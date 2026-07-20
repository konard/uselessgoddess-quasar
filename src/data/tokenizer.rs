//! The byte-level BPE tokenizer, trained on the corpus it will encode.
//!
//! Trained rather than borrowed: a 32k vocabulary fitted to the target corpus
//! beats a borrowed 50k one at this scale twice over — it spends fewer
//! parameters on the embedding (32k × 640 is 13% of `tiny`, 50k would be 19%)
//! and it fertilises better on the text actually being modelled.
//!
//! Digits are split individually and the vocabulary is capped at 65 535 so a
//! token fits the `u16` shard format.

use std::path::Path;

use tokenizers::models::TrainerWrapper;
use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::digits::Digits;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::{AddedToken, PreTokenizerWrapper};

/// The document separator, and the only special token.
pub const EOS: &str = "<|endoftext|>";

/// Largest vocabulary a `u16` shard can address.
pub const MAX_VOCAB: usize = u16::MAX as usize + 1;

#[derive(Debug)]
pub enum Error {
    /// Wraps `tokenizers`' own boxed error, which carries no useful variants.
    Tokenizers(String),
    /// A vocabulary this large cannot be stored in the shard format.
    TooLarge(usize),
    /// The trained vocabulary has no [`EOS`], so documents could not be joined.
    NoEos,
}

/// A trained tokenizer plus the id of [`EOS`].
#[derive(Debug)]
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    eos: u16,
}

impl Tokenizer {
    /// Fit a new vocabulary of `vocab_size` merges over `docs`.
    pub fn train<I, S>(docs: I, vocab_size: usize) -> Result<Self, Error>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
    {
        if vocab_size > MAX_VOCAB {
            return Err(Error::TooLarge(vocab_size));
        }
        let mut inner = tokenizers::Tokenizer::new(BPE::default());
        inner.with_pre_tokenizer(Some(pre_tokenizer())).with_decoder(Some(byte_level()));

        let mut trainer: TrainerWrapper = BpeTrainerBuilder::new()
            .vocab_size(vocab_size)
            .show_progress(true)
            .special_tokens(vec![AddedToken::from(EOS, true)])
            // Seeding with all 256 byte characters is what makes the tokenizer
            // total: any byte sequence encodes, so no `<unk>` is ever needed.
            .initial_alphabet(ByteLevel::alphabet().into_iter().collect())
            .build()
            .into();
        inner.train(&mut trainer, docs).map_err(|e| Error::Tokenizers(e.to_string()))?;

        Self::wrap(inner)
    }

    pub fn load(path: &Path) -> Result<Self, Error> {
        let inner =
            tokenizers::Tokenizer::from_file(path).map_err(|e| Error::Tokenizers(e.to_string()))?;
        Self::wrap(inner)
    }

    pub fn save(&self, path: &Path) -> Result<(), Error> {
        self.inner.save(path, true).map_err(|e| Error::Tokenizers(e.to_string()))
    }

    fn wrap(inner: tokenizers::Tokenizer) -> Result<Self, Error> {
        let size = inner.get_vocab_size(true);
        if size > MAX_VOCAB {
            return Err(Error::TooLarge(size));
        }
        let eos = inner.token_to_id(EOS).ok_or(Error::NoEos)? as u16;
        Ok(Self { inner, eos })
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn eos(&self) -> u16 {
        self.eos
    }

    /// Encode one document, terminated by [`EOS`].
    pub fn encode(&self, text: &str) -> Result<Vec<u16>, Error> {
        let mut ids = self.encode_raw(text)?;
        ids.push(self.eos);
        Ok(ids)
    }

    /// Encode without the terminator, for a prompt that is to be continued.
    pub fn encode_raw(&self, text: &str) -> Result<Vec<u16>, Error> {
        let encoding =
            self.inner.encode(text, false).map_err(|e| Error::Tokenizers(e.to_string()))?;
        Ok(encoding.get_ids().iter().map(|&id| id as u16).collect())
    }

    pub fn decode(&self, ids: &[u16]) -> Result<String, Error> {
        let ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.inner.decode(&ids, false).map_err(|e| Error::Tokenizers(e.to_string()))
    }
}

/// Digits before bytes: `2019` becoming four tokens costs a little compression
/// and buys arithmetic that generalises, which every recent tokenizer does.
fn pre_tokenizer() -> PreTokenizerWrapper {
    Sequence::new(vec![Digits::new(true).into(), byte_level().into()]).into()
}

/// `add_prefix_space` is off: with it on, every digit split by [`Digits`] would
/// come back prefixed with `Ġ` and cost two tokens instead of one.
fn byte_level() -> ByteLevel {
    ByteLevel::new(false, true, true)
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tokenizers(message) => write!(f, "tokenizers: {message}"),
            Self::TooLarge(size) => write!(f, "vocabulary of {size} exceeds the u16 shard format"),
            Self::NoEos => write!(f, "the vocabulary has no {EOS}"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    fn trained() -> Tokenizer {
        let docs = ["the quick brown fox", "the lazy dog", "quick quick brown"];
        Tokenizer::train(docs.into_iter(), 300).unwrap()
    }

    #[test]
    fn text_survives_a_roundtrip() {
        let tokenizer = trained();

        let ids = tokenizer.encode("the quick fox").unwrap();

        assert_eq!(tokenizer.decode(&ids[..ids.len() - 1]).unwrap(), "the quick fox");
    }

    #[test]
    fn unseen_bytes_still_encode() {
        let tokenizer = trained();

        let ids = tokenizer.encode("λ→∅").unwrap();

        assert_eq!(tokenizer.decode(&ids[..ids.len() - 1]).unwrap(), "λ→∅");
    }

    #[test]
    fn every_document_ends_with_eos() {
        let tokenizer = trained();

        let ids = tokenizer.encode("the dog").unwrap();

        assert_eq!(ids.last(), Some(&tokenizer.eos()));
    }

    #[test]
    fn digits_never_merge() {
        let tokenizer = trained();

        let ids = tokenizer.encode("2019").unwrap();

        assert_eq!(ids.len(), 5, "four digits plus eos");
    }
}
