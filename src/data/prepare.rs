//! The one-off pass that turns a corpus into a shard directory.
//!
//! Runs once per corpus and then never again, so it is written for throughput:
//! documents are tokenised a chunk at a time across every core, which is the
//! only part of the pipeline that is CPU-bound.

use std::io;
use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

use crate::data::corpus;
use crate::data::shard::{self, Meta};
use crate::data::tokenizer;
use crate::data::{Corpus, Tokenizer};

/// Documents tokenised per parallel chunk. Large enough to amortise the join,
/// small enough that a chunk of FineWeb-Edu documents stays under a gigabyte.
const CHUNK: usize = 4096;

/// One document in `VALID_EVERY` is held out. At 1 in 200 a 10B-token corpus
/// leaves ~50M validation tokens — far more than the eval loop needs, and the
/// split is by whole document, so no validation text can appear in training.
const VALID_EVERY: u64 = 200;

#[derive(Debug)]
pub enum Error {
    Corpus(corpus::Error),
    Tokenizer(tokenizer::Error),
    Io(io::Error),
}

/// What `prepare` wrote, one [`Meta`] per split.
#[derive(Debug)]
pub struct Prepared {
    pub train: Meta,
    pub valid: Meta,
}

/// Tokenise every document of `corpus` into `out/train` and `out/valid`.
pub fn run(corpus: &Corpus, tokenizer: &Tokenizer, out: &Path) -> Result<Prepared, Error> {
    let (vocab, eos) = (tokenizer.vocab_size(), tokenizer.eos());
    let mut train = shard::Writer::create(&out.join("train"), vocab, eos).map_err(Error::Io)?;
    let mut valid = shard::Writer::create(&out.join("valid"), vocab, eos).map_err(Error::Io)?;

    let bar = ProgressBar::new_spinner().with_style(style());
    let mut index = 0u64;
    for chunk in chunks(corpus.docs()) {
        let encoded: Vec<_> = chunk?
            .par_iter()
            .map(|doc| tokenizer.encode(doc).map(|ids| (ids, doc.len())))
            .collect::<Result<_, _>>()
            .map_err(Error::Tokenizer)?;

        for (ids, bytes) in encoded {
            let writer = if index.is_multiple_of(VALID_EVERY) { &mut valid } else { &mut train };
            writer.push(&ids, bytes).map_err(Error::Io)?;
            index += 1;
        }
        bar.set_message(format!("{index} docs"));
        bar.tick();
    }
    bar.finish();

    Ok(Prepared {
        train: train.finish().map_err(Error::Io)?,
        valid: valid.finish().map_err(Error::Io)?,
    })
}

/// Group a fallible document stream into owned chunks, failing the whole chunk
/// on the first bad document.
fn chunks<I>(docs: I) -> impl Iterator<Item = Result<Vec<String>, Error>>
where
    I: Iterator<Item = Result<String, corpus::Error>>,
{
    let mut docs = docs.fuse();
    std::iter::from_fn(move || {
        let chunk: Result<Vec<String>, _> = docs.by_ref().take(CHUNK).collect();
        match chunk {
            Ok(chunk) if chunk.is_empty() => None,
            Ok(chunk) => Some(Ok(chunk)),
            Err(error) => Some(Err(Error::Corpus(error))),
        }
    })
}

fn style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner} tokenising {msg} in {elapsed}").unwrap()
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Corpus(error) => write!(f, "{error}"),
            Self::Tokenizer(error) => write!(f, "{error}"),
            Self::Io(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Shards;

    fn corpus(docs: &[&str]) -> (tempfile::TempDir, Corpus) {
        let dir = tempfile::tempdir().unwrap();
        let lines: String =
            docs.iter().map(|d| format!("{{\"text\":\"{d}\"}}\n")).collect::<Vec<_>>().concat();
        std::fs::write(dir.path().join("a.jsonl"), lines).unwrap();
        let corpus = Corpus::open(&[dir.path().to_owned()], "text").unwrap();
        (dir, corpus)
    }

    fn tokenizer() -> Tokenizer {
        Tokenizer::train(["the quick brown fox jumps"].into_iter(), 300).unwrap()
    }

    #[test]
    fn every_document_lands_in_exactly_one_split() {
        let (_dir, corpus) = corpus(&["one", "two", "three"]);
        let out = tempfile::tempdir().unwrap();

        let prepared = run(&corpus, &tokenizer(), out.path()).unwrap();

        assert_eq!(prepared.train.docs + prepared.valid.docs, 3);
    }

    #[test]
    fn the_first_document_is_held_out() {
        let (_dir, corpus) = corpus(&["one", "two", "three"]);
        let out = tempfile::tempdir().unwrap();

        let prepared = run(&corpus, &tokenizer(), out.path()).unwrap();

        assert_eq!(prepared.valid.docs, 1);
    }

    #[test]
    fn the_shards_are_readable_afterwards() {
        let (_dir, corpus) = corpus(&["the quick brown fox", "jumps over"]);
        let out = tempfile::tempdir().unwrap();

        let prepared = run(&corpus, &tokenizer(), out.path()).unwrap();

        let shards = Shards::open(&out.path().join("train")).unwrap();
        assert_eq!(shards.len() as u64, prepared.train.tokens);
    }
}
