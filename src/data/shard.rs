//! The on-disk token format: little-endian `u16` shards plus one JSON sidecar.
//!
//! `u16` is why [`crate::data::Tokenizer`] caps the vocabulary at 65 535 — it
//! halves the corpus on disk and doubles the effective loader bandwidth against
//! a `u32` format, and nothing at this scale wants a 128k vocabulary anyway.
//!
//! Shards are read through `mmap`, so the page cache is the dataloader. At the
//! throughput this card reaches, a 150M model consumes well under 1 MB/s of
//! shard — there is no reason for a prefetch thread.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use serde::{Deserialize, Serialize};

/// Bytes per shard file before rotating. 512 MB keeps a shard mappable
/// everywhere and keeps the file count sane for a 100B-token corpus.
const SHARD_BYTES: usize = 512 << 20;

/// What a shard directory contains, beside the tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meta {
    pub tokens: u64,
    pub docs: u64,
    /// UTF-8 bytes of the source text, so evaluation can report bits-per-byte —
    /// the only measure comparable across tokenizers.
    pub bytes: u64,
    pub vocab_size: usize,
    pub eos: u16,
}

/// A read-only, memory-mapped corpus: the shards concatenated end to end.
#[derive(Debug)]
pub struct Shards {
    maps: Vec<Mmap>,
    /// `starts[i]` is the token index where shard `i` begins; the last entry is
    /// the total, so a binary search over it locates any global index.
    starts: Vec<usize>,
    meta: Meta,
}

impl Shards {
    pub fn open(dir: &Path) -> io::Result<Self> {
        let meta: Meta = serde_json::from_reader(File::open(dir.join("meta.json"))?)
            .map_err(io::Error::other)?;

        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "bin"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(io::Error::new(io::ErrorKind::NotFound, "no .bin shards"));
        }

        let mut maps = Vec::with_capacity(files.len());
        let mut starts = vec![0];
        for path in files {
            // SAFETY: the shards are written once by `Writer` and treated as
            // immutable afterwards; a concurrent truncation would be a caller
            // bug, and is the same assumption every mmap-backed loader makes.
            let map = unsafe { Mmap::map(&File::open(path)?)? };
            starts.push(starts.last().unwrap() + map.len() / 2);
            maps.push(map);
        }
        Ok(Self { maps, starts, meta })
    }

    pub fn meta(&self) -> &Meta {
        &self.meta
    }

    /// Total tokens across all shards.
    pub fn len(&self) -> usize {
        *self.starts.last().unwrap()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copy `len` tokens starting at global index `start` into `out`.
    ///
    /// # Panics
    /// If the window runs past the end of the corpus.
    pub fn read(&self, start: usize, len: usize, out: &mut Vec<u16>) {
        assert!(start + len <= self.len(), "window {start}+{len} past {}", self.len());
        out.clear();
        let mut shard = self.starts.partition_point(|&s| s <= start) - 1;
        let mut cursor = start;
        while out.len() < len {
            let local = cursor - self.starts[shard];
            let take = (len - out.len()).min(self.maps[shard].len() / 2 - local);
            let bytes = &self.maps[shard][local * 2..(local + take) * 2];
            out.extend(bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])));
            cursor += take;
            shard += 1;
        }
    }
}

/// Writes a shard directory, rotating files at [`SHARD_BYTES`].
pub struct Writer {
    dir: PathBuf,
    file: BufWriter<File>,
    index: usize,
    in_shard: usize,
    meta: Meta,
}

impl Writer {
    pub fn create(dir: &Path, vocab_size: usize, eos: u16) -> io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        Ok(Self {
            dir: dir.to_owned(),
            file: BufWriter::new(File::create(shard_path(dir, 0))?),
            index: 0,
            in_shard: 0,
            meta: Meta { tokens: 0, docs: 0, bytes: 0, vocab_size, eos },
        })
    }

    /// Append one document's tokens; the caller has already appended EOS.
    pub fn push(&mut self, tokens: &[u16], bytes: usize) -> io::Result<()> {
        for &token in tokens {
            self.file.write_all(&token.to_le_bytes())?;
        }
        self.in_shard += tokens.len() * 2;
        self.meta.tokens += tokens.len() as u64;
        self.meta.docs += 1;
        self.meta.bytes += bytes as u64;
        if self.in_shard >= SHARD_BYTES {
            self.rotate()?;
        }
        Ok(())
    }

    fn rotate(&mut self) -> io::Result<()> {
        self.file.flush()?;
        self.index += 1;
        self.in_shard = 0;
        self.file = BufWriter::new(File::create(shard_path(&self.dir, self.index))?);
        Ok(())
    }

    /// Flush the last shard and write `meta.json`.
    pub fn finish(mut self) -> io::Result<Meta> {
        self.file.flush()?;
        drop(self.file);
        if self.in_shard == 0 && self.index > 0 {
            std::fs::remove_file(shard_path(&self.dir, self.index))?;
        }
        let json = serde_json::to_string_pretty(&self.meta).map_err(io::Error::other)?;
        std::fs::write(self.dir.join("meta.json"), json)?;
        Ok(self.meta)
    }
}

fn shard_path(dir: &Path, index: usize) -> PathBuf {
    dir.join(format!("shard_{index:04}.bin"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn written(docs: &[&[u16]]) -> (tempfile::TempDir, Shards) {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = Writer::create(dir.path(), 64, 0).unwrap();
        for doc in docs {
            writer.push(doc, doc.len()).unwrap();
        }
        writer.finish().unwrap();
        let shards = Shards::open(dir.path()).unwrap();
        (dir, shards)
    }

    #[test]
    fn a_roundtrip_preserves_every_token() {
        let (_dir, shards) = written(&[&[1, 2, 3], &[4, 5]]);

        let mut out = Vec::new();
        shards.read(0, 5, &mut out);

        assert_eq!(out, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn meta_counts_documents_and_bytes() {
        let (_dir, shards) = written(&[&[1, 2, 3], &[4, 5]]);

        let meta = shards.meta();

        assert_eq!((meta.tokens, meta.docs, meta.bytes), (5, 2, 5));
    }

    #[test]
    fn a_window_may_start_anywhere() {
        let (_dir, shards) = written(&[&[1, 2, 3, 4, 5, 6]]);

        let mut out = Vec::new();
        shards.read(2, 3, &mut out);

        assert_eq!(out, [3, 4, 5]);
    }
}
