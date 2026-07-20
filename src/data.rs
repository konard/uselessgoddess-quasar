//! The data path: raw documents in, `[batch, seq]` tensors out.
//!
//! Three stages, each usable alone: [`Corpus`] reads documents out of parquet /
//! JSONL / text, [`Tokenizer`] turns them into ids, and [`shard`] stores those
//! ids as memory-mapped `u16` so [`Batcher`] can draw windows without a copy.

pub mod corpus;
pub mod shard;

mod batch;
mod tokenizer;

pub use batch::{Batch, Batcher};
pub use corpus::Corpus;
pub use shard::{Meta, Shards};
pub use tokenizer::Tokenizer;
