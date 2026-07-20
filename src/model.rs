//! The network: a hybrid Mamba-3 / sliding-window-attention stack.

mod attention;
mod block;
mod ffn;
mod init;
mod lm;

pub use attention::Attention;
pub use block::Block;
pub use ffn::Ffn;
pub use lm::{Loss, Quasar};
