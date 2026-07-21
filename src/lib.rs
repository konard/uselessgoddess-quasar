//! quasar — a Mamba-3 language model family trained on one consumer GPU.
//!
//! The crate is four layers: [`config`] describes a model and its budget,
//! [`model`] builds it, [`data`] feeds it, and [`train`] runs it.

pub mod config;
pub mod data;
pub mod eval;
pub mod generate;
/// A fused RMSNorm kernel written directly in CubeCL. Only compiled with the
/// `cubecl-kernel` feature (pulled in by `cpu`), which brings in the CubeCL
/// runtime a `#[cube]` kernel needs.
#[cfg(feature = "cubecl-kernel")]
pub mod kernel;
pub mod model;
pub mod train;
