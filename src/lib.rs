//! quasar — a Mamba-3 language model family trained on one consumer GPU.
//!
//! The crate is four layers: [`config`] describes a model and its budget,
//! [`model`] builds it, [`data`] feeds it, and [`train`] runs it.

pub mod config;
pub mod model;
