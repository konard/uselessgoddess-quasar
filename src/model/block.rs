//! One residual layer: a token mixer, then a feed-forward.

use burn::nn::{RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn_mamba::mamba3::prelude::{Mamba3, Mamba3SsdPath};

use crate::config::{self, Mixer};
use crate::model::{Attention, Ffn};

/// Whatever mixes tokens in this layer.
#[derive(Module, Debug)]
pub enum Mix {
    Ssm(Mamba3),
    Attn(Attention),
}

/// Pre-norm mixer sublayer plus pre-norm feed-forward sublayer, both residual.
///
/// Same shape for SSM and attention layers, so the stack is a uniform `Vec` and
/// the hybrid schedule is data, not control flow.
#[derive(Module, Debug)]
pub struct Block {
    norm_mix: RmsNorm,
    mix: Mix,
    norm_ffn: RmsNorm,
    ffn: Ffn,
}

impl Block {
    pub fn new(cfg: &config::Model, layer: usize, device: &Device) -> Self {
        let mix = match cfg.mixer(layer) {
            Mixer::Ssm => Mix::Ssm(cfg.mamba().init(device)),
            Mixer::Attention => Mix::Attn(Attention::new(cfg, device)),
        };
        Self {
            norm_mix: RmsNormConfig::new(cfg.d_model).init(device),
            mix,
            norm_ffn: RmsNormConfig::new(cfg.d_model).init(device),
            ffn: Ffn::new(cfg, device),
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let mixed = match &self.mix {
            // `SerialRecalculated` (the default path) recomputes the SSD
            // intermediates in the backward instead of storing them — the
            // difference between fitting 16 GB and not.
            Mix::Ssm(ssm) => {
                ssm.forward(self.norm_mix.forward(x.clone()), None, Mamba3SsdPath::default()).0
            }
            Mix::Attn(attn) => attn.forward(self.norm_mix.forward(x.clone())),
        };
        let x = x + mixed;
        let ffn = self.ffn.forward(self.norm_ffn.forward(x.clone()));
        x + ffn
    }
}
