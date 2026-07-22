//! One residual layer: a token mixer, then a feed-forward.

use burn::nn::{RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn_mamba::mamba3::prelude::{Mamba3, Mamba3SsdPath};

use crate::config::{self, Mixer, SsdMode};
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
    /// Chunk length of the SSD scan, resolved once at build time so the forward
    /// never falls back to burn-mamba's `None` (which pads the sequence).
    ssd_chunk: usize,
    /// Whether SSD intermediates are retained or recomputed in the backward.
    /// It changes execution only, not parameters or checkpoint records.
    #[module(skip)]
    ssd_mode: SsdMode,
}

impl Block {
    pub fn new(cfg: &config::Model, layer: usize, ssd_mode: SsdMode, device: &Device) -> Self {
        let mix = match cfg.mixer(layer) {
            Mixer::Ssm => Mix::Ssm(cfg.mamba().init(device)),
            Mixer::Attention => Mix::Attn(Attention::new(cfg, device)),
        };
        Self {
            norm_mix: RmsNormConfig::new(cfg.d_model).init(device),
            mix,
            norm_ffn: RmsNormConfig::new(cfg.d_model).init(device),
            ffn: Ffn::new(cfg, device),
            ssd_chunk: cfg.ssd_chunk_len(),
            ssd_mode,
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let mixed = match &self.mix {
            // `SerialRecalculated` recomputes the SSD intermediates in the
            // backward instead of storing them — the difference between fitting
            // 16 GB and not. The chunk length is passed explicitly: left unset,
            // burn-mamba picks √(state_rank · head_dim) rounded to 32, which for
            // the shipped presets does not divide `seq_len` and makes every SSM
            // layer pad its sequence with six `cat` allocations.
            Mix::Ssm(ssm) => {
                let path = match self.ssd_mode {
                    SsdMode::Serial => Mamba3SsdPath::Serial(Some(self.ssd_chunk)),
                    SsdMode::Recalculated => {
                        Mamba3SsdPath::SerialRecalculated(Some(self.ssd_chunk))
                    }
                };
                ssm.forward(self.norm_mix.forward(x.clone()), None, path).0
            }
            Mix::Attn(attn) => attn.forward(self.norm_mix.forward(x.clone())),
        };
        let x = x + mixed;
        let ffn = self.ffn.forward(self.norm_ffn.forward(x.clone()));
        x + ffn
    }
}
