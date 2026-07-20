//! The optimizer, and which half of the model each part of it sees.
//!
//! AdamW keeps two fp32 moments per parameter, so weights, gradients and states
//! come to 16 B/param — 16.65 GiB for `base` before a single activation is
//! allocated, which is why `base` did not fit. Muon keeps one momentum buffer
//! instead, 12 B/param, and orthogonalises the update, which is the better
//! optimizer for a hidden matrix anyway.
//!
//! What Muon cannot do is anything that is not a matrix: its step asserts a 2-D
//! tensor, so norms and biases are out by construction, and the embedding and
//! the head are out by meaning — their rows are tokens, and a step touches only
//! the tokens in the batch, so orthogonalising over the vocabulary would mix
//! rows that were never seen. Those stay on AdamW. Splitting them is what
//! [`GradientsParams::from_params`] is for: it *removes* the parameters it
//! matches, so the `from_module` that follows collects exactly the remainder.

use std::path::Path;

use burn::grad_clipping::GradientClippingConfig;
use burn::module::ParamId;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{
    AdamWConfig, AdjustLrFn, GradientsAccumulator, GradientsParams, ModuleOptimizer, MuonConfig,
};
use burn::store::{ModuleSnapshot, RecordError};
use burn::tensor::Gradients;

use crate::model::Quasar;
use crate::train::Run;

/// Both optimizers and the accumulator each of them feeds from.
pub struct Optim {
    adamw: ModuleOptimizer,
    adamw_grads: GradientsAccumulator<Quasar>,
    muon: ModuleOptimizer,
    muon_grads: GradientsAccumulator<Quasar>,
    /// The parameters Muon steps; empty when it is switched off, which is the
    /// only thing that distinguishes an AdamW-only run.
    hidden: Vec<ParamId>,
}

impl Optim {
    pub fn new(run: &Run, model: &Quasar) -> Self {
        let clip = GradientClippingConfig::Norm(run.clip);
        Self {
            adamw: AdamWConfig::new()
                .with_weight_decay(run.weight_decay)
                .with_grad_clipping(Some(clip.clone()))
                .init(),
            adamw_grads: GradientsAccumulator::new(),
            muon: MuonConfig::new()
                .with_weight_decay(Some(WeightDecayConfig::new(run.weight_decay)))
                // Moonshot's scaling gives an orthogonalised update the same RMS
                // as an AdamW one, so the schedule stays as it was tuned.
                .with_adjust_lr_fn(AdjustLrFn::MatchRmsAdamW)
                .init()
                .with_grad_clipping(clip.init()),
            muon_grads: GradientsAccumulator::new(),
            hidden: if run.muon { hidden(model) } else { Vec::new() },
        }
    }

    /// Split one micro-batch's gradients between the two optimizers.
    pub fn accumulate(&mut self, model: &Quasar, mut grads: Gradients) {
        if !self.hidden.is_empty() {
            let matrices = GradientsParams::from_params(&mut grads, model, &self.hidden);
            self.muon_grads.accumulate(model, matrices);
        }
        let rest = GradientsParams::from_module(&mut grads, model);
        self.adamw_grads.accumulate(model, rest);
    }

    /// Apply everything accumulated since the last step.
    pub fn step(&mut self, lr: f64, model: Quasar) -> Quasar {
        let model = match self.hidden.is_empty() {
            true => model,
            false => self.muon.step(lr, model, self.muon_grads.grads()),
        };
        self.adamw.step(lr, model, self.adamw_grads.grads())
    }

    pub fn save(&self, dir: &Path) -> Result<(), RecordError> {
        self.adamw.save(dir.join("optim.bpk"))?;
        self.muon.save(dir.join("muon.bpk"))
    }

    pub fn load(self, dir: &Path) -> Result<Self, RecordError> {
        let Self { adamw, muon, adamw_grads, muon_grads, hidden } = self;
        Ok(Self {
            adamw: adamw.load(dir.join("optim.bpk"))?,
            muon: muon.load(dir.join("muon.bpk"))?,
            adamw_grads,
            muon_grads,
            hidden,
        })
    }
}

/// The parameters Muon may step: the two-dimensional weights that are not the
/// vocabulary. That is the projections of every mixer and feed-forward, which is
/// the overwhelming majority of the parameters and so of the optimizer state.
fn hidden(model: &Quasar) -> Vec<ParamId> {
    model
        .collect(None, None, false)
        .iter()
        .filter(|tensor| tensor.shape.num_dims() == 2 && !vocab(&tensor.full_path()))
        .filter_map(|tensor| tensor.tensor_id)
        .collect()
}

/// Whether a parameter path names a matrix whose rows are tokens.
fn vocab(path: &str) -> bool {
    path.contains("embed") || path.contains("head")
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::prelude::*;

    use crate::config;

    fn model() -> Quasar {
        Quasar::new(&config::Model::toy(), &Device::default())
    }

    #[test]
    fn muon_takes_the_matrices_and_leaves_the_vocabulary() {
        let model = model();
        let hidden = hidden(&model);

        for tensor in model.collect(None, None, false) {
            let matrix = tensor.shape.num_dims() == 2;
            let path = tensor.full_path();
            let taken = hidden.contains(&tensor.tensor_id.unwrap());
            assert_eq!(taken, matrix && !vocab(&path), "{path} {:?}", tensor.shape);
        }
        assert!(!hidden.is_empty(), "the toy model has projections to orthogonalise");
    }

    /// The split is only correct if it is a partition: a matrix that reaches
    /// neither optimizer silently stops training, and one that reaches Muon
    /// without being a matrix panics inside the Newton-Schulz iteration.
    #[test]
    fn a_step_moves_both_halves_of_the_split() {
        let (device, run) = (Device::default().autodiff(), Run::new());
        let model = Quasar::new(&config::Model::toy(), &device);
        let before = model.clone().collect(None, None, false);
        let mut optim = Optim::new(&run, &model);

        let tokens = Tensor::<2, Int>::zeros([2, 8], &device);
        let grads = model.loss(tokens.clone(), tokens).total.backward();
        optim.accumulate(&model, grads);
        let model = optim.step(1e-2, model);

        let after = model.collect(None, None, false);
        assert_eq!(before.len(), after.len());
        // Every matrix has a gradient after one forward, whichever optimizer
        // owns it — the vocabulary through AdamW, the rest through Muon.
        for (before, after) in before.iter().zip(&after).filter(|(t, _)| t.shape.num_dims() == 2) {
            let was = before.to_data().unwrap().to_vec::<f32>().unwrap();
            let now = after.to_data().unwrap().to_vec::<f32>().unwrap();
            let step = was.iter().zip(&now).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(step > 0.0, "{} never moved", before.full_path());
        }
    }
}
