//! The language model: embedding, a stack of [`Block`]s, and the head.

use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::activation::log_softmax;

use crate::config;
use crate::model::{Block, init};

/// A quasar model, built from [`config::Model`].
#[derive(Module, Debug)]
pub struct Quasar {
    embed: Embedding,
    blocks: Vec<Block>,
    norm: RmsNorm,
    /// `None` when embeddings are tied, in which case the embedding matrix is
    /// transposed into the head at every forward.
    head: Option<Linear>,
    z_loss: f64,
}

impl Quasar {
    pub fn new(cfg: &config::Model, device: &Device) -> Self {
        Self::new_with_ssd(cfg, config::SsdMode::default(), device)
    }

    /// Build the same model with an explicit SSD memory/speed tradeoff.
    ///
    /// `ssd_mode` changes only which mathematically equivalent burn-mamba
    /// backward implementation runs; it does not change parameters or records.
    pub fn new_with_ssd(cfg: &config::Model, ssd_mode: config::SsdMode, device: &Device) -> Self {
        cfg.validate().expect("model config is invalid");
        Self {
            embed: EmbeddingConfig::new(cfg.vocab_size, cfg.d_model)
                .with_initializer(init::normal())
                .init(device),
            blocks: (0..cfg.n_layers)
                .map(|i| Block::new(cfg, i, ssd_mode.clone(), device))
                .collect(),
            norm: RmsNormConfig::new(cfg.d_model).init(device),
            head: (!cfg.tied_embeddings).then(|| {
                LinearConfig::new(cfg.d_model, cfg.vocab_size)
                    .with_bias(false)
                    .with_initializer(init::normal())
                    .init(device)
            }),
            z_loss: cfg.z_loss,
        }
    }

    /// `[batch, seq] -> [batch, seq, vocab]`.
    pub fn forward(&self, tokens: Tensor<2, Int>) -> Tensor<3> {
        let x = self.blocks.iter().fold(self.embed.forward(tokens), |x, b| b.forward(x));
        let x = self.norm.forward(x);
        match &self.head {
            Some(head) => head.forward(x),
            None => x.matmul(self.embed.weight.val().transpose().unsqueeze()),
        }
    }

    /// Next-token loss over every position.
    pub fn loss(&self, tokens: Tensor<2, Int>, targets: Tensor<2, Int>) -> Loss {
        Loss::new(self.forward(tokens), targets, self.z_loss)
    }
}

/// The training objective and its parts, kept separate so a run can log the
/// cross-entropy that perplexity is computed from rather than the total.
#[derive(Debug)]
pub struct Loss {
    pub nll: Tensor<1>,
    pub z: Tensor<1>,
    pub total: Tensor<1>,
}

impl Loss {
    fn new(logits: Tensor<3>, targets: Tensor<2, Int>, z_loss: f64) -> Self {
        let [batch, seq, _] = logits.dims();
        let logp = log_softmax(logits.clone(), 2);
        let nll = -logp.clone().gather(2, targets.reshape([batch, seq, 1])).mean();

        // `log_softmax` subtracts the log-normaliser, so recovering it needs no
        // second reduction: any vocabulary column of `logits - logp` is it.
        let column = [0..batch, 0..seq, 0..1];
        let z = (logits.slice(column.clone()) - logp.slice(column)).powi_scalar(2).mean();

        let total = nll.clone() + z.clone().mul_scalar(z_loss);
        Self { nll, z, total }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::{Optim, Run};

    #[test]
    fn analytic_budget_matches_the_real_module() {
        let cfg = config::Model::toy();

        let model = Quasar::new(&cfg, &Device::default());

        assert_eq!(model.num_params(), cfg.budget().total, "\n{}", cfg.budget());
    }

    #[test]
    fn tied_and_untied_stacks_agree_on_shape() {
        let cfg = config::Model::toy();
        let device = Device::default();
        let tokens = Tensor::<2, Int>::zeros([2, 8], &device);

        let tied = Quasar::new(&cfg.clone().with_tied_embeddings(true), &device);
        let untied = Quasar::new(&cfg.clone().with_tied_embeddings(false), &device);

        assert_eq!(tied.forward(tokens.clone()).dims(), [2, 8, cfg.vocab_size]);
        assert_eq!(untied.forward(tokens).dims(), [2, 8, cfg.vocab_size]);
    }

    #[test]
    fn an_untrained_model_loses_about_log_vocab() {
        let cfg = config::Model::toy();
        let device = Device::default();
        let tokens = Tensor::<2, Int>::zeros([2, 8], &device);

        let loss = Quasar::new(&cfg, &device).loss(tokens.clone(), tokens);

        let uniform = (cfg.vocab_size as f32).ln();
        assert!((loss.nll.into_scalar::<f32>() - uniform).abs() < 0.5, "expected ≈{uniform}");
    }

    #[test]
    fn a_later_token_sees_an_earlier_one() {
        let cfg = config::Model::toy();
        let device = Device::default();
        let base = Tensor::<2, Int>::zeros([1, 8], &device);
        let poked = base
            .clone()
            .slice_assign([0..1, 2..3], Tensor::<2, Int>::ones([1, 1], &device).mul_scalar(7));
        let model = Quasar::new(&cfg, &device);

        let delta = (model.forward(poked) - model.forward(base)).abs();

        let before = delta.clone().slice([0..1, 0..2, 0..cfg.vocab_size]).max();
        let after = delta.slice([0..1, 3..8, 0..cfg.vocab_size]).max();
        assert!(before.into_scalar::<f32>() < 1e-6, "position 2 must not leak backwards");
        assert!(after.into_scalar::<f32>() > 1e-6, "position 2 must reach later positions");
    }

    #[test]
    fn ssd_modes_agree_after_an_optimizer_step() {
        let cfg = config::Model::toy();
        let device = Device::default().autodiff().gradient_checkpointing();
        let record = Quasar::new(&cfg, &device).into_record();

        let step = |mode: config::SsdMode| -> f32 {
            let tokens = Tensor::<2, Int>::zeros([2, cfg.seq_len], &device);
            // Device seeding is global to the backend and races with other
            // tests running in parallel. Loading one shared record makes the
            // initial parameters identical and isolates the SSD algorithm.
            let mut model =
                Quasar::new_with_ssd(&cfg, mode.clone(), &device).load_record(record.clone());
            let run = Run::new().with_ssd_mode(Some(mode));
            let mut optim = Optim::new(&run, &model);

            let grads = model.loss(tokens.clone(), tokens.clone()).total.backward();
            optim.accumulate(&model, grads);
            model = optim.step(run.lr, model);

            model.loss(tokens.clone(), tokens).nll.into_scalar::<f32>()
        };

        let serial = step(config::SsdMode::Serial);
        let recalculated = step(config::SsdMode::Recalculated);
        let minimal = step(config::SsdMode::Minimal);

        assert!((serial - recalculated).abs() < 1e-4, "serial {serial} vs {recalculated}");
        assert!((minimal - recalculated).abs() < 1e-4, "minimal {minimal} vs {recalculated}");
    }
}
