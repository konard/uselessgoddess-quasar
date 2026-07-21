//! The model family and its parameter arithmetic.
//!
//! Every parameter count claimed here is checked against the module burn
//! actually builds (`model::tests::analytic_budget_matches_the_real_module`).
//! A design document that disagrees with the code is worse than no document,
//! so the arithmetic lives next to the thing it describes.
//!
//! The two shipped presets are [`Model::tiny`] and [`Model::base`];
//! `docs/DESIGN.md` justifies each number.

use burn::prelude::*;
use burn_mamba::mamba3::prelude::Mamba3Config;

/// What mixes tokens in a layer.
///
/// A pure-SSM stack recalls an arbitrary earlier token poorly — its memory is a
/// fixed-size summary — and every published hybrid (Nemotron-H, Samba, Jamba,
/// MiniMax-01, and Mamba-3's own) spends 8–17% of layers on attention to fix it.
/// Which layers those are is [`Model::attn_period`], so the ablation is a config
/// edit rather than a rewrite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mixer {
    /// Mamba-3 SSD: trapezoidal discretisation, data-dependent RoPE, MIMO.
    Ssm,
    /// Grouped-query attention over a sliding window.
    Attention,
}

/// One quasar model.
#[derive(Config, Debug, PartialEq)]
pub struct Model {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,

    /// Sequence length the model trains at. Both mixers are position-relative
    /// (data-dependent RoPE in the SSM, a sliding window in attention), so this
    /// is a training-cost choice, not a hard limit at inference.
    #[config(default = 2048)]
    pub seq_len: usize,

    // ── SSM ──────────────────────────────────────────────────────────────────
    /// SSM state width `N`. The recurrent memory is `n_heads · head_dim · N`
    /// scalars per layer — this is the knob that buys recall.
    #[config(default = 128)]
    pub state_rank: usize,
    #[config(default = 2)]
    pub expand: usize,
    #[config(default = 64)]
    pub head_dim: usize,
    /// B/C sharing groups across SSM heads.
    #[config(default = 1)]
    pub n_groups: usize,
    /// MIMO rank `R`: parallel read/write channels into one state. Raises the
    /// arithmetic intensity of the recurrence without widening the state.
    #[config(default = 1)]
    pub mimo_rank: usize,
    /// Fraction of state dimensions carrying data-dependent RoPE. burn-mamba
    /// accepts 0.0, 0.5 or 1.0; 0.5 is what restores state tracking.
    #[config(default = 0.5)]
    pub rope_fraction: f64,

    // ── attention ────────────────────────────────────────────────────────────
    /// Every `attn_period`-th layer is attention instead of SSM; `None` gives a
    /// pure-SSM stack.
    #[config(default = "None")]
    pub attn_period: Option<usize>,
    #[config(default = 12)]
    pub attn_heads: usize,
    #[config(default = 2)]
    pub attn_kv_heads: usize,
    /// Sliding-window radius in tokens; `None` is full causal attention. Samba
    /// showed a window suffices: global attention buys little once an SSM
    /// carries the long-range state, and it costs quadratic memory.
    #[config(default = "Some(1024)")]
    pub attn_window: Option<usize>,

    // ── feed-forward ─────────────────────────────────────────────────────────
    /// SwiGLU hidden width as a multiple of `d_model`, rounded up to 64.
    #[config(default = 2.5)]
    pub ffn_mult: f64,

    // ── head ─────────────────────────────────────────────────────────────────
    /// Share the embedding matrix with the output projection.
    #[config(default = true)]
    pub tied_embeddings: bool,
    /// Coefficient of the auxiliary `log²Z` term holding the softmax normaliser
    /// near 1. Zero disables it.
    #[config(default = 1e-4)]
    pub z_loss: f64,
}

impl Model {
    /// `quasar-tiny`, 164M — the model intended for the first full run.
    ///
    /// Deep and thin (24 × 640) because MobileLLM measured depth beating width
    /// below 1B. Its compute-efficient target is about 3.25B tokens; the actual
    /// wall-clock budget is derived from measured backend throughput.
    pub fn tiny() -> Self {
        Self::new(32_768, 640, 24)
            .with_seq_len(2048)
            .with_state_rank(128)
            .with_head_dim(64)
            .with_n_groups(2)
            .with_mimo_rank(2)
            .with_attn_period(Some(6))
            .with_attn_heads(10)
            .with_attn_kv_heads(2)
            .with_attn_window(Some(1024))
            .with_ffn_mult(2.5)
            .with_tied_embeddings(true)
    }

    /// `quasar-base`, 1.14B — the largest model whose weights, gradients and
    /// Adam moments still fit 16 GB in bf16 (8 B/param ≈ 9.1 GB).
    pub fn base() -> Self {
        Self::new(32_768, 1536, 28)
            .with_seq_len(2048)
            .with_state_rank(128)
            .with_head_dim(64)
            .with_n_groups(4)
            .with_mimo_rank(4)
            .with_attn_period(Some(7))
            .with_attn_heads(24)
            .with_attn_kv_heads(4)
            .with_attn_window(Some(1024))
            .with_ffn_mult(2.5)
            // Untied here: the embedding is only 4.4% of this budget, and a tied
            // matrix is shaped by output prediction at the input's expense.
            .with_tied_embeddings(false)
    }

    /// A model small enough to train a few steps inside a unit test.
    pub fn toy() -> Self {
        Self::new(64, 32, 4)
            .with_seq_len(16)
            .with_state_rank(8)
            .with_head_dim(8)
            .with_mimo_rank(2)
            .with_attn_period(Some(2))
            .with_attn_heads(4)
            .with_attn_kv_heads(2)
            .with_attn_window(Some(8))
            .with_ffn_mult(2.0)
    }

    /// The mixer of layer `index`.
    ///
    /// Attention lands on the *last* layer of each period, so layer 0 — the one
    /// reading raw embeddings — is always an SSM.
    pub fn mixer(&self, index: usize) -> Mixer {
        match self.attn_period {
            Some(period) if period > 0 && (index + 1).is_multiple_of(period) => Mixer::Attention,
            _ => Mixer::Ssm,
        }
    }

    /// SwiGLU hidden width, rounded to a multiple of 64 so GEMM tiles are whole.
    pub fn d_ff(&self) -> usize {
        let raw = (self.d_model as f64 * self.ffn_mult).round() as usize;
        raw.div_ceil(64) * 64
    }

    pub fn d_inner(&self) -> usize {
        self.expand * self.d_model
    }

    pub fn n_heads(&self) -> usize {
        self.d_inner() / self.head_dim
    }

    /// The burn-mamba block config for one SSM layer.
    pub fn mamba(&self) -> Mamba3Config {
        Mamba3Config::new(self.d_model)
            .with_state_rank(self.state_rank)
            .with_expand(self.expand)
            .with_per_head_dim(self.head_dim)
            .with_ngroups(self.n_groups)
            .with_mimo_rank(self.mimo_rank)
            .with_rope_fraction(self.rope_fraction)
            // The per-head gated RMSNorm before `out_proj` keeps the output
            // scale stable once MIMO sums several read channels into it.
            .with_has_outproj_norm(true)
    }

    pub fn validate(&self) -> Result<(), Invalid> {
        use Invalid::*;
        if !self.state_rank.is_multiple_of(2) {
            return Err(OddStateRank(self.state_rank));
        }
        if !self.d_inner().is_multiple_of(self.head_dim) {
            return Err(HeadDim { head_dim: self.head_dim, d_inner: self.d_inner() });
        }
        if !self.n_heads().is_multiple_of(self.n_groups) {
            return Err(Groups { groups: self.n_groups, heads: self.n_heads() });
        }
        if self.mimo_rank == 0 {
            return Err(MimoRank);
        }
        if self.attn_period.is_some() {
            if !self.d_model.is_multiple_of(self.attn_heads) {
                return Err(AttnHeads { heads: self.attn_heads, d_model: self.d_model });
            }
            if !self.attn_heads.is_multiple_of(self.attn_kv_heads) {
                return Err(KvHeads { kv: self.attn_kv_heads, heads: self.attn_heads });
            }
        }
        Ok(())
    }

    /// The analytic parameter budget, broken down by what it is spent on.
    pub fn budget(&self) -> Budget {
        let d = self.d_model;
        let embedding = self.vocab_size * d;
        let head = if self.tied_embeddings { 0 } else { self.vocab_size * d };

        let (mut ssm, mut attention) = (0, 0);
        for layer in 0..self.n_layers {
            match self.mixer(layer) {
                Mixer::Ssm => ssm += self.ssm_params(),
                Mixer::Attention => attention += self.attn_params(),
            }
        }
        let ffn = self.n_layers * 3 * d * self.d_ff();
        // Two pre-norms per layer, plus the final norm before the head.
        let norms = (2 * self.n_layers + 1) * d;

        Budget {
            embedding,
            head,
            ssm,
            attention,
            ffn,
            norms,
            total: embedding + head + ssm + attention + ffn + norms,
        }
    }

    /// Parameters of one Mamba-3 block, mirroring `Mamba3Config::init`.
    fn ssm_params(&self) -> usize {
        let cfg = self.mamba();
        let (heads, n, r) = (self.n_heads(), self.state_rank, self.mimo_rank);
        let projections = self.d_model * cfg.d_in_proj() + self.d_inner() * self.d_model;
        let mimo = if r > 1 { 3 * heads * r * self.head_dim } else { 0 };
        // dt_bias and D, the two B/C RMSNorm gammas, their per-head biases, and
        // the gated out-norm's gamma.
        let small = 2 * heads + 2 * n + 2 * heads * r * n + self.head_dim;
        projections + mimo + small
    }

    /// Parameters of one GQA block: `q`, `k`, `v`, `o` bias-free, plus the two
    /// QK-norm gammas.
    fn attn_params(&self) -> usize {
        let d = self.d_model;
        let head = d / self.attn_heads;
        2 * d * d + 2 * d * (self.attn_kv_heads * head) + 2 * head
    }

    /// Forward FLOPs per token, counting a multiply-add as two.
    ///
    /// Backward is taken as 2× forward — the usual convention — so a training
    /// step costs `3 ×` this. Attention is quadratic inside its window only.
    pub fn flops_per_token(&self) -> f64 {
        let d = self.d_model as f64;
        let mut flops = 0.0;
        for layer in 0..self.n_layers {
            flops += match self.mixer(layer) {
                Mixer::Ssm => {
                    let cfg = self.mamba();
                    let projections = 2.0 * d * (cfg.d_in_proj() + self.d_inner()) as f64;
                    // The SSD recurrence reads and writes the whole state once
                    // per token per MIMO channel.
                    let state = (self.n_heads() * self.head_dim * self.state_rank) as f64;
                    projections + 4.0 * state * self.mimo_rank as f64
                }
                Mixer::Attention => {
                    let head = (self.d_model / self.attn_heads) as f64;
                    let kv = self.attn_kv_heads as f64 * head;
                    let span = self.attn_window.unwrap_or(self.seq_len).min(self.seq_len) as f64;
                    2.0 * d * (2.0 * d + 2.0 * kv) + 4.0 * span * d
                }
            };
            flops += 6.0 * d * self.d_ff() as f64;
        }
        // The unembedding is a full vocab GEMM at every position.
        flops + 2.0 * d * self.vocab_size as f64
    }
}

/// Why a [`Model`] cannot be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Invalid {
    OddStateRank(usize),
    HeadDim { head_dim: usize, d_inner: usize },
    Groups { groups: usize, heads: usize },
    MimoRank,
    AttnHeads { heads: usize, d_model: usize },
    KvHeads { kv: usize, heads: usize },
}

impl std::fmt::Display for Invalid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OddStateRank(n) => write!(f, "state_rank {n} is odd, RoPE pairs dimensions"),
            Self::HeadDim { head_dim, d_inner } => {
                write!(f, "head_dim {head_dim} does not divide d_inner {d_inner}")
            }
            Self::Groups { groups, heads } => {
                write!(f, "n_groups {groups} does not divide {heads} SSM heads")
            }
            Self::MimoRank => write!(f, "mimo_rank must be at least 1"),
            Self::AttnHeads { heads, d_model } => {
                write!(f, "attn_heads {heads} does not divide d_model {d_model}")
            }
            Self::KvHeads { kv, heads } => {
                write!(f, "attn_kv_heads {kv} does not divide attn_heads {heads}")
            }
        }
    }
}

impl std::error::Error for Invalid {}

/// A parameter budget, in parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    pub embedding: usize,
    pub head: usize,
    pub ssm: usize,
    pub attention: usize,
    pub ffn: usize,
    pub norms: usize,
    pub total: usize,
}

impl std::fmt::Display for Budget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let m = |n: usize| n as f64 / 1e6;
        writeln!(f, "embedding {:>9.1}M", m(self.embedding))?;
        writeln!(f, "lm_head   {:>9.1}M", m(self.head))?;
        writeln!(f, "ssm       {:>9.1}M", m(self.ssm))?;
        writeln!(f, "attention {:>9.1}M", m(self.attention))?;
        writeln!(f, "ffn       {:>9.1}M", m(self.ffn))?;
        writeln!(f, "norms     {:>9.1}M", m(self.norms))?;
        write!(f, "total     {:>9.1}M", m(self.total))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_lands_on_the_last_layer_of_each_period() {
        let cfg = Model::tiny().with_attn_period(Some(6));

        let mixers: Vec<_> = (0..12).map(|i| cfg.mixer(i)).collect();

        assert_eq!(mixers[0], Mixer::Ssm, "the layer reading embeddings stays an SSM");
        assert_eq!(mixers[5], Mixer::Attention);
        assert_eq!(mixers[11], Mixer::Attention);
    }

    #[test]
    fn tying_the_head_removes_a_vocab_matrix() {
        let cfg = Model::tiny();

        let saved = cfg.clone().with_tied_embeddings(false).budget().total
            - cfg.clone().with_tied_embeddings(true).budget().total;

        assert_eq!(saved, cfg.vocab_size * cfg.d_model);
    }

    #[test]
    fn presets_stay_inside_their_size_class() {
        let (tiny, base) = (Model::tiny().budget().total, Model::base().budget().total);

        assert!((100e6..200e6).contains(&(tiny as f64)), "tiny is {tiny}");
        assert!((1.0e9..1.5e9).contains(&(base as f64)), "base is {base}");
    }

    #[test]
    fn presets_validate() {
        Model::tiny().validate().unwrap();
        Model::base().validate().unwrap();
        Model::toy().validate().unwrap();
    }

    #[test]
    fn an_odd_state_rank_is_rejected() {
        let cfg = Model::tiny().with_state_rank(63);

        assert_eq!(cfg.validate(), Err(Invalid::OddStateRank(63)));
    }
}
