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
use burn_mamba::mamba3::prelude::{Mamba3Config, Mamba3SsdPath};

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
    /// Chunk length of the SSD scan. `None` resolves to [`Model::ssd_chunk_len`],
    /// which stays a divisor of [`Model::seq_len`] — burn-mamba pads the
    /// sequence up to a multiple of this value, and a padded sequence costs six
    /// extra `cat` allocations per SSM layer for tokens that are then discarded.
    #[config(default = "None")]
    pub ssd_chunk: Option<usize>,

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

    /// `quasar-tiny-turbo`, 78M — the cheapest model still worth training.
    ///
    /// Every cut here is one the memory accounting says is paid for twice, in
    /// activations and in FLOPs, and none of them is a quality knob of the first
    /// order: `mimo_rank 1` divides the intra-chunk SSD tensors by four,
    /// `state_rank 64` halves the retained `B`/`C`, a 256-token window scores a
    /// fifth of the pairs a 1024 one does, and a 1024-token sequence halves what
    /// is left. `docs/MEMORY.md` has the per-knob arithmetic. What it buys is a
    /// micro-batch above one on 16 GB, which is where the throughput was lost.
    pub fn tiny_turbo() -> Self {
        Self::new(32_768, 512, 20)
            .with_seq_len(1024)
            .with_state_rank(64)
            .with_head_dim(64)
            .with_n_groups(1)
            .with_mimo_rank(1)
            .with_attn_period(Some(5))
            .with_attn_heads(8)
            .with_attn_kv_heads(2)
            .with_attn_window(Some(256))
            .with_ffn_mult(2.0)
            .with_tied_embeddings(true)
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

    /// The resolved SSD chunk length.
    ///
    /// burn-mamba's rule of thumb is `√(state_rank · head_dim)` rounded up to a
    /// multiple of 32 (`Mamba3SsdPath::optimal_chunk_len`), and its scan pads the
    /// sequence up to a multiple of whatever chunk length it is given. For `tiny`
    /// that rule yields 96, which does not divide 2048: every SSM layer would pad
    /// 2048 → 2112 and build six concatenated copies of its largest tensors to do
    /// it. The default here is therefore the largest divisor of `seq_len` not
    /// exceeding the rule's value — 64 for `tiny` — which removes the padding and
    /// shrinks the `[chunk, chunk]` intra-chunk scores at the same time.
    pub fn ssd_chunk_len(&self) -> usize {
        if let Some(chunk) = self.ssd_chunk {
            return chunk.max(1);
        }
        let optimal = Mamba3SsdPath::optimal_chunk_len(self.state_rank, self.head_dim);
        (1..=optimal.min(self.seq_len))
            .rev()
            .find(|chunk| self.seq_len.is_multiple_of(*chunk))
            .unwrap_or(optimal)
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

    /// Score pairs one attention layer materialises for a whole sequence.
    ///
    /// This is the loop in `model::attention::Attention::windowed` written as
    /// arithmetic: block `i` holds `[block, keys]` scores, and the mask decides
    /// nothing about the allocation. A window wider than the sequence — or none
    /// at all — falls back to the dense `t²`.
    pub fn attn_pairs(&self) -> usize {
        let seq = self.seq_len;
        match self.attn_window {
            Some(w) if w < seq => (0..seq)
                .step_by(w)
                .map(|start| {
                    let end = (start + w).min(seq);
                    (end - start) * (end - (start + 1).saturating_sub(w))
                })
                .sum(),
            _ => seq * seq,
        }
    }

    /// Activation bytes one micro-batch holds until its backward, in fp32.
    ///
    /// An estimate, not a measurement — burn fuses and frees on its own schedule
    /// and `--checkpointing` trades some of this back for recompute. It counts
    /// what the pinned code demonstrably keeps: the SSD leaves that
    /// `SerialRecalculated` retains (`v`, `B`, `C`, `dA`, `γ`, `scale`) rather
    /// than the intra-chunk tensors it recomputes, the in-projection and gated
    /// out-norm around them, the attention scores and their softmax, the SwiGLU
    /// intermediates, and the two vocabulary-wide tensors of the loss. It is
    /// meant for the question the OOM actually asks — which micro-batch fits —
    /// and `docs/MEMORY.md` derives every term.
    pub fn activations(&self, batch: usize) -> Activations {
        let (d, seq) = (self.d_model, self.seq_len);
        let (heads, m, r, p) = (self.n_heads(), self.mimo_rank, self.state_rank, self.head_dim);
        let bytes = |per_token: usize| (batch * seq * per_token * 4) as f64;

        // Per SSM layer: the in-projection output, the six retained SSD leaves,
        // the gated RMSNorm chain (x, x², mean, rms, silu(z), product) and the
        // out-projection.
        let ssm_layer = self.mamba().d_in_proj()
            + m * heads * p
            + 2 * m * heads * r
            + 3 * heads
            + 6 * self.d_inner()
            + d;
        // Per attention layer: q, k, v and their rope and repeat copies, then the
        // scores and the softmax over them — the term that grows with the window.
        let attn_layer = 6 * d + 2 * self.attn_heads * self.attn_pairs() / seq;
        // Per layer: two norms, the residual, and SwiGLU's gate, up, silu and
        // product.
        let block = 4 * d + 4 * self.d_ff();

        let (mut ssm, mut attention) = (0.0, 0.0);
        for layer in 0..self.n_layers {
            match self.mixer(layer) {
                Mixer::Ssm => ssm += bytes(ssm_layer),
                Mixer::Attention => attention += bytes(attn_layer),
            }
        }
        // The logits and their log-softmax, both vocabulary-wide and both alive
        // until the backward.
        let head = bytes(2 * self.vocab_size + d);
        let ffn = bytes(self.n_layers * block);

        Activations { ssm, attention, ffn, head, total: ssm + attention + ffn + head }
    }

    /// The largest micro-batch whose states and activations fit `bytes`.
    ///
    /// `per_param` is the optimizer's cost per parameter — 12 B for Muon on the
    /// hidden matrices, 16 B for pure fp32 AdamW. Zero means nothing fits, which
    /// is the honest answer when the states alone are over budget.
    pub fn micro_batch_within(&self, bytes: f64, per_param: f64) -> usize {
        let states = self.budget().total as f64 * per_param;
        let one = self.activations(1).total;
        if states + one > bytes { 0 } else { ((bytes - states) / one) as usize }
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

/// An activation budget, in bytes per micro-batch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Activations {
    pub ssm: f64,
    pub attention: f64,
    pub ffn: f64,
    pub head: f64,
    pub total: f64,
}

impl std::fmt::Display for Activations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mib = |bytes: f64| bytes / (1u64 << 20) as f64;
        writeln!(f, "ssm       {:>9.0} MiB", mib(self.ssm))?;
        writeln!(f, "attention {:>9.0} MiB", mib(self.attention))?;
        writeln!(f, "ffn       {:>9.0} MiB", mib(self.ffn))?;
        writeln!(f, "head      {:>9.0} MiB", mib(self.head))?;
        write!(f, "total     {:>9.0} MiB", mib(self.total))
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
        Model::tiny_turbo().validate().unwrap();
    }

    #[test]
    fn turbo_is_the_smaller_and_cheaper_tiny() {
        let (tiny, turbo) = (Model::tiny(), Model::tiny_turbo());

        assert!(turbo.budget().total < tiny.budget().total);
        assert!(turbo.flops_per_token() < tiny.flops_per_token());
        assert!(
            turbo.activations(1).total * 4.0 < tiny.activations(1).total,
            "turbo {:.0} MiB vs tiny {:.0} MiB",
            turbo.activations(1).total / 1048576.0,
            tiny.activations(1).total / 1048576.0
        );
    }

    #[test]
    fn a_window_narrower_than_the_sequence_scores_fewer_pairs() {
        let cfg = Model::tiny();
        let dense = cfg.clone().with_attn_window(None).attn_pairs();

        assert_eq!(dense, cfg.seq_len * cfg.seq_len);
        assert!(cfg.attn_pairs() < dense, "the default 1024 window");
        assert!(cfg.clone().with_attn_window(Some(256)).attn_pairs() * 4 < dense);
    }

    #[test]
    fn activations_scale_with_the_micro_batch() {
        let cfg = Model::tiny();

        let (one, four) = (cfg.activations(1), cfg.activations(4));

        assert!((four.total - 4.0 * one.total).abs() < 1.0);
    }

    /// The `tiny` run in issue #11 held `micro_batch 1` and OOM'd above it. The
    /// estimate reproduces that for the dense scores the old attention built —
    /// `attn_window = None` is exactly what a mask-only window cost — and shows
    /// where the fixes move the line.
    #[test]
    fn the_estimate_reproduces_the_reported_ceiling() {
        let gib = (16u64 << 30) as f64;
        let dense = Model::tiny().with_attn_window(None);

        assert_eq!(dense.micro_batch_within(gib, 12.0), 1, "what the issue reported");
        assert_eq!(Model::tiny().micro_batch_within(gib, 12.0), 2, "blocking the window");
        assert!(Model::tiny_turbo().micro_batch_within(gib, 12.0) >= 8, "turbo leaves room");
    }

    #[test]
    fn the_ssd_chunk_divides_the_sequence() {
        for cfg in [Model::tiny(), Model::base(), Model::toy()] {
            let (chunk, seq) = (cfg.ssd_chunk_len(), cfg.seq_len);

            assert!(seq.is_multiple_of(chunk), "chunk {chunk} pads a {seq}-token sequence");
        }
    }

    #[test]
    fn the_default_ssd_chunk_stays_under_burn_mambas_rule() {
        let cfg = Model::tiny();

        let rule = Mamba3SsdPath::optimal_chunk_len(cfg.state_rank, cfg.head_dim);

        assert_eq!(rule, 96, "burn-mamba's own rule of thumb for tiny");
        assert_eq!(cfg.ssd_chunk_len(), 64, "the largest divisor of 2048 below it");
    }

    #[test]
    fn an_explicit_ssd_chunk_overrides_the_default() {
        assert_eq!(Model::tiny().with_ssd_chunk(Some(128)).ssd_chunk_len(), 128);
    }

    #[test]
    fn an_odd_state_rank_is_rejected() {
        let cfg = Model::tiny().with_state_rank(63);

        assert_eq!(cfg.validate(), Err(Invalid::OddStateRank(63)));
    }
}
