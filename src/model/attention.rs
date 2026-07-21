//! Grouped-query attention over a sliding window.
//!
//! This is the recall path of the hybrid: a handful of these layers give the
//! stack the exact-token lookup a fixed-size SSM state cannot do. It is kept
//! deliberately plain — no KV cache, no flash kernel — because it runs in one
//! layer out of six or seven and is not where the time goes.

use burn::nn::{
    Linear, LinearConfig, RmsNorm, RmsNormConfig, RotaryEncoding, RotaryEncodingConfig,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;

use crate::config;
use crate::model::init;

/// Attention with `heads` queries sharing `kv_heads` key/value heads.
#[derive(Module, Debug)]
pub struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    /// QK-norm: normalising q and k per head bounds the logits, which is what
    /// makes a larger learning rate and f16 compute survivable without a loss
    /// scaler — burn has no `GradScaler`.
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rope: RotaryEncoding,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    window: Option<usize>,
}

impl Attention {
    pub fn new(cfg: &config::Model, device: &Device) -> Self {
        let (d, heads, kv_heads) = (cfg.d_model, cfg.attn_heads, cfg.attn_kv_heads);
        let head_dim = d / heads;
        let linear = |a, b, init| {
            LinearConfig::new(a, b).with_bias(false).with_initializer(init).init(device)
        };
        Self {
            q: linear(d, d, init::normal()),
            k: linear(d, kv_heads * head_dim, init::normal()),
            v: linear(d, kv_heads * head_dim, init::normal()),
            o: linear(d, d, init::residual(cfg.n_layers)),
            q_norm: RmsNormConfig::new(head_dim).init(device),
            k_norm: RmsNormConfig::new(head_dim).init(device),
            rope: RotaryEncodingConfig::new(cfg.seq_len, head_dim).init(device),
            heads,
            kv_heads,
            head_dim,
            window: cfg.attn_window,
        }
    }

    /// `[batch, seq, d_model] -> [batch, seq, d_model]`.
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, seq, _] = x.dims();
        let heads = |t: Tensor<3>, n| {
            t.reshape([batch, seq, n, self.head_dim]).swap_dims(1, 2) // [b, h, t, dh]
        };

        let q = self.q_norm.forward(heads(self.q.forward(x.clone()), self.heads));
        let k = self.k_norm.forward(heads(self.k.forward(x.clone()), self.kv_heads));
        let v = heads(self.v.forward(x), self.kv_heads);

        let (q, k) = (self.rope.forward(q), self.rope.forward(k));
        let (k, v) = (self.repeat_kv(k), self.repeat_kv(v));

        let out = match self.window {
            Some(w) if w < seq => self.windowed(q, k, v, w),
            _ => self.dense(q, k, v),
        };

        self.o.forward(out.swap_dims(1, 2).reshape([batch, seq, self.heads * self.head_dim]))
    }

    /// The whole `[t, t]` score matrix at once — what a window wider than the
    /// sequence, or no window at all, reduces to.
    fn dense(&self, q: Tensor<4>, k: Tensor<4>, v: Tensor<4>) -> Tensor<4> {
        let [_, _, seq, _] = q.dims();
        let device = q.device();
        let scores = self.scores(q, k) + self.bias(0, 0, seq, seq, &device);
        softmax(scores, 3).matmul(v)
    }

    /// Sliding-window attention that only ever materialises the scores it keeps.
    ///
    /// The mask alone saves nothing: `q · kᵀ` is `[batch, heads, t, t]` whether
    /// or not most of it is about to be set to `-1e9`, and autodiff holds both it
    /// and the softmax output until the backward — 335 MB per attention layer per
    /// sequence at `tiny`'s 10 heads × 2048². Cutting the queries into blocks of
    /// `w` and giving each block only the `< 2w` keys its window can reach makes
    /// the score memory `t · 2w` instead of `t²`, which is what
    /// [`config::Model::flops_per_token`] already claims the window costs. That
    /// is 25% off at the default 2048/1024 and 4.3× off at a 256-token window —
    /// so shrinking the window now actually shrinks the footprint.
    fn windowed(&self, q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, w: usize) -> Tensor<4> {
        let [batch, heads, seq, _] = q.dims();
        let device = q.device();
        let mut out = Vec::with_capacity(seq.div_ceil(w));
        for start in (0..seq).step_by(w) {
            let end = (start + w).min(seq);
            // The window admits keys `start - w < s <= t`, so nothing before this
            // is reachable from any query in the block.
            let first = (start + 1).saturating_sub(w);
            let block = |t: Tensor<4>| t.slice([0..batch, 0..heads, first..end]);
            let queries = q.clone().slice([0..batch, 0..heads, start..end]);

            let scores = self.scores(queries, block(k.clone()))
                + self.bias(start, first, end - start, end - first, &device);
            out.push(softmax(scores, 3).matmul(block(v.clone())));
        }
        Tensor::cat(out, 2)
    }

    fn scores(&self, q: Tensor<4>, k: Tensor<4>) -> Tensor<4> {
        q.matmul(k.swap_dims(2, 3)).div_scalar((self.head_dim as f64).sqrt())
    }

    /// Broadcast each key/value head to the queries that share it. `expand` on a
    /// unit axis keeps the grouping contiguous, which `repeat_dim` would not.
    fn repeat_kv(&self, t: Tensor<4>) -> Tensor<4> {
        let [batch, kv_heads, seq, dim] = t.dims();
        let group = self.heads / self.kv_heads;
        t.reshape([batch, kv_heads, 1, seq, dim])
            .expand([batch, kv_heads, group, seq, dim])
            .reshape([batch, self.heads, seq, dim])
    }

    /// Additive `-1e9` outside the causal sliding window, shaped
    /// `[1, 1, queries, keys]` so it broadcasts over batch and heads.
    ///
    /// `query` and `key` are the positions the block starts at, so the same
    /// arithmetic serves the whole sequence (`0, 0`) and any window block: a
    /// query at local `i` is token `query + i`, a key at local `j` is token
    /// `key + j`, and shifting the mask diagonals by `query - key` re-centres it.
    ///
    /// Additive rather than `mask_fill` because the value must survive an f16
    /// cast: `-1e9` saturates to `-inf`, and no row is ever fully masked (the
    /// diagonal is always in the window), so no `inf - inf` NaN can appear.
    ///
    /// burn's `triu_mask`/`tril_mask` are *keep* masks — `false` marks the named
    /// triangle and `true` everything else — so they read inverted from the
    /// torch convention and are combined before the single negation.
    fn bias(
        &self,
        query: usize,
        key: usize,
        queries: usize,
        keys: usize,
        device: &Device,
    ) -> Tensor<4> {
        let shift = query as i64 - key as i64;
        let past = Tensor::<2, Bool>::triu_mask([queries, keys], shift + 1, device);
        let keep = match self.window {
            Some(w) => past.bool_and(Tensor::tril_mask([queries, keys], shift - w as i64, device)),
            None => past,
        };
        keep.bool_not().float().mul_scalar(-1e9).reshape([1, 1, queries, keys])
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Distribution, Tolerance};

    use super::*;

    /// `q`, `k` and `v` for a toy attention layer, already in `[b, h, t, dh]`.
    fn qkv(attn: &Attention, seq: usize) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        let device = Device::default();
        let shape = [2, attn.heads, seq, attn.head_dim];
        let draw = || Tensor::random(shape, Distribution::Normal(0.0, 1.0), &device);
        (draw(), draw(), draw())
    }

    fn attention(window: usize, seq: usize) -> Attention {
        let cfg = config::Model::toy().with_seq_len(seq).with_attn_window(Some(window));
        Attention::new(&cfg, &Device::default())
    }

    #[test]
    fn blocking_the_window_does_not_change_the_output() {
        let (window, seq) = (4, 16);
        let attn = attention(window, seq);
        let (q, k, v) = qkv(&attn, seq);

        let blocked = attn.windowed(q.clone(), k.clone(), v.clone(), window);
        let dense = attn.dense(q, k, v);

        blocked.into_data().assert_approx_eq::<f32>(&dense.into_data(), Tolerance::default());
    }

    /// The last block is short whenever the window does not divide the sequence,
    /// and its key range starts mid-sequence — the case an off-by-one lands in.
    #[test]
    fn a_window_that_does_not_divide_the_sequence_still_matches() {
        let (window, seq) = (5, 16);
        let attn = attention(window, seq);
        let (q, k, v) = qkv(&attn, seq);

        let blocked = attn.windowed(q.clone(), k.clone(), v.clone(), window);
        let dense = attn.dense(q, k, v);

        blocked.into_data().assert_approx_eq::<f32>(&dense.into_data(), Tolerance::default());
    }

    #[test]
    fn a_block_bias_is_the_dense_bias_shifted() {
        let (window, seq) = (4, 16);
        let attn = attention(window, seq);
        let device = Device::default();

        let block = attn.bias(8, 5, 4, 7, &device);
        let dense = attn.bias(0, 0, seq, seq, &device).slice([0..1, 0..1, 8..12, 5..12]);

        block.into_data().assert_eq(&dense.into_data(), true);
    }

    /// The window is the memory knob: `t · 2w` scores instead of `t²`.
    #[test]
    fn a_narrower_window_scores_fewer_pairs() {
        let seq = 2048;
        let pairs = |w: usize| {
            (0..seq)
                .step_by(w)
                .map(|start| {
                    let end = (start + w).min(seq);
                    (end - start) * (end - (start + 1).saturating_sub(w))
                })
                .sum::<usize>()
        };

        assert_eq!(pairs(2048), seq * seq, "a window as wide as the sequence is dense");
        assert!(pairs(1024) < seq * seq, "the default window already scores fewer pairs");
        assert!(pairs(256) * 4 < seq * seq, "a 256-token window drops four fifths");
    }
}
