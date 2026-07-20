//! Grouped-query attention over a sliding window.
//!
//! This is the recall path of the hybrid: a handful of these layers give the
//! stack the exact-token lookup a fixed-size SSM state cannot do. It is kept
//! deliberately plain — no KV cache, no flash kernel — because it runs in one
//! layer out of six or seven and is not where the time goes.

use burn::nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig, RotaryEncoding, RotaryEncodingConfig};
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
        let device = x.device();
        let heads = |t: Tensor<3>, n| {
            t.reshape([batch, seq, n, self.head_dim]).swap_dims(1, 2) // [b, h, t, dh]
        };

        let q = self.q_norm.forward(heads(self.q.forward(x.clone()), self.heads));
        let k = self.k_norm.forward(heads(self.k.forward(x.clone()), self.kv_heads));
        let v = heads(self.v.forward(x), self.kv_heads);

        let (q, k) = (self.rope.forward(q), self.rope.forward(k));
        let (k, v) = (self.repeat_kv(k), self.repeat_kv(v));

        let scale = (self.head_dim as f64).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)).div_scalar(scale) + self.bias(seq, &device);
        let out = softmax(scores, 3).matmul(v);

        self.o.forward(out.swap_dims(1, 2).reshape([batch, seq, self.heads * self.head_dim]))
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

    /// Additive `-1e9` outside the causal sliding window, shaped `[1, 1, t, t]`
    /// so it broadcasts over batch and heads.
    ///
    /// Additive rather than `mask_fill` because the value must survive an f16
    /// cast: `-1e9` saturates to `-inf`, and no row is ever fully masked (the
    /// diagonal is always in the window), so no `inf - inf` NaN can appear.
    ///
    /// burn's `triu_mask`/`tril_mask` are *keep* masks — `false` marks the named
    /// triangle and `true` everything else — so they read inverted from the
    /// torch convention and are combined before the single negation.
    fn bias(&self, seq: usize, device: &Device) -> Tensor<4> {
        let past = Tensor::<2, Bool>::triu_mask([seq, seq], 1, device);
        let keep = match self.window {
            Some(w) => past.bool_and(Tensor::tril_mask([seq, seq], -(w as i64), device)),
            None => past,
        };
        keep.bool_not().float().mul_scalar(-1e9).reshape([1, 1, seq, seq])
    }
}
