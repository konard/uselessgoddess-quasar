"""Hybrid Quasar model using upstream's fused Mamba-3 Triton SSD."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba3 import Mamba3
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


class RotaryEmbedding(nn.Module):
  def __init__(self, length: int, head_dim: int, device: torch.device):
    super().__init__()
    inverse = 1.0 / (
      10_000 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    angles = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inverse)
    self.register_buffer("cos", angles.cos(), persistent=False)
    self.register_buffer("sin", angles.sin(), persistent=False)

  def forward(self, value: torch.Tensor) -> torch.Tensor:
    length = value.shape[-2]
    cos = self.cos[:length].to(dtype=value.dtype)
    sin = self.sin[:length].to(dtype=value.dtype)
    even, odd = value[..., 0::2], value[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
  """Sliding-window GQA dispatched through PyTorch's fused SDPA."""

  def __init__(self, config: ModelConfig, device: torch.device, dtype: torch.dtype):
    super().__init__()
    d = config.d_model
    self.heads = config.attn_heads
    self.kv_heads = config.attn_kv_heads
    self.head_dim = d // self.heads
    kwargs = {"device": device, "dtype": dtype, "bias": False}
    self.q = nn.Linear(d, d, **kwargs)
    self.k = nn.Linear(d, self.kv_heads * self.head_dim, **kwargs)
    self.v = nn.Linear(d, self.kv_heads * self.head_dim, **kwargs)
    self.out = nn.Linear(d, d, **kwargs)
    self.q_norm = nn.RMSNorm(self.head_dim, device=device, dtype=dtype)
    self.k_norm = nn.RMSNorm(self.head_dim, device=device, dtype=dtype)
    self.rotary = RotaryEmbedding(config.seq_len, self.head_dim, device)

    row = torch.arange(config.seq_len, device=device)[:, None]
    column = torch.arange(config.seq_len, device=device)[None, :]
    keep = column <= row
    if config.attn_window is not None:
      keep &= column > row - config.attn_window
    self.register_buffer("mask", keep, persistent=False)

    for projection in (self.q, self.k, self.v):
      nn.init.normal_(projection.weight, mean=0.0, std=0.02)
    nn.init.normal_(
      self.out.weight,
      mean=0.0,
      std=0.02 / math.sqrt(2.0 * config.n_layers),
    )

  def forward(self, value: torch.Tensor) -> torch.Tensor:
    batch, length, _ = value.shape

    def heads(projected: torch.Tensor, count: int) -> torch.Tensor:
      return projected.view(batch, length, count, self.head_dim).transpose(1, 2)

    query = self.rotary(self.q_norm(heads(self.q(value), self.heads)))
    key = self.rotary(self.k_norm(heads(self.k(value), self.kv_heads)))
    values = heads(self.v(value), self.kv_heads)
    mixed = F.scaled_dot_product_attention(
      query,
      key,
      values,
      attn_mask=self.mask[:length, :length],
      dropout_p=0.0,
      enable_gqa=True,
    )
    return self.out(mixed.transpose(1, 2).contiguous().view(batch, length, -1))


class FeedForward(nn.Module):
  def __init__(self, config: ModelConfig, device: torch.device, dtype: torch.dtype):
    super().__init__()
    kwargs = {"device": device, "dtype": dtype, "bias": False}
    self.gate = nn.Linear(config.d_model, config.d_ff, **kwargs)
    self.up = nn.Linear(config.d_model, config.d_ff, **kwargs)
    self.down = nn.Linear(config.d_ff, config.d_model, **kwargs)
    nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
    nn.init.normal_(self.up.weight, mean=0.0, std=0.02)
    nn.init.normal_(
      self.down.weight,
      mean=0.0,
      std=0.02 / math.sqrt(2.0 * config.n_layers),
    )

  def forward(self, value: torch.Tensor) -> torch.Tensor:
    return self.down(F.silu(self.gate(value)) * self.up(value))


class Block(nn.Module):
  def __init__(
    self,
    config: ModelConfig,
    layer: int,
    device: torch.device,
    dtype: torch.dtype,
  ):
    super().__init__()
    self.norm_mixer = nn.RMSNorm(config.d_model, device=device, dtype=dtype)
    if layer in config.attention_layers:
      self.mixer = Attention(config, device, dtype)
    else:
      self.mixer = Mamba3(
        d_model=config.d_model,
        d_state=config.state_rank,
        expand=config.expand,
        headdim=config.head_dim,
        ngroups=config.n_groups,
        rope_fraction=config.rope_fraction,
        is_outproj_norm=True,
        is_mimo=False,
        chunk_size=config.chunk_size,
        layer_idx=layer,
        device=device,
        dtype=dtype,
      )
      # Match the residual-aware initialization used by the upstream LM stack.
      with torch.no_grad():
        self.mixer.out_proj.weight.div_(math.sqrt(2.0 * config.n_layers))
    self.norm_ffn = nn.RMSNorm(config.d_model, device=device, dtype=dtype)
    self.ffn = FeedForward(config, device, dtype)

  def forward(self, value: torch.Tensor) -> torch.Tensor:
    value = value + self.mixer(self.norm_mixer(value))
    return value + self.ffn(self.norm_ffn(value))


@dataclass
class Loss:
  nll: torch.Tensor
  z: torch.Tensor
  total: torch.Tensor


class Quasar(nn.Module):
  def __init__(
    self,
    config: ModelConfig,
    device: torch.device,
    dtype: torch.dtype,
    checkpointing: bool,
  ):
    super().__init__()
    config.validate()
    self.config = config
    self.checkpointing = checkpointing
    self.embedding = nn.Embedding(config.vocab_size, config.d_model, device=device, dtype=dtype)
    nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    self.blocks = nn.ModuleList(
      Block(config, layer, device, dtype) for layer in range(config.n_layers)
    )
    self.norm = nn.RMSNorm(config.d_model, device=device, dtype=dtype)
    self.head = None
    if not config.tied_embeddings:
      self.head = nn.Linear(
        config.d_model,
        config.vocab_size,
        bias=False,
        device=device,
        dtype=dtype,
      )
      nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    value = self.embedding(tokens)
    for block in self.blocks:
      if self.checkpointing and self.training and torch.is_grad_enabled():
        value = checkpoint(block, value, use_reentrant=False)
      else:
        value = block(value)
    value = self.norm(value)
    weight = self.embedding.weight if self.head is None else self.head.weight
    return F.linear(value, weight)

  def loss(self, tokens: torch.Tensor, targets: torch.Tensor) -> Loss:
    logits = self(tokens)
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    log_normalizer = torch.logsumexp(logits, dim=-1)
    z = log_normalizer.square().mean()
    return Loss(nll=nll, z=z, total=nll + self.config.z_loss * z)
