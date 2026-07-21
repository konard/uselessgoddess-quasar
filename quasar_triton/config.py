"""Dependency-free model and training configuration for the Triton path."""

from dataclasses import asdict, dataclass
from typing import Any


def _round_up(value: float, multiple: int) -> int:
  return int(value + multiple - 1) // multiple * multiple


@dataclass(frozen=True)
class ModelConfig:
  """One Quasar preset expressed in upstream Mamba-3 terms.

  The optimized upstream Triton implementation is SISO. Mamba-3 MIMO is a
  separate TileLang kernel, so this path deliberately fixes ``mimo_rank`` to 1
  instead of claiming to use Triton while dispatching to another backend.
  """

  vocab_size: int
  d_model: int
  n_layers: int
  seq_len: int = 2_048
  state_rank: int = 128
  expand: int = 2
  head_dim: int = 64
  n_groups: int = 1
  mimo_rank: int = 1
  rope_fraction: float = 0.5
  attn_period: int | None = None
  attn_heads: int = 12
  attn_kv_heads: int = 2
  attn_window: int | None = 1_024
  ffn_mult: float = 2.5
  tied_embeddings: bool = True
  z_loss: float = 1e-4
  chunk_size: int = 64

  @classmethod
  def tiny(cls, vocab_size: int) -> "ModelConfig":
    return cls(
      vocab_size=vocab_size,
      d_model=640,
      n_layers=24,
      n_groups=2,
      attn_period=6,
      attn_heads=10,
      attn_kv_heads=2,
    )

  @classmethod
  def base(cls, vocab_size: int) -> "ModelConfig":
    return cls(
      vocab_size=vocab_size,
      d_model=1_536,
      n_layers=28,
      n_groups=4,
      attn_period=7,
      attn_heads=24,
      attn_kv_heads=4,
      tied_embeddings=False,
    )

  @classmethod
  def toy(cls, vocab_size: int) -> "ModelConfig":
    return cls(
      vocab_size=vocab_size,
      d_model=32,
      n_layers=4,
      seq_len=16,
      state_rank=8,
      head_dim=8,
      attn_period=2,
      attn_heads=4,
      attn_kv_heads=2,
      attn_window=8,
      ffn_mult=2.0,
      chunk_size=16,
    )

  @classmethod
  def preset(cls, name: str, vocab_size: int) -> "ModelConfig":
    constructors = {"tiny": cls.tiny, "base": cls.base, "toy": cls.toy}
    try:
      config = constructors[name](vocab_size)
    except KeyError as error:
      raise ValueError(f"unknown preset {name!r}") from error
    config.validate()
    return config

  @property
  def ssd_kernel(self) -> str:
    return "mamba3_siso_combined"

  @property
  def d_inner(self) -> int:
    return self.expand * self.d_model

  @property
  def ssm_heads(self) -> int:
    return self.d_inner // self.head_dim

  @property
  def d_ff(self) -> int:
    return _round_up(self.d_model * self.ffn_mult, 64)

  @property
  def attention_layers(self) -> tuple[int, ...]:
    if not self.attn_period:
      return ()
    return tuple(
      layer for layer in range(self.n_layers) if (layer + 1) % self.attn_period == 0
    )

  def validate(self) -> None:
    if self.mimo_rank != 1:
      raise ValueError("the Triton SSD path is SISO and requires mimo_rank=1")
    if self.state_rank % 2:
      raise ValueError("state_rank must be even for data-dependent RoPE")
    if self.d_inner % self.head_dim:
      raise ValueError("head_dim must divide expand * d_model")
    if self.ssm_heads % self.n_groups:
      raise ValueError("n_groups must divide the number of SSM heads")
    if self.attn_period:
      if self.d_model % self.attn_heads:
        raise ValueError("attn_heads must divide d_model")
      if self.attn_heads % self.attn_kv_heads:
        raise ValueError("attn_kv_heads must divide attn_heads")
    if self.chunk_size <= 0 or self.seq_len % self.chunk_size:
      raise ValueError("chunk_size must divide seq_len")

  def parameter_budget(self) -> int:
    """Analytic count matching :mod:`quasar_triton.model`."""
    d = self.d_model
    embedding = self.vocab_size * d
    head = 0 if self.tied_embeddings else self.vocab_size * d
    angles = int(self.state_rank * self.rope_fraction) // 2
    in_projection = (
      2 * self.d_inner
      + 2 * self.state_rank * self.n_groups
      + 3 * self.ssm_heads
      + angles
    )
    ssm_small = (
      2 * self.ssm_heads
      + 2 * self.state_rank
      + 2 * self.ssm_heads * self.state_rank
      + self.d_inner
    )
    one_ssm = d * in_projection + self.d_inner * d + ssm_small
    head_dim = d // self.attn_heads
    one_attention = (
      2 * d * d + 2 * d * self.attn_kv_heads * head_dim + 2 * head_dim
    )
    attention_count = len(self.attention_layers)
    ssm = (self.n_layers - attention_count) * one_ssm
    attention = attention_count * one_attention
    ffn = self.n_layers * 3 * d * self.d_ff
    norms = (2 * self.n_layers + 1) * d
    return embedding + head + ssm + attention + ffn + norms

  def flops_per_token(self) -> float:
    """Approximate forward FLOPs, counting a multiply-add as two."""
    d = self.d_model
    angles = int(self.state_rank * self.rope_fraction) // 2
    in_projection = (
      2 * self.d_inner
      + 2 * self.state_rank * self.n_groups
      + 3 * self.ssm_heads
      + angles
    )
    ssm_state = self.ssm_heads * self.head_dim * self.state_rank
    ssm = 2 * d * (in_projection + self.d_inner) + 4 * ssm_state
    head_dim = d // self.attn_heads
    kv = self.attn_kv_heads * head_dim
    span = min(self.attn_window or self.seq_len, self.seq_len)
    attention = 2 * d * (2 * d + 2 * kv) + 4 * span * d
    mixers = (
      (self.n_layers - len(self.attention_layers)) * ssm
      + len(self.attention_layers) * attention
    )
    ffn = self.n_layers * 6 * d * self.d_ff
    unembedding = 2 * d * self.vocab_size
    return float(mixers + ffn + unembedding)

  def as_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass(frozen=True)
class RunConfig:
  steps: int = 12_500
  micro_batch: int = 1
  accum: int = 128
  lr: float = 3e-3
  lr_floor: float = 0.1
  warmup: int = 400
  decay: int = 2_500
  weight_decay: float = 0.1
  clip: float = 1.0
  seed: int = 1_337
  log_every: int = 20
  eval_every: int = 1_000
  eval_batches: int = 20
  save_every: int = 2_000
  checkpointing: bool = True
  compile: bool = False
  dtype: str = "bfloat16"

  def validate(self) -> None:
    positive = {
      "steps": self.steps,
      "micro_batch": self.micro_batch,
      "accum": self.accum,
      "lr": self.lr,
      "log_every": self.log_every,
      "eval_batches": self.eval_batches,
    }
    for name, value in positive.items():
      if value <= 0:
        raise ValueError(f"{name} must be positive")
    if self.warmup < 0 or self.decay < 0 or self.warmup + self.decay > self.steps:
      raise ValueError("warmup and decay must be non-negative and fit within steps")
    if not 0.0 <= self.lr_floor <= 1.0:
      raise ValueError("lr_floor must be between zero and one")
    if self.weight_decay < 0.0 or self.clip < 0.0:
      raise ValueError("weight_decay and clip must be non-negative")
    if self.save_every < 0 or self.eval_every < 0:
      raise ValueError("save_every and eval_every must be non-negative")
    if self.dtype not in {"bfloat16", "float32"}:
      raise ValueError(f"unsupported dtype {self.dtype!r}")

  def tokens_per_step(self, model: ModelConfig) -> int:
    return self.micro_batch * self.accum * model.seq_len

  def total_tokens(self, model: ModelConfig) -> int:
    return self.steps * self.tokens_per_step(model)

  def steps_for_tokens(self, model: ModelConfig, tokens: int) -> int:
    per_step = self.tokens_per_step(model)
    return (tokens + per_step - 1) // per_step

  def as_dict(self) -> dict[str, Any]:
    return asdict(self)
