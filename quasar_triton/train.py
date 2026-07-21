"""Single-GPU training loop for the upstream Mamba-3 Triton SSD path."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .config import ModelConfig, RunConfig
from .data import Batch, Batcher, Shards
from .model import Quasar
from .schedule import learning_rate


def _write_json(path: Path, value: dict[str, Any]) -> None:
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
  os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text())


def _prepare_run(out: Path, model: ModelConfig, run: RunConfig) -> None:
  out.mkdir(parents=True, exist_ok=True)
  expected = {"model.json": model.as_dict(), "run.json": run.as_dict()}
  for name, value in expected.items():
    path = out / name
    if path.exists():
      if _read_json(path) != value:
        raise ValueError(f"{path} does not match this invocation; use a new --out directory")
    else:
      _write_json(path, value)


def _checkpoint_directory(out: Path, step: int) -> Path:
  return out / f"checkpoint_{step:08d}"


def _latest_checkpoint(out: Path) -> Path | None:
  candidates = [
    path
    for path in out.glob("checkpoint_*")
    if path.is_dir() and (path / "checkpoint.pt").is_file()
  ]
  return max(candidates, key=lambda path: path.name, default=None)


def _save_checkpoint(
  out: Path,
  step: int,
  model: Quasar,
  optimizer: torch.optim.Optimizer,
  config: ModelConfig,
  run: RunConfig,
) -> Path:
  directory = _checkpoint_directory(out, step)
  directory.mkdir(parents=True, exist_ok=True)
  destination = directory / "checkpoint.pt"
  temporary = directory / "checkpoint.pt.tmp"
  torch.save(
    {
      "step": step,
      "model": model.state_dict(),
      "optimizer": optimizer.state_dict(),
      "model_config": config.as_dict(),
      "run_config": run.as_dict(),
    },
    temporary,
  )
  os.replace(temporary, destination)
  print(f"checkpoint step={step} path={destination}", flush=True)
  return directory


def _resume(
  out: Path,
  model: Quasar,
  optimizer: torch.optim.Optimizer,
  config: ModelConfig,
  run: RunConfig,
) -> int:
  directory = _latest_checkpoint(out)
  if directory is None:
    return 0
  state = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=False)
  if state["model_config"] != config.as_dict() or state["run_config"] != run.as_dict():
    raise ValueError(f"checkpoint configuration does not match {out}")
  model.load_state_dict(state["model"])
  optimizer.load_state_dict(state["optimizer"])
  step = int(state["step"])
  if step < 0 or step > run.steps:
    raise ValueError(f"checkpoint step {step} is outside a {run.steps}-step run")
  print(f"resume step={step} path={directory}", flush=True)
  return step


def _optimizer(model: Quasar, run: RunConfig) -> torch.optim.Optimizer:
  decay: list[torch.nn.Parameter] = []
  no_decay: list[torch.nn.Parameter] = []
  for parameter in model.parameters():
    if parameter.ndim < 2 or getattr(parameter, "_no_weight_decay", False):
      no_decay.append(parameter)
    else:
      decay.append(parameter)
  return torch.optim.AdamW(
    [
      {"params": decay, "weight_decay": run.weight_decay},
      {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=run.lr,
    betas=(0.9, 0.95),
    fused=True,
  )


def _loss(
  logits: torch.Tensor,
  targets: torch.Tensor,
  z_loss: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
  z = torch.logsumexp(logits, dim=-1).square().mean()
  return nll, z, nll + z_loss * z


def _tensor_batch(batch: Batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
  inputs = torch.tensor(batch.inputs, dtype=torch.long, device=device)
  targets = torch.tensor(batch.targets, dtype=torch.long, device=device)
  return inputs, targets


@torch.inference_mode()
def _evaluate(
  model: torch.nn.Module,
  raw_model: Quasar,
  batcher: Batcher,
  batches: int,
  device: torch.device,
) -> tuple[float, float, float]:
  was_training = raw_model.training
  raw_model.eval()
  nlls: list[torch.Tensor] = []
  zs: list[torch.Tensor] = []
  for index in range(min(batches, batcher.evaluation_batches)):
    inputs, targets = _tensor_batch(batcher.evaluation(index), device)
    nll, z, _ = _loss(model(inputs), targets, raw_model.config.z_loss)
    nlls.append(nll)
    zs.append(z)
  if was_training:
    raw_model.train()
  if not nlls:
    raise ValueError("validation split has no complete batches")
  nll = torch.stack(nlls).mean().item()
  z = torch.stack(zs).mean().item()
  bits_per_byte = nll * batcher.tokens_per_byte / math.log(2.0)
  return nll, z, bits_per_byte


def _check_runtime() -> str:
  if not sys.platform.startswith("linux"):
    raise RuntimeError(
      "the official Mamba/Triton package targets Linux; native Windows ROCm wheels "
      "cannot run this kernel"
    )
  if not torch.cuda.is_available():
    raise RuntimeError("PyTorch reports no CUDA/HIP device")
  try:
    import triton
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import (
      mamba3_siso_combined,
    )
  except ImportError as error:
    raise RuntimeError(f"upstream Mamba-3 Triton SSD is unavailable: {error}") from error
  if not callable(mamba3_siso_combined):
    raise RuntimeError("mamba3_siso_combined is not callable")
  # Upstream's SISO wrapper casts kernel operands to bf16 even when surrounding
  # module weights are fp32.
  if not torch.cuda.is_bf16_supported():
    raise RuntimeError("the upstream SISO kernel requires bfloat16 support")
  backend = f"ROCm {torch.version.hip}" if torch.version.hip else f"CUDA {torch.version.cuda}"
  return f"{backend}; Triton {triton.__version__}"


def train(preset: str, data: Path, out: Path, run: RunConfig) -> None:
  """Train one preset, automatically resuming the newest complete checkpoint."""
  run.validate()
  if run.compile:
    # Static batches plus expandable segments avoid fragmentation during the
    # compiler's warm-up. Respect an allocator configuration supplied by the user.
    allocator = "expandable_segments:True"
    os.environ.setdefault("PYTORCH_ALLOC_CONF", allocator)
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", allocator)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", allocator)

  dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[run.dtype]
  runtime = _check_runtime()
  device = torch.device("cuda")  # PyTorch intentionally uses this name for HIP too.
  torch.set_float32_matmul_precision("high")
  torch.manual_seed(run.seed)
  torch.cuda.manual_seed_all(run.seed)

  with Shards(data / "train") as train_shards, Shards(data / "valid") as valid_shards:
    if train_shards.meta.vocab_size != valid_shards.meta.vocab_size:
      raise ValueError("train and valid vocabulary sizes differ")
    config = ModelConfig.preset(preset, train_shards.meta.vocab_size)
    _prepare_run(out, config, run)
    train_batches = Batcher(
      train_shards,
      config.seq_len,
      run.micro_batch,
      run.seed,
    )
    valid_batches = Batcher(
      valid_shards,
      config.seq_len,
      run.micro_batch,
      run.seed,
    )

    raw_model = Quasar(config, device, dtype, run.checkpointing)
    parameters = sum(parameter.numel() for parameter in raw_model.parameters())
    if parameters != config.parameter_budget():
      raise RuntimeError(
        f"upstream model shape changed: built {parameters:,} parameters, "
        f"expected {config.parameter_budget():,}"
      )
    optimizer = _optimizer(raw_model, run)
    completed = _resume(out, raw_model, optimizer, config, run)
    model: torch.nn.Module = raw_model
    if run.compile:
      model = torch.compile(
        raw_model,
        dynamic=False,
        mode="max-autotune-no-cudagraphs",
      )

    raw_model.train()
    tokens_per_step = run.tokens_per_step(config)
    print(
      f"device={torch.cuda.get_device_name(0)} backend={runtime} "
      f"kernel={config.ssd_kernel} dtype={run.dtype}",
      flush=True,
    )
    print(
      f"preset={preset} params={parameters:,} seq={config.seq_len} "
      f"effective_batch={run.micro_batch * run.accum} "
      f"tokens/step={tokens_per_step:,} total_tokens={run.total_tokens(config):,} "
      f"checkpointing={run.checkpointing} compile={run.compile}",
      flush=True,
    )

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    window_started = time.perf_counter()
    window_start_step = completed
    try:
      while completed < run.steps:
        step = completed
        rate = learning_rate(
          step,
          run.lr,
          run.lr_floor,
          run.warmup,
          run.decay,
          run.steps,
        )
        for group in optimizer.param_groups:
          group["lr"] = rate

        nll_sum = torch.zeros((), device=device)
        z_sum = torch.zeros((), device=device)
        for micro in range(run.accum):
          batch_index = step * run.accum + micro
          inputs, targets = _tensor_batch(train_batches.train(batch_index), device)
          nll, z, total = _loss(model(inputs), targets, config.z_loss)
          (total / run.accum).backward()
          nll_sum += nll.detach()
          z_sum += z.detach()

        if run.clip:
          torch.nn.utils.clip_grad_norm_(raw_model.parameters(), run.clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        completed += 1

        if completed % run.log_every == 0 or completed == run.steps:
          torch.cuda.synchronize()
          elapsed = time.perf_counter() - window_started
          window_steps = completed - window_start_step
          throughput = window_steps * tokens_per_step / elapsed
          remaining = (run.steps - completed) * tokens_per_step
          eta_hours = remaining / throughput / 3_600.0
          tflops = throughput * 3.0 * config.flops_per_token() / 1e12
          print(
            f"train step={completed}/{run.steps} "
            f"nll={(nll_sum / run.accum).item():.5f} "
            f"z={(z_sum / run.accum).item():.4f} lr={rate:.3e} "
            f"tok/s={throughput:,.0f} tflop/s={tflops:.2f} eta_h={eta_hours:.2f}",
            flush=True,
          )
          window_started = time.perf_counter()
          window_start_step = completed

        ancillary_work = False
        if run.eval_every and completed % run.eval_every == 0:
          nll, z, bits_per_byte = _evaluate(
            model,
            raw_model,
            valid_batches,
            run.eval_batches,
            device,
          )
          print(
            f"valid step={completed} nll={nll:.5f} ppl={math.exp(min(nll, 20.0)):.3f} "
            f"z={z:.4f} bits/byte={bits_per_byte:.5f}",
            flush=True,
          )
          ancillary_work = True

        if run.save_every and completed % run.save_every == 0:
          _save_checkpoint(out, completed, raw_model, optimizer, config, run)
          ancillary_work = True
        if ancillary_work:
          torch.cuda.synchronize()
          window_started = time.perf_counter()
          window_start_step = completed
    except KeyboardInterrupt:
      print("interrupted; saving the last complete optimizer step", flush=True)
      _save_checkpoint(out, completed, raw_model, optimizer, config, run)
      return

    if not run.save_every or completed % run.save_every:
      _save_checkpoint(out, completed, raw_model, optimizer, config, run)
    print(f"complete step={completed} tokens={completed * tokens_per_step:,}", flush=True)
