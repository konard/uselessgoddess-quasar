"""Command-line parsing without importing PyTorch or allocating a GPU."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import RunConfig


DESCRIPTION = "Train Quasar with the upstream Mamba-3 Triton SSD kernel."


def build_parser() -> argparse.ArgumentParser:
  defaults = RunConfig()
  parser = argparse.ArgumentParser(description=DESCRIPTION)
  parser.add_argument("preset", choices=("tiny", "base", "toy"), nargs="?", default="tiny")
  parser.add_argument("--data", type=Path, default=Path("data/shards"))
  parser.add_argument("--out", type=Path, default=None)
  parser.add_argument("--steps", type=int, default=defaults.steps)
  parser.add_argument("--micro-batch", type=int, default=defaults.micro_batch)
  parser.add_argument("--accum", type=int, default=defaults.accum)
  parser.add_argument("--lr", type=float, default=defaults.lr)
  parser.add_argument("--lr-floor", type=float, default=defaults.lr_floor)
  parser.add_argument("--warmup", type=int, default=defaults.warmup)
  parser.add_argument("--decay", type=int, default=defaults.decay)
  parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
  parser.add_argument("--clip", type=float, default=defaults.clip)
  parser.add_argument("--seed", type=int, default=defaults.seed)
  parser.add_argument("--log-every", type=int, default=defaults.log_every)
  parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
  parser.add_argument("--eval-batches", type=int, default=defaults.eval_batches)
  parser.add_argument("--save-every", type=int, default=defaults.save_every)
  parser.add_argument(
    "--checkpointing",
    action=argparse.BooleanOptionalAction,
    default=defaults.checkpointing,
    help="recompute block activations to reduce VRAM (default: enabled)",
  )
  parser.add_argument(
    "--compile",
    action=argparse.BooleanOptionalAction,
    default=defaults.compile,
    help="experimentally wrap the model in torch.compile (Triton SSD is used either way)",
  )
  parser.add_argument(
    "--dtype",
    choices=("bfloat16", "float32"),
    default=defaults.dtype,
  )
  return parser


def run_config(args: argparse.Namespace) -> RunConfig:
  return RunConfig(
    steps=args.steps,
    micro_batch=args.micro_batch,
    accum=args.accum,
    lr=args.lr,
    lr_floor=args.lr_floor,
    warmup=args.warmup,
    decay=args.decay,
    weight_decay=args.weight_decay,
    clip=args.clip,
    seed=args.seed,
    log_every=args.log_every,
    eval_every=args.eval_every,
    eval_batches=args.eval_batches,
    save_every=args.save_every,
    checkpointing=args.checkpointing,
    compile=args.compile,
    dtype=args.dtype,
  )


def main(argv: Sequence[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  run = run_config(args)
  out = args.out or Path("runs") / f"{args.preset}-triton"

  # Keep `--help` usable on corpus-preparation and CI hosts without GPU wheels.
  try:
    from .train import train
  except ImportError as error:
    raise SystemExit(
      "Triton training dependencies are unavailable. Follow docs/TRITON.md; "
      f"the first import error was: {error}"
    ) from error

  train(args.preset, args.data, out, run)
  return 0
