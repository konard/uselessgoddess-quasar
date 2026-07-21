"""Warmup-stable-decay learning-rate schedule."""

import math


def learning_rate(
  step: int,
  peak: float,
  floor_ratio: float,
  warmup: int,
  decay: int,
  steps: int,
) -> float:
  if warmup + decay > steps:
    raise ValueError(f"warmup and decay overlap in {steps} steps")
  if step < 0 or step >= steps:
    raise ValueError(f"step {step} outside a {steps}-step run")
  if step < warmup:
    return peak * (step + 1) / warmup
  stable_end = steps - decay
  if step < stable_end:
    return peak
  progress = (step - stable_end + 1) / decay
  floor = peak * floor_ratio
  return floor + (peak - floor) * (1.0 - math.sqrt(progress))
