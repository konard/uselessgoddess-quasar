"""Read Quasar's little-endian ``u16`` shards without copying the corpus."""

import bisect
import json
import mmap
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Meta:
  tokens: int
  docs: int
  bytes: int
  vocab_size: int
  eos: int


@dataclass(frozen=True)
class Batch:
  inputs: tuple[tuple[int, ...], ...]
  targets: tuple[tuple[int, ...], ...]


class Shards:
  """A read-only, memory-mapped concatenation of prepared token shards."""

  def __init__(self, directory: str | Path):
    self.directory = Path(directory)
    self.meta = Meta(**json.loads((self.directory / "meta.json").read_text()))
    paths = sorted(self.directory.glob("*.bin"))
    if not paths:
      raise FileNotFoundError(f"no .bin shards in {self.directory}")

    self._files: list[Any] = []
    self._maps: list[mmap.mmap] = []
    self._starts = [0]
    try:
      for path in paths:
        file = path.open("rb")
        if path.stat().st_size % 2:
          file.close()
          raise ValueError(f"odd byte count in u16 shard {path}")
        mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        self._files.append(file)
        self._maps.append(mapping)
        self._starts.append(self._starts[-1] + len(mapping) // 2)
    except Exception:
      self.close()
      raise

    if len(self) != self.meta.tokens:
      self.close()
      raise ValueError(
        f"meta says {self.meta.tokens} tokens but shards contain {len(self)}"
      )

  def __len__(self) -> int:
    return self._starts[-1]

  def read(self, start: int, length: int) -> list[int]:
    if start < 0 or length < 0 or start + length > len(self):
      raise IndexError(f"window {start}+{length} outside {len(self)} tokens")
    output: list[int] = []
    shard = bisect.bisect_right(self._starts, start) - 1
    cursor = start
    while len(output) < length:
      local = cursor - self._starts[shard]
      take = min(length - len(output), len(self._maps[shard]) // 2 - local)
      view = self._maps[shard][local * 2 : (local + take) * 2]
      output.extend(value[0] for value in struct.iter_unpack("<H", view))
      cursor += take
      shard += 1
    return output

  def close(self) -> None:
    for mapping in getattr(self, "_maps", ()):
      mapping.close()
    for file in getattr(self, "_files", ()):
      file.close()
    self._maps = []
    self._files = []

  def __enter__(self) -> "Shards":
    return self

  def __exit__(self, *_error: object) -> None:
    self.close()


class Batcher:
  """Deterministic fixed-shape windows suitable for ``torch.compile``."""

  def __init__(self, shards: Shards, seq_len: int, batch_size: int, seed: int):
    if len(shards) <= seq_len + 1:
      raise ValueError("corpus is shorter than one training window")
    if batch_size <= 0:
      raise ValueError("batch_size must be positive")
    self.shards = shards
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.seed = seed

  def train(self, index: int) -> Batch:
    randomizer = random.Random(self.seed ^ index)
    last = len(self.shards) - self.seq_len - 1
    starts = [randomizer.randrange(last + 1) for _ in range(self.batch_size)]
    return self._gather(starts)

  def evaluation(self, index: int) -> Batch:
    starts = [
      (index * self.batch_size + sample) * self.seq_len
      for sample in range(self.batch_size)
    ]
    return self._gather(starts)

  @property
  def evaluation_batches(self) -> int:
    return (len(self.shards) - 1) // self.seq_len // self.batch_size

  @property
  def tokens_per_byte(self) -> float:
    return self.shards.meta.tokens / self.shards.meta.bytes

  def _gather(self, starts: list[int]) -> Batch:
    inputs: list[tuple[int, ...]] = []
    targets: list[tuple[int, ...]] = []
    for start in starts:
      window = self.shards.read(start, self.seq_len + 1)
      inputs.append(tuple(window[:-1]))
      targets.append(tuple(window[1:]))
    return Batch(tuple(inputs), tuple(targets))

