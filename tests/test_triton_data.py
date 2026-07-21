import json
import struct
import tempfile
import unittest
from pathlib import Path

from quasar_triton.data import Batcher, Shards


def write_shards(root: Path, tokens: list[int], split_at: int = 9) -> None:
  root.mkdir(parents=True, exist_ok=True)
  (root / "meta.json").write_text(
    json.dumps(
      {
        "tokens": len(tokens),
        "docs": 1,
        "bytes": len(tokens) * 2,
        "vocab_size": 64,
        "eos": 0,
      }
    )
  )
  for index, shard in enumerate((tokens[:split_at], tokens[split_at:])):
    (root / f"shard_{index:04}.bin").write_bytes(
      struct.pack(f"<{len(shard)}H", *shard)
    )


class ShardsTest(unittest.TestCase):
  def test_a_window_can_cross_a_shard_boundary(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      write_shards(root, list(range(32)))

      with Shards(root) as shards:
        self.assertEqual(shards.read(7, 6), [7, 8, 9, 10, 11, 12])

  def test_batches_are_deterministic_and_shift_targets_by_one(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      write_shards(root, list(range(64)), split_at=31)

      with Shards(root) as shards:
        batcher = Batcher(shards, seq_len=8, batch_size=2, seed=7)
        first = batcher.train(41)
        again = batcher.train(41)

      self.assertEqual(first, again)
      for inputs, targets in zip(first.inputs, first.targets, strict=True):
        self.assertEqual(inputs[1:], targets[:-1])


if __name__ == "__main__":
  unittest.main()
