import unittest

from quasar_triton.config import ModelConfig, RunConfig


class TritonPresetTest(unittest.TestCase):
  def test_tiny_uses_the_upstream_triton_ssd_contract(self):
    config = ModelConfig.tiny(vocab_size=32_768)

    self.assertEqual(config.ssd_kernel, "mamba3_siso_combined")
    self.assertEqual(config.mimo_rank, 1)
    self.assertEqual(config.chunk_size, 64)
    self.assertEqual(config.attention_layers, (5, 11, 17, 23))
    self.assertTrue(100_000_000 <= config.parameter_budget() <= 200_000_000)

  def test_default_run_keeps_the_compute_efficient_token_budget(self):
    model = ModelConfig.tiny(vocab_size=32_768)
    run = RunConfig()

    self.assertEqual((run.micro_batch, run.accum), (1, 128))
    self.assertEqual(run.tokens_per_step(model), 262_144)
    self.assertTrue(3_000_000_000 <= run.total_tokens(model) <= 3_500_000_000)
    self.assertEqual(run.steps_for_tokens(model, 10_000_000_000), 38_147)

  def test_invalid_run_fails_before_allocating_a_model(self):
    with self.assertRaisesRegex(ValueError, "warmup and decay"):
      RunConfig(steps=10, warmup=6, decay=5).validate()


if __name__ == "__main__":
  unittest.main()
