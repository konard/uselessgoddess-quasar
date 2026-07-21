import unittest

from quasar_triton.schedule import learning_rate


class WsdTest(unittest.TestCase):
  def test_warmup_stable_and_decay_phases_join(self):
    args = dict(peak=3e-3, floor_ratio=0.1, warmup=400, decay=2_500, steps=12_500)

    self.assertAlmostEqual(learning_rate(step=399, **args), 3e-3)
    self.assertAlmostEqual(learning_rate(step=400, **args), 3e-3)
    self.assertAlmostEqual(learning_rate(step=9_999, **args), 3e-3)
    self.assertAlmostEqual(learning_rate(step=12_499, **args), 3e-4)


if __name__ == "__main__":
  unittest.main()
