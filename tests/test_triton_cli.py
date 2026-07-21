import subprocess
import sys
import unittest


class TritonCliTest(unittest.TestCase):
  def test_help_is_available_without_importing_gpu_dependencies(self):
    completed = subprocess.run(
      [sys.executable, "-m", "quasar_triton", "--help"],
      check=False,
      capture_output=True,
      text=True,
    )

    self.assertEqual(completed.returncode, 0, completed.stderr)
    self.assertIn("upstream Mamba-3 Triton SSD", completed.stdout)
    self.assertIn("--compile", completed.stdout)


if __name__ == "__main__":
  unittest.main()
