import ast
from pathlib import Path
import unittest


class TritonModelContractTest(unittest.TestCase):
  def test_mamba_layers_select_upstream_siso_kernel(self):
    source = Path(__file__).parents[1] / "quasar_triton" / "model.py"
    tree = ast.parse(source.read_text())
    calls = [
      node
      for node in ast.walk(tree)
      if isinstance(node, ast.Call)
      and isinstance(node.func, ast.Name)
      and node.func.id == "Mamba3"
    ]

    self.assertEqual(len(calls), 1)
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    self.assertIs(keywords["is_mimo"].value, False)
    self.assertEqual(ast.unparse(keywords["chunk_size"]), "config.chunk_size")


if __name__ == "__main__":
  unittest.main()
