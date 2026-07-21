#!/usr/bin/env bash
set -euo pipefail

venv_dir=.venv-triton
python3.12 -m venv "${venv_dir}"
source "${venv_dir}/bin/activate"

python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
python -m pip install -e .
python -m pip install --no-build-isolation --no-deps \
  "mamba-ssm @ git+https://github.com/state-spaces/mamba.git@f577286d052741c35d39cd43bdc3fad27120f22c"

python - <<'PY'
import torch
import triton
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined

assert callable(mamba3_siso_combined)
print(f"torch={torch.__version__} hip={torch.version.hip} triton={triton.__version__}")
print("upstream mamba3_siso_combined import: ok")
PY
