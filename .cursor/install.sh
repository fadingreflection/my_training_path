#!/usr/bin/env bash
# Idempotent environment bootstrap for the my_training_path project.
# Creates a Python virtualenv and installs the pinned CPU dependency stack.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Ensure a venv module is available (Debian/Ubuntu splits it out).
if ! python3 -c "import venv, ensurepip" >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y "python3-venv" || sudo apt-get install -y "python$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')-venv"
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt

echo "Environment ready. Python: $(python --version)"
python - <<'PY'
import torch, transformers, trl, peft, datasets
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"transformers={transformers.__version__} trl={trl.__version__} "
      f"peft={peft.__version__} datasets={datasets.__version__}")
PY
