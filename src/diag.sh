#!/usr/bin/env bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qml
export PYTHONNOUSERSITE=1
echo "which python: $(which python)"
echo "which pip:    $(which pip)"
echo "pip points to:"
pip --version
python - <<'PY'
import sys
print("sys.prefix:", sys.prefix)
print("--- sys.path ---")
for p in sys.path:
    print(p)
PY
