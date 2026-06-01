#!/usr/bin/env bash
# Sets up the qml conda env with the full GPU quantum stack. Idempotent-ish.
set -e
source ~/miniconda3/etc/profile.d/conda.sh

if ! conda env list | grep -q "^qml "; then
  echo ">> Creating conda env qml (python 3.10)"
  conda create -y -n qml -c conda-forge --override-channels python=3.10 >/dev/null
fi
conda activate qml

echo ">> Installing PyTorch (CUDA 12.1)"
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121

echo ">> Verifying GPU"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory/1e9, 2))
PY
echo "TORCH_STEP_DONE"
