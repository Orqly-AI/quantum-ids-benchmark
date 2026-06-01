#!/usr/bin/env bash
# One clean, correct setup of the qml env. Uses `python -m pip` so installs always
# land in the conda env, and isolates from ~/.local via PYTHONNOUSERSITE.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qml
export PYTHONNOUSERSITE=1

# Persist isolation for every future `conda activate qml`
ACT_DIR="$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$ACT_DIR"
echo 'export PYTHONNOUSERSITE=1' > "$ACT_DIR/00_nouser_site.sh"

echo ">> Ensuring pip exists INSIDE the env"
python -m ensurepip --upgrade
python -m pip install --quiet --upgrade pip
echo "env pip: $(python -m pip --version)"

echo ">> Installing PyTorch (CUDA 12.1) into env"
python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121

echo ">> Installing PennyLane + ML stack into env"
python -m pip install --quiet pennylane scikit-learn xgboost imbalanced-learn pandas numpy matplotlib seaborn tqdm

echo ">> Installing GPU quantum backend (lightning.gpu + cuQuantum custatevec)"
python -m pip install --quiet pennylane-lightning-gpu custatevec-cu12 || echo "WARN: lightning.gpu install issue (CPU fallback will be used)"

echo ">> SMOKE TEST"
python - <<'PY'
import torch
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "|", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu"))

import pennylane as qml
from pennylane import numpy as np
print("pennylane:", qml.version(), "| from:", qml.__file__)

n_qubits = 6
try:
    dev = qml.device("lightning.gpu", wires=n_qubits)
    backend = "lightning.gpu (CUDA)"
except Exception as e:
    dev = qml.device("lightning.qubit", wires=n_qubits)
    backend = f"lightning.qubit (CPU fallback) -- {type(e).__name__}: {e}"

@qml.qnode(dev)
def circuit(x, w):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    qml.StronglyEntanglingLayers(w, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=n_qubits)
print("Quantum backend:", backend)
print("Circuit output:", float(circuit(np.random.random(n_qubits), np.random.random(shape))))
PY
echo "ALL_DONE"
