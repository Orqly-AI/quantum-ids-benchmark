#!/usr/bin/env bash
# Installs PennyLane + GPU quantum backend + ML stack into the qml env, then smoke-tests.
# Isolates from ~/.local user site-packages (PYTHONNOUSERSITE) to avoid leakage.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qml
export PYTHONNOUSERSITE=1

# Make the isolation permanent for this env (every future `conda activate qml`)
ACT_DIR="$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$ACT_DIR"
echo 'export PYTHONNOUSERSITE=1' > "$ACT_DIR/00_nouser_site.sh"

echo ">> Python in use: $(which python)"

echo ">> Installing PennyLane + ML stack (into env, ignoring ~/.local)"
pip install --quiet --ignore-installed pennylane scikit-learn xgboost imbalanced-learn pandas numpy matplotlib seaborn tqdm

echo ">> Installing GPU quantum backend (lightning.gpu + cuQuantum custatevec)"
pip install --quiet pennylane-lightning-gpu custatevec-cu12 || echo "WARN: lightning.gpu install issue (will fall back to lightning.qubit on CPU)"

echo ">> Smoke test: VQC"
python - <<'PY'
import pennylane as qml
from pennylane import numpy as np
print("PennyLane loaded from:", qml.__file__)

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
w = np.random.random(shape)
x = np.random.random(n_qubits)
print("Backend:", backend)
print("Circuit output:", float(circuit(x, w)))
print("PennyLane version:", qml.version())
PY
echo "QUANTUM_STEP_DONE"
