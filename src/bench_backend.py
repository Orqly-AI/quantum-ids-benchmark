"""Quick CPU(lightning.qubit) vs GPU(lightning.gpu) timing for the HybridQNN,
at small qubit counts, to decide the default backend for the experiment sweeps.
"""
import time, sys
import numpy as np
import torch
from data import load
from quantum_model import HybridQNN

def bench(n_qubits, n_layers, n_train, epochs, force_cpu):
    Xtr, ytr, Xte, yte = load("nslkdd", n_features=n_qubits, reduction="pca", binary=True)
    Xtr, ytr = Xtr[:n_train], ytr[:n_train]
    m = HybridQNN(in_dim=n_qubits, n_qubits=n_qubits, n_layers=n_layers, encoding="angle")
    # Optionally force CPU lightning.qubit by swapping the device
    if force_cpu:
        import pennylane as qml
        from quantum_model import build_qnode
        dev = qml.device("lightning.qubit", wires=n_qubits)
        qnode, ws = build_qnode(n_qubits, n_layers, dev, "angle")
        m.qlayer = qml.qnn.TorchLayer(qnode, ws)
        m.backend = "lightning.qubit"
    dev_t = torch.device("cpu")  # keep torch on CPU; quantum layer dominates
    m = m.to(dev_t)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    lossf = torch.nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    t0 = time.time()
    for ep in range(epochs):
        for i in range(0, len(Xt), 64):
            opt.zero_grad()
            loss = lossf(m(Xt[i:i+64]), yt[i:i+64])
            loss.backward(); opt.step()
    dt = time.time() - t0
    print(f"backend={m.backend:18s} qubits={n_qubits} train={n_train} epochs={epochs} "
          f"=> {dt:.1f}s  ({dt/epochs:.1f}s/epoch)")
    return dt

if __name__ == "__main__":
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    n_train, epochs = 256, 1
    print(f"--- timing {n_train} samples x {epochs} epoch, {nq} qubits ---")
    bench(nq, 2, n_train, epochs, force_cpu=True)
    bench(nq, 2, n_train, epochs, force_cpu=False)
