"""QNN throughput benchmark (task #14): measure real HybridQNN train/eval cost on
lightning.qubit and recommend a feasible #7 sweep budget.

Times forward and forward+backward at several qubit counts / depths / batch sizes,
checks whether qml.qnn.TorchLayer broadcasts a batch in one device call or loops
sample-by-sample (the suspected bottleneck), and confirms adjoint diff is active.

Run: bash ~/run.sh bench_qnn.py [--full]
"""
from __future__ import annotations
import time
import argparse
import numpy as np
import torch
import pennylane as qml

from quantum_model import HybridQNN, QuantumConfig, build_vqc_qnode, make_device


def _time(fn, repeats=3):
    """Median wall-time of fn() over `repeats` (after 1 warmup)."""
    fn()  # warmup (graph build / caching)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def bench_torchlayer_batch(n_qubits, n_layers, batch, encoding="angle"):
    """Time one forward and one forward+backward through a HybridQNN for a batch."""
    m = HybridQNN(in_dim=n_qubits, n_qubits=n_qubits, n_layers=n_layers,
                  encoding=encoding, device="cpu")
    x = torch.rand(batch, n_qubits)
    y = torch.randint(0, 2, (batch,))
    lossf = torch.nn.CrossEntropyLoss()

    def fwd():
        with torch.no_grad():
            m(x)

    def fwd_bwd():
        m.zero_grad()
        loss = lossf(m(x), y)
        loss.backward()

    t_fwd = _time(fwd)
    t_fb = _time(fwd_bwd)
    return m.backend, t_fwd, t_fb


def check_batch_broadcast(n_qubits=4, n_layers=2):
    """Does the raw QNode broadcast a batch in ONE call, or loop per sample?

    We compare time for batch=1 vs batch=32 on the bare QNode. If the device
    broadcasts, batch=32 is ~batch=1 cost; if it loops, it is ~32x.
    """
    dev, backend = make_device(n_qubits, None, "cpu")
    cfg = QuantumConfig(in_dim=n_qubits, n_qubits=n_qubits, n_layers=n_layers)
    qnode, wshapes = build_vqc_qnode(cfg, dev)
    w = torch.rand(wshapes["weights"])

    def call(b):
        x = torch.rand(b, n_qubits) if b > 1 else torch.rand(n_qubits)
        qml.math.stack(qnode(x, w))

    t1 = _time(lambda: call(1), repeats=5)
    t32 = _time(lambda: call(32), repeats=3)
    ratio = t32 / t1 if t1 else float("nan")
    return backend, t1, t32, ratio


def confirm_adjoint(n_qubits=4, n_layers=2):
    """Confirm the QNode's diff_method resolves to adjoint on lightning.qubit."""
    dev, backend = make_device(n_qubits, None, "cpu")
    cfg = QuantumConfig(in_dim=n_qubits, n_qubits=n_qubits, n_layers=n_layers)
    qnode, _ = build_vqc_qnode(cfg, dev)
    # The QNode stores its diff_method; surface it.
    dm = getattr(qnode, "diff_method", None)
    return backend, dm


def epoch_estimate(t_fb_per_batch, batch, n_train, epochs=1):
    """Estimate wall-time for `epochs` over n_train samples at this batch cost."""
    n_batches = int(np.ceil(n_train / batch))
    return t_fb_per_batch * n_batches * epochs


def main(full=False):
    print("=" * 64)
    print("QNN THROUGHPUT BENCHMARK (task #14) — lightning.qubit (CPU)")
    print("=" * 64)

    be, dm = confirm_adjoint()
    print(f"\n[adjoint] backend={be} qnode.diff_method={dm!r} "
          f"(expect 'adjoint' for lightning.qubit)")

    print("\n[batch-broadcast] bare QNode: time batch=1 vs batch=32")
    be, t1, t32, ratio = check_batch_broadcast()
    print(f"  backend={be} t(b=1)={t1*1000:.1f}ms t(b=32)={t32*1000:.1f}ms "
          f"ratio={ratio:.1f}x  -> {'LOOPS per-sample' if ratio > 8 else 'broadcasts'}")

    print("\n[hybrid fwd / fwd+bwd] per-batch wall-time (median of 3):")
    print(f"  {'qubits':>6} {'depth':>5} {'batch':>5} {'backend':>16} "
          f"{'fwd(ms)':>9} {'fwd+bwd(ms)':>12} {'samp/s(fb)':>11}")
    grid_q = [4, 8, 12] if full else [4, 8]
    grid_layers = [2, 3] if full else [2]
    grid_batch = [32, 64] if full else [64]
    rows = []
    for q in grid_q:
        for L in grid_layers:
            for b in grid_batch:
                be, tf, tfb = bench_torchlayer_batch(q, L, b)
                sps = b / tfb if tfb else float("nan")
                print(f"  {q:>6} {L:>5} {b:>5} {be:>16} "
                      f"{tf*1000:>9.1f} {tfb*1000:>12.1f} {sps:>11.1f}")
                rows.append((q, L, b, tfb, sps))

    # Budget projection at the most representative config (8q, depth 2-3, batch 64).
    print("\n[budget] projected wall-time per EPOCH (forward+backward):")
    print(f"  {'qubits':>6} {'depth':>5} {'batch':>5} {'n_train':>8} "
          f"{'sec/epoch':>10} {'min/epoch':>10}")
    for (q, L, b, tfb, sps) in rows:
        for n_train in (1000, 2000):
            sec = epoch_estimate(tfb, b, n_train, epochs=1)
            print(f"  {q:>6} {L:>5} {b:>5} {n_train:>8} {sec:>10.1f} {sec/60:>10.2f}")

    print("\nDone. See results/qnn_throughput.md for the recommended #7 budget.")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="benchmark the full 4/8/12-qubit x depth x batch grid")
    a = ap.parse_args()
    main(full=a.full)
