"""End-to-end NSL-KDD comparison: classical baselines vs hybrid quantum NN.

Thin wrapper over the reproducible harness (config.py + runner.py): it builds
one ExperimentConfig per model and runs each through run_config(), so every
model gets a provenance-stamped results JSON and a per-run log just like a
sweep. Designed for the 4GB-VRAM RTX 3050 Ti: small qubit counts, modest
subsample for the (slower) quantum training.

This preserves the original CLI:
    python run_experiment.py --qubits 8 --layers 3 --epochs 30 \
        --subsample 8000 --encoding angle
For arbitrary single configs or sweeps, prefer: python runner.py --config <yaml>
"""
from __future__ import annotations
import argparse

from config import ExperimentConfig
from runner import run_config
from logging_utils import get_logger

# The model line-up reproduced from the original script. Each tuple is
# (run name, model_type, extra overrides).
_MODELS = [
    ("RandomForest", "random_forest", {}),
    ("SVM_RBF", "svm_rbf", {}),
    ("ClassicalMLP", "classical_mlp", {}),
    ("HybridQNN", "hybrid_qnn", {}),
]


def run(n_qubits=8, n_layers=3, epochs=30, q_subsample=8000,
        encoding="angle", seed=42):
    log = get_logger()
    log.info("== NSL-KDD comparison | qubits=%d layers=%d encoding=%s ==",
             n_qubits, n_layers, encoding)

    records = {}
    for name, model_type, extra in _MODELS:
        cfg = ExperimentConfig(
            name=name, dataset="nslkdd", n_features=n_qubits, binary=True,
            scale="minmax", model_type=model_type, n_qubits=n_qubits,
            n_layers=n_layers, encoding=encoding, seed=seed, epochs=epochs,
            batch_size=64, lr=5e-3,
            # RF/MLP train on the full set; SVM/QNN subsample for speed.
            q_subsample=q_subsample if model_type in ("svm_rbf", "hybrid_qnn") else 0,
            extra=extra,
        )
        records[name] = run_config(cfg)["metrics"]

    # Pretty summary table (same columns as the original script).
    log.info("\n================ SUMMARY (NSL-KDD test) ================")
    hdr = (f"{'Model':<14}{'Acc':>7}{'F1':>7}{'DR':>7}{'FPR':>7}"
           f"{'AUC':>7}{'time(s)':>9}")
    log.info(hdr)
    log.info("-" * len(hdr))
    for name, m in records.items():
        log.info("%-14s%7.3f%7.3f%7.3f%7.3f%7.3f%9.1f", name,
                 m.get("accuracy", float("nan")), m.get("f1", float("nan")),
                 m.get("detection_rate", float("nan")),
                 m.get("false_positive_rate", float("nan")),
                 m.get("roc_auc", float("nan")), m.get("train_time_s", float("nan")))
    return records


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--subsample", type=int, default=8000)
    ap.add_argument("--encoding", choices=["angle", "amplitude"], default="angle")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run(a.qubits, a.layers, a.epochs, a.subsample, a.encoding, a.seed)
