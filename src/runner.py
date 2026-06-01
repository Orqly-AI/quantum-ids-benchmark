"""Run a single ExperimentConfig and write a self-describing results JSON.

The runner is model-agnostic: it loads the dataset, builds a model via
model_factory, trains it (torch mini-batch loop or sklearn fit), evaluates the
standard IDS metrics, and writes results/<slug>_<timestamp>.json containing the
full config + environment snapshot + timestamp + metrics. This is the single
entry point sweeps and run_experiment.py call into.

Usage:
    python runner.py --config configs/nslkdd_qnn.yaml
    python runner.py --config configs/sweep_qubits.yaml   # runs each in the sweep
"""
from __future__ import annotations
import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np

from config import ExperimentConfig, load_configs
from seeding import seed_everything
from logging_utils import get_logger
from provenance import env_snapshot
from model_factory import build_model

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


# ---------------------------------------------------------------- data dispatch
def load_data(cfg: ExperimentConfig):
    """Dispatch to data-engineer's unified loader (task #3, data.load).

    Contract: ``data.load(dataset, n_features, reduction, binary, scale, seed)``
    returns ``(X_train, y_train, X_test, y_test)`` reduced to ``cfg.n_features``
    dims. Falls back to the legacy ``load_nslkdd`` if ``load`` is unavailable.
    """
    import data as data_mod
    if hasattr(data_mod, "load"):
        return data_mod.load(
            cfg.dataset, n_features=cfg.n_features, reduction=cfg.reduction,
            binary=cfg.binary, scale=cfg.scale, seed=cfg.seed,
        )
    if cfg.dataset == "nslkdd":
        return data_mod.load_nslkdd(
            n_features=cfg.n_features, binary=cfg.binary,
            scale=cfg.scale, seed=cfg.seed,
        )
    raise NotImplementedError(
        f"dataset {cfg.dataset!r}: data.load not available yet "
        f"(owned by data-engineer, task #3)."
    )


# -------------------------------------------------------------------- metrics
def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    """Compute the full IDS metric set.

    Delegates to ai-engineer's evaluation.metrics (task #5/#8) so every run
    records the same metric set as the standalone baseline runner: Acc, P, R/DR,
    F1, FPR, plus (when y_score is given) ROC-AUC, AUPRC, TPR@0.1%/1% FPR, Brier,
    ECE. Falls back to a minimal binary set if evaluation.py is unavailable.
    For multiclass, evaluation.metrics is binary-oriented, so we use macro
    averages here instead.
    """
    binary = len(np.unique(y_true)) <= 2
    if binary:
        try:
            import evaluation
            return evaluation.metrics(y_true, y_pred, y_score)
        except ImportError:
            pass
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix)
    avg = "binary" if binary else "macro"
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
    }
    if binary:
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            out["detection_rate"] = float(tp / (tp + fn)) if (tp + fn) else 0.0
            out["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) else 0.0
    if y_score is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            out["roc_auc"] = float("nan")
    return out


# ---------------------------------------------------------------- torch train
def _train_torch(model, Xtr, ytr, Xte, yte, cfg, log):
    import torch
    import torch.nn as nn
    # A HybridQNN's quantum TorchLayer runs on a CPU PennyLane device
    # (lightning.qubit). Putting the torch tensors/classical layers on CUDA then
    # forces a host<->device copy for EVERY sample through the per-sample QNode
    # loop, which is pathologically slow (a q8/2000-sample epoch can hang for
    # hours). Keep quantum models entirely on CPU so no cross-device sync occurs;
    # only pure-classical torch models (no PennyLane backend) use CUDA.
    is_quantum = getattr(model, "backend", None) is not None
    use_cuda = torch.cuda.is_available() and not is_quantum
    device = torch.device("cuda" if use_cuda else "cpu")
    if is_quantum:
        log.info("  quantum model -> training on CPU (avoids per-sample GPU<->CPU sync)")
    model = model.to(device)
    # L2 weight decay via extra.weight_decay (default 0.0 = unchanged). Used by the
    # attribution audit's MLP-regularization sweep (quantum-dev #9).
    wd = float((cfg.extra or {}).get("weight_decay", 0.0))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=wd)
    # Cost-sensitive loss: balanced class weights unless imbalance disabled
    # (under SMOTE the data is already balanced, so weights would double-count).
    weight = None
    if cfg.imbalance == "class_weight":
        classes, counts = np.unique(ytr, return_counts=True)
        w = counts.sum() / (len(classes) * counts)  # sklearn 'balanced' scheme
        weight = torch.tensor(w, dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=weight)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    n = len(Xtr_t)
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % 5 == 0 or ep == 0:
            log.info("  [%s] epoch %d/%d loss=%.4f (%.1fs)",
                     cfg.model_type, ep + 1, cfg.epochs, tot / n, time.time() - t0)
    train_time = time.time() - t0

    model.eval()
    if is_quantum:
        # Full test set evaluated across processes (quantum-dev #15): the QNode
        # forward is a per-sample loop, so a single-process eval over a large test
        # set (e.g. CICIDS ~312k rows) is the campaign bottleneck. parallel_predict
        # chunks Xte across `eval_workers` processes and recombines IN ORIGINAL
        # ORDER (pred[i] aligns with Xte[i], so the predictions block + compare.py
        # stay valid), with a serial fallback on small N / any error. Probabilities
        # are byte-identical to the serial path. OMP_NUM_THREADS=1 keeps each worker
        # from spawning BLAS threads and oversubscribing the box.
        from quantum_model import parallel_predict
        workers = int((cfg.extra or {}).get("eval_workers", 0)) or None  # None -> min(16, cpu)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        log.info("  quantum eval -> parallel_predict (workers=%s) over %d test rows",
                 workers if workers else "auto", len(Xte))
        pred, proba = parallel_predict(model, Xte, workers=workers)
        score_is_proba = proba is not None
    else:
        with torch.no_grad():
            Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
            logits = model(Xte_t)
            sm = torch.softmax(logits, dim=1)
            pred = logits.argmax(1).cpu().numpy()
            # Positive-class score only makes sense for binary; multiclass skips AUC.
            proba = sm[:, 1].cpu().numpy() if logits.shape[1] == 2 else None
        score_is_proba = True  # softmax output is a true probability in [0,1]
    return pred, proba, train_time, str(device), score_is_proba


# -------------------------------------------------------------- sklearn train
def _train_sklearn(model, Xtr, ytr, Xte, yte, cfg, log):
    # Slow models (e.g. SVM/QSVM) optionally train on a subsample.
    if cfg.q_subsample and cfg.q_subsample < len(Xtr):
        log.info("  subsampling train to %d rows", cfg.q_subsample)
        Xtr, ytr = Xtr[:cfg.q_subsample], ytr[:cfg.q_subsample]
    t0 = time.time()
    model.fit(Xtr, ytr)
    train_time = time.time() - t0
    pred = model.predict(Xte)
    proba = None
    score_is_proba = False  # decision_function scores are NOT in [0,1]
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(Xte)[:, 1]
            score_is_proba = True
        except Exception:
            proba = None
    if proba is None and hasattr(model, "decision_function"):
        try:
            proba = model.decision_function(Xte)  # e.g. QSVM: raw margin, not proba
            score_is_proba = False
        except Exception:
            proba = None
    return pred, proba, train_time, "cpu/sklearn", score_is_proba


# --------------------------------------------------------------------- run one
def run_config(cfg: ExperimentConfig, results_dir: str = RESULTS,
               save_predictions: bool = True) -> dict:
    """Execute one config end to end and persist a results JSON. Returns it.

    save_predictions: also store per-instance y_true/y_pred/y_score in the JSON
    (for downstream paired significance tests). Default True.
    """
    cfg.validate()
    os.makedirs(results_dir, exist_ok=True)
    log = get_logger(run_slug=cfg.slug())
    log.info("=== Run %s | %s ===", cfg.name, cfg.slug())
    seed_everything(cfg.seed)

    Xtr, ytr, Xte, yte = load_data(cfg)
    log.info("data: train=%s test=%s", Xtr.shape, Xte.shape)

    # Quantum VQC training cost scales with #samples (per-sample QNode evals), so
    # subsample the TRAIN set for the hybrid QNN here. (Sklearn-side slow models
    # such as SVM/QSVM are subsampled in _train_sklearn; classical torch baselines
    # MLP/CNN are fast and intentionally train on the full set.) Seeded + shuffled
    # so both classes are present even when the loader is not pre-shuffled.
    if cfg.model_type == "hybrid_qnn" and cfg.q_subsample and cfg.q_subsample < len(Xtr):
        idx = np.random.RandomState(cfg.seed).permutation(len(Xtr))[:cfg.q_subsample]
        Xtr, ytr = Xtr[idx], ytr[idx]
        log.info("  subsampling quantum train to %d rows (seeded)", cfg.q_subsample)

    # Tell the model factory the true class count (multiclass output dim).
    n_classes = int(len(np.unique(ytr)))
    extra = dict(cfg.extra or {})
    if not cfg.binary or n_classes > 2:
        extra["n_classes"] = n_classes

    # Class-imbalance handling (cfg.imbalance). SMOTE resamples the TRAIN set
    # here (test untouched); 'class_weight' is applied inside the estimator
    # (sklearn) or the loss (torch). scale_pos_weight for XGB is derived from y.
    if cfg.imbalance == "smote" and n_classes == 2:
        try:
            from baselines import resample_smote
            before = len(Xtr)
            Xtr, ytr = resample_smote(Xtr, ytr, seed=cfg.seed)
            log.info("SMOTE: train %d -> %d rows", before, len(Xtr))
        except Exception as exc:  # noqa: BLE001
            log.warning("SMOTE failed (%s); proceeding without resampling", exc)
    if cfg.model_type == "xgboost" and cfg.imbalance == "class_weight" and n_classes == 2:
        pos = float((ytr == 1).sum()); neg = float((ytr == 0).sum())
        extra["scale_pos_weight"] = neg / pos if pos else 1.0
    cfg.extra = extra

    model, kind = build_model(cfg)
    backend = getattr(model, "backend", None)
    log.info("model: %s (kind=%s%s)", cfg.model_type, kind,
             f", backend={backend}" if backend else "")

    if kind == "torch":
        pred, proba, train_time, device, score_is_proba = _train_torch(
            model, Xtr, ytr, Xte, yte, cfg, log)
    else:
        pred, proba, train_time, device, score_is_proba = _train_sklearn(
            model, Xtr, ytr, Xte, yte, cfg, log)

    t0 = time.time()
    metrics = compute_metrics(yte, pred, proba)
    metrics["train_time_s"] = round(train_time, 3)
    metrics["eval_time_s"] = round(time.time() - t0, 3)
    if backend:
        metrics["quantum_backend"] = backend
    log.info("metrics: %s", {k: round(v, 4) if isinstance(v, float) else v
                             for k, v in metrics.items()})

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(results_dir, f"{cfg.slug()}_{ts}.json")

    record = {
        "config": cfg.to_dict(),
        "env": env_snapshot(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "metrics": metrics,
    }

    # Per-instance predictions for downstream paired significance tests
    # (ai-engineer #5/#9: McNemar / paired bootstrap across model pairs on the
    # same fixed test set). The test-set order is deterministic (seeded loader),
    # so y_pred arrays from different result files align by index for a given
    # (dataset, seed). y_score is P(attack) when score_is_proba, else a raw
    # decision/margin score (rank metrics valid; Brier/ECE skipped downstream).
    if save_predictions:
        record["predictions"] = _save_predictions(
            yte, pred, proba, score_is_proba, out_path, log)

    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    log.info("saved -> %s", out_path)
    return record


# Inline predictions up to this many test rows; larger -> compressed .npz sidecar
# (ai-engineer's request, keeps CICIDS2017's ~840k-row JSONs small).
_INLINE_PRED_LIMIT = 100_000


def _save_predictions(y_true, y_pred, y_score, score_is_proba, out_path, log) -> dict:
    """Build the record['predictions'] block; sidecar .npz for large test sets."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = None if y_score is None else np.asarray(y_score, dtype=float)
    if len(y_true) <= _INLINE_PRED_LIMIT:
        return {
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "y_score": (y_score.tolist() if y_score is not None else None),
            "y_score_is_proba": bool(score_is_proba),
        }
    # Large test set: store arrays in a compressed sidecar next to the JSON.
    sidecar = os.path.splitext(out_path)[0] + ".preds.npz"
    np.savez_compressed(
        sidecar, y_true=y_true, y_pred=y_pred,
        y_score=(y_score if y_score is not None else np.array([])),
    )
    log.info("predictions -> sidecar %s (%d rows)", sidecar, len(y_true))
    return {
        "sidecar": os.path.basename(sidecar),
        "y_score_is_proba": bool(score_is_proba),
        "has_score": y_score is not None,
        "n": int(len(y_true)),
    }


def run_file(path: str, results_dir: str = RESULTS,
             save_predictions: bool = True) -> list[dict]:
    """Run every config in a YAML file (single config or a sweep)."""
    configs = load_configs(path)
    log = get_logger()
    log.info("loaded %d config(s) from %s", len(configs), path)
    records = []
    import glob as _glob
    for i, cfg in enumerate(configs, 1):
        log.info("---- [%d/%d] %s ----", i, len(configs), cfg.name)
        # Per-run resumability: if a result JSON for this exact slug already
        # exists, skip it. Lets an interrupted sweep (e.g. killed by a WSL
        # restart) resume where it left off instead of recomputing from run 1.
        if _glob.glob(os.path.join(results_dir, cfg.slug() + "_*.json")):
            log.info("     skip (result already exists for slug %s)", cfg.slug())
            continue
        try:
            records.append(run_config(cfg, results_dir, save_predictions))
        except Exception as exc:  # noqa: BLE001
            # One failing config shouldn't sink an overnight sweep; log full
            # traceback to the run log and continue to the next config.
            log.exception("config %s FAILED: %s: %s",
                          cfg.name, type(exc).__name__, exc)
    return records


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run experiment config(s).")
    ap.add_argument("--config", required=True, help="YAML config or sweep file")
    ap.add_argument("--results-dir", default=RESULTS)
    ap.add_argument("--no-predictions", action="store_true",
                    help="don't store per-instance y_true/y_pred/y_score in the JSON")
    a = ap.parse_args()
    run_file(a.config, a.results_dir, save_predictions=not a.no_predictions)
