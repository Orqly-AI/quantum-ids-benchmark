"""Quantum-vs-best-classical significance comparison from saved results.

Consumes the reference-seed predictions/scores persisted by the runners and runs
the three significance tests from evaluation.py:
  * McNemar    -- per-instance hard predictions on the fixed test set.
  * paired bootstrap -- AUPRC / TPR@1%FPR / ROC-AUC on the probability scores.
  * paired t-test / Wilcoxon -- over per-seed metric values, when available.

It reads two sources of predictions (whichever are present), so it works whether
runs came from the standalone baseline runner or the unified harness:

  A) Standalone baselines JSON (run_baselines.py):
       results/baselines_<dataset>_<view>_<imbalance>.json
       -> reference_predictions.by_model[name] = {"pred":[...], "score":[...]}
       -> per_seed_metrics[name] = [ {metric: val}, ... ]
  B) Harness per-run JSON (runner.py):
       results/<slug>_<ts>.json with top-level "predictions" {y_true,y_pred,y_score}
       and "config" {dataset, model_type, seed, ...}

Usage:
    python compare.py --dataset nslkdd --view pca \
        --quantum hybrid_qnn --against auto
    python compare.py --baselines-json results/baselines_nslkdd_pca_class_weight.json \
        --quantum-json results/nslkdd_hybrid_qnn_q8_l3_angle_s42_*.json

`--against auto` picks the best classical model by mean test F1.
Writes results/compare_<dataset>_<view>.json and prints a summary.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import evaluation

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

_QUANTUM = {"hybrid_qnn", "qsvm", "quantum_kernel_svm"}


# --------------------------------------------------------------------------- #
# Loading predictions/scores from either result format
# --------------------------------------------------------------------------- #
def _load_baselines_json(path: str):
    """Return (per_seed, preds, scores, y_true) from a run_baselines.py JSON."""
    d = json.load(open(path))
    per_seed = d.get("per_seed_metrics", {})
    rp = d.get("reference_predictions", {})
    y_true = np.asarray(rp.get("y_true", []))
    preds, scores = {}, {}
    for name, bm in rp.get("by_model", {}).items():
        if bm.get("pred") is not None:
            preds[name] = {"y_true": y_true, "y_pred": np.asarray(bm["pred"])}
        if bm.get("score") is not None:
            scores[name] = {"y_true": y_true, "y_score": np.asarray(bm["score"])}
    return per_seed, preds, scores, y_true


def _load_harness_json(path: str):
    """Parse a runner.py JSON's predictions block (inline or sidecar).

    Returns (model_type, seed, y_true, y_pred, y_score|None, score_is_proba).
    `score_is_proba` is False for models exposing only decision_function (QSVM,
    LinearSVC): their y_score is a raw margin, valid for rank metrics (AUC/AUPRC/
    TPR@FPR) but NOT for Brier/ECE — the caller skips those.
    """
    d = json.load(open(path))
    cfg = d.get("config", {})
    pred_block = d.get("predictions")
    if pred_block is None:
        return None
    is_proba = bool(pred_block.get("y_score_is_proba", True))
    # sidecar support (large datasets, e.g. CICIDS2017)
    if "sidecar" in pred_block:
        npz = np.load(os.path.join(os.path.dirname(path), pred_block["sidecar"]))
        yt, yp = npz["y_true"], npz["y_pred"]
        ys = npz["y_score"] if ("y_score" in npz and pred_block.get("has_score", True)) else None
    else:
        yt = np.asarray(pred_block.get("y_true", []))
        yp = np.asarray(pred_block.get("y_pred", []))
        ys = pred_block.get("y_score")
        ys = np.asarray(ys) if ys is not None else None
    return cfg.get("model_type"), cfg.get("seed"), yt, yp, ys, is_proba


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def compare(baselines_json: str, quantum_json: str | None,
            quantum_name: str, against: str,
            bootstrap_metrics=("auprc", "tpr_at_1pct_fpr", "roc_auc")) -> dict:
    per_seed, preds, scores, y_true = _load_baselines_json(baselines_json)

    classical_models = [m for m in per_seed if m not in _QUANTUM]
    # Choose the classical model to compare against.
    if against == "auto":
        agg = {m: evaluation.aggregate_seeds(per_seed[m]) for m in classical_models}
        classical = evaluation.best_classical(
            {m: {"f1": agg[m]["f1"]} for m in agg if "f1" in agg[m]}, metric="f1")
    else:
        classical = against
    if classical is None:
        raise SystemExit("no classical model found to compare against")

    # Pull the quantum model's reference predictions/scores.
    bootstrap_metrics = tuple(bootstrap_metrics)
    if quantum_json:
        loaded = _load_harness_json(quantum_json)
        if loaded is None:
            raise SystemExit(f"{quantum_json} has no 'predictions' block")
        qmt, qseed, qyt, qyp, qys, q_is_proba = loaded
        if len(qyt) != len(y_true) or not np.array_equal(qyt, y_true):
            raise SystemExit(
                "quantum and classical test labels differ (different split/order) "
                "— cannot pair McNemar/bootstrap. Ensure same dataset/seed.")
        preds[quantum_name] = {"y_true": y_true, "y_pred": qyp}
        if qys is not None:
            scores[quantum_name] = {"y_true": y_true, "y_score": qys}
        # If the quantum score is a raw decision margin (QSVM), drop the rank
        # metrics that need it to be a calibrated probability. AUC/AUPRC/TPR@FPR
        # are rank-based and stay valid; Brier/ECE would be meaningless on margins.
        if not q_is_proba:
            bootstrap_metrics = tuple(m for m in bootstrap_metrics
                                      if m not in ("brier",))
    elif quantum_name not in preds:
        raise SystemExit(
            f"quantum model {quantum_name!r} not in {baselines_json} and no "
            f"--quantum-json given; nothing to compare.")

    out = {
        "dataset_file": os.path.basename(baselines_json),
        "quantum": quantum_name, "classical": classical,
    }
    # McNemar on hard preds.
    if quantum_name in preds and classical in preds:
        pc, pq = preds[classical], preds[quantum_name]
        out["mcnemar"] = evaluation.mcnemar_test(
            pc["y_true"], pc["y_pred"], pq["y_pred"]).to_dict()
    # Paired bootstrap on scores.
    if quantum_name in scores and classical in scores:
        sc, sq = scores[classical], scores[quantum_name]
        out["bootstrap"] = {
            bm: evaluation.paired_bootstrap(
                sc["y_true"], sc["y_score"], sq["y_score"], metric=bm).to_dict()
            for bm in bootstrap_metrics
        }
    # Per-seed paired test (only if BOTH have per-seed lists in the baselines file).
    if quantum_name in per_seed and classical in per_seed:
        out["paired_seed_test"] = evaluation.paired_t_test(
            [m["f1"] for m in per_seed[classical]],
            [m["f1"] for m in per_seed[quantum_name]], metric="f1").to_dict()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines-json", required=True,
                    help="results/baselines_<dataset>_<view>_<imbalance>.json")
    ap.add_argument("--quantum-json", default=None,
                    help="harness JSON of the quantum run (with predictions block)")
    ap.add_argument("--quantum", default="hybrid_qnn",
                    help="name to label the quantum model as")
    ap.add_argument("--against", default="auto",
                    help="'auto' (best classical by F1) or a model name")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    # Allow globbed --quantum-json (take newest match).
    qj = a.quantum_json
    if qj and any(c in qj for c in "*?["):
        matches = sorted(glob.glob(qj))
        qj = matches[-1] if matches else None

    res = compare(a.baselines_json, qj, a.quantum, a.against)
    out = a.out or os.path.join(RESULTS, "compare_summary.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n=== {res['quantum']} vs {res['classical']} ===")
    if "mcnemar" in res:
        m = res["mcnemar"]
        print(f"McNemar: n01(B>A)={m['n01']} n10(A>B)={m['n10']} "
              f"p={m['p_value']:.3e} ({m['method']})")
    for bm, b in res.get("bootstrap", {}).items():
        print(f"Bootstrap {bm}: classical={b['score_a']:.4f} quantum={b['score_b']:.4f} "
              f"Δ={b['observed_diff']:+.4f} 95%CI=[{b['ci95_low']:+.4f},{b['ci95_high']:+.4f}] "
              f"p={b['p_value']:.3e}")
    if "paired_seed_test" in res:
        p = res["paired_seed_test"]
        print(f"Paired-seed F1: classical={p['mean_a']:.4f} quantum={p['mean_b']:.4f} "
              f"Δ={p['mean_diff']:+.4f} t-p={p['t_p_value']:.3e} "
              f"Wilcoxon-p={p['wilcoxon_p_value']:.3e} d={p['cohens_d']:.2f}")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
