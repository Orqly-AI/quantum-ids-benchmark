"""Publication figures for the benchmark campaign (task #7).

Reads the harness result JSONs in results/ (each carries ``config``, ``metrics``,
and a per-instance ``predictions`` block) and produces camera-ready matplotlib
figures into figures/:

  1. comparison bar charts    -- per dataset, models x a chosen metric (mean+/-std
                                 over seeds), quantum models highlighted.
  2. ROC curves               -- per dataset, one curve per model (uses the stored
                                 y_true / y_score; needs y_score_is_proba or a
                                 rank-valid score).
  3. ablation line plots      -- HybridQNN metric vs n_qubits (one line per
                                 encoding), and vs n_layers.
  4. confusion matrices       -- the best quantum model vs the best classical
                                 model on each dataset.

The companion noise-degradation figures live in noise_plots.py (task #10); the
main LaTeX results table is emitted by aggregate.py. This module is metric-aware
but not metric-hardcoded: pass --metric to chart any key present in the JSONs.

Usage:
    python figures.py --results-dir ../results --out ../figures --metric f1
"""
from __future__ import annotations
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless (WSL, no display)
import matplotlib.pyplot as plt

QUANTUM_MODELS = {"hybrid_qnn", "qsvm"}
LOWER_BETTER = {"false_positive_rate", "brier", "ece", "train_time_s"}
_PRETTY = {
    "f1": "F1", "accuracy": "Accuracy", "roc_auc": "ROC-AUC", "auprc": "AUPRC",
    "detection_rate": "Detection rate", "false_positive_rate": "False-positive rate",
    "tpr_at_1pct_fpr": "TPR @ 1% FPR", "tpr_at_0.1pct_fpr": "TPR @ 0.1% FPR",
    "ece": "ECE", "brier": "Brier",
}


# ----------------------------------------------------------------- loading
def load_records(results_dir: str) -> list[dict]:
    """Load every harness result JSON (those with config+metrics)."""
    recs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(path) as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if "config" in rec and "metrics" in rec:
            rec["_dir"] = results_dir
            recs.append(rec)
    return recs


def _predictions(rec: dict):
    """Return (y_true, y_score, is_proba) or (None, None, None).

    Handles inline predictions and the .npz sidecar written for large test sets.
    """
    p = rec.get("predictions")
    if not p:
        return None, None, None
    if "sidecar" in p:
        path = os.path.join(rec.get("_dir", "."), p["sidecar"])
        if not os.path.exists(path):
            return None, None, None
        d = np.load(path)
        ys = d["y_score"]
        ys = ys if ys.size else None
        return d["y_true"], ys, p.get("y_score_is_proba", False)
    yt = p.get("y_true")
    ys = p.get("y_score")
    if yt is None:
        return None, None, None
    return (np.asarray(yt), np.asarray(ys, dtype=float) if ys is not None else None,
            p.get("y_score_is_proba", False))


def _group(recs):
    """dataset -> model_type -> list of records (one per seed)."""
    g: dict = defaultdict(lambda: defaultdict(list))
    for r in recs:
        c = r["config"]
        g[c.get("dataset", "?")][c.get("model_type", "?")].append(r)
    return g


def _agg(records, metric):
    """(mean, std) of `metric` over a model's seed records; NaN-safe."""
    vals = [r["metrics"].get(metric) for r in records]
    vals = [float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return float("nan"), 0.0
    return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)


# ----------------------------------------------------------------- 1. bars
def bar_chart(recs, out_dir, metric="f1"):
    g = _group(recs)
    for ds, models in g.items():
        names, means, stds, is_q = [], [], [], []
        for mt, rs in sorted(models.items()):
            m, s = _agg(rs, metric)
            names.append(mt); means.append(m); stds.append(s)
            is_q.append(mt in QUANTUM_MODELS)
        order = np.argsort(means)  # ascending; lower-better metrics read naturally
        names = [names[i] for i in order]; means = [means[i] for i in order]
        stds = [stds[i] for i in order]; is_q = [is_q[i] for i in order]
        colors = ["#d62728" if q else "#1f77b4" for q in is_q]  # quantum=red

        fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(names)), 4))
        ax.bar(range(len(names)), means, yerr=stds, capsize=4, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel(_PRETTY.get(metric, metric))
        ax.set_title(f"{ds}: {_PRETTY.get(metric, metric)} by model (mean$\\pm$std over seeds)")
        handles = [plt.Rectangle((0, 0), 1, 1, color="#d62728"),
                   plt.Rectangle((0, 0), 1, 1, color="#1f77b4")]
        ax.legend(handles, ["quantum", "classical"], frameon=False)
        fig.tight_layout()
        _save(fig, out_dir, f"bar_{ds}_{metric}")


# ----------------------------------------------------------------- 2. ROC
def roc_curves(recs, out_dir):
    from sklearn.metrics import roc_curve, roc_auc_score
    g = _group(recs)
    for ds, models in g.items():
        fig, ax = plt.subplots(figsize=(5, 5))
        plotted = 0
        for mt, rs in sorted(models.items()):
            # one representative seed (the lowest) with a usable score
            rs_sorted = sorted(rs, key=lambda r: r["config"].get("seed", 0))
            for r in rs_sorted:
                yt, ys, _ = _predictions(r)
                if yt is None or ys is None or len(np.unique(yt)) < 2:
                    continue
                fpr, tpr, _ = roc_curve(yt, ys)
                try:
                    auc = roc_auc_score(yt, ys)
                except ValueError:
                    auc = float("nan")
                ax.plot(fpr, tpr, lw=1.6,
                        ls="--" if mt in QUANTUM_MODELS else "-",
                        label=f"{mt} (AUC={auc:.3f})")
                plotted += 1
                break
        if not plotted:
            plt.close(fig)
            continue
        ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls=":")
        ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
        ax.set_title(f"{ds}: ROC curves")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.tight_layout()
        _save(fig, out_dir, f"roc_{ds}")


# ------------------------------------------------------------- 3. ablations
def ablation_plots(recs, out_dir, metric="f1"):
    """HybridQNN `metric` vs n_qubits (line per encoding) and vs n_layers."""
    q = [r for r in recs if r["config"].get("model_type") == "hybrid_qnn"]
    if not q:
        return
    by_ds = defaultdict(list)
    for r in q:
        by_ds[r["config"].get("dataset", "?")].append(r)

    for ds, rs in by_ds.items():
        # --- vs n_qubits, one line per encoding ---
        enc_groups = defaultdict(lambda: defaultdict(list))  # enc -> nq -> records
        for r in rs:
            c = r["config"]
            enc_groups[c.get("encoding", "?")][c.get("n_qubits")].append(r)
        if any(len(nq) >= 2 for nq in enc_groups.values()):
            fig, ax = plt.subplots(figsize=(5, 4))
            for enc, nqmap in sorted(enc_groups.items()):
                xs = sorted(nqmap)
                ys = [_agg(nqmap[x], metric)[0] for x in xs]
                es = [_agg(nqmap[x], metric)[1] for x in xs]
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=enc)
            ax.set_xlabel("number of qubits"); ax.set_ylabel(_PRETTY.get(metric, metric))
            ax.set_title(f"{ds}: HybridQNN {_PRETTY.get(metric, metric)} vs qubits")
            ax.legend(frameon=False, title="encoding")
            fig.tight_layout()
            _save(fig, out_dir, f"ablation_qubits_{ds}_{metric}")

        # --- vs n_layers (depth), fixed at the modal qubit count ---
        depth_map = defaultdict(list)
        for r in rs:
            depth_map[r["config"].get("n_layers")].append(r)
        if len(depth_map) >= 2:
            xs = sorted(d for d in depth_map if d is not None)
            ys = [_agg(depth_map[x], metric)[0] for x in xs]
            es = [_agg(depth_map[x], metric)[1] for x in xs]
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.errorbar(xs, ys, yerr=es, marker="s", capsize=3, color="#d62728")
            ax.set_xlabel("ansatz depth (n_layers)"); ax.set_ylabel(_PRETTY.get(metric, metric))
            ax.set_title(f"{ds}: HybridQNN {_PRETTY.get(metric, metric)} vs depth")
            fig.tight_layout()
            _save(fig, out_dir, f"ablation_depth_{ds}_{metric}")


# --------------------------------------------------- 4. confusion matrices
def confusion_matrices(recs, out_dir, metric="f1"):
    from sklearn.metrics import confusion_matrix
    g = _group(recs)
    for ds, models in g.items():
        # best quantum + best classical by mean metric
        best = {}
        for mt, rs in models.items():
            m, _ = _agg(rs, metric)
            grp = "quantum" if mt in QUANTUM_MODELS else "classical"
            if grp not in best or (m > best[grp][1]) ^ (metric in LOWER_BETTER):
                best[grp] = (mt, m, rs)
        picks = [(grp, *best[grp]) for grp in ("quantum", "classical") if grp in best]
        if not picks:
            continue
        fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 3.6))
        if len(picks) == 1:
            axes = [axes]
        for ax, (grp, mt, _m, rs) in zip(axes, picks):
            r = sorted(rs, key=lambda r: r["config"].get("seed", 0))[0]
            yt, _, _ = _predictions(r)
            p = r.get("predictions") or {}
            yp = p.get("y_pred")
            if yt is None or yp is None:
                ax.set_visible(False)
                continue
            cm = confusion_matrix(np.asarray(yt), np.asarray(yp))
            im = ax.imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v > cm.max() / 2 else "black")
            ax.set_title(f"{grp}: {mt}")
            ax.set_xlabel("predicted"); ax.set_ylabel("true")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["normal", "attack"]); ax.set_yticklabels(["normal", "attack"])
        fig.suptitle(f"{ds}: confusion matrices (best models)")
        fig.tight_layout()
        _save(fig, out_dir, f"confusion_{ds}")


# ----------------------------------------------------------------- helpers
def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):  # png for review, pdf for the manuscript
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] {name}.png/.pdf")


def main():
    ap = argparse.ArgumentParser(description="Generate benchmark figures (task #7).")
    HERE = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--results-dir", default=os.path.join(HERE, "..", "results"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "figures"))
    ap.add_argument("--metric", default="f1", help="metric for bars/ablations/best-pick")
    a = ap.parse_args()

    recs = load_records(a.results_dir)
    print(f"[figures] loaded {len(recs)} result record(s) from {a.results_dir}")
    if not recs:
        print("[figures] nothing to plot.")
        return
    bar_chart(recs, a.out, a.metric)
    roc_curves(recs, a.out)
    ablation_plots(recs, a.out, a.metric)
    confusion_matrices(recs, a.out, a.metric)
    print(f"[figures] done -> {a.out}")


if __name__ == "__main__":
    main()
