"""Plotting helpers for the NISQ noise-robustness study (task #10).

Reads the harness result JSONs (results/*.json), each of which carries
``config`` (with the noise spec under ``extra.noise = {type, p}``) and
``metrics`` (operating-point + calibration metrics from evaluation.py). Produces
degradation curves of the operationally-important IDS metrics vs noise strength:

  * TPR @ low FPR  -- detection rate at a fixed alarm budget (keys
                      ``tpr_at_0.1pct_fpr`` / ``tpr_at_1pct_fpr`` from #8).
  * ECE            -- expected calibration error (key ``ece`` from #8).

One line per noise channel; the p=0 (noiseless) point is shared across channels
as the baseline. Metric names are kept in lock-step with ai-engineer's
evaluation.probability_metrics (task #8) -- see METRIC_KEYS below.

Usage::

    python noise_plots.py --results-dir ../results --out ../results/figures
    # or programmatically:
    from noise_plots import load_noise_records, plot_metric_vs_noise
"""
from __future__ import annotations
import os
import glob
import json
import argparse
from collections import defaultdict

# Metric keys, matched to evaluation.probability_metrics (task #8). Edit here if
# ai-engineer renames a key, and every plot/label updates consistently.
METRIC_KEYS = {
    "tpr_at_0.1pct_fpr": "TPR @ 0.1% FPR",
    "tpr_at_1pct_fpr": "TPR @ 1% FPR",
    "ece": "Expected Calibration Error",
    "roc_auc": "ROC-AUC",
    "auprc": "AUPRC",
}


def _noise_of(record: dict) -> tuple[str, float]:
    """Extract (channel, p) from a result record. Noiseless -> ('noiseless', 0.0)."""
    cfg = record.get("config", {})
    extra = cfg.get("extra", {}) or {}
    noise = extra.get("noise")
    # Some runs may also stash a normalized noise on the model config snapshot.
    if not noise:
        noise = cfg.get("noise")
    if not noise or not noise.get("p"):
        return "noiseless", 0.0
    return str(noise.get("type", "depolarizing")), float(noise["p"])


def _model_label(record: dict) -> str:
    cfg = record.get("config", {})
    return cfg.get("name") or cfg.get("model_type", "model")


def load_noise_records(results_dir: str) -> list[dict]:
    """Load every results/*.json into a list of dicts (skips unreadable files)."""
    out = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(path) as f:
                out.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def collect_curves(records: list[dict], metric: str, model: str | None = None):
    """Return {channel: [(p, value), ...sorted]} for one metric.

    The noiseless baseline (p=0) is appended to every channel's curve so each
    line starts from the ideal point. If `model` is given, only that model's
    records are used (match on config.name or model_type).
    """
    by_channel: dict[str, dict[float, float]] = defaultdict(dict)
    baseline: float | None = None
    for r in records:
        if model is not None and _model_label(r) != model:
            continue
        val = r.get("metrics", {}).get(metric)
        if val is None:
            continue
        ch, p = _noise_of(r)
        if ch == "noiseless":
            baseline = float(val)
        else:
            by_channel[ch][p] = float(val)
    curves: dict[str, list[tuple[float, float]]] = {}
    for ch, pts in by_channel.items():
        if baseline is not None:
            pts = {0.0: baseline, **pts}
        curves[ch] = sorted(pts.items())
    return curves


def plot_metric_vs_noise(records, metric="tpr_at_1pct_fpr", model=None,
                         out_path=None, ax=None, title=None):
    """Plot one metric vs noise strength, one line per channel.

    Returns the matplotlib Axes. Saves a PNG if out_path is given. Import of
    matplotlib is deferred so the rest of the module is usable headless.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless: WSL has no display
    import matplotlib.pyplot as plt

    curves = collect_curves(records, metric, model=model)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for ch in sorted(curves):
        xs = [p for p, _ in curves[ch]]
        ys = [v for _, v in curves[ch]]
        ax.plot(xs, ys, marker="o", label=ch)
    ax.set_xlabel("noise strength  p")
    ax.set_ylabel(METRIC_KEYS.get(metric, metric))
    ax.set_xscale("symlog", linthresh=1e-3)  # p spans 0 .. 0.05, near-zero matters
    ax.grid(True, alpha=0.3)
    ax.legend(title="channel", fontsize=8)
    ttl = title or (f"{METRIC_KEYS.get(metric, metric)} vs NISQ noise"
                    + (f"  ({model})" if model else ""))
    ax.set_title(ttl)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        ax.get_figure().savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[noise_plots] saved -> {out_path}")
    return ax


def plot_robustness_panel(records, model=None, out_path=None):
    """2x2 panel: TPR@0.1%FPR, TPR@1%FPR, ECE, ROC-AUC vs noise. Returns the Figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["tpr_at_0.1pct_fpr", "tpr_at_1pct_fpr", "ece", "roc_auc"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for m, ax in zip(metrics, axes.ravel()):
        plot_metric_vs_noise(records, metric=m, model=model, ax=ax)
    fig.suptitle("NISQ noise robustness" + (f" -- {model}" if model else ""),
                 fontsize=13)
    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[noise_plots] saved -> {out_path}")
    return fig


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(HERE, "..", "results"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "figures"))
    ap.add_argument("--model", default=None,
                    help="restrict to one config name / model_type")
    a = ap.parse_args()

    recs = load_noise_records(a.results_dir)
    print(f"[noise_plots] loaded {len(recs)} result records from {a.results_dir}")
    if not recs:
        print("[noise_plots] no results yet -- run a noise sweep first.")
    else:
        plot_robustness_panel(recs, model=a.model,
                              out_path=os.path.join(a.out, "noise_robustness.png"))
        for m in ("tpr_at_1pct_fpr", "ece"):
            plot_metric_vs_noise(recs, metric=m, model=a.model,
                                out_path=os.path.join(a.out, f"noise_{m}.png"))
