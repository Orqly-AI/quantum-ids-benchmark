"""Scan results/*.json and emit tidy CSVs plus a publication LaTeX table.

Each results JSON (written by runner.py) is one row of the long-form CSV:
config columns + every metric key found in the record. The aggregator is
**metric-agnostic** — it auto-discovers numeric metric keys, so when ai-engineer
(task #8) adds AUPRC / TPR@FPR / Brier / ECE, or any future metric, they appear
in the outputs without changing this file.

Outputs (default into results/):
  * summary.csv      long form, one row per run (per seed)
  * summary_agg.csv  one row per (dataset, model_type): mean/std/CI over seeds
  * summary.tex      camera-ready LaTeX table: mean +/- std over seeds, with a
                     significance marker vs the best classical model per dataset.

Run after a sweep:
    python aggregate.py
    python aggregate.py --results-dir results --metric f1 --sig-metric f1
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

# Config fields surfaced as columns (everything else is treated as a metric).
_CONFIG_COLS = ["name", "dataset", "model_type", "n_qubits", "n_layers",
                "encoding", "n_features", "reduction", "scale", "imbalance",
                "seed", "epochs"]
# Non-config, non-metric bookkeeping columns to exclude from metric detection.
_NON_METRIC = set(_CONFIG_COLS) | {"timestamp_utc", "device", "source_file",
                                   "quantum_backend", "eval_time_s"}
# Preferred display order + pretty headers; any extra metrics are appended.
_METRIC_ORDER = ["accuracy", "precision", "recall", "f1",
                 "detection_rate", "false_positive_rate",
                 "roc_auc", "auprc", "tpr_at_0.1pct_fpr", "tpr_at_1pct_fpr",
                 "brier", "ece", "train_time_s"]
_METRIC_HEADERS = {
    "accuracy": "Acc", "precision": "P", "recall": "R", "f1": "F1",
    "detection_rate": "DR",
    "false_positive_rate": "FPR", "roc_auc": "AUC", "auprc": "AUPRC",
    "tpr_at_0.1pct_fpr": r"TPR@.1\%", "tpr_at_1pct_fpr": r"TPR@1\%",
    "brier": "Brier", "ece": "ECE", "train_time_s": "Time (s)",
}
# Models treated as "quantum" when picking the best classical for significance.
_QUANTUM_MODELS = {"hybrid_qnn", "qsvm"}
# Metrics where lower is better (so the LaTeX table doesn't bold the wrong way).
_LOWER_BETTER = {"false_positive_rate", "brier", "ece", "train_time_s"}


def load_records(results_dir: str = RESULTS) -> pd.DataFrame:
    """Flatten every results JSON into one tidy DataFrame (one row per run)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(path) as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if "config" not in rec or "metrics" not in rec:
            continue  # skip non-result JSON (e.g. legacy summaries)
        cfg, metrics = rec["config"], rec["metrics"]
        row = {c: cfg.get(c) for c in _CONFIG_COLS}
        row.update(metrics)              # all metric keys, whatever they are
        row["timestamp_utc"] = rec.get("timestamp_utc")
        row["device"] = rec.get("device")
        row["source_file"] = os.path.basename(path)
        rows.append(row)
    return pd.DataFrame(rows)


def metric_columns(df: pd.DataFrame) -> list[str]:
    """Auto-discover numeric metric columns (metric-agnostic), in display order."""
    numeric = [c for c in df.columns
               if c not in _NON_METRIC and pd.api.types.is_numeric_dtype(df[c])]
    ordered = [m for m in _METRIC_ORDER if m in numeric]
    extra = sorted(c for c in numeric if c not in _METRIC_ORDER)
    return ordered + extra


def aggregate_over_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, model_type): mean/std/CI over seeds for each metric."""
    if df.empty:
        return df
    from scipy import stats
    metrics = metric_columns(df)
    out_rows = []
    for (ds, mt), g in df.groupby(["dataset", "model_type"]):
        row = {"dataset": ds, "model_type": mt, "n_seeds": int(len(g))}
        for m in metrics:
            vals = g[m].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_std"] = np.nan
                row[f"{m}_ci95"] = np.nan
                continue
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            ci = (float(stats.t.ppf(0.975, df=vals.size - 1)) * std / np.sqrt(vals.size)
                  if vals.size > 1 else 0.0)
            row[f"{m}_mean"], row[f"{m}_std"], row[f"{m}_ci95"] = mean, std, ci
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _significance(df: pd.DataFrame, dataset: str, model: str,
                  metric: str) -> tuple[float, str]:
    """Paired test of `model` vs the best classical model on `dataset`.

    Returns (p_value, marker). Markers: *** p<.001, ** p<.01, * p<.05, ns, or ''
    (when the comparison is undefined: it IS the best classical, too few aligned
    seeds, or evaluation.py unavailable).
    """
    sub = df[df["dataset"] == dataset]
    classical = sub[~sub["model_type"].isin(_QUANTUM_MODELS)]
    if classical.empty:
        return float("nan"), ""
    # best classical by mean of the chosen metric (respecting lower-is-better)
    means = classical.groupby("model_type")[metric].mean()
    if means.empty:
        return float("nan"), ""
    best_classical = means.idxmin() if metric in _LOWER_BETTER else means.idxmax()
    if model == best_classical:
        return float("nan"), ""
    a = (classical[classical["model_type"] == best_classical]
         .sort_values("seed")[metric].dropna().to_numpy(float))
    b = (sub[sub["model_type"] == model]
         .sort_values("seed")[metric].dropna().to_numpy(float))
    n = min(len(a), len(b))
    if n < 2:
        return float("nan"), ""
    try:
        import evaluation
        res = evaluation.paired_t_test(a[:n], b[:n], metric=metric)
        p = res.t_p_value
    except Exception:  # noqa: BLE001
        from scipy import stats
        p = float(stats.ttest_rel(b[:n], a[:n]).pvalue)
    marker = ("***" if p < 0.001 else "**" if p < 0.01
              else "*" if p < 0.05 else "ns")
    return p, marker


def to_latex(df: pd.DataFrame, *, primary: str = "f1") -> str:
    """Camera-ready LaTeX table: mean +/- std over seeds per (dataset, model).

    A significance marker (vs the best classical model on that dataset, paired
    over seeds on `primary`) is appended to each quantum model's primary metric.
    """
    if df.empty:
        return "% no results found\n"
    metrics = metric_columns(df)
    agg = aggregate_over_seeds(df)
    sort_primary = primary if primary in metrics else metrics[0]
    asc = sort_primary in _LOWER_BETTER
    agg = agg.sort_values(["dataset", f"{sort_primary}_mean"],
                          ascending=[True, asc])

    headers = ["Dataset", "Model"] + [_METRIC_HEADERS.get(m, m) for m in metrics]
    align = "ll" + "r" * len(metrics)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Intrusion-detection performance (mean $\pm$ std over seeds). "
        r"DR = detection rate, FPR = false-positive rate; "
        r"TPR@$x$ = detection rate at $x$ false-positive rate. "
        r"Significance (paired $t$-test on " + _tex_escape(sort_primary) +
        r" vs.\ best classical): {*}$p{<}.05$, {**}$p{<}.01$, {***}$p{<}.001$.}",
        r"\label{tab:results}",
        r"\begin{tabular}{" + align + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    prev_ds = None
    for _, r in agg.iterrows():
        ds_label = r["dataset"] if r["dataset"] != prev_ds else ""
        prev_ds = r["dataset"]
        cells = [_tex_escape(str(ds_label)), _tex_escape(str(r["model_type"]))]
        for m in metrics:
            mean, std = r.get(f"{m}_mean"), r.get(f"{m}_std")
            if pd.isna(mean):
                cells.append("--")
                continue
            prec = 1 if m == "train_time_s" else 3
            cell = (f"{mean:.{prec}f}$\\pm${std:.{prec}f}"
                    if std and not pd.isna(std) and std > 0 else f"{mean:.{prec}f}")
            if m == sort_primary and r["model_type"] in _QUANTUM_MODELS:
                _, marker = _significance(df, r["dataset"], r["model_type"], sort_primary)
                if marker and marker != "ns":
                    cell += rf"$^{{{marker}}}$"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


def main():
    ap = argparse.ArgumentParser(description="Aggregate results -> CSV + LaTeX.")
    ap.add_argument("--results-dir", default=RESULTS)
    ap.add_argument("--csv", default=None, help="default: <results-dir>/summary.csv")
    ap.add_argument("--agg-csv", default=None, help="default: <results-dir>/summary_agg.csv")
    ap.add_argument("--tex", default=None, help="default: <results-dir>/summary.tex")
    ap.add_argument("--metric", default="f1",
                    help="primary metric for sorting + significance (default f1)")
    a = ap.parse_args()

    df = load_records(a.results_dir)
    csv_path = a.csv or os.path.join(a.results_dir, "summary.csv")
    agg_path = a.agg_csv or os.path.join(a.results_dir, "summary_agg.csv")
    tex_path = a.tex or os.path.join(a.results_dir, "summary.tex")

    df.to_csv(csv_path, index=False)
    aggregate_over_seeds(df).to_csv(agg_path, index=False)
    with open(tex_path, "w") as f:
        f.write(to_latex(df, primary=a.metric))

    print(f"[aggregate] {len(df)} run(s) -> {csv_path}")
    print(f"[aggregate] per-group means -> {agg_path}")
    print(f"[aggregate] LaTeX table -> {tex_path}")
    if not df.empty:
        print(f"[aggregate] metrics detected: {metric_columns(df)}")
        show = [c for c in ["dataset", "model_type", "seed", "f1", "roc_auc",
                            "auprc", "tpr_at_1pct_fpr", "ece"] if c in df.columns]
        print(df[show].to_string(index=False))


if __name__ == "__main__":
    main()
