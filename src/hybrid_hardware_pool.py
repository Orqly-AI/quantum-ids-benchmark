"""Pool the hybrid's hardware runs over the disjoint test subsets, per device.

Each run (results/kernel_analysis/hybrid_hardware_run_<backend>[_s<k>].json) measured
the trained 4-qubit hybrid's quantum layer on one class-balanced 2,000-row subset of
NSL-KDD's official test set. Subsets are disjoint, so a device's runs together cover
up to 6,000 rows (3,000 benign), which sets the 1%-FPR threshold from 30 benign scores
rather than 10. On the pooled rows this compares the hardware scores with the exact
simulation, the tuned random forest (seed 42) and the classical twins of
hybrid_attribution.py (seed 42, the seed of the model run on hardware), by paired
bootstrap, and splits detection at 1% FPR into seen and novel attack types.

Run from src/:  python hybrid_hardware_pool.py  -> results/kernel_analysis/hybrid_hardware_pooled.json
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

from evaluation import paired_bootstrap
from kernel_analysis import RESULTS, score_metrics

ATTR = os.path.join(os.path.dirname(RESULTS), "hybrid_attribution")
RF = os.path.join(os.path.dirname(RESULTS), "nslkdd_random_forest_q8_l3_angle_s42_rf_20260601_105130.json")


def names_full():
    from data import load_meta
    _, ym, _, ymt, meta = load_meta("nslkdd", 4, "pca", False, "minmax", 42)
    cn = np.array(meta["class_names"])
    return cn[ymt], set(cn[np.unique(ym)])


def tpr_types(y, s, novel):
    """TPR@1%FPR per attack group against all benign rows (ROC interpolation, tie-safe)."""
    from evaluation import tpr_at_fpr
    b = y == 0
    grp = lambda m: float(tpr_at_fpr(y[b | m], s[b | m], 0.01))
    return {"seen_types": grp((y == 1) & ~novel), "novel_types": grp((y == 1) & novel)}


def boot(y, a, b, mt, n_boot=2000):
    r = paired_bootstrap(y, a, b, mt, n_boot=n_boot)
    return {"diff": float(r.observed_diff), "ci95": [float(r.ci95_low), float(r.ci95_high)], "p": float(r.p_value)}


runs = defaultdict(list)
for f in sorted(glob.glob(os.path.join(RESULTS, "hybrid_hardware_run_*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    runs[d["backend"]].append(d)

names, seen = names_full()
novel_all = np.array([n not in seen for n in names])
rf_all = np.asarray(json.load(open(RF, encoding="utf-8"))["predictions"]["y_score"], float)
twins = {k: np.load(os.path.join(ATTR, f"{k}_s42.npz"))["score"] for k in ("linear", "mlp", "fourier")
         if os.path.exists(os.path.join(ATTR, f"{k}_s42.npz"))}

out = {"devices": {}}
for backend, ds in sorted(runs.items()):
    raws = [np.load(os.path.join(RESULTS, d["raw_file"])) for d in ds]
    idx = np.concatenate([r["test_index"] for r in raws])
    assert len(np.unique(idx)) == len(idx), f"{backend}: subsets overlap"
    y = np.concatenate([r["y_test"] for r in raws])
    s_hw = np.concatenate([r["score_hardware"] for r in raws])
    s_ex = np.concatenate([r["score_exact"] for r in raws])
    zr = float(np.corrcoef(np.concatenate([r["Z_hardware"] for r in raws]).ravel(),
                           np.concatenate([r["Z_exact"] for r in raws]).ravel())[0, 1])
    nov = novel_all[idx]
    assert np.array_equal((names[idx] != "normal").astype(int), y), "labels misaligned"
    o = {"subsets": sorted(d["subset_seed"] for d in ds), "n_rows": int(len(y)), "n_benign": int((y == 0).sum()),
         "job_ids": [j for d in ds for j in d["job_ids"]], "qpu_seconds": [d.get("qpu_seconds") for d in ds],
         "expectation_r": zr,
         "hardware": {**score_metrics(y, s_hw), **tpr_types(y, s_hw, nov)},
         "exact": {**score_metrics(y, s_ex), **tpr_types(y, s_ex, nov)},
         "rf": {**score_metrics(y, rf_all[idx]), **tpr_types(y, rf_all[idx], nov)},
         "twins": {k: {**score_metrics(y, v[idx]), **tpr_types(y, v[idx], nov)} for k, v in twins.items()},
         "paired": {}}
    for mt in ("roc_auc", "tpr_at_1pct_fpr"):
        o["paired"][f"hardware_minus_exact_{mt}"] = boot(y, s_ex, s_hw, mt)
        o["paired"][f"hardware_minus_rf_{mt}"] = boot(y, rf_all[idx], s_hw, mt)
        for k, v in twins.items():
            o["paired"][f"hardware_minus_{k}_{mt}"] = boot(y, v[idx], s_hw, mt)
    out["devices"][backend] = o
    print(f"{backend}: {o['n_rows']} rows (subsets {o['subsets']}), <Z> r {zr:.3f}")
    for src in ("hardware", "exact", "rf"):
        m = o[src]
        print(f"   {src:9s} AUC {m['roc_auc']:.3f} TPR1 {m['tpr_at_1pct_fpr']:.3f} seen {m['seen_types']:.3f} novel {m['novel_types']:.3f}")
    for k, m in o["twins"].items():
        print(f"   {k:9s} AUC {m['roc_auc']:.3f} TPR1 {m['tpr_at_1pct_fpr']:.3f} seen {m['seen_types']:.3f} novel {m['novel_types']:.3f}")
    for k, v in o["paired"].items():
        print(f"     {k:36s} {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] p {v['p']:.3f}")
# every device measured the same subsets, so the non-hardware comparisons are shared
idx_sets = {b: np.concatenate([np.load(os.path.join(RESULTS, d["raw_file"]))["test_index"] for d in ds])
            for b, ds in runs.items()}
first = sorted(idx_sets)[0]
if all(np.array_equal(np.sort(v), np.sort(idx_sets[first])) for v in idx_sets.values()):
    raws = [np.load(os.path.join(RESULTS, d["raw_file"])) for d in runs[first]]
    idx = np.concatenate([r["test_index"] for r in raws])
    y = np.concatenate([r["y_test"] for r in raws])
    s_ex = np.concatenate([r["score_exact"] for r in raws])
    common = {"exact_minus_rf_tpr_at_1pct_fpr": boot(y, rf_all[idx], s_ex, "tpr_at_1pct_fpr"),
              "exact_minus_rf_roc_auc": boot(y, rf_all[idx], s_ex, "roc_auc")}
    if "mlp" in twins:
        common["exact_minus_mlp_tpr_at_1pct_fpr"] = boot(y, twins["mlp"][idx], s_ex, "tpr_at_1pct_fpr")
    for k, v in twins.items():
        common[f"{k}_minus_rf_tpr_at_1pct_fpr"] = boot(y, rf_all[idx], v[idx], "tpr_at_1pct_fpr")
    out["common"] = common
    for k, v in common.items():
        print(f"  common {k:36s} {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] p {v['p']:.3f}")
json.dump(out, open(os.path.join(RESULTS, "hybrid_hardware_pooled.json"), "w"), indent=2)
