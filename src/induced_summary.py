"""Summarise every induced-shift run in results/kernel_analysis/induced_<ds>_<holdout>_s<seed>.json.

For each run and kernel family: the correlation r across the 40 grid settings
between standard-CV ROC-AUC and test ROC-AUC, the test ROC-AUC at the CV-selected
setting, the best test ROC-AUC in the grid (oracle), and the selection loss
(oracle minus CV pick). Per run: the smallest r over the three kernels, the
largest selection loss, whether the kernel ranking at the CV-selected settings
agrees with the ranking at the oracle settings, and the train-to-test shift
statistics recorded by landscape_ds (squared MMD; mmd2_all needs no labels).
Across all runs with a holdout: Spearman correlations between the shift
statistics and the selection-failure measures, with permutation p-values.

Run from src/:  python induced_summary.py   -> results/kernel_analysis/induced_summary.json
"""
import glob
import json
import os
import re

import numpy as np
from scipy.stats import spearmanr

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "kernel_analysis")
FAM = {"Q IQP projected": "quantum", "classical trig + IQP phases": "phase", "classical RBF raw": "rbf"}


def per_family(g):
    keys = list(g)
    s = np.array([g[k]["std_cv"] for k in keys])
    t = np.array([g[k]["test"]["roc_auc"] for k in keys])
    t1 = np.array([g[k]["test"]["tpr_at_1pct_fpr"] for k in keys])
    i, j = int(np.argmax(s)), int(np.argmax(t))
    return {"r": float(np.corrcoef(s, t)[0, 1]), "pick_auc": float(t[i]), "oracle_auc": float(t[j]),
            "loss": float(t[j] - t[i]), "pick_tpr1": float(t1[i]), "best_tpr1": float(t1.max()),
            "pick": keys[i], "oracle": keys[j]}


def perm_spearman(x, y, n=10000, seed=0):
    rho = spearmanr(x, y).correlation
    rng = np.random.RandomState(seed)
    null = np.array([spearmanr(x, rng.permutation(y)).correlation for _ in range(n)])
    return float(rho), float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n + 1))


rows = []
for f in sorted(glob.glob(os.path.join(R, "induced_*_s*.json"))):
    m = re.match(r"induced_(\w+?)_(.+)_s(\d+)\.json$", os.path.basename(f))
    if not m or f.endswith("_bal.json"):
        continue
    d = json.load(open(f, encoding="utf-8"))
    fams = {FAM[k]: per_family(v) for k, v in d["families"].items() if k in FAM}
    order_pick = sorted(fams, key=lambda k: -fams[k]["pick_auc"])
    order_oracle = sorted(fams, key=lambda k: -fams[k]["oracle_auc"])
    rows.append({"file": os.path.basename(f), "dataset": m.group(1), "holdout": d.get("holdout", []),
                 "seed": int(m.group(3)), "unseen_test_fraction": d.get("unseen_test_fraction", 0.0),
                 **{f"shift_{k}": v for k, v in d.get("shift_stats", {}).items()},
                 "families": fams, "min_r": min(v["r"] for v in fams.values()),
                 "mean_r": float(np.mean([v["r"] for v in fams.values()])),
                 "max_loss": max(v["loss"] for v in fams.values()),
                 "winner_pick": order_pick[0], "winner_oracle": order_oracle[0],
                 "ranking_agrees": order_pick == order_oracle,
                 "quantum_minus_rbf_pick_auc": fams["quantum"]["pick_auc"] - fams["rbf"]["pick_auc"],
                 "quantum_minus_rbf_oracle_auc": fams["quantum"]["oracle_auc"] - fams["rbf"]["oracle_auc"]})

held = [r for r in rows if r["holdout"]]
out = {"n_runs": len(rows), "n_holdout_runs": len(held), "runs": rows, "correlations": {}}
for stat in ("shift_mmd2_all", "shift_mmd2_attack", "unseen_test_fraction"):
    x = [r[stat] for r in held if stat in r]
    if len(x) != len(held):
        continue
    for target in ("min_r", "mean_r", "max_loss"):
        rho, p = perm_spearman(x, [r[target] for r in held])
        out["correlations"][f"{stat}~{target}"] = {"spearman": rho, "p_perm": p, "n": len(x)}
out["ranking_agreement"] = {"holdout_runs": int(sum(r["ranking_agrees"] for r in held)),
                            "of": len(held),
                            "baseline_runs": int(sum(r["ranking_agrees"] for r in rows if not r["holdout"])),
                            "baseline_of": sum(1 for r in rows if not r["holdout"])}
json.dump(out, open(os.path.join(R, "induced_summary.json"), "w"), indent=2)

print(f"{'run':52s} {'unseen':>6s} {'mmd_all':>8s} {'mmd_att':>8s} {'min_r':>6s} {'maxloss':>7s}  pick-winner/oracle-winner")
for r in rows:
    print(f"{r['file'][8:-5]:52s} {r['unseen_test_fraction']:6.3f} {r.get('shift_mmd2_all', float('nan')):8.4f} "
          f"{r.get('shift_mmd2_attack', float('nan')):8.4f} {r['min_r']:+6.2f} {r['max_loss']:7.3f}  "
          f"{r['winner_pick']}/{r['winner_oracle']}{'' if r['ranking_agrees'] else '  (ranking differs)'}")
print(json.dumps(out["correlations"], indent=1))
print(out["ranking_agreement"])
