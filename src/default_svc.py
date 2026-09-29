"""Where do scikit-learn's default RBF-SVM settings land on the bandwidth landscape?

Many QML benchmarks compare a quantum kernel with SVC() at its defaults: C=1 and gamma='scale',
i.e. gamma = 1 / (n_features * X.var()). For each dataset protocol of the kernel analysis this
computes that gamma on the 2,000 training rows, the default SVC's test metrics, and, from the
landscape file, where that gamma sits relative to the grid's best and worst RBF settings. The
quantum projected kernel's CV-selected score on the same rows is reported alongside.

Run from src/:  python default_svc.py  -> results/kernel_analysis/default_svc.json
"""
import json
import os

import numpy as np
from sklearn.svm import SVC

from kernel_analysis import RESULTS, load_nslkdd, score_metrics


def landscape_row(fname, gamma):
    d = json.load(open(os.path.join(RESULTS, fname), encoding="utf-8"))
    g = d["families"]["classical RBF raw"]
    by_g = {}
    for k, v in g.items():
        gm, C = (float(x) for x in k.split("|"))
        if C == 1.0:
            by_g[gm] = v["test"]["roc_auc"]
    q = d["families"]["Q IQP projected"]
    kq = max(q, key=lambda k: q[k]["std_cv"])
    grid = np.array(sorted(by_g))
    near = float(grid[np.argmin(np.abs(np.log(grid) - np.log(gamma)))])
    return {"grid_rbf_auc_at_C1": {f"{k:g}": by_g[k] for k in grid},
            "nearest_grid_gamma": near, "rbf_auc_at_nearest_C1": by_g[near],
            "rbf_best_auc_C1": max(by_g.values()), "rbf_worst_auc_C1": min(by_g.values()),
            "quantum_cv_pick": kq, "quantum_cv_pick_test": q[kq]["test"]}


out = {}
# NSL-KDD: the first 2,000 training rows, as in the kernel analysis
Xtr, ytr, Xte, yte = load_nslkdd(8)
Xq, yq = np.asarray(Xtr[:2000], float), np.asarray(ytr[:2000])
Xte, yte = np.asarray(Xte, float), np.asarray(yte)
gamma = 1.0 / (Xq.shape[1] * Xq.var())
s = SVC(kernel="rbf", C=1.0, gamma="scale").fit(Xq, yq).decision_function(Xte)
out["nslkdd"] = {"gamma_scale": float(gamma), "default_svc_test": score_metrics(yte, s),
                 **landscape_row("landscape.json", gamma)}

# UNSW-NB15: the seeded stratified sample of the kernel analysis
from data import load_meta
from sklearn.model_selection import train_test_split
X, ym, Xt, ymt, meta = load_meta("unsw", 8, "pca", False, "minmax", 42)
names = np.array(meta["class_names"])
yb, ybt = (names[ym] != "Normal").astype(int), (names[ymt] != "Normal").astype(int)
itr, _ = train_test_split(np.arange(len(yb)), train_size=2000, stratify=ym, random_state=0)
ite, _ = train_test_split(np.arange(len(ybt)), train_size=20000, stratify=ymt, random_state=0)
Xq, yq, Xe, ye = np.asarray(X[itr], float), yb[itr], np.asarray(Xt[ite], float), ybt[ite]
gamma = 1.0 / (Xq.shape[1] * Xq.var())
s = SVC(kernel="rbf", C=1.0, gamma="scale").fit(Xq, yq).decision_function(Xe)
out["unsw"] = {"gamma_scale": float(gamma), "default_svc_test": score_metrics(ye, s),
               **landscape_row("landscape_unsw.json", gamma)}

json.dump(out, open(os.path.join(RESULTS, "default_svc.json"), "w"), indent=2)
for ds, o in out.items():
    print(f"{ds}: gamma='scale' = {o['gamma_scale']:.3f} (nearest grid {o['nearest_grid_gamma']:g}); "
          f"default SVC test AUC {o['default_svc_test']['roc_auc']:.3f} TPR1 {o['default_svc_test']['tpr_at_1pct_fpr']:.3f}; "
          f"grid RBF at C=1: best {o['rbf_best_auc_C1']:.3f}, worst {o['rbf_worst_auc_C1']:.3f}; "
          f"quantum CV pick {o['quantum_cv_pick']} AUC {o['quantum_cv_pick_test']['roc_auc']:.3f} "
          f"TPR1 {o['quantum_cv_pick_test']['tpr_at_1pct_fpr']:.3f}")
