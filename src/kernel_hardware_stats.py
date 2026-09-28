"""Paired bootstrap: hardware-feature kernel SVM vs exact-feature kernel SVM.

Reads every results/kernel_analysis/hardware_<backend>_<stamp>.npz written by
kernel_hardware.py, refits the projected-kernel SVM (gamma=1, C=1) on the
measured and on the exact features, and bootstraps the paired difference on the
same test rows. Writes hardware_stats.json.

Run from src/:  python kernel_hardware_stats.py
"""
import glob
import json
import os

import numpy as np
from sklearn.svm import SVC

from kernel_analysis import RESULTS, rbf, score_metrics
from evaluation import paired_bootstrap


def scores(Ftr, Fte, y):
    return SVC(kernel="precomputed", C=1.0).fit(rbf(Ftr, Ftr, 1.0), y).decision_function(rbf(Fte, Ftr, 1.0))


out = {}
for path in sorted(glob.glob(os.path.join(RESULTS, "hardware_*_*.npz"))):
    d = np.load(path)
    backend = os.path.basename(path).split("_")[1] + "_" + os.path.basename(path).split("_")[2]
    yq, yte = d["y_train"], d["y_test"]
    s_hw = scores(d["F_train"], d["F_test"], yq)
    s_ex = scores(d["F_train_exact"], d["F_test_exact"], yq)
    row = {"file": os.path.basename(path), "n_train": int(len(yq)), "n_test": int(len(yte)),
           "hardware": score_metrics(yte, s_hw), "exact": score_metrics(yte, s_ex),
           "exact_minus_hardware": {}}
    for mt in ("roc_auc", "auprc", "tpr_at_1pct_fpr"):
        b = paired_bootstrap(yte, s_hw, s_ex, mt, n_boot=2000)     # diff = exact - hardware
        row["exact_minus_hardware"][mt] = {"delta": b.observed_diff,
                                           "ci95": [b.ci95_low, b.ci95_high], "p": b.p_value}
    out[backend] = row
    print(backend, json.dumps(row["exact_minus_hardware"]))

json.dump(out, open(os.path.join(RESULTS, "hardware_stats.json"), "w"), indent=2)
print("wrote hardware_stats.json for", len(out), "device run(s)")
