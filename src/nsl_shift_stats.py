"""Shift statistics for NSL-KDD's official split, which results/kernel_analysis/landscape.json predates.

The same _shift_stats that landscape_ds and the NSL-KDD test-set controls record: unbiased squared MMD
between the first 2,000 training rows and the official test set on the raw features (label-free), and
its class-conditional versions. Used for Figure 6.

Run from src/:  python nsl_shift_stats.py  -> results/kernel_analysis/nsl_shift_stats.json
"""
import json
import os

import numpy as np

from kernel_analysis import RESULTS, _shift_stats, load_nslkdd

Xtr, ytr, Xte, yte = load_nslkdd(8)
st = _shift_stats(np.asarray(Xtr[:2000], float), np.asarray(ytr[:2000]), np.asarray(Xte, float), np.asarray(yte))
json.dump({"landscape.json": st}, open(os.path.join(RESULTS, "nsl_shift_stats.json"), "w"), indent=2)
print(st)
