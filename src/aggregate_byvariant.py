"""Correct per-variant aggregation of the harness result JSONs.

The original aggregate.py groups by (dataset, model_type), which lumps together
distinct experiments that share a model_type (e.g. all hybrid VQC encodings,
qubit counts, and noise runs become one 'hybrid_qnn' bucket). This script groups
by the FULL variant identity (dataset, model_type, n_qubits, encoding, run-name)
so each row is a single experiment averaged over its own seeds, and it excludes
debug / smoke runs. Writes results/summary_byvariant.csv.
"""
import json, glob, os, csv
from collections import defaultdict
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "results")
# debug / smoke run-names to exclude from the published summary
EXCLUDE_NAMES = {"verify_sub"}
# the very first single-seed smoke runs used the default names at q6
EXCLUDE_IF_Q6_DEFAULT = {"RandomForest", "SVM_RBF", "ClassicalMLP"}
METRICS = ["accuracy","precision","recall","f1","detection_rate","false_positive_rate",
           "roc_auc","auprc","tpr_at_0.1pct_fpr","tpr_at_1pct_fpr","brier","ece"]

groups = defaultdict(lambda: defaultdict(list))   # key -> metric -> [values]
seeds  = defaultdict(set)

for f in sorted(glob.glob(os.path.join(RES, "*.json"))):
    if os.path.basename(f).startswith("baselines_"):
        continue
    try:
        d = json.load(open(f))
    except Exception:
        continue
    cfg = d.get("config", {}); m = d.get("metrics", {})
    name = cfg.get("name", "")
    if name in EXCLUDE_NAMES:
        continue
    if name in EXCLUDE_IF_Q6_DEFAULT and cfg.get("n_qubits") == 6:
        continue
    key = (cfg.get("dataset"), cfg.get("model_type"), cfg.get("n_qubits"),
           cfg.get("encoding"), name)
    seeds[key].add(cfg.get("seed"))
    for mk in METRICS:
        if mk in m and isinstance(m[mk], (int, float)):
            groups[key][mk].append(m[mk])

out = os.path.join(RES, "summary_byvariant.csv")
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    header = ["dataset","model_type","n_qubits","encoding","name","n_seeds"]
    for mk in METRICS:
        header += [f"{mk}_mean", f"{mk}_std"]
    w.writerow(header)
    for key in sorted(groups):
        ds, mt, nq, enc, name = key
        row = [ds, mt, nq, enc, name, len(seeds[key])]
        for mk in METRICS:
            vals = groups[key][mk]
            if vals:
                row += [round(float(np.mean(vals)), 4),
                        round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)]
            else:
                row += ["", ""]
        w.writerow(row)

n = len(groups)
print(f"wrote {out} with {n} distinct variants (excluded debug/smoke runs)")
