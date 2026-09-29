"""Generate the training-sample robustness table from results/kernel_analysis/landscape.json (the first
2,000 rows, as published) and landscape_sample_n<N>_s<seed>.json (seeded samples stratified by attack
type, from src/kernel_analysis.py landscape_sample).

One row per training sample; the test set is always NSL-KDD's official one. Each column is the quantum
projected kernel's test metric minus the raw-feature RBF kernel's, each at its own setting chosen by the
stated rule: standard CV over the narrow grid (gamma <= 3; ROC-AUC and TPR@1%FPR), standard CV over the
wide grid (gamma <= 30), shift-aware CV over the wide grid, and the best test ROC-AUC in the wide grid
(oracle). Then the correlation across the 40 settings between the RBF kernel's standard-CV (and
shift-aware CV) score and its test ROC-AUC. Also writes _tab_samples_facts.json.

Run from paper/:  python _gen_tables9.py
"""
import glob
import json
import os
import re

import numpy as np

R = os.path.join("..", "results", "kernel_analysis")
NARROW = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


def pick(g, keep, rule="std_cv"):
    ks = [k for k in g if keep(float(k.split("|")[0]))]
    return max(ks, key=lambda k: g[k][rule])


def stats(d):
    q, r = d["families"]["Q IQP projected"], d["families"]["classical RBF raw"]
    auc = lambda g, k: g[k]["test"]["roc_auc"]
    tpr = lambda g, k: g[k]["test"]["tpr_at_1pct_fpr"]
    qn, rn = pick(q, lambda gm: gm in NARROW), pick(r, lambda gm: gm in NARROW)
    qw, rw = pick(q, lambda gm: True), pick(r, lambda gm: True)
    qs, rs = pick(q, lambda gm: True, "shift_cv"), pick(r, lambda gm: True, "shift_cv")
    t = np.array([auc(r, k) for k in r])
    cor = lambda key: float(np.corrcoef(np.array([r[k][key] for k in r]), t)[0, 1])
    return {"gap_narrow": auc(q, qn) - auc(r, rn), "gap_narrow_tpr": tpr(q, qn) - tpr(r, rn),
            "rbf_pick_narrow": rn, "rbf_pick_wide": rw,
            "gap_wide": auc(q, qw) - auc(r, rw), "gap_wide_tpr": tpr(q, qw) - tpr(r, rw),
            "gap_shift": auc(q, qs) - auc(r, rs), "gap_shift_tpr": tpr(q, qs) - tpr(r, rs),
            "gap_oracle": max(auc(q, k) for k in q) - max(auc(r, k) for k in r),
            "r_std": cor("std_cv"), "r_shift": cor("shift_cv"), "n_shift_folds": d["n_shift_folds"]}


runs = [(2000, "first rows (published)", stats(json.load(open(os.path.join(R, "landscape.json")))))]
key = lambda f: tuple(int(x) for x in re.search(r"_n(\d+)_s(\d+)\.json$", f).groups())
for f in sorted(glob.glob(os.path.join(R, "landscape_sample_n*_s*.json")), key=key):
    n, sd = key(f)
    runs.append((n, f"random, seed {sd}", stats(json.load(open(f)))))
runs.sort(key=lambda x: (x[0], x[1] != "first rows (published)", x[1]))

lines, prev = [], None
for n, lab, st in runs:
    if prev is not None and n != prev:
        lines.append(r"\addlinespace[2pt]")
    prev = n
    lines.append(f"{n:,}".replace(",", "{,}") + f" & {lab} & ${st['gap_narrow']:+.3f}$ & ${st['gap_narrow_tpr']:+.3f}$ & "
                 f"${st['gap_wide']:+.3f}$ & ${st['gap_shift']:+.3f}$ & ${st['gap_oracle']:+.3f}$ & "
                 f"${st['r_std']:+.2f}$ & ${st['r_shift']:+.2f}$" + r"\\")

tab = r"""\begin{table}[!htb]
\centering
\caption{The surrogate artefact across training samples on NSL-KDD. Every row repeats the kernel comparison
of Table~\ref{tab:fair} on a different training sample, the published first 2{,}000 rows or a seeded sample
of the stated size stratified by attack type, and tests on the official test set. Each $\Delta$ is the
quantum projected kernel's test ROC-AUC (or TPR@1\%FPR) minus the raw-feature RBF kernel's, each at its own
setting chosen by standard CV over the narrow grid ($\gamma\le3$), standard CV over the wide grid
($\gamma\le30$), shift-aware CV over the wide grid, or the best test ROC-AUC in the wide grid (oracle). $r$
is the correlation across the 40 settings between the RBF kernel's CV score and its test ROC-AUC, for
standard and shift-aware CV. Generated from the result files.}
\label{tab:samples}
\scriptsize
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{2.5pt}
\begin{tabularx}{\textwidth}{@{}lYccccccc@{}}
\toprule
& & \multicolumn{2}{c}{\textbf{Narrow, std.\ CV}} & \textbf{Wide, std.} & \textbf{Wide, shift} & \textbf{Oracle} & \multicolumn{2}{c}{\textbf{RBF $\bm{r}$}}\\
\cmidrule(lr){3-4}\cmidrule(lr){8-9}
\textbf{Rows} & \textbf{Sample} & $\bm{\Delta}$\textbf{AUC} & $\bm{\Delta}$\textbf{TPR} & $\bm{\Delta}$\textbf{AUC} & $\bm{\Delta}$\textbf{AUC} & $\bm{\Delta}$\textbf{AUC} & \textbf{Std.} & \textbf{Shift}\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
open("_tab_samples.tex", "w", encoding="utf-8").write(tab)
json.dump([{"n": n, "sample": lab, **st} for n, lab, st in runs], open("_tab_samples_facts.json", "w"), indent=2)
print("wrote _tab_samples.tex with", len(runs), "runs")
for n, lab, st in runs:
    print(f"  {n:5d} {lab:24s} narrow {st['gap_narrow']:+.3f} (tpr {st['gap_narrow_tpr']:+.3f}, RBF {st['rbf_pick_narrow']}) | "
          f"wide {st['gap_wide']:+.3f} (tpr {st['gap_wide_tpr']:+.3f}, RBF {st['rbf_pick_wide']}) | "
          f"shift {st['gap_shift']:+.3f} (tpr {st['gap_shift_tpr']:+.3f}) | oracle {st['gap_oracle']:+.3f} | "
          f"r_std {st['r_std']:+.2f} r_shift {st['r_shift']:+.2f} (folds {st['n_shift_folds']})")
