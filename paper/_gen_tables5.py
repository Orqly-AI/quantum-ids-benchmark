"""Generate the induced-shift summary table from results/kernel_analysis/induced_summary.json
(written by src/induced_summary.py).

Rows group the runs: no holdout (one per dataset), UNSW-NB15 with one attack family held out,
UNSW-NB15 holdouts A and B over three sampling seeds, ToN-IoT and CICIDS2017 with one family held
out. Per group: the smallest CV-to-test correlation over the three kernels (median and range over
runs), the largest selection loss (median and maximum), the number of runs in which selection
flips the sign of the quantum-minus-best-classical ROC-AUC gap, the largest error selection
introduces into that gap, and the largest quantum lead at the oracle settings.

Run from paper/:  python _gen_tables5.py
"""
import json
import os

import numpy as np

R = os.path.join("..", "results", "kernel_analysis")
S = json.load(open(os.path.join(R, "induced_summary.json"), encoding="utf-8"))
FAM = {"Q IQP projected": "quantum", "classical trig + IQP phases": "phase", "classical RBF raw": "rbf"}


def per_family(g):
    keys = list(g)
    s = np.array([g[k]["std_cv"] for k in keys]); t = np.array([g[k]["test"]["roc_auc"] for k in keys])
    i = int(np.argmax(s))
    return {"r": float(np.corrcoef(s, t)[0, 1]), "pick_auc": float(t[i]), "oracle_auc": float(t.max()),
            "loss": float(t.max() - t[i])}


def gaps(f):
    gp = f["quantum"]["pick_auc"] - max(f["phase"]["pick_auc"], f["rbf"]["pick_auc"])
    go = f["quantum"]["oracle_auc"] - max(f["phase"]["oracle_auc"], f["rbf"]["oracle_auc"])
    return gp, go


runs = S["runs"]          # includes each dataset's no-holdout run, induced_<ds>_none_s0.json
for r in runs:
    r["min_r"] = min(v["r"] for v in r["families"].values())
    r["max_loss"] = max(v["loss"] for v in r["families"].values())
    r["gap_pick"], r["gap_oracle"] = gaps(r["families"])

AB = ({"exploits", "fuzzers", "reconnaissance"}, {"generic", "dos", "analysis", "backdoor"})
groups = [
    ("No holdout", [r for r in runs if not r["holdout"]]),
    ("UNSW-NB15, single family", [r for r in runs if r["dataset"] == "unsw" and len(r["holdout"]) == 1]),
    ("UNSW-NB15, A and B, 3 seeds", [r for r in runs if r["dataset"] == "unsw"
                                             and {h.lower() for h in r["holdout"]} in AB]),
    ("ToN-IoT, single family", [r for r in runs if r["dataset"] == "toniot" and r["holdout"]]),
    ("CICIDS2017, single family", [r for r in runs if r["dataset"] == "cicids" and r["holdout"]]),
]
lines, facts = [], {}
for label, rs in groups:
    mr = np.array([r["min_r"] for r in rs]); ml = np.array([r["max_loss"] for r in rs])
    err = np.array([r["gap_pick"] - r["gap_oracle"] for r in rs])
    flips = sum(1 for r in rs if np.sign(r["gap_pick"]) != np.sign(r["gap_oracle"]))
    qlead = max(r["gap_oracle"] for r in rs)
    i = int(np.argmax(np.abs(err)))
    facts[label] = dict(n=len(rs), min_r_median=float(np.median(mr)), min_r_min=float(mr.min()),
                        loss_median=float(np.median(ml)), loss_max=float(ml.max()), flips=flips,
                        err_max=float(err[i]), err_run=rs[i]["file"], qlead_oracle_max=float(qlead))
    rng = f"$[{mr.min():+.2f}, {mr.max():+.2f}]$" if len(rs) > 1 else ""
    lines.append(f"{label} & {len(rs)} & ${np.median(mr):+.2f}$ & {rng} & {np.median(ml):.3f} & {ml.max():.3f} & "
                 f"{flips} & ${err[i]:+.3f}$ & ${qlead:+.3f}$" + r"\\")
    if label == "No holdout":
        lines.append(r"\addlinespace[3pt]")

c = S["correlations"]
cap_corr = (f"Across the {S['n_holdout_runs']} runs with a holdout, the label-free squared MMD between the training "
            f"sample and the unlabelled test sample correlates with the smallest $r$ at Spearman "
            f"$\\rho={c['shift_mmd2_all~min_r']['spearman']:+.2f}$ (permutation $p={c['shift_mmd2_all~min_r']['p_perm']:.3f}$) "
            f"and with the largest loss at $\\rho={c['shift_mmd2_all~max_loss']['spearman']:+.2f}$ "
            f"($p={c['shift_mmd2_all~max_loss']['p_perm']:.3f}$).")
tab = r"""\begin{table}[!htb]
\centering
\caption{Inducing shift on three datasets. Whole attack families are removed from the 2{,}000-row training
sample and kept in the 20{,}000-row test sample; every kernel is selected by standard 5-fold CV over the
full grid. \emph{Min $r$} is, per run, the smallest correlation over the three kernels between CV score and
test ROC-AUC across the 40 settings (median and range over the group's runs); \emph{Loss} is the largest
gap between a kernel's best test ROC-AUC in the grid (oracle) and its CV-selected one. \emph{Flips} counts
runs in which the sign of the quantum-minus-best-classical ROC-AUC gap differs between the CV-selected and
the oracle settings; \emph{Sel.\ error} is the largest difference between those two gaps, and \emph{Q lead}
the largest quantum lead at the oracle settings. """ + cap_corr + r""" Generated from the result files.}
\label{tab:induced}
\scriptsize
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{2.5pt}
\begin{tabularx}{\textwidth}{@{}Ycccccccc@{}}
\toprule
& & \multicolumn{2}{c}{\textbf{Min $\bm{r}$}} & \multicolumn{2}{c}{\textbf{Loss}} & & & \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
\textbf{Runs} & \textbf{$\bm{n}$} & \textbf{Median} & \textbf{Range} & \textbf{Median} & \textbf{Max} & \textbf{Flips} & \textbf{Sel.\ error} & \textbf{Q lead}\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
open("_tab_induced.tex", "w", encoding="utf-8").write(tab)
json.dump(facts, open("_tab_induced_facts.json", "w"), indent=2)
print("wrote _tab_induced.tex")
for k, v in facts.items():
    print(f"  {k:40s} {v}")
