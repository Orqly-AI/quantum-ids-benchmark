"""Generate the NSL-KDD shift-decomposition table from results/kernel_analysis/landscape*.json.

The 2,000 training rows, the grid and cross-validation are identical in every row;
only the evaluation set changes: the official test set; the same without the 17 attack
types absent from training; that, importance-weighted to the training file's attack-type
mix; and 20,000 held-out rows of the training file (no shift). For each: the correlation
across the 40 settings between standard-CV and test ROC-AUC for each kernel; under the
narrow grid (gamma <= 3) the quantum-minus-RBF ROC-AUC gap at the CV-selected settings and
the RBF pick's TPR@1%FPR; the same gap under the wide grid; and the shift-aware CV
correlation for the RBF kernel.

Run from paper/:  python _gen_tables8.py
"""
import json
import os

import numpy as np

R = os.path.join("..", "results", "kernel_analysis")
NARROW = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
SETS = [("landscape.json", "Official test set"),
        ("landscape_nsl_seen.json", "Without novel types"),
        ("landscape_nsl_seen_rw.json", r"\ldots{} reweighted mix"),
        ("landscape_nsl_iid.json", "Held-out training rows")]
FAMS = ("Q IQP projected", "classical trig + IQP phases", "classical RBF raw")


def pick(g, keep):
    ks = [k for k in g if keep(float(k.split("|")[0]))]
    return max(ks, key=lambda k: g[k]["std_cv"])


lines, facts = [], {}
for fname, label in SETS:
    d = json.load(open(os.path.join(R, fname), encoding="utf-8"))
    fam = d["families"]
    rs = []
    for f in FAMS:
        g = fam[f]
        s = np.array([g[k]["std_cv"] for k in g]); t = np.array([g[k]["test"]["roc_auc"] for k in g])
        rs.append(float(np.corrcoef(s, t)[0, 1]))
    g_rbf, g_q = fam["classical RBF raw"], fam["Q IQP projected"]
    sh = np.array([g_rbf[k]["shift_cv"] for k in g_rbf]); t = np.array([g_rbf[k]["test"]["roc_auc"] for k in g_rbf])
    r_shift = float(np.corrcoef(sh, t)[0, 1])
    qn, rn = pick(g_q, lambda gm: gm in NARROW), pick(g_rbf, lambda gm: gm in NARROW)
    qw, rw = pick(g_q, lambda gm: True), pick(g_rbf, lambda gm: True)
    gap_n = g_q[qn]["test"]["roc_auc"] - g_rbf[rn]["test"]["roc_auc"]
    gap_w = g_q[qw]["test"]["roc_auc"] - g_rbf[rw]["test"]["roc_auc"]
    tpr_rbf_n = g_rbf[rn]["test"]["tpr_at_1pct_fpr"]
    n_test = d.get("n_test", 22544)
    facts[fname] = dict(r=rs, r_shift=r_shift, gap_narrow=gap_n, gap_wide=gap_w, rbf_tpr_narrow=tpr_rbf_n,
                        rbf_pick_narrow=rn, n_test=n_test)
    lines.append(f"{label} & {n_test:,}".replace(",", "{,}") + " & " + " & ".join(f"${r:+.2f}$" for r in rs)
                 + f" & ${gap_n:+.3f}$ & {tpr_rbf_n:.3f} & ${gap_w:+.3f}$ & ${r_shift:+.2f}$" + r"\\")

tab = r"""\begin{table}[!htb]
\centering
\caption{Decomposing NSL-KDD's shift. The 2{,}000 training rows, the grid and the cross-validation are
identical in every row; only the evaluation set changes. $r$ is the correlation across the 40 grid
settings between standard-CV ROC-AUC and test ROC-AUC. Under the narrow grid ($\gamma\le3$) and the wide
grid ($\gamma\le30$), $\Delta$AUC is the quantum projected kernel's test ROC-AUC minus the raw-feature RBF
kernel's, each at its CV-selected setting; \emph{RBF TPR} is that RBF setting's TPR@1\%FPR. The last
column is the shift-aware (leave-attack-types-out) CV correlation for the RBF kernel. Reweighting gives
each attack type present in the test set its share of the training file. Generated from the result files.}
\label{tab:decomp}
\scriptsize
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{2.5pt}
\begin{tabularx}{\textwidth}{@{}Ycccccccc@{}}
\toprule
& & \multicolumn{3}{c}{\textbf{Standard CV $\bm{r}$}} & \multicolumn{2}{c}{\textbf{Narrow grid}} & \textbf{Wide} & \textbf{Shift-aware}\\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}
\textbf{Test set} & \textbf{Rows} & \textbf{Quantum} & \textbf{Phase} & \textbf{RBF} & $\bm{\Delta}$\textbf{AUC} & \textbf{RBF TPR} & $\bm{\Delta}$\textbf{AUC} & \textbf{RBF $\bm{r}$}\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
open("_tab_decomp.tex", "w", encoding="utf-8").write(tab)
print("wrote _tab_decomp.tex")
for k, v in facts.items():
    print(f"  {k:30s} r(Q,ph,RBF)={[round(x, 2) for x in v['r']]} gap_narrow={v['gap_narrow']:+.3f} "
          f"gap_wide={v['gap_wide']:+.3f} rbf_tpr_narrow={v['rbf_tpr_narrow']:.3f} ({v['rbf_pick_narrow']}) "
          f"r_shift={v['r_shift']:+.2f} n={v['n_test']}")
