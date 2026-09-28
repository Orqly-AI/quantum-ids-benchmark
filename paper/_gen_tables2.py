"""Generate the two-official-splits kernel table from the landscape JSONs.

Run from paper/:  python _gen_tables2.py
"""
import json
import os

import numpy as np

R = os.path.join("..", "results", "kernel_analysis")
load = lambda n: json.load(open(os.path.join(R, n), encoding="utf-8"))
f3 = lambda x: f"{x:.3f}"

sets = [("NSL-KDD", load("landscape.json")["families"]),
        ("UNSW-NB15", load("landscape_unsw.json")["families"])]
label = {"Q IQP projected": "Quantum IQP projected",
         "classical trig + IQP phases": "Phase kernel (classical)",
         "classical RBF raw": "RBF, raw features (classical)"}

lines = []
for ds, fams in sets:
    for i, fam in enumerate(("Q IQP projected", "classical trig + IQP phases", "classical RBF raw")):
        g = fams[fam]
        k = max(g, key=lambda kk: g[kk]["std_cv"])
        m = g[k]["test"]
        t = np.array([v["test"]["roc_auc"] for v in g.values()])
        s = np.array([v["std_cv"] for v in g.values()])
        r = float(np.corrcoef(s, t)[0, 1])
        cell = r"\multirow{3}{*}{" + ds + "}" if i == 0 else ""
        lines.append(f"{cell} & {label[fam]} & {f3(m['roc_auc'])} & {f3(m['auprc'])} & "
                     f"{f3(m['tpr_at_1pct_fpr'])} & {f3(t.min())} & ${r:+.2f}$" + r"\\")
    lines.append(r"\addlinespace[3pt]")
lines = lines[:-1]

head = r"""\begin{table}[t]
\centering
\caption{Kernels on the two datasets with an official train/test split, each selected by standard
5-fold CV over the full grid ($\gamma\le30$, $C\le100$). \emph{Worst} is the lowest test ROC-AUC over all
40 settings; $r$ is the correlation, across those settings, between the CV score and the test ROC-AUC.
NSL-KDD's test set contains 17 attack types absent from training; UNSW-NB15's contains none. Generated
from the result files.}
\label{tab:twosplits}
\scriptsize
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{4pt}
\begin{tabularx}{\textwidth}{@{}lYccccc@{}}
\toprule
\textbf{Dataset} & \textbf{Kernel} & \textbf{Test AUC} & \textbf{AUPRC} & \textbf{TPR@1\%FPR} & \textbf{Worst} & $\bm{r}$\\
\midrule
"""
tail = r"""
\bottomrule
\end{tabularx}
\end{table}
"""
open("_tab_twosplits.tex", "w", encoding="utf-8").write(head + "\n".join(lines) + tail)
print("wrote _tab_twosplits.tex")
