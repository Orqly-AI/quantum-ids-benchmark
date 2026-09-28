"""Generate the induced-shift table from results/kernel_analysis/{landscape_unsw,induced_unsw_a,induced_unsw_b}.json.

Three UNSW-NB15 protocols: the official split (no unseen attack types), and two
in which whole attack categories are removed from the training sample so that
they appear only at test time. For each kernel: the correlation across the 40
grid settings between standard 5-fold CV and test ROC-AUC, the test ROC-AUC of
the CV-selected setting, the best test ROC-AUC in the grid (oracle), and the
TPR@1%FPR of the CV-selected setting against the best in the grid. Standard-CV
statistics do not depend on the leave-attack-types-out fold construction.

Run from paper/:  python _gen_tables5.py
"""
import json
import os

import numpy as np

R = os.path.join("..", "results", "kernel_analysis")
load = lambda n: json.load(open(os.path.join(R, n), encoding="utf-8"))
f3 = lambda x: f"{x:.3f}"
label = {"Q IQP projected": "Quantum IQP projected",
         "classical trig + IQP phases": "Phase kernel (classical)",
         "classical RBF raw": "RBF, raw features (classical)"}
short = {"Exploits": "Exploits", "Fuzzers": "Fuzzers", "Reconnaissance": "Recon.", "Generic": "Generic",
         "DoS": "DoS", "Analysis": "Analysis", "Backdoor": "Backdoor"}

sets = [("official split", load("landscape_unsw.json")),
        ("A", load("induced_unsw_a.json")), ("B", load("induced_unsw_b.json"))]
lines, notes = [], []
for tag, d in sets:
    hold = d.get("holdout") or []
    frac = d.get("unseen_test_fraction", 0.0)
    if hold:
        notes.append(f"{tag}: " + ", ".join(short.get(h, h) for h in hold) + f" ({100 * frac:.0f}\\% of test rows)")
        head = r"\multirow{3}{*}{Holdout " + tag + r" (" + f"{100 * frac:.0f}" + r"\%)}"
    else:
        head = r"\multirow{3}{*}{None (0\%)}"
    for i, fam in enumerate(("Q IQP projected", "classical trig + IQP phases", "classical RBF raw")):
        g = d["families"][fam]
        keys = list(g)
        s = np.array([g[k]["std_cv"] for k in keys])
        t = np.array([g[k]["test"]["roc_auc"] for k in keys])
        t1 = np.array([g[k]["test"]["tpr_at_1pct_fpr"] for k in keys])
        k = keys[int(np.argmax(s))]
        m = g[k]["test"]
        r = float(np.corrcoef(s, t)[0, 1])
        cell = head if i == 0 else ""
        lines.append(f"{cell} & {label[fam]} & ${r:+.2f}$ & {f3(m['roc_auc'])} & {f3(t.max())} & "
                     f"{f3(m['tpr_at_1pct_fpr'])} & {f3(t1.max())}" + r"\\")
    lines.append(r"\addlinespace[3pt]")
lines = lines[:-1]

head = r"""\begin{table}[!htb]
\centering
\caption{Inducing novel-attack shift on UNSW-NB15. Whole attack categories are removed from the
2{,}000-row training sample and kept in the 20{,}000-row test sample: holdout A removes """ + notes[0].split(": ")[1] + r"""; holdout B removes """ + notes[1].split(": ")[1] + r""". Each kernel is
selected by standard 5-fold CV over the full grid ($\gamma\le30$, $C\le100$); $r$ is the correlation
across the 40 settings between the CV score and the test ROC-AUC, and \emph{oracle} the best test ROC-AUC
in the grid. The last two columns give TPR@1\%FPR at the CV-selected setting and the best in the grid.
Generated from the result files.}
\label{tab:induced}
\scriptsize
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{@{}lYccccc@{}}
\toprule
& & & \multicolumn{2}{c}{\textbf{Test ROC-AUC}} & \multicolumn{2}{c}{\textbf{TPR@1\%FPR}}\\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}
\textbf{Unseen types} & \textbf{Kernel} & $\bm{r}$ & \textbf{CV pick} & \textbf{Oracle} & \textbf{CV pick} & \textbf{Best}\\
\midrule
"""
tail = r"""
\bottomrule
\end{tabularx}
\end{table}
"""
open("_tab_induced.tex", "w", encoding="utf-8").write(head + "\n".join(lines) + tail)
print("wrote _tab_induced.tex")
for n in notes:
    print("  ", n)
