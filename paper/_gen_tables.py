"""Generate the kernel-analysis LaTeX tables directly from the result JSONs, so
no number in the paper is hand-transcribed. Writes _tab_fair.tex and, when the
UNSW results exist, _tab_unsw.tex.

Run from paper/:  python _gen_tables.py
"""
import json
import os

R = os.path.join("..", "results", "kernel_analysis")
load = lambda n: json.load(open(os.path.join(R, n), encoding="utf-8"))
f3 = lambda x: f"{x:.3f}"

fair = load("fair.json")["models"]
deq = load("dequant.json")["models"]
land = load("landscape.json")["families"]


def pick(fam, rule):
    g = land[fam]
    k = max(g, key=lambda kk: g[kk][rule])
    return g[k][rule], g[k]["test"]


def row(label, feats, sel, score, m):
    s = f3(score) if score is not None else "n/a"
    return (f"{label} & {feats} & {sel} & {s} & {f3(m['roc_auc'])} & {f3(m['auprc'])} "
            f"& {f3(m['tpr_at_1pct_fpr'])}\\\\")


rows = [r"\multicolumn{7}{@{}l}{\emph{Quantum}}\\"]
rows.append(row("IQP projected, published", 24, "fixed", None,
                fair["Q proj (published: g=1,C=1)"]["metrics"]))
rows.append(row("IQP projected", 24, r"CV, $\gamma\le3$", deq["Q IQP projected (tuned)"]["cv_auc"],
                deq["Q IQP projected (tuned)"]["metrics"]))
rows.append(row("IQP projected", 24, r"CV, $\gamma\le30$", *pick("Q IQP projected", "std_cv")))
rows.append(row("IQP projected", 24, r"shift-aware CV", *pick("Q IQP projected", "shift_cv")))
rows.append(row("IQP fidelity", 256, "CV, $C$ only", *pick("Q IQP fidelity", "std_cv")))
rows.append(r"\addlinespace[2pt]")
rows.append(r"\multicolumn{7}{@{}l}{\emph{Classical, no quantum state}}\\")
rows.append(row("Random features, 1{,}200 rows, published", 512, "fixed", None,
                fair["rf-kern (published: 1200 rows)"]["metrics"]))
rows.append(row("Random features, 2{,}000 rows", 512, "fixed", None,
                fair["rf-kern (same 2000 rows)"]["metrics"]))
rows.append(row("RBF, raw features", 8, "fixed", None,
                fair["exact RBF (g=1,C=1)"]["metrics"]))
rows.append(row("RBF, raw features", 8, r"CV, $\gamma\le3$", deq["classical RBF raw (tuned)"]["cv_auc"],
                deq["classical RBF raw (tuned)"]["metrics"]))
rows.append(row("RBF, raw features", 8, r"CV, $\gamma\le30$", *pick("classical RBF raw", "std_cv")))
rows.append(row("RBF, raw features", 8, "shift-aware CV", *pick("classical RBF raw", "shift_cv")))
rows.append(row(r"Trig $(\sin x_i,\cos x_i)$ only", 16, r"CV, $\gamma\le3$", deq["classical trig [sin,cos]"]["cv_auc"],
                deq["classical trig [sin,cos]"]["metrics"]))
rows.append(row("Phase kernel", 46, r"CV, $\gamma\le3$",
                deq["classical trig + IQP phases"]["cv_auc"], deq["classical trig + IQP phases"]["metrics"]))
rows.append(row("Phase kernel", 46, r"CV, $\gamma\le30$",
                *pick("classical trig + IQP phases", "std_cv")))
rows.append(row("Phase kernel", 46, "shift-aware CV",
                *pick("classical trig + IQP phases", "shift_cv")))

tab = r"""\begin{table}[t]
\centering
\caption{Kernels on identical data (first 2{,}000 NSL-KDD training rows; official test set of
22{,}544 rows containing 17 attack types absent from training). Exact Gram matrices and one kernel SVM
solver throughout. \emph{Score} is the selection criterion's own value: in-distribution 5-fold CV AUC,
or leave-attack-types-out CV AUC for shift-aware rows. The phase kernel is an RBF kernel on
$(\sin x_i,\cos x_i)$ plus the ZZ feature map's phases, with no quantum state. Fixed rows use
$\gamma=1$, $C=1$. Generated from the result files.}
\label{tab:fair}
\scriptsize
\renewcommand{\arraystretch}{1.12}
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{@{}Ylccccc@{}}
\toprule
\textbf{Kernel} & \textbf{Feat.} & \textbf{Selection} & \textbf{Score} & \textbf{Test AUC} & \textbf{AUPRC} & \textbf{TPR@1\%FPR}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
open("_tab_fair.tex", "w", encoding="utf-8").write(tab)
print("wrote _tab_fair.tex with", len([r for r in rows if r.count("&") == 6]), "data rows")


