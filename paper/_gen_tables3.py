"""Generate the geometric-difference and finite-shot tables from the JSONs.

Run from paper/:  python _gen_tables3.py
"""
import json
import os

R = os.path.join("..", "results", "kernel_analysis")
load = lambda n: json.load(open(os.path.join(R, n), encoding="utf-8"))
f2 = lambda x: f"{x:.2f}"
f3 = lambda x: f"{x:.3f}"

# ---------------- geometric difference ----------------
gt = load("geometry_trig.json")
g0 = load("geometry.json")
names = {"nslkdd": "NSL-KDD", "unsw": "UNSW-NB15", "cicids": "CICIDS2017", "toniot": "NF-ToN-IoT-v2"}
lam = f"lam={gt['nslkdd']['lambda']:g}"
rows = []
for ds in ("nslkdd", "unsw", "cicids", "toniot"):
    ang = g0[ds]["kernels"]["angle fidelity"][lam]["g_all"]
    ang_min = min(v for k, v in ang.items() if k != "linear")
    for i, (qn, lab) in enumerate((("IQP projected", "projected"), ("IQP fidelity", "fidelity"))):
        k = gt[ds]["kernels"][qn]
        first = names[ds] if i == 0 else ""
        angc = r"\multirow{2}{*}{" + f2(ang_min) + "}" if i == 0 else ""
        rows.append(f"{first} & {lab} & {f2(k['RBF raw']['g_min'])} & {f2(k['trig [sin,cos]']['g_min'])} & "
                    r"\textbf{" + f2(k['trig + IQP phases']['g_min']) + "} & " + angc + r"\\")
N = gt["nslkdd"]["N"]
geom = r"""\begin{table}[t]
\centering
\caption{Geometric difference $g(K_C\Vert K_Q)$, minimised over classical bandwidth
($N=""" + str(N) + r"""$, $\lambda=10^{-2}$, $\sqrt{N}\approx """ + f"{N ** 0.5:.1f}" + r"""$). A small $g$ means the
classical kernel can learn anything the quantum kernel can. The last column gives the angle-encoded
fidelity kernel against its best RBF. Rank-deficient kernels are excluded. Generated from the result
files.}
\label{tab:geom}
\footnotesize
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llcccc@{}}
\toprule
& & \multicolumn{3}{c}{\textbf{Classical family, vs IQP kernel}} & \textbf{Angle fidelity}\\
\cmidrule(lr){3-5}\cmidrule(lr){6-6}
\textbf{Dataset} & \textbf{IQP kernel} & \textbf{RBF raw} & \textbf{Trig} & \textbf{Phase kernel} & \textbf{vs RBF}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open("_tab_geom.tex", "w", encoding="utf-8").write(geom)

# ---------------- finite shots ----------------
sh = load("shots.json")
srows = []
for M, r in sh["budgets"].items():
    cell = lambda k: f"${r[k]['mean']:.3f}\\pm{r[k]['std']:.3f}$"
    srows.append(f"{int(M):,} & {3 * int(M):,} & {cell('roc_auc')} & {cell('auprc')} & {cell('tpr_at_1pct_fpr')}"
                 + r"\\")
e = sh["exact"]
srows.append(r"\addlinespace[2pt]")
srows.append(f"exact & $\\infty$ & ${f3(e['roc_auc'])}$ & ${f3(e['auprc'])}$ & ${f3(e['tpr_at_1pct_fpr'])}$" + r"\\")
shots = r"""\begin{table}[t]
\centering
\caption{Finite-shot robustness of the published IQP projected QSVM on NSL-KDD. $M$ shots per basis in
each of the X, Y and Z settings; mean and standard deviation over 5 sampling seeds. Generated from the
result files.}
\label{tab:shots}
\footnotesize
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{6pt}
\begin{tabular}{@{}rrccc@{}}
\toprule
\textbf{Shots / basis} & \textbf{Shots / sample} & \textbf{ROC-AUC} & \textbf{AUPRC} & \textbf{TPR@1\%FPR}\\
\midrule
""" + "\n".join(srows).replace(",", "{,}") + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open("_tab_shots.tex", "w", encoding="utf-8").write(shots)
print("wrote _tab_geom.tex and _tab_shots.tex")
