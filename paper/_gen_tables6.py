"""Generate the hybrid-on-hardware table from results/kernel_analysis/hybrid_hardware_pooled.json
(written by src/hybrid_hardware_pool.py).

Per device, the three disjoint class-balanced 2,000-row subsets of NSL-KDD's official test set are
pooled (6,000 rows, 3,000 benign). Rows: the exact simulation of the model run on hardware, each
device, and, on the same rows, the tuned random forest and the three classical twins of the same
seed (Table attrib). Columns: correlation of the measured <Z_i> with exact; ROC-AUC; TPR@1%FPR on
all attacks, on attacks of types seen in training and on the novel types; and paired-bootstrap
differences in TPR@1%FPR against the random forest and against the tanh MLP twin.

Run from paper/:  python _gen_tables6.py
"""
import json
import os

R = os.path.join("..", "results", "kernel_analysis")
P = json.load(open(os.path.join(R, "hybrid_hardware_pooled.json"), encoding="utf-8"))
dev, common = P["devices"], P.get("common", {})
first = dev[sorted(dev)[0]]
f3 = lambda x: f"{x:.3f}"


def pfmt(p):
    return "$<0.001$" if p < 0.001 else f"${p:.3f}$" if p < 0.1 else f"${p:.2f}$"


def d(b):
    return f"${b['diff']:+.3f}$ [{pfmt(b['p'])}]" if b else ""


def row(label, r, m, vs_rf=None, vs_mlp=None):
    return (f"{label} & {r} & {f3(m['roc_auc'])} & {f3(m['tpr_at_1pct_fpr'])} & {f3(m['seen_types'])} & "
            f"{f3(m['novel_types'])} & {d(vs_rf)} & {d(vs_mlp)}" + r"\\")


rows = [row("Exact simulation", "1.000", first["exact"], common.get("exact_minus_rf_tpr_at_1pct_fpr"),
            common.get("exact_minus_mlp_tpr_at_1pct_fpr"))]
for b in sorted(dev):
    o = dev[b]
    rows.append(row(r"\texttt{" + b.replace("_", r"\_") + "}", f3(o["expectation_r"]), o["hardware"],
                    o["paired"]["hardware_minus_rf_tpr_at_1pct_fpr"], o["paired"].get("hardware_minus_mlp_tpr_at_1pct_fpr")))
rows.append(r"\addlinespace[2pt]")
rows.append(row("Random forest, 8 features", "", first["rf"]))
for k, lab in (("linear", "No middle layer"), ("mlp", "Tanh MLP twin"), ("fourier", "Fourier twin")):
    if k in first["twins"]:
        rows.append(row(lab, "", first["twins"][k], common.get(f"{k}_minus_rf_tpr_at_1pct_fpr")))

qpu = sorted({q for o in dev.values() for q in o["qpu_seconds"] if q is not None})
n_rows, n_ben = first["n_rows"], first["n_benign"]
fmt = lambda n: f"{n:,}".replace(",", "{,}")
tab = r"""\begin{table}[!htb]
\centering
\caption{The 4-qubit hybrid on hardware: NSL-KDD, """ + fmt(n_rows) + r""" class-balanced test rows per device (three disjoint
subsets of """ + fmt(n_rows // 3) + r""", one job each, """ + f"{qpu[0]}" + r""" to """ + f"{qpu[-1]}" + r""" QPU-seconds per job). Only the four $\langle Z_i\rangle$ that
feed the trained classical head are measured; $r$ is their correlation with the exact values. The lower
block gives, on the same rows, the tuned random forest and the classical twins of Table~\ref{tab:attrib}
(same seed as the model run on hardware). TPR@1\%FPR is also split into attacks of types seen in training
and of the novel types. The last two columns are paired-bootstrap differences in TPR@1\%FPR against the
random forest and against the tanh MLP twin (two-sided $p$ in brackets). Generated from the result files.}
\label{tab:hybridhw}
\scriptsize
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{2.5pt}
\begin{tabularx}{\textwidth}{@{}Yccccccc@{}}
\toprule
& & & \multicolumn{3}{c}{\textbf{TPR@1\%FPR}} & \multicolumn{2}{c}{$\bm{\Delta}$\textbf{TPR@1\%FPR vs}}\\
\cmidrule(lr){4-6}\cmidrule(lr){7-8}
\textbf{Source} & $\bm{r}$ & \textbf{ROC-AUC} & \textbf{All} & \textbf{Seen} & \textbf{Novel} & \textbf{Random forest} & \textbf{MLP twin}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
open("_tab_hybridhw.tex", "w", encoding="utf-8").write(tab)
print("wrote _tab_hybridhw.tex with", len(rows), "rows")
