"""Generate the hybrid-on-hardware table from results/kernel_analysis/hybrid_hardware_*.json.

One row per source of the four <Z_i> that feed the trained classical head: the
exact statevector, shot noise alone (binomial resampling of the exact values),
a noisy local simulation of a Heron device, and each real-device run present;
plus the tuned random forest on the same test rows. The agreement column is the
Pearson correlation of the source's expectations with the exact ones; metric
columns are the hybrid's own on that source; the last column is the
paired-bootstrap difference in TPR@1%FPR against the random forest on the same
rows. (The correlation of the final scores with the exact ones, 0.99 on every
device, is quoted in the text.)

Run from paper/:  python _gen_tables6.py
"""
import glob
import json
import os

R = os.path.join("..", "results", "kernel_analysis")
load = lambda n: json.load(open(os.path.join(R, n), encoding="utf-8"))
f3 = lambda x: f"{x:.3f}"


def pfmt(p):
    return "$<0.001$" if p < 0.001 else f"${p:.3f}$" if p < 0.01 else f"${p:.2f}$"


def row(label, r_z, m, vs_rf=None):
    cell = (f"${vs_rf['diff']:+.3f}$ [{pfmt(vs_rf['p'])}]" if vs_rf else "")
    return (f"{label} & {r_z} & {f3(m['roc_auc'])} & {f3(m['auprc'])} & "
            f"{f3(m['tpr_at_1pct_fpr'])} & {cell}" + r"\\")


runs = sorted(glob.glob(os.path.join(R, "hybrid_hardware_run_*.json")))
dry = load("hybrid_hardware_dryrun.json")
sh = load("hybrid_hardware_shots.json")
ref = json.load(open(runs[0], encoding="utf-8")) if runs else dry     # any run carries the RF rows
n_test, shots = dry["n_test"], dry["shots"]
assert sh["n_test"] == n_test and str(shots) in sh

rows, notes = [], []
rows.append(row("Exact statevector", "1.000", dry["exact"], ref.get("exact_minus_rf_tpr_at_1pct_fpr")))
m_sh = {k: v["mean"] for k, v in sh[str(shots)].items()}
rows.append(row(f"Shot noise only ({shots} shots)", "", m_sh))
rows.append(row(f"FakeFez simulation ({shots} shots)", f3(dry["expectation_agreement"]["pearson_r"]),
                dry["hardware"]))
score_r = []
for path in runs:
    d = json.load(open(path, encoding="utf-8"))
    label = r"\texttt{" + d["backend"].replace("_", r"\_") + "} (" + str(d["shots"]) + " shots)"
    rows.append(row(label, f3(d["expectation_agreement"]["pearson_r"]), d["hardware"],
                    d.get("hardware_minus_rf_tpr_at_1pct_fpr")))
    score_r.append(d["score_agreement"]["pearson_r"])
    q = d.get("qpu_seconds")
    notes.append(f"{d['backend']}: job {', '.join(d['job_ids'])}" + (f", {q} QPU seconds" if q is not None else "")
                 + f", score r {d['score_agreement']['pearson_r']:.3f}")
if "rf_same_rows" in ref:
    rows.append(r"\addlinespace[2pt]")
    rows.append(row("Tuned random forest, same rows", "", ref["rf_same_rows"]))

tab = r"""\begin{table}[!htb]
\centering
\caption{The 4-qubit hybrid on hardware: NSL-KDD, """ + f"{n_test:,}".replace(",", "{,}") + r""" class-balanced test rows.
Only the four $\langle Z_i\rangle$ that feed the trained classical head are measured; every other part of
the model is the published one. $r$ is the Pearson correlation of the source's expectations with the
exact statevector values. Metric columns are the hybrid's own on that source; the shot-noise row is the
mean of """ + str(sh["reps"]) + r""" binomial resamplings of the exact expectations. The last column is the
paired-bootstrap difference in TPR@1\%FPR against the tuned random forest on the same rows (positive
favours the hybrid; two-sided $p$ in brackets). Generated from the result files.}
\label{tab:hybridhw}
\scriptsize
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{@{}Yccccc@{}}
\toprule
& & \multicolumn{3}{c}{\textbf{Hybrid QNN}} & \\
\cmidrule(lr){3-5}
\textbf{Source of $\langle Z_i\rangle$} & $\bm{r}$ & \textbf{ROC-AUC} & \textbf{AUPRC} & \textbf{TPR@1\%FPR} & $\bm{\Delta}$\textbf{TPR vs RF}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
open("_tab_hybridhw.tex", "w", encoding="utf-8").write(tab)
print("wrote _tab_hybridhw.tex with", len(rows), "rows;", len(runs), "device runs")
for n in notes:
    print("  ", n)
