"""Generate the hardware-validation table from results/kernel_analysis/hardware_*.json.

One row per feature source: exact statevector, noisy local simulation of a Heron
device, and each real-device run present. Agreement columns compare that source's
features and Gram matrix with the exact ones; metric columns are the projected-kernel
SVM trained and tested on that source's features.

Run from paper/:  python _gen_tables4.py
"""
import glob
import json
import os

R = os.path.join("..", "results", "kernel_analysis")
f3 = lambda x: f"{x:.3f}"


def row(label, r_feat, r_gram, m, extra=""):
    return (f"{label} & {r_feat} & {r_gram} & {f3(m['roc_auc'])} & {f3(m['auprc'])} & "
            f"{f3(m['tpr_at_1pct_fpr'])}{extra}" + r"\\")


rows, notes = [], []
dry = json.load(open(os.path.join(R, "hardware_dryrun.json"), encoding="utf-8"))
rows.append(row("Exact statevector", "1.000", "1.000", dry["exact_train_exact_test"]))
rows.append(row(r"Noisy simulation (FakeFez, " + str(dry["shots"]) + " shots)",
                f3(dry["feature_agreement"]["pearson_r_test"]), f3(dry["gram_agreement"]["pearson_r_offdiag"]),
                dry["hardware_train_hardware_test"]))
for path in sorted(glob.glob(os.path.join(R, "hardware_run_*.json"))):
    d = json.load(open(path, encoding="utf-8"))
    label = r"\texttt{" + d["backend"].replace("_", r"\_") + "} (" + str(d["shots"]) + " shots)"
    rows.append(row(label, f3(d["feature_agreement"]["pearson_r_test"]),
                    f3(d["gram_agreement"]["pearson_r_offdiag"]), d["hardware_train_hardware_test"]))
    q = d.get("qpu_seconds")
    notes.append(f"{d['backend']}: job {', '.join(d['job_ids'])}"
                 + (f", {q} QPU seconds" if q is not None else ""))

n_tr, n_te = dry["n_train"], dry["n_test"]
tab = r"""\begin{table}[t]
\centering
\caption{Hardware validation of the IQP projected kernel on NSL-KDD (""" + f"{n_tr}" + r""" training and
""" + f"{n_te}" + r""" test samples, the latter class-balanced). Agreement columns are Pearson correlations
of the source's Pauli features and of its Gram matrix (off-diagonal) with the exact statevector values.
Metric columns are for the projected-kernel SVM trained and tested on that source's own features
($\gamma=1$, $C=1$). Generated from the result files.}
\label{tab:hardware}
\footnotesize
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{@{}Yccccc@{}}
\toprule
& \multicolumn{2}{c}{\textbf{Agreement with exact}} & \multicolumn{3}{c}{\textbf{Projected-kernel SVM}}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-6}
\textbf{Feature source} & \textbf{Feat.\ $r$} & \textbf{Gram $r$} & \textbf{ROC-AUC} & \textbf{AUPRC} & \textbf{TPR@1\%FPR}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
open("_tab_hardware.tex", "w", encoding="utf-8").write(tab)
print("wrote _tab_hardware.tex with", len(rows), "rows")
for n in notes:
    print("  ", n)
