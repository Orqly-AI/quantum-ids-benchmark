"""Generate the hybrid attribution table from results/hybrid_attribution/{summary,certificate_s42}.json.

The 4-qubit hybrid and its classical twins (identical pipeline, quantum layer
replaced) on NSL-KDD's official test set, mean and standard deviation over the
seeds present, and the tuned random forest for reference. The last column
counts the seeds on which the hybrid's TPR@1%FPR differs significantly (paired
bootstrap, p < 0.05) from that model's, in either direction.

Run from paper/:  python _gen_tables7.py
"""
import json
import os

R = os.path.join("..", "results", "hybrid_attribution")
S = json.load(open(os.path.join(R, "summary.json"), encoding="utf-8"))
cert = json.load(open(os.path.join(R, "certificate_s42.json"), encoding="utf-8"))
agg, paired = S["aggregate"], S["paired"]

LABEL = {"hybrid": r"Hybrid QNN (quantum circuit)",
         "fourier": r"Fourier twin",
         "mlp": r"Tanh MLP twin",
         "linear": r"No middle layer",
         "rf": r"RF, 8 features (published)",
         "rf_f8_sub": r"RF, 8 features, 2{,}000 rows",
         "rf_f4_full": r"RF, 4 features",
         "rf_f4_sub": r"RF, 4 features, 2{,}000 rows"}
PARAMS = {"hybrid": "36", "fourier": "328", "mlp": "40", "linear": "0",
          "rf": "", "rf_f8_sub": "", "rf_f4_full": "", "rf_f4_sub": ""}


def ms(d):
    return f"{d['mean']:.3f} $\\pm$ {d['sd']:.3f}"


def sig_count(other):
    keys = [k for k in paired if k.startswith(f"hybrid_minus_{other}_s") and k.endswith("tpr_at_1pct_fpr")]
    if not keys:
        return ""
    n_sig = sum(1 for k in keys if paired[k]["p"] < 0.05)
    return f"{n_sig}/{len(keys)}"


rows = []
for kind in ("hybrid", "fourier", "mlp", "linear", "rf", "rf_f8_sub", "rf_f4_full", "rf_f4_sub"):
    if kind not in agg:
        continue
    a = agg[kind]
    if kind == "rf":
        rows.append(r"\addlinespace[2pt]")
    rows.append(f"{LABEL[kind]} & {PARAMS[kind]} & {ms(a['roc_auc'])} & "
                f"{ms(a['tpr_at_1pct_fpr'])} & {ms(a['tpr1_seen_types'])} & {ms(a['tpr1_novel_types'])} & "
                f"{'' if kind == 'hybrid' else sig_count(kind)}" + r"\\")
    if kind == "hybrid":
        rows.append(r"\addlinespace[2pt]")

tab = r"""\begin{table}[!htb]
\centering
\caption{Attributing the four-qubit hybrid's low-false-positive result on NSL-KDD. The upper block shares
the hybrid's pipeline (the same four PCA features, seeded 2{,}000-row training subsample, $\mathrm{Linear}(4,4)$
and $\tanh$ input layer, $\mathrm{Linear}(4,2)$ output layer, optimiser, loss, epochs and seeds), and only the
middle layer differs; \emph{Params} counts its trainable parameters. The lower block is the tuned random
forest (RF), as published and on the hybrid's input. Values are mean $\pm$ standard deviation over five
seeds for the upper block and three for the lower, on the full official test set ("""
tab += f"{S['n_test']:,}".replace(",", "{,}") + r""" rows). TPR@1\%FPR is also split into attacks of types present in the training file ("""
tab += f"{S['n_attack_seen']:,}".replace(",", "{,}") + r""" rows) and of the novel types ("""
tab += f"{S['n_attack_novel']:,}".replace(",", "{,}") + r""" rows), at the same threshold. \emph{Sig.} counts the seeds on
which the hybrid's TPR@1\%FPR differs from that model's at $p<0.05$ (paired bootstrap). Generated from the
result files.}
\label{tab:attrib}
\scriptsize
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{2.5pt}
\begin{tabularx}{\textwidth}{@{}Ycccccc@{}}
\toprule
& & & \multicolumn{3}{c}{\textbf{TPR@1\%FPR}} & \\
\cmidrule(lr){4-6}
\textbf{Model} & \textbf{Params} & \textbf{ROC-AUC} & \textbf{All} & \textbf{Seen types} & \textbf{Novel types} & \textbf{Sig.}\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
open("_tab_attrib.tex", "w", encoding="utf-8").write(tab)
print("wrote _tab_attrib.tex with", len(rows), "rows")
print("certificate: basis", cert["n_basis"], "fit err", cert["max_abs_err_fit"], "held-out err",
      cert["max_abs_err_heldout_angles"], "score diff", cert["max_abs_score_diff_on_test_set"])
