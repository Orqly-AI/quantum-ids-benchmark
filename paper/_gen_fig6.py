"""Generate Figure 6 (_fig_induced.tex): label-free shift against the reliability of CV selection.

One point per kernel landscape: x is the unbiased squared MMD (RBF, median-heuristic bandwidth)
between the 2,000 training rows and the unlabelled test sample, computable without test labels;
y is the smallest correlation, over the three kernels, between standard-CV and test ROC-AUC across
the 40 grid settings. Filled markers are runs with attack families held out of training, open
markers the same datasets without a holdout; NSL-KDD's official split and its i.i.d. control are
added for reference.

Run from paper/:  python _gen_fig6.py
"""
import json
import os

import numpy as np

R = os.path.join("..", "results", "kernel_analysis")
S = json.load(open(os.path.join(R, "induced_summary.json"), encoding="utf-8"))
FAMS = ("Q IQP projected", "classical trig + IQP phases", "classical RBF raw")


def min_r(d):
    rs = []
    for f in FAMS:
        g = d["families"][f]
        s = np.array([g[k]["std_cv"] for k in g]); t = np.array([g[k]["test"]["roc_auc"] for k in g])
        rs.append(float(np.corrcoef(s, t)[0, 1]))
    return min(rs)


pts = {"unsw": [], "toniot": [], "cicids": []}
base = []
annot = None
for r in S["runs"]:
    xy = (r["shift_mmd2_all"], r["min_r"])
    (pts[r["dataset"]] if r["holdout"] else base).append(xy)
    if r["dataset"] == "cicids" and [h.lower() for h in r["holdout"]] == ["portscan"]:
        annot = xy
nsl_off = (json.load(open(os.path.join(R, "nsl_shift_stats.json")))["landscape.json"]["mmd2_all"],
           min_r(json.load(open(os.path.join(R, "landscape.json")))))
d_iid = json.load(open(os.path.join(R, "landscape_nsl_iid.json")))
nsl_iid = (d_iid["shift_stats"]["mmd2_all"], min_r(d_iid))

coords = lambda ps: " ".join(f"({x:.5f},{y:.4f})" for x, y in ps)
fig = r"""\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[
  width=0.82\textwidth, height=6.4cm,
  xmin=-0.0025, xmax=0.042, ymin=-0.4, ymax=1.08,
  scaled x ticks=false,
  xtick={0,0.01,0.02,0.03,0.04},
  xticklabel style={/pgf/number format/.cd,fixed,fixed zerofill,precision=2},
  ytick={0,0.25,0.5,0.75,1.0},
  yticklabel style={/pgf/number format/.cd,fixed,fixed zerofill,precision=2},
  xlabel={Label-free shift: squared MMD, training sample vs.\ unlabelled test sample},
  ylabel={Smallest CV-to-test correlation $r$},
  tick label style={font=\small}, label style={font=\small},
  ymajorgrids, grid style={gray!18},
  axis line style={gray!70}, tick style={gray!70},
  legend style={at={(0.5,-0.24)}, anchor=north, legend columns=3, draw=none, font=\small,
    /tikz/every even column/.append style={column sep=10pt}},
  every axis plot/.append style={only marks},
]
\addplot[cvviolet, mark=*, mark size=2.6pt] coordinates {""" + coords(pts["unsw"]) + r"""};
\addplot[cvaqua, mark=square*, mark size=2.4pt] coordinates {""" + coords(pts["toniot"]) + r"""};
\addplot[cvorange, mark=triangle*, mark size=3.4pt] coordinates {""" + coords(pts["cicids"]) + r"""};
\addplot[gray, mark=o, mark size=2.6pt, line width=0.9pt] coordinates {""" + coords(base) + r"""};
\addplot[cvblue, mark=diamond*, mark size=3.6pt] coordinates {""" + coords([nsl_off]) + r"""};
\addplot[cvblue, mark=diamond, mark size=3.6pt, line width=0.9pt] coordinates {""" + coords([nsl_iid]) + r"""};
\legend{UNSW-NB15 holdout, ToN-IoT holdout, CICIDS2017 holdout, No holdout, NSL-KDD official split, NSL-KDD i.i.d.\ control}
\node[font=\scriptsize, anchor=west, align=left] at (axis cs:""" + f"{annot[0] + 0.0012:.4f},{annot[1] + 0.02:.3f}" + r""")
  {CICIDS2017, PortScan held out:\\selection creates a $0.17$ quantum lead};
\node[font=\scriptsize, anchor=north] at (axis cs:""" + f"{nsl_off[0]:.4f},{nsl_off[1] - 0.045:.3f}" + r""") {NSL-KDD official};
\end{axis}
\end{tikzpicture}
\caption{Label-free shift predicts when cross-validation stops ranking kernel settings. Each point is one
kernel landscape (three kernels, 40 settings each): attack families held out of the 2{,}000-row training
sample on UNSW-NB15, ToN-IoT and CICIDS2017 (filled), the same datasets without a holdout (open), and
NSL-KDD's official split and its i.i.d.\ control (diamonds). The horizontal axis needs no test labels.
Across the """ + str(S["n_holdout_runs"]) + r""" holdout runs, Spearman $\rho=""" + f"{S['correlations']['shift_mmd2_all~min_r']['spearman']:+.2f}" + r"""$
(permutation $p=""" + f"{S['correlations']['shift_mmd2_all~min_r']['p_perm']:.3f}" + r"""$). Generated from the result files.}
\label{fig:induced}
\end{figure}
"""
open("_fig_induced.tex", "w", encoding="utf-8").write(fig)
print("wrote _fig_induced.tex:", {k: len(v) for k, v in pts.items()}, "baselines", len(base),
      "NSL official", nsl_off, "NSL iid", nsl_iid, "annot", annot)
