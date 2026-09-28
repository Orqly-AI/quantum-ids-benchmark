"""Insert the induced-shift experiment and the hybrid-on-hardware results into manuscript.tex.

Reads:
  _sec_induced_methods.tex       methods paragraph -> after the 'Fair surrogates' paragraph (Sec. 4.6)
  _sec_hybridhw_methods.tex      methods text      -> end of the 'Hardware execution' paragraph (Sec. 4.6)
  _sec_induced.tex               new subsection    -> before 'Geometry' in Section 7 (@@TAB_INDUCED@@)
  _tab_induced.tex               table             -> inside that subsection
  _sec_hybridhw_results.tex      paragraph + table -> end of 'On real hardware' (@@TAB_HYBRIDHW@@)
  _tab_hybridhw.tex              table             -> inside that paragraph
  _sec_hardware_limitation2.tex  replacement for the 'Hardware.' limitation paragraph
and rewrites the abstract, contribution, Section 7.2, 7.3, summary and conclusion sentences that
described shift-aware selection as a general remedy.

Idempotent: refuses to run twice.
"""
import re

rd = lambda p: open(p, encoding="utf-8").read().strip()
s = open("manuscript.tex", encoding="utf-8").read()
assert r"\label{sec:kinduced}" not in s, "already inserted"


def rep(old, new, where):
    """Replace exactly one occurrence; whitespace in `old` matches any whitespace run."""
    global s
    pat = r"\s+".join(re.escape(part) for part in old.split())
    hits = re.findall(pat, s)
    assert len(hits) == 1, f"{where}: {len(hits)} matches"
    s = re.sub(pat, lambda m: new, s, count=1)


# ---- methods (Section 4.6) ----
anchor = "stratified cross-validation or the shift-aware variant described in Section~\\ref{sec:kcv}."
rep(anchor, anchor + "\n\n" + rd("_sec_induced_methods.tex"), "induced methods")
anchor = "device built from its calibration data."
rep(anchor, anchor + "\n\n" + rd("_sec_hybridhw_methods.tex"), "hybrid methods")

# ---- Section 7.2: state the fold count of the construction used ----
rep("never saw. It reproduces the condition of the official split using training data alone. Its score "
    "correlates with test ROC-AUC at $r=+0.79$ for the RBF kernel,",
    "never saw. It reproduces the condition of the official split using training data alone; attack types\n"
    "are assigned to folds by GroupKFold, and the three of five folds in which both classes occur are used.\n"
    "Its score correlates with test ROC-AUC at $r=+0.79$ for the RBF kernel,", "kcv fold count")

# ---- Section 7.3: pointer instead of the 'one usable fold' sentence ----
rep("not claim it as a property of quantum kernels. Shift-aware CV is not informative on UNSW-NB15, where the "
    "dominance of a single attack category leaves only one usable fold, and we report no result from it there.",
    "not claim it as a property of quantum kernels. Shift-aware CV on UNSW-NB15, whose official split has no\n"
    "novel attacks for it to anticipate, is examined with the induced-shift experiment in\n"
    "Section~\\ref{sec:kinduced}.", "kunsw pointer")

# ---- new subsection 7.4 with its table ----
res = rd("_sec_induced.tex")
assert res.count("@@TAB_INDUCED@@") == 1
res = res.replace("@@TAB_INDUCED@@", rd("_tab_induced.tex"))
anchor = "\\subsection{Geometry: the IQP kernel is close to its own classical phases}\\label{sec:kgeom}"
assert s.count(anchor) == 1, "geometry anchor"
s = s.replace(anchor, res + "\n\n" + anchor)

# ---- hybrid-on-hardware paragraph + table at the end of 7.6 ----
res = rd("_sec_hybridhw_results.tex")
assert res.count("@@TAB_HYBRIDHW@@") == 1
res = res.replace("@@TAB_HYBRIDHW@@", rd("_tab_hybridhw.tex"))
anchor = "\\subsection{Summary}\\label{sec:ksum}"
assert s.count(anchor) == 1, "summary anchor"
s = s.replace(anchor, res + "\n\n" + anchor)

# ---- summary paragraph ----
rep("Two findings of general use come out of the exercise. Leave-attack-types-out cross-validation restores "
    "reliable model selection under novel-attack shift, and the geometric difference identifies, without "
    "labels, when a quantum kernel is classically reproducible, which here it is from its own circuit phases "
    "on every dataset. The projected kernel itself runs on current hardware: on three IBM Heron processors "
    "its measured Gram matrix correlates at $0.97$ to $0.98$ with simulation and its classifier stays within "
    "$0.01$ ROC-AUC of the exact kernel, so the object under analysis is realisable, and its classical "
    "reproducibility is not an artefact of simulation.",
    "Three findings of general use come out of the\n"
    "exercise. The selection failure can be induced at will by holding attack families out of training, and\n"
    "no training-only rule repairs it in general: leave-attack-types-out cross-validation restores selection\n"
    "on NSL-KDD, where the unseen attacks are relatives of seen ones, but not under the induced shift, so\n"
    "kernel comparisons under shift should be reported under several rules with the oracle bound. The\n"
    "geometric difference identifies, without labels, when a quantum kernel is classically reproducible, which\n"
    "here it is from its own circuit phases on every dataset. And the objects under analysis run on current\n"
    "hardware: on three IBM Heron processors the projected kernel's measured Gram matrix correlates at $0.97$\n"
    "to $0.98$ with simulation and its classifier stays within $0.01$ ROC-AUC of the exact kernel, and the\n"
    "four-qubit hybrid keeps its low-false-positive margin over the Random Forest, so neither the classical\n"
    "reproducibility nor the one surviving positive is an artefact of simulation.", "summary")

# ---- limitations ----
start = s.index("\\textbf{Hardware.}")
end = s.index("\\textbf{Qubit budget.}", start)
s = s[:start] + rd("_sec_hardware_limitation2.tex") + "\n" + s[end:]

# ---- abstract ----
rep("false positives. On UNSW-NB15, which has no unseen attack types, standard selection is reliable and "
    "classical kernels win outright. We introduce leave-attack-types-out cross-validation, which restores "
    "reliable selection under shift, and show with the geometric difference of Huang et al.\\ that the IQP "
    "kernel lies within $1.4$ to $2.4$ of a classical kernel built on the circuit's own phases on all four "
    "datasets, which reproduces it with no quantum state. On three IBM Heron processors the projected kernel's "
    "classifier stays within $0.01$ ROC-AUC of its exact simulation, so the analysed object is realisable on "
    "current hardware. Significance is not attribution. Code, seeds, splits and raw device measurements are "
    "released.",
    "false positives. On UNSW-NB15, which has no unseen attack types, standard selection is reliable and\n"
    "classical kernels win outright, and holding whole attack families out of its training data recreates\n"
    "the failure on demand. Leave-attack-types-out cross-validation repairs selection on NSL-KDD but not\n"
    "under that induced shift, so no training-only rule is a general remedy. The geometric difference of\n"
    "Huang et al.\\ places the IQP kernel within $1.4$ to $2.4$ of a classical kernel on the circuit's own\n"
    "phases on all four datasets. On three IBM Heron processors the projected kernel's classifier stays\n"
    "within $0.01$ ROC-AUC of its exact simulation, and the four-qubit hybrid's one surviving result, higher\n"
    "detection than a tuned Random Forest at $1\\%$ false positives, reproduces on every device. Significance\n"
    "is not attribution. Code, seeds, splits and raw device measurements are released.", "abstract")
rep("quantum kernel significantly out-ranks its classical surrogate. Dissecting that result with exact kernels "
    "on identical data, the gap is set by the classical bandwidth grid:",
    "quantum kernel significantly out-ranks its classical surrogate. With exact kernels on identical data,\n"
    "the gap is set by the classical bandwidth grid:", "abstract dissecting")

# ---- contribution item ----
rep("\\item \\textbf{Shift-aware model selection and a label-free dequantisation test.} Leave-attack-types-out "
    "cross-validation restores reliable model selection under novel-attack shift. The geometric difference of "
    "Huang et al., computed on all four datasets, shows the IQP kernel to be reproducible by a classical kernel "
    "on its own circuit phases, and the projected kernel proves robust to finite-shot measurement and, on three "
    "IBM Heron processors, to real device noise.",
    "\\item \\textbf{An induced-shift test, the limits of shift-aware selection, and a label-free\n"
    "dequantisation test.} Holding whole attack families out of UNSW-NB15's training data reproduces the\n"
    "selection failure on demand. Leave-attack-types-out cross-validation repairs it on NSL-KDD, where the\n"
    "unseen attacks are variants of seen families, but not under the induced shift, so we recommend reporting\n"
    "kernel comparisons under several selection rules with the oracle bound. The geometric difference of\n"
    "Huang et al., computed on all four datasets, shows the IQP kernel to be reproducible by a classical\n"
    "kernel on its own circuit phases; the projected kernel proves robust to finite-shot measurement and, on\n"
    "three IBM Heron processors, to real device noise, and the four-qubit hybrid's low-false-positive result,\n"
    "the audit's one surviving positive, reproduces on the same three devices.", "contribution")

# ---- conclusion ----
rep("how its classical baselines were tuned, over what range, and by which selection rule, and should prefer "
    "shift-aware selection to standard cross-validation.",
    "how its classical baselines were tuned, over what range, and by which selection rule, and should report\n"
    "the comparison under more than one rule with the oracle bound, since no training-only rule, shift-aware\n"
    "selection included, proved reliable under every shift we induced.", "conclusion rule")
rep("We regard the result as a hypothesis for real-hardware testing rather than "
    "evidence of quantum advantage, and we note that it compares a variational model with a tree ensemble, a "
    "comparison the kernel-level controls of Section~\\ref{sec:kernel} do not address.",
    "On three IBM Heron processors it reproduces with the quantum layer's\n"
    "expectations measured rather than simulated (Section~\\ref{sec:khw}), so it is not an artefact of\n"
    "noiseless simulation; but it remains a comparison between a variational model and a tree ensemble on one\n"
    "dataset, a comparison the kernel-level controls of Section~\\ref{sec:kernel} do not address, and we do\n"
    "not regard it as evidence of quantum advantage.", "conclusion hybrid")
rep("leave-attack-types-out cross-validation addresses the model-selection failure behind it; and the backend "
    "crossover",
    "leave-attack-types-out\ncross-validation addresses the model-selection failure behind it on NSL-KDD, and "
    "the induced-shift\nexperiment shows where such rules stop working; and the backend crossover", "conclusion three")
rep("the kernel-level diagnostics, and a shift-aware selection rule, and release them",
    "the kernel-level diagnostics, an induced-shift\ntest, and a shift-aware selection rule together with its "
    "limits, and release them", "conclusion final")

open("manuscript.tex", "w", encoding="utf-8").write(s)
print("induced-shift subsection, hybrid-on-hardware results, methods, limitation and framing inserted")
