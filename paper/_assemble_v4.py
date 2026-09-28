"""Insert the shift decomposition, the three-dataset induced-shift study, the hybrid attribution and the
pooled hardware results into manuscript.tex, and rewrite the framing that depended on the earlier drafts.

Fragments (_v4_*.tex) carry the new prose; generated tables and the new figure replace @@...@@ markers:
  _tab_decomp.tex (tab:decomp), _tab_induced.tex (tab:induced), _fig_induced.tex (fig:induced),
  _tab_attrib.tex (tab:attrib), _tab_hybridhw.tex (tab:hybridhw).
Every anchor must occur exactly once. Idempotent: refuses to run twice.
"""
import re

rd = lambda p: open(p, encoding="utf-8").read().strip()
s = open("manuscript.tex", encoding="utf-8").read()
assert r"\label{sec:kattr}" not in s, "already assembled"

FILL = {"@@TAB_DECOMP@@": "_tab_decomp.tex", "@@TAB_INDUCED@@": "_tab_induced.tex",
        "@@FIG_INDUCED@@": "_fig_induced.tex", "@@TAB_ATTRIB@@": "_tab_attrib.tex",
        "@@TAB_HYBRIDHW@@": "_tab_hybridhw.tex"}


def frag(path):
    t = rd(path)
    for k, v in FILL.items():
        if k in t:
            t = t.replace(k, rd(v))
    assert "@@" not in t, path
    return t


def pat(old):
    return r"\s+".join(re.escape(part) for part in old.split())


def rep(old, new, where):
    """Replace exactly one occurrence; whitespace in `old` matches any whitespace run."""
    global s
    hits = re.findall(pat(old), s)
    assert len(hits) == 1, f"{where}: {len(hits)} matches"
    s = re.sub(pat(old), lambda m: new, s, count=1)


def span(start, end, new, where, keep_end=True, sep="\n\n"):
    """Replace from the (unique) start anchor up to the (first following) end anchor."""
    global s
    a = [m.start() for m in re.finditer(pat(start), s)]
    assert len(a) == 1, f"{where}: start anchor {len(a)} matches"
    m = re.compile(pat(end)).search(s, a[0] + 1)
    assert m, f"{where}: end anchor missing"
    s = s[:a[0]] + new + (sep if keep_end else "") + s[m.start() if keep_end else m.end():]


# ---- abstract ----
i, j = s.index("\\begin{abstract}") + len("\\begin{abstract}"), s.index("\\end{abstract}")
s = s[:i] + "\n" + rd("_v4_abstract.tex") + "\n" + s[j:]

# ---- introduction, last sentence ----
rep("Last, we dissect the audit's strongest positive at the level of the kernels themselves, and show that "
    "it was produced by the tuning of the classical baseline under a distribution shift that standard "
    "cross-validation cannot see, rather than by the quantum state.",
    "Last, we dissect the audit's two surviving positives, the kernel result at the level of the kernels\n"
    "themselves and the hybrid result through classical twins of its quantum layer, and show that both are\n"
    "classical: the first was produced by the tuning of the classical baseline under a distribution shift that\n"
    "standard cross-validation cannot see, the second by the input and the neural pipeline around the quantum\n"
    "layer. Both models reproduce their simulated results on IBM Heron processors.", "intro")

# ---- contributions ----
span("\\item \\textbf{A kernel-level dissection of the audit's strongest positive.}",
     "\\item \\textbf{Operationally meaningful evaluation under noise.}", rd("_v4_contrib.tex"), "contrib")

# ---- methods (Section 4.6) ----
span("\\paragraph{Induced shift.}", "\\paragraph{Geometric difference and model complexity.}",
     rd("_v4_methods_shift.tex"), "methods shift")
span("The same devices also run the variational model behind the audit's one surviving positive,",
     "\\paragraph{Classical phase kernels.}", rd("_v4_methods_hybridhw.tex"), "methods hybrid")

# ---- Section 6 pointers ----
rep("Second, the \\emph{four-qubit} hybrid beats the tuned Random Forest at the low-$\\fpr$ operating point "
    "($\\tpr$@$1\\%\\fpr$ $+0.050$, $p=0.005$, BH $q=0.030$; Section~\\ref{sec:op}).",
    "Second, the \\emph{four-qubit} hybrid beats the tuned Random Forest at the low-$\\fpr$ operating point\n"
    "($\\tpr$@$1\\%\\fpr$ $+0.050$, $p=0.005$, BH $q=0.030$; Section~\\ref{sec:op}); Section~\\ref{sec:kattr}\n"
    "shows that this one, too, is classical.", "audit pointer")
rep("This is the pattern a genuine, if narrow, quantum signal would produce, though it survives FDR and not "
    "Holm correction (Section~\\ref{sec:eff}).",
    "This is the pattern a genuine, if narrow, quantum signal would produce, though it survives FDR and not\n"
    "Holm correction (Section~\\ref{sec:eff}). Section~\\ref{sec:kattr} tests that reading and rejects it:\n"
    "classical twins that replace only the quantum layer reproduce the result.", "op pointer")
rep("Statistical robustness is not attribution, however: Section~\\ref{sec:kernel} shows that the Holm-robust "
    "kernel result is produced by its classical control.",
    "Statistical robustness is not attribution, however: Section~\\ref{sec:kernel} shows that the Holm-robust\n"
    "kernel result is produced by its classical control, and Section~\\ref{sec:kattr} that the FDR-robust hybrid\n"
    "result is reproduced by classical twins of its quantum layer.", "eff pointer")

# ---- Section 7 title and introduction ----
rep("\\section{Kernel-Level Analysis: How Quantum Is the Kernel Advantage?}",
    "\\section{Kernel- and Model-Level Analysis: How Quantum Are the Surviving Positives?}", "sec7 title")
rep("This section removes each difference in turn, examines how the result depends on hyperparameter "
    "selection, tests it on a second official split, and asks whether what remains requires the quantum "
    "state, and finally runs the kernel on real devices.",
    "This section removes each difference in turn, examines how the result depends on hyperparameter\n"
    "selection and on the shift between training and test data, tests it on a second official split and under\n"
    "shift induced on three datasets, and asks whether what remains requires the quantum state. It then asks\n"
    "the same question of the audit's other surviving positive, the four-qubit hybrid, and finally runs both\n"
    "models on real devices.", "sec7 intro")

# ---- 7.2 ----
rep("\\subsection{Cross-validation cannot see the novel-attack shift}",
    "\\subsection{Cross-validation cannot see the train-to-test shift}", "kcv title")
span("Figure~\\ref{fig:landscape} shows why the grid matters so much.",
     "This motivates \\emph{shift-aware model selection}:", frag("_v4_kcv_p1.tex"), "kcv p1")
rep("Its score correlates with test ROC-AUC at $r=+0.79$ for the RBF kernel, and it selects robust settings",
    "Its score correlates with test ROC-AUC at $r=+0.79$ for the RBF kernel ($+0.60$ and $+0.65$ on the two\n"
    "reduced test sets of Table~\\ref{tab:decomp}, but only $+0.11$ on held-out training rows, where there is no\n"
    "shift for it to anticipate), and it selects robust settings", "kcv shift r")
rep("(a) Within the training distribution every kernel scores $0.95$ to $0.99$ at every bandwidth, so "
    "cross-validation cannot tell good settings from bad ones. (b) On the official test set, whose 17 "
    "attack types never occur in training, the raw-feature RBF kernel collapses",
    "(a) Within the training distribution every kernel scores $0.95$ to $0.99$ at every bandwidth, as it does\n"
    "on held-out training rows (Table~\\ref{tab:decomp}): without shift there are no bad settings to avoid.\n"
    "(b) On the official test set, shifted from the training distribution (Section~\\ref{sec:kcv}), the\n"
    "raw-feature RBF kernel collapses", "fig5 caption")

# ---- 7.3 ----
rep("If the collapse of standard CV is caused by novel attack types, it should vanish on a dataset whose test "
    "set has none.",
    "If the collapse of standard CV is caused by the shift, it should vanish on a split whose test set holds\n"
    "no novel attack types and little shift.", "kunsw first")

# ---- 7.4 (replaced whole, with its table and the new figure) ----
span("\\subsection{Inducing the shift}\\label{sec:kinduced}",
     "\\subsection{Geometry: the IQP kernel is close to its own classical phases}", frag("_v4_kinduced.tex"),
     "kinduced")

# ---- new subsection: the hybrid's attribution, before the hardware subsection ----
anchor = "\\subsection{On real hardware}\\label{sec:khw}"
assert s.count(anchor) == 1
s = s.replace(anchor, frag("_v4_kattr.tex") + "\n\n" + anchor)

# ---- hardware: hybrid paragraph and table ----
span("The device runs also settle the status of the audit's one surviving positive.",
     "\\subsection{Summary}\\label{sec:ksum}", frag("_v4_hwhybrid.tex"), "hw hybrid")

# ---- summary ----
span("\\subsection{Summary}\\label{sec:ksum}", "\\section{Discussion}", rd("_v4_summary.tex"), "summary")

# ---- discussion ----
rep("\\textbf{How quantum is the advantage?} On this benchmark and in the practical NISQ regime, almost none "
    "of it.",
    "\\textbf{How quantum is the advantage?} On this benchmark and in the practical NISQ regime, none that\n"
    "survives attribution.", "disc opening")
rep("A QML-IDS claim evaluated on a split with unseen attack types should report how its classical baselines "
    "were tuned, over what range, and by which selection rule, and should report the comparison under more "
    "than one rule with the oracle bound, since no training-only rule, shift-aware selection included, "
    "proved reliable under every shift we induced.",
    "A QML-IDS claim evaluated under distribution shift should report how its classical baselines were tuned,\n"
    "over what range, and by which selection rule; should report the comparison under more than one rule with\n"
    "the oracle bound, since no training-only rule, shift-aware selection included, proved reliable under\n"
    "every shift we induced; and can report a label-free shift statistic between training and test traffic\n"
    "as a warning.", "disc rule")
span("One narrower positive remains from the audit.", "\\textbf{What remains classical.}",
     rd("_v4_disc_hybrid.tex"), "disc hybrid")
rep("Three methodological findings generalise. The $\\approx 0.20$ F1 CV-to-test gap is a caution for the "
    "whole QML-IDS literature; leave-attack-types-out cross-validation addresses the model-selection failure "
    "behind it on NSL-KDD, and the induced-shift experiment shows where such rules stop working; and the "
    "backend crossover",
    "Four methodological findings generalise. The $\\approx 0.20$ F1 CV-to-test gap is a caution for the\n"
    "whole QML-IDS literature; leave-attack-types-out cross-validation addresses the model-selection failure\n"
    "behind it on NSL-KDD, the induced-shift experiment shows where such rules stop working, and a label-free\n"
    "MMD between training and test traffic flags the risk; classical twins that replace only the quantum\n"
    "layer are the control a variational claim needs; and the backend crossover", "disc classical")

# ---- limitations ----
span("\\textbf{Hardware.} The projected quantum kernel", "\\textbf{Qubit budget.}", rd("_v4_limit_hw.tex"),
     "limit hw", keep_end=True, sep="\n")
rep("\\textbf{Kernel analysis.} The kernel-level analysis uses eight qubits, the angle and IQP feature maps, "
    "noiseless statevectors, and the two datasets with official splits; the geometric difference covers all "
    "four datasets, at $N=800$. Shift-aware cross-validation was informative on NSL-KDD but not on UNSW-NB15, "
    "where one attack category dominates, and it requires attack-type labels that production data may not "
    "provide.",
    "\\textbf{Kernel and model analysis.} The kernel-level analysis uses eight qubits, the angle and IQP\n"
    "feature maps, noiseless statevectors, the two datasets with official splits and induced shifts on three\n"
    "datasets, each with 2{,}000 training rows; the geometric difference covers all four datasets, at $N=800$.\n"
    "Shift-aware cross-validation requires attack-type labels that production data may not provide; the MMD\n"
    "statistic is a coarse, correlational signal from 26 runs, not a calibrated test; and the composition\n"
    "reweighting of Table~\\ref{tab:decomp} treats each attack type's test rows as representative of that type.\n"
    "The classical twins attribute the four-qubit, angle-encoded hybrid only: wider or data-re-uploading\n"
    "circuits have larger trigonometric spectra, and the exact expansion grows as $3^n$.", "limit kernel")

# ---- conclusion ----
span("The audit's strongest positive, a quantum kernel that out-ranked its classical surrogate",
     "\\paragraph{Reproducibility.}",
     "Neither of the audit's two positives survived attribution. The quantum kernel that out-ranked its\n"
     "classical surrogate under both false-discovery-rate and family-wise correction was produced by the\n"
     "surrogate's confounds and by the classical bandwidth grid, under a train-to-test shift that standard\n"
     "cross-validation cannot see and that, induced on three further datasets, manufactures quantum leads of\n"
     "the same size; a classical kernel on the circuit's own phases reproduces the quantum kernel on all four\n"
     "datasets. The four-qubit hybrid's advantage at a strict alarm budget belongs to its input and its neural\n"
     "pipeline: its trained circuit is an exact 81-term trigonometric polynomial, and a tanh layer in its place\n"
     "reproduces the result. Both models run on IBM Heron processors with their simulated results intact, so\n"
     "hardware realisability is not attribution either. The lesson reaches beyond this paper: in QML\n"
     "benchmarks evaluated under distribution shift, significance is not attribution, and the tuning of the\n"
     "classical baseline, like the classical twin of the quantum layer, must be reported as carefully as the\n"
     "quantum model itself. We contribute the benchmark and harness, the kernel-level diagnostics, test-set\n"
     "shift controls and an induced-shift protocol with a label-free warning statistic, classical twins for\n"
     "variational models, and a shift-aware selection rule together with its limits, and release them so that\n"
     "QML-IDS claims can be stated, and contested, on common, honest ground.", "conclusion")

open("manuscript.tex", "w", encoding="utf-8").write(s)
print("v4 assembled: decomposition, induced shift on three datasets, hybrid attribution, pooled hardware, framing")
