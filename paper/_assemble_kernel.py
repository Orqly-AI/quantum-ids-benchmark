"""Assemble the kernel-level section into manuscript.tex.

1. fill the section template with the generated tables and the landscape figure;
2. insert the methods subsection before 'Class imbalance and noise';
3. insert the new section before 'Discussion';
4. revise the Section 6.2 and 6.5 passages whose claims the analysis overturns.
"""
rd = lambda p: open(p, encoding="utf-8").read()

sec = rd("_sec_kernel_template.tex")
for tag, path in (("@@TAB_FAIR@@", "_tab_fair.tex"), ("@@TAB_TWOSPLITS@@", "_tab_twosplits.tex"),
                  ("@@TAB_GEOM@@", "_tab_geom.tex"), ("@@TAB_SHOTS@@", "_tab_shots.tex"),
                  ("@@FIG_LANDSCAPE@@", "_fig_landscape.tex")):
    assert sec.count(tag) == 1, tag
    sec = sec.replace(tag, rd(path).strip())
assert "@@" not in sec

methods = rd("_sec_kdiag_methods.tex").replace(
    "This costs $O(N)$ circuit simulations instead of $O(N^{2})$,",
    "This costs $O(N)$ circuit simulations instead of $O(N^{2})$, a shortcut available only in simulation,")

s = rd("manuscript.tex")
assert r"\label{sec:kernel}" not in s, "kernel section already present"

anchor_m = "\\subsection{Class imbalance and noise}"
assert s.count(anchor_m) == 1
s = s.replace(anchor_m, methods.strip() + "\n\n" + anchor_m)

anchor_d = "\\section{Discussion}"
assert s.count(anchor_d) == 1
s = s.replace(anchor_d, sec.strip() + "\n\n" + anchor_d)

revisions = [
    ("confirmation of the Bellante hypothesis on a full multi-dataset benchmark. Two exceptions survive",
     "confirmation of the Bellante hypothesis on this dataset. Two exceptions survive"),
    ("informative results in the paper. First, and most cleanly, the \\textbf{quantum-kernel SVM significantly\n"
     "out-performs its direct classical analogue}, the random-feature",
     "informative results in the audit. First, the quantum-kernel SVM significantly\n"
     "out-performs the random-feature"),
    ("$p=0.017$, $q=0.047$). Because the random-feature kernel is the explicit classical surrogate for the\n"
     "quantum feature map, this is the audit's sharpest isolation of a genuinely quantum contribution: a\n"
     "comparably-cheap classical kernel does \\emph{not} reproduce the quantum kernel's ranking quality.",
     "$p=0.017$, $q=0.047$). Taken at face value this would isolate a genuinely quantum contribution, since\n"
     "the random-feature kernel is meant as the classical surrogate for the quantum feature map.\n"
     "Section~\\ref{sec:kernel} shows that it does not: the control differed from the QSVM in four further\n"
     "ways, and once those are removed the gap is set by the classical bandwidth search, while a classical\n"
     "kernel on the circuit's own phases reproduces the quantum kernel."),
    ("width rather than a quantum benefit. The corrected picture is therefore precise: classical wins on\n"
     "aggregate, while two specific, FDR-robust quantum advantages persist: the quantum kernel over its\n"
     "classical surrogate, and the small hybrid at the strict operating point.",
     "width rather than a quantum benefit. Within the audit, then, classical wins on aggregate and two\n"
     "FDR-robust quantum differences persist: the quantum kernel over the random-feature control, which\n"
     "Section~\\ref{sec:kernel} traces to the control, and the small hybrid at the strict operating point."),
    ("The two\nquantum advantages that survive FDR control are",
     "The two\nquantum differences that survive FDR control are"),
    ("correction across the full surface, under which only the quantum-kernel-vs-surrogate results remain\n"
     "significant; we therefore treat the quantum-kernel result as the study's most robust positive.",
     "correction across the full surface, under which only the quantum-kernel-vs-surrogate results remain\n"
     "significant. Statistical robustness is not attribution, however: Section~\\ref{sec:kernel} shows that\n"
     "the Holm-robust kernel result is produced by its classical control."),
]
for old, new in revisions:
    n = s.count(old)
    assert n == 1, f"expected 1 match, got {n}: {old[:80]!r}"
    s = s.replace(old, new)

open("manuscript.tex", "w", encoding="utf-8").write(s)
print(f"inserted methods + section; applied {len(revisions)} revisions")
