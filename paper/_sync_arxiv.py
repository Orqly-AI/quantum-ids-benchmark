"""Regenerate a figure-externalised copy of manuscript.tex: identical body, but
the four inline TikZ/pgfplots figure bodies are replaced by
\\includegraphics{FigN} (built from figs.tex by _sync_figs.py).

The rewrite is done one figure environment at a time. Matching a bare
\\centering across the whole file is unsafe -- it can start inside a preceding
table and swallow it, or swallow an already-substituted figure -- so we first
isolate each \\begin{figure}...\\end{figure} block and only edit within it.

Usage:
  python _sync_arxiv.py                                  -> arxiv/main.tex, PDF figures
  python _sync_arxiv.py --ext png --out arxiv/main.tex   -> PNG figures
"""
import os
import re
import sys

ext = "pdf"
out_path = os.path.join("arxiv", "main.tex")
if "--ext" in sys.argv:
    ext = sys.argv[sys.argv.index("--ext") + 1]
if "--out" in sys.argv:
    out_path = sys.argv[sys.argv.index("--out") + 1]

# caption anchor -> (width, figure number); widths mirror the manuscript layout
FIGS = [
    ("Hybrid quantum-classical classifier.", r"\textwidth",      1),
    ("The attribution audit.",               r"0.95\textwidth",  2),
    ("Best classical (same-budget)",         r"\textwidth",      3),
    ("Qubit-count ablation",                 r"0.92\textwidth",  4),
    ("Kernel performance across the RBF bandwidth", r"\textwidth", 5),
]
FIGS = [(anchor, rf"\includegraphics[width={w}]{{Fig{n}.{ext}}}")
        for anchor, w, n in FIGS]

src = open("manuscript.tex", encoding="utf-8").read()
seen = []


def rewrite_figure(match):
    block = match.group(0)
    for anchor, inc in FIGS:
        if "\\caption{" + anchor in block:
            new, n = re.subn(
                r"(\\centering\n).*?(\n\\caption\{)",
                lambda m: m.group(1) + inc + m.group(2),
                block, count=1, flags=re.DOTALL,
            )
            assert n == 1, f"could not rewrite the body of figure {anchor!r}"
            seen.append(anchor)
            return new
    raise AssertionError(f"figure block matched no known anchor:\n{block[:200]}")


out, n_fig = re.subn(r"\\begin\{figure\}.*?\\end\{figure\}", rewrite_figure,
                     src, flags=re.DOTALL)

assert n_fig == 5, f"expected 5 figure environments, found {n_fig}"
assert len(seen) == 5 and len(set(seen)) == 5, f"anchor mismatch: {seen}"
assert out.count("\\includegraphics") == 5, \
    f"expected 5 includegraphics, got {out.count(chr(92) + 'includegraphics')}"
assert "tikzpicture" not in out, "a tikzpicture survived the rewrite"
# the tables must come through untouched
assert out.count("\\begin{tabular}") == src.count("\\begin{tabular}")
assert out.count("\\begin{tabularx}") == src.count("\\begin{tabularx}")
assert "%" not in "\n".join(
    l for l in out.split("\n") if l.lstrip().startswith("%")), "comment leaked"

os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
open(out_path, "w", encoding="utf-8").write(out)
print(f"{out_path} regenerated; figures replaced: {len(seen)} (.{ext})")
