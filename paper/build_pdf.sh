#!/usr/bin/env bash
# Build the manuscript PDF and an arXiv-ready source zip from manuscript.md.
# Requires pandoc + tectonic (installed in the qml conda env). Run:
#   bash paper/build_pdf.sh
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qml
cd "/mnt/c/Research work 2/quantum-ids/paper"

TITLE="How Quantum Is the Advantage? A Fair, Calibration- and Noise-Aware Benchmark and Attribution Audit of Quantum Machine Learning for Network Intrusion Detection"

echo ">> 1/4  Markdown -> LaTeX (preserving \\cite, math, tables)"
# +raw_tex keeps the inline \cite{...}; we strip the manual author block lines so
# pandoc's title block (set via metadata) is the single source of the heading.
pandoc manuscript.md \
  -f markdown+raw_tex+tex_math_dollars \
  -t latex -s \
  -V documentclass=article \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V linkcolor=blue -V urlcolor=blue \
  -M title="$TITLE" \
  -M author="Syeda Anshrah Gillani (Hamdard University, Karachi, Pakistan); Mirza Samad Ahmed Baig (Fandaqah, Al Khobar, Saudi Arabia)" \
  -M date="2026" \
  -o manuscript.tex

# Ensure a bibliography is emitted at the end (natbib/bibtex over references.bib).
if ! grep -q "bibliography{references}" manuscript.tex; then
  # insert before \end{document}
  sed -i 's:\\end{document}:\\bibliographystyle{unsrt}\n\\bibliography{references}\n\\end{document}:' manuscript.tex
fi

echo ">> 2/4  Compile PDF with tectonic (auto-runs bibtex)"
tectonic manuscript.tex --keep-intermediates --reruns 3 >/dev/null 2>&1 || tectonic manuscript.tex >/dev/null 2>&1 || true
if [ -f manuscript.pdf ]; then echo "   PDF OK: $(du -h manuscript.pdf | cut -f1)"; else echo "   WARN: PDF not produced (see tectonic output)"; fi

echo ">> 3/4  Assemble arXiv source zip"
rm -rf arxiv_build arxiv.zip
mkdir -p arxiv_build/figures
cp manuscript.tex references.bib arxiv_build/
# arXiv compiles LaTeX; include the PDF figures the .tex references.
cp ../figures/*.pdf arxiv_build/figures/ 2>/dev/null || true
# arXiv needs the .bbl when it can't run bibtex; include it if tectonic made one.
[ -f manuscript.bbl ] && cp manuscript.bbl arxiv_build/ || true
( cd arxiv_build && zip -qr ../arxiv.zip . )
echo "   arxiv.zip contents:"; unzip -l arxiv.zip | tail -n +4 | head -40

echo ">> 4/4  Done"
echo "PDF:       quantum-ids/paper/manuscript.pdf"
echo "arXiv zip: quantum-ids/paper/arxiv.zip"
