#!/usr/bin/env bash
# Compile the PDF (keeping the .bbl) and assemble an arXiv-ready source zip.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qml
cd "/mnt/c/Research work 2/quantum-ids/paper"

echo ">> compiling (keep intermediates for .bbl)"
tectonic manuscript.tex --keep-intermediates --reruns 5 > /tmp/tex.log 2>&1 || true
echo "   undefined-citation count: $(grep -ci 'Citation.*undefined' /tmp/tex.log || echo 0)"
ls -lh manuscript.pdf manuscript.bbl 2>/dev/null

echo ">> assembling arxiv.zip"
rm -rf arxiv_build arxiv.zip
mkdir -p arxiv_build/figures
cp manuscript.tex references.bib arxiv_build/
[ -f manuscript.bbl ] && cp manuscript.bbl arxiv_build/
# only the PDF figures the .tex actually \includegraphics
for f in bar_nslkdd_f1 bar_unsw_f1 roc_nslkdd ablation_qubits_nslkdd_f1; do
  cp "../figures/$f.pdf" arxiv_build/figures/ 2>/dev/null || true
done
# arXiv autodetects the main .tex; add a minimal 00README for clarity
cat > arxiv_build/00README.XXX <<'EOF'
Main document: manuscript.tex  (compile with pdflatex + bibtex, or the included .bbl)
Figures in figures/.  TikZ diagrams are inline in manuscript.tex.
EOF
( cd arxiv_build && zip -qr ../arxiv.zip . )
echo "   arxiv.zip:"; unzip -l arxiv.zip | tail -n +4
