#!/usr/bin/env bash
# Compile manuscript.tex from clean and report errors, undefined refs, overflows.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qml
cd "$(dirname "$0")"
rm -f manuscript.aux manuscript.bbl
timeout 900 tectonic manuscript.tex --keep-intermediates --reruns 5 > /tmp/mk.log 2>&1
echo "exit=$?"
echo "--- errors ---";            grep -iE '^error|^! ' /tmp/mk.log | head -8
echo "--- undefined refs ---";    grep -ciE 'undefined' /tmp/mk.log
echo "--- overfull (>3pt) ---"
grep -E 'Overfull' /tmp/mk.log | sort -u | awk '{ for (i=1;i<=NF;i++) if ($i ~ /^\(/) { v=$i; gsub(/[(pt]/,"",v); if (v+0 > 3) print; break } }'
echo "--- pages ---";             pdfinfo manuscript.pdf | grep Pages
