#!/usr/bin/env bash
# Rasterise the four vector figures to PNG for the arXiv build.
# 600 dpi keeps line art crisp; PDF remains the master for the journal package.
set -eu
cd "$(dirname "$0")"
DPI=600
for n in 1 2 3 4 5; do
  pdftoppm -png -r "$DPI" -singlefile "arxiv/Fig${n}.pdf" "arxiv/Fig${n}"
  px=$(python3 -c "import struct,sys;d=open('arxiv/Fig${n}.png','rb').read();print(struct.unpack('>II',d[16:24])[0])")
  kb=$(( $(wc -c < "arxiv/Fig${n}.png") / 1024 ))
  echo "Fig${n}.png  ${px}px wide  ${kb} KB"
done
