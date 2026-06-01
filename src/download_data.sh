#!/usr/bin/env bash
# Downloads NSL-KDD (train/test) into the project data/ dir. Tries mirrors with fallback.
set -e
DATA_DIR="/mnt/c/Research work 2/quantum-ids/data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

fetch () { # url outfile
  if [ -s "$2" ]; then echo "exists: $2"; return 0; fi
  echo ">> downloading $2"
  curl -fsSL "$1" -o "$2" && [ -s "$2" ] && echo "ok: $2 ($(wc -l < "$2") lines)" || { echo "FAILED: $1"; rm -f "$2"; return 1; }
}

# Primary mirror (widely used NSL-KDD copy)
fetch "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt" "KDDTrain+.txt" \
  || fetch "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt" "KDDTrain+.txt" \
  || fetch "https://raw.githubusercontent.com/HoaNP/NSL-KDD-DataSet/master/KDDTrain%2B.txt" "KDDTrain+.txt"

fetch "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt" "KDDTest+.txt" \
  || fetch "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt" "KDDTest+.txt" \
  || fetch "https://raw.githubusercontent.com/HoaNP/NSL-KDD-DataSet/master/KDDTest%2B.txt" "KDDTest+.txt"

echo "=== data dir ==="
ls -lh "$DATA_DIR"
