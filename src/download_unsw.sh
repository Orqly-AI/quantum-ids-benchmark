#!/usr/bin/env bash
# Downloads the canonical UNSW-NB15 train/test CSVs into the project data/ dir.
# Canonical split: 175,341 train rows / 82,332 test rows. Tries mirrors w/ fallback.
set -e
DATA_DIR="/mnt/c/Research work 2/quantum-ids/data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

fetch () { # url outfile
  if [ -s "$2" ]; then echo "exists: $2 ($(wc -l < "$2") lines)"; return 0; fi
  echo ">> downloading $2"
  curl -fsSL "$1" -o "$2" && [ -s "$2" ] && echo "ok: $2 ($(wc -l < "$2") lines)" \
    || { echo "FAILED: $1"; rm -f "$2"; return 1; }
}

# Training set
fetch "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv" "UNSW_NB15_training-set.csv" \
  || fetch "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv" "UNSW_NB15_training-set.csv"

# Testing set
fetch "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv" "UNSW_NB15_testing-set.csv" \
  || fetch "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_testing-set.csv" "UNSW_NB15_testing-set.csv"

echo "=== UNSW-NB15 files ==="
ls -lh "$DATA_DIR"/UNSW_NB15_*.csv
