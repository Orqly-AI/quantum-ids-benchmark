#!/usr/bin/env bash
# Downloads the CICIDS2017 "MachineLearningCVE" day CSVs (the per-flow feature
# CSVs: 78 features + Label) and concatenates them into data/cicids2017.csv
# (header kept once). Source: HuggingFace dataset c01dsnap/CIC-IDS2017 (the 8
# canonical *.pcap_ISCX.csv files, uncompressed -> no unzip needed).
#
# NOTE: ~885 MB total. Set CICIDS_DAYS to a space-separated subset of basenames
# to fetch fewer days (e.g. for a quick pipeline check):
#   CICIDS_DAYS="Monday-WorkingHours Friday-WorkingHours-Afternoon-PortScan" bash download_cicids.sh
set -e
DATA_DIR="/mnt/c/Research work 2/quantum-ids/data"
WORK="$DATA_DIR/_cicids_raw"
OUT="$DATA_DIR/cicids2017.csv"
BASE="https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main"
mkdir -p "$WORK"
cd "$WORK"

if [ -s "$OUT" ]; then
  echo "exists: $OUT ($(wc -l < "$OUT") lines)"; exit 0
fi

ALL_DAYS=(
  "Monday-WorkingHours"
  "Tuesday-WorkingHours"
  "Wednesday-workingHours"
  "Thursday-WorkingHours-Morning-WebAttacks"
  "Thursday-WorkingHours-Afternoon-Infilteration"
  "Friday-WorkingHours-Morning"
  "Friday-WorkingHours-Afternoon-PortScan"
  "Friday-WorkingHours-Afternoon-DDos"
)
# Allow overriding with a subset via env var.
if [ -n "${CICIDS_DAYS:-}" ]; then
  read -r -a DAYS <<< "$CICIDS_DAYS"
else
  DAYS=("${ALL_DAYS[@]}")
fi

valid_csv () { # file -> 0 if it looks like a CICIDS flow CSV (not HTML/error)
  [ -s "$1" ] || return 1
  head -c 64 "$1" | grep -qi "<html\|<!doctype" && return 1   # reject HTML pages
  head -1 "$1" | grep -qi "Label" || return 1                  # must have a Label header
  return 0
}

fetch_day () { # basename
  local name="$1" f="${1}.pcap_ISCX.csv"
  if valid_csv "$f"; then echo "ok (cached): $f ($(wc -l < "$f") lines)"; return 0; fi
  echo ">> downloading $f"
  curl -fSL "$BASE/$f" -o "$f" || { echo "FAILED: $f"; rm -f "$f"; return 1; }
  if valid_csv "$f"; then echo "ok: $f ($(wc -l < "$f") lines)"; return 0; fi
  echo "INVALID (not a CSV): $f"; rm -f "$f"; return 1
}

ok=0
for d in "${DAYS[@]}"; do
  if fetch_day "$d"; then ok=$((ok+1)); fi
done
if [ "$ok" -eq 0 ]; then
  echo "No CICIDS day CSVs downloaded successfully."
  echo "Manual: get the MachineLearningCVE CSVs (https://www.unb.ca/cic/datasets/ids-2017.html"
  echo "        or https://huggingface.co/datasets/c01dsnap/CIC-IDS2017), put *.pcap_ISCX.csv"
  echo "        into $WORK/ , then re-run."
  exit 1
fi

# Concatenate valid day CSVs, keeping one header. Strip CR if present.
mapfile -t CSVS < <(find "$WORK" -maxdepth 1 -iname "*.pcap_ISCX.csv" | sort)
echo "concatenating ${#CSVS[@]} CSV files -> $OUT"
first=1
: > "$OUT"
for f in "${CSVS[@]}"; do
  valid_csv "$f" || { echo "skip invalid: $f"; continue; }
  if [ "$first" -eq 1 ]; then
    tr -d '\r' < "$f" >> "$OUT"; first=0
  else
    tail -n +2 "$f" | tr -d '\r' >> "$OUT"
  fi
done

echo "=== CICIDS2017 ==="
echo "rows (incl header): $(wc -l < "$OUT")"
ls -lh "$OUT"
