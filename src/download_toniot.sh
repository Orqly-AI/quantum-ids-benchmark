#!/usr/bin/env bash
# Downloads NF-ToN-IoT-v2 (NetFlow-format ToN_IoT, a standard IoT-IDS benchmark)
# into data/nf_toniot.csv. Source: HuggingFace Hmehdi515/NF-ToN-IoT-v2.csv.
#
# The full train.csv is 1.86 GB; we fetch the smaller test split (~535 MB, still
# contains all 10 classes) and let src/data.py do its own stratified split +
# optional row cap (TONIOT_MAX_ROWS, default 200k). Set TONIOT_FILE=train.csv to
# grab the full train split instead.
set -e
DATA_DIR="/mnt/c/Research work 2/quantum-ids/data"
OUT="$DATA_DIR/nf_toniot.csv"
FILE="${TONIOT_FILE:-test.csv}"
BASE="https://huggingface.co/datasets/Hmehdi515/NF-ToN-IoT-v2.csv/resolve/main"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

valid_csv () { # file -> 0 if it has an 'Attack' or 'Label' header (not HTML)
  [ -s "$1" ] || return 1
  head -c 64 "$1" | grep -qi "<html\|<!doctype" && return 1
  head -1 "$1" | grep -qiE "attack|label" || return 1
  return 0
}

if valid_csv "$OUT"; then
  echo "exists: $OUT ($(wc -l < "$OUT") lines)"; exit 0
fi

echo ">> downloading NF-ToN-IoT-v2 ($FILE) -> nf_toniot.csv"
curl -fSL "$BASE/$FILE" -o "$OUT" || {
  echo "FAILED to download $BASE/$FILE"; rm -f "$OUT"
  echo "Manual: grab train/test.csv from https://huggingface.co/datasets/Hmehdi515/NF-ToN-IoT-v2.csv"
  echo "        save as $OUT , then re-run the loader."
  exit 1
}
if ! valid_csv "$OUT"; then
  echo "INVALID download (not a CSV): $OUT"; rm -f "$OUT"; exit 1
fi

echo "=== NF-ToN-IoT-v2 ==="
echo "rows (incl header): $(wc -l < "$OUT")"
head -1 "$OUT" | tr ',' '\n' | head -50
ls -lh "$OUT"
