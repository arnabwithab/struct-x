#!/usr/bin/env bash
# Fetch only XJTU-DV-Beam.zip (1.7 GB). Rotor/Pump out of scope. See design sec 11.
set -euo pipefail
ZENODO_URL="https://zenodo.org/records/17447083/files/XJTU-DV-Beam.zip?download=1"
EXPECTED_MD5="d3af5b54a227be70fa932d93323d5920"
OUT="${1:-XJTU-DV-Beam.zip}"

if [ -f "$OUT" ]; then
  echo "Resuming $OUT ..."
  curl -C - -L -o "$OUT" "$ZENODO_URL"
else
  echo "Downloading $OUT ..."
  curl -L -o "$OUT" "$ZENODO_URL"
fi

echo "Verifying md5 ..."
ACTUAL=$(md5sum "$OUT" | awk '{print $1}')
if [ "$ACTUAL" != "$EXPECTED_MD5" ]; then
  echo "MD5 MISMATCH: got $ACTUAL want $EXPECTED_MD5" >&2
  echo "Recorded new hash in run log; inspect before proceeding." >&2
  exit 1
fi
echo "md5 OK: $ACTUAL"
echo "Inventory (df + zip listing):"
df -h .
unzip -l "$OUT" | head -n 50
