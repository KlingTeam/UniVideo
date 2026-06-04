#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DITTO_EXAMPLES="${DITTO_EXAMPLES:-100}"
OPENS2V_LIMIT="${OPENS2V_LIMIT:-30}"
PICO_LIMIT="${PICO_LIMIT:-32}"
PICO_SCAN_LIMIT="${PICO_SCAN_LIMIT:-512}"
KIWI_LIMIT="${KIWI_LIMIT:-32}"
DOWNLOAD_DITTO_ARCHIVES="${DOWNLOAD_DITTO_ARCHIVES:-1}"

echo "[1/5] Preparing Ditto JSONL (${DITTO_EXAMPLES} examples)"
python scripts/prepare_ditto_subset.py \
  --metadata global_style \
  --num-examples "${DITTO_EXAMPLES}" \
  --allow-missing-videos \
  --output data/ditto_100_v2v.jsonl

echo "[2/5] Extracting Ditto videos"
ditto_args=(
  --jsonl data/ditto_100_v2v.jsonl
)
if [[ "${DOWNLOAD_DITTO_ARCHIVES}" == "1" ]]; then
  ditto_args+=(--download-archives)
fi
python scripts/extract_ditto_subset_videos.py "${ditto_args[@]}"

echo "[3/5] Preparing OpenS2V multi-reference data"
python scripts/prepare_opens2v_multiid_smoke.py \
  --limit "${OPENS2V_LIMIT}"

echo "[4/5] Preparing Pico-Banana image-edit data"
python scripts/prepare_pico_banana_i2i_smoke.py \
  --limit "${PICO_LIMIT}" \
  --scan-limit "${PICO_SCAN_LIMIT}"

echo "[5/5] Preparing Kiwi/RefVIE reference-image + v2v data"
if ! python - <<'PY'
import importlib.util
import sys
from pathlib import Path

if importlib.util.find_spec("pyarrow") is not None:
    raise SystemExit(0)
target = Path(".deps/pyarrow")
if target.exists():
    sys.path.insert(0, str(target))
    if importlib.util.find_spec("pyarrow") is not None:
        raise SystemExit(0)
raise SystemExit(1)
PY
then
  echo "pyarrow is required for the Kiwi/RefVIE parquet smoke set."
  echo "Install it with:"
  echo "  python -m pip install --target .deps/pyarrow pyarrow"
  exit 1
fi

python scripts/prepare_kiwi_refvie_i_v2v_smoke.py \
  --limit "${KIWI_LIMIT}" \
  --download-parquet \
  --allow-fallback-videos

echo "Smoke data ready:"
wc -l \
  data/ditto_100_v2v.jsonl \
  data/opens2v_multiid_smoke.jsonl \
  data/pico_banana_i2i_smoke.jsonl \
  data/kiwi_refvie_i_v2v_smoke.jsonl
