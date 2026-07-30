#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

rm -rf episodes/t0_dummy episodes/t0_corrupt
python -m schema.make_dummy episodes/t0_dummy --frames 8 --width 16 --height 16 --seed 123
python -m schema.validate episodes/t0_dummy

cp -R episodes/t0_dummy episodes/t0_corrupt
python - <<'PY'
import json
from pathlib import Path

path = Path("episodes/t0_corrupt/meta.json")
meta = json.loads(path.read_text())
del meta["model_target"]
path.write_text(json.dumps(meta))
PY

if python -m schema.validate episodes/t0_corrupt; then
  echo "expected corrupted episode validation to fail" >&2
  exit 1
fi

pytest -q
