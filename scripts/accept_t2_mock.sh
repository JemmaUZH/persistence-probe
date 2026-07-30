#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

rm -rf episodes/t2_dummy results/t2_dummy
python -m schema.make_dummy episodes/t2_dummy --frames 12 --width 16 --height 16 --seed 456
python -m schema.validate episodes/t2_dummy

python -m runner.run \
  --model mock \
  --episodes episodes/t2_dummy \
  --protocol teacher_forced \
  --mock \
  --seed 789 \
  --overwrite

FIRST_HASH="$(python - <<'PY'
import hashlib
from pathlib import Path

digest = hashlib.sha256()
for path in sorted(Path("results/t2_dummy/mock").glob("gen_*.png")):
    digest.update(path.read_bytes())
print(digest.hexdigest())
PY
)"

python -m runner.run \
  --model mock \
  --episodes episodes/t2_dummy \
  --protocol teacher_forced \
  --mock \
  --seed 789 \
  --overwrite

SECOND_HASH="$(python - <<'PY'
import hashlib
from pathlib import Path

digest = hashlib.sha256()
for path in sorted(Path("results/t2_dummy/mock").glob("gen_*.png")):
    digest.update(path.read_bytes())
print(digest.hexdigest())
PY
)"

if [[ "$FIRST_HASH" != "$SECOND_HASH" ]]; then
  echo "mock runner is not deterministic under fixed seed" >&2
  exit 1
fi

pytest -q
