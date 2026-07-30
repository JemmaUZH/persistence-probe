#!/usr/bin/env bash
set -euo pipefail

source .minerl-codespaces-env

OUT_ROOT="${1:-episodes}"
ENV_ID="${T1_ENV_ID:-MineRLBasaltFindCave-v0}"
WORKER_RETRIES="${T1_WORKER_RETRIES:-4}"

xvfb-run -a -s "-screen 0 1280x720x24 +extension RANDR +extension GLX +render" \
  python -m recorder.record \
    --scenario break_gold \
    --N 16 \
    --pairs 2 \
    --output-root "$OUT_ROOT" \
    --env-id "$ENV_ID" \
    --worker-retries "$WORKER_RETRIES"

for episode in "$OUT_ROOT"/break_gold_*_N16_*; do
  python -m schema.validate "$episode"
  python -m recorder.contact_sheet "$episode"
done

python - "$OUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
episodes = sorted(root.glob("break_gold_*_N16_*"))
if len(episodes) != 4:
    raise SystemExit(f"expected 4 N=16 episodes under {root}, found {len(episodes)}")
for episode in episodes:
    frames = sorted((episode / "frames").glob("*.png"))
    events = [json.loads(line) for line in (episode / "events.jsonl").read_text().splitlines()]
    print(f"{episode}: {len(frames)} frames, events={events}")
PY
