#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-scratch/t3_real_template}"
EPISODES="$ROOT/episodes"
RESULTS="$ROOT/results"
MODEL="identity"

rm -rf "$ROOT"

python -m recorder.record \
  --mock \
  --scenario break_gold \
  --N 8 \
  --pairs 1 \
  --output-root "$EPISODES" \
  --width 96 \
  --height 72 \
  --K1 4 \
  --K2 8 \
  --seed 123

python - "$EPISODES" "$RESULTS" "$MODEL" <<'PY'
import json
import shutil
import sys
from pathlib import Path

episodes_root = Path(sys.argv[1])
results_root = Path(sys.argv[2])
model = sys.argv[3]

for episode_dir in sorted(episodes_root.glob("break_gold_*")):
    events = [json.loads(line) for line in (episode_dir / "events.jsonl").read_text().splitlines()]
    return_start = next(row["frame_idx"] for row in events if row["event"] == "return_start")
    output_dir = results_root / episode_dir.name / model
    output_dir.mkdir(parents=True)
    for out_idx, frame_path in enumerate(sorted((episode_dir / "frames").glob("*.png"))[return_start:]):
        shutil.copyfile(frame_path, output_dir / f"gen_{out_idx:06d}.png")
PY

python -m readout.run \
  --episodes "$EPISODES" \
  --results "$RESULTS" \
  --model "$MODEL" \
  --output "$RESULTS/readout"

python - "$RESULTS/readout/metrics.json" <<'PY'
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text())
if metrics["mock_readout"]:
    raise SystemExit("expected real template readout, got mock readout")
if not metrics["calibrations"]:
    raise SystemExit("expected real-frame calibration metadata")
for calibration in metrics["calibrations"]:
    if calibration["real_frame_accuracy"] < 0.95:
        raise SystemExit(f"calibration below acceptance threshold: {calibration}")
for row in metrics["per_episode"]:
    if row["correct_state"] != 1:
        raise SystemExit(f"identity readout should be correct: {row}")
print(f"real template readout accepted: {sys.argv[1]}")
PY
