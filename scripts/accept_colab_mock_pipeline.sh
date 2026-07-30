#!/usr/bin/env bash
set -euo pipefail

python -m recorder.record --mock --scenario break_gold --N 8 --pairs 1 --width 64 --height 64 --K1 4 --K2 6 --seed 123
python -m runner.run --model mock --episodes episodes --protocol teacher_forced --mock --seed 456 --overwrite
python -m readout.run --episodes episodes --results results --model mock --output results/readout --mock
python -m recorder.contact_sheet episodes/break_gold_000_N8_control
pytest -q
