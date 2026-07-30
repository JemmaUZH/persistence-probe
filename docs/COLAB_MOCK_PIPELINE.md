# Colab Mock Pipeline: T1 + T2 + T3

This runs the CPU-safe pipeline pieces in Colab:

- T1 mock paired episode recorder
- T2 mock model runner
- T3 mock readout metrics and audit

It does not run real MineRL. Real MineRL recording is a separate Linux/Docker
task because headless Minecraft, Java, xvfb, and determinism are the risky part.

## Run

```bash
%cd /content/persistence-probe
!pip install jsonschema Pillow pytest torch torchvision av
```

Generate one paired control/intervene mock episode at `N=8`:

```bash
!python -m recorder.record \
  --mock \
  --scenario break_gold \
  --N 8 \
  --pairs 1 \
  --width 64 \
  --height 64 \
  --K1 4 \
  --K2 6 \
  --seed 123
```

Run T2 mock generations:

```bash
!python -m runner.run \
  --model mock \
  --episodes episodes \
  --protocol teacher_forced \
  --mock \
  --seed 456 \
  --overwrite
```

Run T3 mock readout:

```bash
!python -m readout.run \
  --episodes episodes \
  --results results \
  --model mock \
  --output results/readout \
  --mock
```

Render a contact sheet:

```bash
!python -m recorder.contact_sheet episodes/break_gold_000_N8_control
```

Inspect outputs:

```bash
!find episodes -maxdepth 2 -type f | head -40
!find results -maxdepth 3 -type f | head -40
!cat results/readout/metrics.json
```

Display audit/contact sheet:

```python
from IPython.display import HTML, Image, display
display(Image("/content/persistence-probe/episodes/break_gold_000_N8_control/contact_sheet.png"))
display(HTML(open("/content/persistence-probe/results/readout/audit.html").read()))
```

Run tests:

```bash
!pytest -q
```
