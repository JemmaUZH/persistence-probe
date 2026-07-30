# T1 MineRL In GitHub Codespaces

Codespaces is for the T1 environment probe and real MineRL recorder work. Colab
should remain the Oasis GPU path.

## 0. Push Or Upload This Repo

Codespaces starts from a GitHub repository. Create a GitHub repo for
`persistence-probe`, then upload or push this local folder.

Do not commit:

- `external/`
- `episodes/`
- `results/`
- `.safetensors` / `.pt` weights

Those are already covered by `.gitignore`.

## 1. Open Codespaces

On GitHub:

1. Open the `persistence-probe` repo.
2. Click `Code`.
3. Click `Codespaces`.
4. Create a new codespace on the main branch.

The `.devcontainer/` config uses Python 3.8, xvfb, ffmpeg, and Java 11. MineRL
official docs prefer Java 8, but Java 8 packages are awkward on current base
images. First try Java 11; if MineRL/Malmo rejects it, switch to a custom Java 8
base image.

## 2. Verify Base Environment

In the Codespaces terminal:

```bash
python --version
java -version
which xvfb-run
python -m pip --version
```

Expected shape:

```text
Python 3.8.x
openjdk version "11..."
/usr/bin/xvfb-run
```

## 3. Install MineRL

Do this manually instead of in the devcontainer build so failures are visible:

```bash
python -m pip install --upgrade "pip<24" "setuptools<60" wheel
python -m pip install "gym==0.19.0"
python -m pip install "minerl==0.4.4"
```

If `gym==0.19.0` fails, stop and paste the full error back into Codex.

If MineRL fails with Java/Malmo errors, stop and paste the full error. Do not
start changing recorder logic until `import minerl` works.

## 4. Import Probe

```bash
python - <<'PY'
import gym
import minerl
print("gym", gym.__version__)
print("minerl", getattr(minerl, "__version__", "unknown"))
PY
```

## 5. MineRL Reset/Step Smoke

```bash
xvfb-run -a python scripts/minerl_smoke.py \
  --env MineRLBasaltFindCave-v0 \
  --steps 5 \
  --output scratch/minerl_smoke
```

Success means:

- environment resets
- five no-op steps run
- PNG frames appear in `scratch/minerl_smoke/`

Check:

```bash
ls -lh scratch/minerl_smoke
```

## 6. Decision Gate

If the smoke passes, the next implementation task is the real T1 backend:

```text
Implement recorder.record real MineRL backend:
- fixed seed/spawn where MineRL supports it
- scripted control/intervene action sequence
- save frames/actions/state/events per SPEC.md
- assert paired action equality outside intervention frame
- render contact sheet
```

If the smoke fails after one focused attempt, use a Linux VM with a more
controlled Docker image. Do not spend days debugging Codespaces/Malmo blindly.
