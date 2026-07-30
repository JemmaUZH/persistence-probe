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

The `.devcontainer/` config uses Python 3.8, xvfb, ffmpeg, Java 11, and the
X11/Mesa tools needed by LWJGL. MineRL/Malmo still needs Java 8 at runtime; use
the setup script below to install a user-local JDK8.

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

Run the Codespaces setup script:

```bash
bash scripts/setup_minerl_codespaces.sh
```

What it does:

- pins old pip/setuptools/wheel versions that can install `gym==0.19.0`
- installs user-local JDK8 under `$HOME/.jdks/jdk8`
- patches MineRL's missing `MixinGradle:dcfaf61` dependency via a local Maven repo
- starts a local asset proxy because old ForgeGradle uses disabled HTTP
  Minecraft asset URLs
- builds MineRL 0.4.4 and writes `.minerl-codespaces-env`

## 4. Import Probe

```bash
source .minerl-codespaces-env
python - <<'PY'
import gym
import minerl
print("gym", gym.__version__)
print("minerl", getattr(minerl, "__version__", "unknown"))
PY
```

## 5. MineRL Reset/Step Smoke

```bash
xvfb-run -a -s "-screen 0 1280x720x24 +extension RANDR +extension GLX +render" \
  python scripts/minerl_smoke.py \
  --env MineRLBasaltFindCave-v0 \
  --steps 3 \
  --output scratch/minerl_smoke
```

Success means:

- environment resets
- three no-op steps run
- PNG frames appear in `scratch/minerl_smoke/`

Check:

```bash
ls -lh scratch/minerl_smoke
```

## 6. Decision Gate

The current real recorder acceptance command is:

```bash
bash scripts/accept_t1_real_codespaces.sh episodes
```

This records two paired `N=16` MineRL episodes with the stable
`MineRLBasaltFindCave-v0` backend, validates all four episode directories, and
renders contact sheets. Set `T1_ENV_ID=PersistenceProbeBreakGold-v0` only when
debugging the custom gold-block Malmo mission; that custom mission is not yet
the accepted backend.

If the smoke passes, the next implementation task is the final gold-block T1 backend:

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
