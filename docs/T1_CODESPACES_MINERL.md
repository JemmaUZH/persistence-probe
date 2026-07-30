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
export JAVA_HOME=/home/vscode/.jdks/jdk8
export PATH="$JAVA_HOME/bin:$HOME/.local/bin:$PATH"
export PYTHONPATH=/tmp/minerl_build_proxy_1785423349/minerl-0.4.4/build/lib.linux-x86_64-3.8:$PYTHONPATH
export LIBGL_ALWAYS_SOFTWARE=1
export JAVA_TOOL_OPTIONS="-Dorg.lwjgl.opengl.Display.allowSoftwareOpenGL=true"
T1_ENV_ID=PersistenceProbeBreakGold-v0 T1_WORKER_RETRIES=2 T1_WORKER_TIMEOUT=420 \
  bash scripts/accept_t1_real_codespaces.sh episodes
```

This records two paired `N=16` MineRL episodes with the custom
`PersistenceProbeBreakGold-v0` gold-block backend, validates all four episode
directories, and renders contact sheets. To run the older stable BASALT plumbing
backend instead, leave `T1_ENV_ID` unset.

The T1 backend now covers:

```text
- fixed seed/spawn where MineRL supports it
- scripted control/intervene action sequence
- save frames/actions/state/events per SPEC.md
- assert paired action equality outside intervention frame
- render contact sheet
```

If this command fails after one focused retry, inspect
`episodes/_worker_logs/*.log` and the newest `logs/mc_*.log`.
