# Colab Oasis Smoke Run

Use Colab only for real Oasis inference. The local Mac path should stay on `--mock`
because the official `open-oasis/generate.py` requires CUDA.

## 1. Runtime

In Colab, choose a GPU runtime:

`Runtime` -> `Change runtime type` -> `T4` or better.

Check it:

```bash
!nvidia-smi
```

## 2. Get The Project

Upload or clone `persistence-probe` into Colab. If using Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then set:

```bash
%cd /content/persistence-probe
```

## 3. Install Dependencies

```bash
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install einops diffusers timm av safetensors jsonschema Pillow pytest
```

## 4. Get open-oasis And Weights

```bash
!mkdir -p external
![ -d external/open-oasis ] || git clone --depth 1 https://github.com/etched-ai/open-oasis.git external/open-oasis
```

The model is gated on Hugging Face. Log in with a fresh token that has access to
`Etched/oasis-500m`:

```bash
!hf auth login
```

Download:

```bash
%cd /content/persistence-probe/external/open-oasis
!hf download Etched/oasis-500m oasis500m.safetensors --local-dir .
!hf download Etched/oasis-500m vit-l-20.safetensors --local-dir .
%cd /content/persistence-probe
```

## 5. Prepare One Episode For Oasis

For a smoke test with the dummy episode:

```bash
!python -m schema.make_dummy episodes/colab_dummy --frames 12 --width 640 --height 360 --seed 123
!python -m schema.validate episodes/colab_dummy
!python -m runner.oasis_io prepare episodes/colab_dummy /content/oasis_inputs/colab_dummy --overwrite
```

For real MineRL episodes, replace `episodes/colab_dummy` with the target episode
directory.

## 6. Run open-oasis

```bash
%cd /content/persistence-probe/external/open-oasis
!python generate.py \
  --oasis-ckpt oasis500m.safetensors \
  --vae-ckpt vit-l-20.safetensors \
  --prompt-path /content/oasis_inputs/colab_dummy/prompt.mp4 \
  --actions-path /content/oasis_inputs/colab_dummy/actions.one_hot_actions.pt \
  --n-prompt-frames 8 \
  --num-frames 12 \
  --fps 20 \
  --ddim-steps 10 \
  --output-path /content/oasis_inputs/colab_dummy/oasis.mp4
%cd /content/persistence-probe
```

Use the `n_prompt_frames` and `total_frames` printed by `runner.oasis_io` for
non-dummy episodes.

## 7. Bring Output Back

The smoke output video is:

```text
/content/oasis_inputs/colab_dummy/oasis.mp4
```

For pipeline integration, the next task is to add automatic extraction of only
the generated return frames into `results/{episode}/oasis/gen_%06d.png`:

```bash
%cd /content/persistence-probe
!python -m runner.oasis_io import-video \
  /content/oasis_inputs/colab_dummy/oasis.mp4 \
  /content/oasis_inputs/colab_dummy/manifest.json \
  results/colab_dummy/oasis \
  --overwrite
!ls -lh results/colab_dummy/oasis
```
