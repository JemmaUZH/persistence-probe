from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from runner.run import RunnerError, _teacher_forced_batch
from schema.validate import ValidationFailure, validate_episode


ACTION_KEYS = [
    "inventory",
    "ESC",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "forward",
    "back",
    "left",
    "right",
    "cameraX",
    "cameraY",
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
]


@dataclass(frozen=True)
class OasisPreparedInputs:
    prompt_path: Path
    actions_path: Path
    manifest_path: Path
    n_prompt_frames: int
    total_frames: int
    generated_frames: int


@dataclass(frozen=True)
class OasisImportedVideo:
    output_dir: Path
    manifest_path: Path
    generated_frames: int


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def encode_oasis_actions(actions: list[dict[str, Any]]) -> torch.Tensor:
    encoded = torch.zeros((len(actions), len(ACTION_KEYS)), dtype=torch.float32)
    for i, action in enumerate(actions):
        for j, key in enumerate(ACTION_KEYS):
            if key == "cameraX":
                value = action.get("camera", [40, 40])[0]
                value = (float(value) - 40) / 40
                if not -1 - 1e-3 <= value <= 1 + 1e-3:
                    raise RunnerError(f"cameraX normalized value outside [-1, 1]: {value}")
            elif key == "cameraY":
                value = action.get("camera", [40, 40])[1]
                value = (float(value) - 40) / 40
                if not -1 - 1e-3 <= value <= 1 + 1e-3:
                    raise RunnerError(f"cameraY normalized value outside [-1, 1]: {value}")
            else:
                value = float(action.get(key, 0))
                if not 0 <= value <= 1:
                    raise RunnerError(f"{key} value outside [0, 1]: {value}")
            encoded[i, j] = value
    return encoded


def _frames_to_video(frame_paths: list[Path], output_path: Path, fps: int) -> None:
    try:
        import av
    except ImportError as exc:
        raise RunnerError("PyAV is required to write prompt.mp4; install with `pip install av`") from exc

    with Image.open(frame_paths[0]) as first_img:
        width, height = first_img.size

    container = av.open(str(output_path), mode="w")
    try:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        for path in frame_paths:
            with Image.open(path) as img:
                frame = av.VideoFrame.from_image(img.convert("RGB"))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _read_video_frames(video_path: Path) -> list[Image.Image]:
    try:
        import av
    except ImportError as exc:
        raise RunnerError("PyAV is required to read Oasis output video; install with `pip install av`") from exc

    if not video_path.exists():
        raise RunnerError(f"Oasis output video does not exist: {video_path}")

    frames: list[Image.Image] = []
    container = av.open(str(video_path))
    try:
        for frame in container.decode(video=0):
            frames.append(frame.to_image().convert("RGB"))
    finally:
        container.close()

    if not frames:
        raise RunnerError(f"no frames decoded from Oasis output video: {video_path}")
    return frames


def prepare_oasis_inputs(
    episode_dir: str | Path,
    output_dir: str | Path,
    *,
    ctx_limit: int | None = None,
    fps: int = 20,
    overwrite: bool = False,
) -> OasisPreparedInputs:
    episode_path = Path(episode_dir)
    output_path = Path(output_dir)
    validate_episode(episode_path)
    batch = _teacher_forced_batch(episode_path, ctx_limit)

    if output_path.exists():
        if overwrite:
            shutil.rmtree(output_path)
        else:
            raise RunnerError(f"output directory already exists: {output_path}")
    output_path.mkdir(parents=True)

    all_actions = batch.context_actions + batch.future_actions
    if len(all_actions) < 2:
        raise RunnerError("Oasis action export requires at least two total frames")

    prompt_path = output_path / "prompt.mp4"
    actions_path = output_path / "actions.one_hot_actions.pt"
    manifest_path = output_path / "manifest.json"

    _frames_to_video(batch.context_frames, prompt_path, fps=fps)
    # open-oasis load_actions prepends a zero row, so save frame actions 1..T-1.
    torch.save(encode_oasis_actions(all_actions[1:]), actions_path)

    manifest = {
        "episode": episode_path.name,
        "protocol": "teacher_forced",
        "return_start": batch.return_start,
        "ctx_limit": ctx_limit,
        "n_prompt_frames": len(batch.context_frames),
        "total_frames": len(all_actions),
        "generated_frames": len(batch.future_actions),
        "prompt_path": str(prompt_path),
        "actions_path": str(actions_path),
        "action_encoding_keys": ACTION_KEYS,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return OasisPreparedInputs(
        prompt_path=prompt_path,
        actions_path=actions_path,
        manifest_path=manifest_path,
        n_prompt_frames=len(batch.context_frames),
        total_frames=len(all_actions),
        generated_frames=len(batch.future_actions),
    )


def import_oasis_video(
    video_path: str | Path,
    prepared_manifest: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> OasisImportedVideo:
    video = Path(video_path)
    manifest_path = Path(prepared_manifest)
    result_path = Path(output_dir)

    if not manifest_path.exists():
        raise RunnerError(f"prepared manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    n_prompt_frames = int(manifest["n_prompt_frames"])
    generated_frames = int(manifest["generated_frames"])

    if result_path.exists():
        if overwrite:
            shutil.rmtree(result_path)
        else:
            raise RunnerError(f"output directory already exists: {result_path}")
    result_path.mkdir(parents=True)

    frames = _read_video_frames(video)
    expected_total = n_prompt_frames + generated_frames
    if len(frames) < expected_total:
        raise RunnerError(
            f"Oasis video has {len(frames)} frames, expected at least {expected_total} "
            f"({n_prompt_frames} prompt + {generated_frames} generated)"
        )

    generated = frames[n_prompt_frames : n_prompt_frames + generated_frames]
    for idx, frame in enumerate(generated):
        frame.save(result_path / f"gen_{idx:06d}.png")

    result_manifest = {
        "episode": manifest["episode"],
        "model": "oasis",
        "protocol": manifest["protocol"],
        "mock": False,
        "source_video": str(video),
        "prepared_manifest": str(manifest_path),
        "return_start": manifest["return_start"],
        "ctx_limit": manifest["ctx_limit"],
        "context_frames": n_prompt_frames,
        "output_frames": generated_frames,
        "video_total_frames_decoded": len(frames),
    }
    imported_manifest_path = result_path / "manifest.json"
    imported_manifest_path.write_text(json.dumps(result_manifest, indent=2) + "\n", encoding="utf-8")

    return OasisImportedVideo(
        output_dir=result_path,
        manifest_path=imported_manifest_path,
        generated_frames=generated_frames,
    )


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] not in {"prepare", "import-video", "-h", "--help"}:
        argv = ["prepare", *argv]

    parser = argparse.ArgumentParser(description="Prepare/import open-oasis files for persistence-probe.")
    subparsers = parser.add_subparsers(dest="command")

    prepare_parser = subparsers.add_parser("prepare", help="Prepare a validated episode for open-oasis generate.py.")
    prepare_parser.add_argument("episode_dir", type=Path)
    prepare_parser.add_argument("output_dir", type=Path)
    prepare_parser.add_argument("--ctx-limit", type=int, default=None)
    prepare_parser.add_argument("--fps", type=int, default=20)
    prepare_parser.add_argument("--overwrite", action="store_true")

    import_parser = subparsers.add_parser("import-video", help="Import Oasis mp4 output into results gen_*.png frames.")
    import_parser.add_argument("video_path", type=Path)
    import_parser.add_argument("prepared_manifest", type=Path)
    import_parser.add_argument("output_dir", type=Path)
    import_parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.command in {None, "prepare"}:
            if args.command is None:
                parser.error("missing command; use `prepare` or `import-video`")
            prepared = prepare_oasis_inputs(
                args.episode_dir,
                args.output_dir,
                ctx_limit=args.ctx_limit,
                fps=args.fps,
                overwrite=args.overwrite,
            )
            print(f"prompt: {prepared.prompt_path}")
            print(f"actions: {prepared.actions_path}")
            print(f"n_prompt_frames: {prepared.n_prompt_frames}")
            print(f"total_frames: {prepared.total_frames}")
        elif args.command == "import-video":
            imported = import_oasis_video(
                args.video_path,
                args.prepared_manifest,
                args.output_dir,
                overwrite=args.overwrite,
            )
            print(f"results: {imported.output_dir}")
            print(f"generated_frames: {imported.generated_frames}")
        else:
            parser.error(f"unknown command: {args.command}")
    except (RunnerError, ValidationFailure) as exc:
        print(f"oasis io failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
