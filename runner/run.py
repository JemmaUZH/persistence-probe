from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from schema.validate import ValidationFailure, validate_episode


SUPPORTED_PROTOCOLS = {"teacher_forced"}


class RunnerError(Exception):
    """Raised when runner inputs or configuration are invalid."""


@dataclass(frozen=True)
class TeacherForcedBatch:
    context_frames: list[Path]
    context_actions: list[dict[str, Any]]
    future_actions: list[dict[str, Any]]
    return_start: int
    output_count: int


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _stable_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json(config)).hexdigest()


def _episode_dirs(root: Path) -> list[Path]:
    if not root.exists():
        raise RunnerError(f"episodes root does not exist: {root}")
    if (root / "meta.json").exists():
        return [root]
    episodes = sorted(path for path in root.iterdir() if path.is_dir() and (path / "meta.json").exists())
    if not episodes:
        raise RunnerError(f"no episode directories with meta.json under: {root}")
    return episodes


def _teacher_forced_batch(episode_dir: Path, ctx_limit: int | None) -> TeacherForcedBatch:
    frames = sorted((episode_dir / "frames").glob("*.png"))
    actions = _load_jsonl(episode_dir / "actions.jsonl")
    events = _load_jsonl(episode_dir / "events.jsonl")
    return_events = [event for event in events if event["event"] == "return_start"]
    if len(return_events) != 1:
        raise RunnerError(f"{episode_dir}: expected exactly one return_start event, found {len(return_events)}")

    return_start = return_events[0]["frame_idx"]
    if return_start <= 0:
        raise RunnerError(f"{episode_dir}: return_start must leave at least one context frame")
    if return_start >= len(frames):
        raise RunnerError(f"{episode_dir}: return_start is outside available frames")

    context_frames = frames[:return_start]
    context_actions = actions[:return_start]
    if ctx_limit is not None:
        if ctx_limit <= 0:
            raise RunnerError("ctx_limit must be positive when provided")
        context_frames = context_frames[-ctx_limit:]
        context_actions = context_actions[-ctx_limit:]

    future_actions = actions[return_start:]
    return TeacherForcedBatch(
        context_frames=context_frames,
        context_actions=context_actions,
        future_actions=future_actions,
        return_start=return_start,
        output_count=len(future_actions),
    )


def generate_mock(
    context_frames: list[Path],
    context_actions: list[dict[str, Any]],
    future_actions: list[dict[str, Any]],
    seed: int,
    ctx_limit: int | None = None,
) -> list[Image.Image]:
    """Generate deterministic noise frames for CPU-only pipeline tests."""
    if not context_frames:
        raise RunnerError("mock generation requires at least one context frame")

    with Image.open(context_frames[-1]) as last_context:
        width, height = last_context.size

    digest = hashlib.sha256()
    digest.update(str(seed).encode("ascii"))
    digest.update(str(ctx_limit).encode("ascii"))
    digest.update(_stable_json(context_actions))
    digest.update(_stable_json(future_actions))
    for frame in context_frames:
        digest.update(_sha256_file(frame).encode("ascii"))

    base_seed = int.from_bytes(digest.digest()[:8], "big")
    images: list[Image.Image] = []
    for idx, action in enumerate(future_actions):
        frame_rng = random.Random(base_seed + idx)
        action_tint = hashlib.sha256(_stable_json(action)).digest()
        pixels = bytearray(width * height * 3)
        for offset in range(0, len(pixels), 3):
            pixels[offset] = frame_rng.randrange(256) ^ action_tint[0]
            pixels[offset + 1] = frame_rng.randrange(256) ^ action_tint[1]
            pixels[offset + 2] = frame_rng.randrange(256) ^ action_tint[2]
        images.append(Image.frombytes("RGB", (width, height), bytes(pixels)))
    return images


def _write_manifest(
    output_dir: Path,
    *,
    episode_dir: Path,
    model: str,
    protocol: str,
    mock: bool,
    seed: int,
    ctx_limit: int | None,
    batch: TeacherForcedBatch,
    config: dict[str, Any],
) -> None:
    manifest = {
        "episode": episode_dir.name,
        "model": model,
        "protocol": protocol,
        "mock": mock,
        "seed": seed,
        "ctx_limit": ctx_limit,
        "return_start": batch.return_start,
        "context_frames": len(batch.context_frames),
        "future_actions": len(batch.future_actions),
        "output_frames": batch.output_count,
        "config_hash": _config_hash(config),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _run_mock_episode(
    episode_dir: Path,
    output_dir: Path,
    *,
    model: str,
    protocol: str,
    seed: int,
    ctx_limit: int | None,
    overwrite: bool,
) -> None:
    validate_episode(episode_dir)
    batch = _teacher_forced_batch(episode_dir, ctx_limit)
    meta = _load_json(episode_dir / "meta.json")
    config = {
        "model": model,
        "protocol": protocol,
        "mock": True,
        "episode_resolution": meta["resolution"],
        "episode_fps": meta["fps"],
    }

    if output_dir.exists():
        if not overwrite:
            existing = sorted(output_dir.glob("gen_*.png"))
            if len(existing) == batch.output_count and (output_dir / "manifest.json").exists():
                return
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = generate_mock(
        batch.context_frames,
        batch.context_actions,
        batch.future_actions,
        seed=seed,
        ctx_limit=ctx_limit,
    )
    for idx, frame in enumerate(frames):
        frame.save(output_dir / f"gen_{idx:06d}.png")

    _write_manifest(
        output_dir,
        episode_dir=episode_dir,
        model=model,
        protocol=protocol,
        mock=True,
        seed=seed,
        ctx_limit=ctx_limit,
        batch=batch,
        config=config,
    )


def run(
    *,
    model: str,
    episodes: Path,
    results: Path,
    protocol: str,
    mock: bool,
    seed: int,
    ctx_limit: int | None,
    overwrite: bool,
) -> list[Path]:
    if protocol not in SUPPORTED_PROTOCOLS:
        raise RunnerError(f"unsupported protocol: {protocol}")
    if not mock:
        raise RunnerError(
            "real model execution is not configured yet; pass --mock until Oasis repo/weights are provided"
        )

    written: list[Path] = []
    for episode_dir in _episode_dirs(episodes):
        output_dir = results / episode_dir.name / model
        _run_mock_episode(
            episode_dir,
            output_dir,
            model=model,
            protocol=protocol,
            seed=seed,
            ctx_limit=ctx_limit,
            overwrite=overwrite,
        )
        written.append(output_dir)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run model generation over validated episodes.")
    parser.add_argument("--model", required=True, help="Model name for results/{episode}/{model}/.")
    parser.add_argument("--episodes", type=Path, required=True, help="Episode directory or root of episodes.")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--protocol", default="teacher_forced")
    parser.add_argument("--mock", action="store_true", help="Use deterministic CPU mock generation.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ctx-limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        outputs = run(
            model=args.model,
            episodes=args.episodes,
            results=args.results,
            protocol=args.protocol,
            mock=args.mock,
            seed=args.seed,
            ctx_limit=args.ctx_limit,
            overwrite=args.overwrite,
        )
    except (RunnerError, ValidationFailure) as exc:
        print(f"runner failed: {exc}", file=sys.stderr)
        return 1

    for output in outputs:
        print(f"wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
