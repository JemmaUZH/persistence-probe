from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def make_dummy_episode(
    episode_dir: str | Path,
    *,
    frame_count: int = 24,
    width: int = 64,
    height: int = 64,
    seed: int = 0,
) -> Path:
    if frame_count < 3:
        raise ValueError("frame_count must be at least 3 so required events can be placed")

    episode_path = Path(episode_dir)
    if episode_path.exists():
        shutil.rmtree(episode_path)
    frames_dir = episode_path / "frames"
    frames_dir.mkdir(parents=True)

    rng = random.Random(seed)
    intervention_frame = frame_count // 3
    look_away_frame = frame_count // 2
    return_start_frame = (frame_count * 2) // 3

    meta = {
        "scenario": "dummy_probe",
        "pair_id": "000",
        "arm": "control",
        "N_away": max(0, return_start_frame - look_away_frame),
        "world_seed": seed,
        "start_pos": [0.0, 64.0, 0.0],
        "probe_block": {"pos": [3.0, 64.0, 6.0], "type": "gold_block"},
        "fps": 20,
        "resolution": [width, height],
        "model_target": "dummy",
    }
    (episode_path / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    actions = []
    states = []
    for idx in range(frame_count):
        pixels = bytes(rng.randrange(256) for _ in range(width * height * 3))
        Image.frombytes("RGB", (width, height), pixels).save(frames_dir / f"{idx:06d}.png")

        actions.append(
            {
                "forward": idx < look_away_frame,
                "back": idx >= return_start_frame,
                "left": False,
                "right": False,
                "jump": False,
                "attack": False,
                "camera": [0.0, 0.0],
            }
        )
        states.append(
            {
                "probe_block_state": "present",
                "time_of_day": float((idx * 100) % 24000),
                "player_pos": [float(idx) * 0.05, 64.0, float(idx) * 0.02],
                "player_yaw_pitch": [180.0 if idx >= look_away_frame else 0.0, 0.0],
            }
        )

    events = [
        {"frame_idx": intervention_frame, "event": "intervention"},
        {"frame_idx": look_away_frame, "event": "look_away"},
        {"frame_idx": return_start_frame, "event": "return_start"},
    ]

    _write_jsonl(episode_path / "actions.jsonl", actions)
    _write_jsonl(episode_path / "state.jsonl", states)
    _write_jsonl(episode_path / "events.jsonl", events)
    return episode_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate one synthetic dummy episode.")
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    path = make_dummy_episode(
        args.episode_dir,
        frame_count=args.frames,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    print(f"wrote dummy episode: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
