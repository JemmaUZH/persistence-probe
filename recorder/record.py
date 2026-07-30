from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

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
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
]


class RecorderError(Exception):
    """Raised when recording cannot satisfy the episode contract."""


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def _base_action() -> dict[str, Any]:
    action = {key: 0 for key in ACTION_KEYS}
    action["camera"] = [40, 40]
    return action


def _script_action(frame_idx: int, intervention_frame: int, look_away_frame: int, return_start: int, arm: str) -> dict[str, Any]:
    action = _base_action()
    if frame_idx < intervention_frame:
        action["forward"] = 1
    elif frame_idx == intervention_frame and arm == "intervene":
        action["attack"] = 1
    elif look_away_frame <= frame_idx < return_start:
        action["forward"] = 1
        action["camera"] = [80, 40] if frame_idx == look_away_frame else [40, 40]
    elif frame_idx >= return_start:
        action["back"] = 1
    return action


def _draw_mock_frame(width: int, height: int, rng: random.Random, present: bool, away: bool) -> Image.Image:
    img = Image.new("RGB", (width, height), (126, 188, 234))
    draw = ImageDraw.Draw(img)
    horizon = int(height * 0.48)
    draw.rectangle((0, horizon, width, height), fill=(94, 151, 74))
    draw.rectangle((0, int(height * 0.82), width, height), fill=(99, 99, 99))
    for _ in range(50):
        x = rng.randrange(width)
        y = rng.randrange(height)
        shade = rng.randrange(-8, 9)
        base = img.getpixel((x, y))
        img.putpixel((x, y), tuple(max(0, min(255, channel + shade)) for channel in base))
    if present and not away:
        box_w = max(8, width // 11)
        box_h = max(8, height // 7)
        cx = width // 2
        cy = int(height * 0.55)
        draw.rectangle((cx - box_w, cy - box_h, cx + box_w, cy + box_h), fill=(238, 201, 48), outline=(86, 63, 12), width=2)
    if away:
        draw.rectangle((int(width * 0.42), int(height * 0.35), int(width * 0.58), int(height * 0.8)), fill=(90, 73, 59))
    return img


def _assert_paired_actions(control_actions: list[dict[str, Any]], intervene_actions: list[dict[str, Any]], intervention_frame: int) -> None:
    if len(control_actions) != len(intervene_actions):
        raise RecorderError("paired action logs have different lengths")
    for idx, (control, intervene) in enumerate(zip(control_actions, intervene_actions)):
        if idx == intervention_frame:
            continue
        if control != intervene:
            raise RecorderError(f"paired actions differ outside intervention window at frame {idx}")


def _write_mock_episode(
    episode_dir: Path,
    *,
    scenario: str,
    pair_id: int,
    arm: str,
    n_away: int,
    k1: int,
    k2: int,
    width: int,
    height: int,
    fps: int,
    seed: int,
) -> list[dict[str, Any]]:
    frames_dir = episode_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    intervention_frame = k1
    look_away_frame = k1 + 1
    return_start = look_away_frame + n_away
    total_frames = return_start + k2
    rng = random.Random(seed + pair_id * 1000 + (1 if arm == "intervene" else 0))

    meta = {
        "scenario": scenario,
        "pair_id": f"{pair_id:03d}",
        "arm": arm,
        "N_away": n_away,
        "world_seed": seed + pair_id,
        "start_pos": [0.0, 64.0, 0.0],
        "probe_block": {"pos": [3.0, 64.0, 6.0], "type": "gold_block"},
        "fps": fps,
        "resolution": [width, height],
        "model_target": "mock",
    }
    (episode_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    actions: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    for frame_idx in range(total_frames):
        away = look_away_frame <= frame_idx < return_start
        present = not (arm == "intervene" and frame_idx >= intervention_frame)
        _draw_mock_frame(width, height, rng, present=present, away=away).save(frames_dir / f"{frame_idx:06d}.png")
        actions.append(_script_action(frame_idx, intervention_frame, look_away_frame, return_start, arm))
        states.append(
            {
                "probe_block_state": "present" if present else "absent",
                "time_of_day": float((frame_idx * 100) % 24000),
                "player_pos": [float(frame_idx) * 0.05, 64.0, float(frame_idx) * 0.02],
                "player_yaw_pitch": [180.0 if away else 0.0, 0.0],
            }
        )

    events = [
        {"frame_idx": intervention_frame, "event": "intervention"},
        {"frame_idx": look_away_frame, "event": "look_away"},
        {"frame_idx": return_start, "event": "return_start"},
    ]
    _write_jsonl(episode_dir / "actions.jsonl", actions)
    _write_jsonl(episode_dir / "state.jsonl", states)
    _write_jsonl(episode_dir / "events.jsonl", events)
    return actions


def record_mock(
    *,
    output_root: Path,
    scenario: str,
    n_values: list[int],
    pairs: int,
    k1: int = 8,
    k2: int = 32,
    width: int = 640,
    height: int = 360,
    fps: int = 20,
    seed: int = 0,
) -> list[Path]:
    written: list[Path] = []
    for n_away in n_values:
        for pair_idx in range(pairs):
            pair_id = f"{pair_idx:03d}_N{n_away}"
            control_dir = output_root / f"{scenario}_{pair_id}_control"
            intervene_dir = output_root / f"{scenario}_{pair_id}_intervene"
            control_actions = _write_mock_episode(
                control_dir,
                scenario=scenario,
                pair_id=pair_idx,
                arm="control",
                n_away=n_away,
                k1=k1,
                k2=k2,
                width=width,
                height=height,
                fps=fps,
                seed=seed,
            )
            intervene_actions = _write_mock_episode(
                intervene_dir,
                scenario=scenario,
                pair_id=pair_idx,
                arm="intervene",
                n_away=n_away,
                k1=k1,
                k2=k2,
                width=width,
                height=height,
                fps=fps,
                seed=seed,
            )
            _assert_paired_actions(control_actions, intervene_actions, intervention_frame=k1)
            for episode_dir in (control_dir, intervene_dir):
                validate_episode(episode_dir)
                written.append(episode_dir)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record MineRL episodes, or schema-valid mock episodes for CI/Colab.")
    parser.add_argument("--scenario", default="break_gold")
    parser.add_argument("--N", dest="n_values", type=int, nargs="+", required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("episodes"))
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--K1", type=int, default=8)
    parser.add_argument("--K2", type=int, default=32)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if not args.mock:
        print(
            "real MineRL recording is not implemented in Colab path; use --mock or run the MineRL-specific task on Linux/Docker",
            file=sys.stderr,
        )
        return 1

    try:
        written = record_mock(
            output_root=args.output_root,
            scenario=args.scenario,
            n_values=args.n_values,
            pairs=args.pairs,
            k1=args.K1,
            k2=args.K2,
            width=args.width,
            height=args.height,
            fps=args.fps,
            seed=args.seed,
        )
    except (RecorderError, ValidationFailure) as exc:
        print(f"record failed: {exc}", file=sys.stderr)
        return 1

    for episode_dir in written:
        print(f"wrote: {episode_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
