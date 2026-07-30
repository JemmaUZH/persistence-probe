from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import random
import sys
import time
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


def _json_safe_action(action: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in action.items():
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        safe[key] = value
    return safe


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


def _to_minerl_action(action_space: Any, logged_action: dict[str, Any]) -> dict[str, Any]:
    env_action = copy.deepcopy(action_space.noop())
    for key in ("forward", "back", "left", "right", "jump", "sneak", "sprint", "attack", "use"):
        if key in env_action and key in logged_action:
            env_action[key] = int(logged_action[key])
    if "camera" in env_action and "camera" in logged_action:
        camera_x, camera_y = logged_action["camera"]
        env_action["camera"] = [float(camera_y) - 40.0, float(camera_x) - 40.0]
    if "equip" in env_action and logged_action.get("attack"):
        env_action["equip"] = "diamond_pickaxe" if "diamond_pickaxe" in str(action_space) else env_action["equip"]
    return env_action


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


def _make_probe_env_spec(width: int, height: int, total_frames: int) -> Any:
    import numpy as np

    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]

    from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
    from minerl.herobraine.hero import handlers

    class PersistenceProbeEnvSpec(SimpleEmbodimentEnvSpec):
        def __init__(self) -> None:
            super().__init__(
                name="PersistenceProbeBreakGold-v0",
                resolution=(width, height),
                max_episode_steps=total_frames + 64,
                reward_threshold=0.0,
            )

        def create_observables(self) -> list[Any]:
            return [
                handlers.POVObservation((width, height)),
                handlers.ObservationFromCurrentLocation(),
            ]

        def create_rewardables(self) -> list[Any]:
            return []

        def create_agent_start(self) -> list[Any]:
            return [
                handlers.AgentStartPlacement(x=0.5, y=5.0, z=0.5, yaw=0.0, pitch=0.0),
                handlers.AgentStartBreakSpeedMultiplier(1000.0),
                handlers.SimpleInventoryAgentStart([{"type": "diamond_pickaxe", "quantity": 1}]),
            ]

        def create_agent_handlers(self) -> list[Any]:
            return []

        def create_server_world_generators(self) -> list[Any]:
            return [handlers.FlatWorldGenerator(force_reset=True, generatorString="1;7,2x3,2;1")]

        def create_server_decorators(self) -> list[Any]:
            return [
                handlers.DrawingDecorator(
                    """
                    <DrawBlock x="0" y="5" z="4" type="gold_block"/>
                    <DrawBlock x="0" y="4" z="4" type="stone"/>
                    <DrawCuboid x1="-2" y1="4" z1="6" x2="2" y2="7" z2="6" type="stone"/>
                    """
                )
            ]

        def create_server_quit_producers(self) -> list[Any]:
            return [handlers.ServerQuitFromTimeUp(max((total_frames + 512) * 50, 300_000))]

        def create_server_initial_conditions(self) -> list[Any]:
            return [
                handlers.TimeInitialCondition(allow_passage_of_time=False, start_time=6000),
                handlers.SpawningInitialCondition(allow_spawning=False),
            ]

        def determine_success_from_rewards(self, rewards: list[Any]) -> bool:
            return False

        def is_from_folder(self, folder: str) -> bool:
            return False

        def get_docstring(self) -> str:
            return "Custom MineRL/Malmo persistence probe environment."

    return PersistenceProbeEnvSpec()


def _make_real_env(env_id: str, width: int, height: int, total_frames: int) -> Any:
    if env_id == "PersistenceProbeBreakGold-v0":
        return _make_probe_env_spec(width, height, total_frames).make()

    import gym
    import minerl  # noqa: F401

    return gym.make(env_id)


def _frame_from_obs(obs: Any, width: int, height: int) -> Image.Image:
    pov = obs["pov"] if isinstance(obs, dict) and "pov" in obs else obs
    frame = Image.fromarray(pov).convert("RGB")
    if frame.size != (width, height):
        resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
        frame = frame.resize((width, height), resample)
    return frame


def _float_from_mapping(mapping: Any, *keys: str, default: float = 0.0) -> float:
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        if key in mapping:
            try:
                return float(mapping[key])
            except (TypeError, ValueError):
                return default
    return default


def _player_state_from_obs(obs: Any) -> tuple[list[float], list[float]]:
    location = obs.get("location_stats", {}) if isinstance(obs, dict) else {}
    pos = [
        _float_from_mapping(location, "xpos", "x", "XPos"),
        _float_from_mapping(location, "ypos", "y", "YPos", default=64.0),
        _float_from_mapping(location, "zpos", "z", "ZPos"),
    ]
    yaw_pitch = [
        _float_from_mapping(location, "yaw", "Yaw"),
        _float_from_mapping(location, "pitch", "Pitch"),
    ]
    return pos, yaw_pitch


def _probe_state_for_frame(frame_idx: int, intervention_frame: int, arm: str) -> str:
    if arm == "intervene" and frame_idx >= intervention_frame:
        return "absent"
    return "present"


def _write_real_episode(
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
    env_id: str,
) -> list[dict[str, Any]]:
    if episode_dir.exists():
        shutil.rmtree(episode_dir)
    frames_dir = episode_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    intervention_frame = k1
    look_away_frame = k1 + 1
    return_start = look_away_frame + n_away
    total_frames = return_start + k2

    meta = {
        "scenario": scenario,
        "pair_id": f"{pair_id:03d}",
        "arm": arm,
        "N_away": n_away,
        "world_seed": seed + pair_id,
        "start_pos": [0.5, 5.0, 0.5],
        "probe_block": {"pos": [0.0, 5.0, 4.0], "type": "gold_block"},
        "fps": fps,
        "resolution": [width, height],
        "model_target": "minerl",
    }
    (episode_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    actions: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    env = _make_real_env(env_id, width, height, total_frames)
    try:
        if hasattr(env, "seed"):
            env.seed(seed + pair_id)
        obs = env.reset()
        for frame_idx in range(total_frames):
            _frame_from_obs(obs, width, height).save(frames_dir / f"{frame_idx:06d}.png")
            logged_action = _script_action(frame_idx, intervention_frame, look_away_frame, return_start, arm)
            actions.append(logged_action)
            player_pos, player_yaw_pitch = _player_state_from_obs(obs)
            states.append(
                {
                    "probe_block_state": _probe_state_for_frame(frame_idx, intervention_frame, arm),
                    "time_of_day": 6000.0,
                    "player_pos": player_pos,
                    "player_yaw_pitch": player_yaw_pitch,
                }
            )
            if frame_idx < total_frames - 1:
                obs, _reward, done, _info = env.step(_to_minerl_action(env.action_space, logged_action))
                if done:
                    raise RecorderError(f"{episode_dir.name}: MineRL env ended early at frame {frame_idx}")
    finally:
        env.close()

    events = [
        {"frame_idx": intervention_frame, "event": "intervention"},
        {"frame_idx": look_away_frame, "event": "look_away"},
        {"frame_idx": return_start, "event": "return_start"},
    ]
    _write_jsonl(episode_dir / "actions.jsonl", actions)
    _write_jsonl(episode_dir / "state.jsonl", states)
    _write_jsonl(episode_dir / "events.jsonl", events)
    return actions


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _run_real_episode_worker(
    *,
    output_root: Path,
    scenario: str,
    pair_idx: int,
    arm: str,
    n_away: int,
    k1: int,
    k2: int,
    width: int,
    height: int,
    fps: int,
    seed: int,
    env_id: str,
    retries: int,
) -> list[dict[str, Any]]:
    cmd = [
        sys.executable,
        "-m",
        "recorder.record",
        "--scenario",
        scenario,
        "--N",
        str(n_away),
        "--pairs",
        "1",
        "--output-root",
        str(output_root),
        "--K1",
        str(k1),
        "--K2",
        str(k2),
        "--width",
        str(width),
        "--height",
        str(height),
        "--fps",
        str(fps),
        "--seed",
        str(seed),
        "--env-id",
        env_id,
        "--_single-real-arm",
        arm,
        "--_pair-idx",
        str(pair_idx),
    ]
    result = None
    for attempt in range(1, retries + 1):
        result = subprocess.run(cmd, text=True)
        if result.returncode == 0:
            break
        if attempt < retries:
            print(f"worker retry {attempt}/{retries} for pair {pair_idx:03d} {arm}", file=sys.stderr)
            time.sleep(10 * attempt)
    if result is None or result.returncode != 0:
        raise RecorderError(f"worker failed for pair {pair_idx:03d} {arm} after {retries} attempts")
    pair_id = f"{pair_idx:03d}_N{n_away}"
    episode_dir = output_root / f"{scenario}_{pair_id}_{arm}"
    return _read_jsonl(episode_dir / "actions.jsonl")


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


def record_real(
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
    env_id: str = "PersistenceProbeBreakGold-v0",
    worker_retries: int = 3,
) -> list[Path]:
    written: list[Path] = []
    for n_away in n_values:
        for pair_idx in range(pairs):
            pair_id = f"{pair_idx:03d}_N{n_away}"
            control_dir = output_root / f"{scenario}_{pair_id}_control"
            intervene_dir = output_root / f"{scenario}_{pair_id}_intervene"
            control_actions = _run_real_episode_worker(
                output_root=output_root,
                scenario=scenario,
                pair_idx=pair_idx,
                arm="control",
                n_away=n_away,
                k1=k1,
                k2=k2,
                width=width,
                height=height,
                fps=fps,
                seed=seed,
                env_id=env_id,
                retries=worker_retries,
            )
            time.sleep(3)
            intervene_actions = _run_real_episode_worker(
                output_root=output_root,
                scenario=scenario,
                pair_idx=pair_idx,
                arm="intervene",
                n_away=n_away,
                k1=k1,
                k2=k2,
                width=width,
                height=height,
                fps=fps,
                seed=seed,
                env_id=env_id,
                retries=worker_retries,
            )
            time.sleep(3)
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
    parser.add_argument("--env-id", default="PersistenceProbeBreakGold-v0")
    parser.add_argument("--worker-retries", type=int, default=3)
    parser.add_argument("--_single-real-arm", choices=["control", "intervene"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_pair-idx", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        if args._single_real_arm is not None:
            if len(args.n_values) != 1:
                raise RecorderError("single real worker expects exactly one N value")
            pair_id = f"{args._pair_idx:03d}_N{args.n_values[0]}"
            episode_dir = args.output_root / f"{args.scenario}_{pair_id}_{args._single_real_arm}"
            _write_real_episode(
                episode_dir,
                scenario=args.scenario,
                pair_id=args._pair_idx,
                arm=args._single_real_arm,
                n_away=args.n_values[0],
                k1=args.K1,
                k2=args.K2,
                width=args.width,
                height=args.height,
                fps=args.fps,
                seed=args.seed,
                env_id=args.env_id,
            )
            validate_episode(episode_dir)
            written = [episode_dir]
        elif args.mock:
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
        else:
            written = record_real(
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
                env_id=args.env_id,
            )
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"record failed: MineRL dependencies are unavailable: {exc}", file=sys.stderr)
        return 1
    except (RecorderError, ValidationFailure) as exc:
        print(f"record failed: {exc}", file=sys.stderr)
        return 1

    for episode_dir in written:
        print(f"wrote: {episode_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
