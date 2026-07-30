from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import jsonschema
from PIL import Image


SCHEMA_DIR = Path(__file__).with_name("schemas")
FRAME_RE = re.compile(r"^\d{6}\.png$")


class ValidationFailure(Exception):
    """Raised when an episode violates the T0 data contract."""


def _load_schema(name: str) -> dict[str, Any]:
    with (SCHEMA_DIR / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise ValidationFailure(f"missing required file: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationFailure(f"{path.name}: invalid JSON at line {exc.lineno}: {exc.msg}") from exc


def _load_jsonl(path: Path, schema: dict[str, Any]) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValidationFailure(f"missing required file: {path.name}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                raise ValidationFailure(f"{path.name}:{line_no}: blank JSONL lines are not allowed")
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValidationFailure(f"{path.name}:{line_no}: invalid JSON: {exc.msg}") from exc
            try:
                jsonschema.validate(row, schema)
            except jsonschema.ValidationError as exc:
                raise ValidationFailure(f"{path.name}:{line_no}: {exc.message}") from exc
            rows.append(row)
    return rows


def _validate_frame_files(frames_dir: Path, expected_resolution: tuple[int, int]) -> list[Path]:
    if not frames_dir.is_dir():
        raise ValidationFailure("missing required directory: frames/")

    frames = sorted(path for path in frames_dir.iterdir() if path.is_file())
    if not frames:
        raise ValidationFailure("frames/: expected at least one PNG frame")

    for idx, frame in enumerate(frames):
        expected_name = f"{idx:06d}.png"
        if frame.name != expected_name or not FRAME_RE.match(frame.name):
            raise ValidationFailure(f"frames/: expected {expected_name}, found {frame.name}")
        try:
            with Image.open(frame) as img:
                img.verify()
            with Image.open(frame) as img:
                if img.size != expected_resolution:
                    raise ValidationFailure(
                        f"{frame.relative_to(frames_dir.parent)}: expected resolution "
                        f"{expected_resolution}, found {img.size}"
                    )
        except ValidationFailure:
            raise
        except Exception as exc:
            raise ValidationFailure(f"{frame.relative_to(frames_dir.parent)}: invalid PNG") from exc

    return frames


def validate_episode(episode_dir: str | Path) -> None:
    episode_path = Path(episode_dir)
    if not episode_path.is_dir():
        raise ValidationFailure(f"episode directory does not exist: {episode_path}")

    schemas = {
        "meta": _load_schema("meta.schema.json"),
        "actions": _load_schema("actions.schema.json"),
        "state": _load_schema("state.schema.json"),
        "events": _load_schema("events.schema.json"),
    }

    meta = _load_json(episode_path / "meta.json")
    try:
        jsonschema.validate(meta, schemas["meta"])
    except jsonschema.ValidationError as exc:
        raise ValidationFailure(f"meta.json: {exc.message}") from exc

    width, height = meta["resolution"]
    frames = _validate_frame_files(episode_path / "frames", (width, height))
    actions = _load_jsonl(episode_path / "actions.jsonl", schemas["actions"])
    states = _load_jsonl(episode_path / "state.jsonl", schemas["state"])
    events = _load_jsonl(episode_path / "events.jsonl", schemas["events"])

    frame_count = len(frames)
    if len(actions) != frame_count:
        raise ValidationFailure(
            f"actions.jsonl: expected one line per frame ({frame_count}), found {len(actions)}"
        )
    if len(states) != frame_count:
        raise ValidationFailure(
            f"state.jsonl: expected one line per frame ({frame_count}), found {len(states)}"
        )
    for row_no, event in enumerate(events, start=1):
        if event["frame_idx"] >= frame_count:
            raise ValidationFailure(
                f"events.jsonl:{row_no}: frame_idx {event['frame_idx']} outside frame range 0..{frame_count - 1}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a persistence-probe episode directory.")
    parser.add_argument("episode_dir", type=Path)
    args = parser.parse_args(argv)

    try:
        validate_episode(args.episode_dir)
    except ValidationFailure as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1

    print(f"validation passed: {args.episode_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
