from __future__ import annotations

import json
import shutil

import pytest

from schema.make_dummy import make_dummy_episode
from schema.validate import ValidationFailure, validate_episode


def test_dummy_episode_validates(tmp_path):
    episode = make_dummy_episode(tmp_path / "dummy_000_control", frame_count=8, width=16, height=16)

    validate_episode(episode)


def test_missing_required_meta_field_fails_loudly(tmp_path):
    episode = make_dummy_episode(tmp_path / "dummy_000_control", frame_count=8, width=16, height=16)
    meta_path = episode / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["model_target"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValidationFailure, match="meta.json"):
        validate_episode(episode)


def test_jsonl_length_must_match_frame_count(tmp_path):
    episode = make_dummy_episode(tmp_path / "dummy_000_control", frame_count=8, width=16, height=16)
    actions_path = episode / "actions.jsonl"
    actions_path.write_text("\n".join(actions_path.read_text(encoding="utf-8").splitlines()[:-1]) + "\n")

    with pytest.raises(ValidationFailure, match="one line per frame"):
        validate_episode(episode)


def test_frame_names_must_be_contiguous(tmp_path):
    episode = make_dummy_episode(tmp_path / "dummy_000_control", frame_count=8, width=16, height=16)
    shutil.move(episode / "frames" / "000007.png", episode / "frames" / "000009.png")

    with pytest.raises(ValidationFailure, match="expected 000007.png"):
        validate_episode(episode)


def test_event_frame_idx_must_be_in_range(tmp_path):
    episode = make_dummy_episode(tmp_path / "dummy_000_control", frame_count=8, width=16, height=16)
    events_path = episode / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    events[0]["frame_idx"] = 99
    events_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    with pytest.raises(ValidationFailure, match="outside frame range"):
        validate_episode(episode)
