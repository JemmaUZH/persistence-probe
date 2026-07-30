from __future__ import annotations

import hashlib
import json

import pytest

from runner.run import RunnerError, _teacher_forced_batch, run
from schema.make_dummy import make_dummy_episode


def _hash_outputs(output_dir):
    digest = hashlib.sha256()
    for path in sorted(output_dir.glob("gen_*.png")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_teacher_forced_batch_starts_future_actions_at_return_start(tmp_path):
    episode = make_dummy_episode(tmp_path / "dummy_000_control", frame_count=12, width=16, height=16)
    events = [json.loads(line) for line in (episode / "events.jsonl").read_text().splitlines()]
    return_start = next(event["frame_idx"] for event in events if event["event"] == "return_start")

    batch = _teacher_forced_batch(episode, ctx_limit=None)

    assert batch.return_start == return_start
    assert len(batch.context_frames) == return_start
    assert len(batch.context_actions) == return_start
    assert len(batch.future_actions) == 12 - return_start
    assert batch.output_count == len(batch.future_actions)


def test_mock_runner_is_deterministic_under_fixed_seed(tmp_path):
    episode = make_dummy_episode(tmp_path / "episodes" / "dummy_000_control", frame_count=10, width=16, height=16)
    results = tmp_path / "results"

    outputs = run(
        model="mock",
        episodes=episode,
        results=results,
        protocol="teacher_forced",
        mock=True,
        seed=123,
        ctx_limit=None,
        overwrite=True,
    )
    first_hash = _hash_outputs(outputs[0])

    outputs = run(
        model="mock",
        episodes=episode,
        results=results,
        protocol="teacher_forced",
        mock=True,
        seed=123,
        ctx_limit=None,
        overwrite=True,
    )
    second_hash = _hash_outputs(outputs[0])

    assert first_hash == second_hash
    manifest = json.loads((outputs[0] / "manifest.json").read_text())
    assert manifest["mock"] is True
    assert manifest["protocol"] == "teacher_forced"
    assert manifest["output_frames"] == len(list(outputs[0].glob("gen_*.png")))


def test_ctx_limit_truncates_context(tmp_path):
    episode = make_dummy_episode(tmp_path / "dummy_000_control", frame_count=12, width=16, height=16)

    batch = _teacher_forced_batch(episode, ctx_limit=3)

    assert len(batch.context_frames) == 3
    assert len(batch.context_actions) == 3


def test_real_model_without_mock_fails_until_configured(tmp_path):
    episode = make_dummy_episode(tmp_path / "dummy_000_control", frame_count=8, width=16, height=16)

    with pytest.raises(RunnerError, match="pass --mock"):
        run(
            model="oasis",
            episodes=episode,
            results=tmp_path / "results",
            protocol="teacher_forced",
            mock=False,
            seed=0,
            ctx_limit=None,
            overwrite=False,
        )
