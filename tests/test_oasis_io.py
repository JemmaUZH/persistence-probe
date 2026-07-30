from __future__ import annotations

import json

import pytest
import torch

from runner.oasis_io import (
    ACTION_KEYS,
    _frames_to_video,
    encode_oasis_actions,
    import_oasis_video,
    prepare_oasis_inputs,
)
from runner.run import RunnerError
from schema.make_dummy import make_dummy_episode


def test_encode_oasis_actions_fills_missing_binary_keys_with_zero():
    encoded = encode_oasis_actions([{"forward": 1, "camera": [40, 20]}])

    assert encoded.shape == (1, len(ACTION_KEYS))
    assert encoded[0, ACTION_KEYS.index("forward")] == 1
    assert encoded[0, ACTION_KEYS.index("attack")] == 0
    assert encoded[0, ACTION_KEYS.index("cameraX")] == 0
    assert encoded[0, ACTION_KEYS.index("cameraY")] == -0.5


def test_prepare_oasis_inputs_exports_prompt_video_actions_and_manifest(tmp_path):
    episode = make_dummy_episode(tmp_path / "episodes" / "dummy_000_control", frame_count=12, width=64, height=64)

    prepared = prepare_oasis_inputs(episode, tmp_path / "oasis_inputs", overwrite=True)

    assert prepared.prompt_path.exists()
    assert prepared.actions_path.exists()
    assert prepared.manifest_path.exists()
    manifest = json.loads(prepared.manifest_path.read_text())
    actions = torch.load(prepared.actions_path, weights_only=True)
    assert manifest["n_prompt_frames"] == prepared.n_prompt_frames
    assert manifest["total_frames"] == prepared.total_frames
    assert actions.shape == (prepared.total_frames - 1, len(ACTION_KEYS))


def test_import_oasis_video_exports_only_generated_return_frames(tmp_path):
    episode = make_dummy_episode(tmp_path / "episodes" / "dummy_000_control", frame_count=12, width=64, height=64)
    prepared = prepare_oasis_inputs(episode, tmp_path / "oasis_inputs", overwrite=True)
    oasis_video = tmp_path / "oasis.mp4"
    _frames_to_video(sorted((episode / "frames").glob("*.png")), oasis_video, fps=20)

    imported = import_oasis_video(
        oasis_video,
        prepared.manifest_path,
        tmp_path / "results" / "dummy_000_control" / "oasis",
        overwrite=True,
    )

    assert imported.generated_frames == prepared.generated_frames
    assert len(list(imported.output_dir.glob("gen_*.png"))) == prepared.generated_frames
    manifest = json.loads(imported.manifest_path.read_text())
    assert manifest["context_frames"] == prepared.n_prompt_frames
    assert manifest["output_frames"] == prepared.generated_frames
    assert manifest["output_frame_mode"] == "prompt_plus_generated"


def test_import_oasis_video_accepts_generated_only_video(tmp_path):
    episode = make_dummy_episode(tmp_path / "episodes" / "dummy_000_control", frame_count=12, width=64, height=64)
    prepared = prepare_oasis_inputs(episode, tmp_path / "oasis_inputs", overwrite=True)
    manifest = json.loads(prepared.manifest_path.read_text())
    return_start = int(manifest["return_start"])
    generated_frames = sorted((episode / "frames").glob("*.png"))[return_start:]
    oasis_video = tmp_path / "oasis_generated_only.mp4"
    _frames_to_video(generated_frames, oasis_video, fps=20)

    imported = import_oasis_video(
        oasis_video,
        prepared.manifest_path,
        tmp_path / "results" / "dummy_000_control" / "oasis",
        overwrite=True,
    )

    assert imported.generated_frames == prepared.generated_frames
    assert len(list(imported.output_dir.glob("gen_*.png"))) == prepared.generated_frames
    imported_manifest = json.loads(imported.manifest_path.read_text())
    assert imported_manifest["output_frame_mode"] == "generated_only"


def test_import_oasis_video_fails_when_video_is_too_short(tmp_path):
    episode = make_dummy_episode(tmp_path / "episodes" / "dummy_000_control", frame_count=12, width=64, height=64)
    prepared = prepare_oasis_inputs(episode, tmp_path / "oasis_inputs", overwrite=True)
    manifest = json.loads(prepared.manifest_path.read_text())
    manifest["generated_frames"] = 99
    prepared.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RunnerError, match="expected either"):
        import_oasis_video(
            prepared.prompt_path,
            prepared.manifest_path,
            tmp_path / "results" / "dummy_000_control" / "oasis",
            overwrite=True,
        )
