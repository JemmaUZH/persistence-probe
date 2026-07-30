from __future__ import annotations

import json
import shutil

from readout.run import run_readout
from recorder.record import record_mock
from runner.run import run


def test_mock_readout_writes_metrics_and_audit(tmp_path):
    record_mock(
        output_root=tmp_path / "episodes",
        scenario="break_gold",
        n_values=[8],
        pairs=1,
        width=64,
        height=64,
        k1=4,
        k2=6,
    )
    run(
        model="mock",
        episodes=tmp_path / "episodes",
        results=tmp_path / "results",
        protocol="teacher_forced",
        mock=True,
        seed=123,
        ctx_limit=None,
        overwrite=True,
    )

    rows = run_readout(
        episodes_root=tmp_path / "episodes",
        results_root=tmp_path / "results",
        model="mock",
        output_dir=tmp_path / "results" / "readout",
        mock=True,
    )

    assert len(rows) == 2
    metrics = json.loads((tmp_path / "results" / "readout" / "metrics.json").read_text())
    assert metrics["mock_readout"] is True
    assert len(metrics["summary"]) == 2
    assert (tmp_path / "results" / "readout" / "metrics.csv").exists()
    assert (tmp_path / "results" / "readout" / "audit.html").exists()


def _copy_return_frames_to_results(episodes_root, results_root, model):
    for episode_dir in sorted(episodes_root.glob("break_gold_*")):
        events = [json.loads(line) for line in (episode_dir / "events.jsonl").read_text().splitlines()]
        return_start = next(row["frame_idx"] for row in events if row["event"] == "return_start")
        output_dir = results_root / episode_dir.name / model
        output_dir.mkdir(parents=True)
        for out_idx, frame_path in enumerate(sorted((episode_dir / "frames").glob("*.png"))[return_start:]):
            shutil.copyfile(frame_path, output_dir / f"gen_{out_idx:06d}.png")


def test_real_template_readout_calibrates_on_real_frames_and_scores_generated(tmp_path):
    record_mock(
        output_root=tmp_path / "episodes",
        scenario="break_gold",
        n_values=[8],
        pairs=1,
        width=96,
        height=72,
        k1=4,
        k2=8,
    )
    _copy_return_frames_to_results(tmp_path / "episodes", tmp_path / "results", "identity")

    rows = run_readout(
        episodes_root=tmp_path / "episodes",
        results_root=tmp_path / "results",
        model="identity",
        output_dir=tmp_path / "results" / "readout_real",
        mock=False,
    )

    assert len(rows) == 2
    assert {row["predicted_state"] for row in rows} == {"present", "absent"}
    assert all(row["correct_state"] == 1 for row in rows)
    metrics = json.loads((tmp_path / "results" / "readout_real" / "metrics.json").read_text())
    assert metrics["mock_readout"] is False
    assert metrics["readout_mode"] == "template_ssim"
    assert metrics["calibrations"][0]["real_frame_accuracy"] >= 0.95
    assert metrics["stale_resurrection"][0]["stale_resurrection_rate"] == 0.0
