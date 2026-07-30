from __future__ import annotations

import json
import shutil

from figures.make import make_figure
from recorder.record import record_mock


def _copy_generated_frames(episodes_root, results_root, model):
    for episode_dir in sorted(episodes_root.glob("break_gold_*")):
        events = [json.loads(line) for line in (episode_dir / "events.jsonl").read_text().splitlines()]
        return_start = next(row["frame_idx"] for row in events if row["event"] == "return_start")
        output_dir = results_root / episode_dir.name / model
        output_dir.mkdir(parents=True)
        for out_idx, frame_path in enumerate(sorted((episode_dir / "frames").glob("*.png"))[return_start:]):
            shutil.copyfile(frame_path, output_dir / f"gen_{out_idx:06d}.png")


def test_make_figure_writes_main_grid_and_prelim(tmp_path):
    episodes_root = tmp_path / "episodes"
    results_root = tmp_path / "results"
    model = "identity"
    record_mock(
        output_root=episodes_root,
        scenario="break_gold",
        n_values=[8],
        pairs=1,
        width=96,
        height=72,
        k1=4,
        k2=8,
    )
    _copy_generated_frames(episodes_root, results_root, model)
    metrics_path = results_root / "readout" / "metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics = {
        "model": model,
        "readout_mode": "template_ssim",
        "calibrations": [
            {
                "scenario": "break_gold",
                "pair_id": "000",
                "N_away": 8,
                "crop_box": [36, 30, 54, 48],
                "vote_offsets": [2, 4, 6],
                "threshold": 12.0,
                "real_frame_accuracy": 1.0,
                "calibration_frames": 6,
            }
        ],
        "per_episode": [
            {
                "episode": "break_gold_000_N8_control",
                "scenario": "break_gold",
                "pair_id": "000",
                "arm": "control",
                "N_away": 8,
                "model": model,
                "target_state": "present",
                "predicted_state": "present",
                "correct_state": 1,
                "generated_frames": 8,
                "voted_frames": 3,
                "crop_box": json.dumps([36, 30, 54, 48]),
            },
            {
                "episode": "break_gold_000_N8_intervene",
                "scenario": "break_gold",
                "pair_id": "000",
                "arm": "intervene",
                "N_away": 8,
                "model": model,
                "target_state": "absent",
                "predicted_state": "absent",
                "correct_state": 1,
                "generated_frames": 8,
                "voted_frames": 3,
                "crop_box": json.dumps([36, 30, 54, 48]),
            },
        ],
        "summary": [
            {"arm": "control", "N_away": 8, "episodes": 1, "p_correct": 1.0},
            {"arm": "intervene", "N_away": 8, "episodes": 1, "p_correct": 1.0},
        ],
        "paired_delta": [
            {"N_away": 8, "p_correct_control": 1.0, "p_correct_intervene": 1.0, "delta_intervene_minus_control": 0.0}
        ],
        "stale_resurrection": [{"N_away": 8, "episodes": 1, "stale_resurrection_rate": 0.0}],
    }
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    outputs = make_figure(
        metrics_path=metrics_path,
        episodes_root=episodes_root,
        results_root=results_root,
        model=model,
        output_dir=tmp_path / "figures",
        config_path=tmp_path / "missing.yaml",
        context_boundary=32,
    )

    assert outputs["main"].exists()
    assert outputs["main"].with_suffix(".pdf").exists()
    assert outputs["grid"].exists()
    prelim = outputs["prelim"].read_text(encoding="utf-8")
    assert "P(correct state)" in prelim
    assert "Stale Resurrection" in prelim
