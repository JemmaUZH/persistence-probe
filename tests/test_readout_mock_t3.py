from __future__ import annotations

import json

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
