from __future__ import annotations

import json

from recorder.contact_sheet import render_contact_sheet
from recorder.record import record_mock
from schema.validate import validate_episode


def test_record_mock_writes_valid_paired_episodes_and_contact_sheet(tmp_path):
    episodes = record_mock(
        output_root=tmp_path / "episodes",
        scenario="break_gold",
        n_values=[8],
        pairs=1,
        width=64,
        height=64,
        k1=4,
        k2=6,
    )

    assert len(episodes) == 2
    for episode in episodes:
        validate_episode(episode)
        assert render_contact_sheet(episode).exists()

    control = next(path for path in episodes if path.name.endswith("_control"))
    intervene = next(path for path in episodes if path.name.endswith("_intervene"))
    control_actions = [json.loads(line) for line in (control / "actions.jsonl").read_text().splitlines()]
    intervene_actions = [json.loads(line) for line in (intervene / "actions.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (control / "events.jsonl").read_text().splitlines()]
    intervention_frame = next(row["frame_idx"] for row in events if row["event"] == "intervention")

    for idx, (control_action, intervene_action) in enumerate(zip(control_actions, intervene_actions)):
        if idx == intervention_frame:
            assert control_action != intervene_action
        else:
            assert control_action == intervene_action
