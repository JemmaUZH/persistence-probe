from __future__ import annotations

import json

from recorder.contact_sheet import render_contact_sheet
from recorder import record as record_mod
from recorder.record import _probe_state_for_frame, _script_action, _to_minerl_action, record_mock
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


class _DummyActionSpace:
    def noop(self):
        return {
            "attack": 0,
            "back": 0,
            "camera": [0.0, 0.0],
            "forward": 0,
            "jump": 0,
            "left": 0,
            "right": 0,
            "sneak": 0,
            "sprint": 0,
            "use": 0,
        }


def test_real_action_adapter_preserves_scripted_intervention_boundary():
    k1 = 4
    look_away = 5
    return_start = 8

    control = [_script_action(idx, k1, look_away, return_start, "control") for idx in range(12)]
    intervene = [_script_action(idx, k1, look_away, return_start, "intervene") for idx in range(12)]

    for idx, (control_action, intervene_action) in enumerate(zip(control, intervene)):
        if idx == k1:
            assert _to_minerl_action(_DummyActionSpace(), intervene_action)["attack"] == 1
            assert control_action != intervene_action
        else:
            assert control_action == intervene_action


def test_probe_state_schedule_matches_control_and_intervention_arms():
    assert [_probe_state_for_frame(idx, 3, "control") for idx in range(5)] == [
        "present",
        "present",
        "present",
        "present",
        "present",
    ]
    assert [_probe_state_for_frame(idx, 3, "intervene") for idx in range(5)] == [
        "present",
        "present",
        "present",
        "absent",
        "absent",
    ]


def test_real_cli_passes_worker_retry_and_timeout_options(monkeypatch, tmp_path):
    captured = {}

    def fake_record_real(**kwargs):
        captured.update(kwargs)
        return [tmp_path / "episode"]

    monkeypatch.setattr(record_mod, "record_real", fake_record_real)

    result = record_mod.main(
        [
            "--scenario",
            "break_gold",
            "--N",
            "16",
            "--pairs",
            "2",
            "--output-root",
            str(tmp_path),
            "--env-id",
            "PersistenceProbeBreakGold-v0",
            "--worker-retries",
            "7",
            "--worker-timeout",
            "11",
        ]
    )

    assert result == 0
    assert captured["worker_retries"] == 7
    assert captured["worker_timeout"] == 11
