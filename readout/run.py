from __future__ import annotations

import argparse
import base64
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from schema.validate import ValidationFailure, validate_episode


class ReadoutError(Exception):
    """Raised when readout inputs are missing or invalid."""


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _episode_dirs(root: Path) -> list[Path]:
    episodes = sorted(path for path in root.iterdir() if path.is_dir() and (path / "meta.json").exists())
    if not episodes:
        raise ReadoutError(f"no episodes found under {root}")
    return episodes


def _return_start(episode_dir: Path) -> int:
    events = _load_jsonl(episode_dir / "events.jsonl")
    matches = [row["frame_idx"] for row in events if row["event"] == "return_start"]
    if len(matches) != 1:
        raise ReadoutError(f"{episode_dir}: expected exactly one return_start event")
    return matches[0]


def _target_state(episode_dir: Path) -> str:
    states = _load_jsonl(episode_dir / "state.jsonl")
    return_start = _return_start(episode_dir)
    return states[return_start]["probe_block_state"]


def _mock_classify(frame_path: Path) -> str:
    with Image.open(frame_path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        crop = rgb.crop((int(width * 0.43), int(height * 0.42), int(width * 0.57), int(height * 0.68)))
        pixels = list(crop.getdata())
    if not pixels:
        return "absent"
    gold_score = statistics.mean((r + g) / 2 - b for r, g, b in pixels)
    return "present" if gold_score > 35 else "absent"


def _majority_vote(labels: list[str]) -> str:
    present = sum(1 for label in labels if label == "present")
    absent = len(labels) - present
    return "present" if present >= absent else "absent"


def run_readout(
    *,
    episodes_root: Path,
    results_root: Path,
    model: str,
    output_dir: Path,
    mock: bool,
    audit_sample: int = 50,
) -> list[dict[str, Any]]:
    if not mock:
        raise ReadoutError("real readout classifier is not implemented yet; pass --mock for placeholder audit/metrics")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    audit_items: list[dict[str, str]] = []

    for episode_dir in _episode_dirs(episodes_root):
        validate_episode(episode_dir)
        meta = _load_json(episode_dir / "meta.json")
        generated_dir = results_root / episode_dir.name / model
        generated_frames = sorted(generated_dir.glob("gen_*.png"))
        if not generated_frames:
            raise ReadoutError(f"missing generated frames: {generated_dir}/gen_*.png")

        frame_predictions = [_mock_classify(path) for path in generated_frames]
        predicted_state = _majority_vote(frame_predictions)
        target_state = _target_state(episode_dir)
        correct = int(predicted_state == target_state)
        row = {
            "episode": episode_dir.name,
            "scenario": meta["scenario"],
            "pair_id": meta["pair_id"],
            "arm": meta["arm"],
            "N_away": meta["N_away"],
            "model": model,
            "mock_readout": True,
            "target_state": target_state,
            "predicted_state": predicted_state,
            "correct_state": correct,
            "generated_frames": len(generated_frames),
        }
        rows.append(row)
        for frame_path, prediction in zip(generated_frames, frame_predictions):
            if len(audit_items) >= audit_sample:
                break
            audit_items.append(
                {
                    "episode": episode_dir.name,
                    "frame": str(frame_path),
                    "prediction": prediction,
                    "target": target_state,
                }
            )

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["arm"], int(row["N_away"]))].append(row)

    summary = []
    for (arm, n_away), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        p_correct = sum(row["correct_state"] for row in group) / len(group)
        summary.append({"arm": arm, "N_away": n_away, "episodes": len(group), "p_correct": p_correct})

    metrics = {
        "mock_readout": True,
        "warning": "Placeholder deterministic crop/color rule for pipeline testing only; not valid experimental evidence.",
        "model": model,
        "per_episode": rows,
        "summary": summary,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    _write_audit_html(output_dir / "audit.html", audit_items)
    return rows


def _write_audit_html(path: Path, items: list[dict[str, str]]) -> None:
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Readout Audit</title>",
        "<style>body{font-family:Arial,sans-serif} .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}.item{border:1px solid #ddd;padding:8px}img{width:100%;image-rendering:pixelated}</style>",
        "<h1>Readout Audit</h1>",
        "<p>Mock placeholder classifier; not experimental evidence.</p>",
        "<div class='grid'>",
    ]
    for item in items:
        frame_path = Path(item["frame"])
        encoded = base64.b64encode(frame_path.read_bytes()).decode("ascii")
        parts.append(
            "<div class='item'>"
            f"<img src='data:image/png;base64,{encoded}'>"
            f"<p><b>{item['episode']}</b></p>"
            f"<p>pred: {item['prediction']} target: {item['target']}</p>"
            "</div>"
        )
    parts.append("</div>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run readout and metrics over generated frames.")
    parser.add_argument("--episodes", type=Path, default=Path("episodes"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--model", default="mock")
    parser.add_argument("--output", type=Path, default=Path("results/readout"))
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--audit-sample", type=int, default=50)
    args = parser.parse_args(argv)

    try:
        rows = run_readout(
            episodes_root=args.episodes,
            results_root=args.results,
            model=args.model,
            output_dir=args.output,
            mock=args.mock,
            audit_sample=args.audit_sample,
        )
    except (ReadoutError, ValidationFailure) as exc:
        print(f"readout failed: {exc}", file=sys.stderr)
        return 1

    print(f"wrote: {args.output / 'metrics.json'}")
    print(f"wrote: {args.output / 'metrics.csv'}")
    print(f"wrote: {args.output / 'audit.html'}")
    print(f"episodes: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
