from __future__ import annotations

import argparse
import base64
import csv
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw

from schema.validate import ValidationFailure, validate_episode


class ReadoutError(Exception):
    """Raised when readout inputs are missing or invalid."""


@dataclass(frozen=True)
class TemplateReadout:
    crop_box: tuple[int, int, int, int]
    threshold: float
    real_frame_accuracy: float
    calibration_frames: int
    present_refs: tuple[Image.Image, ...]
    absent_refs: tuple[Image.Image, ...]

    def classify(self, frame_path: Path) -> tuple[str, float]:
        with Image.open(frame_path) as img:
            crop = _normalized_crop(img, self.crop_box)
        present_distance = min(_mean_abs_distance(crop, ref) for ref in self.present_refs)
        absent_distance = min(_mean_abs_distance(crop, ref) for ref in self.absent_refs)
        score = absent_distance - present_distance
        return ("present" if score >= self.threshold else "absent"), score


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


def _return_frame_indices(episode_dir: Path) -> list[int]:
    frames = sorted((episode_dir / "frames").glob("*.png"))
    return list(range(_return_start(episode_dir), len(frames)))


def _target_state(episode_dir: Path) -> str:
    states = _load_jsonl(episode_dir / "state.jsonl")
    return_start = _return_start(episode_dir)
    return states[return_start]["probe_block_state"]


def _state_at_frame(episode_dir: Path, frame_idx: int) -> str:
    states = _load_jsonl(episode_dir / "state.jsonl")
    if frame_idx >= len(states):
        raise ReadoutError(f"{episode_dir}: missing state row for frame {frame_idx}")
    return states[frame_idx]["probe_block_state"]


def _frame_path(episode_dir: Path, frame_idx: int) -> Path:
    path = episode_dir / "frames" / f"{frame_idx:06d}.png"
    if not path.exists():
        raise ReadoutError(f"missing frame: {path}")
    return path


def _gold_pixel_score(pixel: tuple[int, int, int]) -> float:
    red, green, blue = pixel
    yellow = max(0.0, (red + green) / 2.0 - blue)
    saturation = max(0.0, min(red, green) - blue)
    balance_penalty = abs(red - green) * 0.15
    return yellow + saturation - balance_penalty


def _gold_score(img: Image.Image) -> float:
    pixels = list(img.convert("RGB").getdata())
    if not pixels:
        return float("-inf")
    return statistics.mean(_gold_pixel_score(pixel) for pixel in pixels)


def _candidate_crop_sizes(width: int, height: int) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for width_frac, height_frac in ((0.07, 0.12), (0.10, 0.16), (0.14, 0.22), (0.18, 0.28)):
        crop_width = max(8, int(width * width_frac))
        crop_height = max(8, int(height * height_frac))
        sizes.append((crop_width, crop_height))
    return sizes


def _find_probe_crop_box(frame_path: Path) -> tuple[int, int, int, int]:
    with Image.open(frame_path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size

        best_score = float("-inf")
        best_box = (int(width * 0.43), int(height * 0.42), int(width * 0.57), int(height * 0.68))
        for crop_width, crop_height in _candidate_crop_sizes(width, height):
            step_x = max(4, crop_width // 3)
            step_y = max(4, crop_height // 3)
            x_min = max(0, int(width * 0.15))
            x_max = min(width - crop_width, int(width * 0.85))
            y_min = max(0, int(height * 0.10))
            y_max = min(height - crop_height, int(height * 0.85))
            for y in range(y_min, y_max + 1, step_y):
                for x in range(x_min, x_max + 1, step_x):
                    box = (x, y, x + crop_width, y + crop_height)
                    center_bias = abs((x + crop_width / 2) - width / 2) / width
                    score = _gold_score(rgb.crop(box)) - center_bias * 5.0
                    if score > best_score:
                        best_score = score
                        best_box = box
    return best_box


def _localized_crop_box(control_episode: Path) -> tuple[int, int, int, int]:
    candidates = [
        _frame_path(control_episode, frame_idx)
        for frame_idx in _return_frame_indices(control_episode)
        if _state_at_frame(control_episode, frame_idx) == "present"
    ]
    if not candidates:
        raise ReadoutError(f"{control_episode}: no present return frames available for probe localization")
    return _find_probe_crop_box(candidates[0])


def _normalized_crop(img: Image.Image, crop_box: tuple[int, int, int, int]) -> Image.Image:
    resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    return img.convert("RGB").crop(crop_box).resize((32, 32), resample)


def _mean_abs_distance(left: Image.Image, right: Image.Image) -> float:
    diff = ImageChops.difference(left.convert("RGB"), right.convert("RGB"))
    pixels = list(diff.getdata())
    return statistics.mean((red + green + blue) / 3.0 for red, green, blue in pixels)


def _score_crop(crop: Image.Image, present_refs: list[Image.Image], absent_refs: list[Image.Image]) -> float:
    present_distance = min(_mean_abs_distance(crop, ref) for ref in present_refs)
    absent_distance = min(_mean_abs_distance(crop, ref) for ref in absent_refs)
    return absent_distance - present_distance


def _choose_threshold(scores: list[tuple[float, str]]) -> float:
    if not scores:
        raise ReadoutError("cannot choose readout threshold without labeled scores")
    values = sorted(score for score, _label in scores)
    candidates = [values[0] - 1.0, values[-1] + 1.0]
    candidates.extend((left + right) / 2.0 for left, right in zip(values, values[1:]))

    best_threshold = candidates[0]
    best_accuracy = -1.0
    for threshold in candidates:
        correct = sum(1 for score, label in scores if ("present" if score >= threshold else "absent") == label)
        accuracy = correct / len(scores)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    return best_threshold


def _labeled_return_crops(episode_dir: Path, crop_box: tuple[int, int, int, int]) -> list[tuple[Image.Image, str]]:
    rows: list[tuple[Image.Image, str]] = []
    for frame_idx in _return_frame_indices(episode_dir):
        frame_path = _frame_path(episode_dir, frame_idx)
        with Image.open(frame_path) as img:
            rows.append((_normalized_crop(img, crop_box), _state_at_frame(episode_dir, frame_idx)))
    return rows


def _build_template_readout(control_episode: Path, intervene_episode: Path) -> TemplateReadout:
    crop_box = _localized_crop_box(control_episode)
    labeled = _labeled_return_crops(control_episode, crop_box) + _labeled_return_crops(intervene_episode, crop_box)
    if len({label for _crop, label in labeled}) != 2:
        raise ReadoutError(f"{control_episode.parent}: real-frame calibration needs both present and absent labels")

    train = [row for idx, row in enumerate(labeled) if idx % 2 == 0]
    heldout = [row for idx, row in enumerate(labeled) if idx % 2 == 1] or train
    present_refs = [crop for crop, label in train if label == "present"]
    absent_refs = [crop for crop, label in train if label == "absent"]
    if not present_refs or not absent_refs:
        present_refs = [crop for crop, label in labeled if label == "present"]
        absent_refs = [crop for crop, label in labeled if label == "absent"]

    threshold = _choose_threshold([(_score_crop(crop, present_refs, absent_refs), label) for crop, label in train])
    correct = sum(
        1
        for crop, label in heldout
        if ("present" if _score_crop(crop, present_refs, absent_refs) >= threshold else "absent") == label
    )
    accuracy = correct / len(heldout)
    if accuracy < 0.95:
        raise ReadoutError(
            f"real-frame classifier accuracy {accuracy:.3f} below 0.950 for "
            f"{control_episode.name}/{intervene_episode.name}; crop_box={crop_box}"
        )

    return TemplateReadout(
        crop_box=crop_box,
        threshold=threshold,
        real_frame_accuracy=accuracy,
        calibration_frames=len(labeled),
        present_refs=tuple(present_refs),
        absent_refs=tuple(absent_refs),
    )


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


def _episode_groups(episodes: list[Path]) -> dict[tuple[str, str, int], dict[str, Path]]:
    groups: dict[tuple[str, str, int], dict[str, Path]] = defaultdict(dict)
    for episode_dir in episodes:
        meta = _load_json(episode_dir / "meta.json")
        key = (meta["scenario"], meta["pair_id"], int(meta["N_away"]))
        groups[key][meta["arm"]] = episode_dir
    return groups


def _summaries(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["arm"], int(row["N_away"]))].append(row)

    summary = []
    for (arm, n_away), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        p_correct = sum(row["correct_state"] for row in group) / len(group)
        summary.append({"arm": arm, "N_away": n_away, "episodes": len(group), "p_correct": p_correct})

    by_n: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_n[int(row["N_away"])][row["arm"]].append(row)

    paired_delta = []
    stale_resurrection = []
    for n_away, arms in sorted(by_n.items()):
        control = arms.get("control", [])
        intervene = arms.get("intervene", [])
        if control and intervene:
            p_control = sum(row["correct_state"] for row in control) / len(control)
            p_intervene = sum(row["correct_state"] for row in intervene) / len(intervene)
            paired_delta.append(
                {
                    "N_away": n_away,
                    "p_correct_control": p_control,
                    "p_correct_intervene": p_intervene,
                    "delta_intervene_minus_control": p_intervene - p_control,
                }
            )
        if intervene:
            stale = sum(1 for row in intervene if row["predicted_state"] == "present") / len(intervene)
            stale_resurrection.append({"N_away": n_away, "episodes": len(intervene), "stale_resurrection_rate": stale})

    return summary, paired_delta, stale_resurrection


def run_readout(
    *,
    episodes_root: Path,
    results_root: Path,
    model: str,
    output_dir: Path,
    mock: bool,
    audit_sample: int = 50,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    audit_items: list[dict[str, Any]] = []
    episodes = _episode_dirs(episodes_root)
    episode_to_readout: dict[str, TemplateReadout] = {}
    calibrations: list[dict[str, Any]] = []

    if not mock:
        for key, arms in _episode_groups(episodes).items():
            if "control" not in arms or "intervene" not in arms:
                raise ReadoutError(f"real readout needs paired control/intervene episodes for {key}")
            template = _build_template_readout(arms["control"], arms["intervene"])
            episode_to_readout[arms["control"].name] = template
            episode_to_readout[arms["intervene"].name] = template
            scenario, pair_id, n_away = key
            calibrations.append(
                {
                    "scenario": scenario,
                    "pair_id": pair_id,
                    "N_away": n_away,
                    "crop_box": list(template.crop_box),
                    "threshold": template.threshold,
                    "real_frame_accuracy": template.real_frame_accuracy,
                    "calibration_frames": template.calibration_frames,
                }
            )

    for episode_dir in episodes:
        validate_episode(episode_dir)
        meta = _load_json(episode_dir / "meta.json")
        generated_dir = results_root / episode_dir.name / model
        generated_frames = sorted(generated_dir.glob("gen_*.png"))
        if not generated_frames:
            raise ReadoutError(f"missing generated frames: {generated_dir}/gen_*.png")

        frame_scores: list[float | None] = []
        if mock:
            frame_predictions = [_mock_classify(path) for path in generated_frames]
            frame_scores = [None for _path in generated_frames]
            crop_box = None
            real_frame_accuracy = None
        else:
            template = episode_to_readout[episode_dir.name]
            predictions_and_scores = [template.classify(path) for path in generated_frames]
            frame_predictions = [prediction for prediction, _score in predictions_and_scores]
            frame_scores = [score for _prediction, score in predictions_and_scores]
            crop_box = list(template.crop_box)
            real_frame_accuracy = template.real_frame_accuracy

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
            "mock_readout": mock,
            "readout_mode": "mock_color" if mock else "template_ssim",
            "target_state": target_state,
            "predicted_state": predicted_state,
            "correct_state": correct,
            "generated_frames": len(generated_frames),
            "crop_box": json.dumps(crop_box) if crop_box is not None else "",
            "real_frame_accuracy": real_frame_accuracy if real_frame_accuracy is not None else "",
        }
        rows.append(row)
        for frame_path, prediction, score in zip(generated_frames, frame_predictions, frame_scores):
            if len(audit_items) >= audit_sample:
                break
            audit_items.append(
                {
                    "episode": episode_dir.name,
                    "frame": str(frame_path),
                    "prediction": prediction,
                    "target": target_state,
                    "score": score,
                    "crop_box": crop_box,
                }
            )

    summary, paired_delta, stale_resurrection = _summaries(rows)

    metrics = {
        "mock_readout": mock,
        "readout_mode": "mock_color" if mock else "template_ssim",
        "warning": "Placeholder deterministic crop/color rule for pipeline testing only; not valid experimental evidence."
        if mock
        else "",
        "model": model,
        "calibrations": calibrations,
        "per_episode": rows,
        "summary": summary,
        "paired_delta": paired_delta,
        "stale_resurrection": stale_resurrection,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    _write_audit_html(output_dir / "audit.html", audit_items)
    return rows


def _image_data_uri(img: Image.Image) -> str:
    import io

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _write_audit_html(path: Path, items: list[dict[str, Any]]) -> None:
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Readout Audit</title>",
        "<style>body{font-family:Arial,sans-serif} .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}.item{border:1px solid #ddd;padding:8px}img{width:100%;image-rendering:pixelated}.crop{width:96px}</style>",
        "<h1>Readout Audit</h1>",
        "<div class='grid'>",
    ]
    for item in items:
        frame_path = Path(item["frame"])
        with Image.open(frame_path) as raw_img:
            img = raw_img.convert("RGB")
            crop_uri = ""
            crop_box = item.get("crop_box")
            if crop_box is not None:
                box = tuple(int(value) for value in crop_box)
                draw = ImageDraw.Draw(img)
                draw.rectangle(box, outline=(255, 0, 0), width=max(2, img.size[0] // 160))
                crop_uri = _image_data_uri(raw_img.convert("RGB").crop(box).resize((96, 96)))
            frame_uri = _image_data_uri(img)
        score = item.get("score")
        score_text = "" if score is None else f"<p>score: {float(score):.3f}</p>"
        crop_html = "" if not crop_uri else f"<img class='crop' src='{crop_uri}'>"
        parts.append(
            "<div class='item'>"
            f"<img src='{frame_uri}'>"
            f"{crop_html}"
            f"<p><b>{item['episode']}</b></p>"
            f"<p>pred: {item['prediction']} target: {item['target']}</p>"
            f"{score_text}"
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
