from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


class FigureError(Exception):
    """Raised when Figure 1 inputs are missing or inconsistent."""


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_context_boundary(config_path: Path) -> int | None:
    if not config_path.exists():
        return None
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("max_context_frames:"):
            try:
                return int(stripped.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _summary_from_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    if metrics.get("summary"):
        by_key = {(row["arm"], int(row["N_away"])): dict(row) for row in metrics["summary"]}
        for row in metrics.get("per_episode", []):
            key = (row["arm"], int(row["N_away"]))
            if key in by_key:
                by_key[key]["successes"] = by_key[key].get("successes", 0) + int(row["correct_state"])
        return list(by_key.values())

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in metrics.get("per_episode", []):
        grouped[(row["arm"], int(row["N_away"]))].append(row)
    summary = []
    for (arm, n_away), rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        successes = sum(int(row["correct_state"]) for row in rows)
        summary.append(
            {
                "arm": arm,
                "N_away": n_away,
                "episodes": len(rows),
                "p_correct": successes / len(rows),
                "successes": successes,
            }
        )
    return summary


def _plot_main(metrics: dict[str, Any], output_path: Path, *, context_boundary: int | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _summary_from_rows(metrics)
    if not summary:
        raise FigureError("metrics contain no summary/per_episode rows")

    colors = {"control": "#2f6f9f", "intervene": "#b2453a"}
    labels = {"control": "Control", "intervene": "Intervene"}
    fig, ax = plt.subplots(figsize=(3.35, 2.25), dpi=220)

    x_values_all: list[int] = []
    for arm in ("control", "intervene"):
        rows = sorted((row for row in summary if row["arm"] == arm), key=lambda row: int(row["N_away"]))
        if not rows:
            continue
        xs = [int(row["N_away"]) for row in rows]
        ys = [float(row["p_correct"]) for row in rows]
        lowers = []
        uppers = []
        for row, y in zip(rows, ys):
            total = int(row["episodes"])
            successes = int(row.get("successes", round(y * total)))
            low, high = _wilson_interval(successes, total)
            lowers.append(y - low)
            uppers.append(high - y)
        x_values_all.extend(xs)
        ax.errorbar(
            xs,
            ys,
            yerr=[lowers, uppers],
            marker="o",
            markersize=4,
            linewidth=1.3,
            capsize=2.5,
            color=colors.get(arm),
            label=labels.get(arm, arm),
        )

    if context_boundary is not None:
        ax.axvline(context_boundary, color="#555555", linewidth=0.9, linestyle="--")
        ax.text(context_boundary, 0.03, "context", rotation=90, va="bottom", ha="right", fontsize=7, color="#555555")
        x_values_all.append(context_boundary)

    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("Look-away duration N (frames)")
    ax.set_ylabel("P(correct state)")
    ax.set_title("Persistence Probe Readout", fontsize=9)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower left", fontsize=7)

    if x_values_all:
        xmin = max(1, min(x_values_all))
        xmax = max(x_values_all)
        if xmin == xmax:
            xmin = max(1, xmin // 2)
            xmax = xmax * 2
        ax.set_xlim(xmin * 0.75, xmax * 1.35)
    fig.tight_layout(pad=0.5)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def _read_events(episode_dir: Path) -> list[dict[str, Any]]:
    with (episode_dir / "events.jsonl").open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _return_start(episode_dir: Path) -> int:
    matches = [row["frame_idx"] for row in _read_events(episode_dir) if row["event"] == "return_start"]
    if len(matches) != 1:
        raise FigureError(f"{episode_dir}: expected exactly one return_start event")
    return int(matches[0])


def _image_or_blank(path: Path, size: tuple[int, int]) -> Image.Image:
    if not path.exists():
        return Image.new("RGB", size, (238, 238, 238))
    with Image.open(path) as img:
        return img.convert("RGB")


def _fit_image(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    fitted = Image.new("RGB", size, (255, 255, 255))
    img = img.copy()
    img.thumbnail(size, resample)
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    fitted.paste(img, (x, y))
    return fitted


def _draw_scaled_box(
    draw: ImageDraw.ImageDraw,
    source_size: tuple[int, int],
    cell_origin: tuple[int, int],
    cell_size: tuple[int, int],
    crop_box: list[int] | None,
) -> None:
    if not crop_box:
        return
    src_w, src_h = source_size
    scale = min(cell_size[0] / src_w, cell_size[1] / src_h)
    fitted_w = src_w * scale
    fitted_h = src_h * scale
    pad_x = cell_origin[0] + (cell_size[0] - fitted_w) / 2
    pad_y = cell_origin[1] + (cell_size[1] - fitted_h) / 2
    x1, y1, x2, y2 = crop_box
    box = [
        int(pad_x + x1 * scale),
        int(pad_y + y1 * scale),
        int(pad_x + x2 * scale),
        int(pad_y + y2 * scale),
    ]
    draw.rectangle(box, outline=(220, 32, 32), width=3)


def _episode_row(metrics: dict[str, Any], n_away: int, arm: str) -> dict[str, Any] | None:
    rows = [
        row
        for row in metrics.get("per_episode", [])
        if int(row["N_away"]) == n_away and row["arm"] == arm
    ]
    return sorted(rows, key=lambda row: row["episode"])[0] if rows else None


def _crop_box(row: dict[str, Any] | None) -> list[int] | None:
    if not row:
        return None
    raw = row.get("crop_box")
    if not raw:
        return None
    if isinstance(raw, list):
        return [int(value) for value in raw]
    return [int(value) for value in json.loads(raw)]


def _plot_grid(metrics: dict[str, Any], episodes_root: Path, results_root: Path, model: str, output_path: Path) -> None:
    n_values = sorted({int(row["N_away"]) for row in metrics.get("per_episode", [])})
    if not n_values:
        raise FigureError("metrics contain no per_episode rows for grid")

    cell_size = (240, 135)
    label_h = 28
    row_label_w = 68
    gap = 10
    columns = ["real return", "generated control", "generated intervene"]
    width = row_label_w + len(columns) * cell_size[0] + (len(columns) + 1) * gap
    height = label_h + len(n_values) * (cell_size[1] + label_h + gap) + gap
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for col_idx, label in enumerate(columns):
        x = row_label_w + gap + col_idx * (cell_size[0] + gap)
        draw.text((x, 8), label, fill=(20, 20, 20), font=font)

    for row_idx, n_away in enumerate(n_values):
        y = label_h + gap + row_idx * (cell_size[1] + label_h + gap)
        draw.text((8, y + cell_size[1] // 2 - 6), f"N={n_away}", fill=(20, 20, 20), font=font)
        control = _episode_row(metrics, n_away, "control")
        intervene = _episode_row(metrics, n_away, "intervene")
        if control is None or intervene is None:
            continue
        control_episode = episodes_root / control["episode"]
        return_start = _return_start(control_episode)
        vote_offset = 0
        calibrations = [
            row
            for row in metrics.get("calibrations", [])
            if int(row["N_away"]) == n_away and row["pair_id"] == control["pair_id"]
        ]
        if calibrations and calibrations[0].get("vote_offsets"):
            offsets = calibrations[0]["vote_offsets"]
            vote_offset = int(offsets[len(offsets) // 2])

        image_paths = [
            control_episode / "frames" / f"{return_start + vote_offset:06d}.png",
            results_root / control["episode"] / model / f"gen_{vote_offset:06d}.png",
            results_root / intervene["episode"] / model / f"gen_{vote_offset:06d}.png",
        ]
        boxes = [_crop_box(control), _crop_box(control), _crop_box(intervene)]
        captions = [
            f"target {control['target_state']}",
            f"pred {control['predicted_state']}",
            f"pred {intervene['predicted_state']}",
        ]

        for col_idx, path in enumerate(image_paths):
            x = row_label_w + gap + col_idx * (cell_size[0] + gap)
            img = _image_or_blank(path, cell_size)
            source_size = img.size
            fitted = _fit_image(img, cell_size)
            canvas.paste(fitted, (x, y))
            _draw_scaled_box(draw, source_size, (x, y), cell_size, boxes[col_idx])
            draw.rectangle((x, y, x + cell_size[0], y + cell_size[1]), outline=(210, 210, 210), width=1)
            draw.text((x, y + cell_size[1] + 7), captions[col_idx], fill=(40, 40, 40), font=font)

    canvas.save(output_path)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_prelim(metrics: dict[str, Any], output_path: Path, *, figure_path: Path, grid_path: Path) -> None:
    summary = _summary_from_rows(metrics)
    paired = metrics.get("paired_delta", [])
    stale = metrics.get("stale_resurrection", [])
    calibrations = metrics.get("calibrations", [])
    lines = [
        "# Preliminary Results",
        "",
        "Generated automatically from readout metrics. Human narrative goes here.",
        "",
        f"- Model: `{metrics.get('model', '')}`",
        f"- Readout mode: `{metrics.get('readout_mode', '')}`",
        f"- Main figure: `{figure_path}`",
        f"- Qualitative grid: `{grid_path}`",
        "",
        "## P(correct state)",
        "",
        _markdown_table(summary, ["N_away", "arm", "episodes", "p_correct"]),
        "",
        "## Paired Delta",
        "",
        _markdown_table(paired, ["N_away", "p_correct_control", "p_correct_intervene", "delta_intervene_minus_control"]),
        "",
        "## Stale Resurrection",
        "",
        _markdown_table(stale, ["N_away", "episodes", "stale_resurrection_rate"]),
        "",
        "## Readout Calibration",
        "",
        _markdown_table(calibrations, ["N_away", "pair_id", "real_frame_accuracy", "calibration_frames", "vote_offsets"]),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def make_figure(
    *,
    metrics_path: Path,
    episodes_root: Path,
    results_root: Path,
    model: str,
    output_dir: Path,
    config_path: Path,
    context_boundary: int | None,
) -> dict[str, Path]:
    if not metrics_path.exists():
        raise FigureError(f"metrics file does not exist: {metrics_path}")
    metrics = _load_json(metrics_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    boundary = context_boundary if context_boundary is not None else _parse_context_boundary(config_path)

    main_path = output_dir / "figure1_main.png"
    grid_path = output_dir / "figure1_grid.png"
    prelim_path = output_dir / "PRELIM.md"
    _plot_main(metrics, main_path, context_boundary=boundary)
    _plot_grid(metrics, episodes_root, results_root, model, grid_path)
    _write_prelim(metrics, prelim_path, figure_path=main_path, grid_path=grid_path)
    return {"main": main_path, "grid": grid_path, "prelim": prelim_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Figure 1 assets from readout metrics.")
    parser.add_argument("--metrics", type=Path, default=Path("results/readout_oasis/metrics.json"))
    parser.add_argument("--episodes", type=Path, default=Path("episodes"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--model", default="oasis")
    parser.add_argument("--output", type=Path, default=Path("figures"))
    parser.add_argument("--config", type=Path, default=Path("configs/oasis.yaml"))
    parser.add_argument("--context-boundary", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        outputs = make_figure(
            metrics_path=args.metrics,
            episodes_root=args.episodes,
            results_root=args.results,
            model=args.model,
            output_dir=args.output,
            config_path=args.config,
            context_boundary=args.context_boundary,
        )
    except FigureError as exc:
        print(f"figure failed: {exc}", file=sys.stderr)
        return 1

    for output in outputs.values():
        print(f"wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
