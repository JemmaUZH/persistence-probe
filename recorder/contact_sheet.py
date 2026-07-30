from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def _load_events(episode_dir: Path) -> dict[str, int]:
    events = {}
    with (episode_dir / "events.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            events[row["event"]] = row["frame_idx"]
    return events


def render_contact_sheet(episode_dir: str | Path, output_path: str | Path | None = None) -> Path:
    episode_path = Path(episode_dir)
    events = _load_events(episode_path)
    frame_indices = [
        max(0, events["intervention"] - 1),
        events["intervention"],
        events["look_away"],
        events["return_start"],
    ]
    labels = ["pre", "intervention", "look_away", "return_start"]
    frames = [Image.open(episode_path / "frames" / f"{idx:06d}.png").convert("RGB") for idx in frame_indices]
    thumb_w, thumb_h = frames[0].size
    sheet = Image.new("RGB", (thumb_w * len(frames), thumb_h + 28), "white")
    draw = ImageDraw.Draw(sheet)
    for col, (label, frame) in enumerate(zip(labels, frames)):
        x = col * thumb_w
        sheet.paste(frame, (x, 0))
        draw.text((x + 8, thumb_h + 8), f"{label} f={frame_indices[col]}", fill=(0, 0, 0))

    out = Path(output_path) if output_path is not None else episode_path / "contact_sheet.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render key-frame contact sheet for an episode.")
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    out = render_contact_sheet(args.episode_dir, args.output)
    print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
