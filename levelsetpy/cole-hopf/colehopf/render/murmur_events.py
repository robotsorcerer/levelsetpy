"""Build the static three-event murmuration figure from final GIF frames."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


EVENTS = (
    ("cordon", ("2.0e-03", "1.3e-03", "6.7e-04", "0.0e+00")),
    ("flash", ("6.9e-04", "4.6e-04", "2.3e-04", "0.0e+00")),
    ("vacuole", ("2.0e-03", "1.3e-03", "6.7e-04", "0.0e+00")),
)
TICK_CENTRES = (112, 178, 244, 311)
STRIP_HEIGHT = 390
CANVAS_WIDTH = 960
LABEL_X = 864


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def final_frame(path: Path) -> Image.Image:
    image = Image.open(path)
    image.seek(image.n_frames - 1)
    return image.convert("RGB")


def event_strip(path: Path, labels: tuple[str, ...], font: ImageFont.FreeTypeFont) -> Image.Image:
    frame = final_frame(path)
    if frame.width != 924 or frame.height < STRIP_HEIGHT:
        raise ValueError(f"unexpected animation frame size for {path}: {frame.size}")

    strip = frame.crop((0, 0, frame.width, STRIP_HEIGHT))
    canvas = Image.new("RGB", (CANVAS_WIDTH, STRIP_HEIGHT), "white")
    canvas.paste(strip, (0, 0))
    draw = ImageDraw.Draw(canvas)

    # Remove the original 8-point labels while preserving the colorbar and its tick marks.
    draw.rectangle((862, 100, CANVAS_WIDTH, 322), fill="white")
    for centre, label in zip(TICK_CENTRES, labels):
        draw.text((LABEL_X, centre), label, fill="black", font=font, anchor="lm")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--animation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--font",
        type=Path,
        default=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    )
    args = parser.parse_args()

    font = ImageFont.truetype(str(args.font), 14)
    strips = []
    sources = []
    for event, labels in EVENTS:
        source = args.animation_dir / f"roysoc_murmuration_{event}.gif"
        strips.append(event_strip(source, labels, font))
        sources.append({"event": event, "path": str(source), "sha256": digest(source)})

    output = Image.new("RGB", (CANVAS_WIDTH, STRIP_HEIGHT * len(strips)), "white")
    for index, strip in enumerate(strips):
        output.paste(strip, (0, index * STRIP_HEIGHT))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".png":
        output.save(args.output, optimize=True)
    else:
        output.save(args.output, quality=98, subsampling=0, optimize=True)
    metadata = {
        "sources": sources,
        "crop_height_pixels": STRIP_HEIGHT,
        "canvas_pixels": list(output.size),
        "colorbar_tick_font": str(args.font),
        "colorbar_tick_source_pixels": 14,
        "image_format": args.output.suffix.lower().lstrip("."),
        "figure_sha256": digest(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
