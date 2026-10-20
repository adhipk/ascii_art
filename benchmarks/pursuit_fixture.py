"""Create a real-font native fixture for the TypeScript search harness.

PYTHONPATH=src uv run --no-sync python benchmarks/pursuit_fixture.py
bun benchmarks/pursuit_run.ts /tmp/glyph-pursuit-fixture.json

Uses bundled Fira Code through Pillow; not a browser-font visual test.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CHARACTERS = json.loads((ROOT / "web/pursuit-dictionary.json").read_text())[
    "characters"
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "web/examples/taylor.png")
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/glyph-pursuit-fixture.json")
    )
    parser.add_argument("--characters", default=CHARACTERS)
    args = parser.parse_args()
    characters = args.characters
    if not characters or any(ord(c) < 32 or ord(c) > 126 for c in characters):
        raise ValueError("Characters must be printable ASCII")
    image = Image.open(args.image).convert("RGBA")
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    font = ImageFont.truetype(str(ROOT / "fonts/FiraCode-Regular.ttf"), 61)
    ascent, descent = font.getmetrics()
    baseline, advance = (64 - ascent - descent) / 2 + ascent, font.getlength("M")
    masks = []
    for char in characters:
        cell = Image.new("L", (64, 64))
        ImageDraw.Draw(cell).text(
            ((64 - advance) / 2, baseline), char, font=font, anchor="ls", fill=255
        )
        masks.append((np.asarray(cell, np.float32) / 255).ravel().tolist())
    metric = {
        "width": 64,
        "height": 64,
        "baseline": baseline,
        "advance": advance,
        "fontSize": 61,
        "densities": [],
        "kernelCharacters": characters,
        "masks": masks,
    }
    data = {
        "width": image.width,
        "height": image.height,
        "pixels": np.asarray(image).ravel().tolist(),
        "settings": {
            "columns": 80,
            "font_size": 12,
            "layout": "pursuit",
            "use_edge": True,
            "edge_method": "acerola",
            "edge_sensitivity": 50,
            "edge_amount": 100,
            "use_color": True,
            "background": "#ffffff",
            "background_mode": "source",
            "foreground": "#242424",
            "pursuit_metrics": metric,
            "pursuit_beam": 2,
            "pursuit_edge_weight": 3,
        },
    }
    args.output.write_text(json.dumps(data))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
