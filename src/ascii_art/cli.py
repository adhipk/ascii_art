"""Command-line image and plain-text ASCII conversion."""
import argparse
from pathlib import Path
from typing import Sequence

import cv2

from ascii_art.generators.sprite_generator import SpriteASCIIGenerator


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate DoG-guided ASCII art from an image."
    )
    parser.add_argument("image_path")
    parser.add_argument("-o", "--output", default="result.png")
    parser.add_argument("--font-size", type=int, default=12)
    parser.add_argument("--font-path", help="Optional TrueType/OpenType font file")
    parser.add_argument("--columns", type=int, help="Output width in characters")
    parser.add_argument("--no-edge", action="store_true")
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=70,
        help="Edge strength percentile, 0–100",
    )
    parser.add_argument(
        "--color", action="store_true", help="Use average source color per character"
    )
    parser.add_argument("--foreground", default="#242424")
    parser.add_argument("--background", default="#FFFFFF")
    parser.add_argument(
        "--text", action="store_true", help="Write a plain-text character grid"
    )
    args = parser.parse_args(argv)
    img = cv2.imread(args.image_path)
    if img is None:
        parser.error(f"could not load image: {args.image_path}")
    try:
        generator = SpriteASCIIGenerator(
            {
                "font_size": args.font_size,
                "font_path": args.font_path,
                "columns": args.columns,
                "use_edge": not args.no_edge,
                "edge_threshold": args.edge_threshold,
                "use_color": args.color,
                "foreground": args.foreground,
                "background": args.background,
            }
        )
        if args.text:
            Path(args.output).write_text(
                generator.generate_text(img) + "\n", encoding="utf-8"
            )
        elif not cv2.imwrite(args.output, generator.generate_ascii(img)):
            parser.error(f"could not save image: {args.output}")
    except (ValueError, OSError, cv2.error) as exc:
        parser.error(str(exc))
    print(f"ASCII art saved to {args.output}")


if __name__ == "__main__":
    main()
