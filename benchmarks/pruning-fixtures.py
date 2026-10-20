"""Generate real shaped masks and deterministic image patches for the Bun POC."""
import argparse
import itertools
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

parser = argparse.ArgumentParser()
parser.add_argument("--font", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
font = ImageFont.truetype(args.font, 16)
width, height = round(font.getlength("MM")), 24
assert width % 2 == 0
alphabet = " .:-=+*#%@/|\\<>!~_"
sequences = ["".join(pair) for pair in itertools.product(alphabet, repeat=2)]
def mask(text, enabled=True):
    image = Image.new("L", (width, height))
    ImageDraw.Draw(image).text((0, 2), text, font=font, fill=255,
                              features=["liga", "calt"] if enabled else ["-liga", "-calt"])
    return list(image.getdata())

masks = [mask(text) for text in sequences]
changed = [text for text, pixels in zip(sequences, masks) if pixels != mask(text, False)]
unique = {}
for text, pixels in zip(sequences, masks):
    unique.setdefault(bytes(pixels), {"text": text, "pixels": pixels})
rng = random.Random(42)
groups = {}
root = Path(__file__).resolve().parents[1]
for name in ["test.png", "taylor.png", "webb-poster.webp"]:
    # Compare coverage masks to required dark-ink coverage on white paper.
    image = ImageOps.invert(Image.open(root / "web/examples" / name).convert("L"))
    image.thumbnail((640, 640))
    groups[name] = [list(image.crop((x, y, x + width, y + height)).getdata())
                    for x, y in [(rng.randrange(image.width - width + 1),
                                  rng.randrange(image.height - height + 1)) for _ in range(300)]]
groups["exact-masks"] = [rng.choice(masks) for _ in range(300)]
groups["flat"] = [[value] * (width * height) for value in range(0, 256, 4)]
groups["noise"] = [[rng.randrange(256) for _ in range(width * height)] for _ in range(300)]
Path(args.output).write_text(json.dumps(dict(width=width, height=height,
    font=args.font, rawCandidates=len(masks), changedSequences=changed,
    candidates=list(unique.values()), groups=groups)))
