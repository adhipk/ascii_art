"""Rasterize every recorded native iteration and build an offline scrubber.

uv run --no-sync python benchmarks/pursuit_preview.py
Produces /tmp/glyph-pursuit-taylor/index.html; this is native evidence, not browser QA.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", type=Path, default=Path("/tmp/glyph-pursuit-fixture.json")
    )
    parser.add_argument("--run", type=Path, default=Path("/tmp/glyph-pursuit-taylor"))
    parser.add_argument(
        "--last-only",
        action="store_true",
        help="Render only the final frame for quick quality comparisons",
    )
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    result = json.loads((args.run / "trace.json").read_text())
    trace, metric = result["pursuit"], fixture["settings"]["pursuit_metrics"]
    frames = trace["frames"][-1:] if args.last_only else trace["frames"]
    width, height = trace["analysisWidth"], trace["analysisHeight"]
    factor = result["width"] / width
    base = np.array(result["paper"][:3], np.float32) / 255
    masks = {}
    sizes = {
        round(glyph["size"] / factor) for frame in frames for glyph in frame["glyphs"]
    }
    for size in sizes:
        for char, flat in zip(metric["kernelCharacters"], metric["masks"]):
            original = np.array(flat, np.float32).reshape(
                metric["height"], metric["width"]
            )
            for angle in (0, 90):
                source = original if angle == 0 else np.rot90(original, -1)
                mask = np.zeros((size, size), np.float32)
                for y in range(size):
                    for x in range(size):
                        x0, x1 = int(x * metric["width"] / size), int(
                            np.ceil((x + 1) * metric["width"] / size)
                        )
                        y0, y1 = int(y * metric["height"] / size), int(
                            np.ceil((y + 1) * metric["height"] / size)
                        )
                        mask[y, x] = source[y0:y1, x0:x1].mean()
                masks[char, size, angle] = mask
    original = Image.fromarray(
        np.array(fixture["pixels"], np.uint8).reshape(
            fixture["height"], fixture["width"], 4
        )
    )
    original.convert("RGB").resize(
        (width * 2, height * 2), Image.Resampling.LANCZOS
    ).save(args.run / "original.png")
    for frame in frames:
        pixels = np.broadcast_to(base, (height, width, 3)).copy()
        for glyph in frame["glyphs"]:
            x, y, size = (round(glyph[key] / factor) for key in ("x", "y", "size"))
            alpha = masks[glyph["char"], size, glyph.get("angle", 0)][..., None]
            region = pixels[y : y + size, x : x + size]
            region[:] = region * (1 - alpha) + np.array(glyph["color"]) / 255 * alpha
        Image.fromarray(np.rint(pixels * 255).clip(0, 255).astype(np.uint8)).resize(
            (width * 2, height * 2), Image.Resampling.NEAREST
        ).save(args.run / f"frame-{frame['iteration']}.png")
    # Self-contained controls, relative PNGs, no external scripts or runtime.
    metadata = [
        {
            k: frame[k]
            for k in (
                "iteration",
                "energy",
                "pixelError",
                "contourError",
                "diversity",
                "action",
            )
        }
        for frame in frames
    ]
    page = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Glyph search iteration lab — native Taylor run</title>
<style>body{font:15px system-ui;background:#17202a;color:#edf2f6;margin:24px}main{max-width:1000px;margin:auto}.images{display:flex;gap:16px}.images figure{margin:0;flex:1;min-width:0}img{width:100%;height:auto}input{width:100%;margin:24px 0}button{padding:8px 16px;margin-right:8px}pre{white-space:pre-wrap}small{color:#b3c3d4}@media(max-width:500px){body{margin:12px}}</style>
<main><h1>Glyph search iteration lab</h1><p>Native Fira Code run. Left: original. Right: recorded reconstruction.</p>
<div class="images"><figure><figcaption>Original</figcaption><img src="original.png" alt="Original Taylor source"></figure><figure><figcaption id="label">Reconstruction</figcaption><img id="frame" alt="Reconstruction at selected iteration"></figure></div>
<label for="iteration">Iteration</label><input id="iteration" type="range" min="0" step="1">
<button id="prev">Previous</button><button id="play">Play</button><button id="next">Next</button>
<pre id="metrics"></pre><small>Every image is rendered from the actual saved composition. This native view does not validate system-font SVG rendering.</small></main>
<script>const frames=METADATA;const slider=document.getElementById('iteration');let timer;slider.max=frames.length-1;slider.value=slider.max;
function show(){const f=frames[Number(slider.value)];document.getElementById('frame').src='frame-'+f.iteration+'.png';document.getElementById('label').textContent='Iteration '+f.iteration+' / '+slider.max;document.getElementById('metrics').textContent=JSON.stringify(f,null,2);}
function pause(){clearInterval(timer);timer=undefined;document.getElementById('play').textContent='Play';}
slider.oninput=()=>{pause();show();};document.getElementById('prev').onclick=()=>{pause();slider.value=Math.max(0,Number(slider.value)-1);show();};document.getElementById('next').onclick=()=>{pause();slider.value=Math.min(Number(slider.max),Number(slider.value)+1);show();};
document.getElementById('play').onclick=()=>{if(timer){pause();return;}if(slider.value===slider.max)slider.value=0;show();document.getElementById('play').textContent='Pause';timer=setInterval(()=>{slider.value=Math.min(Number(slider.max),Number(slider.value)+1);show();if(slider.value===slider.max)pause();},150);};show();</script>"""
    (args.run / "index.html").write_text(
        page.replace("METADATA", json.dumps(metadata).replace("<", "\\u003c"))
    )
    print(f"Wrote {args.run / 'index.html'} and {len(metadata)} frame images")


if __name__ == "__main__":
    main()
