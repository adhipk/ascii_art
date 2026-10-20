import type { ShapeAtlas } from "./shape-grid";

export const SHAPE_FONT = "ASCII Fira Code";
const cache = new Map<number, ShapeAtlas>();
let loading: Promise<void> | undefined;
export function loadShapeFont(): Promise<void> {
  return loading ??= (async () => {
    const face = new FontFace(SHAPE_FONT, 'url("./fonts/FiraCode.ttf")', {weight: "400"});
    await face.load(); document.fonts.add(face);
  })().catch(error => { loading = undefined; throw error; });
}

export function shapeAtlas(size: number): ShapeAtlas {
  const cached = cache.get(size); if (cached) return cached;
  if (!document.fonts.check(`${size}px "${SHAPE_FONT}"`)) throw new Error("Ligature font is not loaded");
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d", {willReadFrequently: true})!;
  context.font = `400 ${size}px "${SHAPE_FONT}"`;
  const advance = context.measureText("M").width;
  const width = Math.ceil(advance), height = Math.ceil(size * 1.4);
  const measured = context.measureText("M");
  const baseline = (height - (measured.fontBoundingBoxAscent ?? size) - (measured.fontBoundingBoxDescent ?? size*.25))/2 + (measured.fontBoundingBoxAscent ?? size);
  const alphabet = " .:-=+*#%@/|\\<>!~_";
  const candidates: ShapeAtlas["candidates"] = [];
  for (const span of [1,2]) {
    const texts = span === 1 ? [...alphabet] : [...alphabet].flatMap(a => [...alphabet].map(b => a+b));
    const seen = new Set<string>();
    for (const text of texts) {
      canvas.width = width * span; canvas.height = height;
      context.font = `400 ${size}px "${SHAPE_FONT}"`;
      context.fontKerning = "none";
      context.fillStyle = "white";
      // Center the natural shaped advance inside the integer-width span.
      context.fillText(text, (width*span-context.measureText(text).width)/2, baseline);
      const rgba = context.getImageData(0,0,canvas.width,height).data;
      const mask = Array.from({length: canvas.width*height}, (_,i) => rgba[i*4+3]/255);
      const key = mask.join(","); if (seen.has(key)) continue; seen.add(key);
      candidates.push({text, span, mask});
    }
  }
  const atlas = {width, height, size, baseline, candidates}; cache.set(size, atlas); return atlas;
}
