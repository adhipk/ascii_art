import type { RenderResult } from "./renderer";
import { shapePixels } from "./shape-grid";
export const CHARACTERS = " .:-=+*#%@";
export const FONT_FAMILY = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
export interface TextMetrics {
  width: number; height: number; baseline: number; advance: number; densities: number[];
  fontSize?: number;
}
const cache = new Map<number, TextMetrics>();
const squareCache = new Map<number, TextMetrics>();
export function squareTextMetrics(size: number): TextMetrics {
  const cached = squareCache.get(size);
  if (cached) return cached;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const context = canvas.getContext("2d", { willReadFrequently: true })!;
  const fontSize = size * .8;
  context.font = `600 ${fontSize}px ${FONT_FAMILY}`;
  const measured = context.measureText("M");
  const ascent = measured.fontBoundingBoxAscent ?? fontSize;
  const descent = measured.fontBoundingBoxDescent ?? fontSize * .25;
  const baseline = (size - ascent - descent) / 2 + ascent;
  const advance = measured.width;
  context.fillStyle = "white";
  const densities = [...CHARACTERS].map((char) => {
    context.clearRect(0, 0, size, size);
    context.fillText(char, (size - advance) / 2, baseline);
    const pixels = context.getImageData(0, 0, size, size).data;
    let ink = 0;
    for (let i = 3; i < pixels.length; i += 4) ink += pixels[i] / 255;
    return ink / (size * size);
  });
  const metrics = { width: size, height: size, baseline, advance, densities, fontSize };
  squareCache.set(size, metrics);
  return metrics;
}
export function textMetrics(size: number): TextMetrics {
  const cached = cache.get(size);
  if (cached) return cached;
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d", { willReadFrequently: true })!;
  const font = `600 ${size}px ${FONT_FAMILY}`;
  context.font = font;
  const measured = context.measureText("M");
  const advance = measured.width;
  const width = Math.ceil(advance);
  const height = Math.ceil(size * 1.4);
  const ascent = measured.fontBoundingBoxAscent ?? size;
  const descent = measured.fontBoundingBoxDescent ?? size * .25;
  const baseline = (height - ascent - descent) / 2 + ascent;
  canvas.width = width; canvas.height = height;
  context.font = font;
  context.textBaseline = "alphabetic";
  context.fillStyle = "white";
  const densities = [...CHARACTERS].map((char) => {
    context.clearRect(0, 0, width, height);
    context.fillText(char, 0, baseline);
    const pixels = context.getImageData(0, 0, width, height).data;
    let ink = 0;
    for (let i = 3; i < pixels.length; i += 4) ink += pixels[i] / 255;
    return ink / (width * height);
  });
  const metrics = { width, height, baseline, advance, densities };
  cache.set(size, metrics);
  return metrics;
}
function rgb(data: Uint8Array, offset: number): string {
  return `rgb(${data[offset]} ${data[offset + 1]} ${data[offset + 2]})`;
}
function same(data: Uint8Array, a: number, b: number): boolean {
  return data[a] === data[b] && data[a + 1] === data[b + 1] && data[a + 2] === data[b + 2];
}
export function textPreview(result: RenderResult): DocumentFragment {
  const fragment = document.createDocumentFragment();
  if (result.shapes) {
    const canvas = shapeCanvas(result);
    canvas.style.width = canvas.style.height = "100%";
    canvas.style.display = "block";
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", "ASCII art with measured character and ligature shapes");
    fragment.append(canvas); return fragment;
  }
  if (result.tiles) {
    const namespace = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(namespace, "svg");
    svg.setAttribute("viewBox", `0 0 ${result.width} ${result.height}`);
    svg.setAttribute("width", "100%"); svg.setAttribute("height", "100%");
    svg.style.display = "block";
    for (const tile of result.tiles) {
      if (tile.char === " ") continue;
      const metrics = result.tileMetrics?.[String(tile.size)];
      if (!metrics) throw new Error("Missing adaptive font metrics");
      const cell = document.createElementNS(namespace, "svg");
      for (const [key, value] of Object.entries({ x: tile.x, y: tile.y, width: tile.size, height: tile.size, overflow: "hidden" })) cell.setAttribute(key, String(value));
      const glyph = document.createElementNS(namespace, "text");
      glyph.setAttribute("x", String((tile.size - metrics.advance) / 2));
      glyph.setAttribute("y", String(metrics.baseline));
      glyph.setAttribute("font-size", String(metrics.fontSize));
      glyph.setAttribute("font-family", FONT_FAMILY);
      glyph.setAttribute("font-weight", "600");
      glyph.setAttribute("fill", `rgb(${tile.color.join(" ")})`);
      glyph.textContent = tile.char;
      cell.append(glyph); svg.append(cell);
    }
    fragment.append(svg);
    return fragment;
  }
  const lines = result.text.split("\n");
  if (lines.length !== result.rows || lines.some((line) => line.length !== result.columns)
    || result.colors.length !== result.rows * result.columns * 3 || result.paper.length !== result.height * 3) throw new Error("Invalid character grid");
  for (let row = 0; row < result.rows; ++row) {
    const line = document.createElement("div");
    line.className = "ascii-line";
    for (let start = 0; start < result.columns;) {
      let end = start + 1;
      while (end < result.columns && same(result.colors, (row * result.columns + start) * 3, (row * result.columns + end) * 3)) ++end;
      const run = document.createElement("span");
      run.style.color = rgb(result.colors, (row * result.columns + start) * 3);
      run.textContent = lines[row].slice(start, end);
      line.append(run);
      start = end;
    }
    fragment.append(line);
  }
  return fragment;
}
export function paperBackground(result: RenderResult): string {
  // Sample the generated paper; no interpolation in a different color space.
  const stops = Array.from({ length: 17 }, (_, i) => {
    const row = Math.round(i * (result.height - 1) / 16);
    return `${rgb(result.paper, row * 3)} ${i * 100 / 16}%`;
  });
  return `linear-gradient(to bottom, ${stops.join(",")})`;
}
export function exportTextPng(result: RenderResult, size: number): Promise<Blob> {
  if (result.shapes) {
    const canvas = shapeCanvas(result);
    return new Promise((resolve,reject) => canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error("PNG export failed")), "image/png"));
  }
  const canvas = document.createElement("canvas");
  const scale = Math.min(2, Math.sqrt(12_000_000 / (result.width * result.height)));
  canvas.width = Math.round(result.width * scale); canvas.height = Math.round(result.height * scale);
  const context = canvas.getContext("2d")!;
  context.scale(scale, scale);
  for (let y = 0; y < result.height; ++y) {
    context.fillStyle = rgb(result.paper, y * 3);
    context.fillRect(0, y, result.width, 1);
  }
  context.font = `600 ${size}px ${FONT_FAMILY}`;
  context.textBaseline = "alphabetic";
  context.fontKerning = "none";
  if (result.tiles) {
    for (const tile of result.tiles) {
      const metrics = result.tileMetrics?.[String(tile.size)];
      if (!metrics) throw new Error("Missing adaptive font metrics");
      context.save();
      context.beginPath(); context.rect(tile.x, tile.y, tile.size, tile.size); context.clip();
      context.font = `600 ${metrics.fontSize}px ${FONT_FAMILY}`;
      context.fillStyle = `rgb(${tile.color.join(" ")})`;
      context.fillText(tile.char, tile.x + (tile.size - metrics.advance) / 2, tile.y + metrics.baseline);
      context.restore();
    }
    return new Promise((resolve, reject) => canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("PNG export failed")), "image/png"));
  }
  const metrics = textMetrics(size);
  const lines = result.text.split("\n");
  for (let row = 0; row < result.rows; ++row) for (let col = 0; col < result.columns; ++col) {
    context.fillStyle = rgb(result.colors, (row * result.columns + col) * 3);
    context.fillText(lines[row][col], col * result.cellWidth, row * result.cellHeight + metrics.baseline);
  }
  return new Promise((resolve, reject) => canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("PNG export failed")), "image/png"));
}

function shapeCanvas(result: RenderResult): HTMLCanvasElement {
  const canvas=document.createElement("canvas"); canvas.width=result.width; canvas.height=result.height;
  canvas.getContext("2d")!.putImageData(new ImageData(shapePixels(result),result.width,result.height),0,0);
  return canvas;
}
