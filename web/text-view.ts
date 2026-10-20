import {splitGlyphs, uniqueGlyphs} from "./scale-characters";
import type { RenderResult } from "./renderer";
import pursuitDictionary from "./pursuit-dictionary.json";
export const CHARACTERS = " .:-=+*#%@";
export const FONT_FAMILY = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
export interface TextMetrics {
  width: number; height: number; baseline: number; advance: number; densities: number[];
  fontSize?: number;
  kernelGlyphs?: string[]; kernelCharacters?: string; masks?: number[][];
}
const cache = new Map<number, TextMetrics>();
const squareCache = new Map<number, TextMetrics>();
export function squareTextMetrics(size:number):TextMetrics {
  const cached=squareCache.get(size);if(cached)return cached;
  const canvas=document.createElement("canvas");canvas.width=canvas.height=size;
  const context=canvas.getContext("2d",{willReadFrequently:true})!,fontSize=size*.8;
  context.font=`600 ${fontSize}px ${FONT_FAMILY}`;context.fontKerning="none";context.fillStyle="white";
  const measured=context.measureText("M"),advance=measured.width;
  const ascent=measured.fontBoundingBoxAscent??fontSize,descent=measured.fontBoundingBoxDescent??fontSize*.25;
  const baseline=(size-ascent-descent)/2+ascent,kernelCharacters=CHARACTERS+"/|\\";
  const masks=[...kernelCharacters].map(char=>{context.clearRect(0,0,size,size);context.fillText(char,(size-advance)/2,baseline);
    const data=context.getImageData(0,0,size,size).data;return Array.from({length:size*size},(_,i)=>data[i*4+3]/255);});
  const metrics={width:size,height:size,baseline,advance,fontSize,kernelCharacters,masks,densities:masks.slice(0,CHARACTERS.length).map(mask=>mask.reduce((a,b)=>a+b,0)/mask.length)};
  squareCache.set(size,metrics);return metrics;
}
export const PURSUIT_CHARACTERS=pursuitDictionary.characters;
const pursuitAtlases=new Map<string,TextMetrics>();
export function pursuitMetrics(input:string|string[]=PURSUIT_CHARACTERS):TextMetrics {
  const glyphs=typeof input==="string"?uniqueGlyphs(input):[...new Set(input)],key=JSON.stringify(glyphs);
  if(!glyphs.length)throw new Error("Glyph calibration requires characters");
  const cached=pursuitAtlases.get(key);if(cached)return cached;
  const size=64,fontSize=size*.95,canvas=document.createElement("canvas");canvas.width=canvas.height=size;
  const context=canvas.getContext("2d",{willReadFrequently:true})!;
  context.font=`600 ${fontSize}px ${FONT_FAMILY}`;context.fontKerning="none";context.fillStyle="white";
  const measured=context.measureText("M"),advance=measured.width;
  const ascent=measured.fontBoundingBoxAscent??fontSize,descent=measured.fontBoundingBoxDescent??fontSize*.25;
  const baseline=(size-ascent-descent)/2+ascent;
  const masks=glyphs.map(char=>{
    context.clearRect(0,0,size,size);const bounds=context.measureText(char);
    const left=bounds.actualBoundingBoxLeft??0,right=bounds.actualBoundingBoxRight??bounds.width;
    const top=bounds.actualBoundingBoxAscent??ascent,bottom=bounds.actualBoundingBoxDescent??descent;
    const fit=Math.min(1,(size-2)/Math.max(1,left+right),(size-2)/Math.max(1,top+bottom));
    context.save();context.translate(size/2,size/2);context.scale(fit,fit);context.fillText(char,(left-right)/2,(top-bottom)/2);context.restore();
    const data=context.getImageData(0,0,size,size).data;return Array.from({length:size*size},(_,i)=>data[i*4+3]/255);
  });
  const atlas={width:size,height:size,baseline,advance,fontSize,kernelGlyphs:glyphs,kernelCharacters:glyphs.join(""),masks,densities:masks.map(mask=>mask.reduce((a,b)=>a+b,0)/mask.length)};
  if(pursuitAtlases.size>=8)pursuitAtlases.delete(pursuitAtlases.keys().next().value!);
  pursuitAtlases.set(key,atlas);return atlas;
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
const glyphStampCaches=new WeakMap<TextMetrics,Map<string,HTMLCanvasElement>>();
function drawGlyphMasks(context:CanvasRenderingContext2D,result:RenderResult):void {
  const atlas=result.glyphAtlas!,glyphs=atlas.kernelGlyphs??splitGlyphs(atlas.kernelCharacters??"");
  let stamps=glyphStampCaches.get(atlas);if(!stamps){stamps=new Map();glyphStampCaches.set(atlas,stamps);}
  for(const tile of result.tiles??[]) {
    const key=JSON.stringify([tile.char,tile.color]);let stamp=stamps.get(key);
    if(!stamp) {
      const mask=atlas.masks?.[glyphs.indexOf(tile.char)];if(!mask) throw new Error("Missing measured Unicode glyph");
      stamp=document.createElement("canvas");stamp.width=atlas.width;stamp.height=atlas.height;
      const ctx=stamp.getContext("2d")!,pixels=ctx.createImageData(atlas.width,atlas.height);
      for(let i=0;i<mask.length;i++){pixels.data[i*4]=tile.color[0];pixels.data[i*4+1]=tile.color[1];pixels.data[i*4+2]=tile.color[2];pixels.data[i*4+3]=Math.round(mask[i]*255);}
      ctx.putImageData(pixels,0,0);if(stamps.size>=512)stamps.delete(stamps.keys().next().value!);stamps.set(key,stamp);
    }
    context.save();context.translate(tile.x+tile.size/2,tile.y+tile.size/2);context.rotate((tile.angle??0)*Math.PI/180);
    context.drawImage(stamp,-tile.size/2,-tile.size/2,tile.size,tile.size);context.restore();
  }
}
export function textPreview(result: RenderResult): DocumentFragment {
  const fragment = document.createDocumentFragment();
  if(result.glyphAtlas && result.tiles) {
    const canvas=document.createElement("canvas");canvas.width=result.width;canvas.height=result.height;
    canvas.style.width="100%";canvas.style.height="100%";canvas.style.display="block";
    drawGlyphMasks(canvas.getContext("2d")!,result);fragment.append(canvas);return fragment;
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
      if (tile.angle) glyph.setAttribute("transform", `rotate(${tile.angle} ${tile.size / 2} ${tile.size / 2})`);
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
  if(result.glyphAtlas && result.tiles) {
    drawGlyphMasks(context,result);
    return new Promise((resolve,reject)=>canvas.toBlob(blob=>blob?resolve(blob):reject(new Error("PNG export failed")),"image/png"));
  }
  if (result.tiles) {
    for (const tile of result.tiles) {
      const metrics = result.tileMetrics?.[String(tile.size)];
      if (!metrics) throw new Error("Missing adaptive font metrics");
      context.save();
      context.beginPath(); context.rect(tile.x, tile.y, tile.size, tile.size); context.clip();
      context.font = `600 ${metrics.fontSize}px ${FONT_FAMILY}`;
      context.fillStyle = `rgb(${tile.color.join(" ")})`;
      if (tile.angle) {
        context.translate(tile.x + tile.size / 2, tile.y + tile.size / 2);
        context.rotate(tile.angle * Math.PI / 180);
        context.fillText(tile.char, -metrics.advance / 2, metrics.baseline - tile.size / 2);
      } else {
        context.fillText(tile.char, tile.x + (tile.size - metrics.advance) / 2, tile.y + metrics.baseline);
      }
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
