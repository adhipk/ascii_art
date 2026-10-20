import { KernelBlurObjective, type BlurCurvePoint } from "./pursuit-blur";
import { createWebGPUScorer, type CandidateBatch, type CandidateScorer } from "./pursuit-gpu";
import dictionary from "./pursuit-dictionary.json";
import { repeatAddition, repeatError } from "./pursuit-repeat";
import {penaltyAtScale} from "./penalty-weight";
import {allocateLayerBudgets} from "./layer-budgets";
import {sharpenTarget} from "./pursuit-sharpen";
import { splitGlyphs, hasGlyph, defaultPursuitScales, defaultScaleCharacters, normalizeScaleCharacters, type ScaleCharacters, type VocabularyProfile } from "./scale-characters";
import type { AdaptiveTile, RenderResult, RenderSettings } from "./renderer";
import type { TextMetrics } from "./text-view";

/** A bounded beam search over freely positioned, overlapping glyphs.
 * All fitting happens in a canonical coordinate system. Display resolution is
 * applied only after optimization, so it does not change the search decisions.
 */
export interface PursuitFrame {
  stage:number; activeScale:number|null; searchRound:number;
  iteration: number; energy: number; pixelError: number; contourError: number;
  overlapError?: number;
  repeatError?: number;
  glyphs: AdaptiveTile[]; diversity: number; action: string; candidates: number;
  scaleCandidates: Record<string, number>;
  residual: Uint8Array; residualWidth: number; residualHeight: number;
}
export interface PursuitStage {
  index:number; size:number|null; searchRounds:number; startFrame:number; endFrame:number; addedGlyphs:number; converged:boolean;
}
export interface PursuitTrace {
  frames: PursuitFrame[]; stopReason: string; analysisWidth: number; analysisHeight: number;
  budget: number; beamWidth: number; edgeWeight: number; elapsedMs: number; scales: number[];
  backend: "cpu" | "webgpu" | "mixed"; backendReason?: string; gpuBatches?: number; gpuCandidates?: number; gpuBatchMs?: number; gpuValidationMaxError?: number;
  gpuResident?:boolean; gpuReadbacks?:number; gpuInitMs?:number; gpuRunMs?:number;
  blurStrength:number; blurCurve:BlurCurvePoint[];
  coarseThreshold:number; coarseRounds:number; schedule:"coarse-to-fine"|"all"|"independent"; stages:PursuitStage[];
  objectiveScope?:"independent-layer-mean";
  budgetPerLayer?:number;
  targetSharpen?:number;
  layerBudgets?:Record<string,number>;
  layerRuns?:{size:number;budget:number;targetSharpen:number;glyphCount:number;energy:number;stopReason:string;elapsedMs:number;backend:string}[];
  compositeMetrics?:{pixelError:number;contourError:number;reconstructionEnergy:number};
  effort: "fast" | "thorough";
  vocabulary: VocabularyProfile; kernelCounts: Record<string, number>; overlapLimit:number; hiddenGlyphsRemoved:number;
  charactersByScale?: Record<string,string>;
  overlapPenalty?: number;
  repeatPenalty?: number;
  penaltyMinSize?: number;
}
interface Glyph { x: number; y: number; size: number; char: string; angle: number; color: number[]; }
interface Kernel { size: number; char: string; angle: number; alpha: Float32Array; inkX?:number; inkY?:number; }
interface State { glyphs: Glyph[]; pixels: Float32Array; coverage: Float32Array; penaltyCoverage:Float32Array; overlap: number; energy: number; }
interface Proposal { glyph: Glyph; kernel: Kernel; gain: number; }
interface Problem {
  width: number; height: number; target: Float32Array; paper: Float32Array;
  activeSize?:number; hotspotOffset?:number;
  repeatPenalty?:number;
  targetSharpen?:number;
  penaltyMinSize?:number; penaltyCoverage?:Float32Array;
  blur?:KernelBlurObjective;
  effort?: "fast" | "thorough";
  kernels: Kernel[]; edgeWeight: number; color: boolean; coverage:Float32Array; maxOverlap:number; overlapPenalty:number; evaluated: number; scaleCandidates: Record<string, number>;
}
const luminance = (pixels: Float32Array, i: number): number =>
  pixels[i * 3] * .299 + pixels[i * 3 + 1] * .587 + pixels[i * 3 + 2] * .114;
const clamp = (x: number, low = 0, high = 1): number => Math.max(low, Math.min(high, x));

export function energyTerms(target: Float32Array, pixels: Float32Array, width: number, height: number): { pixel: number; contour: number } {
  let pixel = 0, contour = 0;
  for (let i = 0; i < width * height; ++i) {
    for (let c = 0; c < 3; ++c) pixel += (target[i * 3 + c] - pixels[i * 3 + c]) ** 2 / 3;
    const residual = luminance(target, i) - luminance(pixels, i);
    if (i % width + 1 < width) contour += (residual - luminance(target, i + 1) + luminance(pixels, i + 1)) ** 2;
    if (i + width < width * height) contour += (residual - luminance(target, i + width) + luminance(pixels, i + width)) ** 2;
  }
  return { pixel: pixel / (width * height), contour: contour / (width * height) };
}

export function glyphAllowedAtScale(char:string,size:number,largest:number,profile:VocabularyProfile):boolean {
  if(profile==="shape-detail") return hasGlyph(defaultScaleCharacters(size,largest,profile),char);
  // Custom dictionaries/brush IDs are unrestricted unless part of this vocabulary.
  if(profile==="full" || !hasGlyph(dictionary.characters,char)) return true;
  if(size>=largest/2) return hasGlyph(dictionary.coarse,char);
  if(size>=largest/8) return hasGlyph(dictionary.medium,char);
  return true;
}

function kernelSet(metric: TextMetrics, sizes: number[], profile:VocabularyProfile, characters:ScaleCharacters): Kernel[] {
  const chars = metric.kernelGlyphs ?? splitGlyphs(metric.kernelCharacters ?? "");
  if (!chars.length || !metric.masks || metric.masks.length !== chars.length || metric.width !== metric.height)
    throw new Error("Top-down mode requires measured square glyph masks");
  const kernels: Kernel[] = [];
  const unique = new Set<string>();
  for (let k = 0; k < chars.length; ++k) {
    const mask = metric.masks[k];
    if (mask.length !== metric.width ** 2 || mask.some(v => !Number.isFinite(v) || v < 0 || v > 1)) throw new Error("Invalid top-down glyph mask");
    if (chars[k] === " " || !mask.some(v => v > .01)) continue;
    for (const size of sizes) for (const angle of [0, 90]) {
      if(characters[String(size)]!==undefined ? !hasGlyph(characters[String(size)],chars[k]) : !glyphAllowedAtScale(chars[k],size,Math.max(...sizes),profile)) continue;
      const alpha = new Float32Array(size * size);
      for (let y = 0; y < size; ++y) for (let x = 0; x < size; ++x) {
        // Area sampling retains thin strokes when reducing the normalized atlas.
        const x0 = Math.floor(x * metric.width / size), x1 = Math.max(x0 + 1, Math.ceil((x + 1) * metric.width / size));
        const y0 = Math.floor(y * metric.height / size), y1 = Math.max(y0 + 1, Math.ceil((y + 1) * metric.height / size));
        let sum = 0;
        for (let sy = y0; sy < y1; ++sy) for (let sx = x0; sx < x1; ++sx)
          sum += angle === 0 ? mask[sy * metric.width + sx] : mask[(metric.width - 1 - sx) * metric.width + sy];
        alpha[y * size + x] = sum / ((x1 - x0) * (y1 - y0));
      }
      const signature = `${size}:` + Array.from(alpha, v => Math.round(v * 255)).join(",");
      if (unique.has(signature)) continue;
      unique.add(signature);
      let mass=0,inkX=0,inkY=0;
      for(let y=0;y<size;++y)for(let x=0;x<size;++x){const a=alpha[y*size+x];mass+=a;inkX+=(x+.5)*a;inkY+=(y+.5)*a;}
      kernels.push({ size, char: chars[k], angle, alpha, inkX:inkX/mass, inkY:inkY/mass });
    }
  }
  if (!kernels.length) throw new Error("The glyph dictionary has no visible kernels");
  return kernels;
}

function draw(pixels: Float32Array, glyph: Glyph, kernel: Kernel, width: number, coverage?:Float32Array, penaltyCoverage=coverage, minSize=0): number {
  let overlap=0;
  for (let y = 0; y < glyph.size; ++y) for (let x = 0; x < glyph.size; ++x) {
    const alpha = kernel.alpha[y * glyph.size + x], i = ((y + glyph.y) * width + x + glyph.x) * 3;
    for (let c = 0; c < 3; ++c) pixels[i + c] += alpha * (glyph.color[c] - pixels[i + c]);
    if(penaltyCoverage && glyph.size>=minSize){overlap+=alpha*penaltyCoverage[i/3];penaltyCoverage[i/3]+=alpha*(1-penaltyCoverage[i/3]);}
    if(coverage && (coverage!==penaltyCoverage || glyph.size<minSize))coverage[i/3]+=alpha*(1-coverage[i/3]);
  }
  return coverage ? overlap/coverage.length : 0;
}

/** Exact local energy difference, including all gradient pairs crossing the patch boundary. */
function evaluate(problem: Problem, pixels: Float32Array, glyph: Glyph, kernel: Kernel): number {
  const { width, height, target, edgeWeight } = problem, { x: left, y: top, size } = glyph;
  const value = (x: number, y: number, channel: number): number => {
    const old = pixels[(y * width + x) * 3 + channel];
    if (x < left || y < top || x >= left + size || y >= top + size) return old;
    return old + kernel.alpha[(y - top) * size + x - left] * (glyph.color[channel] - old);
  };
  const residual = (x: number, y: number): number => {
    const i = y * width + x;
    return luminance(target, i) - (.299 * value(x, y, 0) + .587 * value(x, y, 1) + .114 * value(x, y, 2));
  };
  let delta = 0;
  for (let y = top; y < top + size; ++y) for (let x = left; x < left + size; ++x) {
    const i = (y * width + x) * 3;
    delta+=penaltyAtScale(problem.overlapPenalty,size,problem.penaltyMinSize)*kernel.alpha[(y-top)*size+x-left]*(problem.penaltyCoverage??problem.coverage)[i/3];
    for (let c = 0; c < 3; ++c) delta += ((target[i + c] - value(x, y, c)) ** 2 - (target[i + c] - pixels[i + c]) ** 2) / 3;
  }
  for (let y = Math.max(0, top - 1); y < Math.min(height, top + size); ++y)
    for (let x = Math.max(0, left - 1); x < Math.min(width, left + size); ++x) {
      const i = y * width + x, before = luminance(target, i) - luminance(pixels, i), after = residual(x, y);
      if (x + 1 < width) {
        const old = before - luminance(target, i + 1) + luminance(pixels, i + 1);
        delta += edgeWeight * ((after - residual(x + 1, y)) ** 2 - old ** 2);
      }
      if (y + 1 < height) {
        const old = before - luminance(target, i + width) + luminance(pixels, i + width);
        delta += edgeWeight * ((after - residual(x, y + 1)) ** 2 - old ** 2);
      }
    }
  return -delta / (width * height);
}

function fit(problem: Problem, pixels: Float32Array, kernel: Kernel, left: number, top: number, step: number): Proposal {
  let denominator = 0, before = 0, after = 0, alphaMass=0, coveredMass=0, penaltyMass=0;
  const numerator = [0, 0, 0], { size } = kernel;
  for (let y = 0; y < size; y += step) for (let x = 0; x < size; x += step) {
    const alpha = kernel.alpha[y * size + x], i = ((top + y) * problem.width + left + x) * 3;
    denominator += alpha * alpha;
    alphaMass+=alpha;coveredMass+=alpha*problem.coverage[i/3];penaltyMass+=alpha*(problem.penaltyCoverage??problem.coverage)[i/3];
    for (let c = 0; c < 3; ++c) numerator[c] += alpha * (problem.target[i + c] - (1 - alpha) * pixels[i + c]);
  }
  if(alphaMass>0 && coveredMass/alphaMass>problem.maxOverlap+1e-7) {
    ++problem.evaluated;problem.scaleCandidates[String(size)]=(problem.scaleCandidates[String(size)]??0)+1;
    return {glyph:{x:left,y:top,size,char:kernel.char,angle:kernel.angle,color:[]},kernel,gain:-1};
  }
  let color = numerator.map(v => clamp(v / Math.max(denominator, 1e-10)));
  if (!problem.color) color.fill(color[0] * .299 + color[1] * .587 + color[2] * .114);
  if (step === 1 && denominator > 1e-10) {
    // Fit brightness against the same pixel + contour objective. A shared
    // RGB offset preserves chroma and cannot tint grayscale originals.
    let gradientDenominator = 0, gradientNumerator = 0;
    const alphaAt = (x: number, y: number): number => x >= left && x < left + size && y >= top && y < top + size ? kernel.alpha[(y - top) * size + x - left] : 0;
    const pair = (x: number, y: number, nx: number, ny: number): void => {
      const i = y * problem.width + x, j = ny * problem.width + nx;
      const a = alphaAt(x, y), b = alphaAt(nx, ny), difference = a - b;
      const fixed = (1 - a) * luminance(pixels, i) - (1 - b) * luminance(pixels, j);
      gradientDenominator += problem.edgeWeight * difference * difference;
      gradientNumerator += problem.edgeWeight * difference * (luminance(problem.target, i) - luminance(problem.target, j) - fixed);
    };
    for (let y = Math.max(0, top - 1); y < Math.min(problem.height, top + size); ++y)
      for (let x = Math.max(0, left - 1); x < Math.min(problem.width, left + size); ++x) {
        if (x + 1 < problem.width) pair(x, y, x + 1, y);
        if (y + 1 < problem.height) pair(x, y, x, y + 1);
      }
    if (problem.color) {
      const weights = [.299, .587, .114], raw = numerator.map(v => v / denominator);
      const dot = raw.reduce((sum, v, c) => sum + weights[c] * v, 0);
      const adjustment = (gradientNumerator - gradientDenominator * dot) / (denominator + gradientDenominator);
      color = raw.map(v => clamp(v + adjustment));
    } else {
      const gray = clamp((numerator.reduce((sum, v) => sum + v, 0) / 3 + gradientNumerator) / (denominator + gradientDenominator));
      color.fill(gray);
    }
  }
  const correctedResidual = (x: number, y: number): number => {
    const i = y * problem.width + x;
    const alpha = x >= left && y >= top && x < left + size && y < top + size ? kernel.alpha[(y - top) * size + x - left] : 0;
    const nextLuma = luminance(pixels, i) * (1 - alpha) + (color[0] * .299 + color[1] * .587 + color[2] * .114) * alpha;
    return luminance(problem.target, i) - nextLuma;
  };
  for (let y = 0; y < size; y += step) for (let x = 0; x < size; x += step) {
    const alpha = kernel.alpha[y * size + x], px = left + x, py = top + y, i = (py * problem.width + px) * 3;
    for (let c = 0; c < 3; ++c) {
      before += (problem.target[i + c] - pixels[i + c]) ** 2 / 3;
      after += (problem.target[i + c] - pixels[i + c] - alpha * (color[c] - pixels[i + c])) ** 2 / 3;
    }
    const residual = luminance(problem.target, i / 3) - luminance(pixels, i / 3);
    for (const [nx, ny] of [[px + 1, py], [px, py + 1]]) {
      if (nx >= problem.width || ny >= problem.height) continue;
      const j = ny * problem.width + nx;
      before += problem.edgeWeight * (residual - luminance(problem.target, j) + luminance(pixels, j)) ** 2;
      after += problem.edgeWeight * (correctedResidual(px, py) - correctedResidual(nx, ny)) ** 2;
    }
  }
  ++problem.evaluated;
  problem.scaleCandidates[String(size)] = (problem.scaleCandidates[String(size)] ?? 0) + 1;
  return { glyph: { x: left, y: top, size, char: kernel.char, angle: kernel.angle, color }, kernel,
    gain: (before - after - penaltyAtScale(problem.overlapPenalty,size,problem.penaltyMinSize)*penaltyMass) * step ** 2 / (problem.width * problem.height) };
}

interface CandidateSpec { kernel: number; x: number; y: number; }
function candidateSpecs(problem: Problem, state: State, focus?: Glyph): CandidateSpec[] {
  const { width, height } = problem, target=problem.blur?.guide(state.pixels)??problem.target;
  // Rank residual blocks, rather than random proposals, for deterministic replay.
  const hotspots: { x: number; y: number; error: number }[] = [];
  const block = 8;
  for (let top = 0; top < height; top += block) for (let left = 0; left < width; left += block) {
    let error = 0, weightedX = 0, weightedY = 0;
    for (let y = top; y < Math.min(height, top + block); ++y) for (let x = left; x < Math.min(width, left + block); ++x) {
      const i = y * width + x;
      let local = 0;
      for (let c = 0; c < 3; ++c) local += (target[i * 3 + c] - state.pixels[i * 3 + c]) ** 2;
      const r = luminance(target, i) - luminance(state.pixels, i);
      if (x + 1 < width) local += problem.edgeWeight * (r - luminance(target, i + 1) + luminance(state.pixels, i + 1)) ** 2;
      if (y + 1 < height) local += problem.edgeWeight * (r - luminance(target, i + width) + luminance(state.pixels, i + width)) ** 2;
      error += local; weightedX += (x + .5) * local; weightedY += (y + .5) * local;
    }
    hotspots.push({ x: error > 0 ? weightedX / error : left + block / 2, y: error > 0 ? weightedY / error : top + block / 2, error });
  }
  hotspots.sort((a, b) => b.error - a.error);
  const focusedKernel=focus?problem.kernels.find(k=>k.char===focus.char&&k.size===focus.size&&k.angle===focus.angle):undefined;
  const anchors = focus ? [{ x: focus.x + (focusedKernel?.inkX??focus.size/2), y: focus.y + (focusedKernel?.inkY??focus.size/2), error: 1 }] : hotspots.slice(problem.hotspotOffset??0,(problem.hotspotOffset??0)+(problem.effort === "fast" ? 3 : 6));
  const specs: CandidateSpec[] = [];
  // Share positions only when glyph ink centres agree, including rotation.
  const positions = new Map<string, {x:number;y:number}[]>();
  for (const [kernelIndex, kernel] of problem.kernels.entries()) {
    if (kernel.size > width || kernel.size > height || (problem.activeSize!==undefined && kernel.size!==problem.activeSize)) continue;
    const inkX=kernel.inkX??kernel.size/2,inkY=kernel.inkY??kernel.size/2;
    const positionKey=`${kernel.size}:${inkX.toFixed(4)}:${inkY.toFixed(4)}`;
    let locations = positions.get(positionKey);
    if (!locations) {
      locations = [];
      const seen = new Set<number>();
      for (const anchor of anchors) for (const [offsetX, offsetY] of [[0, 0], [-.25, 0], [.25, 0], [0, -.25], [0, .25]]) {
        const x = Math.round(clamp(anchor.x - inkX + offsetX * kernel.size, 0, width - kernel.size));
        const y = Math.round(clamp(anchor.y - inkY + offsetY * kernel.size, 0, height - kernel.size));
        const key = y * width + x;
        if (seen.has(key)) continue;
        seen.add(key); locations.push({x,y});
      }
      positions.set(positionKey, locations);
    }
    for (const {x,y} of locations) specs.push({kernel:kernelIndex,x,y});
  }
  return specs;
}

function finishProposals(problem: Problem, state: State, specs: CandidateSpec[], scores: ArrayLike<number>, fitted?: Proposal[]): Proposal[] {
  problem.coverage=state.coverage;problem.penaltyCoverage=state.penaltyCoverage;
  const byScale = new Map<number, { spec: CandidateSpec; gain: number; index: number }[]>();
  for (let i = 0; i < specs.length; ++i) {
    const kernel=problem.kernels[specs[i].kernel];
    const score=scores[i]-(problem.repeatPenalty ? problem.repeatPenalty*repeatAddition({...specs[i],size:kernel.size,char:kernel.char},state.glyphs,problem.width*problem.height,problem.penaltyMinSize) : 0);
    if (!Number.isFinite(score) || score <= 1e-10) continue;
    const size = problem.kernels[specs[i].kernel].size, shortlist = byScale.get(size) ?? [];
    if (shortlist.length === 12 && score <= shortlist[11].gain) continue;
    shortlist.push({spec: specs[i], gain: score, index:i});
    shortlist.sort((a,b)=>b.gain-a.gain);
    if (shortlist.length > 12) shortlist.pop();
    byScale.set(size,shortlist);
  }
  const verify = ({spec, index}: {spec:CandidateSpec;index?:number}): Proposal => {
    const kernel=problem.kernels[spec.kernel];
    const repeatCost=problem.repeatPenalty ? problem.repeatPenalty*repeatAddition({...spec,size:kernel.size,char:kernel.char},state.glyphs,problem.width*problem.height,problem.penaltyMinSize) : 0;
    if(problem.blur){
      let mass=0,covered=0,penaltyMass=0;
      for(let y=0;y<kernel.size;++y)for(let x=0;x<kernel.size;++x){const a=kernel.alpha[y*kernel.size+x];mass+=a;covered+=a*state.coverage[(spec.y+y)*problem.width+spec.x+x];penaltyMass+=a*state.penaltyCoverage[(spec.y+y)*problem.width+spec.x+x];}
      ++problem.evaluated;
      problem.scaleCandidates[String(kernel.size)]=(problem.scaleCandidates[String(kernel.size)]??0)+1;
      const glyph={x:spec.x,y:spec.y,size:kernel.size,char:kernel.char,angle:kernel.angle,color:[] as number[]};
      if(mass>0&&covered/mass>problem.maxOverlap+1e-7)return {glyph,kernel,gain:-1};
      const fitted=problem.blur.fit(state.pixels,spec.x,spec.y,kernel.size,kernel.alpha,problem.color);
      return {glyph:{...glyph,color:fitted.color},kernel,gain:fitted.gain-penaltyAtScale(problem.overlapPenalty,kernel.size,problem.penaltyMinSize)*penaltyMass/(problem.width*problem.height)-repeatCost};
    }
    // Fine kernels already used every alpha sample during CPU scoring.
    const exact = fitted && index!==undefined && Math.floor(kernel.size/10) < 2
      ? fitted[index] : fit(problem,state.pixels,kernel,spec.x,spec.y,1);
    if (exact.gain === -1 && !exact.glyph.color.length) return exact;
    return {...exact,gain:evaluate(problem,state.pixels,exact.glyph,kernel)-repeatCost};
  };
  const quota=problem.effort === "fast" ? 4 : 12;
  let verified=[...byScale.values()].flatMap(list=>list.slice(0,quota)).map(verify);
  // A narrow shortlist must not prematurely stop a run: broaden before giving up.
  if (quota < 12 && !verified.some(p=>p.gain>1e-9))
    verified=verified.concat([...byScale.values()].flatMap(list=>list.slice(quota)).map(verify));
  verified=verified.filter(p=>p.gain>1e-9).sort((a,b)=>b.gain-a.gain);
  // Hotspot centroids/quarter-cell offsets alone miss pixel-aligned contours.
  // Refine the best survivor locally using the same exact objective and overlap cap.
  if(verified.length){
    let best=verified[0];
    const kernelIndex=problem.kernels.indexOf(best.kernel);
    for(const step of new Set([Math.max(1,Math.round(best.kernel.size/16)),1]))for(let round=0;round<2;++round){
      const start=best;
      for(const [dx,dy] of [[-step,0],[step,0],[0,-step],[0,step]]){
        const x=start.glyph.x+dx,y=start.glyph.y+dy;
        if(x<0||y<0||x+start.kernel.size>problem.width||y+start.kernel.size>problem.height)continue;
        const candidate=verify({spec:{kernel:kernelIndex,x,y}});
        if(candidate.gain>best.gain+1e-9)best=candidate;
      }
      if(best===start)break;
    }
    verified[0]=best;
  }
  return verified.slice(0,3);
}

function proposals(problem: Problem, state: State, focus?: Glyph): Proposal[] {
  problem.coverage=state.coverage;problem.penaltyCoverage=state.penaltyCoverage;
  const specs = candidateSpecs(problem,state,focus);
  const original=problem.target;
  problem.target=problem.blur?.guide(state.pixels)??original;
  const fitted = specs.map(spec => {
    const kernel = problem.kernels[spec.kernel];
    return fit(problem,state.pixels,kernel,spec.x,spec.y,Math.max(1,Math.floor(kernel.size/10)));
  });
  problem.target=original;
  return finishProposals(problem,state,specs,fitted.map(p=>p.gain),fitted);
}

/** Reference for adapter self-checks and reproducible GPU/CPU parity tests. */
export function referenceCandidateScores(batch: CandidateBatch): Float64Array {
  const problem: Problem = {width:batch.width,height:batch.height,target:batch.guidance??batch.target,paper:new Float32Array(),
    kernels:batch.kernels.map(k=>({...k,char:"",angle:0})),edgeWeight:batch.edgeWeight,color:batch.color,coverage:batch.coverage??new Float32Array(batch.width*batch.height),maxOverlap:batch.maxOverlap??1,overlapPenalty:batch.overlapPenalty??0,penaltyCoverage:batch.penaltyCoverage,penaltyMinSize:batch.penaltyMinSize,evaluated:0,scaleCandidates:{}};
  return Float64Array.from(batch.candidates, spec => {
    const kernel=problem.kernels[spec.kernel];
    return fit(problem,batch.pixels,kernel,spec.x,spec.y,Math.max(1,Math.floor(kernel.size/10))).gain;
  });
}

function normalizedSource(pixels: Uint8Array, sourceWidth: number, sourceHeight: number, width: number, height: number, settings: RenderSettings): Float32Array {
  const target = new Float32Array(width * height * 3);
  const paper = [1, 3, 5].map(i => parseInt(settings.background.slice(i, i + 2), 16) / 255);
  for (let y = 0; y < height; ++y) for (let x = 0; x < width; ++x) {
    const x0 = Math.floor(x * sourceWidth / width), x1 = Math.max(x0 + 1, Math.ceil((x + 1) * sourceWidth / width));
    const y0 = Math.floor(y * sourceHeight / height), y1 = Math.max(y0 + 1, Math.ceil((y + 1) * sourceHeight / height));
    const sum = [0, 0, 0]; let count = 0;
    for (let sy = y0; sy < y1; ++sy) for (let sx = x0; sx < x1; ++sx) {
      const i = (sy * sourceWidth + sx) * 4, alpha = pixels[i + 3] / 255;
      for (let c = 0; c < 3; ++c) sum[c] += pixels[i + c] / 255 * alpha + paper[c] * (1 - alpha);
      ++count;
    }
    const gray = (sum[0] * .299 + sum[1] * .587 + sum[2] * .114) / count;
    for (let c = 0; c < 3; ++c) target[(y * width + x) * 3 + c] = settings.use_color ? sum[c] / count : gray;
  }
  return target;
}

export function preparePursuit(pixels:Uint8Array,sourceWidth:number,sourceHeight:number,settings:RenderSettings) {
  const started = performance.now(), budget = settings.pursuit_budget ?? 128, beamWidth = settings.pursuit_beam ?? 1;
  const edgeWeight = settings.pursuit_edge_weight ?? 3;
  let metric = settings.pursuit_metrics;
  if (!Number.isInteger(budget) || budget < 1 || budget > 2048 || !Number.isInteger(beamWidth) || beamWidth < 1 || beamWidth > 3 || !Number.isFinite(edgeWeight) || edgeWeight < 0 || edgeWeight > 8)
    throw new Error("Invalid top-down search budget or weights");
  if (!Number.isInteger(settings.columns) || settings.columns < 20 || settings.columns > 180 || !/^#[0-9a-fA-F]{6}$/.test(settings.background))
    throw new Error("Invalid output dimensions or paper color");
  if (Math.max(sourceWidth, sourceHeight) / Math.min(sourceWidth, sourceHeight) > 32)
    throw new Error("Top-down mode supports aspect ratios up to 32:1");
  if (!metric || !Number.isInteger(sourceWidth) || !Number.isInteger(sourceHeight) || sourceWidth < 1 || sourceHeight < 1 || sourceWidth * sourceHeight > 1_440_000 || pixels.length !== sourceWidth * sourceHeight * 4)
    throw new Error("Invalid top-down image or glyph calibration");
  const resolution = settings.pursuit_resolution ?? 192;
  if (![128, 192, 320].includes(resolution)) throw new Error("Invalid fitting resolution");
  const width = Math.max(6, Math.round(resolution * sourceWidth / Math.max(sourceWidth, sourceHeight)));
  const height = Math.max(6, Math.round(resolution * sourceHeight / Math.max(sourceWidth, sourceHeight)));
  const requestedSizes=settings.pursuit_scales??defaultPursuitScales(resolution);
  if(!Array.isArray(requestedSizes)||!requestedSizes.length||requestedSizes.length>8||requestedSizes.some(size=>!Number.isInteger(size)||size<3||size>resolution/2))throw new Error("Invalid kernel scales");
  const sizes=[...new Set(requestedSizes)].sort((a,b)=>b-a);
  if(settings.pursuit_solid_brush) {
    const glyphs=metric.kernelGlyphs??splitGlyphs(metric.kernelCharacters??""),masks=[...(metric.masks??[])];
    const index=glyphs.indexOf("█"),solid=Array(metric.width*metric.height).fill(1);
    const nextGlyphs=[...glyphs];if(index>=0)masks[index]=solid;else{nextGlyphs.push("█");masks.push(solid);}
    metric={...metric,kernelGlyphs:nextGlyphs,kernelCharacters:nextGlyphs.join(""),masks};
  }
  const scaleCharacters=normalizeScaleCharacters(settings.pursuit_characters_by_scale,sizes,metric.kernelGlyphs??metric.kernelCharacters??"");
  if(settings.pursuit_solid_brush) for(const size of sizes) {
    const chars=scaleCharacters[String(size)]??defaultScaleCharacters(size,sizes[0],settings.pursuit_vocabulary??"scale-aware");
    if(chars.length)scaleCharacters[String(size)]=[...new Set([...splitGlyphs(chars),"█"])].join("");
  }
  let target = normalizedSource(pixels, sourceWidth, sourceHeight, width, height, settings);
  const detailSharpen=settings.pursuit_detail_sharpen??0;
  if(!Number.isFinite(detailSharpen)||detailSharpen<0||detailSharpen>3)throw new Error("Detail sharpening must be between 0 and 3");
  const appliedSharpen=sizes.length===1&&sizes[0]===8?detailSharpen:0;
  const paper = new Float32Array(target.length), solid = [1, 3, 5].map(i => parseInt(settings.background.slice(i, i + 2), 16) / 255);
  let paperColor = solid;
  if (settings.background_mode === "source") {
    const samples = Array.from({ length: width * height }, (_, i) => i).sort((a, b) => luminance(target, a) - luminance(target, b));
    const dark = luminance(target, samples[Math.floor(samples.length / 2)]) < .5;
    const selected = samples[Math.floor(samples.length * (dark ? .05 : .95))];
    paperColor = [0, 1, 2].map(c => target[selected * 3 + c]);
  }
  for (let i = 0; i < width * height; ++i) for (let c = 0; c < 3; ++c) paper[i * 3 + c] = paperColor[c];
  if(appliedSharpen)target=sharpenTarget(target,width,height,appliedSharpen);
  const schedule=settings.pursuit_schedule??"coarse-to-fine";
  if(schedule!=="coarse-to-fine" && schedule!=="all" && schedule!=="independent") throw new Error("Invalid scale schedule");
  const coarseThreshold=settings.pursuit_coarse_threshold??.002;
  if(!Number.isFinite(coarseThreshold)||coarseThreshold<0||coarseThreshold>.05)throw new Error("Invalid broad early-exit threshold");
  const coarseRounds=settings.pursuit_coarse_rounds??3;
  if(!Number.isInteger(coarseRounds)||coarseRounds<1||coarseRounds>8)throw new Error("Invalid broad search rounds");
  const effort=settings.pursuit_effort??"fast";
  if(effort!=="fast" && effort!=="thorough") throw new Error("Invalid search effort");
  const vocabulary=settings.pursuit_vocabulary??"scale-aware";
  if(!["scale-aware","full","shape-detail"].includes(vocabulary)) throw new Error("Invalid glyph vocabulary profile");
  const maxOverlap=settings.pursuit_max_overlap??1;
  if(!Number.isFinite(maxOverlap)||maxOverlap<0||maxOverlap>1)throw new Error("Invalid ink overlap limit");
  const overlapPenalty=settings.pursuit_overlap_penalty??0;
  if(!Number.isFinite(overlapPenalty)||overlapPenalty<0||overlapPenalty>5)throw new Error("Overlap penalty must be between 0 and 5");
  const repeatPenalty=settings.pursuit_repeat_penalty??0;
  if(!Number.isFinite(repeatPenalty)||repeatPenalty<0||repeatPenalty>5)throw new Error("Repeat penalty must be between 0 and 5");
  const penaltyMinSize=settings.pursuit_penalty_min_size??0;
  if(!Number.isFinite(penaltyMinSize)||penaltyMinSize<0)throw new Error("Penalty minimum size must be nonnegative");
  const blurStrength=settings.pursuit_blur??1;
  const blurModel=new KernelBlurObjective(target,width,height,sizes,blurStrength,edgeWeight);
  const blur=blurModel.weights.length>1?blurModel:undefined;
  const totalEnergy=(pixels:Float32Array,errors=energyTerms(target,pixels,width,height)):number=>blur?.energy(pixels)??errors.pixel+edgeWeight*errors.contour;
  const problem: Problem = { width, height, target, paper, effort, blur, coverage:new Float32Array(width*height),maxOverlap,overlapPenalty,kernels: kernelSet(metric, sizes,vocabulary,scaleCharacters), edgeWeight, color: settings.use_color, evaluated: 0, scaleCandidates: {} };
  const kernelCounts=Object.fromEntries(sizes.map(size=>[String(size),problem.kernels.filter(kernel=>kernel.size===size).length]));
  problem.repeatPenalty=repeatPenalty;problem.penaltyMinSize=penaltyMinSize;problem.targetSharpen=appliedSharpen;
  const charactersByScale=Object.fromEntries(sizes.map(size=>[String(size),[...new Set(problem.kernels.filter(k=>k.size===size).map(k=>k.char))].join("")]));
  return {started,budget,beamWidth,edgeWeight,metric,resolution,width,height,sizes,target,paper,paperColor,schedule,coarseThreshold,coarseRounds,effort,vocabulary,maxOverlap,overlapPenalty,blurStrength,blurModel,blur,totalEnergy,problem,kernelCounts,charactersByScale,appliedSharpen};
}

function independentLayerSettings(setup:ReturnType<typeof preparePursuit>,settings:RenderSettings,size:number):RenderSettings {
  const active=setup.sizes.filter(scale=>setup.kernelCounts[String(scale)]>0);
  const budget=allocateLayerBudgets(setup.sizes,active,setup.budget)[String(size)];
  const layer:RenderSettings={...settings,pursuit_budget:budget,pursuit_metrics:setup.metric,pursuit_scales:[size],pursuit_schedule:"all",
    pursuit_blur:size===8&&(settings.pursuit_detail_sharpen??0)>0?0:settings.pursuit_blur,
    pursuit_characters_by_scale:{[size]:setup.charactersByScale[String(size)]}};
  return layer;
}

function combineIndependentLayers(setup:ReturnType<typeof preparePursuit>,results:RenderResult[],started:number):RenderResult {
  const traces=results.map(result=>result.pursuit!),count=results.length;
  const current=traces.map(trace=>trace.frames[0]),frames:PursuitFrame[]=[],stages:PursuitStage[]=[];
  const completed:AdaptiveTile[]=[];
  const record=(layer:number,frame:PursuitFrame,action:string):void=>{
    current[layer]=frame;
    const mean=(get:(frame:PursuitFrame)=>number)=>current.reduce((sum,value)=>sum+get(value),0)/count;
    const glyphs=[...completed,...frame.glyphs];
    frames.push({...frame,stage:layer,activeScale:traces[layer].scales[0],iteration:frames.length,action,glyphs,
      energy:mean(value=>value.energy),pixelError:mean(value=>value.pixelError),contourError:mean(value=>value.contourError),
      overlapError:mean(value=>value.overlapError??0),repeatError:mean(value=>value.repeatError??0),
      diversity:new Set(glyphs.map(glyph=>glyph.char)).size,candidates:current.reduce((sum,value)=>sum+value.candidates,0),
      scaleCandidates:Object.assign({},...current.map(value=>value.scaleCandidates))});
  };
  record(0,current[0],"Start independent layers with paper");
  for(const [layer,trace] of traces.entries()){
    const startFrame=frames.length-1;
    for(const frame of trace.frames.slice(1))record(layer,frame,`${trace.scales[0]}px layer · ${frame.action}`);
    completed.push(...results[layer].tiles??[]);
    stages.push({index:layer,size:trace.scales[0],searchRounds:1,startFrame,endFrame:frames.length-1,
      addedGlyphs:results[layer].tiles?.length??0,converged:trace.stopReason!=="Glyph budget reached"});
  }
  const pixels=setup.paper.slice(),factor=results[0].width/setup.width;
  for(const tile of completed){
    const glyph={...tile,x:Math.round(tile.x/factor),y:Math.round(tile.y/factor),size:Math.round(tile.size/factor),angle:tile.angle??0,color:tile.color.map(value=>value/255)};
    const kernel=setup.problem.kernels.find(candidate=>candidate.size===glyph.size&&candidate.char===glyph.char&&candidate.angle===glyph.angle)!;
    draw(pixels,glyph,kernel,setup.width);
  }
  const errors=energyTerms(setup.target,pixels,setup.width,setup.height);
  const backends=new Set(traces.map(trace=>trace.backend));
  return {...results[0],tiles:completed,tileMetrics:Object.assign({},...results.map(result=>result.tileMetrics)),pursuit:{
    ...traces[0],frames,stages,scales:setup.sizes,schedule:"independent",objectiveScope:"independent-layer-mean",
    budgetPerLayer:setup.budget,budget:setup.budget*count,
    layerBudgets:allocateLayerBudgets(setup.sizes,traces.map(trace=>trace.scales[0]),setup.budget),elapsedMs:performance.now()-started,
    kernelCounts:setup.kernelCounts,charactersByScale:setup.charactersByScale,blurCurve:setup.blurModel.curve,
    overlapPenalty:setup.overlapPenalty,repeatPenalty:setup.problem.repeatPenalty,penaltyMinSize:setup.problem.penaltyMinSize,
    backend:backends.size===1?traces[0].backend:"mixed",gpuResident:false,
    gpuBatches:traces.reduce((sum,trace)=>sum+(trace.gpuBatches??0),0),gpuCandidates:traces.reduce((sum,trace)=>sum+(trace.gpuCandidates??0),0),
    gpuBatchMs:traces.reduce((sum,trace)=>sum+(trace.gpuBatchMs??0),0),
    hiddenGlyphsRemoved:traces.reduce((sum,trace)=>sum+trace.hiddenGlyphsRemoved,0),
    stopReason:"Independent layers complete · "+traces.map(trace=>`${trace.scales[0]}px: ${trace.stopReason}`).join("; "),
    layerRuns:traces.map((trace,index)=>({size:trace.scales[0],budget:trace.budget,targetSharpen:trace.targetSharpen??0,glyphCount:results[index].tiles?.length??0,energy:trace.frames.at(-1)!.energy,
      stopReason:trace.stopReason,elapsedMs:trace.elapsedMs,backend:trace.backend})),
    compositeMetrics:{pixelError:errors.pixel,contourError:errors.contour,reconstructionEnergy:setup.totalEnergy(pixels,errors)}
  }};
}

interface SearchRequest {problem: Problem; state: State; focus?: Glyph;}
function* searchPursuit(pixels: Uint8Array, sourceWidth: number, sourceHeight: number, settings: RenderSettings, progress: (message: string) => void = () => {}): Generator<SearchRequest, RenderResult, Proposal[]> {
  const {started,budget,beamWidth,edgeWeight,metric,width,height,sizes,target,paper,paperColor,schedule,coarseThreshold,coarseRounds,effort,vocabulary,maxOverlap,blurStrength,blurModel,blur,totalEnergy,problem,kernelCounts,charactersByScale}=preparePursuit(pixels,sourceWidth,sourceHeight,settings);
  const terms = energyTerms(target, paper, width, height);
  let beam: State[] = [{ glyphs: [], pixels: paper.slice(), coverage:new Float32Array(width*height), penaltyCoverage:new Float32Array(width*height), overlap:0, energy: totalEnergy(paper,terms) }];
  const frames: PursuitFrame[] = [],stages:PursuitStage[]=[];
  let stageIndex=0,searchRound=0;
  problem.activeSize=schedule==="coarse-to-fine"?sizes[0]:undefined;
  const outputWidth = settings.columns * 8, factor = outputWidth / width, outputHeight = height * factor;
  const displayCache = new WeakMap<Glyph, AdaptiveTile>();
  const displayGlyph = (g: Glyph): AdaptiveTile => {
    const cached = displayCache.get(g);
    if (cached) return cached;
    const tile: AdaptiveTile = ({ x: g.x * factor, y: g.y * factor, size: g.size * factor, char: g.char, angle: g.angle, color: g.color.map(v => v * 255) });
    displayCache.set(g, tile);
    return tile;
  };
  function record(state: State, action: string): void {
    const errors = energyTerms(target, state.pixels, width, height);
    state.energy = totalEnergy(state.pixels,errors)+problem.overlapPenalty*state.overlap+(problem.repeatPenalty??0)*repeatError(state.glyphs,width*height,problem.penaltyMinSize);
    const thumbnailScale = Math.min(1, (budget > 512 ? 64 : 96) / Math.max(width, height));
    const residualWidth = Math.max(1, Math.round(width * thumbnailScale)), residualHeight = Math.max(1, Math.round(height * thumbnailScale));
    const residual = new Uint8Array(residualWidth * residualHeight);
    for (let y = 0; y < residualHeight; ++y) for (let x = 0; x < residualWidth; ++x) {
      let signed = 0;
      for (let sy = Math.floor(y * height / residualHeight); sy < Math.ceil((y + 1) * height / residualHeight); ++sy)
        for (let sx = Math.floor(x * width / residualWidth); sx < Math.ceil((x + 1) * width / residualWidth); ++sx) {
          const value = luminance(target, sy * width + sx) - luminance(state.pixels, sy * width + sx);
          if (Math.abs(value) > Math.abs(signed)) signed = value;
        }
      residual[y * residualWidth + x] = Math.round(128 + 127 * signed);
    }
    frames.push({ stage:stageIndex,activeScale:problem.activeSize??null,searchRound,iteration: frames.length, energy: state.energy, pixelError: errors.pixel, contourError: errors.contour,overlapError:state.overlap,
      repeatError:repeatError(state.glyphs,width*height,problem.penaltyMinSize),glyphs: state.glyphs.map(displayGlyph), diversity: new Set(state.glyphs.map(g => g.char)).size, action, candidates: problem.evaluated, scaleCandidates: { ...problem.scaleCandidates }, residual, residualWidth, residualHeight });
  }
  record(beam[0], "Start with paper");
  let stopReason = "Glyph budget reached",additions=0;
  let best=beam[0];
  const passes:(number|undefined)[]=schedule==="coarse-to-fine"?[...sizes].sort((a,b)=>b-a):[undefined];
  for(const [passIndex,size] of passes.entries()){
    stageIndex=passIndex;problem.activeSize=size;problem.hotspotOffset=0;searchRound=0;
    const minimumGain=size!==undefined&&size>=sizes[0]/2?Math.max(1e-9,best.energy*coarseThreshold):1e-9;
    const roundLimit=size!==undefined&&size>=sizes[0]/2&&problem.kernels.some(k=>k.size===size)?coarseRounds:1;
    // Keep the best underpainting. Alternative beam branches belong to the old pass.
    beam=[best];
    const startFrame=frames.length-1,startAdditions=additions;
    let refined=false,converged=false;
    progress(size===undefined?"Searching all scales…":`Pass ${passIndex+1} / ${passes.length} · ${size}px kernels…`);
    while(true){
      const canAdd=additions<budget;
      if(canAdd){
        problem.scaleCandidates={};
        const next:{state:State;proposal:Proposal;energy:number}[]=[];
        for(const effortPass of effort==="fast"?["fast","thorough"] as const:["thorough"] as const){
          problem.effort=effortPass;
          for(const state of beam)for(const proposal of (yield {problem,state}))
            next.push({state,proposal,energy:state.energy-proposal.gain});
          next.sort((a,b)=>a.energy-b.energy);
          if(next.length&&next[0].energy<beam[0].energy-minimumGain)break;
        }
        problem.effort=effort;
        if(next.length&&next[0].energy<beam[0].energy-minimumGain){
          beam=next.slice(0,beamWidth).map(({state,proposal,energy})=>{
            const pixels=state.pixels.slice(),coverage=state.coverage.slice(),penaltyCoverage=state.penaltyCoverage.slice();
            const overlap=state.overlap+draw(pixels,proposal.glyph,proposal.kernel,width,coverage,penaltyCoverage,problem.penaltyMinSize);
            return {glyphs:[...state.glyphs,proposal.glyph],pixels,coverage,penaltyCoverage,overlap,energy};
          });
          best=beam[0];++additions;
          record(best,`Add ${best.glyphs.at(-1)!.char} · ${best.glyphs.at(-1)!.size} analysis px`);
          if(additions%8===1)progress(`${size===undefined?"All scales":`${size}px pass`} · glyph ${additions} / ${budget}…`);
          continue;
        }
      }
      // Before advancing, revisit up to eight placements from this pass only.
      if(!refined){
        refined=true;
        const before=best.energy,eligible=best.glyphs.filter(g=>size===undefined||g.size===size);
        const revisit=Array.from({length:Math.min(8,eligible.length)},(_,i)=>eligible[(Math.floor(i*eligible.length/Math.min(8,eligible.length))+searchRound)%eligible.length]);
        for(const old of revisit){
          const remaining=best.glyphs.filter(g=>g!==old),base=paper.slice(),coverage=new Float32Array(width*height),penaltyCoverage=new Float32Array(width*height);
          let overlap=0;
          for(const g of remaining)overlap+=draw(base,g,problem.kernels.find(k=>k.char===g.char&&k.size===g.size&&k.angle===g.angle)!,width,coverage,penaltyCoverage,problem.penaltyMinSize);
          const errors=energyTerms(target,base,width,height),state={glyphs:remaining,pixels:base,coverage,penaltyCoverage,overlap,energy:totalEnergy(base,errors)+problem.overlapPenalty*overlap+(problem.repeatPenalty??0)*repeatError(remaining,width*height,problem.penaltyMinSize)};
          problem.scaleCandidates={};
          const proposal=(yield {problem,state,focus:old})[0];
          if(proposal&&state.energy-proposal.gain<best.energy-1e-9){
            overlap+=draw(base,proposal.glyph,proposal.kernel,width,coverage,penaltyCoverage,problem.penaltyMinSize);blur?.invalidate(base);
            best={glyphs:[...remaining,proposal.glyph],pixels:base,coverage,penaltyCoverage,overlap,energy:state.energy-proposal.gain};
            record(best,`Refine ${old.char} → ${proposal.glyph.char} · ${size??"all"} analysis px`);
          }
        }
        beam=[best];
        // Refinement can reveal another useful addition at the same scale.
        if(canAdd&&best.energy<before-1e-9)continue;
      }
      if(canAdd&&best.energy>1e-9&&searchRound+1<roundLimit){
        ++searchRound;refined=false;problem.hotspotOffset=searchRound*6;
        progress(`${size}px broad pass · search round ${searchRound+1} / ${roundLimit}…`);
        continue;
      }
      converged=canAdd;
      break;
    }
    stages.push({index:passIndex,size:size??null,searchRounds:searchRound+1,startFrame,endFrame:frames.length-1,addedGlyphs:additions-startAdditions,converged});
    if(!converged)break;
    progress(`${size===undefined?"All scales":`${size}px pass`} · no improving proposal in bounded search${passIndex+1<passes.length?"; moving to smaller kernels":""}`);
  }
  if(additions<budget)stopReason=schedule==="coarse-to-fine"?"No improving proposal at any remaining scale (bounded search)":"No improving proposal in the bounded search";
  let hiddenGlyphsRemoved=0;
  const hidden=fullyHiddenGlyphIndices(best.glyphs.map(glyph=>({x:glyph.x,y:glyph.y,size:glyph.size,
    alpha:problem.kernels.find(k=>k.char===glyph.char&&k.size===glyph.size&&k.angle===glyph.angle)!.alpha})),width,height);
  if(hidden.length){
    const remaining=best.glyphs.filter((_,index)=>!hidden.includes(index)),base=paper.slice(),coverage=new Float32Array(width*height),penaltyCoverage=new Float32Array(width*height);
    let overlap=0;
    for(const glyph of remaining)overlap+=draw(base,glyph,problem.kernels.find(k=>k.char===glyph.char&&k.size===glyph.size&&k.angle===glyph.angle)!,width,coverage,penaltyCoverage,problem.penaltyMinSize);
    const errors=energyTerms(target,base,width,height),energy=totalEnergy(base,errors)+problem.overlapPenalty*overlap+(problem.repeatPenalty??0)*repeatError(remaining,width*height,problem.penaltyMinSize);
    if(energy<=best.energy+1e-10){
      hiddenGlyphsRemoved=hidden.length;best={glyphs:remaining,pixels:base,coverage,penaltyCoverage,overlap,energy};
      record(best,`Remove ${hidden.length} fully hidden glyphs`);
    }
  }
  const tileMetrics: Record<string, TextMetrics> = {};
  for (const size of sizes) {
    const displaySize = size * factor, scale = displaySize / metric.width;
    tileMetrics[String(displaySize)] = { width: displaySize, height: displaySize, baseline: metric.baseline * scale, advance: metric.advance * scale,
      fontSize: (metric.fontSize ?? metric.height * .8) * scale, densities: [] };
  }
  const finalWidth = Math.round(outputWidth), finalHeight = Math.round(outputHeight);
  if (finalWidth * finalHeight > 6_000_000) throw new Error("Output is too large; reduce columns");
  const paperRows = new Uint8Array(finalHeight * 3);
  for (let i = 0; i < paperRows.length; ++i) paperRows[i] = Math.round(paperColor[i % 3] * 255);
  return { width: finalWidth, height: finalHeight, columns: 0, rows: 0, cellWidth: 0, cellHeight: 0,
    text: "", colors: new Uint8Array(), paper: paperRows, tiles: frames.at(-1)!.glyphs, tileMetrics, glyphAtlas:metric,
    pursuit: { targetSharpen:problem.targetSharpen,frames, stopReason, analysisWidth: width, analysisHeight: height, budget, beamWidth, edgeWeight, scales: sizes, backend: "cpu", blurStrength, blurCurve:blurModel.curve, schedule, stages, coarseThreshold, coarseRounds, effort, vocabulary, kernelCounts, charactersByScale, overlapLimit:maxOverlap,overlapPenalty:problem.overlapPenalty,repeatPenalty:problem.repeatPenalty,penaltyMinSize:problem.penaltyMinSize, hiddenGlyphsRemoved, elapsedMs: performance.now() - started } };
}

/** Back-to-front visibility uses stroke alpha rather than bounding rectangles. */
export function fullyHiddenGlyphIndices(strokes:readonly {x:number;y:number;size:number;alpha:Float32Array}[],width:number,height:number):number[] {
  const transmission=new Float32Array(width*height).fill(1),hidden:number[]=[];
  for(let index=strokes.length-1;index>=0;--index){
    const stroke=strokes[index];let total=0,visible=0;
    for(let y=0;y<stroke.size;++y)for(let x=0;x<stroke.size;++x){
      const alpha=stroke.alpha[y*stroke.size+x],i=(stroke.y+y)*width+stroke.x+x;
      total+=alpha;visible+=alpha*transmission[i];transmission[i]*=1-alpha;
    }
    if(total>0&&visible/total<1e-6)hidden.push(index);
  }
  return hidden;
}

/** Synchronous CPU reference used by native runs and deterministic tests. */
export function renderPursuit(pixels: Uint8Array, sourceWidth: number, sourceHeight: number, settings: RenderSettings, progress: (message: string) => void = () => {}): RenderResult {
  if(settings.pursuit_schedule==="independent"){
    const started=performance.now(),setup=preparePursuit(pixels,sourceWidth,sourceHeight,settings);
    const results=setup.sizes.filter(size=>setup.kernelCounts[String(size)]>0).map(size=>
      renderPursuit(pixels,sourceWidth,sourceHeight,independentLayerSettings(setup,settings,size),message=>progress(`${size}px independent · ${message}`)));
    return combineIndependentLayers(setup,results,started);
  }
  const search=searchPursuit(pixels,sourceWidth,sourceHeight,settings,progress);
  let step=search.next();
  while(!step.done) {
    const {problem,state,focus}=step.value;
    step=search.next(proposals(problem,state,focus));
  }
  return step.value;
}

export async function renderPursuitAsync(pixels: Uint8Array, sourceWidth: number, sourceHeight: number, settings: RenderSettings,
  progress: (message: string) => void = () => {}, scorerFactory: () => Promise<CandidateScorer> = createWebGPUScorer): Promise<RenderResult> {
  const started=performance.now(), requested=settings.pursuit_backend ?? "auto";
  if(!["auto","cpu","webgpu"].includes(requested)) throw new Error("Invalid candidate scoring backend");
  if(settings.pursuit_schedule==="independent"){
    const setup=preparePursuit(pixels,sourceWidth,sourceHeight,settings);
    const results=await Promise.all(setup.sizes.filter(size=>setup.kernelCounts[String(size)]>0).map(size=>
      renderPursuitAsync(pixels,sourceWidth,sourceHeight,independentLayerSettings(setup,settings,size),message=>progress(`${size}px independent · ${message}`),scorerFactory)));
    return combineIndependentLayers(setup,results,started);
  }
  let residentReason:string|undefined;
  if((settings.pursuit_repeat_penalty??0)>0 || ((settings.pursuit_penalty_min_size??0)>0 && (settings.pursuit_overlap_penalty??0)>0))residentReason="Scale-dependent penalties use CPU acceptance with optional GPU candidate scoring";
  if(!residentReason && requested!=="cpu" && scorerFactory===createWebGPUScorer && (settings.pursuit_beam??1)===1 && (settings.pursuit_schedule??"coarse-to-fine")==="coarse-to-fine"){
    try{const {renderResidentPursuit}=await import("./pursuit-resident");return await renderResidentPursuit(pixels,sourceWidth,sourceHeight,settings,progress);}
    catch(error){residentReason=error instanceof Error?error.message:String(error);progress(`GPU loop unavailable · ${residentReason} · trying fallback…`);}
  }
  let scorer:CandidateScorer|undefined, reason:string|undefined=residentReason, gpuBatches=0, gpuCandidates=0, gpuBatchMs=0, maxError:number|undefined, cpuBatches=0;
  if(requested!=="cpu") {
    try {scorer=await scorerFactory();progress("WebGPU candidate scoring · CPU verifies accepted edits");}
    catch(error) {reason=error instanceof Error?error.message:String(error);progress(`CPU scoring · ${reason}`);}
  }
  const search=searchPursuit(pixels,sourceWidth,sourceHeight,settings,progress);
  try {
    let step=search.next();
    while(!step.done) {
      const {problem,state,focus}=step.value;
      if(!scorer) {++cpuBatches;step=search.next(proposals(problem,state,focus));continue;}
      problem.coverage=state.coverage;problem.penaltyCoverage=state.penaltyCoverage;
      const specs=candidateSpecs(problem,state,focus);
      const batch:CandidateBatch={width:problem.width,height:problem.height,target:problem.target,pixels:state.pixels,
        guidance:problem.blur?.guide(state.pixels),edgeWeight:problem.edgeWeight,color:problem.color,coverage:state.coverage,penaltyCoverage:state.penaltyCoverage,penaltyMinSize:problem.penaltyMinSize,maxOverlap:problem.maxOverlap,overlapPenalty:problem.overlapPenalty,kernels:problem.kernels,candidates:specs};
      let scores:Float32Array;
      try {
        const batchStarted=performance.now();
        scores=await scorer.score(batch);
        gpuBatchMs+=performance.now()-batchStarted;
        if(scores.length!==specs.length || scores.some(value=>!Number.isFinite(value))) throw new Error("WebGPU returned invalid candidate scores");
        if(maxError===undefined && specs.length) {
          // Uniformly spaced probes cover glyphs, scales, orientations and positions.
          const indices=[...new Set(Array.from({length:Math.min(48,specs.length)},(_,i)=>Math.round(i*(specs.length-1)/Math.max(1,Math.min(48,specs.length)-1))))];
          const reference=referenceCandidateScores({...batch,candidates:indices.map(i=>specs[i])});
          maxError=0;
          for(let j=0;j<indices.length;++j) {
            const error=Math.abs(scores[indices[j]]-reference[j]);
            maxError=Math.max(maxError,error);
            if(error>5e-6+Math.abs(reference[j])*.002) throw new Error("WebGPU/CPU score self-check failed");
          }
        }
        ++gpuBatches;gpuCandidates+=specs.length;
        problem.evaluated+=specs.length;
        for(const spec of specs) {const size=String(problem.kernels[spec.kernel].size);problem.scaleCandidates[size]=(problem.scaleCandidates[size]??0)+1;}
      } catch(error) {
        reason=error instanceof Error?error.message:String(error);
        scorer.dispose();scorer=undefined;
        progress(`CPU fallback · ${reason}`);
        ++cpuBatches;step=search.next(proposals(problem,state,focus));continue;
      }
      step=search.next(finishProposals(problem,state,specs,scores));
    }
    Object.assign(step.value.pursuit!,{backend:gpuBatches?(cpuBatches?"mixed":"webgpu"):"cpu",backendReason:reason,
      gpuBatches,gpuCandidates,gpuBatchMs,gpuValidationMaxError:maxError,elapsedMs:performance.now()-started});
    return step.value;
  } finally {scorer?.dispose();}
}

/** Pure replay: arbitrary seek, including iteration zero, without another render. */
export function pursuitFrame(result: RenderResult, iteration: number): RenderResult {
  if (!result.pursuit) return result;
  const frame = result.pursuit.frames[Math.max(0, Math.min(result.pursuit.frames.length - 1, Number.isFinite(iteration) ? Math.round(iteration) : 0))];
  return { ...result, tiles: frame.glyphs };
}
