import { expect, test } from "bun:test";
import { energyTerms, fullyHiddenGlyphIndices, glyphAllowedAtScale, pursuitFrame, renderPursuit } from "./pursuit";
import { PURSUIT_CHARACTERS } from "./text-view";
import type { RenderSettings } from "./renderer";

function settings(overrides: Partial<RenderSettings> = {}): RenderSettings {
  const mask = (char: string) => Array.from({length: 64}, (_, i) => char === "X" ? 1 : char === "|" ? Number(i % 8 === 4) : char === "-" ? Number(Math.floor(i / 8) === 4) : 0);
  return { columns: 24, font_size: 12, layout: "pursuit", use_edge: false, edge_method: "acerola",
    edge_sensitivity: 50, edge_amount: 100, use_color: true, background: "#000000", background_mode: "solid", foreground: "#ffffff",
    pursuit_coarse_threshold:0, pursuit_schedule:"all", pursuit_blur:0, pursuit_budget: 6, pursuit_beam: 2, pursuit_edge_weight: 3,
    pursuit_metrics: {width:8,height:8,baseline:7,advance:8,fontSize:8,densities:[],kernelCharacters:" X|-",masks:[..." X|-"].map(mask)}, ...overrides };
}
function image(width = 96, height = 96): Uint8Array {
  const pixels = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; ++y) for (let x = 0; x < width; ++x) {
    const i = (y * width + x) * 4, bright = x >= width / 4 && x < width / 2 && y >= height / 4 && y < height * 3 / 4;
    pixels[i] = bright ? 220 : 0; pixels[i + 1] = bright ? 100 : 0; pixels[i + 2] = bright ? 40 : 0; pixels[i + 3] = 255;
  }
  return pixels;
}

test("omitted overlap limit defaults to unrestricted layering",()=>{
  const result=renderPursuit(image(),96,96,settings({pursuit_budget:1}));
  expect(result.pursuit!.overlapLimit).toBe(1);
});

test("edits improve energy or remove hidden glyphs within the visible budget", () => {
  const result = renderPursuit(image(), 96, 96, settings());
  const frames = result.pursuit!.frames;
  expect(frames.length).toBeGreaterThan(2);
  expect(frames[0].glyphs).toEqual([]);
  for (let i = 1; i < frames.length; ++i) {
    if(frames[i].action.startsWith("Remove")){
      expect(frames[i].energy).toBeLessThanOrEqual(frames[i-1].energy+1e-10);
      expect(frames[i].glyphs.length).toBeLessThan(frames[i-1].glyphs.length);
    }else expect(frames[i].energy).toBeLessThan(frames[i-1].energy);
    expect(frames[i].energy).toBeCloseTo(frames[i].pixelError + 3 * frames[i].contourError, 10);
    expect(frames[i].glyphs.length).toBeLessThanOrEqual(6);
  }
  expect(frames.at(-1)!.energy).toBeLessThan(frames[0].energy * .5);
  expect(result.tiles).toEqual(frames.at(-1)!.glyphs);
});

test("scrubbing replays exact snapshots backward and forward without changing history", () => {
  const result = renderPursuit(image(), 96, 96, settings({pursuit_budget: 3, pursuit_beam: 1}));
  const snapshot = JSON.stringify(result.pursuit);
  expect(pursuitFrame(result, 0).tiles).toEqual([]);
  const end = pursuitFrame(result, 999).tiles;
  expect(pursuitFrame(result, 1).tiles).toEqual(result.pursuit!.frames[1].glyphs);
  expect(pursuitFrame(result, -1).tiles).toEqual([]);
  expect(pursuitFrame(result, result.pursuit!.frames.length - 1).tiles).toEqual(end);
  expect(JSON.stringify(result.pursuit)).toBe(snapshot);
});

test("input resize and output columns preserve normalized composition", () => {
  const first = renderPursuit(image(96,96), 96,96,settings({pursuit_budget: 3,pursuit_beam:1}));
  const second = renderPursuit(image(192,192), 192,192,settings({columns:48,pursuit_blur:0,pursuit_budget:3,pursuit_beam:1}));
  expect(second.tiles!.length).toBe(first.tiles!.length);
  for (let i=0; i<first.tiles!.length; ++i) {
    const a = first.tiles![i], b = second.tiles![i];
    expect(b.char).toBe(a.char); expect(b.angle).toBe(a.angle);
    expect(b.x / second.width).toBe(a.x / first.width);
    expect(b.y / second.height).toBe(a.y / first.height);
    expect(b.size / second.width).toBe(a.size / first.width);
    expect(b.color).toEqual(a.color);
  }
});

test("paper-only input stops cleanly with an inspectable iteration zero", () => {
  const pixels = new Uint8Array(16*16*4); for(let i=3;i<pixels.length;i+=4) pixels[i]=255;
  const result = renderPursuit(pixels,16,16,settings());
  expect(result.pursuit!.frames.length).toBe(1);
  expect(result.pursuit!.frames[0].energy).toBe(0);
  expect(result.pursuit!.stopReason).toContain("No improving");
});

test("transparent hidden RGB does not introduce glyphs", () => {
  const pixels = new Uint8Array(16*16*4).fill(255); for(let i=3;i<pixels.length;i+=4) pixels[i]=0;
  const result = renderPursuit(pixels,16,16,settings());
  expect(result.tiles).toEqual([]);
});

test("pixel and contour errors distinguish an exactly matched image from a displaced edge", () => {
  const target = new Float32Array(8*8*3), shifted = target.slice();
  for(let y=0;y<8;++y) for(let c=0;c<3;++c) {target[(y*8+3)*3+c]=1; shifted[(y*8+4)*3+c]=1;}
  expect(energyTerms(target,target,8,8)).toEqual({pixel:0,contour:0});
  const errors=energyTerms(target,shifted,8,8);
  expect(errors.pixel).toBeGreaterThan(0); expect(errors.contour).toBeGreaterThan(0);
});

test.each([-1,0,2049,Infinity,NaN])("rejects invalid glyph budget %s", budget => {
  expect(() => renderPursuit(image(),96,96,settings({pursuit_budget:budget}))).toThrow("budget");
});

test("thin off-center contours retain fine-scale candidates instead of stopping at coarse fills", () => {
  const pixels = new Uint8Array(96*96*4).fill(255);
  for(let y=12;y<84;++y) for(const x of [29,57]) {
    const i=(y*96+x)*4; pixels[i]=pixels[i+1]=pixels[i+2]=0;
  }
  const result = renderPursuit(pixels,96,96,settings({background:"#ffffff",pursuit_scales:[48,24,12,6,3],pursuit_budget:24,pursuit_edge_weight:8,pursuit_max_overlap:.35}));
  expect(result.pursuit!.frames.length).toBeGreaterThan(12);
  expect(result.tiles!.some(g=>g.size<=6)).toBe(true);
  expect(result.pursuit!.frames.at(-1)!.energy).toBeLessThan(result.pursuit!.frames[0].energy);
});


test("every search step evaluates every sampled scale, including the finest", () => {
  const result = renderPursuit(image(),96,96,settings({pursuit_budget:4,pursuit_beam:1}));
  for(const frame of result.pursuit!.frames.slice(1)) {
    for(const size of result.pursuit!.scales) expect(frame.scaleCandidates[String(size)]).toBeGreaterThan(0);
  }
  expect(result.pursuit!.scales).toEqual([32,16,8]);
});

test("signed residual snapshots distinguish missing brightness and excess brightness", () => {
  const dark = renderPursuit(image(),96,96,settings({pursuit_budget:1,pursuit_beam:1}));
  const light = renderPursuit(image(),96,96,settings({background:"#ffffff",pursuit_budget:1,pursuit_beam:1}));
  expect(dark.pursuit!.frames[0].residual.some(v=>v>128)).toBe(true);
  expect(light.pursuit!.frames[0].residual.some(v=>v<128)).toBe(true);
  for(const frame of dark.pursuit!.frames) {
    expect(frame.residual.length).toBe(frame.residualWidth*frame.residualHeight);
    expect(Math.max(frame.residualWidth,frame.residualHeight)).toBeLessThanOrEqual(96);
  }
});

test("contour fitting cannot introduce a color tint into a grayscale original", () => {
  const pixels=image();
  for(let i=0;i<pixels.length;i+=4) pixels[i+1]=pixels[i+2]=pixels[i];
  const result=renderPursuit(pixels,96,96,settings({pursuit_blur:0,pursuit_budget:3,pursuit_beam:1,pursuit_edge_weight:8}));
  for(const frame of result.pursuit!.frames) for(const glyph of frame.glyphs) {
    expect(glyph.color[0]).toBeCloseTo(glyph.color[1],6);
    expect(glyph.color[1]).toBeCloseTo(glyph.color[2],6);
  }
});


test("custom Unicode shapes are allowed by unrestricted and scale-aware profiles", () => {
  expect(PURSUIT_CHARACTERS).toMatch(/^[\x20-\x7e]+$/);
  for(const char of "/|\\()-+*#%@") expect(PURSUIT_CHARACTERS).toContain(char);
  for(const char of "░▒▓█▀▄▌▐╭╮╰╯┌┐└┘├┤┬┴┼─│") {
    expect(PURSUIT_CHARACTERS).not.toContain(char);
    for(const profile of ["full", "scale-aware"] as const)
      for(const size of [48,24,12,6,3]) expect(glyphAllowedAtScale(char,size,48,profile)).toBe(true);
  }
  expect(new Set([...PURSUIT_CHARACTERS]).size).toBe([...PURSUIT_CHARACTERS].length);
});


test("coarse scales use simple geometry while fine scales retain the full vocabulary",()=>{
  for(const char of "/|\\()~_") expect(glyphAllowedAtScale(char,48,48,"scale-aware")).toBe(true);
  for(const char of ".,:;*#%@") {
    expect(glyphAllowedAtScale(char,48,48,"scale-aware")).toBe(false);
    expect(glyphAllowedAtScale(char,12,48,"scale-aware")).toBe(false);
    expect(glyphAllowedAtScale(char,3,48,"scale-aware")).toBe(true);
    expect(glyphAllowedAtScale(char,48,48,"full")).toBe(true);
  }
  expect(glyphAllowedAtScale("+",48,48,"scale-aware")).toBe(false);
  expect(glyphAllowedAtScale("+",12,48,"scale-aware")).toBe(true);
  expect(glyphAllowedAtScale("custom-brush",48,48,"scale-aware")).toBe(true);
});


test("a strict ink-overlap budget prevents repeated paint in the final composition",()=>{
  const config=settings({pursuit_budget:12,pursuit_beam:1,pursuit_max_overlap:.05});
  const result=renderPursuit(image(),96,96,config),coverage=new Float64Array(192*192);
  for(const glyph of result.tiles!) {
    const original=config.pursuit_metrics!.masks![[...config.pursuit_metrics!.kernelCharacters!].indexOf(glyph.char)];
    const size=glyph.size;let total=0,covered=0;const samples:{i:number;alpha:number}[]=[];
    for(let y=0;y<size;++y)for(let x=0;x<size;++x){
      let alpha=0,count=0;
      for(let sy=Math.floor(y*8/size);sy<Math.ceil((y+1)*8/size);++sy)
        for(let sx=Math.floor(x*8/size);sx<Math.ceil((x+1)*8/size);++sx){alpha+=glyph.angle===90?original[(7-sx)*8+sy]:original[sy*8+sx];++count;}
      alpha/=count;const i=(glyph.y+y)*192+glyph.x+x;
      total+=alpha;covered+=alpha*coverage[i];samples.push({i,alpha});
    }
    expect(covered/Math.max(total,1e-10)).toBeLessThanOrEqual(.050001);
    for(const {i,alpha} of samples)coverage[i]+=alpha*(1-coverage[i]);
  }
  expect(result.pursuit!.overlapLimit).toBe(.05);
});

test("visibility cleanup identifies hidden ink without deleting partial strokes or crossings",()=>{
  const solid=new Float32Array(16).fill(1),line=Float32Array.from({length:16},(_,i)=>Number(i%4===2));
  expect(fullyHiddenGlyphIndices([{x:0,y:0,size:4,alpha:solid},{x:0,y:0,size:4,alpha:solid}],8,8)).toEqual([0]);
  expect(fullyHiddenGlyphIndices([{x:0,y:0,size:4,alpha:solid},{x:2,y:0,size:4,alpha:solid}],8,8)).toEqual([]);
  expect(fullyHiddenGlyphIndices([{x:0,y:0,size:4,alpha:solid},{x:0,y:0,size:4,alpha:line}],8,8)).toEqual([]);
});


test("iteration snapshots share unchanged display glyphs instead of copying each placement", () => {
  const frames=renderPursuit(image(),96,96,settings()).pursuit!.frames;
  expect(frames.length).toBeGreaterThan(2);
  expect(frames[1].glyphs[0]).toBe(frames[2].glyphs[0]);
  const copy=structuredClone(frames);
  expect(copy[1].glyphs[0]).toBe(copy[2].glyphs[0]);
});

test("extended budget accepts 2048 and retains an inspectable blank frame", () => {
  const result=renderPursuit(new Uint8Array(8*8*4),8,8,settings({pursuit_budget:2048}));
  expect(result.pursuit!.budget).toBe(2048);
  expect(result.pursuit!.frames).toHaveLength(1);
});


test("fast search reduces scoring work while retaining all scales and exact monotonic energy", () => {
  const fast=renderPursuit(image(),96,96,settings({pursuit_effort:"fast",pursuit_beam:1,pursuit_budget:4})).pursuit!;
  const thorough=renderPursuit(image(),96,96,settings({pursuit_effort:"thorough",pursuit_beam:1,pursuit_budget:4})).pursuit!;
  expect(fast.frames.at(-1)!.candidates).toBeLessThan(thorough.frames.at(-1)!.candidates);
  for (let i=1;i<fast.frames.length;++i) {
    if(fast.frames[i].action.startsWith("Remove")){
      expect(fast.frames[i].energy).toBeLessThanOrEqual(fast.frames[i-1].energy+1e-10);
      expect(fast.frames[i].glyphs.length).toBeLessThan(fast.frames[i-1].glyphs.length);
    }else expect(fast.frames[i].energy).toBeLessThan(fast.frames[i-1].energy);
    for (const size of fast.scales) expect(fast.frames[i].scaleCandidates[String(size)]).toBeGreaterThan(0);
  }
  expect(fast.effort).toBe("fast"); expect(thorough.effort).toBe("thorough");
});

test("kernel-linked blur uses one decreasing objective across different glyph sizes",()=>{
  const result=renderPursuit(image(),96,96,settings({pursuit_blur:1,pursuit_budget:8,pursuit_beam:1}));
  const trace=result.pursuit!;
  expect(trace.blurStrength).toBe(1);expect(trace.blurCurve.map(p=>p.size)).toEqual(trace.scales);
  expect(trace.frames.length).toBeGreaterThan(2);
  for(let i=1;i<trace.frames.length;++i){
    if(trace.frames[i].action.startsWith("Remove"))expect(trace.frames[i].energy).toBeLessThanOrEqual(trace.frames[i-1].energy+1e-10);
    else expect(trace.frames[i].energy).toBeLessThan(trace.frames[i-1].energy);
  }
  expect(trace.blurCurve[0].sigma).toBeGreaterThan(0);
});

test("painterly passes converge from coarse to fine and preserve the underpainting",()=>{
  const pixels=image();
  for(let y=70;y<73;++y)for(let x=75;x<78;++x)for(let c=0;c<3;++c)pixels[(y*96+x)*4+c]=255;
  const result=renderPursuit(pixels,96,96,settings({pursuit_schedule:"coarse-to-fine",pursuit_budget:96,pursuit_blur:1,pursuit_beam:1,pursuit_max_overlap:.35}));
  const trace=result.pursuit!;
  expect(trace.schedule).toBe("coarse-to-fine");
  expect(trace.stages.map(s=>s.size)).toEqual(trace.scales);
  expect(trace.stages.every(s=>s.converged)).toBe(true);
  let previous=Infinity;
  for(let i=1;i<trace.frames.length;++i){
    const frame=trace.frames[i];if(frame.action.startsWith("Remove"))continue;
    expect(frame.activeScale!).toBeLessThanOrEqual(previous);previous=frame.activeScale!;
    expect(Object.keys(frame.scaleCandidates)).toEqual([String(frame.activeScale)]);
    expect(frame.energy).toBeLessThan(trace.frames[i-1].energy);
    if(frame.action.startsWith("Add")&&frame.stage>trace.frames[i-1].stage)
      for(const glyph of trace.frames[i-1].glyphs)expect(frame.glyphs).toContain(glyph);
  }
  expect(new Set(trace.frames.slice(1).map(f=>f.activeScale)).size).toBeGreaterThan(1);
});

test("a scale with no eligible glyphs advances to a finer scale instead of stopping the run",()=>{
  const metrics={width:8,height:8,baseline:7,advance:8,fontSize:8,densities:[],kernelCharacters:" :",masks:[Array(64).fill(0),Array(64).fill(1)]};
  const trace=renderPursuit(image(),96,96,settings({pursuit_schedule:"coarse-to-fine",pursuit_scales:[48,24,12,6,3],pursuit_metrics:metrics,pursuit_budget:1,pursuit_beam:1})).pursuit!;
  expect(trace.stages.map(s=>s.size)).toEqual(trace.scales);
  for(const stage of trace.stages.slice(0,-1)){expect(stage.addedGlyphs).toBe(0);expect(stage.converged).toBe(true);}
  expect(trace.frames.at(-1)!.activeScale).toBe(3);
  expect(trace.stages.at(-1)!.addedGlyphs).toBe(1);
});

test("global glyph budget stops the current pass without falsely claiming convergence",()=>{
  const trace=renderPursuit(image(),96,96,settings({pursuit_schedule:"coarse-to-fine",pursuit_budget:1,pursuit_beam:1})).pursuit!;
  expect(trace.stages).toHaveLength(1);expect(trace.stages[0].size).toBe(trace.scales[0]);
  expect(trace.stages[0].converged).toBe(false);expect(trace.stopReason).toBe("Glyph budget reached");
  expect(trace.frames.at(-1)!.glyphs).toHaveLength(1);
});


test.each(["_","|","L"])("off-centre %s ink aligns to the original contour in one coarse glyph",char=>{
  const mask=Array.from({length:64},(_,i)=>Number(char==="_"?Math.floor(i/8)===7:char==="|"?i%8===1:i%8===1||Math.floor(i/8)===7));
  const pixels=new Uint8Array(128*128*4);for(let i=3;i<pixels.length;i+=4)pixels[i]=255;
  for(let y=0;y<32;++y)for(let x=0;x<32;++x)for(let c=0;c<3;++c)pixels[((44+y)*128+44+x)*4+c]=mask[Math.floor(y/4)*8+Math.floor(x/4)]*255;
  const result=renderPursuit(pixels,128,128,settings({columns:24,pursuit_schedule:"coarse-to-fine",pursuit_resolution:128,pursuit_budget:1,pursuit_beam:1,pursuit_blur:1,use_color:false,
    pursuit_metrics:{width:8,height:8,baseline:7,advance:8,fontSize:8,densities:[],kernelCharacters:` ${char}`,masks:[Array(64).fill(0),mask]}}));
  expect(result.tiles).toHaveLength(1);
  const glyph=result.tiles![0],factor=result.width/128;
  expect(glyph.x/factor).toBe(44);expect(glyph.y/factor).toBe(44);expect(glyph.size/factor).toBe(32);
  expect(glyph.angle).toBe(0);expect(glyph.char).toBe(char);
  expect(result.pursuit!.frames.at(-1)!.energy).toBeLessThan(1e-10);
});

test("additional broad rounds explore residual regions missed by the first hotspot group",()=>{
  const pixels=new Uint8Array(128*128*4);for(let i=3;i<pixels.length;i+=4)pixels[i]=255;
  for(const x of [12,28,44,60,76,92])for(let c=0;c<3;++c)pixels[(12*128+x)*4+c]=255;
  for(let y=80;y<112;++y)for(let x=100;x<104;++x)for(let c=0;c<3;++c)pixels[(y*128+x)*4+c]=20;
  const metrics={width:8,height:8,baseline:7,advance:8,fontSize:8,densities:[],kernelCharacters:" |",masks:[Array(64).fill(0),Array.from({length:64},(_,i)=>Number(i%8===1))]};
  const run=(rounds:number)=>renderPursuit(pixels,128,128,settings({pursuit_schedule:"coarse-to-fine",pursuit_resolution:128,pursuit_budget:64,pursuit_beam:1,pursuit_blur:1,pursuit_metrics:metrics,pursuit_coarse_rounds:rounds,use_color:false}));
  const once=run(1),more=run(3);
  expect(once.pursuit!.stages[0].searchRounds).toBe(1);
  expect(more.pursuit!.stages[0].searchRounds).toBe(3);
  expect(more.pursuit!.stages[0].addedGlyphs).toBeGreaterThan(once.pursuit!.stages[0].addedGlyphs);
  expect(more.pursuit!.frames[more.pursuit!.stages[0].endFrame].energy).toBeLessThan(once.pursuit!.frames[once.pursuit!.stages[0].endFrame].energy);
  for(const pass of more.pursuit!.stages.filter(s=>s.size!==null&&s.size<16))expect(pass.searchRounds).toBe(1);
});

test.each([0,9,1.5,NaN])("rejects invalid broad search rounds %s",rounds=>{
  expect(()=>renderPursuit(image(),96,96,settings({pursuit_coarse_rounds:rounds}))).toThrow("broad search rounds");
});
