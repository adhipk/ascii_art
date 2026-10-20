import {expect,test} from "bun:test";
import {splitGlyphs, defaultPursuitScales,defaultScaleCharacters,normalizeScaleCharacters} from "./scale-characters";
import {preparePursuit,renderPursuit} from "./pursuit";
import {createResidentPlan} from "./pursuit-resident-plan";
import type {RenderSettings} from "./renderer";

const pixels=new Uint8Array(32*32*4);
for(let y=0;y<32;y++)for(let x=0;x<32;x++) {
  const i=(y*32+x)*4;pixels[i]=pixels[i+1]=pixels[i+2]=x>8&&x<24&&y>8&&y<24?220:0;pixels[i+3]=255;
}
function settings():RenderSettings {
  return {columns:24,font_size:12,layout:"pursuit",pursuit_resolution:128,pursuit_scales:[16,8,4],pursuit_budget:4,pursuit_beam:1,pursuit_blur:0,
    pursuit_characters_by_scale:{"16":"X","8":"","4":"|"},
    use_color:false,use_edge:false,edge_method:"acerola",edge_sensitivity:50,edge_amount:0,background:"#000000",background_mode:"solid",foreground:"#ffffff",
    pursuit_metrics:{width:8,height:8,baseline:7,advance:8,fontSize:8,densities:[],kernelCharacters:" X|",
      masks:[Array(64).fill(0),Array(64).fill(1),Array.from({length:64},(_,i)=>Number(i%8===4))]}};
}

test("scale character maps normalize duplicates, permit skipping and reject unsupported input",()=>{
  expect(defaultPursuitScales(128)).toEqual([32,16,8]);
  expect(defaultPursuitScales(320)).toEqual([32,16,8]);
  expect(normalizeScaleCharacters({"16":"XX||","8":""},[16,8],"X|")).toEqual({"16":"X|","8":""});
  for(const map of [{"17":"X"},{"16":"█"},{"16":"\n"},{"16":"A"}])
    expect(()=>normalizeScaleCharacters(map,[16,8],"X|")).toThrow();
});

test("Unicode defaults progress from filled shapes through geometry to fine marks",()=>{
  expect(defaultScaleCharacters(32,32,"shape-detail")).toContain("█");
  expect(defaultScaleCharacters(16,32,"shape-detail")).toContain("┼");
  expect(defaultScaleCharacters(8,32,"shape-detail")).toContain("·");
  expect(defaultScaleCharacters(3,32,"shape-detail")).toContain("·");
});

test("Unicode graphemes retain emoji sequences and exact membership through CPU and GPU planning",()=>{
  const glyphs=["█","界","𝄞","😀","e\u0301","👩‍💻"];
  expect(splitGlyphs(glyphs.join(""))).toEqual(glyphs);
  expect(normalizeScaleCharacters({16:glyphs.join("")+"😀"},[16],glyphs)).toEqual({16:glyphs.join("")});
  expect(()=>normalizeScaleCharacters({16:"👩"},[16],["👩‍💻"])).toThrow("no measured font mask");
  const config=settings();config.pursuit_metrics!.kernelCharacters="😀e\u0301";
  config.pursuit_metrics!.kernelGlyphs=["😀","e\u0301"];
  config.pursuit_metrics!.masks=config.pursuit_metrics!.masks!.slice(1);
  config.pursuit_characters_by_scale={16:"😀",8:"",4:"e\u0301"};
  const setup=preparePursuit(pixels,32,32,config),plan=createResidentPlan(pixels,32,32,config);
  for(const kernel of [...setup.problem.kernels,...plan.kernels]) expect(["😀","e\u0301"]).toContain(kernel.char);
  const result=renderPursuit(pixels,32,32,config);
  expect(result.glyphAtlas?.kernelGlyphs).toEqual(["😀","e\u0301"]);
  expect(result.tiles!.length).toBeGreaterThan(0);
  for(const tile of result.tiles!)expect(["😀","e\u0301"]).toContain(tile.char);
});

test("CPU setup and resident GPU plan obey identical explicit sets and skipped scales",()=>{
  const config=settings(),setup=preparePursuit(pixels,32,32,config),gpu=createResidentPlan(pixels,32,32,config);
  expect(setup.charactersByScale).toEqual({"16":"X","8":"","4":"|"});
  expect(setup.kernelCounts["8"]).toBe(0);
  expect(gpu.stages.find(stage=>stage.size===8)!.count).toBe(0);
  for(const kernel of [...setup.problem.kernels,...gpu.kernels]) expect(config.pursuit_characters_by_scale![String(kernel.size)]).toContain(kernel.char);
  const result=renderPursuit(pixels,32,32,config);
  expect(result.pursuit!.charactersByScale).toEqual(setup.charactersByScale);
  const factor=result.width/result.pursuit!.analysisWidth;
  for(const glyph of result.tiles!) expect(config.pursuit_characters_by_scale![String(glyph.size/factor)]).toContain(glyph.char);
});

test("disabling all scales reports that no visible dictionary is available",()=>{
  expect(()=>preparePursuit(pixels,32,32,{...settings(),pursuit_characters_by_scale:{"16":"","8":"","4":""}})).toThrow("no visible kernels");
});

test("solid square switch adds exact masks at enabled scales and preserves skips",()=>{
  const config={...settings(),pursuit_solid_brush:true};
  const setup=preparePursuit(pixels,32,32,config),plan=createResidentPlan(pixels,32,32,config);
  for(const kernels of [setup.problem.kernels,plan.kernels]) {
    for(const size of [16,4])expect(kernels.some(k=>k.size===size&&k.alpha.every(a=>a===1))).toBe(true);
    expect(kernels.some(k=>k.size===8)).toBe(false);
  }
  expect(preparePursuit(pixels,32,32,settings()).problem.kernels.some(k=>k.char==="█")).toBe(false);
});
