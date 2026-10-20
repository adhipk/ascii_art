import {expect,test} from "bun:test";
import {referenceCandidateScores,renderPursuit,renderPursuitAsync} from "./pursuit";
import type {RenderResult,RenderSettings} from "./renderer";
import type {CandidateBatch} from "./pursuit-gpu";

const pixels=new Uint8Array(32*32*4);
for(let y=0;y<32;y++)for(let x=0;x<32;x++){
  const index=(y*32+x)*4,brightness=x>6&&x<25&&y>4&&y<28?180:30;
  pixels[index]=brightness;pixels[index+1]=brightness;pixels[index+2]=brightness;pixels[index+3]=255;
}
const config:RenderSettings={columns:24,font_size:12,layout:"pursuit",use_edge:false,edge_method:"acerola",edge_sensitivity:50,edge_amount:0,
  use_color:false,background:"#000000",background_mode:"source",foreground:"#ffffff",pursuit_resolution:128,
  pursuit_budget:4,pursuit_beam:1,pursuit_edge_weight:3,pursuit_effort:"fast",pursuit_scales:[32,16,8],
  pursuit_characters_by_scale:{32:"X|",16:"X|",8:"X|"},
  pursuit_metrics:{width:4,height:4,baseline:4,advance:4,fontSize:4,densities:[],kernelCharacters:" X|",
    masks:[Array(16).fill(0),Array(16).fill(.6),Array.from({length:16},(_,index)=>index%4===2?1:0)]}};

function rendering(result:RenderResult):unknown {
  return {tiles:result.tiles,paper:result.paper,tileMetrics:result.tileMetrics,
    frames:result.pursuit!.frames.map(frame=>({glyphs:frame.glyphs,energy:frame.energy,pixel:frame.pixelError,contour:frame.contourError,residual:frame.residual}))};
}

for(const schedule of ["all","coarse-to-fine","independent"] as const)for(const blur of [0,2])
test(`zero new effects equal omitted effects for ${schedule}, blur ${blur}`,()=>{
  const settings={...config,pursuit_schedule:schedule,pursuit_blur:blur};
  const reference=renderPursuit(pixels,32,32,settings);
  const zero=renderPursuit(pixels,32,32,{...settings,pursuit_overlap_penalty:0,pursuit_repeat_penalty:0,pursuit_penalty_min_size:0,pursuit_detail_sharpen:0});
  const cutoff=renderPursuit(pixels,32,32,{...settings,pursuit_overlap_penalty:0,pursuit_repeat_penalty:0,pursuit_penalty_min_size:24,pursuit_detail_sharpen:0});
  expect(rendering(zero)).toEqual(rendering(reference));
  expect(rendering(cutoff)).toEqual(rendering(reference));
});

test("zero effects preserve GPU candidate scoring and acceptance",async()=>{
  const settings={...config,pursuit_schedule:"independent" as const,pursuit_blur:2};
  const scorer=async()=>({score:async(batch:CandidateBatch)=>Float32Array.from(referenceCandidateScores(batch)),dispose(){}});
  const reference=await renderPursuitAsync(pixels,32,32,settings,()=>{},scorer);
  const zero=await renderPursuitAsync(pixels,32,32,{...settings,pursuit_overlap_penalty:0,pursuit_repeat_penalty:0,pursuit_detail_sharpen:0},()=>{},scorer);
  expect(rendering(zero)).toEqual(rendering(reference));
});

test("independent composition is an algorithm change even when all effects are zero",()=>{
  const settings={...config,pursuit_blur:0,pursuit_overlap_penalty:0,pursuit_repeat_penalty:0,pursuit_detail_sharpen:0};
  const shared=renderPursuit(pixels,32,32,{...settings,pursuit_schedule:"all"});
  const independent=renderPursuit(pixels,32,32,{...settings,pursuit_schedule:"independent"});
  expect(independent.tiles).not.toEqual(shared.tiles);
  expect(shared.pursuit!.budget).toBe(4);expect(independent.pursuit!.budget).toBe(12);
  expect(shared.pursuit!.objectiveScope).toBeUndefined();expect(independent.pursuit!.objectiveScope).toBe("independent-layer-mean");
});
