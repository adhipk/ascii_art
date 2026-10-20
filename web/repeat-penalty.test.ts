import {expect,test} from "bun:test";
import {repeatPairCost,repeatError} from "./pursuit-repeat";
import {preparePursuit,renderPursuit,renderPursuitAsync,referenceCandidateScores} from "./pursuit";
import type {RenderSettings} from "./renderer";
import {penaltyWeightsByScale} from "./penalty-weight";

test("repetition is local, scale invariant, whitespace exempt and independent of rotation",()=>{
  const glyph={x:0,y:0,size:16,char:"X"};
  const near={...glyph,x:8},far={...glyph,x:32};
  expect(repeatPairCost(glyph,near,4096)).toBeGreaterThan(repeatPairCost(glyph,{...near,x:16},4096));
  expect(repeatPairCost(glyph,far,4096)).toBe(0);
  expect(repeatPairCost(glyph,{...near,char:"Y"},4096)).toBe(0);
  expect(repeatPairCost({...glyph,char:" "},{...near,char:" "},4096)).toBe(0);
  expect(repeatPairCost(glyph,near,4096)).toBe(repeatPairCost({...glyph,size:32},{...near,x:16,size:32},16384));
  expect(repeatError([glyph,near,{...glyph,x:16}],4096)).toBeGreaterThan(2*repeatPairCost(glyph,near,4096));
});

const source=new Uint8Array(32*32*4).fill(242);
for(let index=3;index<source.length;index+=4)source[index]=255;
const config:RenderSettings={columns:24,font_size:12,layout:"pursuit",use_edge:false,edge_method:"acerola",edge_sensitivity:50,edge_amount:0,
  use_color:false,background:"#000000",background_mode:"solid",foreground:"#ffffff",pursuit_resolution:128,pursuit_scales:[64],
  pursuit_budget:8,pursuit_beam:1,pursuit_coarse_threshold:0,pursuit_max_overlap:1,pursuit_edge_weight:0,pursuit_effort:"fast",
  pursuit_metrics:{width:4,height:4,baseline:4,advance:4,fontSize:4,densities:[],kernelCharacters:" X",masks:[Array(16).fill(0),Array(16).fill(.6)]}};

for(const blur of [0,1])test(`repeat cost is included in exact frame energies with blur ${blur}`,()=>{
  const settings={...config,pursuit_blur:blur,pursuit_repeat_penalty:.5,pursuit_overlap_penalty:.2};
  const setup=preparePursuit(source,32,32,settings),result=renderPursuit(source,32,32,settings),factor=result.width/setup.width;
  let previous=Infinity;
  for(const frame of result.pursuit!.frames){
    const pixels=setup.paper.slice();
    const glyphs=frame.glyphs.map(glyph=>({...glyph,x:Math.round(glyph.x/factor),y:Math.round(glyph.y/factor),size:Math.round(glyph.size/factor)}));
    for(const glyph of glyphs)for(let y=0;y<glyph.size;y++)for(let x=0;x<glyph.size;x++)for(let channel=0;channel<3;channel++){
      const index=((glyph.y+y)*setup.width+glyph.x+x)*3+channel;
      pixels[index]+=Math.fround(.6)*(glyph.color[channel]/255-pixels[index]);
    }
    expect(frame.repeatError!).toBeCloseTo(repeatError(glyphs,setup.width*setup.height),10);
    expect(frame.energy).toBeCloseTo(setup.totalEnergy(pixels)+.5*frame.repeatError!+.2*frame.overlapError!,6);
    expect(frame.energy).toBeLessThanOrEqual(previous+1e-8);previous=frame.energy;
  }
  expect(result.pursuit!.repeatPenalty).toBe(.5);
});

test("positive repeat penalty reduces local repetition",()=>{
  const free=renderPursuit(source,32,32,{...config,pursuit_blur:0});
  const penalized=renderPursuit(source,32,32,{...config,pursuit_blur:0,pursuit_repeat_penalty:2});
  expect(penalized.pursuit!.frames.at(-1)!.repeatError!).toBeLessThan(free.pursuit!.frames.at(-1)!.repeatError!);
});

test("repeat cutoff exempts small glyphs and mixed-size pairs",()=>{
  expect(penaltyWeightsByScale(.2,[48,24,12,6,3],24)).toEqual({"48":.2,"24":.2,"12":0,"6":0,"3":0});
  const large={x:0,y:0,size:32,char:"X"},small={...large,size:12};
  expect(repeatPairCost(large,{...large,x:8},4096,24)).toBeGreaterThan(0);
  expect(repeatPairCost(small,{...small,x:4},4096,24)).toBe(0);
  expect(repeatPairCost(large,small,4096,24)).toBe(0);
});

test("GPU candidate scoring and CPU acceptance use the same repeat objective",async()=>{
  const settings={...config,pursuit_blur:0,pursuit_repeat_penalty:.5};
  const cpu=renderPursuit(source,32,32,settings);
  const gpu=await renderPursuitAsync(source,32,32,settings,()=>{},async()=>({score:async batch=>Float32Array.from(referenceCandidateScores(batch)),dispose(){}}));
  expect(gpu.pursuit!.frames.at(-1)!.energy).toBeCloseTo(cpu.pursuit!.frames.at(-1)!.energy,6);
});

test.each([-1,6,NaN,Infinity])("rejects invalid repeat penalty %s",value=>{
  expect(()=>preparePursuit(source,32,32,{...config,pursuit_repeat_penalty:value})).toThrow("Repeat penalty");
});
