import {expect,test} from "bun:test";
import {referenceCandidateScores,preparePursuit,renderPursuit} from "./pursuit";
import type {CandidateBatch} from "./pursuit-gpu";
import type {RenderSettings} from "./renderer";
import {repeatError} from "./pursuit-repeat";

test("soft overlap cost is ink-weighted and leaves fresh ink free at a 100% hard cap",()=>{
  const batch:CandidateBatch={width:4,height:4,target:new Float32Array(48).fill(.5),pixels:new Float32Array(48),
    coverage:new Float32Array(16).fill(.5),maxOverlap:1,edgeWeight:0,color:false,
    kernels:[{size:2,alpha:new Float32Array(4).fill(1)}],candidates:[{kernel:0,x:1,y:1}]};
  const free=referenceCandidateScores(batch)[0],penalized=referenceCandidateScores({...batch,overlapPenalty:.8})[0];
  expect(free-penalized).toBeCloseTo(.8*4*.5/16,10);
  expect(penalized).toBeLessThan(0);
  expect(referenceCandidateScores({...batch,coverage:new Float32Array(16),overlapPenalty:.8})[0]).toBe(free);
});

const source=new Uint8Array(32*32*4).fill(242);
for(let i=3;i<source.length;i+=4)source[i]=255;
const config:RenderSettings={columns:24,font_size:12,layout:"pursuit",use_edge:false,edge_method:"acerola",edge_sensitivity:50,edge_amount:0,
  use_color:false,background:"#000000",background_mode:"solid",foreground:"#ffffff",pursuit_resolution:128,pursuit_scales:[64],
  pursuit_budget:8,pursuit_beam:1,pursuit_coarse_threshold:0,pursuit_max_overlap:1,pursuit_edge_weight:0,pursuit_effort:"fast",
  pursuit_metrics:{width:4,height:4,baseline:4,advance:4,fontSize:4,densities:[],kernelCharacters:" X",masks:[Array(16).fill(0),Array(16).fill(.6)]}};

for(const blur of [0,1])test(`recorded objective includes recomputable overlap cost with blur ${blur}`,()=>{
  const settings={...config,pursuit_blur:blur,pursuit_overlap_penalty:.5};
  const setup=preparePursuit(source,32,32,settings),result=renderPursuit(source,32,32,settings),n=setup.width*setup.height;
  let previous=Infinity;
  for(const frame of result.pursuit!.frames) {
    const pixels=setup.paper.slice(),coverage=new Float64Array(n);let mass=0;
    const factor=result.width/setup.width,a=Math.fround(.6);
    for(const glyph of frame.glyphs) {
      const left=Math.round(glyph.x/factor),top=Math.round(glyph.y/factor),size=Math.round(glyph.size/factor);
      for(let y=0;y<size;y++)for(let x=0;x<size;x++) {
        const i=(top+y)*setup.width+left+x;mass+=a;coverage[i]+=a*(1-coverage[i]);
        for(let c=0;c<3;c++)pixels[i*3+c]+=a*(glyph.color[c]/255-pixels[i*3+c]);
      }
    }
    const overlap=(mass-coverage.reduce((sum,v)=>sum+v,0))/n;
    expect(frame.overlapError!).toBeCloseTo(overlap,5);
    expect(frame.energy).toBeCloseTo(setup.totalEnergy(pixels)+.5*overlap,5);
    expect(frame.energy).toBeLessThanOrEqual(previous+1e-8);previous=frame.energy;
  }
  expect(result.pursuit!.overlapPenalty).toBe(.5);
});

test("penalizing overlap reduces repeated ink in a translucent layering fixture",()=>{
  const free=renderPursuit(source,32,32,{...config,pursuit_blur:0,pursuit_overlap_penalty:0});
  const penalized=renderPursuit(source,32,32,{...config,pursuit_blur:0,pursuit_overlap_penalty:.5});
  expect(penalized.pursuit!.frames.at(-1)!.overlapError!).toBeLessThan(free.pursuit!.frames.at(-1)!.overlapError!);
});

test.each([-1,6,NaN,Infinity])("rejects invalid overlap penalties %s",value=>{
  expect(()=>preparePursuit(source,32,32,{...config,pursuit_overlap_penalty:value})).toThrow("Overlap penalty");
});

test("small candidates are penalty free without bypassing the hard overlap cap",()=>{
  const batch:CandidateBatch={width:32,height:32,target:new Float32Array(3072).fill(.5),pixels:new Float32Array(3072),
    coverage:new Float32Array(1024).fill(.5),penaltyCoverage:new Float32Array(1024).fill(.5),penaltyMinSize:24,maxOverlap:1,edgeWeight:0,color:false,
    kernels:[{size:16,alpha:new Float32Array(256).fill(1)},{size:24,alpha:new Float32Array(576).fill(1)}],candidates:[{kernel:0,x:0,y:0},{kernel:1,x:0,y:0}]};
  const free=referenceCandidateScores(batch),penalized=referenceCandidateScores({...batch,overlapPenalty:1});
  expect(penalized[0]).toBe(free[0]);expect(penalized[1]).toBeLessThan(free[1]);
  expect(referenceCandidateScores({...batch,maxOverlap:0})[0]).toBe(-1);
  expect(referenceCandidateScores({...batch,overlapPenalty:1,penaltyCoverage:new Float32Array(1024)})).toEqual(free);
});

for(const blur of [0,1])test(`mixed-scale objective charges only coarse ink and coarse repeat pairs with blur ${blur}`,()=>{
  const settings={...config,pursuit_scales:[32,16],pursuit_blur:blur,pursuit_penalty_min_size:24,pursuit_overlap_penalty:.1,pursuit_repeat_penalty:.2};
  const setup=preparePursuit(source,32,32,settings),result=renderPursuit(source,32,32,settings),area=setup.width*setup.height;
  let previous=Infinity;
  for(const frame of result.pursuit!.frames){
    const pixels=setup.paper.slice(),coverage=new Float64Array(area);let overlap=0;
    const factor=result.width/setup.width,alpha=Math.fround(.6);
    const glyphs=frame.glyphs.map(glyph=>({...glyph,x:Math.round(glyph.x/factor),y:Math.round(glyph.y/factor),size:Math.round(glyph.size/factor)}));
    for(const glyph of glyphs)for(let y=0;y<glyph.size;y++)for(let x=0;x<glyph.size;x++){
      const index=(glyph.y+y)*setup.width+glyph.x+x;
      if(glyph.size>=24){overlap+=alpha*coverage[index]/area;coverage[index]+=alpha*(1-coverage[index]);}
      for(let channel=0;channel<3;channel++)pixels[index*3+channel]+=alpha*(glyph.color[channel]/255-pixels[index*3+channel]);
    }
    expect(frame.overlapError!).toBeCloseTo(overlap,6);
    expect(frame.repeatError!).toBeCloseTo(repeatError(glyphs,area,24),8);
    expect(frame.energy).toBeCloseTo(setup.totalEnergy(pixels)+.1*overlap+.2*repeatError(glyphs,area,24),6);
    expect(frame.energy).toBeLessThanOrEqual(previous+1e-8);previous=frame.energy;
  }
  expect(result.pursuit!.penaltyMinSize).toBe(24);
});
