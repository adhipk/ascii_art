import {expect,test} from "bun:test";
import {sharpenTarget} from "./pursuit-sharpen";
import {preparePursuit,renderPursuit} from "./pursuit";
import type {RenderSettings} from "./renderer";

test("unsharp masking strengthens edges while preserving flat regions and source data",()=>{
  const target=Float32Array.from([.4,.4,.4,.4,.4,.4,.6,.6,.6,.6,.6,.6]),original=target.slice();
  expect(sharpenTarget(target,4,1,0)).toEqual(target);
  const sharp=sharpenTarget(target,4,1,1);
  expect(sharp[0]).toBeCloseTo(target[0],7);expect(sharp[9]).toBeCloseTo(target[9],7);
  expect(sharp[3]).toBeLessThan(target[3]);expect(sharp[6]).toBeGreaterThan(target[6]);
  expect(sharpenTarget(target,4,1,2)[6]).toBeGreaterThan(sharp[6]);
  expect(target).toEqual(original);
  expect(sharpenTarget(new Float32Array(27).fill(.5),3,3,3)).toEqual(new Float32Array(27).fill(.5));
  expect(sharpenTarget(Float32Array.from([0,0,0,1,1,1]),2,1,3).every(value=>value>=0&&value<=1)).toBe(true);
});

const source=new Uint8Array(16*16*4);
for(let y=0;y<16;y++)for(let x=0;x<16;x++){
  const index=(y*16+x)*4;source[index]=source[index+1]=source[index+2]=x<8?100:150;source[index+3]=255;
}
const config:RenderSettings={columns:24,font_size:12,layout:"pursuit",use_edge:false,edge_method:"acerola",edge_sensitivity:50,edge_amount:0,
  use_color:false,background:"#000000",background_mode:"source",foreground:"#ffffff",pursuit_resolution:128,
  pursuit_budget:2,pursuit_beam:1,pursuit_schedule:"independent",pursuit_blur:2,pursuit_edge_weight:0,pursuit_effort:"fast",
  pursuit_characters_by_scale:{32:"X",16:"X",8:"X"},
  pursuit_metrics:{width:4,height:4,baseline:4,advance:4,fontSize:4,densities:[],kernelCharacters:" X",masks:[Array(16).fill(0),Array(16).fill(.6)]}};

test("only the 8px target sharpens and paper selection is unchanged",()=>{
  for(const size of [32,16,8]){
    const settings={...config,pursuit_schedule:"all" as const,pursuit_scales:[size],pursuit_characters_by_scale:{[size]:"X"}};
    const original=preparePursuit(source,16,16,settings),sharp=preparePursuit(source,16,16,{...settings,pursuit_detail_sharpen:1});
    expect(sharp.paper).toEqual(original.paper);
    expect(sharp.paperColor).toEqual(original.paperColor);
    if(size===8){expect(sharp.target).not.toEqual(original.target);expect(sharp.appliedSharpen).toBe(1);}
    else{expect(sharp.target).toEqual(original.target);expect(sharp.appliedSharpen).toBe(0);}
  }
});

test("independent detail layer uses the sharpened target with sharp scoring",()=>{
  const settings={...config,pursuit_detail_sharpen:1},result=renderPursuit(source,16,16,settings),factor=result.width/result.pursuit!.analysisWidth;
  const standalone=renderPursuit(source,16,16,{...settings,pursuit_scales:[8],pursuit_characters_by_scale:{8:"X"},pursuit_schedule:"all",pursuit_blur:0});
  expect(result.tiles!.filter(tile=>Math.round(tile.size/factor)===8)).toEqual(standalone.tiles);
  expect(result.pursuit!.layerRuns!.map(run=>[run.size,run.targetSharpen])).toEqual([[32,0],[16,0],[8,1]]);
  expect(preparePursuit(source,16,16,settings).target).toEqual(preparePursuit(source,16,16,config).target);
});

test.each([-1,4,NaN,Infinity])("invalid sharpening strength %s is rejected",amount=>{
  expect(()=>preparePursuit(source,16,16,{...config,pursuit_detail_sharpen:amount})).toThrow("Detail sharpening");
});
