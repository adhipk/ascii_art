import {expect,test} from "bun:test";
import {preparePursuit,referenceCandidateScores,renderPursuit,renderPursuitAsync} from "./pursuit";
import {pursuitFrame} from "./pursuit";
import {visiblePursuitLayers} from "./pursuit-layers";
import type {RenderSettings} from "./renderer";
import {allocateLayerBudgets} from "./layer-budgets";

const source=new Uint8Array(32*32*4).fill(240);
for(let index=3;index<source.length;index+=4)source[index]=255;
const config:RenderSettings={columns:24,font_size:12,layout:"pursuit",use_edge:false,edge_method:"acerola",edge_sensitivity:50,edge_amount:0,
  use_color:false,background:"#000000",background_mode:"solid",foreground:"#ffffff",pursuit_resolution:128,
  pursuit_budget:3,pursuit_beam:1,pursuit_schedule:"independent",pursuit_blur:2,pursuit_edge_weight:0,pursuit_effort:"fast",
  pursuit_characters_by_scale:{32:"X",16:"X",8:"X"},
  pursuit_metrics:{width:4,height:4,baseline:4,advance:4,fontSize:4,densities:[],kernelCharacters:" X",masks:[Array(16).fill(0),Array(16).fill(.6)]}};

test("each layer matches an independent full-image fit and receives its own full budget",()=>{
  const original=JSON.stringify(config),result=renderPursuit(source,32,32,config),trace=result.pursuit!;
  expect(trace.scales).toEqual([32,16,8]);expect(trace.schedule).toBe("independent");
  expect(trace.budgetPerLayer).toBe(3);expect(trace.budget).toBe(9);
  const factor=result.width/trace.analysisWidth,standalone=[];
  for(const size of [32,16,8]){
    const layer=renderPursuit(source,32,32,{...config,pursuit_schedule:"all",pursuit_scales:[size],pursuit_characters_by_scale:{[size]:"X"}});
    standalone.push(layer);
    expect(result.tiles!.filter(tile=>Math.round(tile.size/factor)===size)).toEqual(layer.tiles);
    expect(trace.layerRuns!.find(run=>run.size===size)!.glyphCount).toBe(3);
  }
  expect(result.tiles).toEqual(standalone.flatMap(layer=>layer.tiles!));
  expect(trace.frames.at(-1)!.energy).toBeCloseTo(standalone.reduce((sum,layer)=>sum+layer.pursuit!.frames.at(-1)!.energy,0)/3,10);
  let previous=Infinity;
  for(const frame of trace.frames){expect(frame.energy).toBeLessThanOrEqual(previous+1e-9);previous=frame.energy;}
  expect(trace.frames.at(-1)!.glyphs).toEqual(result.tiles);
  expect(pursuitFrame(result,0).tiles).toEqual([]);
  expect(visiblePursuitLayers(result,{32:false,16:false}).tiles).toEqual(standalone[2].tiles);
  expect(JSON.stringify(config)).toBe(original);
});

test("final composite metrics measure the actual overlaid pixels, separately from layer scores",()=>{
  const setup=preparePursuit(source,32,32,config),result=renderPursuit(source,32,32,config),pixels=setup.paper.slice();
  const factor=result.width/setup.width;
  for(const tile of result.tiles!){
    const left=Math.round(tile.x/factor),top=Math.round(tile.y/factor),size=Math.round(tile.size/factor);
    for(let y=0;y<size;y++)for(let x=0;x<size;x++)for(let channel=0;channel<3;channel++){
      const index=((top+y)*setup.width+left+x)*3+channel;
      pixels[index]+=Math.fround(.6)*(tile.color[channel]/255-pixels[index]);
    }
  }
  expect(result.pursuit!.objectiveScope).toBe("independent-layer-mean");
  expect(result.pursuit!.compositeMetrics!.reconstructionEnergy).toBeCloseTo(setup.totalEnergy(pixels),8);
});

test("GPU batches contain one scale and every layer starts with untouched paper",async()=>{
  const firstPixels=new Map<number,Float32Array>();let factories=0;
  const result=await renderPursuitAsync(source,32,32,config,()=>{},async()=>{
    factories++;
    return {score:async batch=>{
      const sizes=[...new Set(batch.kernels.map(kernel=>kernel.size))];expect(sizes.length).toBe(1);
      if(!firstPixels.has(sizes[0]))firstPixels.set(sizes[0],batch.pixels.slice());
      return Float32Array.from(referenceCandidateScores(batch));
    },dispose(){}};
  });
  expect(factories).toBe(3);expect([...firstPixels.keys()].sort((first,second)=>first-second)).toEqual([8,16,32]);
  for(const pixels of firstPixels.values())expect(pixels.every(value=>value===0)).toBe(true);
  expect(result.pursuit!.layerRuns!.length).toBe(3);
});

test("empty layers do not increase other layers' budgets",()=>{
  const result=renderPursuit(source,32,32,{...config,pursuit_characters_by_scale:{32:"",16:"X",8:"X"}});
  expect(result.pursuit!.kernelCounts["32"]).toBe(0);
  expect(result.pursuit!.layerRuns!.map(run=>run.size)).toEqual([16,8]);
  expect(result.pursuit!.layerRuns!.map(run=>run.budget)).toEqual([3,3]);
  expect(result.pursuit!.layerRuns!.map(run=>run.glyphCount)).toEqual([3,3]);
  expect(result.pursuit!.layerBudgets).toEqual({32:0,16:3,8:3});
  expect(result.pursuit!.budget).toBe(6);
});

test("a lone active layer keeps the exact slider cap on CPU and async paths",async()=>{
  const settings={...config,pursuit_characters_by_scale:{32:"",16:"",8:"X"}};
  const cpu=renderPursuit(source,32,32,settings);
  const asyncResult=await renderPursuitAsync(source,32,32,{...settings,pursuit_backend:"cpu"});
  for(const result of [cpu,asyncResult]){
    expect(result.pursuit!.layerBudgets).toEqual({32:0,16:0,8:3});
    expect(result.pursuit!.layerRuns![0].budget).toBe(3);
    expect(result.tiles!.length).toBe(3);
  }
});

test("allocations never multiply the per-layer cap",()=>{
  expect(allocateLayerBudgets([32,16,8],[8],80)).toEqual({32:0,16:0,8:80});
  expect(allocateLayerBudgets([32,16,8],[16,8],128)).toEqual({32:0,16:128,8:128});
  expect(allocateLayerBudgets([32,16,8],[32,16,8],128)).toEqual({32:128,16:128,8:128});
  const result=renderPursuit(new Uint8Array(source.length),32,32,{...config,pursuit_budget:2048,pursuit_characters_by_scale:{32:"",16:"",8:"X"}});
  expect(result.pursuit!.budget).toBe(2048);
  expect(result.pursuit!.layerRuns![0].budget).toBe(2048);
});
