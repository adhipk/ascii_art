import { expect, test } from "bun:test";
import { createWebGPUScorer, packCandidates, type CandidateBatch, type CandidateScorer } from "./pursuit-gpu";
import { referenceCandidateScores, renderPursuit, renderPursuitAsync } from "./pursuit";
import type { RenderSettings } from "./renderer";

function fixture():{pixels:Uint8Array;settings:RenderSettings} {
  const pixels=new Uint8Array(32*32*4);
  for(let y=0;y<32;++y)for(let x=0;x<32;++x){const i=(y*32+x)*4;pixels[i]=pixels[i+1]=pixels[i+2]=x>8&&x<24&&y>8&&y<24?200:0;pixels[i+3]=255;}
  return {pixels,settings:{columns:24,font_size:12,layout:"pursuit",pursuit_blur:0,pursuit_budget:3,pursuit_beam:1,pursuit_backend:"auto",pursuit_resolution:128,pursuit_edge_weight:3,
    use_color:true,use_edge:false,edge_method:"acerola",edge_sensitivity:50,edge_amount:100,background:"#000000",background_mode:"solid",foreground:"#ffffff",
    pursuit_metrics:{width:4,height:4,baseline:4,advance:4,fontSize:4,densities:[],kernelCharacters:" X|",masks:[Array(16).fill(0),Array(16).fill(1),Array.from({length:16},(_,i)=>Number(i%4===2))]}}};
}

test("explicit CPU backend never requests an adapter and preserves reference results",async()=>{
  const {pixels,settings}=fixture();let called=false;
  const result=await renderPursuitAsync(pixels,32,32,{...settings,pursuit_backend:"cpu"},()=>{},async()=>{called=true;throw new Error("must not run");});
  expect(called).toBe(false);expect(result.pursuit!.backend).toBe("cpu");
  expect(result.tiles).toEqual(renderPursuit(pixels,32,32,settings).tiles);
});

test("missing adapter falls back to CPU with a visible reason",async()=>{
  const {pixels,settings}=fixture();const statuses:string[]=[];
  const result=await renderPursuitAsync(pixels,32,32,settings,s=>statuses.push(s),async()=>{throw new Error("No adapter");});
  expect(result.pursuit!.backend).toBe("cpu");expect(result.pursuit!.backendReason).toBe("No adapter");
  expect(statuses.some(s=>s.includes("CPU scoring"))).toBe(true);
  expect(result.tiles).toEqual(renderPursuit(pixels,32,32,settings).tiles);
});

test("batch scorer integration self-checks scores and still uses exact accepted energy",async()=>{
  const {pixels,settings}=fixture();let disposed=false;
  const scorer:CandidateScorer={score:async batch=>Float32Array.from(referenceCandidateScores(batch)),dispose(){disposed=true;}};
  const result=await renderPursuitAsync(pixels,32,32,settings,()=>{},async()=>scorer);
  expect(disposed).toBe(true);expect(result.pursuit!.backend).toBe("webgpu");
  expect(result.pursuit!.gpuCandidates).toBeGreaterThan(0);expect(result.pursuit!.gpuValidationMaxError).toBeLessThan(1e-5);
  const frames=result.pursuit!.frames;for(let i=1;i<frames.length;++i)expect(frames[i].energy).toBeLessThan(frames[i-1].energy);
  expect(frames.at(-1)!.energy).toBeCloseTo(renderPursuit(pixels,32,32,settings).pursuit!.frames.at(-1)!.energy,5);
});

test("invalid GPU scores are rejected and fall back to the CPU path",async()=>{
  const {pixels,settings}=fixture();let disposed=false;
  const result=await renderPursuitAsync(pixels,32,32,settings,()=>{},async()=>({score:async batch=>new Float32Array(batch.candidates.length).fill(1),dispose(){disposed=true;}}));
  expect(disposed).toBe(true);expect(result.pursuit!.backend).toBe("cpu");expect(result.pursuit!.backendReason).toContain("self-check failed");
  expect(result.tiles).toEqual(renderPursuit(pixels,32,32,settings).tiles);
});

test("device failure after a successful batch continues with a recorded mixed backend",async()=>{
  const {pixels,settings}=fixture();let calls=0;
  const result=await renderPursuitAsync(pixels,32,32,settings,()=>{},async()=>({
    async score(batch){if(++calls>1)throw new Error("Device lost");return Float32Array.from(referenceCandidateScores(batch));},dispose(){},
  }));
  expect(result.pursuit!.backend).toBe("mixed");expect(result.pursuit!.backendReason).toBe("Device lost");
  const frames=result.pursuit!.frames;for(let i=1;i<frames.length;++i)expect(frames[i].energy).toBeLessThan(frames[i-1].energy);
});

test("candidate buffer layout uses alpha offsets and rejects out-of-bounds patches",()=>{
  const batch:CandidateBatch={width:8,height:8,target:new Float32Array(192),pixels:new Float32Array(192),edgeWeight:3,color:true,
    kernels:[{size:2,alpha:new Float32Array(4)},{size:3,alpha:new Float32Array(9)}],candidates:[{kernel:1,x:4,y:5},{kernel:0,x:0,y:0}]};
  expect(Array.from(packCandidates(batch,new Uint32Array([0,4])))).toEqual([4,5,3,4,0,0,2,0]);
  expect(()=>packCandidates({...batch,candidates:[{kernel:0,x:7,y:7}]},new Uint32Array([0,4]))).toThrow("bounds");
});

test("null adapter and shader compilation errors release resources",async()=>{
  await expect(createWebGPUScorer({requestAdapter:async()=>null})).rejects.toThrow("adapter");
  let destroyed=false;
  const device:any={createShaderModule:()=>({getCompilationInfo:async()=>({messages:[{type:"error",message:"invalid shader"}]})}),destroy(){destroyed=true;}};
  await expect(createWebGPUScorer({requestAdapter:async()=>({requestDevice:async()=>device})})).rejects.toThrow("compilation");
  expect(destroyed).toBe(true);
});

test("WebGPU host batches uploads, dispatches, reads scores, and releases all buffers",async()=>{
  const buffers:any[]=[];let dispatches=0;let destroyed=false;
  let bindings:any[]=[];
  const device:any={
    lost:new Promise(()=>{}),pushErrorScope(){},async popErrorScope(){return null;},
    createBuffer({size}:any){const buffer={data:new ArrayBuffer(size),destroyed:false,destroy(){this.destroyed=true;},async mapAsync(){},getMappedRange(offset=0,size?:number){return this.data.slice(offset,size===undefined?undefined:offset+size);},unmap(){}};buffers.push(buffer);return buffer;},
    createShaderModule:()=>({getCompilationInfo:async()=>({messages:[]})}),
    createComputePipelineAsync:async()=>({getBindGroupLayout:()=>({})}),
    createBindGroup({entries}:any){bindings=entries;return {};},
    createCommandEncoder(){const commands:(()=>void)[]=[];return {
      beginComputePass(){return {setPipeline(){},setBindGroup(){},dispatchWorkgroups(count:number){++dispatches;expect(count).toBe(1);commands.push(()=>new Float32Array(bindings[5].resource.buffer.data).set([.125,.25]));},end(){}};},
      copyBufferToBuffer(source:any,so:number,target:any,to:number,size:number){commands.push(()=>new Uint8Array(target.data,to,size).set(new Uint8Array(source.data,so,size)));},finish(){return commands;},
    };},
    queue:{writeBuffer(buffer:any,offset:number,data:ArrayBufferView){new Uint8Array(buffer.data,offset,data.byteLength).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));},submit(commands:any[]){commands.flat().forEach((command:()=>void)=>command());}},
    destroy(){destroyed=true;},
  };
  const scorer=await createWebGPUScorer({requestAdapter:async()=>({requestDevice:async()=>device})});
  const batch:CandidateBatch={width:8,height:8,target:new Float32Array(192),pixels:new Float32Array(192),edgeWeight:3,color:true,
    kernels:[{size:2,alpha:new Float32Array(4).fill(1)}],candidates:[{kernel:0,x:0,y:0},{kernel:0,x:4,y:4}]};
  expect(Array.from(await scorer.score(batch))).toEqual([.125,.25]);
  const allocated=buffers.length;
  batch.guidance=new Float32Array(batch.target.length).fill(.75);
  expect(Array.from(await scorer.score(batch))).toEqual([.125,.25]);
  expect(new Float32Array(bindings[1].resource.buffer.data)[0]).toBe(.75);
  expect(buffers.length).toBe(allocated);expect(dispatches).toBe(2);
  scorer.dispose();scorer.dispose();expect(destroyed).toBe(true);expect(buffers.every(buffer=>buffer.destroyed)).toBe(true);
});

test("blurred residual guidance reaches GPU ranking and exact multiscale acceptance matches CPU",async()=>{
  const {pixels,settings}=fixture();let guided=0;
  const result=await renderPursuitAsync(pixels,32,32,{...settings,pursuit_blur:1},()=>{},async()=>({
    async score(batch){if(batch.guidance){++guided;expect(batch.guidance).not.toBe(batch.target);}return Float32Array.from(referenceCandidateScores(batch));},dispose(){},
  }));
  expect(guided).toBeGreaterThan(0);expect(result.pursuit!.backend).toBe("webgpu");
  const cpu=renderPursuit(pixels,32,32,{...settings,pursuit_blur:1});
  expect(result.pursuit!.frames.at(-1)!.energy).toBeCloseTo(cpu.pursuit!.frames.at(-1)!.energy,6);
});

test("GPU staged search scores exactly one scale per batch in descending order",async()=>{
  const {pixels,settings}=fixture();const sizes:number[]=[];
  const result=await renderPursuitAsync(pixels,32,32,{...settings,pursuit_schedule:"coarse-to-fine",pursuit_budget:12},()=>{},async()=>({
    async score(batch){
      const active=new Set(batch.candidates.map(c=>batch.kernels[c.kernel].size));
      expect(active.size).toBe(1);sizes.push([...active][0]);
      return Float32Array.from(referenceCandidateScores(batch));
    },dispose(){},
  }));
  expect(sizes.length).toBeGreaterThan(0);
  for(let i=1;i<sizes.length;++i)expect(sizes[i]).toBeLessThanOrEqual(sizes[i-1]);
  expect(result.pursuit!.schedule).toBe("coarse-to-fine");
});
