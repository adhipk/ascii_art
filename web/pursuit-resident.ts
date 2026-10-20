import {createResidentPlan, HISTORY_GLYPHS_BYTES,HISTORY_EVENTS_BYTES} from "./pursuit-resident-plan";
import type {Dispatch} from "./pursuit-resident-plan";
import type {BufferHandle,DeviceHandle,GPUProvider} from "./pursuit-gpu";
import type {RenderSettings,RenderResult,AdaptiveTile} from "./renderer";
import type {PursuitFrame,PursuitStage} from "./pursuit";

type Pipeline={getBindGroupLayout(index:number):object};
interface ResidentDevice extends DeviceHandle{
 createBindGroupLayout(descriptor:object):object;createPipelineLayout(descriptor:object):object;
}
interface Context{device:ResidentDevice;layout:object;pipelines:Map<string,Pipeline>;lost?:string;}
let cached:{key:string;promise:Promise<Context>}|undefined;
async function context(shader:string,commands:Dispatch[],provider?:GPUProvider):Promise<Context>{
 if(!provider&&cached?.key===shader)return cached.promise;
 const make=async():Promise<Context>=>{
  const gpu=provider??(globalThis.navigator as Navigator&{gpu?:GPUProvider}|undefined)?.gpu;
  if(!gpu)throw new Error("WebGPU is unavailable in this browser");
  const adapter=await gpu.requestAdapter({powerPreference:"high-performance"});if(!adapter)throw new Error("No WebGPU adapter is available");
  const device=await adapter.requestDevice() as ResidentDevice;
  try{
   const module=device.createShaderModule({code:shader}),info=await module.getCompilationInfo();
   const errors=info.messages.filter(m=>m.type==="error");if(errors.length)throw new Error("GPU shader compilation failed: "+errors.map(m=>m.message).join("; "));
   const layout=device.createBindGroupLayout({entries:Array.from({length:8},(_,binding)=>({binding,visibility:4,buffer:{type:binding===0?"uniform":binding===3||binding===4?"read-only-storage":"storage"}}))});
   const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
   const names=[...new Set(commands.map(c=>c.entry))];
   const pipelines=new Map(await Promise.all(names.map(async name=>[name,await device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint:name}})] as const)));
   const result:Context={device,layout,pipelines};void device.lost.then(info=>{result.lost=info.message||"GPU device lost";if(cached?.key===shader)cached=undefined;});return result;
  }catch(error){device.destroy();throw error;}
 };
 const promise=make();if(!provider){cached={key:shader,promise};void promise.catch(()=>{if(cached?.promise===promise)cached=undefined;});}return promise;
}

export async function renderResidentPursuit(pixels:Uint8Array,sourceWidth:number,sourceHeight:number,settings:RenderSettings,progress:(text:string)=>void=()=>{},provider?:GPUProvider):Promise<RenderResult>{
 if(settings.pursuit_repeat_penalty || (settings.pursuit_penalty_min_size && settings.pursuit_overlap_penalty))throw new Error("Local repeat penalty requires CPU acceptance with optional GPU candidate scoring");
 const started=performance.now(),plan=createResidentPlan(pixels,sourceWidth,sourceHeight,settings),setup=plan.setup;
 progress("Preparing GPU painting loop…");
 const ctx=await context(plan.shader,[...plan.initialize,...plan.iteration,...plan.refinement,...plan.flush],provider),device=ctx.device;
 const initMs=performance.now()-started,resources:BufferHandle[]=[];
 const create=(size:number,usage:number,data?:ArrayBufferView):BufferHandle=>{const b=device.createBuffer({size:Math.max(4,Math.ceil(size/4)*4),usage});resources.push(b);if(data)device.queue.writeBuffer(b,0,data);return b;};
 const buffers=[create(416,64|8,plan.parameters()),create(plan.reference.byteLength,128|8|4,plan.reference),create(plan.reference.byteLength,128|4),create(plan.masks.byteLength,128|8,plan.masks),
  create(plan.metadata.byteLength,128|8,new Uint8Array(plan.metadata)),create(plan.maxCandidates*32,128),create(plan.workBytes,128|8|4),create(plan.historyBytes,128|4)];
 const group=(uniform:BufferHandle)=>device.createBindGroup({layout:ctx.layout,entries:buffers.map((b,binding)=>({binding,resource:{buffer:binding===0?uniform:b}}))});
 const baseGroup=group(buffers[0]);
 const capacity=16,eventBytes=capacity*64,thumbBytes=capacity*plan.thumbWords*4,readback=create(80+eventBytes+thumbBytes,1|8);
 const frames:PursuitFrame[]=[],stages:PursuitStage[]=[],glyphs:{id:number;tile:AdaptiveTile;kernel:number}[]=[];
 const factor=settings.columns*8/setup.width,stageBaseline:number[]=Array(plan.stages.length).fill(0);
 let readbacks=0,seenEvents=0,gpuCandidates=0,currentHeader=new Uint32Array(20),currentFloats=new Float32Array(20);
 const runStart=performance.now();
 const check=()=>{if(ctx.lost)throw new Error(ctx.lost);};
 const encode=(encoder:ReturnType<ResidentDevice["createCommandEncoder"]>,commands:Dispatch[],bindGroup:object=baseGroup,times=1)=>{
  // Separate dispatch scopes give explicit ordering for storage writes/reductions.
  for(let repeat=0;repeat<times;repeat++)for(const command of commands){const pass=encoder.beginComputePass();pass.setPipeline(ctx.pipelines.get(command.entry)!);pass.setBindGroup(0,bindGroup);pass.dispatchWorkgroups(command.x,command.y??1);pass.end();}
 };
 const consume=(bytes:ArrayBuffer):void=>{
  currentHeader=new Uint32Array(bytes.slice(0,80));currentFloats=new Float32Array(bytes.slice(0,80));
  if(currentHeader[0]===3)throw new Error("GPU painting history capacity exceeded");
  const count=currentHeader[5]-seenEvents;if(count<0||count>capacity)throw new Error("Invalid resident GPU history count");
  const u=new Uint32Array(bytes,80,eventBytes/4),f=new Float32Array(bytes,80,eventBytes/4),thumbs=new Uint8Array(bytes,80+eventBytes,thumbBytes);
  for(let i=0;i<count;i++){
   const n=i*16,op=u[n+3],identity=u[n+4],stage=u[n+5],round=u[n+6],kernelIndex=u[n+2],k=plan.kernels[kernelIndex];
   let action="Start with paper";
   if(op!==3){
    if(!k||!Number.isFinite(f[n+11])||[f[n+8],f[n+9],f[n+10]].some(v=>!Number.isFinite(v)||v<0||v>1))throw new Error("Invalid resident GPU stroke");
    const tile:AdaptiveTile={x:u[n]*factor,y:u[n+1]*factor,size:k.size*factor,char:k.char,angle:k.angle,color:[f[n+8]*255,f[n+9]*255,f[n+10]*255]};
    if(op===2){const at=glyphs.findIndex(g=>g.id===identity);if(at<0)throw new Error("GPU refinement references a missing stroke");const old=glyphs.splice(at,1)[0];action=`Refine ${old.tile.char} → ${k.char} · ${k.size} analysis px`;}
    else if(op===1)action=`Add ${k.char} · ${k.size} analysis px`;else throw new Error("Invalid resident GPU history operation");
    glyphs.push({id:identity,tile,kernel:kernelIndex});
   }
   const energy=f[n+11];if(frames.length&&energy>frames.at(-1)!.energy+2e-6)throw new Error("GPU painting energy increased");
   const residual=thumbs.slice(i*plan.thumbWords*4,i*plan.thumbWords*4+plan.thumbWidth*plan.thumbHeight),size=plan.stages[stage]?.size??null;
   frames.push({iteration:frames.length,stage,activeScale:size,searchRound:round,energy,pixelError:f[n+12],contourError:f[n+13],overlapError:f[n+15],glyphs:glyphs.map(g=>g.tile),
    diversity:new Set(glyphs.map(g=>g.tile.char)).size,action,candidates:u[n+7],scaleCandidates:op===3?{}:{[String(size)]:Math.max(0,u[n+7]-stageBaseline[stage])},residual,residualWidth:plan.thumbWidth,residualHeight:plan.thumbHeight});
  }
  seenEvents=currentHeader[5];gpuCandidates=currentHeader[6];
 };
 const snapshot=async(encoder:ReturnType<ResidentDevice["createCommandEncoder"]>)=>{
  check();encoder.copyBufferToBuffer(buffers[6],0,readback,0,80);
  encoder.copyBufferToBuffer(buffers[7],HISTORY_GLYPHS_BYTES+seenEvents*64,readback,80,eventBytes);
  encoder.copyBufferToBuffer(buffers[7],HISTORY_GLYPHS_BYTES+HISTORY_EVENTS_BYTES+seenEvents*plan.thumbWords*4,readback,80+eventBytes,thumbBytes);
  device.queue.submit([encoder.finish()]);await readback.mapAsync(1);++readbacks;
  try{check();consume(readback.getMappedRange().slice(0));}finally{readback.unmap();}
 };
 let minimumGain=1e-9;
 const configure=(stage:number,round:number)=>{device.queue.writeBuffer(buffers[0],0,plan.parameters(stage,round,0,0,minimumGain));device.queue.writeBuffer(buffers[6],0,new Uint32Array([0]));};
 try{
  let encoder=device.createCommandEncoder();encode(encoder,plan.initialize);await snapshot(encoder);
  for(const [stageIndex,stage] of plan.stages.entries()){
   const startFrame=frames.length-1,startCount=glyphs.length;stageBaseline[stageIndex]=gpuCandidates;
   if(!stage.count){stages.push({index:stageIndex,size:stage.size,searchRounds:1,startFrame,endFrame:startFrame,addedGlyphs:0,converged:true});continue;}
   minimumGain=stage.size>=setup.sizes[0]/2?Math.max(1e-9,currentFloats[14]*setup.coarseThreshold):1e-9;
   const rounds=stage.size>=setup.sizes[0]/2?setup.coarseRounds:1;let visited=0;
   for(let round=0;round<rounds;++round){
    ++visited;configure(stageIndex,round);let refined=false;
    progress(`GPU · ${stage.size}px pass · round ${round+1} / ${rounds}…`);
    for(let guard=0;guard<300;++guard){
     encoder=device.createCommandEncoder();encode(encoder,plan.iteration,baseGroup,capacity);encode(encoder,plan.flush);await snapshot(encoder);
     if(currentHeader[0]===0){progress(`GPU · ${stage.size}px · ${glyphs.length} / ${setup.budget} glyphs…`);continue;}
     if(!refined){
      refined=true;const eligible=glyphs.filter(g=>plan.kernels[g.kernel].size===stage.size),count=Math.min(8,eligible.length);
      const identities=Array.from({length:count},(_,i)=>eligible[(Math.floor(i*eligible.length/count)+round)%eligible.length].id);
      const before=seenEvents,uniforms:BufferHandle[]=[];encoder=device.createCommandEncoder();
      for(const identity of identities){const b=create(416,64|8,plan.parameters(stageIndex,round,1,identity,minimumGain));uniforms.push(b);encode(encoder,plan.refinement,group(b));}
      if(identities.length)await snapshot(encoder);
      for(const b of uniforms){b.destroy();resources.splice(resources.indexOf(b),1);}
      configure(stageIndex,round);
      if(seenEvents>before&&glyphs.length<setup.budget)continue;
     }
     break;
    }
    if(currentHeader[0]===0&&glyphs.length<setup.budget)throw new Error("Resident GPU search exceeded its iteration guard");
    if(glyphs.length>=setup.budget||currentFloats[14]<1e-9)break;
   }
   const converged=glyphs.length<setup.budget;
   stages.push({index:stageIndex,size:stage.size,searchRounds:visited,startFrame,endFrame:frames.length-1,addedGlyphs:glyphs.length-startCount,converged});
   if(!converged)break;
  }
  const outputWidth=Math.round(settings.columns*8),outputHeight=Math.round(setup.height*factor);
  if(outputWidth*outputHeight>6_000_000)throw new Error("Output is too large; reduce columns");
  const paper=new Uint8Array(outputHeight*3);for(let i=0;i<paper.length;++i)paper[i]=Math.round(setup.paperColor[i%3]*255);
  const tileMetrics=Object.fromEntries(setup.sizes.map(size=>{const displaySize=size*factor,scale=displaySize/setup.metric.width;return [String(displaySize),{width:displaySize,height:displaySize,baseline:setup.metric.baseline*scale,advance:setup.metric.advance*scale,fontSize:(setup.metric.fontSize??setup.metric.height*.8)*scale,densities:[]}];}));
  return {width:outputWidth,height:outputHeight,columns:0,rows:0,cellWidth:0,cellHeight:0,text:"",colors:new Uint8Array(),paper,tiles:frames.at(-1)!.glyphs,tileMetrics,glyphAtlas:setup.metric,
   pursuit:{targetSharpen:setup.appliedSharpen,frames,stages,schedule:setup.schedule,coarseThreshold:setup.coarseThreshold,coarseRounds:setup.coarseRounds,stopReason:glyphs.length>=setup.budget?"Glyph budget reached":"No improving proposal at any remaining scale (bounded GPU search)",
    budget:setup.budget,beamWidth:1,edgeWeight:setup.edgeWeight,scales:setup.sizes,analysisWidth:setup.width,analysisHeight:setup.height,backend:"webgpu",gpuResident:true,gpuReadbacks:readbacks,
    gpuCandidates,gpuInitMs:initMs,gpuRunMs:performance.now()-runStart,elapsedMs:performance.now()-started,blurStrength:setup.blurStrength,blurCurve:setup.blurModel.curve,effort:setup.effort,vocabulary:setup.vocabulary,kernelCounts:setup.kernelCounts,charactersByScale:setup.charactersByScale,overlapLimit:setup.maxOverlap,overlapPenalty:setup.overlapPenalty,hiddenGlyphsRemoved:0}};
 }catch(error){if(cached?.key===plan.shader)cached=undefined;device.destroy();throw error;}
 finally{for(const resource of resources)resource.destroy();if(provider)device.destroy();}
}
