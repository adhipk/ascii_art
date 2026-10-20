/** Native Bun API: image -> accepted optimization history -> streamed MP4.
 * Requires ffmpeg; image-path input also needs project Python/Pillow.
 */
import {mkdtemp,mkdir,rm,rename} from "node:fs/promises";
import {tmpdir} from "node:os";
import {dirname,resolve,join} from "node:path";
import {renderPursuit} from "../web/pursuit";
import type {PursuitFrame,PursuitStage} from "../web/pursuit";
import {createResidentPlan} from "../web/pursuit-resident-plan";
import type {RenderSettings,RenderResult,AdaptiveTile} from "../web/renderer";
import type {TextMetrics} from "../web/text-view";

const ROOT=resolve(import.meta.dir,"..");
export interface PixelImage{pixels:Uint8Array;width:number;height:number;}
export interface ProcessVideoParameters{
 outputPath:string;
 settings?:Partial<RenderSettings>;
 /** Complete measured atlas required for raw RGBA input. */
 metrics?:TextMetrics;
 glyphs?:string;
 fps?:number;
 width?:number;
 initialHoldSeconds?:number;
 finalHoldSeconds?:number;
 /** Number of accepted edits between video samples; final state is always shown. */
 every?:number;
 maxIterations?:number;
 requireConvergence?:boolean;
 backend?:"auto"|"cpu"|"native-gpu";
 python?:string;
 /** e.g. ["/path/to/python"] with wgpu/numpy installed. Never installed implicitly. */
 gpuPython?:string[];
 ffmpeg?:string;
 signal?:AbortSignal;
 onProgress?:(message:string)=>void;
}
export interface ProcessVideoResult{
 path:string;frames:number;duration:number;iterations:number;converged:boolean;stopReason:string;
 width:number;height:number;backend:"cpu"|"native-gpu";renderMs:number;encodeMs:number;
 backendReason?:string;
}
function checkAbort(signal?:AbortSignal):void{if(signal?.aborted)throw new Error("Video generation cancelled");}
async function command(args:string[],signal?:AbortSignal):Promise<void>{
 checkAbort(signal);const child=Bun.spawn(args,{stdout:"ignore",stderr:"pipe"});
 const stop=()=>child.kill();signal?.addEventListener("abort",stop,{once:true});
 try{const stderr=await new Response(child.stderr).text(),status=await child.exited;checkAbort(signal);if(status!==0)throw new Error(`${args[0]} failed: ${stderr.slice(-2000)}`);}
 finally{signal?.removeEventListener("abort",stop);}
}

async function nativeRender(input:PixelImage,settings:RenderSettings,directory:string,python:string[],signal?:AbortSignal):Promise<RenderResult>{
 const plan=createResidentPlan(input.pixels,input.width,input.height,settings),base=join(directory,"gpu");await mkdir(base);
 await Promise.all([
  Bun.write(join(base,"shader.wgsl"),plan.shader),Bun.write(join(base,"reference.f32"),plan.reference),Bun.write(join(base,"masks.f32"),plan.masks),Bun.write(join(base,"kernels.bin"),plan.metadata),Bun.write(join(base,"params.bin"),plan.parameters()),
  Bun.write(join(base,"plan.json"),JSON.stringify({width:plan.setup.width,height:plan.setup.height,budget:plan.setup.budget,coarseThreshold:plan.setup.coarseThreshold,coarseRounds:plan.setup.coarseRounds,stages:plan.stages,levels:plan.levels,initialize:plan.initialize,iteration:plan.iteration,flush:plan.flush,refinement:plan.refinement,maxCandidates:plan.maxCandidates,workBytes:plan.workBytes,historyBytes:plan.historyBytes,thumbWidth:plan.thumbWidth,thumbHeight:plan.thumbHeight,thumbWords:plan.thumbWords,kernels:plan.kernels.map(k=>({size:k.size,char:k.char,angle:k.angle}))})),
 ]);
 await command([...python,join(ROOT,"benchmarks/pursuit_resident_validate.py"),"--input",base],signal);
 const {report}=await Bun.file(join(base,"result.json")).json();
 const eventBytes=await Bun.file(join(base,"events.bin")).arrayBuffer(),u=new Uint32Array(eventBytes),f=new Float32Array(eventBytes),thumbs=new Uint8Array(await Bun.file(join(base,"thumbs.bin")).arrayBuffer());
 const factor=settings.columns*8/plan.setup.width,active:{id:number;tile:AdaptiveTile}[]=[],frames:PursuitFrame[]=[];
 for(let i=0;i<report.events;++i){const n=i*16,op=u[n+3],id=u[n+4],stage=u[n+5],round=u[n+6],k=plan.kernels[u[n+2]];let action="Start with paper";
  if(op!==3){
   if(op===2){const at=active.findIndex(g=>g.id===id);if(at<0)throw new Error("Invalid GPU video history");active.splice(at,1);}
   active.push({id,tile:{x:u[n]*factor,y:u[n+1]*factor,size:k.size*factor,char:k.char,angle:k.angle,color:[f[n+8]*255,f[n+9]*255,f[n+10]*255]}});
   action=`${op===2?"Refine":"Add"} ${k.char} · ${k.size}px`;
  }
  frames.push({iteration:i,stage,activeScale:plan.stages[stage]?.size??null,searchRound:round,energy:f[n+11],pixelError:f[n+12],contourError:f[n+13],glyphs:active.map(g=>g.tile),diversity:new Set(active.map(g=>g.tile.char)).size,action,candidates:u[n+7],scaleCandidates:{},residual:thumbs.slice(i*plan.thumbWords*4,i*plan.thumbWords*4+plan.thumbWidth*plan.thumbHeight),residualWidth:plan.thumbWidth,residualHeight:plan.thumbHeight});
 }
 let priorEnd=0;
 const stages:PursuitStage[]=plan.stages.slice(0,report.stages.length).map((stage,index)=>{
  const matching=frames.filter(frame=>frame.stage===index),end=matching.at(-1)?.iteration??priorEnd,start=priorEnd;priorEnd=end;
  return {index,size:stage.size,startFrame:start,endFrame:end,searchRounds:report.stages[index].searchRounds??1,addedGlyphs:matching.filter(frame=>frame.action.startsWith("Add")).length,converged:report.stages[index].converged};
 });
 const height=Math.round(plan.setup.height*factor),paper=new Uint8Array(height*3);for(let i=0;i<paper.length;++i)paper[i]=Math.round(plan.setup.paperColor[i%3]*255);
 return {width:settings.columns*8,height,columns:0,rows:0,cellWidth:0,cellHeight:0,text:"",colors:new Uint8Array(),paper,tiles:frames.at(-1)!.glyphs,
  pursuit:{frames,stages,schedule:plan.setup.schedule,coarseThreshold:plan.setup.coarseThreshold,coarseRounds:plan.setup.coarseRounds,stopReason:active.length>=plan.setup.budget?"Glyph budget reached":"No improving proposal at any remaining scale (bounded GPU search)",budget:plan.setup.budget,beamWidth:1,edgeWeight:plan.setup.edgeWeight,scales:plan.setup.sizes,analysisWidth:plan.setup.width,analysisHeight:plan.setup.height,backend:"webgpu",gpuResident:true,gpuReadbacks:report.readbacks,gpuInitMs:report.compileMs,gpuRunMs:report.runMs,elapsedMs:report.compileMs+report.runMs,blurStrength:plan.setup.blurStrength,blurCurve:plan.setup.blurModel.curve,effort:plan.setup.effort,vocabulary:plan.setup.vocabulary,kernelCounts:plan.setup.kernelCounts,overlapLimit:plan.setup.maxOverlap,hiddenGlyphsRemoved:0}};
}

/** Render glyphs from their measured masks. Reuse the canvas for append-only edits. */
export function processFrameRasterizer(result:RenderResult,metric:TextMetrics,width:number,height:number):(frame:PursuitFrame)=>Uint8Array{
 if(!metric.masks||!metric.kernelCharacters)throw new Error("Video needs measured glyph masks");
 const chars=[...metric.kernelCharacters],cache=new Map<string,Float32Array>(),pixels=new Uint8Array(width*height*4),scale=width/result.width;
 let previous:AdaptiveTile[]=[];
 const reset=()=>{for(let y=0;y<height;++y)for(let x=0;x<width;++x){const i=(y*width+x)*4;for(let c=0;c<3;++c)pixels[i+c]=result.paper[Math.min(result.height-1,Math.floor(y*result.height/height))*3+c];pixels[i+3]=255;}};
 const paint=(glyph:AdaptiveTile)=>{
  const size=Math.max(1,Math.round(glyph.size*scale)),left=Math.round(glyph.x*scale),top=Math.round(glyph.y*height/result.height),key=`${glyph.char}:${glyph.angle??0}:${size}`;
  let mask=cache.get(key);
  if(!mask){const index=chars.indexOf(glyph.char);if(index<0)throw new Error(`Missing video mask: ${glyph.char}`);const original=metric.masks![index];mask=new Float32Array(size*size);
   for(let y=0;y<size;++y)for(let x=0;x<size;++x){let sum=0,count=0;for(let sy=Math.floor(y*metric.height/size);sy<Math.ceil((y+1)*metric.height/size);++sy)for(let sx=Math.floor(x*metric.width/size);sx<Math.ceil((x+1)*metric.width/size);++sx){sum+=(glyph.angle??0)===90?original[(metric.width-1-sx)*metric.width+sy]:original[sy*metric.width+sx];++count;}mask[y*size+x]=sum/count;}cache.set(key,mask);
  }
  for(let y=0;y<size;++y)for(let x=0;x<size;++x){if(left+x<0||top+y<0||left+x>=width||top+y>=height)continue;const i=((top+y)*width+left+x)*4,a=mask[y*size+x];for(let c=0;c<3;++c)pixels[i+c]=Math.round(pixels[i+c]+a*(glyph.color[c]-pixels[i+c]));}
 };
 return frame=>{const append=frame.glyphs.length===previous.length+1&&previous.every((g,i)=>g===frame.glyphs[i]);if(!append){reset();for(const glyph of frame.glyphs)paint(glyph);}else paint(frame.glyphs.at(-1)!);previous=frame.glyphs;return pixels;};
}

export async function generateProcessVideo(image:string|PixelImage,parameters:ProcessVideoParameters):Promise<ProcessVideoResult>{
 const {signal,onProgress}=parameters;checkAbort(signal);
 const fps=parameters.fps??30,width=parameters.width??768,every=parameters.every??1,initial=parameters.initialHoldSeconds??.5,final=parameters.finalHoldSeconds??1.5;
 if(!parameters.outputPath||!Number.isInteger(fps)||fps<1||fps>120||!Number.isInteger(width)||width<64||width>4096||width%2||!Number.isInteger(every)||every<1||![initial,final].every(v=>Number.isFinite(v)&&v>=0&&v<=60))throw new Error("Invalid process-video parameters");
 if(parameters.backend&&!['auto','cpu','native-gpu'].includes(parameters.backend))throw new Error("Invalid video backend");
 const ffmpeg=parameters.ffmpeg??Bun.which("ffmpeg");if(!ffmpeg)throw new Error("ffmpeg is required to encode the process video");
 const python=parameters.python??join(ROOT,".venv/bin/python"),temporary=await mkdtemp(join(tmpdir(),"glyph-process-")),output=resolve(parameters.outputPath),partial=join(dirname(output),`.${crypto.randomUUID()}.mp4`);
 let encoder:ReturnType<typeof Bun.spawn>|undefined;
 try{
  let input:PixelImage,defaults:Partial<RenderSettings>={};
  if(typeof image==="string"){
   const fixture=join(temporary,"image.json"),args=[python,join(ROOT,"benchmarks/pursuit_fixture.py"),"--image",resolve(image),"--output",fixture];if(parameters.glyphs)args.push("--characters",parameters.glyphs);
   onProgress?.("Loading image and measuring glyphs…");await command(args,signal);const prepared=await Bun.file(fixture).json();input={pixels:new Uint8Array(prepared.pixels),width:prepared.width,height:prepared.height};defaults=prepared.settings;
  }else input=image;
  const settings:RenderSettings={columns:96,font_size:12,layout:"pursuit",use_edge:false,edge_method:"acerola",edge_sensitivity:50,edge_amount:100,use_color:true,background:"#ffffff",background_mode:"source",foreground:"#242424",...defaults,
   pursuit_vocabulary:"full",pursuit_beam:1,pursuit_schedule:"coarse-to-fine",...parameters.settings,pursuit_metrics:parameters.metrics??parameters.settings?.pursuit_metrics??defaults.pursuit_metrics,pursuit_budget:parameters.maxIterations??parameters.settings?.pursuit_budget??2048};
  const renderStart=performance.now();let result:RenderResult,backend:"cpu"|"native-gpu"="cpu",backendReason:string|undefined;
  if(parameters.backend!=="cpu"){
   try{onProgress?.("Running GPU painting loop…");result=await nativeRender(input,settings,temporary,parameters.gpuPython??[python],signal);backend="native-gpu";}
   catch(error){checkAbort(signal);if(parameters.backend==="native-gpu")throw error;backendReason=error instanceof Error?error.message:String(error);onProgress?.("Native GPU unavailable; using CPU optimizer…");result=renderPursuit(input.pixels,input.width,input.height,{...settings,pursuit_backend:"cpu"},onProgress);}
  }else result=renderPursuit(input.pixels,input.width,input.height,{...settings,pursuit_backend:"cpu"},onProgress);
  const renderMs=performance.now()-renderStart,trace=result.pursuit!,converged=trace.stopReason.startsWith("No improving");
  if((parameters.requireConvergence??true)&&!converged)throw new Error(`Optimization stopped before convergence: ${trace.stopReason}. Increase maxIterations or set requireConvergence:false.`);
  const height=Math.max(2,Math.round(width*result.height/result.width/2)*2),raster=processFrameRasterizer(result,settings.pursuit_metrics!,width,height);
  const selected=trace.frames.filter((_,i)=>i%every===0||i===trace.frames.length-1),initialFrames=Math.round(initial*fps),finalFrames=Math.round(final*fps),frameCount=selected.length+initialFrames+finalFrames;
  await mkdir(dirname(output),{recursive:true});checkAbort(signal);onProgress?.(`Encoding ${frameCount} frames…`);const encodeStart=performance.now();
  encoder=Bun.spawn([ffmpeg,"-hide_banner","-loglevel","error","-y","-f","rawvideo","-pixel_format","rgba","-video_size",`${width}x${height}`,"-framerate",String(fps),"-i","pipe:0","-an","-c:v","libx264","-preset","veryfast","-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",partial],{stdin:"pipe",stdout:"ignore",stderr:"pipe"});
  const child=encoder,stderrPromise=new Response(child.stderr).text(),stop=()=>child.kill();signal?.addEventListener("abort",stop,{once:true});
  try{
   const write=async(bytes:Uint8Array)=>{checkAbort(signal);child.stdin.write(bytes);await child.stdin.flush();};
   let bytes=raster(selected[0]);for(let i=0;i<initialFrames;i++)await write(bytes);
   for(const frame of selected)await write(raster(frame));
   bytes=raster(selected.at(-1)!);for(let i=0;i<finalFrames;i++)await write(bytes);
   child.stdin.end();const status=await child.exited,stderr=await stderrPromise;checkAbort(signal);if(status!==0)throw new Error(`Video encoder failed: ${stderr}`);
  }finally{signal?.removeEventListener("abort",stop);}
  await rename(partial,output);
  return {path:output,frames:frameCount,duration:frameCount/fps,iterations:trace.frames.length-1,converged,stopReason:trace.stopReason,width,height,backend,backendReason,renderMs,encodeMs:performance.now()-encodeStart};
 }catch(error){encoder?.kill();await rm(partial,{force:true});throw error;}
 finally{await rm(temporary,{recursive:true,force:true});}
}

if(import.meta.main){
 const [image,output="/tmp/glyph-process.mp4",json="{}"]=Bun.argv.slice(2);if(!image)throw new Error("Usage: bun tools/generate-process-video.ts image output.mp4 '[JSON parameters]'");
 const result=await generateProcessVideo(image,{...JSON.parse(json),outputPath:output,onProgress:message=>process.stderr.write(message+"\n")});console.log(JSON.stringify(result));
}
