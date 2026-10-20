import { preparePursuit } from "./pursuit";
import { residentShader } from "./pursuit-resident-shader";
import type { RenderSettings } from "./renderer";
export const WORK_HEADER_BYTES=59088, HISTORY_GLYPHS_BYTES=65536, HISTORY_EVENTS_BYTES=262144;
export interface Dispatch {entry:string;x:number;y?:number;}
export function createResidentPlan(pixels:Uint8Array,width:number,height:number,settings:RenderSettings){
 const setup=preparePursuit(pixels,width,height,settings);
 if(setup.beamWidth!==1||setup.schedule!=="coarse-to-fine")throw new Error("Resident GPU search needs one composition and coarse-to-fine painting order");
 const kernels=[...setup.problem.kernels].filter(k=>k.size<=setup.width&&k.size<=setup.height).sort((a,b)=>b.size-a.size),levels=[];
 let w=setup.width,h=setup.height,offset=0,patchOffset=0,maxSide=Math.max(...kernels.map(k=>k.size)),maxHeight=maxSide,maxSamples=0;
 const rowStride=maxSide+4;
 for(let l=0;l<setup.blurModel.weights.length;++l){
  levels.push({offset,width:w,height:h,weight:setup.blurModel.weights[l],patchOffset,maxHeight});
  offset+=w*h;patchOffset+=rowStride*maxHeight;maxSamples+=(maxHeight+1)**2;
  w=Math.ceil(w/2);h=Math.ceil(h/2);maxHeight=Math.ceil(maxHeight/2)+2;
 }
 if(!kernels.length)throw new Error("No GPU kernel fits the image dimensions");
 const partialStride=Math.ceil(maxSamples/128),patchStride=patchOffset,n=setup.width*setup.height;
 const reference=new Float32Array(offset*4);
 for(let i=0;i<n;++i)reference.set(setup.target.subarray(i*3,i*3+3),i*4);
 let maskSize=0;const metadata=new ArrayBuffer(kernels.length*32);
 const mu=new Uint32Array(metadata),mf=new Float32Array(metadata);
 for(const [i,k] of kernels.entries()){mu.set([k.size,maskSize,0,0],i*8);mf.set([k.inkX??k.size/2,k.inkY??k.size/2,0,0],i*8+4);maskSize+=k.alpha.length;}
 const masks=new Float32Array(maskSize);let maskOffset=0;for(const k of kernels){masks.set(k.alpha,maskOffset);maskOffset+=k.alpha.length;}
 const stages=setup.sizes.map(size=>({size,start:Math.max(0,kernels.findIndex(k=>k.size===size)),count:kernels.filter(k=>k.size===size).length}));
 const maxCandidates=Math.max(1,...stages.map(s=>s.count*30));
 const thumbWidth=Math.max(1,Math.round(setup.width*Math.min(1,48/Math.max(setup.width,setup.height))));
 const thumbHeight=Math.max(1,Math.round(setup.height*Math.min(1,48/Math.max(setup.width,setup.height))));
 const thumbWords=Math.ceil(thumbWidth*thumbHeight/4),historyBytes=HISTORY_GLYPHS_BYTES+HISTORY_EVENTS_BYTES+4096*thumbWords*4;
 const dataVectors=2*n+16*patchStride+16*partialStride*2;
 const workBytes=WORK_HEADER_BYTES+dataVectors*16;
 function parameters(stage=0,round=0,mode=0,refineID=0,minimumGain=1e-9):Uint8Array{
  const bytes=new ArrayBuffer(416),u=new Uint32Array(bytes),f=new Float32Array(bytes);
  u.set([setup.width,setup.height,kernels.length,setup.budget,levels.length,stage,round,mode,refineID,Number(setup.effort==="thorough"),thumbWidth,Number(settings.use_color)]);
  f.set([setup.edgeWeight,setup.maxOverlap,minimumGain,setup.overlapPenalty],12);f.set([...setup.paperColor,0],16);
  u.set([rowStride,Math.ceil(offset/128),thumbHeight,0],20);
  levels.forEach((l,i)=>{u.set([l.offset,l.width,l.height,l.patchOffset],24+i*8);f[28+i*8]=l.weight;});
  stages.forEach((s,i)=>u.set([s.start,s.count,s.size,0],72+i*4));
  return new Uint8Array(bytes);
 }
 const d=(entry:string,x=1,y?:number):Dispatch=>({entry,x,...(y?{y}:{})});
 const pyramid=levels.slice(1).map((l,i)=>d(`image_${i+1}`,Math.ceil(l.width*l.height/128)));
 const statistics=[...pyramid,d("statistics",Math.ceil(offset/128)),d("statistics_sum")];
 const record=[d("thumbnails",Math.ceil(thumbWords/128)),d("record_event")];
 const exact:Dispatch[]=[...levels.map((l,i)=>d(`patch_${i}`,Math.ceil(rowStride*l.maxHeight/128),12)),d("fit_parts",partialStride,12),d("fit_colors",12),d("gain_parts",partialStride,12),d("gain_sum",12),d("choose_best")];
 const search=[d("guidance",Math.ceil(n/128)),d("hotspots",Math.ceil(Math.ceil(setup.width/8)*Math.ceil(setup.height/8)/64)),d("choose_hotspots"),d("rank_candidates",Math.ceil(maxCandidates/64)),d("shortlist"),...exact,d("local_large"),...exact,d("local_small"),...exact];
 const iteration=[...statistics,...record,...search,d("paint",Math.ceil(n/128)),d("accept")];
 const initialize=[d("initialize",Math.ceil(n/128)),...levels.slice(1).map((l,i)=>d(`reference_${i+1}`,Math.ceil(l.width*l.height/128))),...statistics,...record];
 const refinement=[d("begin_refine"),d("rebuild",Math.ceil(n/128)),...statistics,...search,d("accept"),d("rebuild",Math.ceil(n/128)),...statistics,...record];
 return {setup,kernels,levels,reference,masks,metadata,parameters,stages,maxCandidates,workBytes,historyBytes,thumbWidth,thumbHeight,thumbWords,patchStride,partialStride,initialize,iteration,refinement,flush:[...statistics,...record],shader:residentShader(patchStride,partialStride)};
}
