import type { RenderResult, RenderSettings } from "./renderer";
import {penaltyWeightsByScale} from "./penalty-weight";

export function debugStatePayload(data:{source:unknown;runSettings:RenderSettings;pendingControls:Record<string,unknown>;result:RenderResult;selectedIteration:number;layerVisibility:Record<string,boolean>}):string {
  const {text_metrics,adaptive_metrics,pursuit_metrics,...settings}=data.runSettings;
  const trace=data.result.pursuit,frame=trace?.frames[data.selectedIteration];
  const {frames,...traceSummary}=trace??{frames:undefined};
  const glyphs=frame?.glyphs??data.result.tiles??[];
  const countsByScale:Record<string,number>={},countsByCharacter:Record<string,number>={};
  const factor=trace ? data.result.width/trace.analysisWidth : 1;
  for(const glyph of glyphs){
    const scale=String(Math.round(glyph.size/factor));
    countsByScale[scale]=(countsByScale[scale]??0)+1;
    countsByCharacter[glyph.char]=(countsByCharacter[glyph.char]??0)+1;
  }
  return JSON.stringify({format:"ascii-debug-state-v1",capturedAt:new Date().toISOString(),source:data.source,
    runSettings:{...settings,pursuit_overlap_penalty:settings.pursuit_overlap_penalty??0,pursuit_repeat_penalty:settings.pursuit_repeat_penalty??0,
      measuredFont:pursuit_metrics?{width:pursuit_metrics.width,height:pursuit_metrics.height,fontSize:pursuit_metrics.fontSize,kernelCharacters:pursuit_metrics.kernelGlyphs??pursuit_metrics.kernelCharacters}:undefined},
    penaltyWeightsByScale:trace?{
      overlap:penaltyWeightsByScale(trace.overlapPenalty??0,trace.scales,trace.penaltyMinSize),
      repeat:penaltyWeightsByScale(trace.repeatPenalty??0,trace.scales,trace.penaltyMinSize)}:undefined,
    pendingControls:data.pendingControls,
    output:{width:data.result.width,height:data.result.height,selectedIteration:frame?data.selectedIteration:null,layerVisibility:data.layerVisibility,glyphCount:glyphs.length,countsByScale,countsByCharacter},
    trace:trace?{...traceSummary,frameCount:frames!.length,selectedFrame:frame?{
      iteration:frame.iteration,stage:frame.stage,activeScale:frame.activeScale,searchRound:frame.searchRound,action:frame.action,
      energy:frame.energy,pixelError:frame.pixelError,contourError:frame.contourError,overlapError:frame.overlapError,repeatError:frame.repeatError,
      overlapCost:(trace.overlapPenalty??0)*(frame.overlapError??0),repeatCost:(trace.repeatPenalty??0)*(frame.repeatError??0),
      diversity:frame.diversity,candidates:frame.candidates,scaleCandidates:frame.scaleCandidates}:undefined}:undefined,
    omitted:["embedded images","glyph placements","font masks","pixel arrays","residual arrays","other frames"]},null,2);
}

/** One self-contained payload works in a text file or a rich clipboard paste. */
export function debugArtifactPayload(data: Record<string,unknown> & {images:{original:string;output:string}}): {json:string;html:string} {
  const json=JSON.stringify({format:"ascii-debug-v1",...data},(_key,value)=>
    ArrayBuffer.isView(value) && !(value instanceof DataView) ? Array.from(value as unknown as ArrayLike<number>) : value,2);
  const escape=(text:string)=>text.replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;");
  return {json,html:`<article><h1>ASCII debug artifact</h1><h2>Original</h2><img alt="Original input" src="${escape(data.images.original)}"><h2>Selected output</h2><img alt="Selected ASCII output" src="${escape(data.images.output)}"><h2>Run and pending configuration</h2><pre>${escape(json)}</pre></article>`};
}
