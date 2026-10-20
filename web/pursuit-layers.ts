import type {RenderResult,AdaptiveTile} from "./renderer";
export type LayerVisibility=Record<string,boolean>;
export function scaleRangeVisibility(scales:readonly number[],minimum:number,maximum:number):LayerVisibility {
  const low=Math.min(minimum,maximum),high=Math.max(minimum,maximum);
  return Object.fromEntries(scales.map(scale=>[String(scale),scale>=low&&scale<=high]));
}
export function pursuitLayers(result:RenderResult,visibility:LayerVisibility={}) {
  const factor=result.pursuit?result.width/result.pursuit.analysisWidth:1;
  const layers:Record<string,{scale:number;visible:boolean;glyphs:{paintIndex:number;tile:AdaptiveTile}[]}>={};
  for(const scale of result.pursuit?.scales??[])layers[String(scale)]={scale,visible:visibility[String(scale)]!==false,glyphs:[]};
  for(const [paintIndex,tile]of (result.tiles??[]).entries()) {
    const scale=Math.round(tile.size/factor),key=String(scale);
    (layers[key]??={scale,visible:visibility[key]!==false,glyphs:[]}).glyphs.push({paintIndex,tile});
  }
  return layers;
}
export function visiblePursuitLayers(result:RenderResult,visibility:LayerVisibility):RenderResult {
  if(!result.pursuit||!result.tiles)return result;
  const factor=result.width/result.pursuit.analysisWidth;
  return {...result,tiles:result.tiles.filter(tile=>visibility[String(Math.round(tile.size/factor))]!==false)};
}
