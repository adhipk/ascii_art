import {penaltyAtScale} from "./penalty-weight";

export interface RepeatGlyph { x:number; y:number; size:number; char:string; }

export function repeatPairCost(first:RepeatGlyph,second:RepeatGlyph,area:number,minSize=0):number {
  if(first.char!==second.char || !first.char.trim() || !penaltyAtScale(1,Math.min(first.size,second.size),minSize))return 0;
  const radius=first.size+second.size;
  const distance=Math.hypot(first.x+first.size/2-second.x-second.size/2,first.y+first.size/2-second.y-second.size/2);
  return Math.min(first.size,second.size)**2/area*Math.max(0,1-distance/radius)**2;
}

export function repeatAddition(glyph:RepeatGlyph,existing:readonly RepeatGlyph[],area:number,minSize=0):number {
  return existing.reduce((sum,other)=>sum+repeatPairCost(glyph,other,area,minSize),0);
}

export function repeatError(glyphs:readonly RepeatGlyph[],area:number,minSize=0):number {
  let total=0;
  for(let index=0;index<glyphs.length;index++)
    for(let previous=0;previous<index;previous++)total+=repeatPairCost(glyphs[index],glyphs[previous],area,minSize);
  return total;
}
