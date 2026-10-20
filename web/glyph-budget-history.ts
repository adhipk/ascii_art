import type { RenderSettings } from "./renderer";

export interface BudgetFrame { glyphs: readonly unknown[]; energy: number; }
/** Each available count maps to its best recorded composition, including removals. */
export function glyphBudgetTimeline(frames: readonly BudgetFrame[]): {count:number;index:number}[] {
  const best=new Map<number,number>();
  frames.forEach((frame,index)=>{
    const count=frame.glyphs.length,previous=best.get(count);
    if(previous===undefined || frame.energy<=frames[previous].energy) best.set(count,index);
  });
  return [...best].sort(([a],[b])=>a-b).map(([count,index])=>({count,index}));
}
export function frameAtGlyphBudget(timeline: readonly {count:number;index:number}[], budget: number): number {
  if(!timeline.length) return 0;
  let selected=timeline[0].index;
  for(const frame of timeline) {if(frame.count>budget) break; selected=frame.index;}
  return selected;
}
/** Changing a budget alone can replay a trace; all other settings and source must match. */
export function budgetHistoryKey(settings: RenderSettings, sourceVersion: number): string {
  const {pursuit_budget,...rest}=settings;
  return JSON.stringify([sourceVersion,rest]);
}
