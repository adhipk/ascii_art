import {expect,test} from "bun:test";
import {glyphBudgetTimeline,frameAtGlyphBudget,budgetHistoryKey} from "./glyph-budget-history";
import type {RenderSettings} from "./renderer";

test("budgets select best recorded compositions despite removals and repeated counts",()=>{
  const frame=(count:number,energy:number)=>({glyphs:Array(count).fill(null),energy});
  const timeline=glyphBudgetTimeline([frame(0,100),frame(1,80),frame(2,60),frame(1,40),frame(4,30),frame(2,50)]);
  expect(timeline).toEqual([{count:0,index:0},{count:1,index:3},{count:2,index:5},{count:4,index:4}]);
  expect(frameAtGlyphBudget(timeline,3)).toBe(5);
  expect(frameAtGlyphBudget(timeline,0)).toBe(0);
  expect(frameAtGlyphBudget(timeline,100)).toBe(4);
  expect(frameAtGlyphBudget([],0)).toBe(0);
});

test("cached histories match budget-only edits and invalidate on source or fitting changes",()=>{
  const settings={layout:"pursuit",columns:100,pursuit_budget:128,pursuit_blur:4} as RenderSettings;
  expect(budgetHistoryKey(settings,1)).toBe(budgetHistoryKey({...settings,pursuit_budget:32},1));
  expect(budgetHistoryKey(settings,1)).not.toBe(budgetHistoryKey(settings,2));
  expect(budgetHistoryKey(settings,1)).not.toBe(budgetHistoryKey({...settings,pursuit_blur:2},1));
});
