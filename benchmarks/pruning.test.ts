import { expect, test } from "bun:test";
import { matcher, summaries } from "./pruning";

test("block bounds never exceed pixel error, including uneven blocks", () => {
  for (const [w,h] of [[8,8], [19,24]]) {
    const a = Array.from({length:w*h}, (_,i) => (i*37)%256);
    const b = Array.from({length:w*h}, (_,i) => (i*71+3)%256);
    const exact = a.reduce((sum,v,i) => sum+(v-b[i])**2,0);
    for (const grid of [1,2,4]) {
      const sa = summaries(a,w,h,grid), sb = summaries(b,w,h,grid);
      expect(sa.reduce((sum,v,i) => sum+(v-sb[i])**2,0)).toBeLessThanOrEqual(exact+1e-6);
    }
  }
});

test("pruning preserves exhaustive winners and ties across deterministic random patches", () => {
  let state = 42;
  const random = () => {state = (Math.imul(state,1664525)+1013904223)>>>0; return state>>>24;};
  const masks = Array.from({length:100}, () => Array.from({length:64}, random));
  masks.push(masks[0].slice());
  const match = matcher(masks,8,8);
  const patches = [...masks, ...Array.from({length:100}, () => Array.from({length:64}, random))];
  for (const patch of patches) {
    const reference = match(patch,false), actual = match(patch,true);
    expect(actual.winner).toBe(reference.winner);
    expect(actual.error).toBe(reference.error);
  }
});

test("candidate-specific tone costs participate in both bounds and final scoring",()=>{
  const masks=[Array(64).fill(0),Array(64).fill(.5),Array(64).fill(1)];
  const match=matcher(masks,8,8), target=Array(64).fill(0);
  const costs=[100,0,0];
  expect(match(target,true,costs).winner).toBe(1);
  expect(match(target,true,costs).error).toBe(match(target,false,costs).error);
});
