import { expect, test } from "bun:test";
import { IterationHistory } from "./iteration-history";

test("history can seek every frame and replacement runs clear playback", async () => {
  const seen: number[] = [];
  const history = new IterationHistory(index => seen.push(index));
  history.load(4); expect(history.index).toBe(3);
  for(const index of [0,1,2,3,2,0]) {history.seek(index); expect(history.index).toBe(index);}
  history.seek(-10); expect(history.index).toBe(0);
  history.seek(10); expect(history.index).toBe(3);
  history.play(); expect(history.playing).toBe(true); expect(history.index).toBe(0);
  await new Promise(resolve => setTimeout(resolve, 180));
  expect(history.index).toBe(1);
  history.load(1); expect(history.playing).toBe(false); expect(history.index).toBe(0);
  history.play(); expect(history.playing).toBe(false);
  expect(seen).toContain(2);
});

test("playback pauses at the final recorded iteration", async () => {
  const history = new IterationHistory(() => {});
  history.load(2); history.play();
  await new Promise(resolve => setTimeout(resolve, 180));
  expect(history.index).toBe(1); expect(history.playing).toBe(false);
});

test("glyph playback follows count order and loops until paused",async()=>{
  const seen:number[]=[];
  const history=new IterationHistory(index=>seen.push(index));
  history.load(4); history.play([2,0,3],true);
  try {
    expect(history.index).toBe(2);
    await new Promise(resolve=>setTimeout(resolve,510));
    expect(history.playing).toBe(true);
    expect(seen).toContain(0); expect(seen).toContain(3);
    expect(seen.filter(i=>i===2).length).toBeGreaterThanOrEqual(2);
  } finally { history.pause(); }
  expect(history.playing).toBe(false);
});
