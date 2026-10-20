import { matcher } from "../web/shape-match";
export { matcher, summaries } from "../web/shape-match";

if (import.meta.main) {
  const data = await Bun.file(process.argv[2]).json();
  const match = matcher(data.candidates.map((c: any) => c.pixels), data.width, data.height);
  const output: any = {font: data.font, dimensions: [data.width, data.height],
    rawCandidates: data.rawCandidates, candidates: data.candidates.length,
    changedSequences: data.changedSequences, groups: {}};
  for (const [name, patches] of Object.entries(data.groups) as [string, number[][]][]) {
    for (const patch of patches) { match(patch, false); match(patch, "early"); match(patch, true); }
    const timing = (pruned: boolean | "early") => {
      const times = [];
      for (let repeat = 0; repeat < 7; repeat++) {
        const start = performance.now();
        for (const patch of patches) match(patch, pruned);
        times.push(performance.now() - start);
      }
      return times.sort((a,b) => a-b)[3];
    };
    const totals = {culled: [0,0,0], full: 0, early: 0, pixels: 0, mismatches: 0};
    for (const patch of patches) {
      const a = match(patch, false), b = match(patch, true);
      const early = match(patch, "early");
      if (a.winner !== b.winner || a.error !== b.error || a.winner !== early.winner || a.error !== early.error) totals.mismatches++;
      b.culled.forEach((n,i) => totals.culled[i] += n);
      totals.full += b.full; totals.early += b.early; totals.pixels += b.pixels;
    }
    output.groups[name] = {patches: patches.length, exhaustiveMs: timing(false), earlyExitMs: timing("early"),
      prunedMs: timing(true), ...totals};
  }
  console.log(JSON.stringify(output, null, 2));
}
