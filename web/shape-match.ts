/** Exact squared-error matching with successive disjoint-block lower bounds. */
export function summaries(pixels: number[], width: number, height: number, grid: number): number[] {
  const values: number[] = [];
  for (let by = 0; by < grid; by++) for (let bx = 0; bx < grid; bx++) {
    let sum = 0;
    const x0 = Math.floor(bx * width / grid), x1 = Math.floor((bx + 1) * width / grid);
    const y0 = Math.floor(by * height / grid), y1 = Math.floor((by + 1) * height / grid);
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) sum += pixels[y * width + x];
    values.push(sum / Math.sqrt((x1 - x0) * (y1 - y0)));
  }
  return values;
}
function distance(a: number[], b: number[], limit = Infinity): [number, number] {
  let error = 0, visited = 0;
  for (; visited < a.length; visited++) {
    error += (a[visited] - b[visited]) ** 2;
    if (error > limit) return [error, visited + 1];
  }
  return [error, visited];
}
export function matcher(masks: number[][], width: number, height: number) {
  const grids = [1, 2, 4];
  const atlas = masks.map(mask => grids.map(grid => summaries(mask, width, height, grid)));
  return (target: number[], mode: boolean | "early", extraCosts?: number[]) => {
    const pruned = mode === true;
    const query = pruned ? grids.map(grid => summaries(target, width, height, grid)) : [];
    // Cheapest descriptor ranks a seed; retain original index order for exact ties.
    let seed = 0;
    if (pruned) for (let i = 1; i < masks.length; i++)
      if (Math.abs(atlas[i][0][0] - query[0][0]) < Math.abs(atlas[seed][0][0] - query[0][0])) seed = i;
    let best = distance(target, masks[seed])[0] + (extraCosts?.[seed] ?? 0), winner = seed;
    const culled = [0, 0, 0];
    let full = 1, early = 0, pixels = target.length;
    for (let i = 0; i < masks.length; i++) {
      if (i === seed) continue;
      let rejected = false;
      if (pruned) for (let stage = 0; stage < grids.length; stage++) {
        // Conservative tolerance avoids rejecting ties due to floating-point roundoff.
        if (distance(query[stage], atlas[i][stage])[0] + (extraCosts?.[i] ?? 0) > best + 1e-6 * Math.max(1, best)) {
          culled[stage]++; rejected = true; break;
        }
      }
      if (rejected) continue;
      const extra=extraCosts?.[i] ?? 0;
      const [pixelError, visited] = distance(target, masks[i], mode ? best-extra : Infinity);
      const error=pixelError+extra;
      pixels += visited;
      if (visited === target.length) full++; else early++;
      if (error < best || (error === best && i < winner)) { best = error; winner = i; }
    }
    return {winner, error: best, culled, full, early, pixels};
  };
}

