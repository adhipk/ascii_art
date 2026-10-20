export function allocateLayerBudgets(scales:readonly number[],active:readonly number[],baseBudget:number):Record<string,number> {
  return Object.fromEntries(scales.map(size=>[String(size),active.includes(size)?baseBudget:0]));
}
