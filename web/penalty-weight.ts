export function penaltyAtScale(weight:number,size:number,minSize=0):number {
  return size<minSize?0:weight;
}

export function penaltyWeightsByScale(weight:number,sizes:readonly number[],minSize=0):Record<string,number> {
  return Object.fromEntries(sizes.map(size=>[String(size),penaltyAtScale(weight,size,minSize)]));
}
