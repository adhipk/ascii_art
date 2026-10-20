import { matcher } from "./shape-match";
import type { RenderResult } from "./renderer";

export interface ShapeCandidate { text: string; span: number; mask: number[]; }
export interface ShapeAtlas { width: number; height: number; size: number; baseline: number; candidates: ShapeCandidate[]; }
export interface ShapeRun { row: number; column: number; candidate: number; color: number[]; }
export interface ShapeResult { atlas: ShapeAtlas; runs: ShapeRun[]; stats: { candidates: number; culled: number; early: number; full: number; error: number; }; }

/** The preview and PNG consume these same pixels, with no second shaping pass. */
export function shapePixels(result: RenderResult): Uint8ClampedArray {
  if (!result.shapes) throw new Error("Missing shape composition");
  const {atlas,runs}=result.shapes;
  const pixels=new Uint8ClampedArray(result.width*result.height*4);
  for(let y=0;y<result.height;y++) for(let x=0;x<result.width;x++) {
    const offset=(y*result.width+x)*4;
    for(let c=0;c<3;c++) pixels[offset+c]=result.paper[y*3+c];
    pixels[offset+3]=255;
  }
  for(const run of runs) {
    const candidate=atlas.candidates[run.candidate], width=atlas.width*candidate.span;
    for(let y=0;y<atlas.height;y++) for(let x=0;x<width;x++) {
      const py=run.row*atlas.height+y, offset=(py*result.width+run.column*atlas.width+x)*4;
      const alpha=candidate.mask[y*width+x];
      for(let c=0;c<3;c++) pixels[offset+c]=Math.round(alpha*run.color[c]+(1-alpha)*result.paper[py*3+c]);
    }
  }
  return pixels;
}

/** Exact row composition for coverage SSE plus a tone penalty from the base grid. */
export function fitShapeGrid(base: RenderResult, rgba: Uint8ClampedArray, atlas: ShapeAtlas, exhaustive = false): RenderResult {
  if (base.cellWidth !== atlas.width || base.cellHeight !== atlas.height || rgba.length !== base.width*base.height*4)
    throw new Error("Shape grid dimensions do not match the atlas");
  const densities=atlas.candidates.map(c=>c.mask.reduce((a,b)=>a+b,0)/c.mask.length);
  const rampMaximum=Math.max(...atlas.candidates.flatMap((c,i)=>c.span===1 && " .:-=+*#%@".includes(c.text) ? [densities[i]] : []));
  const toneByCharacter=new Map(atlas.candidates.flatMap((c,i)=>c.span===1 ? [[c.text,densities[i]/Math.max(rampMaximum,1e-8)] as const] : []));
  const groups = [1,2].map(span => {
    const indices = atlas.candidates.flatMap((c,i) => c.span === span ? [i] : []);
    if (!indices.length) throw new Error("Shape atlas requires singles and pairs");
    return {indices, match: matcher(indices.map(i => atlas.candidates[i].mask), atlas.width*span, atlas.height)};
  });
  const stats = {candidates:0, culled:0, early:0, full:0, error:0};
  const runs: ShapeRun[] = [], lines: string[] = [];
  const original = base.text.split("\n");
  const luma = (r:number,g:number,b:number) => .299*r+.587*g+.114*b;
  for (let row = 0; row < base.rows; row++) {
    const costs = new Float64Array(base.columns+1); costs.fill(Infinity); costs[base.columns]=0;
    const choices: (ShapeRun | undefined)[] = new Array(base.columns);
    for (let column = base.columns-1; column >= 0; column--) {
      for (let span = 1; span <= 2 && column+span <= base.columns; span++) {
        const color = [0,1,2].map(channel => {
          let sum=0; for(let c=0;c<span;c++) sum+=base.colors[(row*base.columns+column+c)*3+channel];
          return Math.round(sum/span);
        });
        const ink = luma(color[0],color[1],color[2]);
        const target: number[] = [];
        for(let y=0;y<atlas.height;y++) {
          const py=row*atlas.height+y, paper=luma(...Array.from(base.paper.slice(py*3,py*3+3)) as [number,number,number]);
          for(let x=0;x<atlas.width*span;x++) {
            const offset=(py*base.width+column*atlas.width+x)*4;
            const alpha=rgba[offset+3]/255;
            const brightness=luma(rgba[offset],rgba[offset+1],rgba[offset+2])*alpha+paper*(1-alpha);
            const delta=ink-paper;
            target.push(Math.abs(delta)<1e-6 ? 0 : Math.max(0,Math.min(1,(brightness-paper)/delta)));
          }
        }
        const group=groups[span-1];
        const desiredTone=[...original[row].slice(column,column+span)].reduce((sum,char)=>sum+(toneByCharacter.get(char)??0),0)/span;
        const toneCosts=group.indices.map(i=>4*target.length*(densities[i]/Math.max(rampMaximum,1e-8)-desiredTone)**2);
        // Preserve explicit contour replacements from the base edge detector.
        const edge = base.contourCells?.subarray(row*base.columns+column,row*base.columns+column+span).some(Boolean) ?? false;
        if (span===2 && edge) continue;
        const forced = span===1 && edge ? group.indices.find(i=>atlas.candidates[i].text===original[row][column]) : undefined;
        if(span===1 && edge && forced===undefined)
          throw new Error(`Shape atlas is missing contour character: ${original[row][column]}`);
        let candidate: number, error: number;
        if(forced!==undefined) {
          candidate=forced; error=target.reduce((sum,v,i)=>sum+(v-atlas.candidates[forced].mask[i])**2,0);
        } else {
          const found=group.match(target,!exhaustive,toneCosts);
          candidate=group.indices[found.winner]; error=found.error;
          stats.candidates+=group.indices.length; stats.culled+=found.culled.reduce((a,b)=>a+b,0);
          stats.early+=found.early; stats.full+=found.full;
        }
        const cost=error+costs[column+span];
        if(cost<costs[column]) {costs[column]=cost; choices[column]={row,column,candidate,color};}
      }
    }
    stats.error+=costs[0];
    let line="";
    for(let col=0;col<base.columns;) {
      const run=choices[col]!; runs.push(run); const candidate=atlas.candidates[run.candidate];
      line+=candidate.text; col+=candidate.span;
    }
    lines.push(line);
  }
  return {...base, text:lines.join("\n"), shapes:{atlas,runs,stats}};
}
