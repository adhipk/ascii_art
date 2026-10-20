import {expect,test} from "bun:test";
import {fitShapeGrid,shapePixels,type ShapeAtlas} from "./shape-grid";
import type {RenderResult} from "./renderer";

const atlas: ShapeAtlas={width:4,height:4,size:8,baseline:3,candidates:[
  {text:" ",span:1,mask:Array(16).fill(0)},
  {text:"@",span:1,mask:Array(16).fill(1)},
  {text:"+",span:1,mask:Array(16).fill(.25)},
  {text:"==",span:2,mask:Array.from({length:32},(_,i)=>i%8===3||i%8===4?1:0)},
]};
function base(columns=3): RenderResult {
  return {text:" ".repeat(columns),columns,rows:1,cellWidth:4,cellHeight:4,width:columns*4,height:4,
    colors:new Uint8Array(columns*3),paper:new Uint8Array(12).fill(255)};
}
test("mixed-span DP selects an exact pair and fills an odd final column",()=>{
  const b=base(),pixels=new Uint8ClampedArray(b.width*b.height*4).fill(255);
  b.text="++ ";
  for(let y=0;y<4;y++) for(let x=3;x<=4;x++) for(let c=0;c<3;c++) pixels[(y*b.width+x)*4+c]=0;
  const fitted=fitShapeGrid(b,pixels,atlas);
  expect(fitted.text).toBe("== ");
  expect(fitted.shapes!.stats.error).toBe(0);
  expect(fitted.shapes!.runs.map(r=>r.column)).toEqual([0,2]);
  expect(shapePixels(fitted)).toEqual(pixels);
});
test("production pruning matches exhaustive row composition on random images",()=>{
  let seed=15;
  const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed>>>24;};
  const vocabulary={...atlas,candidates:[...atlas.candidates,
    ...Array.from({length:40},(_,i)=>({text:i%2?"**":"+",span:i%2?2:1,mask:Array.from({length:i%2?32:16},()=>random()/255)}))]};
  for(let sample=0;sample<40;sample++) {
    const b=base(7),pixels=Uint8ClampedArray.from({length:b.width*b.height*4},(_,i)=>i%4===3?255:random());
    b.text=Array.from({length:7},()=>[" ","@","+"][random()%3]).join("");
    const pruned=fitShapeGrid(b,pixels,vocabulary),full=fitShapeGrid(b,pixels,vocabulary,true);
    expect(pruned.text).toBe(full.text);
    expect(pruned.shapes!.runs).toEqual(full.shapes!.runs);
    expect(pruned.shapes!.stats.error).toBe(full.shapes!.stats.error);
  }
});
test("transparent source matches blank paper and invalid dimensions fail",()=>{
  const b=base();
  expect(fitShapeGrid(b,new Uint8ClampedArray(b.width*b.height*4),atlas).text).toBe("   ");
  expect(()=>fitShapeGrid(b,new Uint8ClampedArray(1),atlas)).toThrow();
});

test("explicit directional contours remain single cells and cannot join a pair",()=>{
  const b=base(); b.text=" - "; b.contourCells=new Uint8Array([0,1,0]);
  const vocabulary={...atlas,candidates:[...atlas.candidates,{text:"-",span:1,mask:Array.from({length:16},(_,i)=>i>=8&&i<12?1:0)}]};
  const fitted=fitShapeGrid(b,new Uint8ClampedArray(b.width*b.height*4).fill(255),vocabulary);
  expect(fitted.text).toBe(" - ");
  expect(fitted.shapes!.runs.map(run=>[run.column,vocabulary.candidates[run.candidate].span])).toEqual([[0,1],[1,1],[2,1]]);
});

 test("tone dashes do not lock cells or block a better pair",()=>{
  const b=base(2); b.text="--"; b.contourCells=new Uint8Array(2);
  const pair=Array.from({length:32},(_,i)=>i%8===3||i%8===4?1:0);
  const vocabulary={...atlas,candidates:[...atlas.candidates,{text:"-",span:1,mask:Array(16).fill(.25)}]};
  const pixels=new Uint8ClampedArray(128).fill(255);
  for(let i=0;i<32;i++)for(let c=0;c<3;c++)pixels[i*4+c]=255*(1-pair[i]);
  const fitted=fitShapeGrid(b,pixels,vocabulary);
  expect(fitted.text).toBe("==");
  expect(fitted.shapes!.stats.error).toBe(0);
 });
