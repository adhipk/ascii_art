import {expect,test} from "bun:test";
import {pursuitLayers,visiblePursuitLayers,scaleRangeVisibility} from "./pursuit-layers";
import type {RenderResult} from "./renderer";
test("cached layer toggles preserve original paint order, frame data, and exact scale mapping",()=>{
 const result={width:100,pursuit:{analysisWidth:128,scales:[32,8]},tiles:[
 {x:0,y:0,size:25,char:"█",color:[255,255,255]},
 {x:1,y:1,size:6.25,char:"é",color:[0,0,0]},
 {x:2,y:2,size:25,char:"界",color:[255,0,0]}]} as RenderResult;
 const off={8:false},layers=pursuitLayers(result,off);
 expect(layers["32"].glyphs.map(g=>g.paintIndex)).toEqual([0,2]);
 expect(layers["8"].visible).toBe(false);
 expect(visiblePursuitLayers(result,off).tiles!.map(t=>t.char)).toEqual(["█","界"]);
 expect(visiblePursuitLayers(result,{}).tiles).toEqual(result.tiles);
 expect(visiblePursuitLayers(result,{8:false,32:false}).tiles).toEqual([]);
 expect(result.tiles!.length).toBe(3);
});
test("scale checkpoints isolate details or underpainting without changing saved layers",()=>{
 const scales=[48,24,12,6,3];
 expect(scaleRangeVisibility(scales,3,12)).toEqual({48:false,24:false,12:true,6:true,3:true});
 expect(scaleRangeVisibility(scales,24,48)).toEqual({48:true,24:true,12:false,6:false,3:false});
 expect(scaleRangeVisibility(scales,12,12)).toEqual({48:false,24:false,12:true,6:false,3:false});
 expect(scaleRangeVisibility(scales,12,3)).toEqual(scaleRangeVisibility(scales,3,12));
 expect(scaleRangeVisibility(scales,3,48)).toEqual({48:true,24:true,12:true,6:true,3:true});
 expect(scales).toEqual([48,24,12,6,3]);
});
