import {expect,test} from "bun:test";
import {encodeProject,decodeProject,exportOptions,type Options} from "./project-state";
import type {RenderSettings} from "./renderer";

const options:Options={layout:"fixed",columns:"100","font-size":"12","shape-matching":false,edges:false,
  "edge-method":"acerola","edge-min-pixels":"8","edge-amount":"20",color:false,"background-mode":"source",background:"#ffffff"};
const metrics={width:8,height:17,advance:7.2,baseline:12,densities:[0,.01,.02,.03,.04,.05,.06,.07,.08,.1]};
const settings={text_metrics:metrics} as RenderSettings;
test("projects preserve exact RGBA bytes, controls and measured metrics",()=>{
  const source={width:2,height:1,data:new Uint8ClampedArray([1,2,3,0,5,6,7,128])};
  const restored=decodeProject(encodeProject(source,options,settings));
  expect(restored.pixels).toEqual(source.data);
  expect(restored.options).toEqual(options);
  expect(restored.metrics.text_metrics).toEqual(metrics);
  expect(exportOptions(options)).toContain("shape-matching: false\n");
  expect(exportOptions(options)).toContain("columns: 100\n");
});
test("projects reject unsupported versions, invalid controls and pixel lengths",()=>{
  const valid=JSON.parse(encodeProject({width:1,height:1,data:new Uint8ClampedArray(4)},options,settings));
  for(const change of [
    (v:any)=>v.format="future",(v:any)=>v.options.columns="10000",
    (v:any)=>v.options.color="false",(v:any)=>v.source.rgba="AA==",
    (v:any)=>v.metrics.text_metrics.width=0,
  ]) {
    const v=structuredClone(valid); change(v); expect(()=>decodeProject(JSON.stringify(v))).toThrow();
  }
});

test("shape projects require every tone-ramp and contour glyph before restoration",()=>{
  const shapeOptions={...options,"shape-matching":true};
  const atlas={width:8,height:17,size:12,baseline:12,candidates:[
    ...[..." .:-=+*#%@/|\\"].map(text=>({text,span:1,mask:Array(136).fill(.25)})),
    {text:"==",span:2,mask:Array(272).fill(.25)}]};
  const complete=JSON.parse(encodeProject({width:1,height:1,data:new Uint8ClampedArray(4)},shapeOptions,{text_metrics:metrics,shape_atlas:atlas} as RenderSettings));
  expect(()=>decodeProject(JSON.stringify(complete))).not.toThrow();
  for(const char of " .:-=+*#%@/|\\") {
    const invalid=structuredClone(complete);
    invalid.metrics.shape_atlas.candidates=invalid.metrics.shape_atlas.candidates.filter((c:any)=>c.text!==char);
    expect(()=>decodeProject(JSON.stringify(invalid))).toThrow("Incomplete shape atlas");
  }
});

test("imported metrics cannot override visible renderer settings",()=>{
  const project=JSON.parse(encodeProject({width:1,height:1,data:new Uint8ClampedArray(4)},options,settings));
  Object.assign(project.metrics,{columns:21,layout:"adaptive",font_size:20,use_color:true,background:"#000000"});
  const restored=decodeProject(JSON.stringify(project));
  expect(Object.keys(restored.metrics).sort()).toEqual(["adaptive_metrics","shape_atlas","text_metrics"]);
  const effective={columns:Number(restored.options.columns),layout:restored.options.layout,font_size:Number(restored.options["font-size"]),use_color:restored.options.color,background:restored.options.background,...restored.metrics};
  expect(effective.columns).toBe(100);
  expect(effective.layout).toBe("fixed");
  expect(effective.font_size).toBe(12);
  expect(effective.use_color).toBe(false);
  expect(effective.background).toBe("#ffffff");
  expect(effective.text_metrics).toEqual(metrics);
});
