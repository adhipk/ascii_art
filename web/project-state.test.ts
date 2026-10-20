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
