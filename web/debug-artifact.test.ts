import {expect,test} from "bun:test";
import {debugArtifactPayload,debugStatePayload} from "./debug-artifact";
import {renderPursuit} from "./pursuit";
import type {RenderSettings} from "./renderer";

test("clipboard JSON embeds both images, exact typed state and distinct pending configuration",()=>{
  const images={original:"data:image/png;base64,original",output:"data:image/png;base64,selected"};
  const payload=debugArtifactPayload({images,runSettings:{pursuit_characters_by_scale:{"32":"A"}},pendingControls:{"scale-characters-32":"B"},output:{selectedIteration:3,paper:new Uint8Array([1,2,3])}});
  const parsed=JSON.parse(payload.json);
  expect(parsed.format).toBe("ascii-debug-v1");expect(parsed.images).toEqual(images);
  expect(parsed.runSettings.pursuit_characters_by_scale["32"]).toBe("A");
  expect(parsed.pendingControls["scale-characters-32"]).toBe("B");
  expect(parsed.output).toEqual({selectedIteration:3,paper:[1,2,3]});
  expect(payload.html).toContain('alt="Original input"');
  expect(payload.html).toContain('alt="Selected ASCII output"');
});

test("rich clipboard escapes character sets and config text",()=>{
  const payload=debugArtifactPayload({images:{original:"data:image/png;base64,x",output:"data:image/png;base64,y"},pendingControls:{glyphs:'</pre><script>alert("x")</script>'}});
  expect(payload.html).not.toContain("<script>");
  expect(payload.html).toContain("&lt;script&gt;");
  expect(JSON.parse(payload.json).pendingControls.glyphs).toContain("<script>");
});

test("compact state preserves accepted settings and selected metrics without bulky render data",()=>{
  const runSettings:RenderSettings={columns:24,font_size:12,layout:"pursuit",use_edge:false,edge_method:"acerola",edge_sensitivity:50,edge_amount:0,
    use_color:false,background:"#000000",background_mode:"solid",foreground:"#ffffff",pursuit_resolution:128,pursuit_scales:[64],
    pursuit_budget:2,pursuit_beam:1,pursuit_coarse_threshold:0,pursuit_edge_weight:0,pursuit_blur:0,pursuit_overlap_penalty:.2,pursuit_repeat_penalty:.5,
    pursuit_metrics:{width:4,height:4,baseline:4,advance:4,fontSize:4,densities:[],kernelCharacters:" X",masks:[Array(16).fill(0),Array(16).fill(.6)]}};
  const source=new Uint8Array(16*16*4).fill(255),result=renderPursuit(source,16,16,runSettings);
  const selectedIteration=1,frame=result.pursuit!.frames[selectedIteration];
  const payload=debugStatePayload({source:{width:16,height:16},runSettings,pendingControls:{"repeat-penalty":"2"},result,selectedIteration,layerVisibility:{"64":false}});
  const parsed=JSON.parse(payload);
  expect(parsed.format).toBe("ascii-debug-state-v1");
  expect(parsed.runSettings.pursuit_repeat_penalty).toBe(.5);
  expect(parsed.pendingControls["repeat-penalty"]).toBe("2");
  expect(parsed.trace.selectedFrame.energy).toBe(frame.energy);
  expect(parsed.trace.selectedFrame.repeatCost).toBe(.5*frame.repeatError!);
  expect(parsed.output.countsByScale).toEqual({"64":1});
  expect(parsed.output.countsByCharacter).toEqual({X:1});
  expect(parsed.output.layerVisibility).toEqual({"64":false});
  expect(parsed.trace.frameCount).toBe(result.pursuit!.frames.length);
  for(const key of ["masks","residual","paper","tiles","images","glyphs","frames"])
    expect(payload).not.toContain(`"${key}":`);
  expect(payload.length).toBeLessThan(8000);
});
