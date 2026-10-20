import {expect, test} from "bun:test";
import {mkdtemp, rm, readdir} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {generateProcessVideo} from "./generate-process-video";

const metrics={width:8,height:8,baseline:7,advance:8,densities:[],kernelCharacters:" X",masks:[Array(64).fill(0),Array(64).fill(1)]};
function input(painted=false){
 const pixels=new Uint8Array(32*32*4);
 for(let y=0;y<32;y++)for(let x=0;x<32;x++){const i=(y*32+x)*4;pixels[i+3]=255;if(painted&&x>=8&&x<24&&y>=8&&y<24)pixels.fill(255,i,i+3);}
 return {pixels,width:32,height:32};
}
const settings={background_mode:"solid" as const,background:"#000000",pursuit_resolution:128 as const,pursuit_scales:[16,8,4],pursuit_blur:0,pursuit_edge_weight:0,pursuit_coarse_rounds:1};

test("encodes converged paper-only input as a valid MP4",async()=>{
 const dir=await mkdtemp(join(tmpdir(),"video-test-"));
 try{
  const result=await generateProcessVideo(input(),{outputPath:join(dir,"result.mp4"),backend:"cpu",metrics,settings,width:64,fps:8,initialHoldSeconds:0,finalHoldSeconds:0});
  expect(result.converged).toBe(true);expect(result.iterations).toBe(0);expect(result.frames).toBe(1);
  const probe=Bun.spawn(["ffprobe","-v","error","-show_entries","stream=width,height,nb_frames","-of","json",result.path],{stdout:"pipe"});
  const stream=(await new Response(probe.stdout).json()).streams[0];expect(await probe.exited).toBe(0);
  expect(stream.width).toBe(64);expect(stream.height).toBe(64);expect(stream.nb_frames).toBe("1");
 }finally{await rm(dir,{recursive:true,force:true});}
});

test("budget exhaustion cannot be labeled convergence; capped previews remain available",async()=>{
 const dir=await mkdtemp(join(tmpdir(),"video-test-"));
 const params={outputPath:join(dir,"result.mp4"),backend:"cpu" as const,metrics,settings,width:64,maxIterations:1,initialHoldSeconds:0,finalHoldSeconds:0};
 try{
  await expect(generateProcessVideo(input(true),params)).rejects.toThrow("before convergence");
  expect(await readdir(dir)).toEqual([]);
  const result=await generateProcessVideo(input(true),{...params,requireConvergence:false});
  expect(result.converged).toBe(false);expect(result.iterations).toBe(1);expect(result.frames).toBe(2);
  const decode=Bun.spawn(["ffmpeg","-v","error","-i",result.path,"-f","rawvideo","-pix_fmt","rgb24","pipe:1"],{stdout:"pipe",stderr:"pipe"});
  const bytes=new Uint8Array(await new Response(decode.stdout).arrayBuffer());expect(await decode.exited).toBe(0);
  expect(bytes.length).toBe(64*64*3*2);expect(bytes.slice(64*64*3).some(v=>v>100)).toBe(true);
  expect(bytes.slice(0,64*64*3).every(v=>v<10)).toBe(true);
 }finally{await rm(dir,{recursive:true,force:true});}
});

test("invalid options and cancellation fail before rendering",async()=>{
 await expect(generateProcessVideo(input(),{outputPath:"/tmp/unused.mp4",width:65})).rejects.toThrow("Invalid");
 const controller=new AbortController();controller.abort();
 await expect(generateProcessVideo(input(),{outputPath:"/tmp/unused.mp4",signal:controller.signal})).rejects.toThrow("cancelled");
});
