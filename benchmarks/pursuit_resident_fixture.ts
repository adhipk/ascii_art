import {mkdir} from "node:fs/promises";
import {createResidentPlan} from "../web/pursuit-resident-plan";
const [input="/tmp/glyph-pursuit-shapes-fixture.json",output="/tmp/pursuit-resident",budget="128"]=Bun.argv.slice(2);
const f=await Bun.file(input).json(),plan=createResidentPlan(new Uint8Array(f.pixels),f.width,f.height,{...f.settings,pursuit_beam:1,pursuit_schedule:"coarse-to-fine",pursuit_budget:Number(budget)});
await mkdir(output,{recursive:true});
await Promise.all([
 Bun.write(`${output}/shader.wgsl`,plan.shader),Bun.write(`${output}/reference.f32`,plan.reference),Bun.write(`${output}/masks.f32`,plan.masks),Bun.write(`${output}/kernels.bin`,plan.metadata),Bun.write(`${output}/params.bin`,plan.parameters()),
 Bun.write(`${output}/exact-reference.json`,JSON.stringify([0,Math.floor(plan.kernels.length/3),Math.floor(plan.kernels.length*2/3),plan.kernels.length-1].map((kernel,i)=>{
  const k=plan.kernels[kernel],x=i%2?plan.setup.width-k.size:0,y=i<2?0:plan.setup.height-k.size;
  return {kernel,x,y,...plan.setup.blurModel.fit(plan.setup.paper,x,y,k.size,k.alpha,f.settings.use_color)};
 }))),
 Bun.write(`${output}/plan.json`,JSON.stringify({width:plan.setup.width,height:plan.setup.height,budget:plan.setup.budget,coarseThreshold:plan.setup.coarseThreshold,coarseRounds:plan.setup.coarseRounds,stages:plan.stages,levels:plan.levels,initialize:plan.initialize,iteration:plan.iteration,flush:plan.flush,refinement:plan.refinement,maxCandidates:plan.maxCandidates,workBytes:plan.workBytes,historyBytes:plan.historyBytes,thumbWidth:plan.thumbWidth,thumbHeight:plan.thumbHeight,thumbWords:plan.thumbWords,patchStride:plan.patchStride,partialStride:plan.partialStride,kernels:plan.kernels.map(k=>({size:k.size,char:k.char,angle:k.angle})),fixture:input})),
]);console.log(JSON.stringify({output,kernels:plan.kernels.length,workBytes:plan.workBytes,historyBytes:plan.historyBytes}));
