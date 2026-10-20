/** Capture a real search batch for native WGSL parity/throughput measurements. */
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { renderPursuitAsync, referenceCandidateScores } from "../web/pursuit";
import { CANDIDATE_SHADER, packCandidates } from "../web/pursuit-gpu";

const [input="/tmp/glyph-pursuit-shapes-fixture.json", output="/tmp/glyph-gpu-benchmark", profile="scale-aware", captureArgument="0"] = Bun.argv.slice(2);
await mkdir(output,{recursive:true});
const fixture=await Bun.file(input).json();
let saved=false,seen=0;
const captureAt=Number(captureArgument);
if(!Number.isInteger(captureAt)||captureAt<0||captureAt>64)throw new Error("Invalid batch index");
await renderPursuitAsync(new Uint8Array(fixture.pixels),fixture.width,fixture.height,
  {...fixture.settings,pursuit_budget:captureAt+1,pursuit_beam:1,pursuit_backend:"webgpu",pursuit_vocabulary:profile as "scale-aware"|"full"},()=>{},async()=>({
    async score(batch) {
      const start=performance.now(),reference=referenceCandidateScores(batch),referenceMs=performance.now()-start;
      if(!saved && seen++===captureAt) {
        saved=true;
        const cpuSamples=[referenceMs];
        for(let repeat=0;repeat<5;++repeat){const start=performance.now();referenceCandidateScores(batch);cpuSamples.push(performance.now()-start);}
        const warm=cpuSamples.slice(1).sort((a,b)=>a-b),referenceWarmMs=warm[Math.floor(warm.length/2)];
        const kernelCounts=Object.fromEntries([...new Set(batch.kernels.map(k=>k.size))].map(size=>[String(size),batch.kernels.filter(k=>k.size===size).length]));
        let length=0;const offsets=new Uint32Array(batch.kernels.length);
        batch.kernels.forEach((kernel,i)=>{offsets[i]=length;length+=kernel.alpha.length;});
        const masks=new Float32Array(length);batch.kernels.forEach((kernel,i)=>masks.set(kernel.alpha,offsets[i]));
        const parameters=new ArrayBuffer(32);
        new Uint32Array(parameters).set([batch.width,batch.height,batch.candidates.length,Number(batch.color)]);
        new Float32Array(parameters)[4]=batch.edgeWeight;
        new Float32Array(parameters)[5]=batch.maxOverlap??1;
        await Promise.all([
          Bun.write(resolve(output,"source.f32"),batch.guidance??batch.target),Bun.write(resolve(output,"pixels.f32"),batch.pixels),
          Bun.write(resolve(output,"coverage.f32"),batch.coverage??new Float32Array(batch.width*batch.height)),Bun.write(resolve(output,"masks.f32"),masks),Bun.write(resolve(output,"candidates.u32"),packCandidates(batch,offsets)),
          Bun.write(resolve(output,"params.bin"),parameters),Bun.write(resolve(output,"reference.f64"),reference),
          Bun.write(resolve(output,"shader.wgsl"),CANDIDATE_SHADER),
          Bun.write(resolve(output,"metadata.json"),JSON.stringify({width:batch.width,height:batch.height,candidates:batch.candidates.length,kernels:batch.kernels.length,kernelCounts,profile,captureAt,referenceMs,referenceWarmMs,referenceSamplesMs:cpuSamples,maxOverlap:batch.maxOverlap??1,cpuRuntime:`Bun ${Bun.version}`,guidance:!!batch.guidance,color:batch.color,edgeWeight:batch.edgeWeight})),
        ]);
        console.log(JSON.stringify({output,candidates:batch.candidates.length,referenceMs,referenceWarmMs,kernelCounts}));
      }
      return Float32Array.from(reference);
    },dispose(){},
  }));
