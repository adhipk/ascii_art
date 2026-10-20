/** WebGPU coarse candidate scorer. Final fitting and acceptance remain on CPU.
 * API/buffer layout follows https://www.w3.org/TR/webgpu/ and WGSL.
 */
export interface CandidateBatch {
  width:number; height:number; target:Float32Array; pixels:Float32Array;
  guidance?:Float32Array;
  edgeWeight:number; color:boolean; coverage?:Float32Array; maxOverlap?:number; overlapPenalty?:number; penaltyCoverage?:Float32Array; penaltyMinSize?:number;
  kernels:readonly {size:number;alpha:Float32Array}[];
  candidates:readonly {kernel:number;x:number;y:number}[];
}
export interface CandidateScorer { score(batch:CandidateBatch):Promise<Float32Array>; dispose():void; }

// Structural types keep this module independent of optional @webgpu/types.
export interface BufferHandle {
  destroy():void; mapAsync(mode:number,offset?:number,size?:number):Promise<void>;
  getMappedRange(offset?:number,size?:number):ArrayBuffer; unmap():void;
}
export interface DeviceHandle {
  lost:Promise<{message:string}>;
  createBuffer(descriptor:{size:number;usage:number}):BufferHandle;
  createShaderModule(descriptor:{code:string}):{getCompilationInfo():Promise<{messages:{type:string;message:string}[]}>};
  createComputePipelineAsync(descriptor:object):Promise<{getBindGroupLayout(index:number):object}>;
  createBindGroup(descriptor:object):object;
  createCommandEncoder():{
    beginComputePass():{setPipeline(pipeline:object):void;setBindGroup(index:number,group:object):void;dispatchWorkgroups(count:number,y?:number,z?:number):void;end():void};
    copyBufferToBuffer(source:BufferHandle,sourceOffset:number,target:BufferHandle,targetOffset:number,size:number):void;
    finish():object;
  };
  pushErrorScope(filter:string):void;popErrorScope():Promise<{message:string}|null>;
  queue:{writeBuffer(buffer:BufferHandle,offset:number,data:ArrayBufferView):void;submit(commands:object[]):void};
  destroy():void;
}
export interface GPUProvider {
  requestAdapter(options?:object):Promise<{requestDevice():Promise<DeviceHandle>}|null>;
}

export const CANDIDATE_SHADER = /* wgsl */ `
struct Params { width:u32, height:u32, count:u32, color:u32, edge_weight:f32, max_overlap:f32, overlap_penalty:f32, min_size:f32 }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> reference_pixels:array<f32>;
@group(0) @binding(2) var<storage,read> pixels:array<f32>;
@group(0) @binding(3) var<storage,read> masks:array<f32>;
// x, y, size, packed alpha offset; 16-byte stride.
@group(0) @binding(4) var<storage,read> candidates:array<vec4<u32>>;
@group(0) @binding(5) var<storage,read_write> scores:array<f32>;
@group(0) @binding(6) var<storage,read> coverage:array<f32>;
const weights=vec3<f32>(0.299,0.587,0.114);
fn source(i:u32)->vec3<f32> {let j=i*3u;return vec3<f32>(reference_pixels[j],reference_pixels[j+1u],reference_pixels[j+2u]);}
fn current(i:u32)->vec3<f32> {let j=i*3u;return vec3<f32>(pixels[j],pixels[j+1u],pixels[j+2u]);}
fn alpha_at(c:vec4<u32>,x:i32,y:i32)->f32 {
  if(x<i32(c.x)||y<i32(c.y)||x>=i32(c.x+c.z)||y>=i32(c.y+c.z)){return 0.0;}
  return masks[c.w+u32(y-i32(c.y))*c.z+u32(x-i32(c.x))];
}
fn gradient_fit(c:vec4<u32>,x:u32,y:u32,nx:u32,ny:u32)->vec2<f32> {
  let i=y*p.width+x;let j=ny*p.width+nx;
  let a=alpha_at(c,i32(x),i32(y));let b=alpha_at(c,i32(nx),i32(ny));let d=a-b;
  let fixed=(1.0-a)*dot(current(i),weights)-(1.0-b)*dot(current(j),weights);
  return p.edge_weight*vec2<f32>(d*d,d*(dot(source(i)-source(j),weights)-fixed));
}
fn corrected_residual(c:vec4<u32>,x:u32,y:u32,color:vec3<f32>)->f32 {
  let i=y*p.width+x;let a=alpha_at(c,i32(x),i32(y));
  return dot(source(i),weights)-dot(current(i),weights)*(1.0-a)-dot(color,weights)*a;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
  if(id.x>=p.count){return;}
  let c=candidates[id.x];let size=c.z;let step=max(1u,size/10u);
  var denominator=0.0;var numerator=vec3<f32>(0.0);var alpha_mass=0.0;var covered_mass=0.0;var penalty_mass=0.0;
  for(var y=0u;y<size;y+=step){for(var x=0u;x<size;x+=step){
    let a=masks[c.w+y*size+x];let i=(c.y+y)*p.width+c.x+x;
    denominator+=a*a;alpha_mass+=a;covered_mass+=a*coverage[i];penalty_mass+=a*coverage[i+p.width*p.height];numerator+=a*(source(i)-(1.0-a)*current(i));
  }}
  if(alpha_mass>0.0&&covered_mass/alpha_mass>p.max_overlap+1e-7){scores[id.x]=-1.0;return;}
  var color=clamp(numerator/max(denominator,1e-10),vec3<f32>(0.0),vec3<f32>(1.0));
  if(p.color==0u){color=vec3<f32>(dot(color,weights));}
  if(step==1u&&denominator>1e-10){
    var gradient=vec2<f32>(0.0);
    let x0=u32(max(0,i32(c.x)-1));let y0=u32(max(0,i32(c.y)-1));
    for(var y=y0;y<min(p.height,c.y+size);y++){for(var x=x0;x<min(p.width,c.x+size);x++){
      if(x+1u<p.width){gradient+=gradient_fit(c,x,y,x+1u,y);}
      if(y+1u<p.height){gradient+=gradient_fit(c,x,y,x,y+1u);}
    }}
    if(p.color!=0u){
      let raw=numerator/denominator;
      let adjustment=(gradient.y-gradient.x*dot(raw,weights))/(denominator+gradient.x);
      color=clamp(raw+vec3<f32>(adjustment),vec3<f32>(0.0),vec3<f32>(1.0));
    }else{
      let gray=clamp(((numerator.x+numerator.y+numerator.z)/3.0+gradient.y)/(denominator+gradient.x),0.0,1.0);
      color=vec3<f32>(gray);
    }
  }
  var before=0.0;var after=0.0;
  for(var y=0u;y<size;y+=step){for(var x=0u;x<size;x+=step){
    let px=c.x+x;let py=c.y+y;let i=py*p.width+px;
    let a=masks[c.w+y*size+x];let residual=source(i)-current(i);
    let next=residual-a*(color-current(i));
    before+=dot(residual,residual)/3.0;after+=dot(next,next)/3.0;
    let old_luma=dot(residual,weights);let new_luma=corrected_residual(c,px,py,color);
    if(px+1u<p.width){
      let old=old_luma-dot(source(i+1u)-current(i+1u),weights);
      let changed=new_luma-corrected_residual(c,px+1u,py,color);
      before+=p.edge_weight*old*old;after+=p.edge_weight*changed*changed;
    }
    if(py+1u<p.height){
      let old=old_luma-dot(source(i+p.width)-current(i+p.width),weights);
      let changed=new_luma-corrected_residual(c,px,py+1u,color);
      before+=p.edge_weight*old*old;after+=p.edge_weight*changed*changed;
    }
  }}
  if(f32(size)>=p.min_size){after+=p.overlap_penalty*penalty_mass;}
  scores[id.x]=(before-after)*f32(step*step)/f32(p.width*p.height);
}
`;

export function packCandidates(batch:CandidateBatch, offsets:Uint32Array):Uint32Array {
  const packed=new Uint32Array(batch.candidates.length*4);
  batch.candidates.forEach((candidate,i)=>{
    const kernel=batch.kernels[candidate.kernel];
    if(!kernel || !Number.isInteger(candidate.x) || !Number.isInteger(candidate.y) || candidate.x<0 || candidate.y<0 || candidate.x+kernel.size>batch.width || candidate.y+kernel.size>batch.height)
      throw new Error("Invalid GPU candidate bounds");
    packed.set([candidate.x,candidate.y,kernel.size,offsets[candidate.kernel]],i*4);
  });
  return packed;
}

class WebGPUScorer implements CandidateScorer {
  private resources=new Set<BufferHandle>();
  private target?:Float32Array;
  private coverageBuffer?:BufferHandle;private targetBuffer?:BufferHandle;private pixelBuffer?:BufferHandle;private maskBuffer?:BufferHandle;private parameterBuffer?:BufferHandle;
  private candidateBuffer?:BufferHandle;private scoreBuffer?:BufferHandle;private readback?:BufferHandle;
  private capacity=0;private offsets=new Uint32Array();private loss?:string;private disposed=false;
  constructor(private device:DeviceHandle,private pipeline:{getBindGroupLayout(index:number):object}) {
    void device.lost.then(info=>{this.loss=info.message||"GPU device lost";});
  }
  private buffer(size:number,usage:number):BufferHandle {
    const buffer=this.device.createBuffer({size:Math.max(4,Math.ceil(size/4)*4),usage});
    this.resources.add(buffer);return buffer;
  }
  async score(batch:CandidateBatch):Promise<Float32Array> {
    if(this.disposed || this.loss) throw new Error(this.loss??"GPU scorer disposed");
    if(!batch.candidates.length) return new Float32Array();
    const {device}=this;
    device.pushErrorScope("validation");
    let scopeOpen=true;
    try {
      if(this.target!==batch.target) {
        if(this.target) throw new Error("A GPU scorer belongs to one optimization run");
        this.target=batch.target;
        this.targetBuffer=this.buffer(batch.target.byteLength,128|8);
        this.pixelBuffer=this.buffer(batch.pixels.byteLength,128|8);
        this.coverageBuffer=this.buffer(batch.width*batch.height*8,128|8);
        let length=0;this.offsets=new Uint32Array(batch.kernels.length);
        batch.kernels.forEach((kernel,i)=>{this.offsets[i]=length;length+=kernel.alpha.length;});
        const masks=new Float32Array(length);
        batch.kernels.forEach((kernel,i)=>masks.set(kernel.alpha,this.offsets[i]));
        this.maskBuffer=this.buffer(masks.byteLength,128|8);
        this.parameterBuffer=this.buffer(32,64|8);
        device.queue.writeBuffer(this.targetBuffer,0,batch.target);
        device.queue.writeBuffer(this.maskBuffer,0,masks);
      }
      if(batch.candidates.length>this.capacity) {
        for(const buffer of [this.candidateBuffer,this.scoreBuffer,this.readback]) if(buffer){buffer.destroy();this.resources.delete(buffer);}
        this.capacity=2**Math.ceil(Math.log2(batch.candidates.length));
        this.candidateBuffer=this.buffer(this.capacity*16,128|8);
        this.scoreBuffer=this.buffer(this.capacity*4,128|4);
        this.readback=this.buffer(this.capacity*4,1|8);
      }
      const packed=packCandidates(batch,this.offsets), parameters=new ArrayBuffer(32);
      new Uint32Array(parameters).set([batch.width,batch.height,batch.candidates.length,Number(batch.color)]);
      new Float32Array(parameters)[4]=batch.edgeWeight;
      new Float32Array(parameters)[5]=batch.maxOverlap??1;
      new Float32Array(parameters)[6]=batch.overlapPenalty??0;
      new Float32Array(parameters)[7]=batch.penaltyMinSize??0;
      device.queue.writeBuffer(this.parameterBuffer!,0,new Uint8Array(parameters));
      device.queue.writeBuffer(this.targetBuffer!,0,batch.guidance??batch.target);
      device.queue.writeBuffer(this.pixelBuffer!,0,batch.pixels);
      const coverage=batch.coverage??new Float32Array(batch.width*batch.height);
      device.queue.writeBuffer(this.coverageBuffer!,0,coverage);
      device.queue.writeBuffer(this.coverageBuffer!,coverage.byteLength,batch.penaltyCoverage??coverage);
      device.queue.writeBuffer(this.candidateBuffer!,0,packed);
      const bindGroup=device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:
        [this.parameterBuffer,this.targetBuffer,this.pixelBuffer,this.maskBuffer,this.candidateBuffer,this.scoreBuffer,this.coverageBuffer].map((buffer,binding)=>({binding,resource:{buffer}}))});
      const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
      pass.setPipeline(this.pipeline);pass.setBindGroup(0,bindGroup);pass.dispatchWorkgroups(Math.ceil(batch.candidates.length/64));pass.end();
      const bytes=batch.candidates.length*4;
      encoder.copyBufferToBuffer(this.scoreBuffer!,0,this.readback!,0,bytes);
      device.queue.submit([encoder.finish()]);
      const validation=await device.popErrorScope();scopeOpen=false;
      if(validation) throw new Error(validation.message);
      await this.readback!.mapAsync(1,0,bytes);
      if(this.loss) throw new Error(this.loss);
      const scores=new Float32Array(this.readback!.getMappedRange(0,bytes).slice(0));
      this.readback!.unmap();return scores;
    } finally {if(scopeOpen) await device.popErrorScope();}
  }
  dispose():void {
    if(this.disposed)return;this.disposed=true;
    for(const buffer of this.resources) buffer.destroy();
    this.resources.clear();this.device.destroy();
  }
}

export async function createWebGPUScorer(provider?:GPUProvider):Promise<CandidateScorer> {
  const gpu=provider??(globalThis.navigator as Navigator & {gpu?:GPUProvider}|undefined)?.gpu;
  if(!gpu) throw new Error("WebGPU is unavailable in this browser");
  const adapter=await gpu.requestAdapter({powerPreference:"high-performance"});
  if(!adapter) throw new Error("No WebGPU adapter is available");
  const device=await adapter.requestDevice();
  try {
    const module=device.createShaderModule({code:CANDIDATE_SHADER});
    const info=await module.getCompilationInfo();
    const errors=info.messages.filter(message=>message.type==="error");
    if(errors.length) throw new Error("GPU shader compilation failed: "+errors.map(error=>error.message).join("; "));
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    return new WebGPUScorer(device,pipeline);
  } catch(error) {device.destroy();throw error;}
}
