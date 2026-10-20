import { CANDIDATE_SHADER } from "./pursuit-gpu";

/** One program, persistent image/mask/history buffers. Entry points are ordered
 * by the host; no CPU scoring or painting occurs between iterations. */
export function residentShader(patchStride:number,partialStride:number):string {
  let rank=CANDIDATE_SHADER.slice(CANDIDATE_SHADER.indexOf("const weights="));
  rank=rank.replaceAll("coverage[i+p.width*p.height]","coverage[i]").replaceAll("p.min_size","0.0").replace(/fn source\(i:u32\)->vec3<f32> \{[^\n]+\}/,"fn source(i:u32)->vec3<f32> {return w.data[p.dims.x*p.dims.y+i].xyz;}")
    .replace(/fn current\(i:u32\)->vec3<f32> \{[^\n]+\}/,"fn current(i:u32)->vec3<f32> {return image[i].xyz;}")
    .replaceAll("p.width","p.dims.x").replaceAll("p.height","p.dims.y").replaceAll("p.edge_weight","p.numbers.x")
    .replaceAll("coverage[i]","w.data[i].x").replaceAll("p.max_overlap","p.numbers.y").replaceAll("p.overlap_penalty","p.numbers.w").replaceAll("p.color","p.control.w")
    .replace("fn main(","fn rank_candidates(")
    .replace("if(id.x>=p.count){return;}","if(w.state.x!=0u || id.x>=candidate_count()){return;}")
    .replace("let c=candidates[id.x];let size=c.z;",`let spec=make_candidate(id.x);candidates[id.x].pos=spec;
      let k=kernels[spec.z];let c=vec4<u32>(spec.xy,k.info.x,k.info.y);let size=c.z;`)
    .replace("scores[id.x]=-1.0;","candidates[id.x].color=vec4<f32>(0.0,0.0,0.0,-1.0);")
    .replace("scores[id.x]=(before-after)*f32(step*step)/f32(p.dims.x*p.dims.y);","candidates[id.x].color=vec4<f32>(color,(before-after)*f32(step*step)/f32(p.dims.x*p.dims.y));");
  return /* wgsl */ `
struct Level { dims:vec4<u32>, config:vec4<f32> }
struct Params {
 dims:vec4<u32>, options:vec4<u32>, control:vec4<u32>, numbers:vec4<f32>, paper:vec4<f32>, limits:vec4<u32>,
 levels:array<Level,6>, stages:array<vec4<u32>,8>
}
struct Kernel {info:vec4<u32>,ink:vec4<f32>}
struct Candidate {pos:vec4<u32>,color:vec4<f32>}
struct Event {pos:vec4<u32>,info:vec4<u32>,color:vec4<f32>,errors:vec4<f32>}
struct Work {
 state:vec4<u32>, counts:vec4<u32>, pending:vec4<u32>, stats:vec4<f32>, saved:vec4<f32>, best:Candidate,
 anchors:array<vec4<f32>,6>, slots:array<Candidate,16>, tiles:array<vec4<f32>,1600>, partial:array<vec4<f32>,2048>, data:array<vec4<f32>>
}
struct History {glyphs:array<Candidate,2048>,events:array<Event,4096>,thumbs:array<u32>}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read_write> reference_image:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read_write> image:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> masks:array<f32>;
@group(0) @binding(4) var<storage,read> kernels:array<Kernel>;
@group(0) @binding(5) var<storage,read_write> candidates:array<Candidate>;
@group(0) @binding(6) var<storage,read_write> w:Work;
@group(0) @binding(7) var<storage,read_write> h:History;
const PATCH_STRIDE=${patchStride}u;
const PARTIAL_STRIDE=${partialStride}u;
fn stage()->vec4<u32>{return p.stages[p.options.y];}
fn candidate_count()->u32{return stage().y*w.state.y*5u;}
fn rgb_luma(v:vec3<f32>)->f32{return dot(v,vec3<f32>(.299,.587,.114));}
fn residual(level:u32,x:u32,y:u32)->vec3<f32>{let l=p.levels[level];let i=l.dims.x+y*l.dims.y+x;return reference_image[i].xyz-image[i].xyz;}
fn mask_at(k:u32,x:i32,y:i32)->f32{let info=kernels[k].info;if(x<0||y<0||x>=i32(info.x)||y>=i32(info.x)){return 0.0;}return masks[info.y+u32(y)*info.x+u32(x)];}
fn make_candidate(index:u32)->vec4<u32>{
 let anchor_count=w.state.y;let k=stage().x+index/(anchor_count*5u);let a=(index/5u)%anchor_count;let offset=index%5u;
 var delta=vec2<f32>(0.0);let size=kernels[k].info.x;
 if(offset==1u){delta.x=-.25;}if(offset==2u){delta.x=.25;}if(offset==3u){delta.y=-.25;}if(offset==4u){delta.y=.25;}
 let origin=clamp(round(w.anchors[a].xy-kernels[k].ink.xy+delta*f32(size)),vec2<f32>(0.0),vec2<f32>(p.dims.xy-vec2<u32>(size)));
 return vec4<u32>(vec2<u32>(origin),k,0u);
}
fn rect(c:Candidate,level:u32)->vec4<u32>{
 var r=vec4<u32>(c.pos.xy,vec2<u32>(kernels[c.pos.z].info.x));
 for(var l=1u;l<=level;l++){
  let begin=vec2<u32>(max(vec2<i32>(0),vec2<i32>(r.xy)/2));
  let end=min(p.levels[l].dims.yz,(r.xy+r.zw)/2u+vec2<u32>(1u));
  r=vec4<u32>(begin,end-begin);
 }
 return r;
}
fn patch_index(slot:u32,level:u32,x:u32,y:u32)->u32{return 2u*p.dims.x*p.dims.y+slot*PATCH_STRIDE+p.levels[level].dims.w+y*p.limits.x+x;}
fn patch_at(slot:u32,level:u32,x:i32,y:i32)->vec4<f32>{
 let r=rect(w.slots[slot],level);
 if(x<i32(r.x)||y<i32(r.y)||x>=i32(r.x+r.z)||y>=i32(r.y+r.w)){return vec4<f32>(0.0);}
 return w.data[patch_index(slot,level,u32(x)-r.x,u32(y)-r.y)];
}
fn sample_count(c:Candidate)->u32{var n=0u;for(var l=0u;l<p.options.x;l++){let r=rect(c,l);let begin=max(vec2<i32>(0),vec2<i32>(r.xy)-vec2<i32>(1));let end=min(p.levels[l].dims.yz,r.xy+r.zw);n+=(end.x-u32(begin.x))*(end.y-u32(begin.y));}return n;}
fn sample_location(c:Candidate,index:u32)->vec3<u32>{
 var i=index;
 for(var l=0u;l<p.options.x;l++){
  let r=rect(c,l);let begin=vec2<u32>(max(vec2<i32>(0),vec2<i32>(r.xy)-vec2<i32>(1)));let end=min(p.levels[l].dims.yz,r.xy+r.zw);let width=end.x-begin.x;let n=width*(end.y-begin.y);
  if(i<n){return vec3<u32>(begin+vec2<u32>(i%width,i/width),l);}i-=n;
 }
 return vec3<u32>(0u,0u,99u);
}
@compute @workgroup_size(128) fn initialize(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;if(i<p.dims.x*p.dims.y){image[i]=p.paper;w.data[i]=vec4<f32>(0.0);}
 if(i==0u){w.pending=vec4<u32>(3u,0u,0u,0u);w.best.color.w=-1.0;}
}
fn reduce_level(level:u32,index:u32,is_reference:bool){
 let dest=p.levels[level];if(index>=dest.dims.y*dest.dims.z){return;}
 let src=p.levels[level-1u];let xy=vec2<i32>(i32(index%dest.dims.y)*2,i32(index/dest.dims.y)*2);var value=vec4<f32>(0.0);
 for(var dy=-1;dy<=1;dy++){for(var dx=-1;dx<=1;dx++){
  let pt=vec2<u32>(clamp(xy+vec2<i32>(dx,dy),vec2<i32>(0),vec2<i32>(src.dims.yz)-vec2<i32>(1)));let j=src.dims.x+pt.y*src.dims.y+pt.x;
  let weight=f32(select(1,2,dx==0)*select(1,2,dy==0))/16.0;
  if(is_reference){value+=weight*reference_image[j];}else{value+=weight*image[j];}
 }}
 if(is_reference){reference_image[dest.dims.x+index]=value;}else{image[dest.dims.x+index]=value;}
}
${Array.from({length:5},(_,i)=>`@compute @workgroup_size(128) fn reference_${i+1}(@builtin(global_invocation_id) id:vec3<u32>){reduce_level(${i+1}u,id.x,true);}\n@compute @workgroup_size(128) fn image_${i+1}(@builtin(global_invocation_id) id:vec3<u32>){reduce_level(${i+1}u,id.x,false);}`).join('\n')}
@compute @workgroup_size(128) fn guidance(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;if(i>=p.dims.x*p.dims.y){return;}let xy=vec2<u32>(i%p.dims.x,i/p.dims.x);var value=image[i].xyz;
 for(var l=0u;l<p.options.x;l++){let level=p.levels[l];value+=level.config.x*residual(l,min(xy.x>>l,level.dims.y-1u),min(xy.y>>l,level.dims.z-1u));}
 w.data[p.dims.x*p.dims.y+i]=vec4<f32>(value,0.0);
}
var<workgroup> sum_a:array<vec4<f32>,128>;
var<workgroup> sum_b:array<vec4<f32>,128>;
@compute @workgroup_size(128) fn statistics(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>){
 var a=vec4<f32>(0.0);var index=id.x;
 for(var l=0u;l<p.options.x;l++){
  let level=p.levels[l];let n=level.dims.y*level.dims.z;
  if(index<n){let x=index%level.dims.y;let y=index/level.dims.y;let r=residual(l,x,y);let pix=dot(r,r)/3.0;var edge=0.0;
   if(x+1u<level.dims.y){let d=rgb_luma(r-residual(l,x+1u,y));edge+=d*d;}if(y+1u<level.dims.z){let d=rgb_luma(r-residual(l,x,y+1u));edge+=d*d;}
   a.z=level.config.x*(pix+p.numbers.x*edge)/f32(n);if(l==0u){a.x=pix/f32(n);a.y=edge/f32(n);a.w=w.data[index].y/f32(n);a.z+=p.numbers.w*a.w;}break;
  }index-=n;
 }
 sum_a[lane]=a;workgroupBarrier();
 for(var stride=64u;stride>0u;stride/=2u){if(lane<stride){sum_a[lane]+=sum_a[lane+stride];}workgroupBarrier();}
 if(lane==0u){w.partial[group.x]=sum_a[0];}
}
@compute @workgroup_size(1) fn statistics_sum(){var total=vec4<f32>(0.0);for(var i=0u;i<p.limits.y;i++){total+=w.partial[i];}w.stats=total;}
@compute @workgroup_size(128) fn thumbnails(@builtin(global_invocation_id) id:vec3<u32>){
 if(w.pending.x==0u){return;}let words=(p.control.z*p.limits.z+3u)/4u;if(id.x>=words){return;}var packed=0u;
 for(var part=0u;part<4u;part++){
  let i=id.x*4u+part;if(i>=p.control.z*p.limits.z){break;}let x=i%p.control.z;let y=i/p.control.z;var signed=0.0;
  let begin=vec2<u32>(x*p.dims.x/p.control.z,y*p.dims.y/p.limits.z);let end=vec2<u32>(((x+1u)*p.dims.x+p.control.z-1u)/p.control.z,((y+1u)*p.dims.y+p.limits.z-1u)/p.limits.z);
  for(var sy=begin.y;sy<end.y;sy++){for(var sx=begin.x;sx<end.x;sx++){let r=rgb_luma(residual(0u,sx,sy));if(abs(r)>abs(signed)){signed=r;}}}
  packed|=u32(clamp(round(128.0+127.0*signed),0.0,255.0))<<(part*8u);
 }
 h.thumbs[w.counts.y*words+id.x]=packed;
}
@compute @workgroup_size(1) fn record_event(){
 if(w.pending.x==0u){return;}if(w.counts.y>=4096u){w.state.x=3u;return;}
 let i=w.counts.y;var c=w.best;if(w.pending.x==3u){c.pos=vec4<u32>(0u);c.color=vec4<f32>(0.0);}
 h.events[i].pos=vec4<u32>(c.pos.xyz,w.pending.x);h.events[i].info=vec4<u32>(w.pending.yzw,w.counts.z);
 h.events[i].color=vec4<f32>(c.color.xyz,w.stats.z);h.events[i].errors=vec4<f32>(w.stats.xy,c.color.w,w.stats.w);
 w.counts.y++;w.pending.x=0u;
}
@compute @workgroup_size(64) fn hotspots(@builtin(global_invocation_id) id:vec3<u32>){
 if(w.state.x!=0u||p.options.w==1u){return;}let columns=(p.dims.x+7u)/8u;let rows=(p.dims.y+7u)/8u;if(id.x>=columns*rows){return;}
 let origin=vec2<u32>(id.x%columns,id.x/columns)*8u;var mass=0.0;var center=vec2<f32>(0.0);
 for(var y=origin.y;y<min(origin.y+8u,p.dims.y);y++){for(var x=origin.x;x<min(origin.x+8u,p.dims.x);x++){
  let i=y*p.dims.x+x;let r=w.data[p.dims.x*p.dims.y+i].xyz-image[i].xyz;var error=dot(r,r);
  if(x+1u<p.dims.x){let d=rgb_luma(r-w.data[p.dims.x*p.dims.y+i+1u].xyz+image[i+1u].xyz);error+=p.numbers.x*d*d;}
  if(y+1u<p.dims.y){let d=rgb_luma(r-w.data[p.dims.x*p.dims.y+i+p.dims.x].xyz+image[i+p.dims.x].xyz);error+=p.numbers.x*d*d;}
  mass+=error;center+=(vec2<f32>(f32(x),f32(y))+.5)*error;
 }}
 w.tiles[id.x]=vec4<f32>(select(vec2<f32>(origin)+4.0,center/max(mass,1e-20),mass>0.0),mass,0.0);
}
@compute @workgroup_size(1) fn choose_hotspots(){
 if(w.state.x!=0u){return;}w.best.color.w=-1.0;
 if(p.options.w==1u){let old=h.glyphs[u32(w.saved.y)];let k=kernels[old.pos.z];w.anchors[0]=vec4<f32>(vec2<f32>(old.pos.xy)+k.ink.xy,1.0,0.0);w.state.y=1u;return;}
 let count=((p.dims.x+7u)/8u)*((p.dims.y+7u)/8u);let wanted=select(3u,6u,p.control.y==1u);let skip=p.options.z*6u;w.state.y=0u;
 for(var rank_index=0u;rank_index<skip+wanted;rank_index++){
  var best=-1.0;var index=0u;for(var i=0u;i<count;i++){if(w.tiles[i].z>best){best=w.tiles[i].z;index=i;}}
  if(best<0.0){break;}if(rank_index>=skip){w.anchors[w.state.y]=w.tiles[index];w.state.y++;}w.tiles[index].z=-1.0;
 }
}
${rank}
@compute @workgroup_size(1) fn shortlist(){
 if(w.state.x!=0u){w.counts.w=0u;return;}let quota=select(4u,12u,p.control.y==1u);w.counts.w=0u;
 for(var i=0u;i<candidate_count();i++){
  let c=candidates[i];if(c.color.w<=1e-10){continue;}
  var duplicate=false;for(var j=0u;j<w.counts.w;j++){if(all(w.slots[j].pos.xyz==c.pos.xyz)){duplicate=true;}}
  if(duplicate){continue;}var place=w.counts.w;
  if(place==quota&&c.color.w<=w.slots[quota-1u].color.w){continue;}place=min(place,quota-1u);
  loop {if(place==0u||w.slots[place-1u].color.w>=c.color.w){break;}if(place<quota){w.slots[place]=w.slots[place-1u];}place--;}
  w.slots[place]=c;w.counts.w=min(quota,w.counts.w+1u);
 }
 w.counts.z+=candidate_count();
}
fn make_patch(level:u32,index:u32,slot:u32){
 if(w.state.x!=0u||slot>=w.counts.w){return;}let c=w.slots[slot];let r=rect(c,level);let x=index%p.limits.x;let y=index/p.limits.x;if(x>=r.z||y>=r.w){return;}
 var value=vec4<f32>(0.0);
 if(level==0u){let a=mask_at(c.pos.z,i32(x),i32(y));let old=image[(r.y+y)*p.dims.x+r.x+x].xyz;value=vec4<f32>(a,a*old);}
 else{let src=p.levels[level-1u];let xy=vec2<i32>(r.xy+vec2<u32>(x,y))*2;
  for(var dy=-1;dy<=1;dy++){for(var dx=-1;dx<=1;dx++){
   let pt=clamp(xy+vec2<i32>(dx,dy),vec2<i32>(0),vec2<i32>(src.dims.yz)-vec2<i32>(1));value+=f32(select(1,2,dx==0)*select(1,2,dy==0))/16.0*patch_at(slot,level-1u,pt.x,pt.y);
  }}
 }
 w.data[patch_index(slot,level,x,y)]=value;
}
${Array.from({length:6},(_,i)=>`@compute @workgroup_size(128) fn patch_${i}(@builtin(global_invocation_id) id:vec3<u32>){make_patch(${i}u,id.x,id.y);}`).join('\n')}
fn parts_offset(slot:u32,group:u32,row:u32)->u32{return 2u*p.dims.x*p.dims.y+16u*PATCH_STRIDE+(slot*PARTIAL_STRIDE+group)*2u+row;}
@compute @workgroup_size(128) fn fit_parts(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>){
 let slot=group.y;var a=vec4<f32>(0.0);var b=vec4<f32>(0.0);
 if(w.state.x==0u&&slot<w.counts.w){let pt=sample_location(w.slots[slot],id.x);let l=pt.z;
  if(l<p.options.x){let level=p.levels[l];let weight=level.config.x/f32(level.dims.y*level.dims.z);let q=patch_at(slot,l,i32(pt.x),i32(pt.y));let r=residual(l,pt.x,pt.y);
   a=vec4<f32>(weight*q.x*(r+q.yzw)/3.0,weight*q.x*q.x/3.0);
   if(l==0u){b.z=q.x;b.w=q.x*w.data[pt.y*p.dims.x+pt.x].x;}
   for(var axis=0u;axis<2u;axis++){
    let next=pt.xy+select(vec2<u32>(1u,0u),vec2<u32>(0u,1u),axis==1u);
    if(next.x<level.dims.y&&next.y<level.dims.z){let t=patch_at(slot,l,i32(next.x),i32(next.y));let da=q.x-t.x;let dr=rgb_luma(r-residual(l,next.x,next.y)+q.yzw-t.yzw);b.x+=weight*p.numbers.x*da*da;b.y+=weight*p.numbers.x*da*dr;}
   }
  }
 }
 sum_a[lane]=a;sum_b[lane]=b;workgroupBarrier();for(var stride=64u;stride>0u;stride/=2u){if(lane<stride){sum_a[lane]+=sum_a[lane+stride];sum_b[lane]+=sum_b[lane+stride];}workgroupBarrier();}
 if(lane==0u&&slot<16u){w.data[parts_offset(slot,group.x,0u)]=sum_a[0];w.data[parts_offset(slot,group.x,1u)]=sum_b[0];}
}
@compute @workgroup_size(1) fn fit_colors(@builtin(global_invocation_id) id:vec3<u32>){
 let slot=id.x;if(w.state.x!=0u||slot>=w.counts.w){return;}var a=vec4<f32>(0.0);var b=vec4<f32>(0.0);
 for(var i=0u;i<PARTIAL_STRIDE;i++){a+=w.data[parts_offset(slot,i,0u)];b+=w.data[parts_offset(slot,i,1u)];}
 if(b.z>0.0&&b.w/b.z>p.numbers.y+1e-7){w.slots[slot].color=vec4<f32>(0.0,0.0,0.0,-1.0);return;}
 var color=a.xyz/max(a.w,1e-20);
 if(p.control.w==1u){let adjustment=(b.y-b.x*rgb_luma(color))/max(3.0*a.w+b.x,1e-20);color=clamp(color+vec3<f32>(adjustment),vec3<f32>(0.0),vec3<f32>(1.0));}
 else{let gray=clamp((a.x+a.y+a.z+b.y)/max(3.0*a.w+b.x,1e-20),0.0,1.0);color=vec3<f32>(gray);}
 w.slots[slot].color=vec4<f32>(color,0.0);
}
@compute @workgroup_size(128) fn gain_parts(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>){
 let slot=group.y;var gain=0.0;
 if(w.state.x==0u&&slot<w.counts.w&&w.slots[slot].color.w>=0.0){let pt=sample_location(w.slots[slot],id.x);let l=pt.z;
  if(l<p.options.x){let level=p.levels[l];let weight=level.config.x/f32(level.dims.y*level.dims.z);let q=patch_at(slot,l,i32(pt.x),i32(pt.y));let r=residual(l,pt.x,pt.y);let color=w.slots[slot].color.xyz;let delta=q.x*color-q.yzw;
   gain=weight*(2.0*dot(r,delta)-dot(delta,delta))/3.0;
   if(l==0u){gain-=p.numbers.w*q.x*w.data[pt.y*p.dims.x+pt.x].x/f32(p.dims.x*p.dims.y);}
   for(var axis=0u;axis<2u;axis++){let next=pt.xy+select(vec2<u32>(1u,0u),vec2<u32>(0u,1u),axis==1u);
    if(next.x<level.dims.y&&next.y<level.dims.z){let t=patch_at(slot,l,i32(next.x),i32(next.y));let dr=rgb_luma(r-residual(l,next.x,next.y));let d=rgb_luma(delta-t.x*color+t.yzw);gain+=weight*p.numbers.x*(2.0*dr*d-d*d);}
   }
  }
 }
 sum_a[lane]=vec4<f32>(gain,0.0,0.0,0.0);workgroupBarrier();for(var stride=64u;stride>0u;stride/=2u){if(lane<stride){sum_a[lane]+=sum_a[lane+stride];}workgroupBarrier();}
 if(lane==0u&&slot<16u){w.data[parts_offset(slot,group.x,0u)]=sum_a[0];}
}
@compute @workgroup_size(1) fn gain_sum(@builtin(global_invocation_id) id:vec3<u32>){
 let slot=id.x;if(w.state.x!=0u||slot>=w.counts.w||w.slots[slot].color.w<0.0){return;}var gain=0.0;for(var i=0u;i<PARTIAL_STRIDE;i++){gain+=w.data[parts_offset(slot,i,0u)].x;}w.slots[slot].color.w=gain;
}
@compute @workgroup_size(1) fn choose_best(){
 if(w.state.x!=0u){return;}for(var i=0u;i<w.counts.w;i++){if(w.slots[i].color.w>w.best.color.w+1e-9){w.best=w.slots[i];}}w.counts.z+=w.counts.w;
}
fn local_slots(small:bool){
 if(w.state.x!=0u||w.best.color.w<=1e-9){w.counts.w=0u;return;}let best=w.best;let size=kernels[best.pos.z].info.x;let step=select(max(1u,u32(round(f32(size)/16.0))),1u,small);w.counts.w=0u;
 for(var dy=-1;dy<=1;dy++){for(var dx=-1;dx<=1;dx++){
  if(dx==0&&dy==0){continue;}if(!small&&dx!=0&&dy!=0){continue;}let pt=vec2<i32>(best.pos.xy)+vec2<i32>(dx,dy)*i32(step);
  if(any(pt<vec2<i32>(0))||any(pt+vec2<i32>(i32(size))>vec2<i32>(p.dims.xy))){continue;}var c=best;c.pos.x=u32(pt.x);c.pos.y=u32(pt.y);w.slots[w.counts.w]=c;w.counts.w++;
 }}
}
@compute @workgroup_size(1) fn local_large(){local_slots(false);}
@compute @workgroup_size(1) fn local_small(){local_slots(true);}
@compute @workgroup_size(128) fn paint(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;if(p.options.w!=0u||w.state.x!=0u||w.best.color.w<=p.numbers.z||i>=p.dims.x*p.dims.y){return;}
 let c=w.best;let xy=vec2<i32>(i32(i%p.dims.x),i32(i/p.dims.x));let a=mask_at(c.pos.z,xy.x-i32(c.pos.x),xy.y-i32(c.pos.y));
 image[i]=vec4<f32>(image[i].xyz+a*(c.color.xyz-image[i].xyz),0.0);w.data[i].y+=a*w.data[i].x;w.data[i].x+=a*(1.0-w.data[i].x);
}
@compute @workgroup_size(1) fn accept(){
 if(w.state.x!=0u){return;}var best=w.best;
 if(p.options.w==1u){
  if(best.color.w>1e-9&&w.stats.z-best.color.w<w.saved.x-1e-9){let index=u32(w.saved.y);let identity=h.glyphs[index].pos.w;best.pos.w=identity;
   for(var i=index;i+1u<w.counts.x;i++){h.glyphs[i]=h.glyphs[i+1u];}h.glyphs[w.counts.x-1u]=best;w.best=best;w.pending=vec4<u32>(2u,identity,p.options.y,p.options.z);
  }w.state.z=0u;w.state.x=1u;return;
 }
 if(best.color.w<=p.numbers.z){w.state.x=1u;return;}
 let identity=w.counts.x;best.pos.w=identity;h.glyphs[w.counts.x]=best;w.best=best;w.counts.x++;w.pending=vec4<u32>(1u,identity,p.options.y,p.options.z);
 if(w.counts.x>=p.dims.w){w.state.x=2u;}
}
@compute @workgroup_size(1) fn begin_refine(){
 w.saved.x=w.stats.z;var found=false;for(var i=0u;i<w.counts.x;i++){if(h.glyphs[i].pos.w==p.control.x){w.saved.y=f32(i);found=true;break;}}
 w.state.x=select(1u,0u,found);w.state.z=select(0u,1u,found);
}
@compute @workgroup_size(128) fn rebuild(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;if(i>=p.dims.x*p.dims.y){return;}var color=p.paper.xyz;var coverage=0.0;var overlap=0.0;
 let xy=vec2<i32>(i32(i%p.dims.x),i32(i/p.dims.x));
 for(var j=0u;j<w.counts.x;j++){let c=h.glyphs[j];if(w.state.z==1u&&c.pos.w==p.control.x){continue;}let a=mask_at(c.pos.z,xy.x-i32(c.pos.x),xy.y-i32(c.pos.y));color+=a*(c.color.xyz-color);overlap+=a*coverage;coverage+=a*(1.0-coverage);}
 image[i]=vec4<f32>(color,0.0);w.data[i]=vec4<f32>(coverage,overlap,0.0,0.0);
}
`;
}
