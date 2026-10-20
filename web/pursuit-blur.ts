/** Linear Gaussian-pyramid approximation: repeated separable [1,2,1]/4 filters.
 * Every edit is compared with the same weighted objective, never a size-specific
 * error baseline. Kernel sizes define the blur curve's sampling locations.
 */
export interface BlurCurvePoint { size:number; level:number; sigma:number; }
interface Image {width:number;height:number;data:Float32Array;}
interface Patch extends Image {x:number;y:number;}
const luma=(a:number,b:number,c:number):number=>.299*a+.587*b+.114*c;
const clamp=(v:number,lo:number,hi:number):number=>Math.max(lo,Math.min(hi,v));

function reduce(image:Image,channels:number):Image {
  const width=Math.ceil(image.width/2),height=Math.ceil(image.height/2),data=new Float32Array(width*height*channels);
  for(let y=0;y<height;++y)for(let x=0;x<width;++x)for(let dy=-1;dy<=1;++dy)for(let dx=-1;dx<=1;++dx){
    const sx=clamp(2*x+dx,0,image.width-1),sy=clamp(2*y+dy,0,image.height-1),w=(dx===0?2:1)*(dy===0?2:1)/16;
    for(let c=0;c<channels;++c)data[(y*width+x)*channels+c]+=w*image.data[(sy*image.width+sx)*channels+c];
  }
  return {width,height,data};
}
function reducePatch(patch:Patch,parent:Image):Patch {
  const x=Math.max(0,Math.ceil((patch.x-1)/2)),y=Math.max(0,Math.ceil((patch.y-1)/2));
  const width=Math.max(0,Math.min(Math.ceil(parent.width/2),Math.floor((patch.x+patch.width)/2)+1)-x);
  const height=Math.max(0,Math.min(Math.ceil(parent.height/2),Math.floor((patch.y+patch.height)/2)+1)-y);
  const data=new Float32Array(width*height*4);
  for(let py=0;py<height;++py)for(let px=0;px<width;++px)for(let dy=-1;dy<=1;++dy)for(let dx=-1;dx<=1;++dx){
    const sx=clamp(2*(x+px)+dx,0,parent.width-1)-patch.x,sy=clamp(2*(y+py)+dy,0,parent.height-1)-patch.y;
    if(sx<0||sy<0||sx>=patch.width||sy>=patch.height)continue;
    const w=(dx===0?2:1)*(dy===0?2:1)/16;
    for(let c=0;c<4;++c)data[(py*width+px)*4+c]+=w*patch.data[(sy*patch.width+sx)*4+c];
  }
  return {x,y,width,height,data};
}

export class KernelBlurObjective {
  readonly curve:BlurCurvePoint[];
  readonly weights:number[];
  private readonly cache=new WeakMap<Float32Array,{levels:Image[];energy:number;guide?:Float32Array}>();
  constructor(readonly target:Float32Array,readonly width:number,readonly height:number,sizes:number[],readonly strength:number,readonly edgeWeight:number){
    if(!Number.isFinite(strength)||strength<0||strength>3)throw new Error("Invalid kernel blur strength");
    this.curve=sizes.map(size=>{const level=clamp(Math.log2(Math.max(1,size*strength/6)),0,5);return {size,level,sigma:Math.sqrt((4**level-1)/6)};});
    this.weights=Array(Math.ceil(Math.max(0,...this.curve.map(p=>p.level)))+1).fill(0);
    this.weights[0]=.15;
    for(const {level} of this.curve){const lo=Math.floor(level),fraction=level-lo,w=.85/this.curve.length;this.weights[lo]+=w*(1-fraction);if(fraction)this.weights[lo+1]+=w*fraction;}
  }
  private prepare(pixels:Float32Array):{levels:Image[];energy:number;guide?:Float32Array}{
    const cached=this.cache.get(pixels);if(cached)return cached;
    const residual=Float32Array.from(this.target,(value,i)=>value-pixels[i]);
    const levels:Image[]=[{width:this.width,height:this.height,data:residual}];
    for(let i=1;i<this.weights.length;++i)levels.push(reduce(levels[i-1],3));
    let energy=0;
    for(let level=0;level<levels.length;++level){
      const {width,height,data}=levels[level],weight=this.weights[level]/(width*height);
      for(let i=0;i<width*height;++i){
        for(let c=0;c<3;++c)energy+=weight*data[i*3+c]**2/3;
        const r=luma(data[i*3],data[i*3+1],data[i*3+2]);
        for(const j of [i%width+1<width?i+1:-1,i+width<width*height?i+width:-1])if(j>=0)
          energy+=weight*this.edgeWeight*(r-luma(data[j*3],data[j*3+1],data[j*3+2]))**2;
      }
    }
    const prepared={levels,energy};this.cache.set(pixels,prepared);return prepared;
  }
  invalidate(pixels:Float32Array):void{this.cache.delete(pixels);}
  energy(pixels:Float32Array):number{return this.prepare(pixels).energy;}
  /** Cheap smoothed residual guidance; final acceptance uses exact pyramid deltas. */
  guide(pixels:Float32Array):Float32Array{
    const prepared=this.prepare(pixels);if(prepared.guide)return prepared.guide;
    const guide=pixels.slice();
    for(let level=0;level<prepared.levels.length;++level){
      const image=prepared.levels[level],scale=2**level,weight=this.weights[level];
      for(let y=0;y<this.height;++y)for(let x=0;x<this.width;++x){
        const i=(y*this.width+x)*3,j=(Math.min(image.height-1,Math.floor(y/scale))*image.width+Math.min(image.width-1,Math.floor(x/scale)))*3;
        for(let c=0;c<3;++c)guide[i+c]+=weight*image.data[j+c];
      }
    }
    prepared.guide=guide;return guide;
  }
  fit(pixels:Float32Array,left:number,top:number,size:number,alpha:Float32Array,colorEnabled:boolean):{color:number[];gain:number}{
    const {levels}=this.prepare(pixels);
    let patch:Patch={x:left,y:top,width:size,height:size,data:new Float32Array(size*size*4)};
    for(let y=0;y<size;++y)for(let x=0;x<size;++x){const a=alpha[y*size+x],i=(y*size+x)*4,j=((top+y)*this.width+left+x)*3;patch.data[i]=a;for(let c=0;c<3;++c)patch.data[i+1+c]=a*pixels[j+c];}
    const patches:Patch[]=[];
    for(let i=0;i<levels.length;++i){patches.push(patch);if(i+1<levels.length)patch=reducePatch(patch,levels[i]);}
    let denominator=0,gD=0,gN=0;const numerator=[0,0,0];
    const at=(p:Patch,x:number,y:number,c:number):number=>x<p.x||y<p.y||x>=p.x+p.width||y>=p.y+p.height?0:p.data[((y-p.y)*p.width+x-p.x)*4+c];
    const pairs=(p:Patch,image:Image,visit:(x:number,y:number,nx:number,ny:number)=>void):void=>{
      for(let y=Math.max(0,p.y-1);y<Math.min(image.height,p.y+p.height);++y)for(let x=Math.max(0,p.x-1);x<Math.min(image.width,p.x+p.width);++x){if(x+1<image.width)visit(x,y,x+1,y);if(y+1<image.height)visit(x,y,x,y+1);}
    };
    for(let level=0;level<levels.length;++level){
      const image=levels[level],p=patches[level],weight=this.weights[level]/(image.width*image.height);
      for(let y=p.y;y<p.y+p.height;++y)for(let x=p.x;x<p.x+p.width;++x){const a=at(p,x,y,0),i=(y*image.width+x)*3;denominator+=weight*a*a/3;for(let c=0;c<3;++c)numerator[c]+=weight*a*(image.data[i+c]+at(p,x,y,c+1))/3;}
      pairs(p,image,(x,y,nx,ny)=>{const a=at(p,x,y,0)-at(p,nx,ny,0),i=(y*image.width+x)*3,j=(ny*image.width+nx)*3;
        const r=luma(image.data[i]-image.data[j]+at(p,x,y,1)-at(p,nx,ny,1),image.data[i+1]-image.data[j+1]+at(p,x,y,2)-at(p,nx,ny,2),image.data[i+2]-image.data[j+2]+at(p,x,y,3)-at(p,nx,ny,3));
        gD+=weight*this.edgeWeight*a*a;gN+=weight*this.edgeWeight*a*r;
      });
    }
    let color=numerator.map(n=>n/Math.max(denominator,1e-20));
    if(colorEnabled){const adjustment=(gN-gD*luma(color[0],color[1],color[2]))/Math.max(3*denominator+gD,1e-20);color=color.map(c=>clamp(c+adjustment,0,1));}
    else color.fill(clamp((numerator.reduce((a,b)=>a+b,0)+gN)/Math.max(3*denominator+gD,1e-20),0,1));
    let gain=0;
    for(let level=0;level<levels.length;++level){const image=levels[level],p=patches[level],weight=this.weights[level]/(image.width*image.height);
      const delta=(x:number,y:number,c:number):number=>at(p,x,y,0)*color[c]-at(p,x,y,c+1);
      for(let y=p.y;y<p.y+p.height;++y)for(let x=p.x;x<p.x+p.width;++x)for(let c=0;c<3;++c){const d=delta(x,y,c),r=image.data[(y*image.width+x)*3+c];gain+=weight*(2*r*d-d*d)/3;}
      pairs(p,image,(x,y,nx,ny)=>{const i=(y*image.width+x)*3,j=(ny*image.width+nx)*3,r=luma(image.data[i]-image.data[j],image.data[i+1]-image.data[j+1],image.data[i+2]-image.data[j+2]),d=luma(delta(x,y,0)-delta(nx,ny,0),delta(x,y,1)-delta(nx,ny,1),delta(x,y,2)-delta(nx,ny,2));gain+=weight*this.edgeWeight*(2*r*d-d*d);});
    }
    return {color,gain};
  }
}
