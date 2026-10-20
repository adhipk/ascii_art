import {expect,test} from "bun:test";
import {KernelBlurObjective} from "./pursuit-blur";
import {energyTerms} from "./pursuit";

const sizes=[48,24,12,6,3];
test("blur curve follows kernel size, responds continuously to strength, and keeps sharp detail",()=>{
  const target=new Float32Array(32*32*3);
  const model=new KernelBlurObjective(target,32,32,sizes,1,3);
  expect(model.weights.reduce((a,b)=>a+b,0)).toBeCloseTo(1,12);
  expect(model.weights[0]).toBeGreaterThanOrEqual(.15);
  for(let i=1;i<model.curve.length;++i)expect(model.curve[i].sigma).toBeLessThanOrEqual(model.curve[i-1].sigma);
  expect(new KernelBlurObjective(target,32,32,sizes,2,3).curve[0].sigma).toBeGreaterThan(model.curve[0].sigma);
  for(const strength of [-1,4,NaN,Infinity])expect(()=>new KernelBlurObjective(target,32,32,sizes,strength,3)).toThrow("blur");
});

test("blur zero recovers sharp error and an exact match scores zero at every blur",()=>{
  const target=Float32Array.from({length:13*17*3},(_,i)=>(i%19)/19),pixels=target.map(v=>v*.7);
  const terms=energyTerms(target,pixels,13,17);
  expect(new KernelBlurObjective(target,13,17,sizes,0,3).energy(pixels)).toBeCloseTo(terms.pixel+3*terms.contour,7);
  expect(new KernelBlurObjective(target,13,17,sizes,2,3).energy(target)).toBe(0);
});

test("high-frequency mismatches are discounted while broad structure remains visible",()=>{
  const target=new Float32Array(32*32*3).fill(.5),checker=target.slice(),broad=target.slice();
  for(let y=0;y<32;++y)for(let x=0;x<32;++x)for(let c=0;c<3;++c){checker[(y*32+x)*3+c]=(x+y)%2;broad[(y*32+x)*3+c]=x<16?0:1;}
  const sharp=new KernelBlurObjective(target,32,32,sizes,0,0),blur=new KernelBlurObjective(target,32,32,sizes,2,0);
  expect(sharp.energy(checker)).toBeCloseTo(sharp.energy(broad),12);
  expect(blur.energy(checker)).toBeLessThan(blur.energy(broad)*.7);
});

test("local pyramid gain matches full rendering at odd dimensions, borders and overlapping ink",()=>{
  const width=31,height=25,target=Float32Array.from({length:width*height*3},(_,i)=>(i*13%101)/101);
  const pixels=target.map((v,i)=>.1+.3*(i%7)/7),model=new KernelBlurObjective(target,width,height,[16,8,4,3],2,3);
  for(const [left,top] of [[0,0],[23,17],[7,9]]){
    const size=8,alpha=Float32Array.from({length:size*size},(_,i)=>(i%11)/11);
    const fit=model.fit(pixels,left,top,size,alpha,true),next=pixels.slice();
    for(let y=0;y<size;++y)for(let x=0;x<size;++x)for(let c=0;c<3;++c){const i=((top+y)*width+left+x)*3+c;next[i]+=alpha[y*size+x]*(fit.color[c]-next[i]);}
    expect(fit.gain).toBeCloseTo(model.energy(pixels)-model.energy(next),7);
    expect(fit.color.every(c=>c>=0&&c<=1)).toBe(true);
  }
});
