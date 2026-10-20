export function sharpenTarget(target:Float32Array,width:number,height:number,amount:number):Float32Array {
  if(!Number.isFinite(amount)||amount<0||amount>3)throw new Error("Detail sharpening must be between 0 and 3");
  const output=target.slice();
  if(amount===0)return output;
  const luma=new Float32Array(width*height),horizontal=new Float32Array(width*height);
  for(let index=0;index<luma.length;index++)luma[index]=target[index*3]*.299+target[index*3+1]*.587+target[index*3+2]*.114;
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const index=y*width+x;
    horizontal[index]=(luma[y*width+Math.max(0,x-1)]+2*luma[index]+luma[y*width+Math.min(width-1,x+1)])/4;
  }
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const index=y*width+x;
    const blurred=(horizontal[Math.max(0,y-1)*width+x]+2*horizontal[index]+horizontal[Math.min(height-1,y+1)*width+x])/4;
    const detail=amount*(luma[index]-blurred);
    for(let channel=0;channel<3;channel++)output[index*3+channel]=Math.max(0,Math.min(1,target[index*3+channel]+detail));
  }
  return output;
}
