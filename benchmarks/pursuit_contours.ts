/** Known recoverable strokes expose cell-centre / ink-centre placement errors. */
import {renderPursuit} from "../web/pursuit";
import type {RenderSettings} from "../web/renderer";
const output=Bun.argv[2]??"/tmp/pursuit-contours.json";
const rows=[];
for(const kind of ["underline","off-center vertical","corner"]){
  const char=kind==="underline"?"_":kind==="corner"?"L":"|";
  const mask=Array.from({length:64},(_,i)=>Number(char==="_"?Math.floor(i/8)===7:char==="|"?i%8===1:i%8===1||Math.floor(i/8)===7));
  const pixels=new Uint8Array(128*128*4),target=new Float32Array(128*128);
  for(let i=3;i<pixels.length;i+=4)pixels[i]=255;
  for(let y=0;y<32;++y)for(let x=0;x<32;++x){const a=mask[Math.floor(y/4)*8+Math.floor(x/4)],i=(44+y)*128+44+x;target[i]=a;for(let c=0;c<3;++c)pixels[i*4+c]=a*255;}
  for(const budget of [1,8]){
    const settings:RenderSettings={columns:24,font_size:12,layout:"pursuit",pursuit_schedule:"coarse-to-fine",pursuit_resolution:128,pursuit_budget:budget,pursuit_beam:1,pursuit_blur:1,pursuit_edge_weight:3,
      pursuit_metrics:{width:8,height:8,baseline:7,advance:8,fontSize:8,densities:[],kernelCharacters:` ${char}`,masks:[Array(64).fill(0),mask]},
      use_edge:false,edge_method:"acerola",edge_sensitivity:50,edge_amount:100,use_color:false,background:"#000000",background_mode:"solid",foreground:"#ffffff"};
    const result=renderPursuit(pixels,128,128,settings),rendered=new Float32Array(128*128),factor=result.width/128;
    for(const glyph of result.tiles!){
      const size=Math.round(glyph.size/factor),left=Math.round(glyph.x/factor),top=Math.round(glyph.y/factor),ink=glyph.color[0]/255;
      for(let y=0;y<size;++y)for(let x=0;x<size;++x){let alpha=0,count=0;
        for(let sy=Math.floor(y*8/size);sy<Math.ceil((y+1)*8/size);++sy)for(let sx=Math.floor(x*8/size);sx<Math.ceil((x+1)*8/size);++sx){alpha+=glyph.angle===90?mask[(7-sx)*8+sy]:mask[sy*8+sx];++count;}
        const i=(top+y)*128+left+x;rendered[i]+=alpha/count*(ink-rendered[i]);
      }
    }
    const edges=(image:Float32Array):number[]=>{
      const points:number[]=[];
      for(let y=0;y<128;++y)for(let x=0;x<128;++x){const i=y*128+x;if((x<127&&Math.abs(image[i]-image[i+1])>.1)||(y<127&&Math.abs(image[i]-image[i+128])>.1))points.push(i);}
      return points;
    };
    const expected=edges(target),actual=edges(rendered);
    const distances=(a:number[],b:number[]):number[]=>a.map(i=>Math.sqrt(Math.min(...b.map(j=>(i%128-j%128)**2+(Math.floor(i/128)-Math.floor(j/128))**2))));
    const precisionDistances=distances(actual,expected),recallDistances=distances(expected,actual),all=[...precisionDistances,...recallDistances];
    const precision=actual.length?precisionDistances.filter(v=>v<=1).length/actual.length:0,recall=expected.length?recallDistances.filter(v=>v<=1).length/expected.length:1;
    rows.push({kind,budget,glyphs:result.tiles!.length,energy:result.pursuit!.frames.at(-1)!.energy,mae:rendered.reduce((sum,v,i)=>sum+Math.abs(v-target[i]),0)/target.length,
      contourF1:precision+recall?2*precision*recall/(precision+recall):0,meanContourDistance:all.length?all.reduce((a,b)=>a+b,0)/all.length:null,ms:result.pursuit!.elapsedMs});
  }
}
await Bun.write(output,JSON.stringify({scope:"Synthetic measured-mask oracle, 128px. Contours are >0.1 neighboring luminance differences; F1 tolerance 1px. No browser-font or photographic quality claim.",rows},null,2)+"\n");
console.log(JSON.stringify(rows));
