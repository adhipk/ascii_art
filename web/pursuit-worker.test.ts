import { expect, test } from "bun:test";

test("multiscale worker runs and reuses source pixels without loading Python or an edge model", async () => {
  const oldSelf = globalThis.self, oldFetch = globalThis.fetch;
  const messages:any[]=[];
  let fetches=0;
  const worker:any={postMessage:(message:any)=>messages.push(message),location:{href:"http://localhost/worker.js"}};
  try {
    globalThis.self=worker;
    globalThis.fetch=async()=>{++fetches;throw new Error("Network must not be used");};
    await import("./worker");
    const pixels=new Uint8Array(16*16*4);for(let i=3;i<pixels.length;i+=4)pixels[i]=255;
    const settings={layout:"pursuit",columns:24,font_size:12,pursuit_budget:1,pursuit_beam:1,pursuit_edge_weight:3,use_color:false,use_edge:true,edge_method:"tiny",background:"#000000",background_mode:"solid",foreground:"#ffffff",pursuit_metrics:{width:4,height:4,baseline:4,advance:4,fontSize:4,densities:[],kernelCharacters:" X",masks:[Array(16).fill(0),Array(16).fill(1)]}};
    await worker.onmessage({data:{type:"render",id:1,sourceVersion:1,pixels:pixels.buffer,width:16,height:16,settings}});
    expect(messages.at(-1).type).toBe("result");expect(messages.at(-1).pursuit.frames.length).toBe(1);
    await worker.onmessage({data:{type:"render",id:2,sourceVersion:1,width:16,height:16,settings}});
    expect(messages.at(-1).type).toBe("result");expect(messages.at(-1).id).toBe(2);
    expect(fetches).toBe(0);
  } finally {globalThis.self=oldSelf;globalThis.fetch=oldFetch;}
});
