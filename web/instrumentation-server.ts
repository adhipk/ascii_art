import {mkdir} from "node:fs/promises";
import {resolve} from "node:path";

export function instrumentationHandler(persist:(kind:"state"|"events",value:unknown)=>Promise<void>=async()=>{}) {
  let state:unknown=null;
  const events:unknown[]=[];
  return async(request:Request):Promise<Response>=>{
    const origin=request.headers.get("origin");
    if(origin){
      let allowed=false;
      try{const url=new URL(origin);allowed=url.protocol==="http:"&&["localhost","127.0.0.1","[::1]"].includes(url.hostname);}catch{}
      if(!allowed)return new Response("Forbidden origin",{status:403});
    }
    const headers={"Cache-Control":"no-store","Vary":"Origin",...(origin?{"Access-Control-Allow-Origin":origin}:{}),
      "Access-Control-Allow-Methods":"GET, POST, OPTIONS","Access-Control-Allow-Headers":"Content-Type"};
    const reply=(value:unknown,status=200)=>Response.json(value,{status,headers});
    if(request.method==="OPTIONS")return new Response(null,{status:204,headers});
    const path=new URL(request.url).pathname;
    if(request.method==="GET"){
      if(path==="/health")return reply({ok:true});
      if(path==="/state")return reply(state);
      if(path==="/events")return reply(events);
      return reply({error:"Not found"},404);
    }
    if(request.method!=="POST")return reply({error:"Method not allowed"},405);
    if(!["/state","/events"].includes(path))return reply({error:"Not found"},404);
    if(!request.headers.get("content-type")?.startsWith("application/json"))return reply({error:"Expected JSON"},415);
    const limit=path==="/state"?512*1024:8192;
    const reader=request.body?.getReader();
    if(!reader)return reply({error:"Missing body"},400);
    const chunks:Uint8Array[]=[];let length=0;
    while(true){const chunk=await reader.read();if(chunk.done)break;length+=chunk.value.length;
      if(length>limit){await reader.cancel();return reply({error:"Payload too large"},413);}chunks.push(chunk.value);}
    const bytes=new Uint8Array(length);let offset=0;
    for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.length;}
    let payload:Record<string,unknown>;
    try{payload=JSON.parse(new TextDecoder().decode(bytes));}catch{return reply({error:"Invalid JSON"},400);}
    if(!payload||typeof payload!=="object"||Array.isArray(payload)||typeof payload.sessionId!=="string")return reply({error:"Expected sessionId and payload"},400);
    if(path==="/state"&&(!payload.state||typeof payload.state!=="object"))return reply({error:"Expected state"},400);
    if(path==="/events"&&typeof payload.event!=="string")return reply({error:"Expected event"},400);
    const entry={...payload,receivedAt:new Date().toISOString()};
    if(path==="/state")state=entry;
    else{events.push(entry);if(events.length>200)events.shift();}
    try{await persist(path==="/state"?"state":"events",path==="/state"?state:[...events]);}
    catch{return reply({error:"Could not persist instrumentation"},500);}
    return reply({ok:true});
  };
}

if(import.meta.main){
  const directory=resolve(import.meta.dir,".debug");await mkdir(directory,{recursive:true});
  let writes=Promise.resolve();
  const handler=instrumentationHandler((kind,value)=>{
    const json=JSON.stringify(value,null,2);
    const next=writes.catch(()=>{}).then(async()=>{await Bun.write(resolve(directory,`${kind}.json`),json);});writes=next;return next;
  });
  const server=Bun.serve({hostname:"127.0.0.1",port:Number(Bun.env.INSTRUMENTATION_PORT??8767),fetch:handler});
  console.log(`ASCII instrumentation: ${server.url} · ${directory}`);
}
