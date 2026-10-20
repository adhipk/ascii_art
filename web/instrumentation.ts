const local=["localhost","127.0.0.1","[::1]"].includes(location.hostname)&&location.protocol==="http:";
const sessionId=crypto.randomUUID();
const endpoint="http://127.0.0.1:8767";
let pending:ReturnType<typeof setTimeout>|undefined;
let requests=Promise.resolve();

function publish(path:string,payload:Record<string,unknown>):void {
  if(!local)return;
  requests=requests.then(async()=>{
    const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),1500);
    try{await fetch(endpoint+path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({sessionId,...payload}),signal:controller.signal});}
    catch{}finally{clearTimeout(timeout);}
  });
}

export function instrumentEvent(event:string,details:Record<string,unknown>={}):void {
  publish("/events",{event,details,at:new Date().toISOString()});
}

export function instrumentState(capture:()=>unknown):void {
  if(!local)return;
  clearTimeout(pending);
  pending=setTimeout(()=>{
    try{publish("/state",{state:capture()});}catch{}
  },300);
}
