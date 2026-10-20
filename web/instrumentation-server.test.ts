import {expect,test} from "bun:test";
import {instrumentationHandler} from "./instrumentation-server";

function post(path:string,payload:unknown,origin="http://localhost:8766"):Request {
  return new Request(`http://127.0.0.1:8767${path}`,{method:"POST",headers:{"Content-Type":"application/json",Origin:origin},body:JSON.stringify(payload)});
}

test("state is readable without clipboard and persisted separately from bounded events",async()=>{
  const saved:Record<string,unknown>={};
  const handle=instrumentationHandler(async(kind,value)=>{saved[kind]=value;});
  expect(await (await handle(new Request("http://localhost/state"))).json()).toBeNull();
  const state={runSettings:{pursuit_repeat_penalty:.5},pendingControls:{"repeat-penalty":"2"}};
  const response=await handle(post("/state",{sessionId:"tab-a",state}));
  expect(response.status).toBe(200);
  expect(response.headers.get("Access-Control-Allow-Origin")).toBe("http://localhost:8766");
  const latest=await (await handle(new Request("http://localhost/state"))).json();
  expect(latest.state).toEqual(state);expect(latest.sessionId).toBe("tab-a");expect(saved.state).toEqual(latest);
  for(let index=0;index<205;index++)await handle(post("/events",{sessionId:"tab-a",event:"progress",index}));
  const events=await (await handle(new Request("http://localhost/events"))).json();
  expect(events.length).toBe(200);expect(events[0].index).toBe(5);expect(events.at(-1).index).toBe(204);
  expect(saved.events).toEqual(events);
});

test("collector rejects remote origins, malformed payloads and excessive bodies",async()=>{
  const handle=instrumentationHandler();
  expect((await handle(post("/state",{sessionId:"x",state:{}},"https://example.com"))).status).toBe(403);
  expect((await handle(post("/state",{}))).status).toBe(400);
  expect((await handle(post("/events",{sessionId:"x"}))).status).toBe(400);
  expect((await handle(post("/events",{sessionId:"x",event:"large",data:"x".repeat(8192)}))).status).toBe(413);
  expect((await handle(new Request("http://localhost/state",{method:"POST",headers:{"Content-Type":"application/json"},body:"{"}))).status).toBe(400);
  expect((await handle(new Request("http://localhost/state",{method:"POST",body:"text"}))).status).toBe(415);
  expect((await handle(new Request("http://localhost/state",{method:"OPTIONS",headers:{Origin:"http://localhost:8766"}}))).status).toBe(204);
});
