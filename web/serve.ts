import { resolve, sep } from "node:path";
import { discoverExamples } from "./example-catalog";
const root = resolve(import.meta.dir, "dist");
const sourceExamples = resolve(import.meta.dir, "examples");

export async function staticResponse(request: Request, examplesDirectory = sourceExamples): Promise<Response> {
  let pathname: string;
  try { pathname = decodeURIComponent(new URL(request.url).pathname); }
  catch { return new Response("Bad request", { status: 400 }); }
  if (pathname === "/examples.json") return Response.json(await discoverExamples(examplesDirectory), { headers: { "Cache-Control": "no-store" } });
  const path = resolve(root, `.${pathname === "/" ? "/index.html" : pathname}`);
  if (!path.startsWith(root + sep)) return new Response("Forbidden", { status: 403 });
  let file = Bun.file(path);
  if (pathname.startsWith("/examples/")) {
    const name = pathname.slice("/examples/".length);
    const catalog = await discoverExamples(examplesDirectory);
    if (!catalog.some(example => example.file === name || example.thumbnail === name)) return new Response("Not found", { status: 404 });
    file = Bun.file(resolve(examplesDirectory, name));
  }
  if (!(await file.exists())) return new Response("Not found", { status: 404 });
  return new Response(file, { headers: {
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
    ...(path.endsWith(".py") ? { "Content-Type": "text/plain; charset=utf-8" } : {}),
  } });
}

if (import.meta.main) {
  const server = Bun.serve({
    hostname: "127.0.0.1", port: Number(Bun.env.PORT ?? 8766), fetch: request => staticResponse(request),
  });
  console.log(`ASCII studio: ${server.url}`);
}
