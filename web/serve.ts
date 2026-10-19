import { resolve, sep } from "node:path";
const root = resolve(import.meta.dir, "dist");

export async function staticResponse(request: Request): Promise<Response> {
  let pathname: string;
  try { pathname = decodeURIComponent(new URL(request.url).pathname); }
  catch { return new Response("Bad request", { status: 400 }); }
  const path = resolve(root, `.${pathname === "/" ? "/index.html" : pathname}`);
  if (!path.startsWith(root + sep)) return new Response("Forbidden", { status: 403 });
  const file = Bun.file(path);
  if (!(await file.exists())) return new Response("Not found", { status: 404 });
  return new Response(file, { headers: {
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
    ...(path.endsWith(".py") ? { "Content-Type": "text/plain; charset=utf-8" } : {}),
  } });
}

if (import.meta.main) {
  const server = Bun.serve({
    hostname: "127.0.0.1", port: Number(Bun.env.PORT ?? 8766), fetch: staticResponse,
  });
  console.log(`ASCII studio: ${server.url}`);
}
