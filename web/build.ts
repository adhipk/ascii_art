import { cp, mkdir, rm } from "node:fs/promises";
import { resolve } from "node:path";
import { discoverExamples } from "./example-catalog";
const root = import.meta.dir;
const dist = resolve(root, "dist");
await rm(dist, { recursive: true, force: true });
await mkdir(resolve(dist, "python"), { recursive: true });
const result = await Bun.build({
  entrypoints: [resolve(root, "app.ts"), resolve(root, "worker.ts")],
  outdir: dist, target: "browser", naming: "[name].[ext]",
});
if (!result.success) throw new AggregateError(result.logs, "Browser build failed");
// CSS has no build-time imports. Keep its stable URL across Bun versions.
for (const file of ["index.html", "style.css"]) await cp(resolve(root, file), resolve(dist, file));
await cp(resolve(root, "../src/ascii_art"), resolve(dist, "python/ascii_art"), {
  recursive: true, filter: (path) => !path.includes("__pycache__") && !path.includes("/assets"),
});
await cp(resolve(root, "examples"), resolve(dist, "examples"), { recursive: true });
await Bun.write(resolve(dist, "examples.json"), JSON.stringify(await discoverExamples(resolve(root, "examples"))));
await cp(resolve(root, "bridge.py"), resolve(dist, "python/bridge.py"));
console.log(`Built static site: ${dist}`);
