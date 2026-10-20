import { expect, test } from "bun:test";
import { staticResponse } from "./serve";
import { resolve } from "node:path";
import {FIT_PRESETS} from "./fit-presets";

test("studio defaults to shared-canvas fitting and retains independent debugging",async()=>{
  const html=await (await staticResponse(new Request("http://localhost/"))).text();
  const select=html.match(/<select id="scale-schedule">([\s\S]*?)<\/select>/)![1];
  expect(select).toContain('<option value="all" selected>');
  expect(select).toContain('<option value="independent">Debug · independent');
  expect(select).not.toContain('<option value="independent" selected>');
  for(const preset of Object.values(FIT_PRESETS))expect(preset["scale-schedule"]).toBe("all");
});

test("static page, modules, and Python sources are served with correct MIME types", async () => {
  for (const [path, mime] of [["/", "text/html"], ["/app.js", "javascript"], ["/worker.js", "javascript"], ["/style.css", "text/css"], ["/python/bridge.py", "text/"]]) {
    const response = await staticResponse(new Request(`http://localhost${path}`));
    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toContain(mime);
    expect((await response.text()).length).toBeGreaterThan(0);
  }
});

test("build copies the actual Python implementation without rewriting it", async () => {
  for (const path of ["__init__.py", "generators/sprite_generator.py", "generators/adaptive.py", "filters/dog_filter.py", "filters/acerola.py", "utils/helpers.py"]) {
    const built = await Bun.file(resolve(import.meta.dir, `dist/python/ascii_art/${path}`)).text();
    const source = await Bun.file(resolve(import.meta.dir, `../src/ascii_art/${path}`)).text();
    expect(built).toBe(source);
  }
});

test("reference detector and its separate vote control are available without a model", async () => {
  const html = await (await staticResponse(new Request("http://localhost/"))).text();
  expect(html).toContain('<option value="acerola" selected>');
  expect(html).toContain('id="edge-min-pixels"');
  const worker = await (await staticResponse(new Request("http://localhost/worker.js"))).text();
  expect(worker).toContain("filters/acerola.py");
});

test("unknown, invalid, and escaping paths are rejected", async () => {
  expect((await staticResponse(new Request("http://localhost/missing"))).status).toBe(404);
  expect((await staticResponse(new Request("http://localhost/%ZZ"))).status).toBe(400);
  expect((await staticResponse(new Request("http://localhost/..%2Fpackage.json"))).status).toBe(403);
});

test("all bundled examples and thumbnails are served intact", async () => {
  const examples = await (await staticResponse(new Request("http://localhost/examples.json"))).json();
  expect(examples.length).toBeGreaterThanOrEqual(4);
  expect(new Set(examples.map((example: any) => example.id)).size).toBe(examples.length);
  for (const example of examples) {
    for (const name of [example.file, example.thumbnail]) {
      const response = await staticResponse(new Request(`http://localhost/examples/${name}`));
      expect(response.status).toBe(200);
      expect(response.headers.get("content-type")).toContain("image/");
      const bytes = new Uint8Array(await response.arrayBuffer());
      const source = new Uint8Array(await Bun.file(resolve(import.meta.dir, `examples/${name}`)).arrayBuffer());
      expect(bytes).toEqual(source);
      if (name !== example.file && name === example.thumbnail) expect(bytes.length).toBeLessThan(30_000);
    }
  }
});

test("the initial view leaves contour effects off", async () => {
  const response = await staticResponse(new Request("http://localhost/"));
  const html = await response.text();
  const checkbox = html.match(/<input[^>]*id="edges"[^>]*>/)?.[0];
  expect(checkbox).toBeDefined();
  expect(checkbox).not.toContain("checked");
  expect(html).toContain("Contour strength");
});

test("preview uses text instead of a raster image", async () => {
  const html = await (await staticResponse(new Request("http://localhost/"))).text();
  expect(html).toContain('id="output" class="ascii-text"');
  expect(html).not.toMatch(/<img[^>]*id="output"/);
});

test("stylesheet contains parsed editor CSS, never an HTML document", async () => {
  const response = await staticResponse(new Request("http://localhost/style.css"));
  expect(response.headers.get("content-type")).toContain("text/css");
  const css = await response.text();
  expect(css).not.toMatch(/<!doctype|<html|<script/i);
  for (const selector of [".workspace", ".stage", ".inspector", ".ascii-text", ".example-list"]) expect(css).toContain(selector);
  expect(css).toContain("grid-template-columns");
});

test("local server prevents stale HTML, CSS, JS and Python assets", async () => {
  for (const path of ["/", "/style.css", "/app.js", "/worker.js", "/python/bridge.py"]) {
    const response = await staticResponse(new Request(`http://localhost${path}`));
    expect(response.headers.get("cache-control")).toBe("no-store");
  }
});

test("stylesheet is copied verbatim to the stable URL used by index.html", async () => {
  const source = await Bun.file(resolve(import.meta.dir,"style.css")).text();
  const response = await staticResponse(new Request("http://localhost/style.css"));
  expect(await response.text()).toBe(source);
  const build = await Bun.file(resolve(import.meta.dir,"build.ts")).text();
  expect(build).toContain('["index.html", "style.css"]');
});
