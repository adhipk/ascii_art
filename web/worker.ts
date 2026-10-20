import { renderPursuitAsync } from "./pursuit";
import { renderInPython } from "./python-render";
import { TinyEdges } from "./tiny-edges";
const tinyEdges = new TinyEdges();

// The page never sends executable code: only pixels and renderer settings.
const CDN = "https://cdn.jsdelivr.net/pyodide/v0.29.3/full/";
let runtime: Promise<any> | undefined;
const publishStatus = (message: string, id: number) => self.postMessage({ type: "status", message, id });
let sourcePixels: Uint8Array | undefined;
let sourceVersion = -1;

async function initialize(id: number) {
  publishStatus("Loading Python WebAssembly runtime…", id);
  const { loadPyodide } = await import(/* @vite-ignore */ `${CDN}pyodide.mjs`);
  const py = await loadPyodide({ indexURL: CDN });
  publishStatus("Loading image libraries… This first download can take a minute.", id);
  await py.loadPackage(["numpy", "opencv-python", "pillow"]);
  const paths = [
    "__init__.py", "generators/__init__.py", "generators/sprite_generator.py", "generators/adaptive.py",
    "filters/__init__.py", "filters/dog_filter.py", "filters/edge_selection.py", "filters/acerola.py", "utils/__init__.py", "utils/helpers.py", "utils/background.py", "utils/color.py",
  ];
  for (const dir of ["/app/ascii_art", "/app/ascii_art/generators", "/app/ascii_art/filters", "/app/ascii_art/utils"]) py.FS.mkdirTree(dir);
  await Promise.all(paths.map(async (path) => {
    const response = await fetch(new URL(`./python/ascii_art/${path}`, self.location.href));
    if (!response.ok) throw new Error(`Could not load renderer source: ${path}`);
    py.FS.writeFile(`/app/ascii_art/${path}`, await response.text());
  }));
  const bridge = await fetch(new URL("./python/bridge.py", self.location.href));
  if (!bridge.ok) throw new Error("Could not load browser adapter");
  py.FS.writeFile("/app/bridge.py", await bridge.text());
  py.runPython("import sys\nsys.path.insert(0, '/app')\nfrom bridge import render_rgba\nimport json");
  return py;
}

self.onmessage = async ({ data }) => {
  if (data.type !== "render") return;
  try {
    if (data.pixels) {
      sourcePixels = new Uint8Array(data.pixels);
      sourceVersion = data.sourceVersion;
    }
    if (!sourcePixels || sourceVersion !== data.sourceVersion) throw new Error("Source pixels are unavailable. Choose the image again.");
    if (data.settings.layout === "pursuit") {
      const output = await renderPursuitAsync(sourcePixels, data.width, data.height, data.settings, message => publishStatus(message, data.id));
      self.postMessage({ type: "result", id: data.id, ...output });
      return;
    }
    runtime ??= initialize(data.id).catch((error) => { runtime = undefined; throw error; });
    const py = await runtime;
    publishStatus("Rendering characters…", data.id);
    let edges;
    if (data.settings.use_edge && data.settings.edge_method === "tiny") {
      try {
        edges = await tinyEdges.detect(sourcePixels, data.width, data.height, data.sourceVersion, (message) => publishStatus(message, data.id));
      } catch (error) {
        throw new Error(`Tiny edge detector failed. Choose Classic DoG to continue. ${error instanceof Error ? error.message : error}`);
      }
    }
    const output = renderInPython(py, sourcePixels, data.width, data.height, data.settings, edges);
    self.postMessage({ type: "result", id: data.id, ...output });
  } catch (error) {
    self.postMessage({ type: "error", id: data.id, message: error instanceof Error ? error.message : String(error) });
  }
};
