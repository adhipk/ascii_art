import { textMetrics, squareTextMetrics, textPreview, paperBackground, exportTextPng, FONT_FAMILY } from "./text-view";
import { EXAMPLES, exampleUrl, type Example } from "./examples";
import { RenderQueue } from "./render-queue";
import { Renderer, type RenderInput, type RenderResult, type RenderSettings } from "./renderer";
import { loadShapeFont, shapeAtlas } from "./shape-font";
import { captureOptions, restoreOptions, exportOptions, encodeProject, decodeProject } from "./project-state";

function element<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing element: ${id}`);
  return node as T;
}
const file = element<HTMLInputElement>("file");
const form = element<HTMLFormElement>("controls");
const stage = element<HTMLDivElement>("stage");
const original = element<HTMLImageElement>("original");
const output = element<HTMLDivElement>("output");
const png = element<HTMLButtonElement>("png");
const text = element<HTMLButtonElement>("text");
const status = element<HTMLSpanElement>("status");
const cancel = element<HTMLButtonElement>("cancel");
const retry = element<HTMLButtonElement>("retry");
const columns = element<HTMLInputElement>("columns");
const edges = element<HTMLInputElement>("edges");
const edgeAmount = element<HTMLInputElement>("edge-amount");
let source: ImageData | undefined;
let sourceVersion = 0;
let fileVersion = 0;
let rendered: RenderResult | undefined;
let renderedFontSize = 12;
let actualSize = false;
let textUrl: string | undefined;
let showingOriginal = false;
let downloadsReady = false;
let disposed = false;
let exampleDownload: AbortController | undefined;
let savedMetrics: Pick<RenderSettings,"text_metrics"|"adaptive_metrics"|"shape_atlas"> | undefined;
let lastSettings: RenderSettings | undefined;
let scheduleVersion = 0;
const exampleButtons = new Map<string, HTMLButtonElement>();

function message(value: string, error = false): void {
  status.textContent = value;
  status.classList.toggle("error", error);
}
function disableDownloads(): void { downloadsReady = false; png.disabled = text.disabled = true; element<HTMLButtonElement>("save-project").disabled=true; }
function releaseOutput(): void {
  if (textUrl) URL.revokeObjectURL(textUrl);
  rendered = undefined;
  textUrl = undefined;
  output.hidden = true;
  output.replaceChildren();
  disableDownloads();
}
function updateView(): void {
  const hasOutput = Boolean(rendered);
  original.hidden = !source || (!showingOriginal && hasOutput);
  output.hidden = showingOriginal || !hasOutput;
  element("empty").hidden = Boolean(source);
  element("view-ascii").setAttribute("aria-pressed", String(!showingOriginal));
  element("view-original").setAttribute("aria-pressed", String(showingOriginal));
}
const renderer = new Renderer((value, id) => { if (queue.isCurrent(id)) message(value); });
const queue = new RenderQueue<RenderInput, RenderResult>({
  render: (input, id) => renderer.render(input, id),
  result: (result, input, id) => { lastSettings=input.settings; displayResult(result, id, input.settings.font_size); },
  error: (error) => {
    message(error instanceof Error ? error.message : "Could not render image", true);
    retry.hidden = false;
  },
  state: (state) => {
    element("stage").setAttribute("aria-busy", String(state !== "idle"));
    document.querySelector(".image-workspace")!.setAttribute("data-busy", String(state !== "idle"));
    cancel.hidden = state === "idle";
    if (state === "waiting") message("Updating…");
    else if (state === "rendering") message(rendered ? "Updating…" : "Preparing preview…");
  },
  delay: 180,
});
function fitText(): void {
  if (!rendered) return;
  const inset = parseFloat(getComputedStyle(stage).paddingLeft) * 2;
  if (stage.clientWidth <= inset || stage.clientHeight <= inset) return;
  const scale = actualSize ? 1 : Math.min((stage.clientWidth - inset) / rendered.width, (stage.clientHeight - inset) / rendered.height);
  output.style.width = `${rendered.width * scale}px`;
  output.style.height = `${rendered.height * scale}px`;
  if (rendered.tiles || rendered.shapes) {
    output.style.letterSpacing = "0px";
    return;
  }
  output.style.fontSize = `${renderedFontSize * scale}px`;
  output.style.lineHeight = `${rendered.cellHeight * scale}px`;
  output.style.letterSpacing = `${rendered.cellWidth * scale - textMetrics(renderedFontSize * scale).advance}px`;
}
function displayResult(result: RenderResult, id: number, fontSize: number): void {
  if (disposed || !queue.isCurrent(id)) return;
  try {
    const content = textPreview(result);
    if (textUrl) URL.revokeObjectURL(textUrl);
    textUrl = URL.createObjectURL(new Blob([result.text], { type: "text/plain;charset=utf-8" }));
    rendered = result;
    renderedFontSize = fontSize;
    output.style.fontFamily = FONT_FAMILY;
    output.style.backgroundImage = paperBackground(result);
    output.replaceChildren(content);
    updateView();
    fitText();
    downloadsReady = true;
    png.disabled = text.disabled = false;
    element<HTMLButtonElement>("save-project").disabled=false;
    element("dimensions").textContent = result.tiles ? `${result.tiles.length} glyphs · 8 / 16 / 32 px` : `${result.columns} × ${result.rows} characters`;
    text.textContent = result.tiles ? "Text · fixed grid" : "Text";
    text.title = result.shapes ? "Plain text needs Fira Code and may shape across pair boundaries. PNG preserves the exact fitted shapes." : result.tiles ? "Plain text uses the separate fixed 8×8 grid. PNG preserves mixed sizes." : "Download character grid";
    element("runtime-note").hidden = true;
    const stats=result.shapes?.stats;
    message(stats ? `Up to date · ${Math.round(100*stats.culled/Math.max(1,stats.candidates))}% candidates pruned` : "Up to date");
  } catch (error) {
    message(error instanceof Error ? error.message : "Could not display preview", true);
    retry.hidden = false;
  }
}
new ResizeObserver(() => fitText()).observe(stage);
function settings(): RenderSettings {
  const background = element<HTMLInputElement>("background").value;
  const rgb = [1, 3, 5].map((i) => parseInt(background.slice(i, i + 2), 16));
  const reference = element<HTMLSelectElement>("edge-method").value === "acerola";
  const adaptive = element<HTMLSelectElement>("layout").value === "adaptive";
  const atlas = !adaptive && element<HTMLInputElement>("shape-matching").checked ? savedMetrics?.shape_atlas ?? shapeAtlas(Number(element<HTMLSelectElement>("font-size").value)) : undefined;
  return {
    columns: Number(columns.value),
    font_size: Number(element<HTMLSelectElement>("font-size").value),
    text_metrics: atlas ? {width:atlas.width,height:atlas.height,advance:atlas.width,baseline:atlas.baseline,
      densities:[..." .:-=+*#%@"].map(char=>{const mask=atlas.candidates.find(c=>c.text===char)!.mask;return mask.reduce((a,b)=>a+b,0)/mask.length;})} : textMetrics(Number(element<HTMLSelectElement>("font-size").value)),
    shape_atlas: atlas,
    layout: adaptive ? "adaptive" : "fixed",
    adaptive_metrics: adaptive ? Object.fromEntries([8, 16, 32].map((size) => [String(size), squareTextMetrics(size)])) : undefined,
    use_edge: edges.checked && (reference || Number(edgeAmount.value) > 0),
    edge_method: element<HTMLSelectElement>("edge-method").value as RenderSettings["edge_method"],
    edge_min_pixels: Number(element<HTMLInputElement>("edge-min-pixels").value),
    // Detection stays fixed; this control changes only gentle tonal emphasis.
    edge_sensitivity: 50,
    edge_amount: reference ? 100 : Number(edgeAmount.value),
    use_color: element<HTMLInputElement>("color").checked,
    background_mode: element<HTMLSelectElement>("background-mode").value as "solid" | "source",
    background, foreground: rgb[0] * .299 + rgb[1] * .587 + rgb[2] * .114 < 128 ? "#ffffff" : "#242424",
    ...savedMetrics,
  };
}
function updateControls(): void {
  element<HTMLInputElement>("shape-matching").disabled = element<HTMLSelectElement>("layout").value === "adaptive";
  element<HTMLSelectElement>("font-size").disabled = element<HTMLSelectElement>("layout").value === "adaptive";
  element<HTMLOutputElement>("columns-value").value = columns.value;
  element<HTMLOutputElement>("edge-value").value = edgeAmount.value;
  edgeAmount.disabled = !edges.checked;
  element<HTMLSelectElement>("edge-method").disabled = !edges.checked;
  const reference = element<HTMLSelectElement>("edge-method").value === "acerola";
  element("reference-edge-controls").hidden = !reference;
  element("tonal-edge-controls").hidden = reference;
  element<HTMLInputElement>("edge-min-pixels").disabled = !edges.checked;
  element<HTMLOutputElement>("edge-min-value").value = element<HTMLInputElement>("edge-min-pixels").value;
  element("edge-help").hidden = !edges.checked || element<HTMLSelectElement>("edge-method").value !== "tiny";
  const automatic = element<HTMLSelectElement>("background-mode").value === "source";
  element("solid-background").hidden = automatic;
  element("background-help").hidden = !automatic;
}
function schedule(immediate = false): void {
  updateControls();
  if (!source || disposed) return;
  disableDownloads();
  retry.hidden = true;
  const version=++scheduleVersion;
  queue.pause(); // Invalidate an old result before an asynchronous font load.
  void (async()=>{
    try {
      if(element<HTMLInputElement>("shape-matching").checked && !savedMetrics?.shape_atlas) await loadShapeFont();
      if(version!==scheduleVersion || !source || disposed) return;
      queue.request({ pixels: source, sourceVersion, settings: settings() }, immediate);
    } catch(error) {message(error instanceof Error ? error.message : "Could not load ligature font",true); retry.hidden=false;}
  })();
}
function acceptSource(pixels: ImageData, name: string, exampleId?: string): void {
  lastSettings=undefined;
  for (const [id, button] of exampleButtons) button.setAttribute("aria-pressed", String(id === exampleId));
  ++sourceVersion;
  
  source = pixels;
  releaseOutput();
  const canvas = document.createElement("canvas");
  canvas.width = pixels.width; canvas.height = pixels.height;
  canvas.getContext("2d")!.putImageData(pixels, 0, 0);
  original.src = canvas.toDataURL("image/png");
  showingOriginal = false;
  element("filename").textContent = name;
  element("dimensions").textContent = "";
  updateView();
  schedule(true);
}
async function openFile(selected: File, exampleId?: string): Promise<void> {
  exampleDownload?.abort();
  exampleDownload = undefined;
  const version = ++fileVersion;
  if (selected.size > 20 * 1024 * 1024) { message("Choose an image smaller than 20 MB.", true); return; }
  // Stop old scheduling before decoding; late results cannot replace a new source.
  queue.pause();
  disableDownloads();
  let bitmap: ImageBitmap | undefined;
  message("Reading image…");
  try {
    bitmap = await createImageBitmap(selected);
    if (version !== fileVersion || disposed) return;
    if (bitmap.width * bitmap.height > 40_000_000) throw new Error("Choose an image smaller than 40 megapixels.");
    const scale = Math.min(1, 1200 / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(bitmap.width * scale));
    canvas.height = Math.max(1, Math.round(bitmap.height * scale));
    const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    acceptSource(ctx.getImageData(0, 0, canvas.width, canvas.height), selected.name, exampleId);
  } catch (error) {
    if (version === fileVersion) { message(error instanceof Error ? error.message : "Could not open image", true); retry.hidden = !source; }
  } finally { bitmap?.close(); }
}
element("choose").addEventListener("click", () => file.click());
file.addEventListener("change", () => { const selected = file.files?.[0]; file.value = ""; if (selected) void openFile(selected); });

function sample(): void {
  exampleDownload?.abort();
  exampleDownload = undefined;
  ++fileVersion;
  const canvas = document.createElement("canvas");
  canvas.width = 800; canvas.height = 560;
  const ctx = canvas.getContext("2d")!;
  const sky = ctx.createLinearGradient(0, 0, 0, 560);
  sky.addColorStop(0, "#73a2bb"); sky.addColorStop(1, "#e4d8b7");
  ctx.fillStyle = sky; ctx.fillRect(0, 0, 800, 560);
  ctx.fillStyle = "#f6e9b6"; ctx.beginPath(); ctx.arc(590, 120, 48, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#667b89"; ctx.beginPath(); ctx.moveTo(0, 400); ctx.lineTo(230, 180); ctx.lineTo(400, 370); ctx.lineTo(540, 220); ctx.lineTo(800, 410); ctx.fill();
  ctx.fillStyle = "#f0ece1"; ctx.beginPath(); ctx.moveTo(157, 250); ctx.lineTo(230, 180); ctx.lineTo(300, 257); ctx.lineTo(244, 238); ctx.lineTo(215, 258); ctx.fill();
  ctx.fillStyle = "#334f4f"; ctx.beginPath(); ctx.moveTo(0, 470); ctx.lineTo(330, 295); ctx.lineTo(630, 500); ctx.lineTo(800, 350); ctx.lineTo(800, 560); ctx.lineTo(0, 560); ctx.fill();
  acceptSource(ctx.getImageData(0, 0, 800, 560), "Sample landscape");
}
element("sample").addEventListener("click", sample);

async function loadExample(example: Example): Promise<void> {
  exampleDownload?.abort();
  const controller = new AbortController();
  exampleDownload = controller;
  const version = ++fileVersion;
  queue.pause();
  disableDownloads();
  message(`Loading ${example.label}…`);
  try {
    const response = await fetch(exampleUrl(example.file), { signal: controller.signal });
    if (!response.ok) throw new Error(`Could not load ${example.label} (${response.status})`);
    const blob = await response.blob();
    if (version !== fileVersion || disposed) return;
    await openFile(new File([blob], example.file, { type: blob.type }), example.id);
  } catch (error) {
    if (controller.signal.aborted || version !== fileVersion || disposed) return;
    message(error instanceof Error ? error.message : "Could not load example", true);
  } finally {
    if (exampleDownload === controller) exampleDownload = undefined;
  }
}
for (const example of EXAMPLES) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "example-button";
  button.setAttribute("aria-pressed", "false");
  button.title = example.label;
  const thumbnail = document.createElement("img");
  thumbnail.src = exampleUrl(example.thumbnail).href;
  thumbnail.alt = ""; // The button's visible label supplies its accessible name.
  thumbnail.width = thumbnail.height = 56;
  thumbnail.decoding = "async";
  const label = document.createElement("span");
  label.textContent = example.label;
  button.append(thumbnail, label);
  button.addEventListener("click", () => { void loadExample(example); });
  exampleButtons.set(example.id, button);
  element("examples").append(button);
}
form.addEventListener("submit", (event) => event.preventDefault());
form.addEventListener("input", event => {
  if(["font-size","layout","shape-matching"].includes((event.target as HTMLElement).id)) savedMetrics=undefined;
  schedule();
});
element("reset").addEventListener("click", () => { savedMetrics=undefined; form.reset(); schedule(); });
retry.addEventListener("click", () => schedule(true));
cancel.addEventListener("click", () => {
  ++scheduleVersion;
  exampleDownload?.abort();
  exampleDownload = undefined;
  ++fileVersion;
  
  queue.cancel();
  renderer.stop();
  message("Paused. Change a setting to resume.");
  retry.hidden = !source;
});
element("view-ascii").addEventListener("click", () => { showingOriginal = false; updateView(); });
element("view-original").addEventListener("click", () => { showingOriginal = true; updateView(); });
function setZoom(actual: boolean): void {
  actualSize = actual;
  stage.classList.toggle("actual", actual);
  element("fit").setAttribute("aria-pressed", String(!actual));
  element("actual").setAttribute("aria-pressed", String(actual));
  fitText();
  stage.scrollTo(0, 0);
}
element("fit").addEventListener("click", () => setZoom(false));
element("actual").addEventListener("click", () => setZoom(true));
let dragDepth = 0;
stage.addEventListener("dragenter", (event) => { event.preventDefault(); ++dragDepth; stage.classList.add("dragging"); });
stage.addEventListener("dragover", (event) => event.preventDefault());
stage.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; stage.classList.remove("dragging"); } });
stage.addEventListener("drop", (event) => {
  event.preventDefault(); dragDepth = 0; stage.classList.remove("dragging");
  const selected = event.dataTransfer?.files[0]; if (selected) void openFile(selected);
});
function download(url: string | undefined, name: string): void {
  if (!url) return;
  const link = document.createElement("a"); link.href = url; link.download = name; link.click();
}
png.addEventListener("click", async () => {
  if (!rendered) return;
  png.disabled = true;
  try {
    const image = await exportTextPng(rendered, renderedFontSize);
    const url = URL.createObjectURL(image);
    download(url, "ascii-art.png");
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) { message(error instanceof Error ? error.message : "PNG export failed", true); }
  finally { png.disabled = !downloadsReady; }
});
text.addEventListener("click", () => download(textUrl, "ascii-art.txt"));
function downloadBlob(blob: Blob, name: string): void {
  const url=URL.createObjectURL(blob); download(url,name); setTimeout(()=>URL.revokeObjectURL(url),1000);
}
element("export-options").addEventListener("click",()=>downloadBlob(new Blob([exportOptions(captureOptions(form))],{type:"text/plain"}),"ascii-options.txt"));
element("save-project").addEventListener("click",()=>{
  if(!source) return;
  try {
    if(!downloadsReady || !lastSettings) throw new Error("Wait for rendering to finish before saving the project");
    const metrics=lastSettings;
    downloadBlob(new Blob([encodeProject(source,captureOptions(form),metrics)],{type:"application/json"}),"ascii-project.json");
  } catch(error) {message(error instanceof Error ? error.message : "Could not save project",true);}
});
const projectFile=element<HTMLInputElement>("project-file");
element("load-project").addEventListener("click",()=>projectFile.click());
projectFile.addEventListener("change",async()=>{
  const selected=projectFile.files?.[0]; projectFile.value=""; if(!selected) return;
  try {
    if(selected.size>30_000_000) throw new Error("Project file exceeds 30 MB");
    const project=decodeProject(await selected.text());
    restoreOptions(form,project.options);
    queue.cancel(); renderer.stop(); ++scheduleVersion;
    exampleDownload?.abort(); ++fileVersion;
    savedMetrics=project.metrics;
    acceptSource(new ImageData(project.pixels,project.width,project.height),"Restored project");
  } catch(error) {message(error instanceof Error ? error.message : "Could not load project",true);}
});
window.addEventListener("pagehide", () => { exampleDownload?.abort(); disposed = true;  queue.cancel(); renderer.stop(); releaseOutput(); });
// BFCache restoration should remain usable after releasing the worker/resources.
window.addEventListener("pageshow", (event) => { if (event.persisted) { disposed = false; schedule(true); } });
updateControls();

// Start with a real local sample, then automatically render every accepted edit.
sample();
