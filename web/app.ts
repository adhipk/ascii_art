import {pursuitLayers,visiblePursuitLayers,scaleRangeVisibility} from "./pursuit-layers";
import { IterationHistory } from "./iteration-history";
import { glyphBudgetTimeline, frameAtGlyphBudget, budgetHistoryKey } from "./glyph-budget-history";
import { FIT_PRESETS } from "./fit-presets";
import { debugArtifactPayload, debugStatePayload } from "./debug-artifact";
import {instrumentEvent,instrumentState} from "./instrumentation";
import {allocateLayerBudgets} from "./layer-budgets";
import { CHARACTER_PRESETS, splitGlyphs, uniqueGlyphs, defaultPursuitScales, defaultScaleCharacters, normalizeScaleCharacters, type ScaleCharacters, type VocabularyProfile } from "./scale-characters";
import { pursuitFrame } from "./pursuit";
import { textMetrics, squareTextMetrics, pursuitMetrics, textPreview, paperBackground, exportTextPng, FONT_FAMILY, PURSUIT_CHARACTERS } from "./text-view";
import { loadExamples, exampleUrl, type Example } from "./examples";
import { RenderQueue } from "./render-queue";
import { Renderer, type RenderInput, type RenderResult, type RenderSettings } from "./renderer";

function element<T extends Element = HTMLElement>(id: string): T {
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
const history = new IterationHistory(() => { if (rendered?.pursuit) showIteration(); });
let budgetTimeline: ReturnType<typeof glyphBudgetTimeline> = [];
let computedHistoryKey: string | undefined;
let renderedInput: RenderInput | undefined;
const characterPresets:Record<string,{label:string;characters:string}>={...CHARACTER_PRESETS};
let scaleCharacterOverrides: ScaleCharacters = {};
let scaleEnabled:Record<string,boolean>={};
let scaleEditorKey="";
function selectedResult(): RenderResult | undefined { return rendered ? visiblePursuitLayers(pursuitFrame(rendered, history.index),scaleEnabled) : undefined; }
function visibleScaleSizes():number[] {return [...(rendered?.pursuit?.scales??[])].sort((first,second)=>first-second);}
function syncScaleRange():void {
  if(!rendered?.pursuit)return;
  const sizes=visibleScaleSizes(),visible=sizes.filter(size=>scaleEnabled[String(size)]!==false);
  for(const [bound,size] of [["min",visible[0]??sizes[0]],["max",visible.at(-1)??sizes.at(-1)]] as const){
    const input=element<HTMLInputElement>(`visible-scale-${bound}`);
    input.max=String(Math.max(0,sizes.length-1));input.value=String(sizes.indexOf(size!));input.disabled=sizes.length<2;
    input.setAttribute("aria-valuetext",`${size}px`);
    element<HTMLOutputElement>(`visible-scale-${bound}-value`).value=`${size}px`;
  }
  const layers=pursuitLayers(pursuitFrame(rendered,history.index),scaleEnabled);
  element("visible-scale-ticks").replaceChildren(...sizes.map((size,index)=>{
    const option=document.createElement("option");option.value=String(index);option.label=`${size}px`;return option;
  }));
  element("visible-scale-checkpoints").replaceChildren(...sizes.map(size=>{
    const button=document.createElement("button");button.type="button";button.dataset.scale=String(size);
    button.textContent=`${size}px`;button.title=`Isolate ${size}px · ${layers[String(size)]?.glyphs.length??0} glyphs`;
    button.setAttribute("aria-pressed",String(scaleEnabled[String(size)]!==false));return button;
  }));
  const visibleCount=Object.values(layers).filter(layer=>layer.visible).reduce((sum,layer)=>sum+layer.glyphs.length,0);
  element("visible-scale-summary").textContent=`${visibleCount} / ${rendered.pursuit.frames[history.index].glyphs.length} glyphs visible · ${visible.length?visible.map(size=>`${size}px`).join(", "):"all scales hidden"}`;
}
function applyScaleRange(minimum:number,maximum:number):void {
  Object.assign(scaleEnabled,scaleRangeVisibility(visibleScaleSizes(),minimum,maximum));
  for(const size of visibleScaleSizes()){
    const checkbox=document.getElementById(`scale-enabled-${size}`) as HTMLInputElement|null;
    if(checkbox)checkbox.checked=scaleEnabled[String(size)];
    const characters=document.getElementById(`scale-characters-${size}`) as HTMLInputElement|null;
    if(characters)updateScaleCharacterCount(characters);
  }
  showIteration();
}
function captureDebugState():string {
  const input=renderedInput,result=rendered;
  const pendingControls=Object.fromEntries([...form.querySelectorAll<HTMLInputElement|HTMLSelectElement>("input[id],select[id]")].map(node=>
    [node.id,node instanceof HTMLInputElement&&node.type==="checkbox"?node.checked:node.value]));
  if(!input||!result)return JSON.stringify({format:"ascii-debug-state-v1",sourceVersion,pendingControls,rendered:false});
  return debugStatePayload({source:{width:input.pixels.width,height:input.pixels.height,sourceVersion:input.sourceVersion},
    runSettings:input.settings,pendingControls,result,selectedIteration:history.index,layerVisibility:{...scaleEnabled}});
}
function publishDebugState():void {instrumentState(()=>JSON.parse(captureDebugState()));}
function showIteration(): void {
  publishDebugState();
  if (!rendered?.pursuit) return;
  const frame = rendered.pursuit.frames[history.index];
  if (!frame) return;
  syncScaleRange();
  for(const row of element("scale-convergence").querySelectorAll<HTMLTableRowElement>("tr"))
    row.classList.toggle("active-stage",Number(row.dataset.stage)===frame.stage);
  output.replaceChildren(textPreview(selectedResult()!));
  const residualCanvas = element<HTMLCanvasElement>("residual-preview");
  residualCanvas.width = frame.residualWidth; residualCanvas.height = frame.residualHeight;
  const residualPixels = new Uint8ClampedArray(frame.residual.length * 4);
  for (let i = 0; i < frame.residual.length; ++i) {
    const signed = (frame.residual[i] - 128) / 127, magnitude = Math.min(1, Math.abs(signed) * 4);
    residualPixels[i * 4] = signed >= 0 ? 255 : Math.round(255 * (1 - magnitude));
    residualPixels[i * 4 + 1] = Math.round(255 * (1 - magnitude));
    residualPixels[i * 4 + 2] = signed <= 0 ? 255 : Math.round(255 * (1 - magnitude));
    residualPixels[i * 4 + 3] = 255;
  }
  residualCanvas.getContext("2d")!.putImageData(new ImageData(residualPixels, frame.residualWidth, frame.residualHeight), 0, 0);
  updateView(); fitText();
  element<HTMLInputElement>("iteration").value = String(history.index);
  element<HTMLOutputElement>("iteration-value").value = `${history.index} / ${history.length - 1}`;
  element<HTMLInputElement>("history-glyph-budget").value=String(frame.glyphs.length);
  element<HTMLOutputElement>("history-budget-value").value=`${frame.glyphs.length} / ${budgetTimeline.at(-1)?.count??0}`;
  element<HTMLButtonElement>("budget-loop").textContent=history.playing ? "Pause" : "Play";
  element<HTMLButtonElement>("budget-loop").setAttribute("aria-label",history.playing ? "Pause glyph playback" : "Play glyph loop");
  element<HTMLButtonElement>("budget-loop").disabled=budgetTimeline.length<2;
  element("iteration-metrics").textContent = `${rendered.pursuit.blurStrength ? "Multiscale objective" : "Sharp objective"} ${frame.energy.toPrecision(7)} · raw pixel ${frame.pixelError.toFixed(5)} · raw contour ${frame.contourError.toFixed(5)} · overlap cost ${((rendered.pursuit.overlapPenalty??0)*(frame.overlapError??0)).toFixed(5)} · repeat cost ${((rendered.pursuit.repeatPenalty??0)*(frame.repeatError??0)).toFixed(5)} · ${frame.glyphs.length} glyphs · ${frame.diversity} distinct · ink overlap ≤ ${Math.round(rendered.pursuit.overlapLimit * 100)}%`;
  element("iteration-scales").textContent = "Candidates by analysis size: " + Object.entries(frame.scaleCandidates ?? {}).map(([size, count]) => `${size}px: ${count} (${rendered!.pursuit!.kernelCounts[size]} kernels)`).join(" · ");
  if(rendered.pursuit.schedule==="independent"){
    element("iteration-metrics").textContent="Mean independent-layer objective · "+element("iteration-metrics").textContent;
    element("iteration-scales").textContent+=" · metrics average the independent fits; error map shows the active layer, not the composite";
  }
  element("iteration-action").textContent = `${frame.activeScale ? `Pass ${frame.stage+1} · ${frame.activeScale}px · round ${(frame.searchRound??0)+1} · ` : ""}${frame.action} · Δ error ${history.index > 0 ? (rendered.pursuit.frames[history.index - 1].energy - frame.energy).toExponential(3) : "—"} · ${frame.candidates.toLocaleString()} candidates evaluated`;
  element("dimensions").textContent = `${selectedResult()!.tiles!.length} visible / ${frame.glyphs.length} fitted glyphs · step ${history.index}`;
  const curve = element<SVGElement>("energy-curve");
  const energies = rendered.pursuit.frames.map(f => f.energy);
  const min = Math.min(...energies), max = Math.max(...energies);
  const curveY = (energy: number): number => 52 - (energy - min) * 48 / Math.max(max - min, 1e-12);
  curve.setAttribute("aria-label", `Total objective including overlap cost from ${min.toPrecision(5)} to ${max.toPrecision(5)}; lower is better within this run, not a perceptual quality score`);
  const points = rendered.pursuit.frames.map((f, i) => `${4 + i * 216 / Math.max(1, history.length - 1)},${curveY(f.energy)}`).join(" ");
  curve.replaceChildren();
  const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  line.setAttribute("points", points); line.setAttribute("fill", "none"); line.setAttribute("stroke", "#2c6fd3"); line.setAttribute("stroke-width", "2");
  const marker = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  marker.setAttribute("cx", String(4 + history.index * 216 / Math.max(1, history.length - 1)));
  marker.setAttribute("cy", String(curveY(frame.energy))); marker.setAttribute("r", "3"); marker.setAttribute("fill", "#17212d");
  const boundaries=new Set<number>();
  for(const pass of rendered.pursuit.stages??[]){
    if(!pass.startFrame||boundaries.has(pass.startFrame))continue;
    boundaries.add(pass.startFrame);
    const separator=document.createElementNS("http://www.w3.org/2000/svg","line");
    const x=String(4+pass.startFrame*216/Math.max(1,history.length-1));
    separator.setAttribute("x1",x);separator.setAttribute("x2",x);separator.setAttribute("y1","4");separator.setAttribute("y2","52");
    separator.setAttribute("stroke","#aeb7c5");separator.setAttribute("stroke-dasharray","2 2");
    curve.append(separator);
  }
  curve.append(line, marker);
}
let actualSize = false;
let textUrl: string | undefined;
let showingOriginal = false;
let downloadsReady = false;
let disposed = false;
let exampleDownload: AbortController | undefined;
const exampleButtons = new Map<string, HTMLButtonElement>();
let examples: Example[] = [];
let catalogSignature = "";
let selectedExampleId: string | undefined;
let refreshingExamples = false;

function message(value: string, error = false): void {
  status.textContent = value;
  status.classList.toggle("error", error);
}
function disableDownloads(): void { downloadsReady = false; png.disabled = text.disabled = true; }
function releaseOutput(): void {
  history.pause();
  element("iteration-lab").hidden = true;
  element("inspect-panel").hidden = true;
  if (textUrl) URL.revokeObjectURL(textUrl);
  rendered = undefined;
  renderedInput=undefined;
  element<HTMLButtonElement>("copy-debug").disabled=true;
  element<HTMLButtonElement>("copy-debug-state").disabled=true;
  textUrl = undefined;
  output.hidden = true;
  output.replaceChildren();
  disableDownloads();
}
function updateView(): void {
  const hasOutput = Boolean(rendered);
  const compare = Boolean(rendered?.pursuit) && element<HTMLInputElement>("compare-original").checked && !showingOriginal;
  stage.classList.toggle("comparing", compare);
  original.hidden = !source || (!showingOriginal && hasOutput && !compare);
  const residualView = Boolean(rendered?.pursuit) && element<HTMLInputElement>("show-residual").checked;
  output.hidden = showingOriginal || !hasOutput || residualView;
  element<HTMLCanvasElement>("residual-preview").hidden = showingOriginal || !hasOutput || !residualView;
  element("empty").hidden = Boolean(source);
  element("view-ascii").setAttribute("aria-pressed", String(!showingOriginal));
  element("view-original").setAttribute("aria-pressed", String(showingOriginal));
}
const renderer = new Renderer((value, id) => { if (queue.isCurrent(id)){message(value);instrumentEvent("render-progress",{id,message:value});} });
const queue = new RenderQueue<RenderInput, RenderResult>({
  render: (input, id) => {instrumentEvent("render-start",{id,sourceVersion:input.sourceVersion});return renderer.render(input, id);},
  result: (result, input, id) => { renderedInput=input; computedHistoryKey=budgetHistoryKey(input.settings,input.sourceVersion); displayResult(result, id, input.settings.font_size);instrumentEvent("render-complete",{id,frames:result.pursuit?.frames.length});publishDebugState(); },
  error: (error) => {
    instrumentEvent("render-error",{message:error instanceof Error?error.message:String(error)});publishDebugState();
    element("fit-state").textContent="Fit failed · adjust settings or retry";
    message(error instanceof Error ? error.message : "Could not render image", true);
    retry.hidden = false;
  },
  state: (state) => {
    element<HTMLButtonElement>("run-fit").disabled=state!=="idle";
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
  const availableWidth = stage.classList.contains("comparing") ? (stage.clientWidth - inset - 16) / 2 : stage.clientWidth - inset;
  const scale = actualSize ? 1 : Math.min(availableWidth / rendered.width, (stage.clientHeight - inset) / rendered.height);
  output.style.width = `${rendered.width * scale}px`;
  output.style.height = `${rendered.height * scale}px`;
  const residualCanvas = element<HTMLCanvasElement>("residual-preview");
  residualCanvas.style.width = output.style.width; residualCanvas.style.height = output.style.height;
  if (rendered.tiles) {
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
    element<HTMLButtonElement>("copy-debug").disabled=false;
    element<HTMLButtonElement>("copy-debug-state").disabled=false;
    element("dimensions").textContent = result.tiles ? `${result.tiles.length} glyphs · 8 / 16 / 32 px` : `${result.columns} × ${result.rows} characters`;
    text.hidden = Boolean(result.pursuit);
    element("iteration-lab").hidden = !result.pursuit;
    element("inspect-panel").hidden = !result.pursuit;
    element("fit-state").textContent="Complete · cached frames are ready to inspect";
    element<HTMLButtonElement>("run-fit").textContent="Run again";
    if (result.pursuit) {
      element("scale-character-summary").textContent="Last run used: "+Object.entries(result.pursuit.charactersByScale??{}).sort(([a],[b])=>Number(b)-Number(a)).map(([size,chars])=>`${size}px: ${splitGlyphs(chars).length} character${splitGlyphs(chars).length===1?"":"s"}`).join(" · ");
      const rows=result.pursuit.stages.map(stage=>{
        const row=document.createElement("tr");row.dataset.stage=String(stage.index);
        const refinements=result.pursuit!.frames.slice(stage.startFrame+1,stage.endFrame+1).filter(frame=>frame.action.startsWith("Refine ")).length;
        for(const value of [stage.size===null?"All":`${stage.size}px`,`${stage.addedGlyphs} add / ${refinements} refine`,String(stage.searchRounds),stage.converged?"Settled":"Budget"]) {
          const cell=document.createElement("td");cell.textContent=value;row.append(cell);
        }
        return row;
      });
      element("scale-convergence").replaceChildren(...rows);
      budgetTimeline=glyphBudgetTimeline(result.pursuit.frames);
      element<HTMLInputElement>("history-glyph-budget").min=String(budgetTimeline[0]?.count??0);
      element<HTMLInputElement>("history-glyph-budget").max=String(budgetTimeline.at(-1)?.count??0);
      element<HTMLInputElement>("history-glyph-budget").disabled=budgetTimeline.length<2;
      element<HTMLInputElement>("iteration").max = String(result.pursuit.frames.length - 1);
      element("iteration-stop").textContent = `${result.pursuit.gpuResident?"GPU painting loop":result.pursuit.backend.toUpperCase()}${result.pursuit.backendReason ? " (" + result.pursuit.backendReason + ")" : ""} · ${result.pursuit.stopReason} · fit ${result.pursuit.analysisWidth} × ${result.pursuit.analysisHeight} · ${(result.pursuit.elapsedMs / 1000).toFixed(2)} s`;
      history.load(result.pursuit.frames.length);
    } else history.pause();
    text.textContent = result.tiles ? "Text · fixed grid" : "Text";
    text.title = result.tiles ? "Plain text uses the separate fixed 8×8 grid. PNG preserves mixed sizes." : "Download character grid";
    element("runtime-note").hidden = true;
    message("Up to date");
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
  const layout = element<HTMLSelectElement>("layout").value as "fixed" | "adaptive" | "pursuit";
  const adaptive = layout === "adaptive";
  const scales=defaultPursuitScales(Number(element<HTMLSelectElement>("fitting-resolution").value));
  const profile=element<HTMLSelectElement>("glyph-vocabulary").value as VocabularyProfile;
  const perScale=layout==="pursuit" ? normalizeScaleCharacters(Object.fromEntries(scales.filter(size=>profile==="shape-detail" || Object.hasOwn(scaleCharacterOverrides,String(size))).map(size=>[String(size),scaleCharacterOverrides[String(size)]??defaultScaleCharacters(size,scales[0],profile)])),scales) : {};
  const atlasCharacters=[...new Set([PURSUIT_CHARACTERS,...Object.values(perScale)].flatMap(splitGlyphs))];
  return {
    columns: Number(columns.value),
    font_size: Number(element<HTMLSelectElement>("font-size").value),
    text_metrics: textMetrics(Number(element<HTMLSelectElement>("font-size").value)),
    layout,
    pursuit_backend: element<HTMLSelectElement>("search-backend").value as "auto" | "cpu" | "webgpu",
    pursuit_max_overlap: Number(element<HTMLInputElement>("overlap-limit").value) / 100,
    pursuit_solid_brush: element<HTMLInputElement>("solid-brush").checked,
    pursuit_overlap_penalty: Number(element<HTMLInputElement>("overlap-penalty").value),
    pursuit_repeat_penalty: Number(element<HTMLInputElement>("repeat-penalty").value),
    pursuit_penalty_min_size:24,
    pursuit_vocabulary: profile,
    pursuit_characters_by_scale: layout === "pursuit" && Object.keys(perScale).length ? perScale : undefined,
    pursuit_metrics: layout === "pursuit" ? pursuitMetrics(atlasCharacters) : undefined,
    pursuit_resolution: Number(element<HTMLSelectElement>("fitting-resolution").value),
    pursuit_effort: element<HTMLSelectElement>("search-effort").value as "fast" | "thorough",
    pursuit_budget: Number(element<HTMLInputElement>("glyph-budget").value),
    pursuit_beam: Number(element<HTMLSelectElement>("beam-width").value),
    pursuit_coarse_threshold: Number(element<HTMLInputElement>("coarse-threshold").value)/100,
    pursuit_coarse_rounds: Number(element<HTMLInputElement>("coarse-rounds").value),
    pursuit_schedule: element<HTMLSelectElement>("scale-schedule").value as "coarse-to-fine" | "all" | "independent",
    pursuit_blur: Number(element<HTMLInputElement>("kernel-blur").value),
    pursuit_detail_sharpen: element<HTMLSelectElement>("scale-schedule").value==="independent"?Number(element<HTMLInputElement>("detail-sharpen").value):0,
    pursuit_edge_weight: Number(element<HTMLInputElement>("contour-weight").value),
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
  };
}
function updateScaleCharacterInputs(): void {
  const sizes=defaultPursuitScales(Number(element<HTMLSelectElement>("fitting-resolution").value));
  const profile=element<HTMLSelectElement>("glyph-vocabulary").value as VocabularyProfile;
  const key=JSON.stringify([sizes,profile]);
  if(key===scaleEditorKey) return;
  scaleEditorKey=key;
  element("scale-character-inputs").replaceChildren(...sizes.map(size=>{
    const wrapper=document.createElement("div");wrapper.className="scale-character-row";
    const label=document.createElement("label");label.htmlFor=`scale-characters-${size}`;label.textContent=`${size}px`;
    const count=document.createElement("output");count.id=`scale-character-count-${size}`;
    const enabled=document.createElement("input");enabled.type="checkbox";enabled.id=`scale-enabled-${size}`;enabled.checked=scaleEnabled[String(size)]!==false;enabled.setAttribute("aria-label",`Show ${size}px layer`);
    enabled.addEventListener("input",()=>{scaleEnabled[String(size)]=enabled.checked;updateScaleCharacterCount(input,count);});
    const toggle=document.createElement("label");toggle.className="scale-enabled";toggle.append(enabled,document.createTextNode("Visible"));
    const heading=document.createElement("div");heading.className="range-label";heading.append(label,toggle,count);
    const input=document.createElement("input");input.type="text";input.id=label.htmlFor;input.dataset.scale=String(size);
    input.spellcheck=false;input.autocomplete="off";input.setAttribute("autocapitalize","off");
    input.placeholder="Empty: skip this scale";
    input.value=scaleCharacterOverrides[String(size)]??defaultScaleCharacters(size,sizes[0],profile);
    input.setAttribute("aria-describedby",count.id);
    const presets=document.createElement("select");presets.id=`scale-preset-${size}`;presets.setAttribute("aria-label",`${size}px character preset`);
    for(const [value,text] of [["custom","Custom"],...Object.entries(characterPresets).map(([key,preset])=>[key,preset.label])]) {const option=document.createElement("option");option.value=value;option.textContent=text;presets.append(option);}
    const presetName=document.createElement("input");presetName.type="text";presetName.id=`scale-preset-name-${size}`;presetName.placeholder="Preset name";presetName.setAttribute("aria-label",`${size}px preset filename`);
    const syncPreset=()=>{presets.value=Object.entries(characterPresets).find(([,preset])=>preset.characters===input.value)?.[0]??"custom";presetName.value=presets.value==="custom"?`custom-${size}px`:presets.value;};syncPreset();
    input.addEventListener("input",syncPreset);
    presets.addEventListener("change",()=>{const preset=characterPresets[presets.value];if(!preset)return;input.value=preset.characters;scaleCharacterOverrides[String(size)]=input.value;updateScaleCharacterCount(input,count);syncPreset();pendingFit();});
    const reset=document.createElement("button");reset.type="button";reset.className="text-button";reset.textContent="Reset to default";
    reset.addEventListener("click",()=>{delete scaleCharacterOverrides[String(size)];delete scaleEnabled[String(size)];enabled.checked=true;input.value=defaultScaleCharacters(size,sizes[0],profile);updateScaleCharacterCount(input,count);syncPreset();pendingFit();});
    const save=document.createElement("button");save.type="button";save.className="text-button";save.textContent="Save .txt";
    save.addEventListener("click",()=>{const name=(presetName.value.trim()||`custom-${size}px`).replace(/\.txt$/i,"").replace(/[\\/]/g,"-");const url=URL.createObjectURL(new Blob([input.value],{type:"text/plain;charset=utf-8"}));download(url,`${name}.txt`);setTimeout(()=>URL.revokeObjectURL(url),1000);});
    const load=document.createElement("button");load.type="button";load.className="text-button";load.textContent="Load .txt";
    const file=document.createElement("input");file.type="file";file.accept=".txt,text/plain";file.hidden=true;file.setAttribute("aria-label",`${size}px load character preset`);
    load.addEventListener("click",()=>file.click());
    file.addEventListener("change",async()=>{const selected=file.files?.[0];if(!selected)return;try {
      const name=selected.name.replace(/\.txt$/i,""),characters=uniqueGlyphs((await selected.text()).replace(/^\uFEFF/,"").replace(/[\r\n]/g,"")).join("");
      const key=`file:${name}`;characterPresets[key]={label:name,characters};scaleCharacterOverrides[String(size)]=characters;
      scaleEditorKey="";pendingFit();
      const nameField=element<HTMLInputElement>(`scale-preset-name-${size}`);nameField.value=name;
    }catch(error){status.textContent=`Could not load preset: ${error instanceof Error?error.message:String(error)}`;}});
    const filenameLabel=document.createElement("label");filenameLabel.htmlFor=presetName.id;filenameLabel.className="preset-filename";filenameLabel.textContent="File name (.txt)";
    const actions=document.createElement("div");actions.className="character-preset-actions";actions.append(reset,load,save);
    wrapper.append(heading,presets,input,filenameLabel,presetName,actions,file); updateScaleCharacterCount(input,count);return wrapper;
  }));
}
function updateScaleCharacterCount(input: HTMLInputElement,count=element<HTMLOutputElement>(`scale-character-count-${input.dataset.scale}`)): void {
  input.setCustomValidity("");
  input.removeAttribute("aria-invalid");
  const n=uniqueGlyphs(input.value).filter(char=>char.trim().length>0).length;
  count.value=scaleEnabled[input.dataset.scale!]===false?"Hidden":n?`${n} char${n===1?"":"s"}${Object.hasOwn(scaleCharacterOverrides,input.dataset.scale!)?" · custom":""}`:"Skipped";
}
function updateControls(): void {
  updateScaleCharacterInputs();
  const layout = element<HTMLSelectElement>("layout").value;
  element<HTMLSelectElement>("font-size").disabled = layout !== "fixed";
  element("pursuit-controls").hidden = layout !== "pursuit";
  element("pursuit-primary").hidden = layout !== "pursuit";
  for(const input of element("scale-character-inputs").querySelectorAll<HTMLInputElement>("input")) input.disabled=layout!=="pursuit";
  element("workflow-nav").hidden = layout !== "pursuit";
  const independent=element<HTMLSelectElement>("scale-schedule").value==="independent";
  element("budget-caption").textContent=independent?"Maximum glyphs per layer":"Maximum glyphs";
  const layerSizes=defaultPursuitScales(Number(element<HTMLSelectElement>("fitting-resolution").value));
  const activeLayers=layerSizes.filter(size=>element<HTMLInputElement>(`scale-characters-${size}`).value.trim().length>0);
  const baseBudget=Number(element<HTMLInputElement>("glyph-budget").value),allocations=allocateLayerBudgets(layerSizes,activeLayers,baseBudget);
  element("budget-help").textContent=independent?`${layerSizes.map(size=>`${size}px: ${allocations[String(size)]}`).join(" · ")}. Each active layer is capped at ${baseBudget} glyphs. Empty character sets skip their layer; unused budgets are not transferred.`:"Total budget shared across the fitting run.";
  element<HTMLInputElement>("coarse-threshold").disabled=independent;
  element<HTMLInputElement>("coarse-rounds").disabled=independent;
  element<HTMLInputElement>("detail-sharpen").disabled=!independent;
  element("fit-plan").textContent=layout==="pursuit" ?
    `Next run: up to ${baseBudget} glyph additions${independent?" per active layer":""} at ${element<HTMLSelectElement>("fitting-resolution").value}px analysis. Existing preview stays available until you run.` :
    `Next run: ${columns.value} columns using the ${layout} grid.`;
  element("grid-help").hidden = layout === "pursuit";
  element("column-caption").textContent = layout === "pursuit" ? "Display columns" : "Columns";
  element("detector-controls").hidden = layout === "pursuit";
  element<HTMLOutputElement>("budget-value").value = element<HTMLInputElement>("glyph-budget").value;
  element<HTMLOutputElement>("overlap-value").value = element<HTMLInputElement>("overlap-limit").value + "%";
  element<HTMLOutputElement>("overlap-penalty-value").value=element<HTMLInputElement>("overlap-penalty").value;
  element<HTMLOutputElement>("repeat-penalty-value").value=element<HTMLInputElement>("repeat-penalty").value;
  element<HTMLOutputElement>("coarse-threshold-value").value=element<HTMLInputElement>("coarse-threshold").value+"%";
  element<HTMLOutputElement>("coarse-rounds-value").value = element<HTMLInputElement>("coarse-rounds").value;
  element<HTMLOutputElement>("kernel-blur-value").value = element<HTMLInputElement>("kernel-blur").value;
  element<HTMLOutputElement>("detail-sharpen-value").value = element<HTMLInputElement>("detail-sharpen").value;
  element<HTMLOutputElement>("contour-weight-value").value = element<HTMLInputElement>("contour-weight").value;
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
  element<HTMLSelectElement>("background-mode").querySelector('option[value="source"]')!.textContent = layout === "pursuit" ? "From image · base paper" : "From image · gradient";
  element("background-help").textContent = layout === "pursuit" ? "A flat base color sampled from the image; fitted glyphs reconstruct its tones." : "A soft gradient from your image’s colors.";
  element("solid-background").hidden = automatic;
  element("background-help").hidden = !automatic;
}
function schedule(immediate = false): void {
  history.pause();
  updateControls();
  if (!source || disposed) return;
  disableDownloads();
  retry.hidden = true;
  element("fit-state").textContent="Computing a new run…";
  try {queue.request({ pixels: source, sourceVersion, settings: settings() }, immediate);}
  catch(error) {message(error instanceof Error?error.message:"Invalid fitting settings",true);element("fit-state").textContent="Fix character sets before fitting";}
}
function acceptSource(pixels: ImageData, name: string, exampleId?: string): void {
  selectedExampleId = exampleId;
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
async function refreshExamples(): Promise<void> {
  if (disposed || refreshingExamples) return;
  refreshingExamples = true;
  try {
    const catalog = await loadExamples();
    const signature = JSON.stringify(catalog);
    if (signature === catalogSignature) return;
    catalogSignature = signature;
    examples = catalog;
    exampleButtons.clear();
    element("examples").replaceChildren();
for (const example of examples) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "example-button";
  button.setAttribute("aria-pressed", String(example.id === selectedExampleId));
  button.title = example.label;
  const thumbnail = document.createElement("img");
  thumbnail.src = exampleUrl(example.thumbnail).href;
  thumbnail.alt = ""; // The button's visible label supplies its accessible name.
  thumbnail.width = thumbnail.height = 56;
  thumbnail.decoding = "async";
  thumbnail.loading = "lazy";
  const label = document.createElement("span");
  label.textContent = example.label;
  button.append(thumbnail, label);
  button.addEventListener("click", () => { void loadExample(example); });
  exampleButtons.set(example.id, button);
  element("examples").append(button);
}
  } catch (error) {
    if (!catalogSignature) element("examples-note").textContent = "Could not load the examples folder. Upload an image or retry.";
  } finally { refreshingExamples = false; }
}
element("refresh-examples").addEventListener("click", () => void refreshExamples());
setInterval(() => { if (!document.hidden) void refreshExamples(); }, 5000);
function pendingFit(): void {
  queue.cancel(); renderer.stop(); history.pause(); updateControls();
  element("fit-state").textContent="Pending changes · preview shows the previous run";
  element<HTMLButtonElement>("run-fit").textContent="Run fit";
  message("Settings changed · press Run fit");
}
form.addEventListener("submit", (event) => {event.preventDefault();schedule(true);});
form.addEventListener("input", event => {
  const id=(event.target as HTMLElement).id;
  if(id.startsWith("scale-enabled-")) {
    if(rendered?.pursuit)showIteration();
    return;
  }
  if(id.startsWith("scale-characters-")) {
    const input=event.target as HTMLInputElement;scaleCharacterOverrides[input.dataset.scale!]=input.value;updateScaleCharacterCount(input);
  }
  if(id==="fit-preset") {
    for(const [key,value] of Object.entries(FIT_PRESETS[element<HTMLSelectElement>("fit-preset").value]??{}))
      element<HTMLInputElement|HTMLSelectElement>(key).value=value;
  } else if(FIT_PRESETS.balanced[id]!==undefined) element<HTMLSelectElement>("fit-preset").value="custom";
  if(id==="glyph-budget" && form.checkValidity() && rendered?.pursuit && rendered.pursuit.schedule!=="independent" &&
    computedHistoryKey===budgetHistoryKey(settings(),sourceVersion) &&
    Number(element<HTMLInputElement>("glyph-budget").value)<=rendered.pursuit.budget) {
    queue.cancel(); renderer.stop(); history.pause(); updateControls();
    history.seek(frameAtGlyphBudget(budgetTimeline,Number(element<HTMLInputElement>("glyph-budget").value)));
    downloadsReady=true; png.disabled=false;
    message("Replaying precomputed glyph budget");
    element("fit-state").textContent="Using cached frames · no new fitting";
    return;
  }
  pendingFit();
});
element("reset").addEventListener("click", () => { form.reset(); scaleCharacterOverrides={};scaleEnabled={};scaleEditorKey="";pendingFit(); });
element("reset-scale-characters").addEventListener("click",()=>{scaleCharacterOverrides={};scaleEnabled={};scaleEditorKey="";pendingFit();});
element("workflow-nav").addEventListener("click",event=>{
  const link=(event.target as HTMLElement).closest<HTMLAnchorElement>("a");
  if(!link) return;
  const target=document.getElementById(link.hash.slice(1)); if(!target) return;
  event.preventDefault();
  if(innerWidth<=760) target.scrollIntoView({block:"start"});
  else {
    const panel=element("workflow-nav").parentElement!;
    panel.scrollTo({top:target.getBoundingClientRect().top-panel.getBoundingClientRect().top+panel.scrollTop-64});
  }
});
retry.addEventListener("click", () => schedule(true));
cancel.addEventListener("click", () => {
  history.pause();
  exampleDownload?.abort();
  exampleDownload = undefined;
  ++fileVersion;
  
  queue.cancel();
  renderer.stop();
  message("Paused · press Run fit to restart.");
  element("fit-state").textContent="Paused · press Run fit to restart";
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
    const image = await exportTextPng(selectedResult()!, renderedFontSize);
    const url = URL.createObjectURL(image);
    download(url, "ascii-art.png");
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) { message(error instanceof Error ? error.message : "PNG export failed", true); }
  finally { png.disabled = !downloadsReady; }
});
text.addEventListener("click", () => download(textUrl, "ascii-art.txt"));
function blobDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve,reject)=>{
    const reader=new FileReader();reader.onload=()=>resolve(String(reader.result));reader.onerror=()=>reject(new Error("Could not encode debug image"));reader.readAsDataURL(blob);
  });
}
async function captureDebugArtifact(): Promise<{json:string;html:string}> {
  const layerVisibility={...scaleEnabled};
  const input=renderedInput,result=selectedResult(),trace=rendered?.pursuit,index=history.index,fontSize=renderedFontSize;
  if(!input || !result) throw new Error("Render an image before copying a debug artifact");
  const pending=Object.fromEntries([...form.querySelectorAll<HTMLInputElement|HTMLSelectElement>("input[id],select[id]")].map(node=>
    [node.id,node instanceof HTMLInputElement && node.type==="checkbox" ? node.checked : node.value]));
  const canvas=document.createElement("canvas");canvas.width=input.pixels.width;canvas.height=input.pixels.height;
  canvas.getContext("2d")!.putImageData(input.pixels,0,0);
  const original=canvas.toDataURL("image/png");
  const png=await exportTextPng(result,fontSize);
  const pngDimensions=new DataView(await png.slice(16,24).arrayBuffer());
  const output=await blobDataUrl(png);
  const {frames,...traceSummary}=trace??{frames:undefined};
  return debugArtifactPayload({capturedAt:new Date().toISOString(),
    images:{original,output},
    source:{width:input.pixels.width,height:input.pixels.height,sourceVersion:input.sourceVersion},
    output:{layerVisibility,layers:trace?pursuitLayers(pursuitFrame(rendered!,index),layerVisibility):undefined,width:pngDimensions.getUint32(0),height:pngDimensions.getUint32(4),logicalWidth:result.width,logicalHeight:result.height,selectedIteration:trace?index:null,text:result.text,tiles:result.tiles,tileMetrics:result.tileMetrics,colors:result.colors,paper:result.paper},
    runSettings:input.settings,pendingControls:pending,
    trace:trace?{...traceSummary,frameCount:frames!.length,selectedFrame:frames![index],otherFramesOmitted:true}:undefined,
    runtime:{userAgent:navigator.userAgent,devicePixelRatio,viewport:{width:innerWidth,height:innerHeight}}});
}
element("copy-debug-state").addEventListener("click",async()=>{
  const input=renderedInput,result=rendered;
  if(!input||!result)return;
  const button=element<HTMLButtonElement>("copy-debug-state");button.disabled=true;
  try{
    const payload=captureDebugState();
    await navigator.clipboard.writeText(payload);
    message(`Debug state copied · ${(new TextEncoder().encode(payload).length/1024).toFixed(1)} KB · paste with your screenshot`);
  }catch(error){message(error instanceof Error?error.message:"Could not copy debug state",true);}
  finally{button.disabled=!renderedInput;}
});
form.addEventListener("input",publishDebugState);
form.addEventListener("change",publishDebugState);
window.addEventListener("error",event=>instrumentEvent("browser-error",{message:event.message,filename:event.filename,line:event.lineno}));
window.addEventListener("unhandledrejection",event=>instrumentEvent("unhandled-rejection",{message:String(event.reason)}));
instrumentEvent("page-loaded");publishDebugState();
element("copy-debug").addEventListener("click",async()=>{
  const button=element<HTMLButtonElement>("copy-debug");button.disabled=true;
  const payload=captureDebugArtifact();
  try {
    if(!navigator.clipboard) throw new Error("Clipboard is unavailable. Open the app on localhost or HTTPS.");
    if(typeof ClipboardItem!=="undefined" && navigator.clipboard.write) {
      await navigator.clipboard.write([new ClipboardItem({
        "text/html":payload.then(value=>new Blob([value.html],{type:"text/html"})),
        "text/plain":payload.then(value=>new Blob([value.json],{type:"text/plain"})),
      })]);
    } else await navigator.clipboard.writeText((await payload).json);
    message("Debug artifact copied · original, selected output and run configuration");
  } catch(error) {await payload.catch(()=>{});message(error instanceof Error?error.message:"Could not copy debug artifact",true);}
  finally {button.disabled=!renderedInput;}
});
element("iteration").addEventListener("input", () => { const index = Number(element<HTMLInputElement>("iteration").value); history.pause(); history.seek(index); });
for(const bound of ["min","max"] as const)element(`visible-scale-${bound}`).addEventListener("input",()=>{
  const sizes=visibleScaleSizes();if(!sizes.length)return;
  let minimum=Number(element<HTMLInputElement>("visible-scale-min").value),maximum=Number(element<HTMLInputElement>("visible-scale-max").value);
  if(minimum>maximum){if(bound==="min")maximum=minimum;else minimum=maximum;}
  applyScaleRange(sizes[minimum],sizes[maximum]);
});
element("visible-scale-checkpoints").addEventListener("click",event=>{
  const button=(event.target as HTMLElement).closest<HTMLButtonElement>("button[data-scale]");
  if(button){const size=Number(button.dataset.scale);applyScaleRange(size,size);}
});
element("visible-scale-all").addEventListener("click",()=>{
  const sizes=visibleScaleSizes();if(sizes.length)applyScaleRange(sizes[0],sizes.at(-1)!);
});
element("history-glyph-budget").addEventListener("input",()=>{
  const budget=Number(element<HTMLInputElement>("history-glyph-budget").value);
  history.pause(); history.seek(frameAtGlyphBudget(budgetTimeline,budget));
});
element("budget-loop").addEventListener("click",()=>{
  const pause=history.playing; history.pause();
  if(!pause) history.play(budgetTimeline.map(frame=>frame.index),true);
});
element("show-residual").addEventListener("change", () => { updateView(); fitText(); });
element("compare-original").addEventListener("change", () => { updateView(); fitText(); });
element("download-trace").addEventListener("click", () => {
  if (!rendered?.pursuit) return;
  const data = { version: 1, width: rendered.width, height: rendered.height, paper: Array.from(rendered.paper), tileMetrics: rendered.tileMetrics, trace: { ...rendered.pursuit, frames: rendered.pursuit.frames.map(frame => ({...frame, residual: Array.from(frame.residual)})) } };
  const url = URL.createObjectURL(new Blob([JSON.stringify(data)], { type: "application/json" }));
  download(url, "glyph-search-trace.json"); setTimeout(() => URL.revokeObjectURL(url), 1000);
});

window.addEventListener("pagehide", () => { history.pause(); exampleDownload?.abort(); disposed = true;  queue.cancel(); renderer.stop(); releaseOutput(); });
// BFCache restoration should remain usable after releasing the worker/resources.
window.addEventListener("pageshow", (event) => { if (event.persisted) { disposed = false; schedule(true); } });
const startup = new URLSearchParams(location.search);
if (startup.get("lab") === "1") element<HTMLSelectElement>("layout").value = "pursuit";
updateControls();
// Direct lab links start without initializing Python or downloading an edge model.
async function start(): Promise<void> {
  await refreshExamples();
  if (source || disposed) return;
  const requested = startup.get("example") ?? "taylor";
  const startupExample = startup.get("lab") === "1" ? examples.find(example => example.id === requested || example.file.replace(/\.[^.]+$/, "") === requested) : undefined;
  if (startupExample) await loadExample(startupExample); else sample();
}
void start();
