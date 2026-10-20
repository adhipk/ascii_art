import type { RenderSettings } from "./renderer";

export type Options = Record<string,string|boolean>;
type Metrics = Pick<RenderSettings,"text_metrics"|"adaptive_metrics"|"shape_atlas">;
export const PROJECT_VERSION = "ascii-studio-v1";
const schema: Record<string, string[] | [number,number] | "boolean"> = {
  layout:["fixed","adaptive"], columns:[20,180], "font-size":["8","12","16","20"],
  "shape-matching":"boolean", edges:"boolean", "edge-method":["acerola","tiny","dog"],
  "edge-min-pixels":[1,64], "edge-amount":[0,100], color:"boolean",
  "background-mode":["source","solid"], background:["hex"],
};
export function validateOptions(value: unknown): Options {
  if(!value || typeof value!=="object" || Array.isArray(value)) throw new Error("Invalid project options");
  const options=value as Options;
  if(Object.keys(options).length!==Object.keys(schema).length) throw new Error("Project options are incomplete");
  for(const [key,rule] of Object.entries(schema)) {
    const v=options[key];
    const valid=rule==="boolean" ? typeof v==="boolean" : typeof v==="string" && (
      rule[0]==="hex" ? /^#[0-9a-f]{6}$/i.test(v) : typeof rule[0]==="number" ?
        /^\d+$/.test(v) && Number(v)>=rule[0] && Number(v)<=Number(rule[1]) : rule.includes(v as never));
    if(!valid) throw new Error(`Invalid option: ${key}`);
  }
  return options;
}
export function captureOptions(form: HTMLFormElement): Options {
  return validateOptions(Object.fromEntries(Object.keys(schema).map(id=>{
    const node=form.querySelector<HTMLInputElement|HTMLSelectElement>(`#${id}`)!;
    return [id,node instanceof HTMLInputElement && node.type==="checkbox" ? node.checked : node.value];
  })));
}
export function restoreOptions(form: HTMLFormElement, options: Options): void {
  validateOptions(options);
  for(const [key,value] of Object.entries(options)) {
    const node=form.querySelector<HTMLInputElement|HTMLSelectElement>(`#${key}`)!;
    if(typeof value==="boolean") (node as HTMLInputElement).checked=value; else node.value=value;
  }
}
export function exportOptions(options: Options): string {
  validateOptions(options);
  return [`format: ${PROJECT_VERSION}`, "algorithm: coverage-tone-sse-block-bounds-v1", "shape-font: bundled-fira-code",
    ...Object.entries(options).map(([key,value])=>`${key}: ${value}`)].join("\n")+"\n";
}
function validateMetrics(value: unknown, options: Options): Metrics {
  if(!value || typeof value!=="object") throw new Error("Missing measured font metrics");
  const metrics=value as Metrics;
  const checkText=(m: any) => {
    if(!m || !Number.isInteger(m.width) || m.width<1 || m.width>64 || !Number.isInteger(m.height) || m.height<2 || m.height>96 ||
      !Array.isArray(m.densities) || m.densities.length!==10 || m.densities.some((x:unknown)=>typeof x!=="number" || !Number.isFinite(x) || x<0 || x>1) ||
      !Number.isFinite(m.advance) || !Number.isFinite(m.baseline)) throw new Error("Invalid measured font metrics");
  };
  checkText(metrics.text_metrics);
  if(options.layout==="adaptive") for(const size of [8,16,32]) {
    const m=metrics.adaptive_metrics?.[String(size)]; checkText(m);
    if(m?.width!==size || m.height!==size) throw new Error("Invalid adaptive font metrics");
  }
  if(options["shape-matching"] && options.layout==="fixed") {
    const atlas=metrics.shape_atlas;
    if(!atlas || atlas.width!==metrics.text_metrics!.width || atlas.height!==metrics.text_metrics!.height || atlas.size!==Number(options["font-size"]) ||
      !Array.isArray(atlas.candidates) || atlas.candidates.length<2 || atlas.candidates.length>400) throw new Error("Invalid shape atlas");
    for(const c of atlas.candidates) if(!c || ![1,2].includes(c.span) || typeof c.text!=="string" || c.text.length!==c.span || !/^[ -~]+$/.test(c.text) ||
      !Array.isArray(c.mask) || c.mask.length!==atlas.width*atlas.height*c.span || c.mask.some(x=>!Number.isFinite(x)||x<0||x>1)) throw new Error("Invalid shape candidate");
    if(![1,2].every(span=>atlas.candidates.some(c=>c.span===span)) ||
      ![..." .:-=+*#%@/|\\"].every(text=>atlas.candidates.some(c=>c.span===1 && c.text===text)))
      throw new Error("Incomplete shape atlas: missing span, tone-ramp or contour character");
  } else if(metrics.shape_atlas) throw new Error("Unexpected shape atlas");
  // Imported objects may contain render settings beyond this measured data.
  // Only these fields may be spread into the current controls' settings.
  return {
    text_metrics: metrics.text_metrics,
    adaptive_metrics: metrics.adaptive_metrics,
    shape_atlas: metrics.shape_atlas,
  };
}
export function encodeProject(source: {width:number;height:number;data:Uint8ClampedArray}, options: Options, settings: RenderSettings): string {
  const metrics={text_metrics:settings.text_metrics,adaptive_metrics:settings.adaptive_metrics,shape_atlas:settings.shape_atlas};
  validateOptions(options); validateMetrics(metrics,options);
  let binary="";
  for(let i=0;i<source.data.length;i+=8192) binary+=String.fromCharCode(...source.data.subarray(i,i+8192));
  return JSON.stringify({format:PROJECT_VERSION,algorithm:"coverage-tone-sse-block-bounds-v1",options,metrics,
    source:{width:source.width,height:source.height,rgba:btoa(binary)}});
}
export function decodeProject(text: string): {width:number;height:number;pixels:Uint8ClampedArray;options:Options;metrics:Metrics} {
  if(text.length>30_000_000) throw new Error("Project file exceeds 30 MB");
  const value=JSON.parse(text);
  if(value?.format!==PROJECT_VERSION || value.algorithm!=="coverage-tone-sse-block-bounds-v1") throw new Error("Unsupported project version");
  const options=validateOptions(value.options), metrics=validateMetrics(value.metrics,options);
  const {width,height,rgba}=value.source ?? {};
  if(!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1||width*height>1_440_000||typeof rgba!=="string" || rgba.length!==4*Math.ceil(width*height*4/3)) throw new Error("Invalid project source");
  const binary=atob(rgba);
  if(binary.length!==width*height*4) throw new Error("Invalid source pixel count");
  return {width,height,pixels:Uint8ClampedArray.from(binary,c=>c.charCodeAt(0)),options,metrics};
}
