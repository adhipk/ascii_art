import type { RenderResult, RenderSettings } from "./renderer";
import type { EdgeMap } from "./tiny-edges";

export const PYTHON_RENDER_CODE = `
render_rgba(
    bytes(_pixels.to_py()), _width, _height, json.loads(_settings),
    bytes(_edge_pixels.to_py()) if _has_edges else None,
    _edge_width, _edge_height, 'float32',
)
`;
interface PythonRuntime {
  globals: { set(name: string, value: unknown): void; delete(name: string): void };
  runPython(code: string): { toJs(options: unknown): RenderResult; destroy(): void };
}
export function renderInPython(py: PythonRuntime, pixels: Uint8Array, width: number, height: number, settings: RenderSettings, edges?: EdgeMap): RenderResult {
  const values = {
    _pixels: pixels, _width: width, _height: height, _settings: JSON.stringify(settings),
    _has_edges: Boolean(edges), _edge_pixels: edges?.pixels ?? new Float32Array(0),
    _edge_width: edges?.width ?? 0, _edge_height: edges?.height ?? 0,
  };
  let result: ReturnType<PythonRuntime["runPython"]> | undefined;
  try {
    for (const [name, value] of Object.entries(values)) py.globals.set(name, value);
    result = py.runPython(PYTHON_RENDER_CODE);
    return result.toJs({ dict_converter: Object.fromEntries });
  } finally {
    result?.destroy();
    for (const name of Object.keys(values)) py.globals.delete(name);
  }
}
