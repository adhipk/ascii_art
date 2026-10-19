import { expect, test } from "bun:test";
import { renderInPython, PYTHON_RENDER_CODE } from "./python-render";
import type { RenderSettings } from "./renderer";

const settings: RenderSettings = { columns: 100, font_size: 12, use_edge: false, edge_method: "tiny", edge_sensitivity: 50, edge_amount: 20, use_color: false, background: "#fffFFF", background_mode: "source", foreground: "#242424" };

test("disabled contours cross the Python boundary with an explicit false flag, never null", () => {
  const values = new Map<string, unknown>();
  const removed: string[] = [];
  let destroyed = false;
  const output = { text: "@" } as any;
  const py = {
    globals: { set: (name: string, value: unknown) => values.set(name, value), delete: (name: string) => { removed.push(name); } },
    runPython(code: string) {
      expect(code).toBe(PYTHON_RENDER_CODE);
      expect(values.get("_has_edges")).toBe(false);
      expect(values.get("_edge_pixels")).toBeInstanceOf(Float32Array);
      expect((values.get("_edge_pixels") as Float32Array).length).toBe(0);
      expect([...values.values()].some((value) => value === null)).toBe(false);
      return { toJs: () => output, destroy: () => { destroyed = true; } };
    },
  };
  expect(renderInPython(py, new Uint8Array(4), 1, 1, settings)).toBe(output);
  expect(destroyed).toBe(true);
  expect(removed).toContain("_has_edges");
  expect(removed).toHaveLength(values.size);
});

test("enabled contours preserve the float confidence buffer", () => {
  const values = new Map<string, unknown>();
  const confidence = new Float32Array([.001, .002]);
  const py = {
    globals: { set: (name: string, value: unknown) => values.set(name, value), delete: () => {} },
    runPython() {
      expect(values.get("_has_edges")).toBe(true);
      expect(values.get("_edge_pixels")).toBe(confidence);
      return { toJs: () => ({} as any), destroy: () => {} };
    },
  };
  renderInPython(py, new Uint8Array(8), 2, 1, settings, { pixels: confidence, width: 2, height: 1 });
});
