/** PiDiNet Tiny input/output adapter. Inference is cached independently of ASCII settings. */
const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";
const MODEL_URL = "https://huggingface.co/bdck/PiDiNet_ONNX/resolve/6b3f899da74e697a7d2c2e7cba10cfce057d6e55/";
const MODEL = "table5_pidinet_tiny.onnx";
const WEIGHTS = `${MODEL}.data`;
const MEAN = [.485, .456, .406];
const STD = [.229, .224, .225];
export interface EdgeMap { pixels: Float32Array; width: number; height: number; }
export interface ModelInput { data: Float32Array; width: number; height: number; paddedWidth: number; paddedHeight: number; }

export function prepareEdges(rgba: Uint8Array, width: number, height: number, maxEdge = 384): ModelInput {
  if (width < 1 || height < 1 || rgba.length !== width * height * 4) throw new Error("Invalid edge source buffer");
  const scale = Math.min(1, maxEdge / Math.max(width, height));
  const w = Math.max(1, Math.round(width * scale));
  const h = Math.max(1, Math.round(height * scale));
  const pw = Math.max(8, Math.ceil(w / 8) * 8);
  const ph = Math.max(8, Math.ceil(h / 8) * 8);
  const plane = pw * ph;
  const data = new Float32Array(plane * 3);
  // Bilinear resize, alpha-composite on neutral gray, and replicate-pad to stride 8.
  for (let y = 0; y < ph; ++y) {
    const sy = Math.max(0, Math.min(height - 1, (Math.min(y, h - 1) + .5) * height / h - .5));
    const y0 = Math.floor(sy), y1 = Math.min(height - 1, y0 + 1), fy = sy - y0;
    for (let x = 0; x < pw; ++x) {
      const sx = Math.max(0, Math.min(width - 1, (Math.min(x, w - 1) + .5) * width / w - .5));
      const x0 = Math.floor(sx), x1 = Math.min(width - 1, x0 + 1), fx = sx - x0;
      const positions = [(y0 * width + x0) * 4, (y0 * width + x1) * 4, (y1 * width + x0) * 4, (y1 * width + x1) * 4];
      const weights = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy];
      for (let c = 0; c < 3; ++c) {
        let value = 0;
        for (let i = 0; i < 4; ++i) {
          const alpha = rgba[positions[i] + 3] / 255;
          value += weights[i] * (rgba[positions[i] + c] / 255 * alpha + .5 * (1 - alpha));
        }
        data[c * plane + y * pw + x] = (value - MEAN[c]) / STD[c];
      }
    }
  }
  return { data, width: w, height: h, paddedWidth: pw, paddedHeight: ph };
}

export function cropEdges(values: ArrayLike<number>, dims: readonly number[], input: ModelInput): EdgeMap {
  if (dims.length !== 4 || dims[0] !== 1 || dims[1] !== 1 || dims[2] !== input.paddedHeight || dims[3] !== input.paddedWidth || values.length !== input.paddedWidth * input.paddedHeight) throw new Error("Unexpected PiDiNet output shape");
  const pixels = new Float32Array(input.width * input.height);
  for (let y = 0; y < input.height; ++y) for (let x = 0; x < input.width; ++x) {
    const value = values[y * input.paddedWidth + x];
    if (!Number.isFinite(value)) throw new Error("Invalid edge confidence");
    pixels[y * input.width + x] = Math.max(0, Math.min(1, value));
  }
  return { pixels, width: input.width, height: input.height };
}

async function modelBytes(name: string): Promise<Uint8Array> {
  const response = await fetch(MODEL_URL + name);
  if (!response.ok) throw new Error(`Could not download ${name}`);
  return new Uint8Array(await response.arrayBuffer());
}
export class TinyEdges {
  private runtime?: Promise<{ ort: any; session: any }>;
  private cached?: { version: number; map: EdgeMap };

  async detect(rgba: Uint8Array, width: number, height: number, version: number, status: (message: string) => void): Promise<EdgeMap> {
    if (this.cached?.version === version) return this.cached.map;
    status("Loading tiny edge detector…");
    this.runtime ??= this.load().catch((error) => { this.runtime = undefined; throw error; });
    const { ort, session } = await this.runtime;
    status("Detecting contours…");
    const input = prepareEdges(rgba, width, height);
    const tensor = new ort.Tensor("float32", input.data, [1, 3, input.paddedHeight, input.paddedWidth]);
    let outputs: Record<string, any> | undefined;
    try {
      outputs = await session.run({ [session.inputNames[0]]: tensor });
      const fused = outputs!["fused"] ?? outputs![session.outputNames.at(-1)];
      if (!fused) throw new Error("PiDiNet fused output is unavailable");
      const map = cropEdges(fused.data, fused.dims, input);
      this.cached = { version, map };
      return map;
    } finally {
      tensor.dispose?.();
      if (outputs) for (const value of Object.values(outputs)) value.dispose?.();
    }
  }

  private async load(): Promise<{ ort: any; session: any }> {
    const ort = await import(`${ORT_URL}ort.wasm.min.mjs`);
    ort.env.wasm.wasmPaths = ORT_URL;
    ort.env.wasm.numThreads = 1; // Static hosting needs no cross-origin isolation headers.
    const [model, weights] = await Promise.all([modelBytes(MODEL), modelBytes(WEIGHTS)]);
    const session = await ort.InferenceSession.create(model, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
      externalData: [{ path: WEIGHTS, data: weights }],
    });
    return { ort, session };
  }
}
