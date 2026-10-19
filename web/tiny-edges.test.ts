import { expect, test } from "bun:test";
import { cropEdges, prepareEdges, TinyEdges } from "./tiny-edges";

test("PiDiNet input uses RGB ImageNet normalization and stride-aligned replicated padding", () => {
  const input = prepareEdges(new Uint8Array([255, 0, 0, 255]), 1, 1);
  expect(input.paddedWidth).toBe(8);
  expect(input.paddedHeight).toBe(8);
  expect(input.data[0]).toBeCloseTo((1 - .485) / .229, 5);
  expect(input.data[64]).toBeCloseTo(-.456 / .224, 5);
  expect(input.data[128]).toBeCloseTo(-.406 / .225, 5);
  expect(input.data[63]).toBe(input.data[0]);
});

test("transparent hidden RGB cannot create neural edges", () => {
  const red = prepareEdges(new Uint8Array([255, 0, 0, 0]), 1, 1);
  const blue = prepareEdges(new Uint8Array([0, 0, 255, 0]), 1, 1);
  expect(red.data).toEqual(blue.data);
});

test("inference budget preserves source aspect ratio and never exceeds 384 pixels", () => {
  const input = prepareEdges(new Uint8Array(1200 * 600 * 4), 1200, 600);
  expect(input.width).toBe(384);
  expect(input.height).toBe(192);
  expect(input.data.length).toBe(384 * 192 * 3);
});

test("output removes padding without min-max normalization that amplifies noise", () => {
  const input = prepareEdges(new Uint8Array(3 * 2 * 4), 3, 2);
  const data = new Float32Array(64).fill(.01);
  data[0] = 0; data[1] = .5; data[2] = 1; data[8] = .25;
  const output = cropEdges(data, [1, 1, 8, 8], input);
  expect(output.pixels).toBeInstanceOf(Float32Array);
  for (const [i, expected] of [0, .5, 1, .25, .01, .01].entries()) expect(output.pixels[i]).toBeCloseTo(expected, 7);
  expect(output.width).toBe(3);
  expect(output.height).toBe(2);
  expect(() => cropEdges(data, [1, 1, 4, 16], input)).toThrow();
});

test("model inference is cached across ASCII settings and reruns for a new source", async () => {
  const detector = new TinyEdges();
  let runs = 0;
  let tensorsDisposed = 0;
  const session = {
    inputNames: ["image"], outputNames: ["fused"],
    async run(feeds: Record<string, { dims: number[] }>) {
      ++runs;
      const [,, h, w] = feeds.image.dims;
      return { fused: { dims: [1, 1, h, w], data: new Float32Array(h * w).fill(.7), dispose() { ++tensorsDisposed; } } };
    },
  };
  // Substitute only the remote runtime: use the real preprocessing and cache.
  (detector as any).load = async () => ({
    session,
    ort: { Tensor: class { constructor(public type: string, public data: Float32Array, public dims: number[]) {} dispose() { ++tensorsDisposed; } } },
  });
  const rgba = new Uint8Array(4 * 4 * 4).fill(255);
  const first = await detector.detect(rgba, 4, 4, 1, () => {});
  const second = await detector.detect(rgba, 4, 4, 1, () => {});
  expect(second).toBe(first);
  expect(runs).toBe(1);
  await detector.detect(rgba, 4, 4, 2, () => {});
  expect(runs).toBe(2);
  expect(tensorsDisposed).toBe(4);
});
