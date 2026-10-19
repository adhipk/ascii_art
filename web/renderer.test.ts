import { afterEach, expect, test } from "bun:test";
import { Renderer, type RenderInput } from "./renderer";

const originalWorker = globalThis.Worker;
class FakeWorker {
  static instances: FakeWorker[] = [];
  onmessage?: (event: { data: unknown }) => void;
  onerror?: () => void;
  onmessageerror?: () => void;
  messages: any[] = [];
  stopped = false;
  constructor() { FakeWorker.instances.push(this); }
  postMessage(message: unknown): void { this.messages.push(message); }
  terminate(): void { this.stopped = true; }
  receive(data: unknown): void { this.onmessage?.({ data }); }
}
function setup() {
  FakeWorker.instances = [];
  globalThis.Worker = FakeWorker as unknown as typeof Worker;
  const statuses: string[] = [];
  const renderer = new Renderer((message) => statuses.push(message));
  const input = {
    pixels: { data: new Uint8ClampedArray(16), width: 2, height: 2 } as ImageData,
    sourceVersion: 1,
    settings: { columns: 100, font_size: 12, use_edge: true, edge_method: "tiny", edge_sensitivity: 50, edge_amount: 20, use_color: false, background: "#ffffff", background_mode: "source", foreground: "#242424" },
  } satisfies RenderInput;
  return { renderer, input, statuses };
}
afterEach(() => { globalThis.Worker = originalWorker; });
const result = (id: number) => ({ type: "result", id, png: new Uint8Array(), text: "ascii", width: 10, height: 10 });

test("pixels are transferred only when the source changes", async () => {
  const { renderer, input } = setup();
  const first = renderer.render(input, 1);
  const worker = FakeWorker.instances[0];
  expect(worker.messages[0].pixels).toBeInstanceOf(ArrayBuffer);
  worker.receive(result(1)); await first;
  const second = renderer.render({ ...input, settings: { ...input.settings, columns: 120 } }, 2);
  expect(worker.messages[1].pixels).toBeUndefined();
  worker.receive(result(2)); await second;
  const third = renderer.render({ ...input, sourceVersion: 2 }, 3);
  expect(worker.messages[2].pixels).toBeInstanceOf(ArrayBuffer);
  worker.receive(result(3)); await third;
  renderer.stop();
});

test("wrong request IDs cannot finish a job or publish stale status", async () => {
  const { renderer, input, statuses } = setup();
  const pending = renderer.render(input, 7);
  const worker = FakeWorker.instances[0];
  worker.receive({ type: "status", id: 6, message: "old" });
  worker.receive(result(6));
  worker.receive({ type: "status", id: 7, message: "current" });
  expect(statuses).toEqual(["current"]);
  worker.receive(result(7));
  expect((await pending).text).toBe("ascii");
  renderer.stop();
});

test("cancellation terminates the worker and a new worker gets source pixels again", async () => {
  const { renderer, input } = setup();
  const old = renderer.render(input, 1);
  const rejection = old.catch((error: Error) => error);
  const worker = FakeWorker.instances[0];
  renderer.stop(); expect((await rejection as Error).message).toBe("Cancelled");
  expect(worker.stopped).toBe(true);
  const next = renderer.render(input, 2);
  const replacement = FakeWorker.instances[1];
  expect(replacement.messages[0].pixels).toBeInstanceOf(ArrayBuffer);
  replacement.receive(result(2)); await next;
  renderer.stop();
});

test("a failed runtime load resends source pixels on retry", async () => {
  const { renderer, input } = setup();
  const failed = renderer.render(input, 1);
  const rejection = failed.catch((error: Error) => error);
  const worker = FakeWorker.instances[0];
  worker.receive({ type: "error", id: 1, message: "CDN failed" });
  expect((await rejection as Error).message).toBe("CDN failed");
  const retry = renderer.render(input, 2);
  expect(worker.messages[1].pixels).toBeInstanceOf(ArrayBuffer);
  worker.receive(result(2)); await retry;
  renderer.stop();
});
