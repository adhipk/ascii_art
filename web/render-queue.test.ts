import { expect, test } from "bun:test";
import { RenderQueue, type Clock, type QueueState } from "./render-queue";

class TestClock implements Clock {
  private next = 0;
  private callbacks = new Map<number, () => void>();
  set(callback: () => void): unknown { const id = ++this.next; this.callbacks.set(id, callback); return id; }
  clear(timer: unknown): void { this.callbacks.delete(timer as number); }
  flush(): void { const callbacks = [...this.callbacks.values()]; this.callbacks.clear(); callbacks.forEach((fn) => fn()); }
  get size(): number { return this.callbacks.size; }
}
async function settle(): Promise<void> { for (let i = 0; i < 8; ++i) await Promise.resolve(); }
function setup() {
  const clock = new TestClock();
  const started: string[] = [];
  const results: string[] = [];
  const errors: unknown[] = [];
  const states: QueueState[] = [];
  const jobs: { resolve(value: string): void; reject(error: Error): void }[] = [];
  const queue = new RenderQueue<string, string>({
    clock, render: (input) => { started.push(input); return new Promise((resolve, reject) => jobs.push({ resolve, reject })); },
    result: (output) => results.push(output), error: (error) => errors.push(error), state: (state) => states.push(state),
  });
  return { clock, started, results, errors, states, jobs, queue };
}

test("rapid slider edits collapse into one render after the quiet interval", async () => {
  const s = setup();
  for (let i = 0; i < 100; ++i) s.queue.request(String(i));
  expect(s.clock.size).toBe(1);
  expect(s.started).toEqual([]);
  s.clock.flush(); await settle();
  expect(s.started).toEqual(["99"]);
  s.jobs[0].resolve("latest"); await settle();
  expect(s.results).toEqual(["latest"]);
  expect(s.states.at(-1)).toBe("idle");
});

test("an active render is never overlapped; intermediate pending edits are dropped", async () => {
  const s = setup();
  s.queue.request("first", true); await settle();
  s.queue.request("second"); s.queue.request("third"); s.clock.flush(); await settle();
  expect(s.started).toEqual(["first"]);
  s.jobs[0].resolve("stale"); await settle();
  expect(s.results).toEqual([]);
  expect(s.started).toEqual(["first", "third"]);
  s.jobs[1].resolve("fresh"); await settle();
  expect(s.results).toEqual(["fresh"]);
});

test("finishing an active job does not bypass an unexpired debounce", async () => {
  const s = setup();
  s.queue.request("first", true); await settle();
  s.queue.request("last");
  s.jobs[0].resolve("stale"); await settle();
  expect(s.started).toEqual(["first"]);
  expect(s.states.at(-1)).toBe("waiting");
  s.clock.flush(); await settle();
  expect(s.started).toEqual(["first", "last"]);
  s.jobs[1].resolve("fresh"); await settle();
});

test("source replacement pauses scheduling but keeps the worker occupied until its old job finishes", async () => {
  const s = setup();
  s.queue.request("old-image", true); await settle();
  s.queue.request("old-settings");
  s.queue.pause();
  s.queue.request("new-image", true); await settle();
  expect(s.started).toEqual(["old-image"]);
  s.jobs[0].resolve("stale"); await settle();
  expect(s.started).toEqual(["old-image", "new-image"]);
  expect(s.results).toEqual([]);
  s.jobs[1].resolve("new-preview"); await settle();
  expect(s.results).toEqual(["new-preview"]);
});

test("cancel drops pending work and invalidates an active result", async () => {
  const s = setup();
  const id = s.queue.request("active", true); await settle();
  s.queue.request("pending"); s.queue.cancel();
  expect(s.queue.isCurrent(id)).toBe(false);
  expect(s.clock.size).toBe(0);
  s.jobs[0].resolve("cancelled"); await settle();
  expect(s.results).toEqual([]);
  expect(s.started).toEqual(["active"]);
});

test("a cancelled session cannot reset or replace a new session", async () => {
  const s = setup();
  s.queue.request("old", true); await settle();
  s.queue.cancel();
  s.queue.request("new", true); await settle();
  s.jobs[0].reject(new Error("cancelled")); await settle();
  expect(s.errors).toEqual([]);
  expect(s.states.at(-1)).toBe("rendering");
  s.jobs[1].resolve("new"); await settle();
  expect(s.results).toEqual(["new"]);
});

test("stale failures are suppressed and the newest pending edit still runs", async () => {
  const s = setup();
  s.queue.request("bad-old", true); await settle();
  s.queue.request("good-new", true);
  s.jobs[0].reject(new Error("old failure")); await settle();
  expect(s.errors).toEqual([]);
  expect(s.started).toEqual(["bad-old", "good-new"]);
  s.jobs[1].resolve("good"); await settle();
  expect(s.results).toEqual(["good"]);
});

test("a current failure is surfaced and retry remains possible", async () => {
  const s = setup();
  s.queue.request("bad", true); await settle();
  s.jobs[0].reject(new Error("failure")); await settle();
  expect(s.errors).toHaveLength(1);
  s.queue.request("retry", true); await settle();
  s.jobs[1].resolve("recovered"); await settle();
  expect(s.results).toEqual(["recovered"]);
});
