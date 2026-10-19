/** Debounce edits, run one job at a time, and discard superseded results. */
export type QueueState = "waiting" | "rendering" | "idle";
export interface Clock {
  set(callback: () => void, delay: number): unknown;
  clear(timer: unknown): void;
}
const clock: Clock = {
  set: (callback, delay) => setTimeout(callback, delay),
  clear: (timer) => clearTimeout(timer as ReturnType<typeof setTimeout>),
};
export class RenderQueue<Input, Output> {
  private revision = 0;
  private pending?: { id: number; input: Input };
  private active?: { id: number; input: Input };
  private timer?: unknown;
  private ready = false;

  constructor(private options: {
    render(input: Input, id: number): Promise<Output>;
    result(output: Output, input: Input, id: number): void;
    error(error: unknown): void;
    state(state: QueueState): void;
    delay?: number;
    clock?: Clock;
  }) {}

  isCurrent(id: number): boolean { return id === this.revision; }

  request(input: Input, immediate = false): number {
    const id = ++this.revision;
    this.pending = { id, input };
    this.ready = immediate;
    const time = this.options.clock ?? clock;
    if (this.timer !== undefined) time.clear(this.timer);
    this.timer = undefined;
    this.options.state("waiting");
    if (immediate) this.drain();
    else this.timer = time.set(() => {
      this.timer = undefined;
      this.ready = true;
      this.drain();
    }, this.options.delay ?? 180);
    return id;
  }

  pause(): void {
    ++this.revision;
    if (this.timer !== undefined) (this.options.clock ?? clock).clear(this.timer);
    this.timer = undefined;
    this.pending = undefined;
    this.ready = false;
    this.options.state("idle");
  }

  cancel(): void {
    this.pause();
    this.active = undefined;
  }

  private drain(): void {
    if (this.active || !this.pending || !this.ready) return;
    const job = this.pending;
    this.pending = undefined;
    this.ready = false;
    this.active = job;
    this.options.state("rendering");
    Promise.resolve().then(() => this.options.render(job.input, job.id)).then(
      (output) => { if (this.isCurrent(job.id)) this.options.result(output, job.input, job.id); },
      (error) => { if (this.isCurrent(job.id)) this.options.error(error); },
    ).finally(() => {
      if (this.active !== job) return; // A cancelled job cannot alter a new session.
      this.active = undefined;
      if (this.pending && this.ready) this.drain();
      else this.options.state(this.pending ? "waiting" : "idle");
    });
  }
}
