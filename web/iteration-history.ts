/** Scrubbing is independent of rendering and never schedules an optimizer job. */
export class IterationHistory {
  index = 0;
  length = 0;
  private timer?: ReturnType<typeof setInterval>;
  constructor(private change: (index: number, playing: boolean) => void) {}
  get playing(): boolean { return this.timer !== undefined; }
  load(length: number): void {
    this.pause();
    this.length = Math.max(0, Math.floor(length));
    this.seek(this.length - 1);
  }
  seek(index: number): void {
    this.index = Math.max(0, Math.min(this.length - 1, Number.isFinite(index) ? Math.round(index) : 0));
    this.change(this.index, this.playing);
  }
  pause(): void {
    if (this.timer !== undefined) clearInterval(this.timer);
    this.timer = undefined;
    this.change(this.index, false);
  }
  play(sequence?: readonly number[], loop = false): void {
    if (this.playing || this.length < 2) return;
    const frames=sequence ? [...new Set(sequence)].filter(i=>Number.isInteger(i)&&i>=0&&i<this.length) : Array.from({length:this.length},(_,i)=>i);
    if(frames.length<2) return;
    let position=frames.indexOf(this.index);
    if(position<0 || position===frames.length-1) {position=0;this.seek(frames[0]);}
    this.timer = setInterval(() => {
      position=(position+1)%frames.length;
      this.seek(frames[position]);
      if (!loop && position===frames.length-1) this.pause();
    }, 150);
    this.change(this.index, true);
  }
}
