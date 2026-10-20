import type { PursuitTrace } from "./pursuit";
import type { TextMetrics } from "./text-view";
import type { VocabularyProfile } from "./scale-characters";

export interface RenderSettings {
  columns: number;
  font_size: number;
  text_metrics?: TextMetrics;
  layout?: "fixed" | "adaptive" | "pursuit";
  pursuit_metrics?: TextMetrics;
  pursuit_backend?: "auto" | "cpu" | "webgpu";
  pursuit_max_overlap?: number;
  pursuit_overlap_penalty?: number;
  pursuit_repeat_penalty?: number;
  pursuit_penalty_min_size?: number;
  pursuit_solid_brush?: boolean;
  pursuit_vocabulary?: VocabularyProfile;
  pursuit_characters_by_scale?: Record<string,string>;
  pursuit_coarse_threshold?: number;
  pursuit_coarse_rounds?: number;
  pursuit_schedule?: "coarse-to-fine" | "all" | "independent";
  pursuit_blur?: number;
  pursuit_detail_sharpen?: number;
  pursuit_budget?: number;
  pursuit_effort?: "fast" | "thorough";
  pursuit_resolution?: number;
  pursuit_scales?: number[];
  pursuit_beam?: number;
  pursuit_edge_weight?: number;
  adaptive_penalty?: number;
  adaptive_objective?: "tone" | "pixels";
  adaptive_metrics?: Record<string, TextMetrics>;
  use_edge: boolean;
  edge_method: "tiny" | "dog" | "acerola";
  edge_min_pixels?: number;
  edge_sensitivity: number;
  edge_amount: number;
  use_color: boolean;
  background: string;
  background_mode: "solid" | "source";
  foreground: string;
}
export interface RenderInput {
  pixels: ImageData;
  sourceVersion: number;
  settings: RenderSettings;
}
export interface AdaptiveTile { x: number; y: number; size: number; char: string; color: number[]; angle?: number; }
export interface RenderResult { pursuit?: PursuitTrace; text: string; colors: Uint8Array; paper: Uint8Array; columns: number; rows: number; cellWidth: number; cellHeight: number; width: number; height: number; tiles?: AdaptiveTile[]; tileMetrics?: Record<string, TextMetrics>; glyphAtlas?: TextMetrics; }

/** Persistent worker: upload pixels once per source, then send settings only. */
export class Renderer {
  private worker?: Worker;
  private sourceVersion = -1;
  private pending?: { id: number; resolve(result: RenderResult): void; reject(error: Error): void; timer: ReturnType<typeof setTimeout> };
  constructor(private status: (message: string, id: number) => void) {}

  render(input: RenderInput, id: number): Promise<RenderResult> {
    return new Promise((resolve, reject) => {
      if (this.pending) { reject(new Error("A render is already active")); return; }
      const search = input.settings.layout === "pursuit";
      const timer = setTimeout(() => this.stop(search ? "Search timed out. Reduce fitting detail, competing compositions, or glyph budget." : "The renderer timed out. Check your connection or reduce the image size."), search ? 600_000 : 180_000);
      this.pending = { id, resolve, reject, timer };
      try {
        if (!this.worker) {
          this.worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
          this.worker.onerror = () => this.stop("Could not start the renderer. Check access to cdn.jsdelivr.net and retry.");
          this.worker.onmessageerror = () => this.stop("Could not read the renderer response. Please retry.");
          this.worker.onmessage = ({ data }) => {
            if (!this.pending || data.id !== this.pending.id) return;
            if (data.type === "status") { this.status(data.message, data.id); return; }
            const job = this.pending;
            this.pending = undefined;
            clearTimeout(job.timer);
            if (data.type === "result") job.resolve(data);
            else if (data.type === "error") { this.sourceVersion = -1; job.reject(new Error(data.message)); }
          };
        }
        const pixels = input.sourceVersion === this.sourceVersion ? undefined : input.pixels.data.slice().buffer;
        this.worker.postMessage({
          type: "render", id, pixels, sourceVersion: input.sourceVersion,
          width: input.pixels.width, height: input.pixels.height, settings: input.settings,
        }, pixels ? [pixels] : []);
        this.sourceVersion = input.sourceVersion;
      } catch (error) { this.stop(error instanceof Error ? error.message : "Could not start renderer"); }
    });
  }

  stop(reason = "Cancelled"): void {
    this.worker?.terminate();
    this.worker = undefined;
    this.sourceVersion = -1;
    if (this.pending) {
      clearTimeout(this.pending.timer);
      this.pending.reject(new Error(reason));
      this.pending = undefined;
    }
  }
}
