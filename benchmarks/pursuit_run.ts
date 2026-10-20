/** Offline optimizer harness. Input is RGBA + measured atlas JSON; no browser. */
import { renderPursuit } from "../web/pursuit";
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";

const [inputPath, outputPath = "/tmp/glyph-pursuit", budgetArgument = "128", effortArgument = "fast", beamArgument,blurArgument,scheduleArgument,coarseArgument] = Bun.argv.slice(2);
if (!inputPath) throw new Error("Usage: bun benchmarks/pursuit_run.ts fixture.json [output-directory] [glyph-budget] [fast|thorough] [beam-width] [blur-strength] [coarse-to-fine|all] [broad-rounds]");
const fixture = await Bun.file(inputPath).json();
const output = resolve(outputPath);
await mkdir(output, { recursive: true });
const result = renderPursuit(new Uint8Array(fixture.pixels), fixture.width, fixture.height,
  { ...fixture.settings, pursuit_coarse_rounds:Number(coarseArgument??fixture.settings.pursuit_coarse_rounds??3), pursuit_schedule:(scheduleArgument??fixture.settings.pursuit_schedule??"coarse-to-fine") as "coarse-to-fine"|"all", pursuit_blur:Number(blurArgument??fixture.settings.pursuit_blur??1), pursuit_budget: Number(budgetArgument), pursuit_effort: effortArgument as "fast" | "thorough", ...(beamArgument ? {pursuit_beam:Number(beamArgument)} : {}) }, message => process.stderr.write(message + "\n"));
await Bun.write(resolve(output, "trace.json"), JSON.stringify({ ...result, paper: Array.from(result.paper), colors: [], pursuit: { ...result.pursuit, frames: result.pursuit!.frames.map(frame => ({...frame, residual: Array.from(frame.residual)})) } }));
const frames = result.pursuit!.frames;
await Bun.write(resolve(output, "energies.csv"), "iteration,energy,pixel_error,contour_error,glyphs,diversity,action\n" + frames.map(f =>
  [f.iteration, f.energy, f.pixelError, f.contourError, f.glyphs.length, f.diversity, JSON.stringify(f.action)].join(",")).join("\n") + "\n");
console.log(JSON.stringify({frames: frames.length, start: frames[0].energy, final: frames.at(-1)!.energy,
  glyphs: frames.at(-1)!.glyphs.length, diversity: frames.at(-1)!.diversity, ms: result.pursuit!.elapsedMs, reason: result.pursuit!.stopReason, output}));
