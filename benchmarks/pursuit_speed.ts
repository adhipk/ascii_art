/** Compare complete CPU optimizer runs, including exact verification and history.
 * Usage: bun benchmarks/pursuit_speed.ts fixture.json [report.json] [budget] [larger-budget]
 * Timings are single-run diagnostics, not browser/GPU latency measurements.
 */
import { cpus } from "node:os";
import { renderPursuit } from "../web/pursuit";

const [input,output="/tmp/pursuit-speed.json",budgetArgument="64",largerArgument="256"]=Bun.argv.slice(2);
if (!input) throw new Error("A measured image/atlas fixture is required");
const fixture=await Bun.file(input).json(),pixels=new Uint8Array(fixture.pixels);
const budget=Number(budgetArgument),largerBudget=Number(largerArgument);
const runs=[];
for (const config of [
  {effort:"thorough" as const,beam:2,budget},
  {effort:"fast" as const,beam:1,budget},
  {effort:"fast" as const,beam:1,budget:largerBudget},
]) {
  const result=renderPursuit(pixels,fixture.width,fixture.height,{
    ...fixture.settings,pursuit_effort:config.effort,pursuit_beam:config.beam,pursuit_budget:config.budget,
  });
  const trace=result.pursuit!,last=trace.frames.at(-1)!;
  runs.push({...config,ms:trace.elapsedMs,glyphs:last.glyphs.length,energy:last.energy,
    pixelError:last.pixelError,contourError:last.contourError,diversity:last.diversity,
    candidates:last.candidates,stopReason:trace.stopReason});
  console.log(JSON.stringify(runs.at(-1)));
}
const report={runtime:`Bun ${Bun.version}`,cpu:cpus()[0]?.model,input,
  atlasCharacters:fixture.settings.pursuit_metrics.kernelCharacters,
  scope:"Single complete native CPU optimizer runs; includes history and verification, excludes export, browser display and GPU",
  runs};
await Bun.write(output,JSON.stringify(report,null,2)+"\n");
