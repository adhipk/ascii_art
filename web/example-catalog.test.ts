import { expect, test } from "bun:test";
import { mkdtemp, writeFile, rm, symlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { discoverExamples } from "./example-catalog";
import { staticResponse } from "./serve";
import { exampleUrl } from "./examples";

test("dropping a new image into the folder updates discovery and serving without rebuilding", async () => {
  const directory = await mkdtemp(join(tmpdir(), "ascii-examples-"));
  try {
    expect(await discoverExamples(directory)).toEqual([]);
    const name = "New portrait #1.png";
    const bytes = new Uint8Array([137,80,78,71,13,10,26,10]);
    await writeFile(join(directory,name),bytes);
    const catalog = await (await staticResponse(new Request("http://localhost/examples.json"),directory)).json();
    expect(catalog).toEqual([{id:name,file:name,label:"New portrait #1",thumbnail:name}]);
    const response = await staticResponse(new Request(`http://localhost/examples/${encodeURIComponent(name)}`),directory);
    expect(response.status).toBe(200);
    expect(new Uint8Array(await response.arrayBuffer())).toEqual(bytes);
    expect(response.headers.get("cache-control")).toBe("no-store");
    await writeFile(join(directory,"New portrait #1.thumb.webp"),bytes);
    expect((await discoverExamples(directory))[0].thumbnail).toBe("New portrait #1.thumb.webp");
    await rm(join(directory,name));
    expect(await discoverExamples(directory)).toEqual([]);
  } finally {await rm(directory,{recursive:true,force:true});}
});

test("catalog ignores thumbnails, unsupported files, hidden files, and symlinks", async () => {
  const directory = await mkdtemp(join(tmpdir(), "ascii-examples-"));
  try {
    for(const name of ["portrait.JPG","portrait.png","portrait.thumb.webp","notes.txt",".hidden.png","unsafe.svg"])
      await writeFile(join(directory,name),"fixture");
    await symlink(join(directory,"portrait.JPG"),join(directory,"linked.jpg"));
    const catalog = await discoverExamples(directory);
    expect(catalog.map(example=>example.file)).toEqual(["portrait.JPG","portrait.png"]);
    expect(new Set(catalog.map(example=>example.id)).size).toBe(2);
    expect((await staticResponse(new Request("http://localhost/examples/linked.jpg"),directory)).status).toBe(404);
    expect((await staticResponse(new Request("http://localhost/examples/..%2Fpackage.json"),directory)).status).toBe(404);
  } finally {await rm(directory,{recursive:true,force:true});}
});

test("example URLs preserve filenames with spaces, percent, and fragment characters", () => {
  expect(exampleUrl("a #1%.png").pathname).toEndWith("/examples/a%20%231%25.png");
  expect(exampleUrl("a #1%.png").hash).toBe("");
});
