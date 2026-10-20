import { readdir } from "node:fs/promises";
import { extname, basename } from "node:path";
import type { Example } from "./examples";

/** Shared by the static build and local server. Symlinks and sidecar thumbnails
 * are not standalone examples. Filenames are stable IDs, including extensions.
 */
export async function discoverExamples(directory: string): Promise<Example[]> {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = new Set(entries.filter(entry => entry.isFile() && !entry.name.startsWith(".")).map(entry => entry.name));
  return [...files].filter(name => /\.(png|jpe?g|webp|gif|avif|bmp)$/i.test(name) && !/\.thumb\.[^.]+$/i.test(name))
    .sort((a, b) => a.localeCompare(b, "en"))
    .map(file => {
      const stem = basename(file, extname(file)), thumbnail = `${stem}.thumb.webp`;
      const label = stem.replace(/[-_]+/g, " ").replace(/^./u, char => char.toUpperCase());
      return { id: file, file, label, thumbnail: files.has(thumbnail) ? thumbnail : file };
    });
}
