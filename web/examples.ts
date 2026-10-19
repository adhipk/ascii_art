/** Bundled fixtures: originals are copied verbatim; small thumbnails load first. */
export const EXAMPLES = [
  { id: "taylor", label: "Taylor", file: "taylor.png", thumbnail: "taylor.thumb.webp" },
  { id: "webb", label: "Webb poster", file: "webb-poster.webp", thumbnail: "webb-poster.thumb.webp" },
  { id: "spacesuit", label: "Spacesuit", file: "spacesuit-patent.webp", thumbnail: "spacesuit-patent.thumb.webp" },
  { id: "test", label: "Test image", file: "test.png", thumbnail: "test.thumb.webp" },
] as const;
export type Example = typeof EXAMPLES[number];

export function exampleUrl(file: string): URL {
  return new URL(`./examples/${file}`, import.meta.url);
}
