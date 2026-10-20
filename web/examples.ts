export interface Example { id: string; label: string; file: string; thumbnail: string; }

export async function loadExamples(): Promise<Example[]> {
  const response = await fetch(new URL("./examples.json", import.meta.url), { cache: "no-store" });
  if (!response.ok) throw new Error(`Could not read examples (${response.status})`);
  const examples: unknown = await response.json();
  const safeFile = (file: unknown): file is string => typeof file === "string" && file.length > 0 && !/[\\/\x00-\x1f]/.test(file) && file !== "." && file !== "..";
  if (!Array.isArray(examples) || examples.some(example => !example || typeof example.id !== "string" || typeof example.label !== "string" || !safeFile(example.file) || !safeFile(example.thumbnail)))
    throw new Error("Invalid example catalog");
  return examples;
}

export function exampleUrl(file: string): URL {
  return new URL(`./examples/${encodeURIComponent(file)}`, import.meta.url);
}
