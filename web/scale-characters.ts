import dictionary from "./pursuit-dictionary.json";
const segmenter = new Intl.Segmenter(undefined, {granularity:"grapheme"});
export const splitGlyphs = (text:string):string[] => Array.from(segmenter.segment(text), part=>part.segment);
export const uniqueGlyphs = (text:string):string[] => [...new Set(splitGlyphs(text))];
export const hasGlyph = (text:string,glyph:string):boolean => splitGlyphs(text).includes(glyph);

export const CHARACTER_PRESETS = {
  blocks:{label:"Blocks",characters:"█▓▒░▀▄▌▐■●"},
  lines:{label:"Lines & corners",characters:"─│╱╲┌┐└┘├┤┬┴┼╭╮╰╯"},
  curves:{label:"Curves",characters:"○●◯◖◗◜◝◞◟(){}"},
  fine:{label:"Fine detail",characters:"·•∙⋅:;.,'`˙˘ˆ˜°∘+×÷=≠<>/\\|_-()[]{}"},
  ascii:{label:"ASCII",characters:Array.from({length:94},(_,i)=>String.fromCharCode(i+33)).join("")},
} as const;
export type ScaleCharacters = Record<string,string>;
export type VocabularyProfile = "scale-aware"|"full"|"shape-detail";
export function defaultPursuitScales(_resolution:number): number[] {
  return [32,16,8];
}
export function defaultScaleCharacters(size:number,largest:number,profile:VocabularyProfile): string {
  if(profile==="shape-detail") {
    if(size>=largest) return CHARACTER_PRESETS.blocks.characters;
    if(size>=largest/2) return CHARACTER_PRESETS.lines.characters;
    return CHARACTER_PRESETS.fine.characters;
  }
  const chars=profile==="full" ? dictionary.characters : size>=largest/2 ? dictionary.coarse : size>=largest/8 ? dictionary.medium : dictionary.characters;
  return chars.replaceAll(" ","");
}
export function normalizeScaleCharacters(value: unknown, sizes: readonly number[], available?: string | string[]): ScaleCharacters {
  if(value===undefined) return {};
  if(!value || typeof value!=="object" || Array.isArray(value)) throw new Error("Per-scale characters must be a map of scale to text");
  const result:ScaleCharacters={};
  for(const [scale,chars] of Object.entries(value)) {
    if(!sizes.some(size=>String(size)===scale)) throw new Error(`Unknown character-set scale: ${scale}`);
    if(typeof chars!=="string") throw new Error(`${scale}px: characters must be text`);
    const normalized=uniqueGlyphs(chars).join("");
    if(available!==undefined) for(const char of splitGlyphs(normalized)) if(char!==" " && !(typeof available==="string"?splitGlyphs(available):available).includes(char)) throw new Error(`${scale}px: no measured font mask for ${JSON.stringify(char)}`);
    result[scale]=normalized;
  }
  return result;
}
