import { afterEach, expect, test } from "bun:test";
import { exportTextPng, squareTextMetrics, textMetrics, textPreview } from "./text-view";
import type { RenderResult } from "./renderer";

const originalDocument = globalThis.document;
test("adaptive SVG and PNG use the same positioned font sizes and clipping", async () => {
  const nodes: any[] = [];
  const makeNode = (tag: string) => {
    const node = { tag, attributes: {} as Record<string,string>, style: {}, children: [] as any[], textContent: "", setAttribute(name: string,value: string) { this.attributes[name] = value; }, append(child: any) { this.children.push(child); } };
    nodes.push(node); return node;
  };
  const draws: any[] = [], clips: any[] = [];
  const context = {font:"",fillStyle:"",textBaseline:"",fontKerning:"",scale(){},fillRect(){},save(){},restore(){},beginPath(){},clip(){},rect(...args: number[]){clips.push(args);},fillText(char: string,x: number,y: number){draws.push({char,x,y});}};
  globalThis.document = {createDocumentFragment:()=>makeNode("fragment"),createElementNS:(_namespace:string,tag:string)=>makeNode(tag),createElement:()=>({width:0,height:0,getContext:()=>context,toBlob:(callback:(blob:Blob)=>void)=>callback(new Blob(["png"]))})} as unknown as Document;
  const metric = (size:number) => ({width:size,height:size,baseline:size*.8,advance:size*.6,fontSize:size*.8,densities:[]});
  const result: RenderResult = {text:"",columns:4,rows:4,cellWidth:8,cellHeight:8,width:32,height:32,colors:new Uint8Array(),paper:new Uint8Array(96),tiles:[{x:0,y:0,size:16,char:"@",color:[1,2,3]},{x:16,y:0,size:8,char:"/",color:[4,5,6]}],tileMetrics:{"8":metric(8),"16":metric(16)}};
  textPreview(result);
  expect(nodes.find(node=>node.tag==="svg").attributes.viewBox).toBe("0 0 32 32");
  const glyphs = nodes.filter(node=>node.tag==="text");
  expect(glyphs.map(node=>node.textContent)).toEqual(["@","/"]);
  expect(glyphs.map(node=>node.attributes["font-size"])).toEqual(["12.8","6.4"]);
  await exportTextPng(result,12);
  expect(clips).toEqual([[0,0,16,16],[16,0,8,8]]);
  expect(draws.map(draw=>[draw.char,draw.x,draw.y])).toEqual([["@",3.2,12.8],["/",17.6,6.4]]);
});
afterEach(() => { globalThis.document = originalDocument; });

test("PNG export draws the real characters and cell colors, with a high-resolution backing", async () => {
  const draws: { char: string; x: number; y: number; color: string }[] = [];
  const canvases: { width: number; height: number }[] = [];
  let lastCharacter = "";
  const context = {
    font: "", fillStyle: "", textBaseline: "", fontKerning: "",
    measureText() { return { width: 7.2, fontBoundingBoxAscent: 11, fontBoundingBoxDescent: 3 }; },
    clearRect() {}, fillRect() {}, scale() {},
    fillText(char: string, x: number, y: number) { lastCharacter = char; draws.push({ char, x, y, color: this.fillStyle }); },
    getImageData() { return { data: new Uint8ClampedArray([255, 255, 255, lastCharacter === " " ? 0 : 255]) }; },
  };
  globalThis.document = {
    createElement(tag: string) {
      expect(tag).toBe("canvas");
      const canvas = { width: 0, height: 0, getContext: () => context, toBlob(callback: (blob: Blob) => void) { callback(new Blob(["png"], { type: "image/png" })); } };
      canvases.push(canvas);
      return canvas;
    },
  } as unknown as Document;
  const metrics = textMetrics(12);
  expect(metrics.width).toBe(8);
  expect(metrics.densities[0]).toBe(0);
  const result: RenderResult = { text: "@%", columns: 2, rows: 1, cellWidth: 8, cellHeight: 17, width: 16, height: 17, colors: new Uint8Array([255, 0, 0, 0, 128, 255]), paper: new Uint8Array(17 * 3).fill(255) };
  draws.length = 0;
  const blob = await exportTextPng(result, 12);
  expect(blob.type).toBe("image/png");
  expect(canvases.at(-1)!.width).toBe(32);
  expect(canvases.at(-1)!.height).toBe(34);
  expect(draws).toEqual([
    { char: "@", x: 0, y: metrics.baseline, color: "rgb(255 0 0)" },
    { char: "%", x: 8, y: metrics.baseline, color: "rgb(0 128 255)" },
  ]);
});


test("adaptive calibration carries spatial alpha masks for the exact positioned font", () => {
  let char = " ";
  const context = {
    font: "", fillStyle: "",
    measureText: () => ({width: 4, fontBoundingBoxAscent: 5, fontBoundingBoxDescent: 1}),
    clearRect() {},
    fillText(value: string, x: number, y: number) {
      char = value;
      expect(x).toBe(2); expect(y).toBe(6);
    },
    getImageData() {
      const data = new Uint8ClampedArray(8 * 8 * 4);
      if (char !== " ") data[7] = 128;
      return { data };
    },
  };
  globalThis.document = { createElement: () => ({width: 0, height: 0, getContext: () => context}) } as unknown as Document;
  const metrics = squareTextMetrics(8);
  expect(metrics.kernelCharacters).toBe(" .:-=+*#%@/|\\");
  expect(metrics.masks!.length).toBe(metrics.kernelCharacters!.length);
  expect(metrics.masks![0]).toEqual(Array(64).fill(0));
  expect(metrics.masks![1][1]).toBe(128 / 255);
  expect(metrics.densities[1]).toBe(128 / 255 / 64);
  expect(metrics.densities.length).toBe(10);
  expect(squareTextMetrics(8)).toBe(metrics);
});

test("rotated pursuit glyphs use matching SVG and Canvas centers", async () => {
  const nodes:any[]=[]; const rotations:number[]=[]; const translations:number[][]=[];
  const make=(tag:string)=>{const node={tag,attributes:{} as Record<string,string>,style:{},setAttribute(k:string,v:string){this.attributes[k]=v;},append(){}};nodes.push(node);return node;};
  const context={font:"",fillStyle:"",textBaseline:"",fontKerning:"",scale(){},fillRect(){},save(){},restore(){},beginPath(){},clip(){},rect(){},fillText(){},translate(x:number,y:number){translations.push([x,y]);},rotate(angle:number){rotations.push(angle);}};
  globalThis.document={createDocumentFragment:()=>make("fragment"),createElementNS:(_:string,tag:string)=>make(tag),createElement:()=>({width:0,height:0,getContext:()=>context,toBlob:(callback:(blob:Blob)=>void)=>callback(new Blob(["png"]))})} as unknown as Document;
  const result:RenderResult={text:"",columns:0,rows:0,cellWidth:0,cellHeight:0,width:48,height:48,colors:new Uint8Array(),paper:new Uint8Array(48*3),tiles:[{x:8,y:12,size:16,char:"|",angle:90,color:[255,255,255]}],tileMetrics:{"16":{width:16,height:16,baseline:12,advance:8,fontSize:14,densities:[]}}};
  textPreview(result); await exportTextPng(result,12);
  expect(nodes.find(node=>node.tag==="text").attributes.transform).toBe("rotate(90 8 8)");
  expect(translations).toEqual([[16,20]]); expect(rotations).toEqual([Math.PI/2]);
});
