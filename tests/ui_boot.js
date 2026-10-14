// A DOM-stub harness for ui/index.html (nnstudio). It loads the page script,
// tolerates the async init that needs the network, and then checks the canvas
// invariants that matter offline: boxes, wires, sizes, nested levels, the
// estimate, undo, copy/paste, GGUF nodes and the deactivate/freeze flags.
//
//   node tests/ui_boot.js
//
const fs = require("fs");
const path = require("path");
const assert = require("assert");

const UI = path.join(__dirname, "..", "ui", "index.html");
const html = fs.readFileSync(UI, "utf8");
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) { console.error("no <script> block in ui/index.html"); process.exit(1); }
const src = m[1];

// ---------------------------------------------------------------- DOM stub
function el(id) {
  const e = {
    id, value: "", textContent: "", hidden: false, disabled: false, checked: false,
    open: false, files: [], dataset: {}, style: {}, children: [], className: "",
    type: "", placeholder: "", title: "", spellcheck: false, autocomplete: "",
    _handlers: {}, _attrs: {}, _parent: null, _html: "",
    offsetWidth: 0, offsetHeight: 0, clientWidth: 600, clientHeight: 400,
    scrollTop: 0, scrollHeight: 0,
    addEventListener(type, fn) { (this._handlers[type] = this._handlers[type] || []).push(fn); },
    removeEventListener() {},
    dispatchEvent(ev) {
      const list = this._handlers[(ev && ev.type) || "click"] || [];
      for (const fn of list) fn(Object.assign({ target: this, stopPropagation() {} }, ev || {}));
      return true;
    },
    click() { this.dispatchEvent({ type: "click" }); },
    setAttribute(k, v) {
      this._attrs[k] = String(v);
      if (k === "class") this.className = String(v);
      if (k === "id") this.id = v;
    },
    getAttribute(k) { return this._attrs[k] != null ? this._attrs[k] : null; },
    appendChild(c) { c._parent = this; this.children.push(c); return c; },
    removeChild(c) { this.children = this.children.filter(x => x !== c); return c; },
    remove() { if (this._parent) this._parent.removeChild(this); },
    get parentNode() { return this._parent; },
    insertAdjacentHTML() {}, setRangeText() {}, focus() {}, blur() {},
    scrollIntoView() {}, getBoundingClientRect() { return { x: 0, y: 0, width: 600, height: 400,
                                                            left: 0, top: 0, right: 600, bottom: 400 }; },
    contains(c) { for (let p = c; p; p = p._parent) if (p === this) return true; return false; },
    closest(sel) { const cls = sel.replace(/^\./, "").split(/[ .]/)[0];
                   for (let p = this; p; p = p._parent)
                     if (String(p.className).split(/\s+/).includes(cls)) return p;
                   return null; },
    querySelector(sel) { return this.querySelectorAll(sel)[0] || null; },
    querySelectorAll(sel) {
      const cls = sel.replace(/^.*\./, "").split(/[ .\[]/)[0];
      const out = [];
      (function walk(n) {
        for (const c of n.children || []) {
          if (String(c.className).split(/\s+/).includes(cls)) out.push(c);
          walk(c);
        }
      })(this);
      return out;
    },
  };
  e.classList = {
    add(...cs) { const s = new Set(String(e.className).split(/\s+/).filter(Boolean)); cs.forEach(c => s.add(c)); e.className = [...s].join(" "); e._attrs["class"] = e.className; },
    remove(...cs) { const s = new Set(String(e.className).split(/\s+/).filter(Boolean)); cs.forEach(c => s.delete(c)); e.className = [...s].join(" "); e._attrs["class"] = e.className; },
    toggle(c, force) { const has = String(e.className).split(/\s+/).includes(c);
                       const on = force === undefined ? !has : !!force;
                       if (on) this.add(c); else this.remove(c); return on; },
    contains(c) { return String(e.className).split(/\s+/).includes(c); },
  };
  Object.defineProperty(e, "innerHTML", {
    get() { return e._html; },
    set(v) { e._html = String(v); e.children = []; },
  });
  Object.defineProperty(e, "id", {
    get() { return e._id || ""; },
    set(v) {
      e._id = String(v);
      if (e._id && typeof els !== "undefined") els.set(e._id, e);
    },
  });
  if (id) e.id = id; else e._id = "";      // via the setter: registers in `els`
  return e;
}
const els = new Map();
const bodyEl = el("body");
// static ids live in one tree too, so a walk finds elements the page appends
// into a container it looked up by id (popovers, drawers, lists …)
const getEl = id => {
  if (!els.has(id)) {
    const e = el(id);
    if (id !== "body") bodyEl.appendChild(e);
  }
  return els.get(id);
};
const findByWalk = id => {
  let hit = null;
  (function walk(e) {
    if (hit) return;
    if (e._attrs && e._attrs.id === id) { hit = e; return; }
    for (const c of e.children || []) { walk(c); if (hit) return; }
  })(bodyEl);
  return hit;
};
const store = new Map();
let onUnhandled = null;
process.on("unhandledRejection", e => { if (onUnhandled) onUnhandled(e); });

// what /api/gguf/canvas hands back: the model as one dedicated nested box (see
// tests/gguf_canvas_test.py) — the UI moves that box's inside into a GGUF node
const GGUF_CANVAS = { type: "canvas", version: 1,
  model: { arch: "test", layers: 2, kv_heads: 2, head_dim: 64, ctx: 4096,
           params: 8192, bytes: 16384, dtypes: { f16: 16384 } },
  nodes: [
    { id: "w1", kind: "input", name: "tokens", x: 0, y: 0, dout: 1 },
    { id: "w2", kind: "custom", name: "Test-Model", x: 320, y: 0, inner: { type: "nested",
      version: 1,
      nodes: [
        { id: "m1", kind: "input", name: "tokens", x: 0, y: 0, dout: 1 },
        { id: "m2", kind: "embed", name: "token embedding", x: 320, y: 0, vocab: 128,
          din: 1, dout: 128, t: "token_embd.weight", params: 8192, wbytes: 16384,
          dtype: "f16" },
        { id: "m3", kind: "output", name: "logits", x: 640, y: 0, din: 128 }],
      edges: [{ from: "m1", to: "m2" }, { from: "m2", to: "m3" }] } },
    { id: "w3", kind: "output", name: "logits", x: 640, y: 0, din: 128 }],
  edges: [{ from: "w1", to: "w2" }, { from: "w2", to: "w3" }] };
// the same model with a second (next-token) logits dot, for the re-crop test
const GGUF_CANVAS2 = JSON.parse(JSON.stringify(GGUF_CANVAS));
GGUF_CANVAS2.nodes.find(n => n.id === "w2").inner.nodes.push(
  { id: "m4", kind: "output", name: "mtp logits", x: 640, y: 120, din: 128 });
GGUF_CANVAS2.nodes.find(n => n.id === "w2").inner.edges.push({ from: "m2", to: "m4" });
GGUF_CANVAS2.nodes.push({ id: "w4", kind: "output", name: "mtp logits", x: 640, y: 120, din: 128 });
GGUF_CANVAS2.edges.push({ from: "w2", to: "w4", fromPort: 1, toPort: 0 });
global.fetch = async (url) => {
  const u = String(url);
  const jr = (body) => ({ ok: true, status: 200, json: async () => body, text: async () => JSON.stringify(body) });
  if (u.startsWith("/api/models")) return jr({ models: [
    { kind: "hf", name: "not-a-gguf", path: "hf/not-a-gguf" },
    { kind: "gguf", name: "test-model.gguf", path: "models/test-model.gguf" }] });
  if (u.startsWith("/api/gguf/canvas")) return jr({ ok: true, canvas: GGUF_CANVAS,
    summary: { arch: "test", blocks: 2, boxes: 3 } });
  return jr({ ok: true });
};
global.document = {
  getElementById: id => els.get(id) || findByWalk(id) || getEl(id),
  querySelector: sel => getEl(sel.replace(/^#/, "")),
  querySelectorAll: () => [],
  createElement: tag => el("<" + tag + ">"),
  createElementNS: (_ns, tag) => el("<" + tag + ">"),
  addEventListener() {},
  body: bodyEl,
  get cookie() { return ""; },
  set cookie(_v) {},
};
global.window = global;
global.addEventListener = () => {};
// the UI keeps a live GPU badge with setInterval — the stub must not hold the
// process open, so timers are inert here
global.setInterval = () => 0;
global.clearInterval = () => {};
global.setTimeout = (fn) => { try { fn(); } catch (e) { /* ignore */ } return 0; };
global.performance = { now: () => Date.now() };
global.location = { origin: "http://localhost:8123", pathname: "/", search: "", hash: "" };
global.history = { replaceState() {} };
global.localStorage = {
  getItem: k => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: k => store.delete(k),
};
global.btoa = s => Buffer.from(s, "binary").toString("base64");
global.atob = s => Buffer.from(s, "base64").toString("binary");
global.URL.createObjectURL = () => "blob:x";
global.URL.revokeObjectURL = () => {};
global.Image = class { set src(_v) { if (this.onerror) this.onerror(); } };
global.FileReader = class { readAsDataURL() {} };
global.requestAnimationFrame = fn => setTimeout(fn, 0);

// ------------------------------------------------------- load + export seam
(0, eval)(src + `
globalThis.__t = {
  bxAdd, bxRemove, bxNode, bxConnect, bxMoveTo, bxConfigure, bxRemoveEdge, bxWireMid, bxWirePath,
  bxSetVia, bxClearVia, bxNaturalMid, bxSelect, bxSelBox, bxSelEdge, bxDeleteSelection,
  get bxSel() { return bxSel; }, bxRender, bxFit, bxZoomAt, bxViewBox,
  bxRenderPalette, bxDropPalette, bxOverCanvas, bxPalDown, bxPalMove, bxPalUp, bxKindLabel,
  bxWheelFactor, bxSize, bxDefaultName, bxForm, bxCheckLink, bxEdgeMismatch, bxSettleSizes, bxAllDocs, bxSettleRound,
  bxExportCanvas, bxExportNested, bxImportJSON, bxJSONRead, bxJSONDocFrom, bxJSONNodeFrom,
  bxJSONNode,
  bxWireState, BX_ACTS, BX_OPS, BX_OPAQUE, bxOpName, bxOpArity, bxDims, bxReshapeSize,
  bxAddTemplate, bxAddKind, bxGGUFImport, bxGGUFImportApply, bxGGUFLoad,
  bxAttentionDoc, bxStateDoc,
  bxMoEDoc, bxTemplateName, bxErrorSet,
  bxMoveInto, bxDropTargetAt, bxHasInside, bxUndoStep, bxRedoStep, bxUndoDepth, bxCommit,
  bxSnapMove, bxSnapGrid, bxClampView, bxSeedBlank, bxNodeH,
  get bxGuides() { return bxGuides; }, set bxGuides(v) { bxGuides = v; },
  bxSelectBoxes, bxSelectedIds, bxToggleSelect, bxMarqueeSelect, bxStartMove, bxMove, bxUp,
  BX_DTYPES, BX_DTYPE_GROUPS, bxBytes, bxDtypeFill, bxEstimate, bxEstimateRuns, bxRenderEstimate, bxPortDefs, bxPortSizes, bxEnter, bxLeave, bxUpBox,
  bxOutSize, bxInSize,
  bxMenuBuild, bxMenuClose, bxClip, bxShape, bxtExport, bxtImport, bxtForm, bxtRefresh,
  bxWeightsForm, bxWeightsApply, bxWeightsExportForm, bxWeightScore, bxTensorDims,
  bxTensorDtype, bxApplyManifest,
  bxtFilesToggle, bxtFilesClose, bxtDropsClose, bxtPaletteToggle, bxtImportToggle,
  bxGGUFImport, bxGGUFImportApply, bxtVramToggle,
  bxSecret, bxCryptoReady, bxAuthCred, bxNameTag, bxSeal, bxUnseal,
  bxTokenRemember, bxTokenRecall, bxAccountLoad, bxtAccountToggle, bxAccountBuild,
  bxAccountCreate, bxAccountRegister, bxAccountSignedIn, bxAccountLocked, bxAccountOut,
  bxAccountReveal, bxAccountLogin, bxAccountUnlockForm, bxAccountSaveForm, bxAccountSave,
  bxAccountOpenForm, bxAccountList, bxAccountOpen, bxAccountOpenApply,
  get BXACCT() { return BXACCT; },
  bxtBackendToggle, bxtBackendPick, get BX_BACKEND() { return BX_BACKEND; },
  bxtRunToggle, bxtRunRefresh, bxtRunSync, bxtRunEnabled, bxtRunGo, bxtRunLoad, bxFlowVerify,
  bxtRunOffload,
  bxSpecRow, bxSpecRows, fetchBackend, bxWeightedBox, bxEveryBox, bxSetPlace,
  bxSetAllPlaces, bxApplyOffload, bxOffloadPlan, bxOffloadRatio, bxBoxWeightTotal,
  bxGB, bxCard, bxWeightSplit, bxNodeWeightBytes, bxStateText, bxNumText,
  bxFlowIngest, bxFlowClear, bxFlowBeginStep, bxRunResult, bxPathId, bxNodePath,
  bxMarkRunError, bxRunErrClear, bxRunStatusClear,
  get bxRunErr() { return bxRunErr; },
  bxBindModels, bxModelNeeded, bxUnboundModels,
  get bxModels() { return bxModels; }, set bxModels(v) { bxModels = v; },
  bxTaskLabel, bxTaskChip, bxTaskProgress, bxAgentShape, bxTaskStart, bxTaskCmd,
  bxTaskFraction, bxArcPath, bxTaskSupersede,
  bxTasksFetch, bxTaskLogFetch, bxTasksApply, bxLogApply, bxLogState,
  bxAgentFill, bxAgentSync, bxAgentResize, bxAgentGrip, bxNodeW, bxNodeH, bxDblClick,
  bxRowsParse, bxRowsText, bxRowsFormat, bxDatasetFill, bxDatasetExpand, bxTrainFill, bxTrainHub,
  bxRunFeeds, bxInputFill, bxInputExpand, bxOutputFill, bxOutputExpand, bxPaneFill,
  bxHotSet, bxHotMove, bxTipHide, bxRunSet, bxRunCanvas, bxFeedsItself,
  bxCsvParse, bxCsvWrite, bxIsBlob, bxBlobIsVideo, bxColLetter, bxDtypeInfer,
  bxTableRead, bxTableWrite, bxTableType, bxTableSetCell, bxTableRenameCol,
  bxTableInsertCol, bxTableDeleteCol, bxTableAddRow, bxTableDeleteRow,
  bxTableAttach,
  get BXTASKS() { return BXTASKS; }, set BXTASKS(v) { BXTASKS = v; },
  bxMarkAudit, bxAuditClear, get bxAudit() { return bxAudit; },
  get bxFlowSet() { return bxFlowSet; }, get bxFlowNow() { return bxFlowNow; },
  get bxDataState() { return bxDataState; }, set bxDataState(v) { bxDataState = v; },
  get bxRunValues() { return bxRunValues; }, set bxRunValues(v) { bxRunValues = v; },
  get BXRUN() { return BXRUN; }, set BXRUN(v) { BXRUN = v; },
  bxCopySelection, bxCopyBoxes, bxPasteBoxes, bxPastePoint, bxOnCopy, bxOnPaste, bxTyping,
  bxLevelOff, bxLevelFrozen, bxBoxOff, bxBoxFrozen, bxOffHere, bxInnerOutOff, bxWarnSet,
  get bxCopied() { return bxCopied; },
  get BX() { return BX; },
};`);

const __t = globalThis.__t;
// the categorized right-click menus: open a category, list what it holds
const menuCats = (m) => m.children.filter(c => String(c.className).split(/\s+/).includes("drop"));
const openCat = (m, label) => {
  const h = menuCats(m).find(c => c.textContent === label);
  if (h) h.click();
  return h;
};
const catItems = (m) => m.children.filter(c => c.className === "mi")
  .map(c => c.textContent).filter(t => t !== "remove this box");   // remove stays top level
// the text a box's window shows (its data and its hints) — one class per query,
// because the stub's selector walk only matches a single class
const paneText = (n) => {
  const d = document.createElement("div");
  __t.bxPaneFill(d, n);
  return d.querySelectorAll(".bxdata").map(e => e.textContent).join(" | ") + " || " +
         d.querySelectorAll(".panehint").map(e => e.textContent).join(" | ");
};
// the app now boots with its blank-network skeleton (input → output); the
// checks below start from a truly empty canvas, as they always have
for (const n of __t.BX.nodes.slice()) __t.bxRemove(n.id);
let failed = 0;
const pending = [];                             // async checks: the summary waits for them
function check(name, fn) {
  const done = (e) => {
    if (e) { failed++; console.log("  FAIL " + name + " — " + e.message);
             if (process.env.STACK) console.log(e.stack); }
    else console.log("  ok   " + name);
  };
  try {
    const out = fn();
    if (out && typeof out.then === "function") pending.push(out.then(() => done(), done));
    else done();
  } catch (e) { done(e); }
}
const eql = (a, b, msg) => assert.deepStrictEqual(a, b, msg || (JSON.stringify(a) + " != " + JSON.stringify(b)));
const ok = (v, msg) => assert.ok(v, msg);
const edgeOf = (from, to, fp, tp) => __t.BX.edges.find(e => e.from === from && e.to === to &&
  (e.fromPort || 0) === (fp || 0) && (e.toPort || 0) === (tp || 0)) || null;

console.log("ui_boot: step 1 — one canvas, three boxes");

check("three boxes: input (out dot only), layer (both), output (in dot only)", () => {
  const a = __t.bxAdd("input", 100, 80, { dout: 5120 });
  const l = __t.bxAdd("layer", 400, 80, { name: "layer 1", din: 5120, dout: 256 });
  const o = __t.bxAdd("output", 700, 80, { din: 256 });
  eql([a.kind, l.kind, o.kind], ["input", "layer", "output"]);
  eql(a.din, null, "an input box has no input");
  eql(a.dout, 5120);
  eql(o.dout, null, "an output box has no output");
  eql(o.din, 256);
  eql([l.din, l.dout], [5120, 256], "a layer box has both sides");
  eql(__t.BX.nodes.length, 3);
  eql(__t.bxShape(a), "5120", "input shows its size only");
  eql(__t.bxShape(o), "256", "output shows its size only");
  eql(__t.bxShape(l), "in 5120 → out 256", "layer shows in → out");
});

check("every box keeps the same fields (and is configured, not retyped)", () => {
  const n = __t.BX.nodes.find(b => b.kind === "layer");
  eql(Object.keys(n).sort(), ["din", "dout", "dtype", "id", "kind", "name", "x", "y"],
    "a box carries exactly name · in · out · kind · place · dtype");
});

check("rename and resize are separate, and work in place on every kind", () => {
  const l = __t.BX.nodes.find(n => n.kind === "layer");
  const keep = { id: l.id, x: l.x, y: l.y };
  const wires = __t.BX.edges.filter(e => e.from === l.id || e.to === l.id).length;
  // the two forms ask only for what they do (fields sit inside their labels)
  const fieldsOf = () => {
    const wrap = getEl("bxmenu").children[0], out = [];
    for (const c of wrap.children || []) {
      if (c.type) out.push(c);
      for (const d of c.children || []) if (d.type) out.push(d);
    }
    return out.map(i => i.type);
  };
  __t.bxForm("layer", null, l, "rename");
  eql(fieldsOf(), ["text"], "rename asks for the name alone");
  __t.bxForm("layer", null, l, "resize");
  eql(fieldsOf(), ["number", "number"], "resize asks for in and out alone");
  __t.bxForm("input", null, __t.BX.nodes.find(n => n.kind === "input"), "resize");
  eql(fieldsOf(), ["number"], "…and an input box for its one size");
  __t.bxMenuClose();
  // renaming leaves the sizes alone
  eql(__t.bxConfigure(l.id, { name: "attn" }), true);
  eql([l.name, l.din, l.dout], ["attn", l.din, l.dout], "a rename touches only the name");
  // resizing leaves the name alone
  __t.bxConfigure(l.id, { din: 5120, dout: 4096 });
  eql([l.name, l.din, l.dout], ["attn", 5120, 4096], "a resize touches only the sizes");
  eql(__t.bxShape(l), "in 5120 → out 4096", "…and the card's shape follows");
  eql([l.id, l.x, l.y], [keep.id, keep.x, keep.y], "the box keeps its identity and place");
  eql(__t.BX.edges.filter(e => e.from === l.id || e.to === l.id).length, wires,
    "…and its wires are untouched");
  // every kind has a name and can be renamed
  const i0 = __t.BX.nodes.find(n => n.kind === "input");
  const o0 = __t.BX.nodes.find(n => n.kind === "output");
  ok(/^input \d+$/.test(i0.name) && /^output \d+$/.test(o0.name),
    "input/output arrive named: " + i0.name + ", " + o0.name);
  __t.bxConfigure(i0.id, { name: "prompt" });
  eql(i0.name, "prompt", "an input box renames");
  __t.bxConfigure(i0.id, { dout: 2048 });
  eql([i0.dout, i0.din, i0.name], [2048, null, "prompt"], "…and resizes, keeping its name");
  eql(__t.bxShape(i0), "2048");
  __t.bxConfigure(o0.id, { din: -5 });
  eql(o0.din, 1, "a nonsense size falls back to something sane");
  __t.bxConfigure(o0.id, { din: 300 });
  eql(o0.din, 300);
  eql(__t.bxDefaultName("layer"), "fully connected " + (__t.BX.nodes.filter(n => n.kind === "layer").length + 1),
    "new boxes keep counting");
  __t.bxMenuClose();
});

check("wiring: out dot → in dot, one wire per input", () => {
  const [a, l, o] = __t.BX.nodes;
  __t.bxConfigure(a.id, { dout: 5120 });            // the sizes this check wires
  __t.bxConfigure(l.id, { din: 5120, dout: 256 });
  __t.bxConfigure(o.id, { din: 256 });
  eql(__t.bxConnect(a.id, l.id), null, "input → layer connects");
  eql(__t.bxConnect(l.id, o.id), null, "layer → output connects");
  eql(__t.BX.edges.length, 2);
  const l2 = __t.bxAdd("layer", 400, 240, { name: "layer 2", din: 5120, dout: 256 });
  __t.bxConnect(a.id, l2.id);
  eql(__t.BX.edges.length, 3, "an output may feed several inputs");
  __t.bxConnect(l2.id, o.id);
  eql(__t.BX.edges.length, 3, "…but an input dot keeps exactly one wire");
  ok(__t.BX.edges.some(e => e.from === l2.id && e.to === o.id), "…the newest one");
  ok(!__t.BX.edges.some(e => e.from === l.id && e.to === o.id), "…replacing the old");
  ok(__t.bxConnect(o.id, l.id), "an output box cannot feed");
  ok(__t.bxConnect(l.id, a.id), "an input box cannot take");
  ok(__t.bxConnect(l.id, l.id), "a box cannot feed itself");
});

check("moving a box does not drag the camera", () => {
  __t.bxRender();
  const view = Object.assign({}, __t.BX.view);
  assert.ok(view.cx != null, "the view is materialised, not re-fitted per render");
  const l2 = __t.BX.nodes.find(n => n.name === "layer 2");
  __t.bxMoveTo(l2.id, 900, 300);
  eql([l2.x, l2.y], [900, 300]);
  eql([__t.BX.view.cx, __t.BX.view.cy, __t.BX.view.k], [view.cx, view.cy, view.k],
    "…so the canvas stays put and the box moves under the cursor");
  const mine = __t.BX.edges.filter(e => e.from === l2.id || e.to === l2.id).length;
  eql(mine, 2, "the box has two wires");
  __t.bxRemove(l2.id);
  eql(__t.BX.nodes.length, 3, "the box is gone");
  eql(__t.BX.edges.filter(e => e.from === l2.id || e.to === l2.id).length, 0,
    "…and both wires went with it");
  ok(__t.BX.edges.length > 0, "the other wires stay");
});

check("the right-click menus: three add items, or remove", () => {
  __t.bxMenuBuild("canvas", null, { clientX: 200, clientY: 150 });
  const menu = getEl("bxmenu");
  const heads = () => menu.children.filter(c => String(c.className).split(/\s+/).includes("drop"));
  const openItems = () => menu.children.filter(c => c.className === "mi").map(c => c.textContent);
  eql(menu.children.map(c => c.textContent),
    ["boundaries", "weights", "text", "tensor ops", "blocks", "containers", "the design"],
    "the categories are listed, all collapsed");
  ok(openItems().length === 0, "…with no items in sight");
  heads()[0].click();
  eql(menu.children.map(c => c.textContent),
    ["boundaries", "input box", "output box", "weights", "text", "tensor ops",
     "blocks", "containers", "the design"],
    "boundaries drops down into its boxes");
  eql(heads()[0].querySelectorAll(".chev")[0].textContent, "\u25be", "…and wears the open chevron");
  heads().find(c => c.textContent === "tensor ops").click();
  eql(openItems(), ["activation box", "operation box", "script box", "sample box"],
    "tensor ops replaces boundaries — one category at a time");
  heads().find(c => c.textContent === "the design").click();
  eql(openItems(), ["export the canvas as JSON", "import JSON …"],
    "the design drops down into its JSON items");
  heads().find(c => c.textContent === "the design").click();
  ok(openItems().length === 0, "…and clicking it again collapses");
  __t.bxMenuBuild("box", __t.BX.nodes[1].id, {});
  const bmenu = getEl("bxmenu");
  eql(bmenu.children.map(c => c.textContent),
    ["this box", "weights", "flow", "remove this box"],
    "a sized box: its categories, and remove");
  openCat(bmenu, "this box");
  eql(catItems(bmenu), ["rename this box", "copy this box", "resize this box"],
    "this box holds rename, copy and resize");
  openCat(bmenu, "weights");
  eql(catItems(bmenu), ["set the dtype", "\u25cf load on the gpu \u2014 video memory",
                        "\u00a0\u00a0 load on the system \u2014 this machine's ram",
                        "load external weights …"],
    "weights holds the dtype, the placement and the files");
  openCat(bmenu, "flow");
  eql(catItems(bmenu), ["deactivate this box", "freeze this box"],
    "flow holds the two switches");
  ok(bmenu.children.map(c => c.textContent).indexOf("remove this box") >= 0,
    "…and remove stays one click away");
  // the layer kind is called Linear
  eql(__t.bxKindLabel("layer"), "Fully Connected", "a layer box is labelled Fully Connected");
  eql(__t.bxKindLabel("custom"), "Nested", "…and a custom box is called Nested");
  eql(__t.bxKindLabel("input"), "input", "…the others keep their names");
  eql(__t.bxKindLabel("train"), "Train", "…the train kind is labelled Train");
  eql(__t.bxDefaultName("train"), "train 1", "…and a fresh one is named train 1");
  const tmp = __t.bxAdd("layer", 0, 0, {});
  ok(/^fully connected \d+$/.test(tmp.name), "a new one is named like: " + tmp.name);
  __t.bxRemove(tmp.id);
  __t.bxMenuClose();
});

check("the canvas view: fit, zoom, pan maths", () => {
  __t.bxFit();
  const v1 = __t.BX.view;
  ok(v1.k > 0 && v1.k <= 1.2, "fit picks a sane zoom: " + v1.k);
  ok(typeof __t.bxViewBox() === "string" && __t.bxViewBox().split(" ").length === 4, "a viewBox");
  __t.bxZoomAt({ clientX: 300, clientY: 200 }, 1.25);
  ok(__t.BX.view.k > v1.k, "the wheel zooms in");
  __t.bxZoomAt({ clientX: 300, clientY: 200 }, 0.8);
  ok(Math.abs(__t.BX.view.k - v1.k) < 0.001, "…and back");
});

check("every wire renders its own hit stroke (what hovering needs)", () => {
  const a = __t.BX.nodes.find(n => n.kind === "input");
  const l = __t.BX.nodes.find(n => n.kind === "layer");
  if (!edgeOf(a.id, l.id)) __t.bxConnect(a.id, l.id, 0, 0);
  const host = getEl("bxcanvas");
  const vis = host.querySelectorAll(".dedge").length, hit = host.querySelectorAll(".dedge-hit").length;
  ok(vis >= 1 && hit === vis, "one hit stroke per wire: " + hit + "/" + vis);
  ok(host.querySelectorAll(".dedge-hit").every(p => String(p.className).indexOf("dedge-hit") === 0),
    "…each carrying its own class, not the wire's");
});

check("the canvas draws a grid that follows the view", () => {
  __t.bxRender();
  const host = getEl("bxcanvas");
  const gridOf = () => {
    const root = host.children[0];
    return root && (root.children || []).find(c => String(c.className).includes("bxgrid"));
  };
  const grid = gridOf();
  ok(grid, "a grid group is drawn");
  eql(grid.children.length, 2, "…with minor and major paths");
  const d = String(grid.children[0]._attrs.d || "");
  ok(d.indexOf("M") === 0 && d.length > 20, "…tiled with lines across the view");
  const before = String(grid.children[1]._attrs.d);
  __t.bxZoomAt({ clientX: 300, clientY: 200 }, 2);
  ok(String(gridOf().children[1]._attrs.d) !== before, "…re-stepping as the view zooms");
  __t.bxZoomAt({ clientX: 300, clientY: 200 }, 0.5);    // back to 1× for the rest
});

check("the run discards the boxes it never touches", () => {
  const errs = () => __t.bxErrorSet();
  const i = __t.bxAdd("input", 0, 0, { dout: 4, values: "1 2 3 4" });
  const l = __t.bxAdd("layer", 200, 0, { din: 4, dout: 4 });
  const o = __t.bxAdd("output", 400, 0, { din: 4 });
  __t.bxConnect(i.id, l.id);
  __t.bxConnect(l.id, o.id);
  const stray = __t.bxAdd("layer", 0, 300, { din: 8, dout: 8 });      // no wires at all
  const branch = __t.bxAdd("layer", 200, 300, { din: 4, dout: 8 });   // fed, leads nowhere
  __t.bxConnect(i.id, branch.id);
  const spare = __t.bxAdd("input", 0, 500, { dout: 8 });              // nothing reads it
  eql([errs().has(stray.id), errs().has(branch.id), errs().has(spare.id)],
      [false, false, false], "a stray box, a dangling branch and a spare input: no errors");
  eql([errs().has(i.id), errs().has(l.id), errs().has(o.id)], [false, false, false],
      "\u2026and nothing on the run's own path is marked");
  const runDoc = __t.bxRunCanvas(__t.bxExportCanvas());
  const ids = runDoc.nodes.map(n => n.id);
  eql([ids.indexOf(stray.id), ids.indexOf(branch.id), ids.indexOf(spare.id)], [-1, -1, -1],
      "the run drops every box it never touches");
  ok(ids.indexOf(i.id) >= 0 && ids.indexOf(l.id) >= 0 && ids.indexOf(o.id) >= 0,
     "\u2026and keeps the path from the input to the output");
  eql(runDoc.edges.filter(e => e.from === l.id || e.to === l.id).length, 2,
      "\u2026with that path's wires alone");
  eql(__t.bxExportCanvas().nodes.map(n => n.id).indexOf(stray.id) >= 0, true,
      "the design itself keeps every box");
  [stray.id, branch.id, spare.id, i.id, l.id, o.id].forEach(x => __t.bxRemove(x));
});

check("a wire is deletable on the spot (hover × , or right-click)", () => {
  const a = __t.BX.nodes.find(n => n.kind === "input");
  const l = __t.BX.nodes.find(n => n.kind === "layer");
  eql(__t.BX.edges.length, 1, "one wire is there");
  eql(__t.bxRemoveEdge(a.id, l.id), true, "…and goes");
  eql(__t.BX.edges.length, 0, "…alone");
  eql(__t.BX.nodes.length, 3, "the boxes stay");
  eql(__t.bxRemoveEdge(a.id, l.id), false, "removing it twice is a no-op");
  __t.bxMenuBuild("edge", a.id + ">" + l.id, {});
  eql(getEl("bxmenu").children.map(c => c.textContent), ["remove this wire"],
    "right-clicking a wire offers the same");
  __t.bxMenuClose();
  __t.bxConnect(a.id, l.id);
  eql(__t.BX.edges.length, 1, "…and it can be wired again");
});

check("the delete × sits in the middle of the wire", () => {
  const m = __t.bxWireMid({ x: 0, y: 100 }, { x: 400, y: 100 });
  eql(Math.round(m.x), 280, "halfway between the two dots (160 → 400)");
  eql(Math.round(m.y), 140, "on the wire's line (y + half a card)");
  const m2 = __t.bxWireMid({ x: 0, y: 100 }, { x: 168, y: 100 });
  ok(m2.x >= 160 && m2.x <= 200, "a very short wire still parks it sensibly: " + Math.round(m2.x));
});

check("hovering a wire reads out the data on it: type, dtype, shape", () => {
  const a = __t.BX.nodes.find(n => n.kind === "input");
  const e = __t.BX.edges[0];
  const tip = getEl("bxtip");
  const line = cls => {
    const hit = tip.querySelectorAll("." + cls)[0];
    return hit ? String(hit.textContent) : null;
  };
  const ev = { clientX: 200, clientY: 120 };
  __t.bxHotSet(e, ev);
  eql(tip.hidden, false, "the card shows while the wire is hot");
  eql(line("who"), "from " + a.name, "…it names where the data comes from");
  eql(line("what"), "matrix", "…and its kind before a run: " + line("what"));
  eql(line("shp"), "[? × 5120]", "…and the shape the wiring declares");
  eql(line("note"), null, "…so no run is needed for the shape");
  eql(tip.style.left, "216px", "…placed by the cursor (x): " + tip.style.left);
  // after a run the card carries the type, the dtype and the shape
  const path = a.id;
  __t.bxRunValues = { [path]: { shape: [12], dtype: "i64", mean: 2, peek: [1, 2] } };
  __t.bxHotSet(e, ev);
  eql(line("what"), "ids", "an integer stream reads as ids: " + line("what"));
  eql(line("shp"), "[12]", "…with its shape");
  __t.bxRunValues = { [path]: { shape: [], dtype: "f32", scalar: 0.5,
                                mean: 0.5, peek: [0.5] } };
  __t.bxHotSet(e, ev);
  eql(line("what"), "matrix · f32", "a lone number is a matrix too: " + line("what"));
  eql(line("shp"), "[1]", "…one element wide");
  __t.bxRunValues = { [path]: { shape: [4, 8], dtype: "f32", mean: 0, peek: [0] } };
  __t.bxHotSet(e, ev);
  eql(line("what"), "matrix \u00b7 f32", "a matrix carries its dtype: " + line("what"));
  eql(line("shp"), "[4\u00d78]", "…and its shape");
  __t.bxRunValues = { [path]: { text: "the cat sat on the mat" } };
  __t.bxHotSet(e, ev);
  eql(line("what"), "text", "a string reads as text");
  ok(String(line("shp")).indexOf("the cat sat") > 0, "…shown, not measured: " + line("shp"));
  __t.bxRunValues = { [path]: { shape: [64, 64, 3], dtype: "f32",
                                image: { w: 64, h: 64, blob: "" } } };
  __t.bxHotSet(e, ev);
  eql(line("what"), "image", "a picture reads as an image");
  eql(line("shp"), "[64\u00d764\u00d73]", "…as an RGB matrix");
  __t.bxRunValues = {};
  // nothing declares this one: an opaque op, until a run fills the shape in
  const op = __t.bxAdd("op", 0, 0, { op: "matmul" });
  const l2 = __t.bxAdd("layer", 200, 0, { din: 8, dout: 8 });
  __t.bxConnect(op.id, l2.id);
  const e2 = __t.BX.edges.find(x => x.from === op.id && x.to === l2.id);
  __t.bxHotSet(e2, ev);
  eql(line("shp"), null, "an opaque op declares no shape…");
  eql(line("note"), "not run yet", "…so it says so until a run: " + line("note"));
  __t.bxRemove(op.id);
  __t.bxRemove(l2.id);
  // …while the wiring tracks the shape back: a layer's width here, and the same
  // width read through a pass-through box on the way
  const l3 = __t.bxAdd("layer", 0, 0, { din: 8, dout: 32 });
  const ac = __t.bxAdd("act", 200, 0, { act: "silu" });
  const l4 = __t.bxAdd("layer", 400, 0, { din: 32, dout: 4 });
  __t.bxConnect(l3.id, ac.id);
  __t.bxConnect(ac.id, l4.id);
  __t.bxHotSet(__t.BX.edges.find(x => x.from === l3.id && x.to === ac.id), ev);
  eql(line("shp"), "[? × 32]", "a layer's width is declared…");
  __t.bxHotSet(__t.BX.edges.find(x => x.from === ac.id && x.to === l4.id), ev);
  eql(line("shp"), "[? × 32]", "…and a pass-through reads it off its neighbour");
  [l3, ac, l4].forEach(x => __t.bxRemove(x.id));
  // a shaped reshape knows its own dims — still behind the batch axis
  const rs = __t.bxAdd("op", 0, 0, { op: "reshape", dims: "8x16" });
  const l5 = __t.bxAdd("layer", 200, 0, { din: 128, dout: 4 });
  __t.bxConnect(rs.id, l5.id);
  __t.bxHotSet(__t.BX.edges.find(x => x.from === rs.id && x.to === l5.id), ev);
  eql(line("shp"), "[? × 8×16]", "a shaped reshape declares its dims");
  __t.bxRemove(rs.id);
  __t.bxRemove(l5.id);
  __t.bxHotSet(null);
  eql(tip.hidden, true, "…and the card goes when the wire is let go");
});

check("hovering an input box names it and the type it emits", () => {
  const tip = getEl("bxtip");
  const line = cls => {
    const hit = tip.querySelectorAll("." + cls)[0];
    return hit ? String(hit.textContent) : null;
  };
  __t.bxHotSet(null);
  const i = __t.bxAdd("input", 0, 0, { dout: 5120 });
  __t.bxHotMove({ target: __t.bxCard(i), clientX: 140, clientY: 90 });
  eql(tip.hidden, false, "the card shows over the box");
  eql(line("who"), "input", "…it names the kind");
  eql(line("what"), i.name, "…and the box's name");
  eql(line("shp"), "emits matrix [? × 5120]", "…and the type it will emit");
  eql(line("note"), null, "no run value yet: " + line("note"));
  // the run's own shape rides under the expectation
  __t.bxRunValues = { [i.id]: { shape: [4, 5120], dtype: "f32", mean: 1, peek: [1] } };
  __t.bxHotSet(null);
  __t.bxHotMove({ target: __t.bxCard(i), clientX: 140, clientY: 90 });
  eql(line("note"), "last run [4×5120] · f32",
      "…with the last run's own shape: " + line("note"));
  // the box's own window is editing space: no card there
  const fo = __t.bxCard(i).querySelectorAll(".bxlogfo")[0];
  __t.bxHotSet(null);
  __t.bxHotMove({ target: fo, clientX: 140, clientY: 90 });
  eql(tip.hidden, true, "…and the card stays out of the box's own window");
  __t.bxRunValues = {};
  __t.bxHotSet(null);
  __t.bxRemove(i.id);
  // an image input knows the picture's shape
  const im = __t.bxAdd("input", 0, 0, { input_mode: "image", dout: 3 });
  im.image_size = 512;                       // set by the editor, not at add time
  __t.bxHotMove({ target: __t.bxCard(im), clientX: 140, clientY: 90 });
  eql(line("shp"), "emits image [512×512×3]",
      "an image input emits its picture: " + line("shp"));
  __t.bxHotSet(null);
  __t.bxRemove(im.id);
});

check("hovering an output box says what it expects", () => {
  const tip = getEl("bxtip");
  const line = cls => {
    const hit = tip.querySelectorAll("." + cls)[0];
    return hit ? String(hit.textContent) : null;
  };
  __t.bxHotSet(null);
  const o = __t.bxAdd("output", 0, 0, { din: 128 });
  __t.bxHotMove({ target: __t.bxCard(o), clientX: 140, clientY: 90 });
  eql(line("who"), "output", "the card names the kind");
  eql(line("what"), o.name, "…and the box's name");
  eql(line("shp"), "expects matrix [? × 128]", "…and what it takes");
  __t.bxRunValues = { [o.id]: { shape: [64, 64, 3], dtype: "f32",
                                image: { w: 64, h: 64, blob: "" } } };
  __t.bxHotSet(null);
  __t.bxHotMove({ target: __t.bxCard(o), clientX: 140, clientY: 90 });
  eql(line("note"), "last run image · [64×64×3]",
      "…a picture that arrived: " + line("note"));
  __t.bxRunValues = {};
  __t.bxHotSet(null);
  __t.bxRemove(o.id);
});

check("a nested box's dots and body read as its own i/o boxes", () => {
  const tip = getEl("bxtip");
  const line = cls => {
    const hit = tip.querySelectorAll("." + cls)[0];
    return hit ? String(hit.textContent) : null;
  };
  __t.bxHotSet(null);
  const c = __t.bxAdd("custom", 400, 0, { name: "Inner-Model" });
  c.inner = { type: "nested", version: 1,
    nodes: [
      { id: "k1", kind: "input", name: "tokens", x: 0, y: 0, dout: 12 },
      { id: "k2", kind: "output", name: "head", x: 320, y: 0, din: 128 }],
    edges: [{ from: "k1", to: "k2" }] };
  const g = __t.bxCard(c);
  const dot = g.querySelectorAll(".dport").find(d => String(d.className).indexOf(" in") >= 0);
  ok(dot, "the nested card has an in dot");
  __t.bxHotMove({ target: dot, clientX: 120, clientY: 80 });
  eql(line("who"), "input", "a dot reads as the inner input box");
  eql(line("what"), "tokens", "…its name");
  eql(line("shp"), "emits matrix [? × 12]", "…and what it emits");
  eql(line("note"), "port 1 of Inner-Model", "…and which dot it is");
  __t.bxHotSet(null);
  __t.bxHotMove({ target: g, clientX: 120, clientY: 80 });
  eql(line("who"), "nested box", "the body lists its dots");
  eql(line("what"), "Inner-Model", "…under the box's name");
  eql(tip.querySelectorAll(".row").map(r => String(r.textContent)),
      ["in  tokens · matrix [? × 12]", "out head · matrix [? × 128]"],
      "…one row per dot, the inner boxes' own metadata");
  const outDot = g.querySelectorAll(".dport").find(d => String(d.className).indexOf(" out") >= 0);
  __t.bxHotSet(null);
  __t.bxHotMove({ target: outDot, clientX: 120, clientY: 80 });
  eql(line("who"), "output", "an out dot reads as the inner output box");
  eql(line("shp"), "expects matrix [? × 128]", "…and what it takes");
  eql(line("note"), "port 1 of Inner-Model", "…port 1 of the nested box");
  // the real DOM hands an SVG dot's .className as an SVGAnimatedString object,
  // and closest() reads the class attribute — emulate that exactly
  const svgDot = {
    className: { baseVal: "dport hit out", animVal: "dport hit out" },
    getAttribute: (k) => (k === "class" ? "dport hit out"
                      : k === "data-port" ? "0" : null),
    closest: (sel) => (sel === ".dport" ? svgDot : (sel === ".dnode" ? g : null)),
  };
  __t.bxHotSet(null);
  __t.bxHotMove({ target: svgDot, clientX: 120, clientY: 80 });
  eql(line("who"), "output", "…even when the DOM hands .className as an object");
  eql(line("shp"), "expects matrix [? × 128]", "…still that box's own shape");
  __t.bxHotSet(null);
  __t.bxRemove(c.id);
});

check("no browser tooltips are left on the canvas", () => {
  const i = __t.bxAdd("input", 0, 0, { dout: 4 });
  const l = __t.bxAdd("layer", 200, 0, { din: 4, dout: 4 });
  const o = __t.bxAdd("output", 400, 0, { din: 4 });
  __t.bxConnect(i.id, l.id);
  __t.bxConnect(l.id, o.id);
  __t.bxRender();
  const titles = (root) => {
    const out = [];
    (function walk(e) {
      for (const c of e.children || []) {
        if (String(c.id) === "<title>") out.push(c);
        walk(c);
      }
    })(root);
    return out;
  };
  eql(titles(getEl("bxcanvas")).length, 0,
      "no <title> on a box, a dot or a wire after a render");
  const e = __t.BX.edges.find(x => x.from === i.id && x.to === l.id);
  __t.bxHotSet(e, { clientX: 120, clientY: 90 });
  eql(titles(getEl("bxcanvas")).length, 0, "\u2026and the hover card adds none");
  __t.bxHotSet(null);
  [i, l, o].forEach(x => __t.bxRemove(x.id));   // only what this check added
});

check("a wire can be shaped: bend it, snap it straight, reset it", () => {
  const a = __t.BX.nodes.find(n => n.kind === "input");
  const l = __t.BX.nodes.find(n => n.kind === "layer");
  const e = __t.BX.edges[0];
  eql([e.from, e.to], [a.id, l.id], "the wire under test");
  const nat = __t.bxNaturalMid(e);
  ok(!e.via, "wires run straight by default");
  eql(__t.bxSetVia(e, 640, 300), true, "a drag marks a bend");
  eql(e.via, { x: 640, y: 300 }, "…where it was dropped");
  eql(__t.bxWireMid(a, l, e.via), { x: 640, y: 300 }, "the curve passes through that point");
  ok(__t.bxWirePath(a, l, e.via).indexOf("C") > 0, "…and is still one smooth cubic");
  ok(Math.hypot(e.via.x - nat.x, e.via.y - nat.y) > 14, "…far from the straight middle");
  __t.bxMenuBuild("edge", [a.id, l.id, 0, 0].join(">"), {});
  eql(getEl("bxmenu").children.map(c => c.textContent),
    ["reset the wire's shape", "remove this wire"], "a bent wire offers a reset");
  __t.bxMenuClose();
  eql(__t.bxClearVia(e), true, "reset puts the straight shape back");
  ok(!e.via, "…with no bend left");
  eql(__t.bxClearVia(e), false, "resetting a straight wire is a no-op");
  __t.bxMenuBuild("edge", [a.id, l.id, 0, 0].join(">"), {});
  eql(getEl("bxmenu").children.map(c => c.textContent), ["remove this wire"],
    "…and only then does the menu drop the reset item");
  __t.bxMenuClose();
});

check("click selects; Delete removes the box (with its wires) or just the wire", () => {
  const l = __t.BX.nodes.find(n => n.kind === "layer");
  __t.bxSelectBoxes([l.id]);
  eql(__t.bxSelBox(l.id), true, "the box is selected");
  eql(__t.bxDeleteSelection(), true, "Delete takes it");
  eql(__t.BX.nodes.some(n => n.id === l.id), false, "…the box is gone");
  eql(__t.BX.edges.some(e => e.from === l.id || e.to === l.id), false, "…and its wires with it");
  eql(__t.bxDeleteSelection(), false, "Delete with nothing selected does nothing");
  const a = __t.BX.nodes.find(n => n.kind === "input");
  const o = __t.BX.nodes.find(n => n.kind === "output");
  __t.bxConfigure(o.id, { din: a.dout });           // matching sizes wire cleanly
  eql(__t.bxConnect(a.id, o.id), null, "a matching pair connects");
  __t.bxSelect({ kind: "edge", from: a.id, to: o.id });
  eql(__t.bxSelEdge({ from: a.id, to: o.id }), true, "a wire can be selected too");
  eql(__t.bxDeleteSelection(), true, "…and Delete takes it");
  eql(__t.BX.edges.length, 0, "…only it");
  eql(__t.BX.nodes.length, 2, "the boxes stay: " + __t.BX.nodes.map(n => n.kind).join(", "));
  __t.bxSelectBoxes([o.id]);
  __t.bxRemove(o.id);
  eql(__t.bxSel, null, "removing the selected box clears the selection");
  __t.bxSelect(null);
});

check("the Palette menu: Input and Output up top, categories below", () => {
  __t.bxRenderPalette();
  const menu = getEl("bxtPalMenu");
  const top = menu.children.filter(c => c.className === "bxtcards")[0];
  eql(top.querySelectorAll(".bxpalitem").map(i => i.getAttribute("data-kind")),
    ["input", "output"], "Input and Output are the first cards, not a category");
  eql(menu.children.filter(c => c.className === "bxtcat").map(c => c.textContent),
    ["AI", "weights", "text", "Dataset", "tensor ops", "blocks", "containers"],
    "and only the categories are listed below");
  // the AI category holds the agent box
  menu.children.find(c => c.textContent === "AI").click();
  eql(menu.querySelectorAll(".bxpalitem").map(i => i.getAttribute("data-kind")),
    ["input", "output", "agent"], "AI holds the agent box");
  menu.children.find(c => c.textContent === "AI").click();
  // expand a category: its visual boxes appear, drawn by the card renderer
  menu.children.find(c => c.textContent === "weights").click();
  const cards = menu.querySelectorAll(".bxpalitem");
  eql(cards.map(i => i.getAttribute("data-kind")),
    ["input", "output", "embed", "layer", "norm"], "weights expands below the two cards");
  ok(cards.slice(2).every(i => i.children[0] && i.children[0]._attrs &&
                              i.children[0]._attrs.viewBox),
    "…each drawn by the same card renderer");
  // the text category holds the tokenizer pair and the hf model
  menu.children.find(c => c.textContent === "text").click();
  eql(menu.querySelectorAll(".bxpalitem").map(i => i.getAttribute("data-kind")),
    ["input", "output", "tokenize", "detokenize", "hf"],
    "text holds the tokenizer pair and the huggingface model");
  // the Dataset category holds only the dataset box; the Train box has its
  // own button up top
  menu.children.find(c => c.textContent === "Dataset").click();
  eql(menu.querySelectorAll(".bxpalitem").map(i => i.getAttribute("data-kind")),
    ["input", "output", "dataset"],
    "Dataset holds only the dataset box (the Train box has its own button)");
  // one category at a time: opening another closes the first
  menu.children.find(c => c.textContent === "tensor ops").click();
  eql(menu.querySelectorAll(".bxpalitem").map(i => i.getAttribute("data-kind")),
    ["input", "output", "act", "op", "script", "sample"],
    "tensor ops replaces text");
  menu.children.find(c => c.textContent === "tensor ops").click();
  eql(menu.querySelectorAll(".bxpalitem").map(i => i.getAttribute("data-kind")),
    ["input", "output"], "…and clicking it again collapses");
});

check("no GGUF card in the Palette — a model comes in through Import", () => {
  __t.bxRenderPalette();
  const menu = getEl("bxtPalMenu");
  const cards = () => menu.querySelectorAll(".bxpalitem").map(i => i.getAttribute("data-kind"));
  menu.children.find(c => c.textContent === "containers").click();
  eql(cards(), ["input", "output", "custom"],
    "containers is just the nested box, at the top level");
  menu.children.find(c => c.textContent === "containers").click();   // collapse again
  // the canvas right-click menu has no GGUF box either
  __t.bxMenuBuild("canvas", null, { clientX: 10, clientY: 10 });
  const heads = () => getEl("bxmenu").children.filter(
    x => String(x.className).split(/\s+/).includes("drop"));
  heads().find(x => x.textContent === "containers").click();
  eql(getEl("bxmenu").children.filter(x => x.className === "mi").map(x => x.textContent),
      ["nested box"], "…and the canvas menu offers only the nested box");
  __t.bxMenuClose();
});

check("the Import menu: one pop-up, one file, no box spawned", () => {
  ok(getEl("bxtImport"), "the toolbar has an Import button");
  eql(getEl("bxtImportMenu").children.map(c => c.textContent), ["GGUF model …"],
      "…offering a GGUF model");
  __t.bxtImportToggle();
  eql(getEl("bxtImportMenu").hidden, false, "a click opens the menu");
  __t.bxtImportToggle();
  eql(getEl("bxtImportMenu").hidden, true, "…and another closes it");
  const before = __t.BX.nodes.length;
  __t.bxGGUFImport();                          // the item: one pop-up, up front
  const menu = getEl("bxmenu");
  eql(__t.BX.nodes.length, before, "the pop-up spawns no box");
  ok(!menu.hidden && menu.style.left === "130px" && menu.style.top === "115px",
     "…it comes up in the middle of the canvas: " + menu.style.left + " × " + menu.style.top);
  eql(menu.querySelectorAll(".bxsel").length, 1, "…with one model list");
  const text = menu.children[0].children.map(c => c.textContent).join(" | ");
  ok(text.indexOf("nested box") >= 0 && text.indexOf("mmproj") < 0 &&
     text.indexOf("repo") < 0,
     "…that says where the model lands — no mmproj, no repo: " + text);
  __t.bxMenuClose();
  eql(__t.BX.nodes.length, before, "cancelling leaves the canvas exactly as it was");
  // the file chosen: one model, nothing else, and it lands as a nested box
  const realFetch = global.fetch;
  let sent = null;
  global.fetch = async (url, opts) => {
    if (String(url) === "/api/gguf/canvas") sent = JSON.parse(opts.body);
    return { ok: true, status: 200, json: async () => ({ ok: true }) };
  };
  __t.bxGGUFImport();
  const form = getEl("bxmenu");
  form.querySelectorAll(".bxsel")[0].value = "models/test-model.gguf";
  form.querySelectorAll(".chip").find(c => String(c.textContent) === "load").click();
  global.fetch = realFetch;
  eql(sent, { model: "models/test-model.gguf" },
      "the load posts the one file — no mmproj, no repo");
  __t.bxMenuClose();
  const n = __t.bxGGUFImportApply(GGUF_CANVAS, "models/test-model.gguf");
  ok(n && n.kind === "custom" && n.file === "models/test-model.gguf",
     "…and the served model lands as one nested box");
  eql(__t.bxJSONNode(n).gguf, undefined, "…with no GGUF flag in its JSON");
  __t.bxUndoStep();
  ok(__t.BX.nodes.length === before &&
     !__t.BX.nodes.some(x => x.file === "models/test-model.gguf"),
     "…and one undo takes the whole import back");
  __t.bxRedoStep();
  ok(__t.BX.nodes.some(x => x.kind === "custom" &&
                            x.file === "models/test-model.gguf"),
     "…redo lands it again");
  __t.bxUndoStep();                            // leave the canvas as it was found

  // an encoder's served cluster lands the same way: one box, its tower inside
  const enc = __t.bxGGUFImportApply({ name: "vision.gguf",
    nodes: [{ id: "image", kind: "input", input_mode: "image", dout: 3, name: "image" },
            { id: "vision", kind: "vision", name: "tower", din: 3, dout: 12 },
            { id: "image embeds", kind: "output", name: "image embeds", din: 12 }],
    edges: [{ from: "image", to: "vision" },
            { from: "vision", to: "image embeds" }] }, "models/mmproj-x.gguf");
  ok(enc && enc.kind === "custom" && enc.inner.nodes.length === 3 &&
     enc.inner.edges.length === 2,
     "an encoder's cluster lands as one nested box with the tower inside");
  eql([enc.file, __t.bxPortDefs(enc, "in").length, __t.bxPortDefs(enc, "out").length],
      ["models/mmproj-x.gguf", 1, 1],
      "…with its file and the image-in / embeds-out ports");
  __t.bxRemove(enc.id);
  // a whole model loads at the top level only
  const box = __t.bxAdd("custom", 0, 0, {});
  __t.bxEnter(box.id);
  __t.bxtRefresh();
  eql(__t.bxtImportToggle(), false, "inside a nested box the menu refuses");
  ok(String(getEl("bxtImport").className).indexOf("off") >= 0,
     "…and the button reads off");
  __t.bxLeave();
  __t.bxtRefresh();
  __t.bxRemove(box.id);
});

check("a toolbar menu opens under its own button, not the strip's left", () => {
  const strip = getEl("bxtools");
  const files = getEl("bxtFiles"), imp = getEl("bxtImport"), men = getEl("bxtImportMenu");
  const keep = { strip: strip.getBoundingClientRect, files: files.getBoundingClientRect,
                 imp: imp.getBoundingClientRect, w: men.offsetWidth };
  strip.getBoundingClientRect = () => ({ left: 0, top: 0, width: 900, height: 36 });
  files.getBoundingClientRect = () => ({ left: 10, top: 0, width: 58, height: 26 });
  imp.getBoundingClientRect = () => ({ left: 210, top: 0, width: 70, height: 26 });
  men.offsetWidth = 180;
  __t.bxtDropsClose();
  __t.bxtImportToggle();
  eql(men.style.left, "210px", "Import's menu lines up with Import");
  __t.bxtDropsClose();
  __t.bxtFilesToggle();
  eql(getEl("bxtFilesMenu").style.left, "10px", "\u2026and Files' with Files");
  __t.bxtDropsClose();
  getEl("bxtVramMenu").style.left = "";
  __t.bxtVramToggle();
  eql(getEl("bxtVramMenu").style.left, "", "the right-anchored VRAM menu is left alone");
  __t.bxtDropsClose();
  strip.getBoundingClientRect = keep.strip;    // the stubs go back untouched
  files.getBoundingClientRect = keep.files;
  imp.getBoundingClientRect = keep.imp;
  men.offsetWidth = keep.w;
});

check("a Palette card: a click drops it in the middle, a drag lands where you let go", () => {
  __t.bxRenderPalette();
  const menu = getEl("bxtPalMenu");
  menu.children.find(c => c.textContent === "weights").click();      // show some cards
  const before = __t.BX.nodes.length;
  // a press that does not move: the box lands in the middle of the view
  __t.bxPalDown({ button: 0, clientX: 60, clientY: 60, preventDefault() {}, stopPropagation() {} }, "layer");
  __t.bxPalUp({ clientX: 60, clientY: 60 });
  eql(__t.BX.nodes.length, before + 1, "a click places a box");
  eql([__t.BX.nodes[before].kind, __t.BX.nodes[before].din, __t.BX.nodes[before].dout],
    ["layer", null, null], "…empty and unconfigured");
  eql(__t.bxSelBox(__t.BX.nodes[before].id), true, "…and selected, ready to configure");
  // the drag gesture: press, move over the canvas, release there
  const n1 = __t.BX.nodes.length;
  __t.bxPalDown({ button: 0, clientX: 60, clientY: 60, preventDefault() {}, stopPropagation() {} }, "output");
  __t.bxPalMove({ clientX: 300, clientY: 200 });
  ok(__t.bxOverCanvas({ clientX: 300, clientY: 200 }), "the cursor is over the canvas");
  __t.bxPalUp({ clientX: 300, clientY: 200 });
  eql(__t.BX.nodes.length, n1 + 1, "a drag out of the menu lands a box");
  eql(__t.BX.nodes[n1].kind, "output");
  // released outside the canvas: nothing happens
  const n2 = __t.BX.nodes.length;
  __t.bxPalDown({ button: 0, clientX: 60, clientY: 60, preventDefault() {}, stopPropagation() {} }, "input");
  __t.bxPalMove({ clientX: 900, clientY: 200 });
  __t.bxPalUp({ clientX: 900, clientY: 200 });
  eql(__t.BX.nodes.length, n2, "…and a release outside is ignored");
  while (__t.BX.nodes.length > before) __t.bxRemove(__t.BX.nodes[__t.BX.nodes.length - 1].id);
  const head = getEl("bxtPalMenu").children.find(c => c.textContent === "weights");
  if (head) head.click();                                          // collapse again
});

check("the Palette and VRAM menus open from the toolbar", () => {
  ok(getEl("bxtPalMenu").hidden, "the Palette menu starts closed");
  ok(__t.bxtPaletteToggle() === true && !getEl("bxtPalMenu").hidden, "Palette opens");
  ok(__t.bxtVramToggle() === true && !getEl("bxtVramMenu").hidden,
    "opening VRAM closes Palette and opens VRAM");
  ok(getEl("bxtPalMenu").hidden, "…and Palette is shut");
  ok(__t.bxtVramToggle() === false && getEl("bxtVramMenu").hidden, "…and VRAM closes");
});

check("the Backend menu: Local and Remote drop down into their instructions", () => {
  const m = getEl("bxtBackendMenu");
  ok(m.hidden, "the Backend menu starts closed");
  ok(__t.bxtBackendToggle() === true && !m.hidden, "Backend opens");
  const heads = () => m.children.filter(c => c.className === "bxtcat");
  eql(heads().map(c => c.textContent), ["Local", "Remote — Modal (BYOK)"],
    "two collapsed sections");
  ok(m.children.every(c => c.className !== "bxbpane"), "nothing is dropped down yet");
  // Local drops down into the agent instructions
  heads()[0].click();
  let pane = m.children.find(c => c.className === "bxbpane");
  ok(pane, "Local drops down");
  const hint = () => pane.querySelectorAll(".bxbhint").map(e => e.textContent).join(" | ");
  ok(hint().indexOf("the agent") >= 0 && hint().indexOf("contacts this app") >= 0,
    "…with the steps");
  ok(pane.querySelectorAll(".bxbcode").map(e => e.textContent).join("").indexOf("nnstudio_agent.py") >= 0,
    "…and the agent command");
  ok(hint().indexOf("waiting for an agent") >= 0, "…and the waiting line while none is connected");
  // Remote replaces it: the key lives there
  heads()[1].click();
  pane = m.children.find(c => c.className === "bxbpane");
  ok(pane && hint().indexOf("Modal token") >= 0, "Remote drops down instead");
  eql(pane.querySelectorAll(".bxbin")[0].type, "password", "…with the key as a password field");
  // the choice is made inside the pane and remembered; the header wears the dot
  pane.querySelectorAll(".chip").find(c => c.textContent === "use modal").click();
  eql(globalThis.localStorage.getItem("nnstudio_backend"), "remote", "use modal remembers it");
  eql(heads()[1].querySelectorAll(".bxton").length, 1, "…and the section wears the in-use dot");
  heads()[0].click();
  pane = m.children.find(c => c.className === "bxbpane");
  pane.querySelectorAll(".chip").find(c => c.textContent === "use this computer").click();
  eql(globalThis.localStorage.getItem("nnstudio_backend"), "local", "…and back to this computer");
  __t.bxtBackendToggle();
  ok(m.hidden, "…and it closes");
});

check("the Train button: the box on and off the canvas", () => {
  const btn = getEl("bxtTrainBtn");
  const panes = () => getEl("bxcanvas").querySelectorAll(".bxlogpane")
    .map(el => el.getAttribute("data-agent"));
  const trainCount = () => __t.BX.nodes.filter(n => n.kind === "train").length;
  const disconnected = { local: { connected: false, agent: null }, inputs: {}, last: null,
                         loaded: null, loading: false, loadKey: null, msg: "", running: false };
  __t.BX_BACKEND.mode = "local";
  __t.BX_BACKEND.key = "";
  __t.BXRUN = disconnected;
  __t.bxtRunRefresh();
  ok(trainCount() === 0, "the canvas starts with no Train box");
  ok(__t.bxtRunEnabled() === false, "…and no backend is connected");
  ok(String(btn.className).indexOf("off") >= 0, "…so the Train button reads off");
  ok(btn.title === "show the Train box", "…saying what a press will do");
  ok(__t.bxtRunToggle() === true, "the button shows the box");
  const hub = __t.bxTrainHub();
  ok(hub && hub.name === "train 1", "…a fresh Train box named train 1");
  ok(panes().indexOf(hub.id) >= 0, "…drawn on the canvas");
  ok(String(btn.className).indexOf("off") < 0 && btn.title === "hide the Train box",
     "…and the button now offers to hide it");
  ok(__t.bxtRunToggle() === false && hub.hidden === true, "a second press hides the box");
  ok(panes().indexOf(hub.id) < 0, "…the box leaves the canvas");
  ok(String(btn.className).indexOf("off") >= 0 && btn.title === "show the Train box",
     "…and the button offers to show it again");
  ok(__t.bxTrainHub() && __t.bxTrainHub().id === hub.id, "…hidden, it is still the hub");
  ok(__t.bxAdd("train", 200, 0, {}) === null, "…so a second one stays refused");
  eql(__t.bxJSONNode(hub).hidden, true, "…and the design carries the hidden flag");
  ok(__t.bxImportJSON(JSON.stringify(__t.bxExportCanvas()), {}).ok, "the canvas round-trips");
  const back = __t.bxTrainHub();
  ok(back && back.hidden === true, "…and the box comes back hidden");
  ok(__t.bxtRunToggle() === true && !back.hidden, "a press shows it again");
  ok(panes().indexOf(back.id) >= 0, "…and the box is back on the canvas");
  // Backend -> Local keeps the connection, and points at the Train box
  __t.BXRUN = { local: { connected: true, agent: { name: "gpu-box", remote: "127.0.0.1:1",
      since: 1, last_seen: 1, specs: { name: "gpu-box", os: "Linux-test", cpu: "Test CPU",
      cores: 16, ram_gb: 64, gpu: "NVIDIA RTX 5090", vram_gb: 32, device: "cuda",
      cuda: true, torch: "2.13.0" } } },
      inputs: {}, last: null, loaded: null, loading: false, loadKey: null,
      msg: "", running: false };
  __t.bxtBackendToggle();
  const bm = getEl("bxtBackendMenu");
  const heads = () => bm.children.filter(c => c.className === "bxtcat");
  const firstOpen = () => ((heads()[0].querySelectorAll(".chev")[0] || {}).textContent === "\u25be");
  if (!firstOpen()) heads()[0].click();
  const pane = bm.children.find(c => c.className === "bxbpane");
  const hint = pane.querySelectorAll(".bxbhint").map(e => e.textContent).join(" | ");
  ok(hint.indexOf("connected") >= 0 && hint.indexOf("live usage are in the Train box") >= 0,
     "Backend → Local keeps the connection and points at the Train box");
  __t.bxtBackendToggle();
  // a saved Modal key still counts as a backend (the run action's readiness)
  __t.BX_BACKEND.mode = "remote";
  __t.BX_BACKEND.key = "ak-test";
  ok(__t.bxtRunEnabled() === true, "a saved Modal key also establishes a backend");
  __t.BX_BACKEND.mode = "local";
  __t.BX_BACKEND.key = "";
  // leave the canvas and the backend as the next block found them
  const fin = __t.bxTrainHub();
  if (fin) __t.bxRemove(fin.id);
  __t.BXRUN = disconnected;
  __t.bxtRunRefresh();
});

check("the Train box carries the machine; the input carries the run", () => {
  const tr = __t.bxAdd("train", 0, 0, {});
  const ib = __t.bxAdd("input", 0, 320, { values: "1 2" });
  ok(!!tr && !!ib, "the boxes exist");
  const fill = (tab) => {
    const d = document.createElement("div");
    if (tab) tr.panel = Object.assign({}, tr.panel, { tab });
    __t.bxTrainFill(d, tr);
    return d;
  };
  const ifill = () => { const d = document.createElement("div"); __t.bxInputFill(d, ib); return d; };
  const bstate = () => fill("spec").querySelectorAll(".bxbhint").map(e => e.textContent).join(" | ");
  const bchips = () => fill("spec").querySelectorAll(".chip").map(e => e.textContent);
  const specsIn = (root) => root.querySelectorAll(".bxbspec")
    .map(r => r.children.map(c => c.textContent).join(" ")).join(" | ");
  eql(fill("exec").querySelectorAll(".runmi").map(e => e.textContent), ["Execute", "Spec"],
      "the Train box's own menu: Execute and Spec");
  ok(fill("exec").querySelectorAll(".mseg").length === 0,
     "…no mode switch — the box is the training");
  const ctrl = fill("exec").querySelectorAll(".chip").map(e => e.textContent).join(" | ");
  ok(ctrl.indexOf("set up the training") >= 0 && ctrl.indexOf("start") >= 0,
     "…and the controls that drive it: " + ctrl);
  ok(fill("exec").querySelectorAll(".bxbspec").length === 0,
     "…the machine stays out of Execute");
  ok(fill("spec").querySelectorAll(".chip").map(c => c.textContent)
       .indexOf("run the design") < 0, "…and the run action stays out of Spec");
  // the run action lives on the input box now
  const goOff = ifill().querySelectorAll(".chip")
    .find(c => c.textContent.indexOf("run the design") >= 0);
  ok(goOff && String(goOff.className).indexOf("off") >= 0,
     "the input box carries the run action, off without a backend");
  ok(String(goOff.className).indexOf("play") >= 0 && goOff.textContent.indexOf("\u25b6") >= 0,
     "…a green play button: " + goOff.textContent);
  __t.BXRUN.error = true;
  __t.BXRUN.msg = "the dataflow is not whole";
  ok(ifill().querySelectorAll(".bxbhint.runbad").map(e => e.textContent).join(" | ")
       .indexOf("dataflow") >= 0, "…and a failed run shows why, in red");
  __t.BXRUN.error = false;
  __t.BXRUN.msg = "";
  __t.BXRUN.last = { ms: 12, device: "cuda", values: { a: {}, b: {} }, texts: {} };
  __t.bxRunValues = { a: { shape: [1] }, b: { shape: [1] } };
  ok(ifill().querySelectorAll(".bxbhint.runok").map(e => e.textContent).join(" | ")
       .indexOf("ran the design") >= 0, "…and a good run says so, in green");
  ok(ifill().querySelectorAll(".bxbhint.runok").map(e => e.textContent).join(" | ")
       .indexOf("cuda") >= 0, "…naming the device it ran on");
  __t.BXRUN.last = null;
  __t.bxRunValues = {};
  ok(ifill().querySelectorAll(".bxbhint").length === 0, "…and an edit clears the verdict");
  // the menu switches views, and the choice sticks to the box
  fill("exec").querySelectorAll(".runmi").find(b => b.textContent === "Spec").click();
  eql(tr.panel.tab, "spec", "…Spec is a press away");
  ok(fill().querySelectorAll(".bxbhint").map(e => e.textContent).join(" | ")
       .indexOf("no backend connected") >= 0, "…and shows the machine's state");
  eql(__t.bxJSONNodeFrom(__t.bxJSONNode(tr)).panel.tab, "spec",
     "…and the design carries the open view");
  fill("spec").querySelectorAll(".runmi").find(b => b.textContent === "Execute").click();
  eql(tr.panel.tab, "exec", "…and back to Execute");
  // a connected backend lights the input's run button
  __t.BX_BACKEND.mode = "local";
  __t.BX_BACKEND.key = "";
  __t.BXRUN = { local: { connected: true, agent: { name: "gpu-box", remote: "127.0.0.1:1",
      since: 1, last_seen: 1, specs: { name: "gpu-box", os: "Linux-test", cpu: "Test CPU",
      cores: 16, ram_gb: 64, gpu: "NVIDIA RTX 5090", vram_gb: 32, device: "cuda",
      cuda: true, torch: "2.13.0" },
      usage: { ram_used_gb: 14.2, ram_total_gb: 64, ram_pct: 22,
               vram_used_gb: 6.1, vram_total_gb: 32, vram_pct: 19 } } },
      inputs: {}, last: null, loaded: null, loading: false, loadKey: null,
      msg: "", running: false };
  const goOn = ifill().querySelectorAll(".chip")
    .find(c => c.textContent.indexOf("run the design") >= 0);
  ok(goOn && String(goOn.className).indexOf("off") < 0, "…and it lights with the backend");
  ok(bstate().indexOf("gpu-box") >= 0 && bstate().indexOf("RTX 5090") >= 0 &&
     bstate().indexOf("cuda") >= 0, "Spec shows the backend machine");
  const specs = specsIn(fill("spec"));
  ok(specs.indexOf("Test CPU") >= 0 && specs.indexOf("16 cores") >= 0 &&
     specs.indexOf("RTX 5090") >= 0, "…the pc's specs");
  ok(specs.indexOf("14.2 / 64.0 GB") >= 0 && specs.indexOf("22%") >= 0,
     "…with current ram usage");
  ok(specs.indexOf("6.1 / 32.0 GB") >= 0 && specs.indexOf("19%") >= 0,
     "…and current vram usage");
  ok(bchips().indexOf("load weights") >= 0 && bchips().indexOf("load all on") < 0,
     "…one load button — the ratio bar decides where it goes");
  ok(bchips().indexOf("offload") >= 0, "…and the offload button that frees the backend");
  // what the backend actually holds, and its going away again
  __t.BXRUN.loaded = { bytes: 2 * 1024 ** 3, on_gpu_bytes: 1024 ** 3,
                       on_system_bytes: 1024 ** 3, params: 500, ms: 12 };
  ok(bstate().indexOf("loaded: 2.00 GB") >= 0 && bstate().indexOf("mixed") >= 0,
     "…the report says what the backend holds");
  __t.BXRUN.loaded = null;
  ok(bstate().indexOf("loaded:") < 0, "…and offload makes it go");
  // a layer's placement: the right-click menu, the card mark, and the JSON
  let layer = __t.BX.nodes.find(n => n.kind === "layer");
  const madeLayer = !layer;
  if (!layer) layer = __t.bxAdd("layer", 0, 480, {});
  if (layer.din == null || layer.dout == null)
    __t.bxConfigure(layer.id, { din: 4, dout: 3 });      // bytes need sizes
  ok(__t.bxWeightedBox(layer), "the layer counts as a weight box");
  __t.bxMenuBuild("box", layer.id, {});
  const lmenu = getEl("bxmenu");
  openCat(lmenu, "weights");
  const lw = catItems(lmenu).join(" | ");
  ok(lw.indexOf("load on the gpu") >= 0 && lw.indexOf("load on the system") >= 0,
     "…menu: load on the gpu or on the system");
  __t.bxMenuClose();
  ok(__t.bxSetPlace(layer.id, "system") === true, "load the layer on the system");
  eql(__t.bxJSONNode(layer).place, "system", "…and the design carries it");
  ok(__t.bxCard(layer).querySelectorAll(".dplace")
       .map(e => e.textContent).join("") === "sys", "…and the card says sys");
  __t.bxSetPlace(layer.id, "gpu");
  ok(__t.bxCard(layer).querySelectorAll(".dplace")
       .map(e => e.textContent).join("") === "gpu", "…and gpu the other way");
  // the weights line: bytes used, and where the layers sit — gpu, cpu or mixed
  let weightCount = 0;
  __t.bxEveryBox(n => { if (__t.bxWeightedBox(n)) weightCount++; });
  ok(weightCount > 0, "the design has weight boxes");
  const extra = weightCount < 2 ? __t.bxAdd("layer", 0, 620, { din: 2, dout: 2 }) : null;
  ok(__t.bxSetAllPlaces("system") > 0, "load all on system pins every one");
  let allSys = true;
  __t.bxEveryBox(n => { if (__t.bxWeightedBox(n) && n.place !== "system") allSys = false; });
  ok(allSys, "…all of them on the system");
  ok(__t.bxSetAllPlaces("gpu") > 0, "load all on vram pins them back");
  let allGpu = true;
  __t.bxEveryBox(n => { if (__t.bxWeightedBox(n) && n.place !== "gpu") allGpu = false; });
  ok(allGpu, "…all of them on the gpu");
  const split = __t.bxWeightSplit(true);
  ok(split.gpu > 0 && split.system === 0, "the split counts bytes, not boxes");
  ok(bstate().indexOf("all on the gpu") >= 0 && bstate().indexOf(" B") >= 0,
     "the weights line: bytes and all on the gpu");
  __t.bxSetAllPlaces("system");
  ok(bstate().indexOf("all on the cpu") >= 0, "…all on the cpu after the other button");
  __t.bxSetPlace(layer.id, "gpu");
  ok(bstate().indexOf("mixed") >= 0 && bstate().indexOf("cpu)") >= 0,
     "…and mixed when one layer sits on the cpu");
  __t.bxEveryBox(n => { delete n.place; });    // leave the design as it was found
  if (extra) __t.bxRemove(extra.id);
  if (madeLayer) __t.bxRemove(layer.id);
  __t.bxRemove(ib.id);
  __t.bxRemove(tr.id);
});


check("the cpu-offload ratio splits the first layers", () => {
  // a small nested model: one box with four layer blocks, 40 000 B each
  const model = __t.bxAdd("custom", 0, 0, {});
  __t.bxEnter(model.id);
  const blocks = [];
  for (let i = 0; i < 4; i++) {
    const b = __t.bxAdd("custom", 0, i * 120, {});
    __t.bxEnter(b.id);
    const l = __t.bxAdd("layer", 0, 0, {});
    __t.bxConfigure(l.id, { din: 100, dout: 100, dtype: "f32" });
    __t.bxLeave();
    blocks.push(b);
  }
  __t.bxLeave();
  const savedModel = __t.BX.model;
  __t.BX.model = Object.assign({}, __t.BX.model, { cpu_offload: 50 });
  const plan = __t.bxApplyOffload("gpu");
  const places = blocks.map(b => b.place || null);
  const p100 = __t.bxOffloadPlan(100);
  const p0 = __t.bxOffloadPlan(0);
  const canvas = __t.bxExportCanvas();          // exactly what the load button posts
  const mbox = canvas && canvas.nodes.find(n => n.kind === "custom");
  const posted = mbox ? mbox.inner.nodes.filter(n => n.kind === "custom").map(b => b.place)
                      : [];
  const noKey = __t.bxtRunLoad.length === 0;
  __t.BX.model = savedModel;
  __t.bxRemove(model.id);
  eql(places, ["gpu", "gpu", "system", "system"],
    "half the ratio: the first half of the layers stays on the gpu");
  eql(posted, ["gpu", "gpu", "system", "system"],
    "…and the canvas the load button posts carries that plan");
  ok(noKey, "the load takes no device key — the ratio decides");
  eql([plan.gpu, plan.system], [80000, 80000], "…and the planned bytes match");
  eql([p100.gpu, p100.system], [0, 160000], "100% keeps everything on the system");
  eql([p0.gpu, p0.system], [160000, 0], "0% keeps everything on the gpu");
});

check("the matrix input: numbers typed in the box, one row per line", () => {
  __t.bxForm("input", null, null, "add");      // adding one asks the same form
  const fsel = getEl("bxmenu").querySelectorAll(".bxsel")[0];
  eql(fsel.children.map(o => o.textContent),
      ["matrix \u2014 type its numbers", "text \u2014 a string"],
      "adding an input asks its data type");
  __t.bxMenuClose();
  const i = __t.bxAdd("input", 0, 0, {});
  __t.bxMenuBuild("box", i.id, {});
  openCat(getEl("bxmenu"), "this box");
  ok(catItems(getEl("bxmenu")).indexOf("set input") >= 0,
    "the input's menu offers the matrix input");
  __t.bxMenuClose();
  __t.bxForm("input", null, i, "matrix");
  eql(getEl("bxmenu").querySelectorAll(".bxmat").length, 1,
      "the form asks for the matrix itself");
  __t.bxMenuClose();
  __t.bxConfigure(i.id, { input_mode: "matrix", values: "1 2 3\n4 5 6", dout: 3 });
  eql(__t.bxJSONNodeFrom(__t.bxJSONNode(i)).values, "1 2 3\n4 5 6",
      "the matrix travels with the design");
  eql(__t.bxShape(i), "2\u00d73", "\u2026and the card reads its shape");
  eql(__t.bxPortDefs(i, "out").map(p => p.size), [3],
      "\u2026and the wire sizes by its width");
  // the hover card calls it a matrix, shape and all
  __t.bxHotSet(null);
  __t.bxHotMove({ target: __t.bxCard(i), clientX: 120, clientY: 90 });
  const tip = getEl("bxtip");
  const line = cls => {
    const e2 = tip.querySelectorAll("." + cls)[0];
    return e2 ? String(e2.textContent) : null;
  };
  eql(line("shp"), "emits matrix [2\u00d73]", "the hover card reads it as a matrix");
  __t.bxConfigure(i.id, { input_mode: "matrix", values: "7 8 9", dout: 3 });
  __t.bxHotSet(null);
  __t.bxHotMove({ target: __t.bxCard(i), clientX: 120, clientY: 90 });
  eql(line("shp"), "emits matrix [3]",
      "a typed line is exact \u2014 the batch ? is only for declared shapes");
  __t.bxConfigure(i.id, { input_mode: "matrix", values: "1 2 3\n4 5 6", dout: 3 });
  __t.bxHotSet(null);
  // the Run box reads the box's own data \u2014 a matrix is not a run field
  __t.BXRUN = { local: { connected: true,
      agent: { name: "pc", specs: { device: "cpu", cpu: "x" } } },
      inputs: {}, last: null, loaded: null, loading: false, loadKey: null,
      msg: "", running: false };
  const feeds = () => { const d = document.createElement("div"); __t.bxRunFeeds(d); return d; };
  const modes = () => feeds().querySelectorAll(".wmode").map(e => e.textContent).join(" | ");
  ok(modes().indexOf("matrix 2\u00d73") >= 0,
     "the Run box reads matrix 2\u00d73 from the box: " + modes());
  eql(feeds().querySelectorAll(".bxbin").length, 0, "\u2026and asks for no run field");
  // a legacy python/random mode reads as the matrix type now
  __t.bxConfigure(i.id, { input_mode: "python" });
  eql(i.input_mode, "matrix", "an old python mode reads as matrix");
  __t.bxConfigure(i.id, { input_mode: "random" });
  eql(i.input_mode, "matrix", "\u2026and so does random");
  eql(__t.bxShape(i), "2\u00d73", "\u2026and the card still reads the matrix");
  __t.BXRUN = { local: { connected: false, agent: null }, inputs: {}, last: null,
                loaded: null, loading: false, loadKey: null, msg: "", running: false };
  __t.bxtRunRefresh();
  __t.bxRemove(i.id);
});

check("the tokenizer and the de-tokenizer: text in, ids through, text out", () => {
  const t = __t.bxAdd("tokenize", 0, 0, { vocab: "ab c", text: "cab" });
  const d = __t.bxAdd("detokenize", 0, 200, { vocab: "ab c" });
  eql(__t.bxKindLabel("tokenize"), "Tokenizer", "the tokenizer is its own kind");
  eql(__t.bxKindLabel("detokenize"), "De-tokenizer", "…and so is the de-tokenizer");
  eql([__t.bxPortDefs(t, "in").length, __t.bxPortDefs(t, "out").length], [1, 1],
    "the tokenizer takes text in and sends ids out");
  eql([__t.bxPortDefs(d, "in").length, __t.bxPortDefs(d, "out").length], [1, 1],
    "the de-tokenizer takes ids in and sends the text out");
  eql(__t.bxPortSizes(t, "out"), [1], "its out-dot carries one id");
  ok(__t.bxShape(t).indexOf("vocab 4") >= 0, "the card shows the vocabulary: " + __t.bxShape(t));
  ok(__t.bxShape(d).indexOf("text") >= 0, "…and the de-tokenizer says text");
  eql(__t.bxConnect(t.id, d.id), null, "the two wire up");
  // any size may arrive — ids (one id) or a model's logits (the vocab)
  const src = __t.bxAdd("layer", 0, 400, { din: 2, dout: 5 });
  ok(!__t.bxCheckLink(src.id, d.id, 0, 0).error, "a logits-sized wire is accepted too");
  eql(__t.bxConnect(src.id, d.id), null, "…and lands");
  const wire = __t.BX.edges.find(e => e.from === src.id && e.to === d.id);
  ok(wire && __t.bxWireState(wire) !== "bad", "…without going red");
  __t.bxRemoveEdge(src.id, d.id);
  __t.bxConnect(t.id, d.id);
  // a huggingface repo: pasted in, loaded, and shown on the card
  __t.bxConfigure(t.id, { repo: "Qwen/Qwen3-8B", tokens: 151936 });
  eql([t.repo, t.tokens], ["Qwen/Qwen3-8B", 151936], "the repo is remembered");
  ok(__t.bxShape(t).indexOf("hf \u00b7 Qwen3-8B") >= 0, "the card says hf: " + __t.bxShape(t));
  eql(__t.bxJSONNode(t).repo, "Qwen/Qwen3-8B", "…and the design carries it");
  __t.bxForm("tokenize", null, t, "token");
  ok(getEl("bxmenu").children[0].children.map(c => c.textContent).join(" | ")
       .indexOf("huggingface") >= 0, "the form asks for a repo");
  __t.bxMenuClose();
  // the de-tokenizer's output is the text — one in-dot, one out-dot
  eql([__t.bxPortDefs(d, "out").length], [1], "the de-tokenizer has an output");
  // a sample box: logits in, one picked id out
  const sm = __t.bxAdd("sample", 0, 600, { temperature: 0, sample_seed: 3 });
  eql(__t.bxKindLabel("sample"), "Sample", "the sample box is its own kind");
  eql([__t.bxPortDefs(sm, "in").length, __t.bxPortDefs(sm, "out").length], [1, 1],
    "logits in, one id out");
  eql(__t.bxPortSizes(sm, "out"), [1], "…one id");
  eql(__t.bxShape(sm), "t 0", "the card shows its temperature");
  __t.bxConfigure(sm.id, { temperature: 0.5, sample_seed: 9 });
  eql([sm.temperature, sm.sample_seed], [0.5, 9], "temperature and seed are remembered");
  const srec = __t.bxJSONNode(sm);
  eql([srec.temperature, srec.sample_seed], [0.5, 9], "…and the JSON carries them");
  __t.bxMenuBuild("box", sm.id, {});
  openCat(getEl("bxmenu"), "this box");
  ok(catItems(getEl("bxmenu")).indexOf("set the sample …") >= 0,
    "the menu offers set the sample");
  __t.bxMenuClose();
  __t.bxForm("sample", null, sm, "sample");
  ok(getEl("bxmenu").children[0].children.map(c => c.textContent).join(" | ")
       .indexOf("temperature") >= 0, "…and its form asks for temperature and seed");
  __t.bxMenuClose();
  __t.bxRemove(sm.id);
  __t.bxConfigure(t.id, { repo: "" });
  ok(!t.repo && !t.tokens, "clearing the repo clears the token count");
  __t.bxMenuBuild("box", t.id, {});
  openCat(getEl("bxmenu"), "this box");
  ok(catItems(getEl("bxmenu")).indexOf("set the tokenizer …") >= 0,
    "the menu offers set the tokenizer");
  __t.bxMenuClose();
  const rec = __t.bxJSONNode(t);
  eql([rec.vocab, rec.text], ["ab c", "cab"], "the design carries the vocab and the text");
  __t.BXRUN = { local: { connected: true, agent: { name: "pc", specs: { device: "cuda" } } },
                inputs: {}, last: null, loaded: null, loading: false, loadKey: null,
                msg: "", running: false };
  const feeds = () => { const d = document.createElement("div"); __t.bxRunFeeds(d); return d; };
  ok(feeds().querySelectorAll(".bxbin").some(f => f.placeholder === "text to tokenize"),
    "the Run box asks for text");
  ok(feeds().querySelectorAll(".mhead").map(e => e.textContent).indexOf("text") >= 0,
    "…under a text heading");
  // an input box in text mode feeds it over the wire — and then it asks for nothing
  const ti = __t.bxAdd("input", 0, 0, { input_mode: "text", text: "hello" });
  eql(__t.bxShape(ti), "text", "a text input says text on the card");
  eql(__t.bxJSONNode(ti).text, "hello", "…and the design carries the text");
  eql(__t.bxConnect(ti.id, t.id), null, "the text input wires into the tokenizer");
  ok(!feeds().querySelectorAll(".bxbin").some(f => f.placeholder === "text to tokenize"),
    "a wired tokenizer asks for no text of its own");
  ok(feeds().querySelectorAll(".bxbin").some(f => f.placeholder === "text"),
    "the input's text field is there instead");
  __t.bxRemoveEdge(ti.id, t.id);
  __t.bxRemove(ti.id);
  __t.BXRUN = { local: { connected: false, agent: null }, inputs: {}, last: null,
                loaded: null, loading: false, loadKey: null, msg: "", running: false };
  __t.bxtRunRefresh();
  __t.bxRemove(t.id);
  __t.bxRemove(d.id);
  __t.bxRemove(src.id);
});

check("the huggingface model box: a repo in, the next token's text out", () => {
  const h = __t.bxAdd("hf", 0, 0, { repo: "Qwen/Qwen2.5-0.5B", text: "Once upon a" });
  eql([h.kind, h.name], ["hf", "hf model 1"], "an hf box, size-less");
  eql([h.repo, h.text], ["Qwen/Qwen2.5-0.5B", "Once upon a"],
    "\u2026carrying its repo and prompt");
  eql(__t.bxKindLabel("hf"), "HF model", "…its own kind");
  eql(__t.bxShape(h), "next \u00b7 Qwen2.5-0.5B", "…its card says what it continues");
  eql([__t.bxPortDefs(h, "in").length, __t.bxPortDefs(h, "out").length], [1, 1],
    "text in, the token's text out");
  ok(!__t.bxErrorSet().has(h.id), "a lone hf box is not an error (both dots optional)");
  __t.bxConfigure(h.id, { temperature: 0.7 });
  const rec = __t.bxJSONNode(h);
  eql([rec.repo, rec.text, rec.temperature],
      ["Qwen/Qwen2.5-0.5B", "Once upon a", 0.7], "…and the design carries all three");
  const made = __t.bxJSONNodeFrom(rec);
  ok(made && made.repo === "Qwen/Qwen2.5-0.5B", "…and it reads back");
  __t.bxMenuBuild("box", h.id, {});
  openCat(getEl("bxmenu"), "this box");
  ok(catItems(getEl("bxmenu")).indexOf("set the model …") >= 0,
    "the menu offers set the model");
  __t.bxMenuClose();
  __t.bxForm("hf", null, h, "hf");
  ok(getEl("bxmenu").children[0].children.map(c => c.textContent).join(" | ")
       .indexOf("temperature") >= 0, "…and its form asks for the prompt and temperature");
  __t.bxMenuClose();
  // the Run box asks for its prompt, and the run sends it as a feed
  __t.BXRUN = { local: { connected: true, agent: { name: "pc", specs: {} } },
                inputs: {}, texts: {}, last: null, loaded: null, loading: false,
                loadKey: null, msg: "", running: false };
  const pane = document.createElement("div");
  __t.bxRunFeeds(pane);
  ok(pane.querySelectorAll(".bxbin").some(f => f.placeholder === "the text to continue"),
    "the Run box asks for the text to continue");
  const realFetch = global.fetch;
  let sent = null;
  global.fetch = async (url, opts) => {
    if (String(url) === "/api/run") sent = JSON.parse(opts.body);
    return { ok: true, status: 200, json: async () => ({ ok: true }) };
  };
  __t.BXRUN.texts[h.id] = "the cat sat";
  __t.bxtRunGo();
  global.fetch = realFetch;
  eql(sent && sent.feeds && sent.feeds[h.id], "the cat sat",
    "the prompt travels as a feed");
  __t.BXRUN.running = false;
  __t.BXRUN.texts = {};
  __t.bxRemove(h.id);
});

check("a generate run posts the ids to grow and the logits to read", () => {
  const t = __t.bxAdd("tokenize", 0, 0, { repo: "Qwen/Qwen3-8B" });
  const m = __t.bxAdd("layer", 0, 200, { din: 1, dout: 3 });
  const o = __t.bxAdd("output", 0, 400, { din: 3 });
  eql(__t.bxConnect(t.id, m.id), null, "the tokenizer's id feeds the layer");
  __t.bxConnect(m.id, o.id);
  __t.BXRUN = { local: { connected: true, agent: { name: "pc", specs: {} } },
                inputs: {}, texts: {}, last: null, loaded: null, loading: false,
                loadKey: null, msg: "", running: false, gen: 3, temp: 0 };
  __t.BXRUN.texts[t.id] = "The capital of France is";
  const realFetch = global.fetch;
  let sent = null;
  global.fetch = async (url, opts) => {
    if (String(url) === "/api/run") sent = JSON.parse(opts.body);
    return { ok: true, status: 200, json: async () => ({ ok: true }) };
  };
  __t.bxtRunGo();
  global.fetch = realFetch;
  __t.BXRUN.running = false;
  eql([sent && sent.generate, sent && sent.input, sent && sent.output,
       sent && sent.temperature], [3, t.id, o.id, 0],
    "the loop grows the tokenizer's ids, samples the logits, and stays greedy");
  __t.BXRUN.gen = 0;                           // a plain run stays a plain forward
  sent = null;
  global.fetch = async (url, opts) => {
    if (String(url) === "/api/run") sent = JSON.parse(opts.body);
    return { ok: true, status: 200, json: async () => ({ ok: true }) };
  };
  __t.bxtRunGo();
  global.fetch = realFetch;
  __t.BXRUN.running = false;
  ok(sent && sent.generate == null, "0 tokens sends no generate at all");
  __t.BXRUN.texts = {};
  __t.bxRemove(t.id);
  __t.bxRemove(m.id);
  __t.bxRemove(o.id);
});

check("an input's own values flow to the output (the browser verifies first)", () => {
  __t.BX.nodes.slice().forEach(x => __t.bxRemove(x.id));
  const i = __t.bxAdd("input", 0, 0, { dout: 3, values: "4 5 6" });
  const o = __t.bxAdd("output", 0, 200, { din: 3 });
  eql(__t.bxConnect(i.id, o.id), null, "the input wires straight to the output");
  eql(__t.bxJSONNodeFrom(__t.bxJSONNode(i)).values, "4 5 6", "…and the design carries its values");
  ok(__t.bxFlowVerify().ok, "the browser verifies the dataflow before a run");
  __t.BXRUN = { local: { connected: true, agent: { name: "pc", specs: {} } },
                inputs: {}, texts: {}, last: null, loaded: null, loading: false,
                loadKey: null, msg: "", running: false };
  const realFetch = global.fetch;
  let sent = null;
  global.fetch = async (url, opts) => {
    if (String(url) === "/api/run") sent = JSON.parse(opts.body);
    return { ok: true, status: 200, json: async () => ({ ok: true }) };
  };
  __t.bxtRunGo();
  global.fetch = realFetch;
  const posted = sent && sent.canvas && (sent.canvas.nodes || []).find(n => n.id === i.id);
  eql(sent && sent.feeds && sent.feeds[i.id], undefined,
      "no feed is sent — the box's own matrix is the data");
  eql(posted && posted.values, "4 5 6", "…and it rides with the design");
  __t.BXRUN.running = false;
  __t.BXRUN.msg = "";
  // a broken dataflow is refused in the browser: the backend is never asked
  sent = null;
  global.fetch = async (url, opts) => {
    if (String(url) === "/api/run") sent = JSON.parse(opts.body);
    return { ok: true, status: 200, json: async () => ({ ok: true }) };
  };
  __t.bxRemoveEdge(i.id, o.id);
  __t.bxtRunGo();
  global.fetch = realFetch;
  ok(sent === null, "a design whose data never reaches the output is not sent");
  ok(__t.BXRUN.msg.indexOf("never reaches") >= 0, "…and the browser says why: " + __t.BXRUN.msg);
  ok(__t.bxFlowVerify().bad.has(o.id), "…marking the output that would starve");
  __t.bxRemove(i.id);
  __t.bxRemove(o.id);
  __t.BXRUN = { local: { connected: false, agent: null }, inputs: {}, last: null,
                loaded: null, loading: false, loadKey: null, msg: "", running: false };
});

check("the AI box: the task explains itself \u2014 type, ask, chip, progress", () => {
  const t = { id: "t1", type: "reverse", params: { repo: "unsloth/Qwen3.8-27B-NVFP4" },
              status: "running",
              todo: [{ state: "done", title: "reading" },
                     { state: "active", title: "reconstruct" }] };
  eql(__t.bxTaskLabel(t), "reverse engineer repo \u00b7 unsloth/Qwen3.8-27B-NVFP4",
    "the task says its type and what the user asked for");
  eql(__t.bxTaskChip(t), "In-Progress", "In-Progress");
  eql(__t.bxTaskChip(Object.assign({}, t, { status: "paused" })), "Paused", "Paused");
  eql(__t.bxTaskChip(Object.assign({}, t, { status: "cancelled" })), "Cancelled", "Cancelled");
  eql(__t.bxTaskChip(Object.assign({}, t, { status: "done", outcome: "exact" })),
    "Done \u00b7 exact", "Done \u00b7 exact");
  eql(__t.bxTaskChip(Object.assign({}, t, { status: "done", outcome: "approximate" })),
    "Done \u00b7 still approximate", "\u2026and honest outcomes");
  eql(__t.bxTaskProgress(t), "1/2 \u00b7 reconstruct", "the checklist line");
  eql(__t.bxTaskFraction(t), 0.5, "the ring reads the checklist: half done");
  eql(__t.bxTaskFraction(Object.assign({}, t, { status: "done", todo: [] })), 1,
    "a done task rings full");
  ok(__t.bxArcPath(10, 10, 5, 0.5).indexOf("A 5 5") >= 0, "the arc draws");
  eql(__t.bxArcPath(10, 10, 5, 0), "", "…and nothing for no progress");
  const py = { id: "t2", type: "python", params: { prompt: "plot the loss" },
               status: "running", todo: [] };
  ok(__t.bxTaskLabel(py).indexOf("python agent") >= 0 &&
     __t.bxTaskLabel(py).indexOf("plot the loss") >= 0,
     "any agent type: " + __t.bxTaskLabel(py));
});

check("the agent box fires, selects and streams its task \u2014 token by token", () => {
  const allText = (el) => {
    let s = el.textContent || "";
    for (const c of el.children || []) s += " " + allText(c);
    return s;
  };
  const calls = [];
  const tasks = [
    { id: "t1", type: "reverse", params: { repo: "org/a" }, status: "running",
      phase: "agent", todo: [{ state: "done", title: "reading" },
                             { state: "active", title: "reconstruct" }] },
    { id: "t2", type: "python", params: { prompt: "plot it" }, status: "paused", todo: [] },
    { id: "t3", type: "reverse", params: { repo: "org/b" }, status: "done",
      outcome: "exact", todo: [] },
    { id: "t4", type: "reverse", params: { repo: "org/c" }, status: "cancelled", todo: [] },
  ];
  const realFetch = global.fetch;
  global.fetch = async (url, opts) => {
    const u = String(url);
    calls.push([u, opts && opts.body ? JSON.parse(opts.body) : null]);
    const jr = (body) => ({ ok: true, status: 200, json: async () => body });
    if (u === "/api/tasks" && !(opts && opts.method === "POST"))
      return jr({ ok: true, tasks });
    if (u === "/api/tasks" && opts && opts.method === "POST") {
      const body = JSON.parse(opts.body);
      tasks.unshift({ id: "t9", type: body.type, params: body.params,
                      status: "running", todo: [] });
      return jr({ ok: true, task: tasks[0] });
    }
    if (u === "/api/tasks/cmd")
      return jr({ ok: true, task: tasks.find(x => x.id === JSON.parse(opts.body).task) });
    if (u.indexOf("/api/tasks/log") === 0) {
      const st = __t.bxLogState("t1");
      if (st.pos === 0) return jr({ ok: true, pos: 100, items: [
        { id: "p1", kind: "say", text: "Hel" },
        { id: "c1", kind: "do", text: "read \u00b7 nn.py", detail: "input:\n{}" }] });
      return jr({ ok: true, pos: 100, items: [{ id: "p1", kind: "say", text: "lo", append: true }] });
    }
    return jr({ ok: true });
  };
  __t.bxTasksApply(tasks);
  eql(__t.BXTASKS.list.length, 4, "the list arrives");
  const b = __t.bxAdd("agent", 0, 0, {});
  __t.bxRender();
  eql([b.kind, __t.bxShape(b)], ["agent", "no task"], "a fresh agent box has no task");
  // the display lives inside the box — no popup, no double click needed
  let panes = getEl("bxcanvas").querySelectorAll(".bxlogpane");
  ok(panes.length === 1, "the box carries its own display pane");
  ok(allText(panes[0]).indexOf("In-Progress") >= 0 &&
     allText(panes[0]).indexOf("Paused") >= 0 &&
     allText(panes[0]).indexOf("Done \u00b7 exact") >= 0 &&
     allText(panes[0]).indexOf("Cancelled") >= 0,
     "every chip is there: " + allText(panes[0]));
  ok(allText(panes[0]).indexOf("new task") >= 0, "\u2026and the new-task form");
  // start a task from inside the box — the only firing path there is
  __t.bxTaskStart("reverse", { repo: "org/new" }, () => {});
  eql(calls.filter(c => c[0] === "/api/tasks" && c[1]).map(c => [c[1].type, c[1].params]),
    [["reverse", { repo: "org/new" }]], "the box posted exactly one task");
  // select one: the box binds to it and the stream folds token by token
  b.task_id = "t1";
  __t.bxLogApply("t1", [{ id: "p1", kind: "say", text: "Hel" },
                        { id: "c1", kind: "do", text: "read \u00b7 nn.py", detail: "input:\n{}" }]);
  __t.bxLogApply("t1", [{ id: "p1", kind: "say", text: "lo", append: true }]);
  __t.bxTaskLogFetch("t1", () => {});
  const st = __t.bxLogState("t1");
  eql(st.items.map(i => [i.id, i.text]), [["p1", "Hello"], ["c1", "read \u00b7 nn.py"]],
    "the stream folds: a growing part updates in place, token by token");
  eql(calls.filter(c => c[0].indexOf("/api/tasks/log") === 0).length, 1,
    "\u2026and the box keeps pulling the stream");
  __t.bxAgentSync();
  panes = getEl("bxcanvas").querySelectorAll(".bxlogpane");
  ok(allText(panes[0]).indexOf("Hello") >= 0 &&
     allText(panes[0]).indexOf("read \u00b7 nn.py") >= 0,
    "the chain of thought streams into the box: " + allText(panes[0]));
  // the controls live in the box
  __t.bxTaskCmd("t1", "pause", null, () => {});
  __t.bxTaskCmd("t1", "cancel", { reason: "stopped by you" }, () => {});
  eql(calls.filter(c => c[0] === "/api/tasks/cmd").map(c => c[1].cmd),
    ["pause", "cancel"], "pause and stop from the box");
  eql(calls.filter(c => c[0] === "/api/tasks/cmd")[1][1].reason, "stopped by you",
    "\u2026with the reason the user gave");
  // the card wears its ring; supersede hands the ask to a fresh task
  ok(__t.bxCard(b).querySelectorAll(".dringbg").length === 1 &&
     __t.bxCard(b).querySelectorAll(".dring").length === 1,
    "the card wears its progress ring");
  __t.bxTaskSupersede(b, () => {});
  const cmds = calls.filter(c => c[0] === "/api/tasks/cmd");
  eql(cmds.map(c => c[1].cmd), ["pause", "cancel", "cancel"],
    "supersede stops the old task first");
  eql(cmds[2][1].reason, "superseded", "\u2026marked superseded \u2014 and its start "
    + "reuses the same ask (the task-start path is proven above)");
  // failures show inside the box, never in a hidden menu
  delete b.task_id;
  b._err = "a repo id is required";
  __t.bxTasksApply(tasks);
  __t.bxRender();
  panes = getEl("bxcanvas").querySelectorAll(".bxlogpane");
  ok(allText(panes[0]).indexOf("a repo id is required") >= 0,
    "a start error shows in the pane: " + allText(panes[0]).slice(0, 80));
  delete b._err;
  b.task_id = "tf";
  __t.bxTasksApply([{ id: "tf", type: "reverse", params: { repo: "org/gated" },
                      status: "done", outcome: "failed", todo: [],
                      meta: { error: "the repo is gated \u2014 set HF_TOKEN" } }]);
  __t.bxRender();
  panes = getEl("bxcanvas").querySelectorAll(".bxlogpane");
  ok(allText(panes[0]).indexOf("gated") >= 0,
    "a failed task shows its reason instead of waiting forever");
  // the size is adjustable in the canvas, snaphotted in the design
  eql([__t.bxNodeW(b), __t.bxNodeH(b)], [480, 320], "the display starts at its size");
  __t.bxAgentResize(b, 800, 440);
  eql([b.box.w, b.box.h, __t.bxNodeW(b), __t.bxNodeH(b)], [800, 440, 800, 440],
    "the grip writes the size, and the box wears it");
  __t.bxAgentResize(b, 100, 100);
  eql([__t.bxNodeW(b), __t.bxNodeH(b)], [320, 160], "\u2026clamped to something usable");
  const back = __t.bxJSONNodeFrom(__t.bxJSONNode(b));
  eql(back.box, { w: 320, h: 160 }, "the size survives a save/load");
  // double-click toggles the king size, in place — no popup
  __t.bxAgentResize(b, 480, 320);
  const card = __t.bxCard(b);
  const ev = { target: card.children[0], preventDefault() {} };
  __t.bxDblClick(ev);
  eql([b.box.w, b.box.h], [800, 480], "double-click expands the display");
  __t.bxDblClick(ev);
  eql([b.box.w, b.box.h], [480, 320], "\u2026and again shrinks it");
  global.fetch = realFetch;
  __t.bxRemove(b.id);
});

check("loading a repo fires nothing \u2014 and removing a box stops nothing", () => {
  const calls = [];
  const realFetch = global.fetch;
  global.fetch = async (url, opts) => {
    calls.push([String(url), opts && opts.body ? JSON.parse(opts.body) : null]);
    return { ok: true, status: 200, json: async () => ({ ok: true }) };
  };
  const box = __t.bxAdd("custom", 0, 0, {});
  const served = { type: "canvas", version: 1,
    nodes: [box, { id: "x1", kind: "input", name: "tokens", x: -200, y: 0, dout: 1 }],
    edges: [] };
  eql(__t.bxGGUFLoad(box.id, served, null, "org/loud"), true, "the model loads");
  eql(box.repo, "org/loud", "the box remembers which repo filled it");
  const a = __t.bxAdd("agent", 0, 200, {});
  a.task_id = "t1";
  __t.bxRemove(a.id);                          // the box goes: the task lives
  global.fetch = realFetch;
  eql(calls.filter(c => c[0].indexOf("/api/tasks") >= 0), [],
    "no task call anywhere \u2014 loading never fires, removing never cancels");
  __t.BX.nodes.slice().forEach(x => __t.bxRemove(x.id));
});

check("an output box shows a picture that travelled to it", () => {
  const o = __t.bxAdd("output", 300, 0, { din: 3 });
  __t.bxRunValues = { [o.id]: { shape: [768, 768, 3], mean: 0.5, min: 0, max: 1,
                                peek: [0.5, 0.5, 0.5],
                                image: { w: 768, h: 768, blob: "blob:abc.png",
                                         name: "image" } } };
  __t.bxRender();
  const d = document.createElement("div");
  __t.bxOutputFill(d, o);
  const img = d.querySelectorAll(".bxoutimg")[0];
  ok(img && String(img.src).indexOf("abc.png") >= 0,
     "…the window shows the picture that arrived");
  eql(__t.bxShape(o), "image", "…and the card says image");
  eql(__t.bxStateText(__t.bxRunValues[o.id]), "image 768\u00d7768",
      "…and the value reads as a picture");
  const bits = d.querySelectorAll(".panehint").map(x => String(x.textContent)).join(" ");
  ok(bits.indexOf("768\u00d7768") >= 0, "…with its size: " + bits.slice(0, 60));
  __t.bxRunValues = {};
  __t.bxRender();
  __t.bxRemove(o.id);
});

check("the image dropzone: drop or paste a picture into the box", () => {
  const i = __t.bxAdd("input", 0, 0, { input_mode: "image" });
  __t.bxSelectBoxes([i.id]);
  const realFetch = global.fetch;
  const calls = [];
  global.fetch = async (url, opts) => {
    calls.push([String(url), opts && opts.body]);
    return { ok: true, status: 200, json: async () => ({ ok: true, id: "abc.png" }) };
  };
  const d = document.createElement("div");
  __t.bxInputFill(d, i);
  const zone = d.querySelectorAll(".bxdrop")[0];
  ok(zone, "the dropzone is there");
  zone.dispatchEvent({ type: "dragover" });
  ok(String(zone.className).indexOf("over") >= 0, "…and lights up while a drag is over it");
  zone.dispatchEvent({ type: "dragleave" });
  ok(String(zone.className).indexOf("over") < 0, "…and calms down when it leaves");
  const file = { name: "cat.png", type: "image/png" };
  zone.dispatchEvent({ type: "drop", dataTransfer: { files: [file] } });
  eql(calls.length, 1, "the drop uploads once");
  eql(calls[0][0].indexOf("/api/blob?name=cat.png"), 0,
      "…to the media store: " + calls[0][0]);
  eql(calls[0][1], file, "…with the file itself");
  zone.dispatchEvent({ type: "drop",
                       dataTransfer: { files: [{ name: "x.txt", type: "text/plain" }] } });
  eql(calls.length, 1, "a non-picture is refused before any upload");
  __t.bxOnPaste({ clipboardData: { files: [{ name: "p.png", type: "image/png" }] },
                  target: null, preventDefault() {} });
  eql(calls.length, 2, "a pasted picture lands in the selected image box");
  ok(calls[1][0].indexOf("p.png") >= 0, "…under its own name");
  global.fetch = realFetch;
  __t.bxRemove(i.id);
});

check("an output box draws a matrix the backend determined to be a picture", () => {
  const o = __t.bxAdd("output", 0, 0, { din: 3 });
  __t.bxRunValues = { [o.id]: { shape: [2, 2, 3], mean: 0.5, min: 0, max: 1,
                                peek: [0.1, 0.2, 0.3],
                                image: { w: 2, h: 2, blob: "",
                                         thumb: { w: 2, h: 2, rgb: "AAAA" } } } };
  __t.bxRender();
  const d = document.createElement("div");
  __t.bxOutputFill(d, o);
  const cv = d.querySelectorAll(".bxoutimg")[0];
  ok(cv && String(cv.id).indexOf("canvas") >= 0, "…the thumbnail canvas is drawn");
  eql([cv.width, cv.height], [2, 2], "…at its own size");
  __t.bxRunValues = {};
  __t.bxRender();
  __t.bxRemove(o.id);
});

check("a vision canvas lands whole: image input, tower and embeds", () => {
  const box = __t.bxAdd("custom", 320, 0, {});
  __t.BX.nodes.slice().forEach(x => { if (x !== box) __t.bxRemove(x.id); });
  const served = { type: "canvas", version: 1,
    model: { vision: { projector: "qwen3vl_merger", image_size: 768, patch_size: 16 } },
    nodes: [
      { id: "t", kind: "input", name: "tokens", x: 0, y: 0, dout: 1 },
      { id: "m1", kind: "custom", name: "Qwen3.8-27B", x: 320, y: 0,
        inner: { type: "nested", version: 1, nodes: [], edges: [] } },
      { id: "o", kind: "output", name: "logits", x: 640, y: 0, din: 8 },
      { id: "image", kind: "input", name: "image", x: 0, y: 400, dout: 768,
        input_mode: "image", image: "blob:abc.png", image_size: 768, patch_size: 16,
        image_mean: "0.5 0.5 0.5", image_std: "0.5 0.5 0.5" },
      { id: "vision", kind: "vision", name: "Qwen3.8-27B vision", x: 320, y: 400,
        file: "models/mmproj/mmproj-x.gguf", projector: "qwen3vl_merger",
        d_model: 1152, heads: 16, blocks: 27, patch: 16, merge: 2,
        image_size: 768, proj_dim: 5120, eps: 1e-6, din: 768, dout: 5120 },
      { id: "image embeds", kind: "output", name: "image embeds", x: 640, y: 400,
        din: 5120 },
    ],
    edges: [
      { from: "t", to: "m1", fromPort: 0, toPort: 0 },
      { from: "m1", to: "o", fromPort: 0, toPort: 0 },
      { from: "image", to: "vision", fromPort: 0, toPort: 0 },
      { from: "vision", to: "image embeds", fromPort: 0, toPort: 0 },
    ] };
  eql(__t.bxGGUFLoad(box.id, served, null, null), true, "the model loads");
  const names = __t.BX.nodes.map(x => x.kind + ":" + x.name).join(", ");
  ok(__t.BX.nodes.some(x => x.id === "image" && x.input_mode === "image"),
     "…the image input lands on the canvas: " + names);
  const tow = __t.BX.nodes.find(x => x.kind === "vision");
  ok(tow, "…with the vision tower");
  eql([tow.projector, tow.blocks, tow.file, tow.dout],
      ["qwen3vl_merger", 27, "models/mmproj/mmproj-x.gguf", 5120],
      "…which keeps its mmproj and geometry");
  ok(__t.BX.nodes.some(x => x.id === "image embeds"), "…and the embeddings output");
  const wired = __t.BX.edges.some(e => e.from === "image" && e.to === "vision") &&
                __t.BX.edges.some(e => e.from === "vision" && e.to === "image embeds");
  ok(wired, "…and the picture flows through the tower");
  __t.BX.nodes.slice().forEach(x => __t.bxRemove(x.id));
});

check("the connection audit marks the boxes it blames", () => {
  const b = __t.bxAdd("custom", 0, 0, { inner: { type: "nested", version: 1,
    nodes: [{ id: "a1", kind: "layer", name: "dead fc", x: 0, y: 0, din: 1, dout: 1 }],
    edges: [] } });
  __t.bxMarkAudit({ ok: false, count: 1, issues: [
    { severity: "warning", where: "m", box: "dead fc", nid: "a1",
      chain: [b.id], what: "feeds nothing \u2014 no wire leaves this box" }] });
  const marked = !!(__t.bxAudit && __t.bxAudit.ids.has("a1") && __t.bxAudit.ids.has(b.id));
  const why = (__t.bxAudit && __t.bxAudit.notes["a1"] || []).join(" ");
  const cls = String(__t.bxCard(b).className);
  __t.bxAuditClear();
  __t.bxRemove(b.id);
  ok(marked, "the box and the block on the way down are marked");
  ok(why.indexOf("feeds nothing") >= 0, "the reason travels with the mark: " + why);
  ok(cls.indexOf("audit") >= 0, "the box wears it: " + cls);
  ok(!__t.bxAudit, "cleared again");
});

check("the input box edits its data right on the canvas", () => {
  const i = __t.bxAdd("input", 0, 0, { values: "4 5 6", dout: 3 });
  const fill = () => { const d = document.createElement("div"); __t.bxInputFill(d, i); return d; };
  const sel = fill().querySelectorAll(".bxsel")[0];
  ok(sel && sel.value === "matrix", "the box shows the datatype: " + (sel && sel.value));
  eql(sel.children.map(o => o.textContent), ["matrix", "text", "image"],
      "…matrix, text or a picture");
  const vals = fill().querySelectorAll(".bxmat")[0];
  eql(vals.value, "4 5 6", "…with the matrix it carries, editable in place");
  vals.value = "7 8";
  vals.dispatchEvent({ type: "change" });
  eql([i.values, i.dout], ["7 8", 2], "typing the numbers sizes the box to fit them");
  vals.value = "7 8\n9 10";                   // a second row makes it a real matrix
  vals.dispatchEvent({ type: "change" });
  eql([i.values, i.dout], ["7 8\n9 10", 2], "…and a second row makes it 2-D, 2 wide");
  eql(__t.bxShape(i), "2×2", "…which the card reads as a shape");
  // the mode switches the editor
  const s2 = fill().querySelectorAll(".bxsel")[0];
  s2.value = "text";
  s2.dispatchEvent({ type: "change" });
  eql(i.input_mode, "text", "switching the datatype lands on the box");
  eql(__t.bxShape(i), "text", "…and the card says text");
  const txt = fill().querySelectorAll(".bxbin")[0];
  ok(txt && String(txt.placeholder).indexOf("string") >= 0,
     "…and the field now asks for the string");
  txt.value = "hello";
  txt.dispatchEvent({ type: "change" });
  eql(__t.bxJSONNodeFrom(__t.bxJSONNode(i)).text, "hello", "…and it travels with the design");
  // the image mode: a picker, the geometry the run sends, a preview once picked
  const s3 = fill().querySelectorAll(".bxsel")[0];
  s3.value = "image";
  s3.dispatchEvent({ type: "change" });
  eql(i.input_mode, "image", "switching to image lands on the box");
  eql(__t.bxShape(i), "image", "…and the card says image");
  eql(fill().querySelectorAll(".bxbin").length, 1,
      "…with the side the pixels are sent at");
  const zone = fill().querySelectorAll(".bxdrop")[0];
  const msgs = zone ? zone.querySelectorAll(".bxdropmsg").map(x => String(x.textContent)) : [];
  ok(msgs.join(" ").indexOf("Drop or choose") >= 0
     && msgs.join(" ").indexOf("paste") >= 0,
     "…with the drop-or-choose zone (drop, paste, choose): " + msgs.join(" / "));
  i.image = "blob:abc123.png";
  i.image_size = 768;
  i.patch_size = 16;
  const img = fill().querySelectorAll(".bxdropimg")[0];
  ok(img && String(img.src).indexOf("abc123.png") >= 0,
     "…the thumbnail shows in the box once picked");
  const chips = fill().querySelectorAll(".chip").map(c => String(c.textContent));
  ok(chips.indexOf("replace") >= 0 && chips.indexOf("clear") >= 0,
     "…with replace and clear at hand");
  eql(__t.bxJSONNodeFrom(__t.bxJSONNode(i)).image, "blob:abc123.png",
      "…and the blob travels with the design");
  __t.bxConfigure(i.id, { input_mode: "image", image: "blob:zz.png",
                          image_size: 512, dout: 3 });
  eql([i.image, i.image_size, i.dout], ["blob:zz.png", 512, 3],
      "…an upload lands on the box: the blob, the size, a 3-wide pixel matrix");
  __t.bxConfigure(i.id, { image: "" });
  eql(i.image, null, "…and clearing it drops the picture");
  __t.bxConfigure(i.id, { input_mode: "image", image: "blob:abc123.png",
                          image_size: 768, dout: 3 });
  // the size and the open editor travel too
  __t.bxInputExpand(i);
  eql([i.box.w, i.box.h], [420, 280], "the editor expands to the bigger size");
  eql(__t.bxJSONNodeFrom(__t.bxJSONNode(i)).box.w, 420, "…and the size travels with the design");
  // a run's value never paints over the editor
  __t.bxRunValues = { [i.id]: { shape: [3], peek: [4, 5, 6], mean: 5 } };
  __t.bxRender();
  ok(__t.bxCard(i).querySelectorAll(".dstate").length === 0,
     "…and the run's value does not draw over the editor");
  ok(fill().querySelectorAll(".panehint").length === 0,
     "…nor as text inside it");
  __t.bxRunValues = {};
  __t.bxRemove(i.id);
});

check("an agent refresh leaves the input box's editor alone", () => {
  const i = __t.bxAdd("input", 0, 0, { values: "4 5 6" });
  __t.bxRender();
  __t.bxAgentSync();                           // the agent state sweeps every pane
  const pane = getEl("bxcanvas").querySelectorAll(".bxlogpane")
    .find(p => p.getAttribute("data-agent") === i.id);
  ok(!!pane, "the input box has its pane");
  ok(pane.querySelectorAll(".bxsel").length === 1 &&
     pane.querySelectorAll(".bxmat").length === 1,
     "…still the input editor");
  ok(pane.querySelectorAll(".bxbbtn").length === 0 &&
     pane.querySelectorAll(".bxptitle").length === 0,
     "…not the agent's buttons and task list");
  __t.bxRemove(i.id);
});

check("a run's blame shows inside the box, never over its window", () => {
  const i = __t.bxAdd("input", 0, 0, { values: "4 5 6" });
  __t.bxMarkRunError({ nid: i.id, error: "input \u201cin\u201d needs numbers" });
  ok(__t.bxCard(i).querySelectorAll(".dstate").length === 0,
     "no blame text is painted over the editor");
  ok(__t.bxCard(i).className.indexOf("runerr") >= 0, "…the card wears the runerr mark");
  const d = document.createElement("div");
  __t.bxPaneFill(d, i);
  ok(d.querySelectorAll(".bxbhint.runbad").map(e => e.textContent).join("|")
       .indexOf("needs numbers") >= 0, "…and the message shows inside its window");
  __t.bxRunErrClear();
  __t.bxRemove(i.id);
});

check("the output window shows whatever data it receives", () => {
  const o = __t.bxAdd("output", 0, 0, { din: 3 });
  ok(__t.bxNodeW(o) >= 240 && __t.bxNodeH(o) >= 120, "the output box holds a window");
  ok(paneText(o).indexOf("no data yet") >= 0, "…saying so before a run");
  __t.bxRunValues = { [o.id]: { shape: [3], mean: 5, min: 4, max: 6, peek: [4, 5, 6] } };
  ok(paneText(o).indexOf("4, 5, 6") >= 0, "…and the received data after one: " + paneText(o));
  ok(paneText(o).indexOf("[3]") >= 0, "…with its shape under it");
  __t.bxRunValues = { [o.id]: { text: "the cat" } };
  ok(paneText(o).indexOf("the cat") >= 0, "…or the text it produced");
  __t.bxRunValues = {};
  __t.bxOutputExpand(o);
  eql([o.box.w, o.box.h], [420, 280], "…and a double-click grows it");
  eql(__t.bxJSONNodeFrom(__t.bxJSONNode(o)).box.w, 420, "…and the size travels with the design");
  __t.bxRemove(o.id);
});

check("re-placing a box keeps the last run's data (an edit still clears it)", () => {
  __t.BX.nodes.slice().forEach(x => __t.bxRemove(x.id));
  const i = __t.bxAdd("input", 0, 0, { dout: 2 });
  const o = __t.bxAdd("output", 240, 0, { din: 2 });
  __t.bxConnect(i.id, o.id);
  __t.bxRunResult({ ms: 5, device: "cuda", loaded: false,
                    values: { [i.id]: { shape: [2], mean: 1, peek: [1, 2] },
                              [o.id]: { shape: [2], mean: 3.5, peek: [3, 4] } },
                    texts: {}, flow: [i.id, o.id], generated: [] });
  const win = (n) => {
    const d = document.createElement("div");
    __t.bxPaneFill(d, n);
    return d.querySelectorAll(".bxdata").map(e => e.textContent).join(" | ");
  };
  ok(win(o).indexOf("3, 4") >= 0, "the output shows what it received: " + win(o));
  // drag the output somewhere else: the run's marks are not stale — they move with it
  __t.bxStartMove([o.id], o.id, 0, 0);
  __t.bxMove({ clientX: 60, clientY: 80 });
  __t.bxUp({});
  ok(o.x !== 240 || o.y !== 0, "the box moved");
  ok(Object.keys(__t.bxRunValues).length === 2, "…the run's values are kept");
  eql(__t.bxFlowSet.size, 2, "…and the flow stays lit");
  __t.bxRender();
  ok(win(o).indexOf("3, 4") >= 0, "…and the output still shows its data");
  // a real edit is a different story
  __t.bxCommit();
  ok(Object.keys(__t.bxRunValues).length === 0 && __t.bxFlowSet.size === 0,
     "an edit clears the run's marks");
  __t.bxRemove(i.id);
  __t.bxRemove(o.id);
});

check("the data boxes show the state of their data after a run", () => {
  const i = __t.bxAdd("input", 0, 0, { dout: 4 });
  const o = __t.bxAdd("output", 0, 200, { din: 4 });
  const s = __t.bxAdd("sample", 0, 400, { din: 4 });
  __t.bxConnect(i.id, o.id);
  const st = (n) => __t.bxCard(n).querySelectorAll(".dstate").map(e => e.textContent).join("");
  const win = (n) => paneText(n);
  __t.bxDataState = { [o.id]: { shape: [2, 2], mean: 0.5, peek: [0.1, 0.2, 0.3, 0.4] } };
  ok(st(o) === "", "the output paints nothing over its window");
  ok(win(o).indexOf("0.1, 0.2, 0.3, 0.4") >= 0, "…it shows the data it received in it");
  __t.bxDataState = { [o.id]: { shape: [4], mean: 2.5, min: 1, max: 4, peek: [1, 2, 3, 4] } };
  ok(win(o).indexOf("1, 2, 3, 4") >= 0, "…with the numbers");
  ok(win(o).indexOf("[4]") >= 0 && win(o).indexOf("2.5") >= 0,
     "…and the shape and mean under them");
  __t.bxDataState = { [o.id]: { text: "cab" } };
  ok(win(o).indexOf("\u201ccab\u201d") >= 0, "…a text result is quoted");
  __t.bxDataState = { [o.id]: { shape: [200], mean: 0.5, peek: [1, 2, 3, 4, 5, 6, 7, 8] } };
  ok(win(o).indexOf("\u2026") >= 0, "…a longer tensor trails off");
  __t.bxDataState = { [o.id]: { shape: [4], mean: 2.5 } };      // no peek: the summary
  ok(win(o).indexOf("[4]") >= 0 && win(o).indexOf("2.5") >= 0,
     "…and without content it falls back to the summary");
  // a box without a window still paints its state on the card
  __t.bxDataState = { [s.id]: { shape: [1], scalar: 15496, peek: [15496] } };
  ok(st(s) === "15496", "a sample box shows the picked id on its card");
  __t.bxCommit();
  ok(Object.keys(__t.bxDataState).length === 0, "an edit clears the state");
  __t.bxRemove(i.id);
  __t.bxRemove(o.id);
  __t.bxRemove(s.id);
});

check("the data flow: lights follow the backend's order across the wires", () => {
  const i = __t.bxAdd("input", 0, 0, { dout: 2 });
  const l = __t.bxAdd("layer", 0, 200, { din: 2, dout: 2 });
  const o = __t.bxAdd("output", 0, 400, { din: 2 });
  __t.bxConnect(i.id, l.id);
  __t.bxConnect(l.id, o.id);
  __t.bxFlowIngest([i.id, l.id, o.id]);
  eql([...__t.bxFlowSet], [i.id, l.id, o.id], "every box lights in order");
  eql(__t.bxFlowNow, o.id, "the newest light marks where the data is now");
  ok(__t.bxCard(i).className.indexOf("flow") >= 0, "the card wears the flow class");
  ok(__t.bxCard(o).className.indexOf("now") >= 0, "the live box wears the now class");
  ok(__t.bxCard(i).className.indexOf("now") < 0, "…the walked-past ones do not");
  __t.bxRender();
  const wires = getEl("bxcanvas").querySelectorAll(".dedge")
    .map(e => String(e.getAttribute("class")));
  eql(wires.filter(c => c.indexOf("flow") >= 0).length, 2, "…and both wires light");
  // feeding the same marks again changes nothing — they are per box
  __t.bxFlowIngest([i.id]);
  eql(__t.bxFlowSet.size, 3, "a repeated mark is not lit twice");
  __t.bxCommit();
  ok(__t.bxFlowSet.size === 0 && __t.bxFlowNow == null, "an edit clears the flow");
  __t.bxRemove(i.id);
  __t.bxRemove(l.id);
  __t.bxRemove(o.id);
});

check("a finished run lands its flow and paints the output at once", () => {
  const i = __t.bxAdd("input", 0, 0, { dout: 2 });
  const o = __t.bxAdd("output", 200, 0, { din: 2 });
  __t.bxConnect(i.id, o.id);
  __t.bxCommit();                              // a fresh edit: no old flow
  __t.bxFlowIngest([i.id]);                    // an animation still mid-flight
  __t.bxRunResult({ ms: 5, device: "cuda", loaded: true,
                    values: { [i.id]: { shape: [2], mean: 1 },
                              [o.id]: { shape: [2], mean: -3.5 } },
                    texts: {}, flow: [i.id, o.id], generated: [] });
  const d = document.createElement("div");
  __t.bxPaneFill(d, o);
  const state = d.querySelectorAll(".bxdata, .panehint").map(e => e.textContent).join(" | ");
  const lit = [...__t.bxFlowSet];
  __t.bxFlowClear();
  __t.bxDataState = {};
  __t.BXRUN.last = null;
  __t.bxRemove(i.id);
  __t.bxRemove(o.id);
  ok(state.indexOf("3.5") >= 0, "the output window shows its stats: " + state);
  eql(lit, [i.id, o.id], "the whole flow is marked at once — nothing left queued");
});

check("the live ring rides the whole chain, at every level", () => {
  const b = __t.bxAdd("custom", 0, 0, {});
  b.inner = { type: "nested", version: 1, nodes: [
      { id: "n1", kind: "layer", name: "l", x: 0, y: 0, din: 2, dout: 2 }], edges: [] };
  __t.bxFlowClear();
  __t.bxFlowIngest([b.id + "/n1"]);            // the data sits deep inside
  ok(__t.bxCard(b).className.indexOf("now") >= 0,
    "the block rings at the top — the live path is inside it");
  ok(__t.bxEnter(b.id), "inside");
  ok(__t.bxCard(b.inner.nodes[0]).className.indexOf("now") >= 0,
    "…and the leaf rings where it actually is");
  __t.bxLeave();
  __t.bxFlowIngest([b.id]);                    // the block itself finished
  ok(__t.bxCard(b).className.indexOf("now") >= 0, "finished: still the live box at the top");
  ok(__t.bxEnter(b.id) && __t.bxCard(b.inner.nodes[0]).className.indexOf("now") < 0,
    "…and inside, nothing pretends to be live");
  __t.bxLeave();
  __t.bxFlowClear();
  __t.bxRemove(b.id);
});

check("the glow paints the existing svg — classes only, never a rebuild", () => {
  const i = __t.bxAdd("input", 0, 0, { dout: 2 });
  const o = __t.bxAdd("output", 0, 200, { din: 2 });
  __t.bxConnect(i.id, o.id);
  __t.bxRender();
  const host = getEl("bxcanvas");
  const root1 = host.querySelector(".bxsvg");
  const wire = () => root1.querySelectorAll(".dedge")
    .map(e => String(e.getAttribute("class")));
  const card = (id) => root1.querySelectorAll(".dnode")
    .filter(g => g.getAttribute("data-id") === id)[0];
  ok(wire().every(c => c.indexOf("flow") < 0), "before: nothing is lit");
  __t.bxFlowIngest([i.id]);                    // the mark lands — no render call
  ok(host.querySelector(".bxsvg") === root1, "the very same svg element is on screen");
  ok(card(i.id).className.indexOf("flow") >= 0, "the box's class was flipped in place");
  ok(wire().some(c => c.indexOf("flow") >= 0), "…and the wire's too");
  ok(card(o.id).className.indexOf("flow") < 0, "the box without a mark stays dark");
  __t.bxFlowClear();
  ok(wire().every(c => c.indexOf("flow") < 0) &&
     card(i.id).className.indexOf("flow") < 0, "clearing strips them again");
  __t.bxRemove(i.id);
  __t.bxRemove(o.id);
});

check("a new forward pass restarts the sweep — a level switch never does", () => {
  __t.bxFlowClear();
  __t.bxFlowBeginStep(0);
  __t.bxFlowIngest(["a", "b"]);
  eql([...__t.bxFlowSet], ["a", "b"], "the pass marks its boxes");
  eql(__t.bxFlowBeginStep(0), false, "the same pass keeps them");
  eql(__t.bxFlowSet.size, 2, "\u2026untouched");
  ok(__t.bxFlowBeginStep(1), "a new pass");
  eql(__t.bxFlowSet.size, 0, "\u2026starts the sweep from scratch");
  eql(__t.bxFlowBeginStep(null), false, "no step at all changes nothing");
  __t.bxFlowClear();
});

check("marks are keyed by path — going inside never resets the walk", () => {
  const b1 = __t.bxAdd("custom", 0, 0, {});
  b1.inner = { type: "nested", version: 1, nodes: [
      { id: "n1", kind: "layer", name: "l", x: 0, y: 0, din: 2, dout: 2 }], edges: [] };
  const b2 = __t.bxAdd("custom", 0, 200, {});
  b2.inner = { type: "nested", version: 1, nodes: [
      { id: "n1", kind: "layer", name: "l", x: 0, y: 0, din: 2, dout: 2 }], edges: [] };
  __t.bxFlowClear();
  __t.bxFlowIngest([b1.id + "/n1", b2.id]);
  // at the top, the done block wears it — not the nested leaf nor a namesake
  ok(__t.bxCard(b2).className.indexOf("flow") >= 0, "the done block glows at the top");
  ok(__t.bxCard(b1).className.indexOf("flow") < 0, "a block that has not run stays dark");
  ok(__t.bxEnter(b1.id), "inside b1");
  const inner = __t.bxCard(b1.inner.nodes[0]).className;
  __t.bxLeave();
  ok(__t.bxEnter(b2.id), "inside b2");
  const other = __t.bxCard(b2.inner.nodes[0]).className;
  __t.bxLeave();
  ok(inner.indexOf("flow") >= 0, "b1's done leaf glows at once — no replay");
  ok(other.indexOf("flow") < 0, "…while b2's namesake stays dark");
  eql(__t.bxFlowSet.size, 2, "the marks survived both trips: " + [...__t.bxFlowSet]);
  __t.bxFlowClear();
  __t.bxRemove(b1.id);
  __t.bxRemove(b2.id);
});

check("a run's values land on whichever nesting level you stand on", () => {
  const b = __t.bxAdd("custom", 0, 0, {});
  b.inner = { type: "nested", version: 1, nodes: [
      { id: "in1", kind: "input", name: "in", x: 0, y: 0, dout: 2 },
      { id: "out1", kind: "output", name: "out", x: 200, y: 0, din: 2 }],
    edges: [] };
  const top = __t.bxAdd("output", 400, 0, { din: 2 });
  __t.bxRender();
  __t.bxRunResult({ ms: 1, flow: [], texts: {}, generated: [],
                    values: { [b.id + "/out1"]: { shape: [2], mean: -1.25 },
                              [top.id]: { shape: [2], mean: 3.5 } } });
  const win = (n) => paneText(n);
  const topState = win(top);
  __t.bxEnter(b.id);
  __t.bxRender();
  const inState = win(b.inner.nodes[1]);
  __t.bxLeave();
  __t.bxFlowClear();
  __t.bxDataState = {};
  __t.BXRUN.last = null;
  __t.bxRemove(b.id);
  __t.bxRemove(top.id);
  ok(topState.indexOf("3.5") >= 0, "the top output paints: " + topState);
  ok(inState.indexOf("1.25") >= 0, "the nested output paints too: " + inState);
});

check("Run inside a nested box says where the design runs", () => {
  const b = __t.bxAdd("custom", 0, 0, {});
  __t.bxEnter(b.id);
  __t.BXRUN.msg = "";
  __t.bxtRunGo();
  const msg = __t.BXRUN.msg;
  __t.bxLeave();
  __t.bxRemove(b.id);
  ok(msg.indexOf("top level") >= 0, "the message: " + msg);
});

check("a run's error line survives the progress that lands after it", () => {
  __t.BXRUN.running = true;
  __t.BXRUN.error = false;
  __t.BXRUN.msg = "building\u2026 40%";
  __t.bxRunStatusClear();
  eql(__t.BXRUN.msg, "", "a plain status line makes way for the flow");
  __t.BXRUN.msg = "could not run on rtx \u2014 box blew up";
  __t.BXRUN.error = true;
  __t.bxRunStatusClear();
  eql(__t.BXRUN.msg, "could not run on rtx \u2014 box blew up", "an error never does");
  __t.BXRUN.error = false;                   // the run ended: a late poll must be mute
  __t.BXRUN.running = false;
  __t.bxRunStatusClear();
  eql(__t.BXRUN.msg, "could not run on rtx \u2014 box blew up", "a late batch cannot wipe it");
  __t.BXRUN.msg = "";
  __t.BXRUN.running = true;
});

check("a failed run marks the block the backend blamed, and its containers", () => {
  const outer = __t.bxAdd("custom", 0, 0, {});
  outer.inner = { nodes: [{ id: "in1", kind: "layer", name: "bad layer", x: 0, y: 0,
                            din: 2, dout: 2 }], edges: [] };
  __t.bxRender();
  __t.bxMarkRunError({ nid: "in1", box: "bad layer", error: "bad layer: it blew up" });
  eql([...__t.bxRunErr.ids].sort(), ["in1", outer.id].sort(),
    "the whole chain is marked — the container and the failing box");
  ok(__t.bxRunErr.here === "in1", "…and the failing box is the one named");
  ok(__t.bxCard(outer).className.indexOf("runerr") >= 0, "the outer card carries the class");
  ok(__t.bxCard(outer).querySelectorAll(".dstate").map(e => e.textContent).join("")
       === "failed inside", "…and says the run failed inside");
  ok(__t.bxEnter(outer.id), "go inside");
  const inner = outer.inner.nodes[0];
  __t.bxRender();
  ok(__t.bxCard(inner).className.indexOf("runerr") >= 0, "the failing box inside carries it too");
  ok(__t.bxCard(inner).querySelectorAll(".dstate").map(e => e.textContent).join("")
       .indexOf("blew up") >= 0, "…with the reason on it");
  __t.bxLeave();
  const l = __t.bxAdd("layer", 0, 200, { din: 2, dout: 2 });
  __t.bxMarkRunError({ nid: l.id, box: l.name, error: "x: no value was given" });
  ok(__t.bxRunErr.here === l.id && __t.bxRunErr.ids.has(l.id) &&
     __t.bxCard(l).className.indexOf("runerr") >= 0, "a top-level box is marked itself");
  ok(__t.bxCard(outer).className.indexOf("runerr") < 0, "…and the old mark is gone");
  __t.bxMarkRunError({ box: "bad layer", error: "bad layer: again" });
  ok(__t.bxRunErr.here === "in1" && __t.bxRunErr.ids.has(outer.id),
    "with no id, the name still finds the chain");
  __t.bxCommit();
  ok(__t.bxRunErr == null, "an edit clears the mark");
  __t.bxRemove(outer.id);
  __t.bxRemove(l.id);
});

check("the wheel zooms gently, and follows the scroll amount", () => {
  const near = (a, b, eps) => Math.abs(a - b) < eps;
  const one = __t.bxWheelFactor(-100, 0);          // one mouse notch up
  ok(near(one, 1.051, 0.002), "a notch is ~5 %: " + one.toFixed(4));
  ok(near(__t.bxWheelFactor(100, 0), 1 / one, 0.001), "…and the same going out");
  ok(__t.bxWheelFactor(-4, 0) < 1.01, "a trackpad tick barely moves it: " +
     __t.bxWheelFactor(-4, 0).toFixed(5));
  ok(near(__t.bxWheelFactor(-3, 1), __t.bxWheelFactor(-48, 0), 0.002),
     "line deltas are normalised (3 lines ≈ 48 px)");
  ok(near(__t.bxWheelFactor(-1, 2), __t.bxWheelFactor(-100, 0), 0.002), "…pages too (×100)");
  ok(__t.bxWheelFactor(-100000, 0) <= Math.pow(1.0005, 240) + 1e-9,
     "a violent scroll still lands one bounded step");
  eql(__t.bxWheelFactor(0, 0), 1, "no scroll, no zoom");
});

check("a box with no size shows ?, and clearing a size brings it back", () => {
  const fresh = __t.bxAdd("input", 0, 0, {});
  eql(fresh.dout, null, "a fresh box has no size");
  eql(__t.bxShape(fresh), "?", "…and reads ?");
  const l = __t.bxAdd("layer", 0, 90, {});
  eql(__t.bxShape(l), "in ? → out ?", "a fresh fully-connected box too");
  eql(__t.bxSize(""), null, "an emptied field means unconfigured");
  eql(__t.bxSize(undefined), null, "…as does a missing one");
  eql(__t.bxSize("5120"), 5120, "a real size sticks");
  eql(__t.bxSize("-9"), 1, "…and is clamped positive");
  eql(__t.bxSize("junk", 4096), 4096, "junk keeps what was there");
  __t.bxConfigure(l.id, { din: 4096, dout: 512 });
  eql(__t.bxShape(l), "in 4096 → out 512", "configuring fills it in");
  __t.bxConfigure(l.id, { din: "" });
  eql(__t.bxShape(l), "in ? → out 512", "clearing one side makes just that side ?");
  __t.bxRemove(fresh.id);
  __t.bxRemove(l.id);
});

check("sizes check on connect: unknown takes the known, a real mismatch errors", () => {
  const a = __t.bxAdd("input", 0, 0, {});           // out: ?
  const l = __t.bxAdd("layer", 200, 0, {});         // in: ?, out: ?
  l.din = 5120;                                     // the known end
  ok(!__t.bxCheckLink(a.id, l.id).error, "an unknown end links fine");
  ok(typeof __t.bxToast === "undefined", "…and with no message drawn anywhere");
  eql(__t.bxConnect(a.id, l.id), null, "…the wire lands");
  eql(a.dout, 5120, "the unknown side takes the known size");
  eql(l.din, 5120, "the known side is untouched");
  const l2 = __t.bxAdd("layer", 400, 0, {});
  l2.din = null; l2.dout = 256;
  const o = __t.bxAdd("output", 600, 0, {});        // in: ?
  eql(__t.bxConnect(l2.id, o.id), null, "a wire into an unknown in");
  eql(o.din, 256, "…the output box takes the source's out");
  const l3 = __t.bxAdd("layer", 800, 0, {});
  l3.din = 300;
  const err = __t.bxConnect(l2.id, l3.id);          // out 256 → in 300
  ok(err && /size mismatch/.test(err), "a real mismatch is refused: " + err);
  ok(!__t.BX.edges.some(e => e.from === l2.id && e.to === l3.id), "…and makes no wire");
  eql(__t.bxEdgeMismatch({ from: l2.id, to: o.id }), null, "matching wires are fine");
  __t.bxConfigure(o.id, { din: 999 });
  const broke = __t.bxEdgeMismatch({ from: l2.id, to: o.id });
  ok(broke && /\u2260/.test(broke), "a resize that breaks a wire is reported: " + broke);
  eql(__t.bxCheckLink(a.id, a.id).error != null, true, "a box cannot feed itself");
  // every re-configure runs the check again on the wires that touch the box
  const p = __t.bxAdd("input", 0, 300, {});
  const q = __t.bxAdd("layer", 200, 300, {});        // both ends unknown
  eql(__t.bxConnect(p.id, q.id), null, "two unknown ends wire fine");
  __t.bxConfigure(q.id, { din: 2048 });
  eql(p.dout, 2048, "configuring one end fills the unknown end through the wire");
  const r = __t.bxAdd("output", 400, 300, {});
  __t.bxConfigure(r.id, { din: 2048 });
  eql(__t.bxConnect(q.id, r.id), null, "a matching pair wires");
  __t.bxConfigure(q.id, { dout: 999 });
  ok(__t.bxEdgeMismatch({ from: q.id, to: r.id }),
    "a re-configure that breaks a wire flags it (never silently rewrites the other end)");
  eql(r.din, 2048, "…and the other end is left alone");
  [p, q, r].forEach(n => __t.bxRemove(n.id));
  [a, l, l2, o, l3].forEach(n => __t.bxRemove(n.id));
});

check("the size check sweeps the whole network, not only the wire it touched", () => {
  // an activation chain has no stored size of its own: the far end has to
  // travel back through two boxes to reach the source box
  const i = __t.bxAdd("input", 0, 0, {});
  const a1 = __t.bxAdd("act", 200, 0, {});
  const a2 = __t.bxAdd("act", 400, 0, {});
  const o = __t.bxAdd("output", 600, 0, {});
  eql(__t.bxConnect(i.id, a1.id), null, "input → activation (both unknown)");
  eql(__t.bxConnect(a1.id, a2.id), null, "activation → activation");
  eql(__t.bxConnect(a2.id, o.id), null, "activation → output");
  __t.bxConfigure(o.id, { din: 4096 });
  eql(i.dout, 4096, "the output's size reaches the input box at the other end of the chain");
  eql(__t.bxWireState(edgeOf(i.id, a1.id)), "match", "…every wire in the chain matches now");
  eql(__t.bxWireState(edgeOf(a1.id, a2.id)), "match", "…including between the activations");
  // and the sweep also walks boxes that were wired but never configured
  const i2 = __t.bxAdd("input", 0, 300, {});
  const m = __t.bxAdd("layer", 200, 300, {});
  const m2 = __t.bxAdd("layer", 400, 300, {});
  const o2 = __t.bxAdd("output", 600, 300, {});
  eql(__t.bxConnect(i2.id, m.id), null, "input → fully connected");
  eql(__t.bxConnect(m.id, m2.id), null, "fully connected → fully connected (both unknown)");
  eql(__t.bxConnect(m2.id, o2.id), null, "fully connected → output");
  __t.bxConfigure(i2.id, { dout: 5120 });
  eql(m.din, 5120, "the input's size walks one hop");
  __t.bxConfigure(o2.id, { din: 2560 });
  eql(m2.dout, 2560, "the output's size walks back one hop");
  eql(m.dout, null, "…and stops at the chain's middle (a fully connected box is not a pass-through)");
  eql(m2.din, null, "…on both sides");
  [i, a1, a2, o, i2, m, m2, o2].forEach(n => __t.bxRemove(n.id));
});

check("sizes the check filled follow along when the source moves", () => {
  const i = __t.bxAdd("input", 0, 0, {});
  const a = __t.bxAdd("act", 200, 0, {});
  const l = __t.bxAdd("layer", 400, 0, {});
  __t.bxConnect(i.id, a.id);
  __t.bxConnect(a.id, l.id);
  __t.bxConfigure(i.id, { dout: 128 });
  eql(l.din, 128, "the unknown end took the size");
  __t.bxConfigure(i.id, { dout: 256 });
  eql([l.din, __t.bxWireState(edgeOf(a.id, l.id))], [256, "match"],
    "the filled size follows the source — no stale red wire");
  __t.bxConfigure(l.id, { din: 300 });          // typed now: it holds its ground
  __t.bxConfigure(i.id, { dout: 512 });
  eql([l.din, __t.bxWireState(edgeOf(a.id, l.id))], [300, "bad"],
    "a typed size holds its ground — the mismatch stays visible");
  // and the follow-through crosses nested boundaries
  const src = __t.bxAdd("input", 0, 300, {});
  const nest = __t.bxAdd("custom", 200, 300, {});
  nest.inner = { nodes: [{ id: "ii", kind: "input", name: "in", x: 0, y: 0 },
                         { id: "io", kind: "output", name: "out", x: 300, y: 0 }],
                 edges: [{ from: "ii", to: "io" }] };
  const dst = __t.bxAdd("output", 600, 300, {});
  __t.bxConnect(src.id, nest.id);
  __t.bxConnect(nest.id, dst.id);
  __t.bxConfigure(src.id, { dout: 40 });
  eql(nest.inner.nodes[0].dout, 40, "an outer size fills the inner boundary");
  __t.bxConfigure(src.id, { dout: 80 });
  eql([nest.inner.nodes[0].dout, nest.inner.nodes[1].din, dst.din], [80, 80, 80],
    "…and the whole run follows when it moves");
  [i, a, l, src, nest, dst].forEach(n => __t.bxRemove(n.id));
});

check("the sweep reaches inside nested boxes, and their ports carry it out", () => {
  const src = __t.bxAdd("input", 0, 0, {});
  const nest = __t.bxAdd("custom", 200, 0, {});
  const dst = __t.bxAdd("output", 600, 0, {});
  ok(__t.bxEnter(nest.id), "go inside");
  const ii = __t.bxAdd("input", 0, 0, {});
  const mid = __t.bxAdd("layer", 200, 0, {});
  const io = __t.bxAdd("output", 400, 0, {});
  eql(__t.bxConnect(ii.id, mid.id), null, "inner input → inner fully connected");
  eql(__t.bxConnect(mid.id, io.id), null, "inner fully connected → inner output");
  __t.bxLeave();
  eql(__t.bxConnect(src.id, nest.id), null, "input → nested");
  eql(__t.bxConnect(nest.id, dst.id), null, "nested → output");
  __t.bxConfigure(src.id, { dout: 5120 });
  eql(ii.dout, 5120, "the outer input's size runs through the nested box to its inner input");
  eql(mid.din, 5120, "…and one more hop, inside, onto the inner box");
  __t.bxConfigure(dst.id, { din: 2560 });
  eql(io.din, 2560, "the outer output's size runs back inside to its inner output");
  eql(mid.dout, 2560, "…and onto the inner box's other side");
  eql(__t.bxWireState(edgeOf(src.id, nest.id)), "match", "the outer wire matches through the nested ports");
  eql(__t.bxWireState(edgeOf(nest.id, dst.id)), "match", "…both of them");
  ok(__t.bxAllDocs().length >= 2, "…because the sweep sees both levels");
  [src, nest, dst].forEach(n => __t.bxRemove(n.id));
});

check("a canvas round-trips through JSON, nested boxes and all", () => {
  // a spec's own op attrs (the scan's heads, a reshape's kept dims) may not
  // be modeled by the UI, but a run must never receive a stripped box
  const spec = __t.bxAdd("op", 0, 0, {});
  __t.bxConfigure(spec.id, { op: "scan" });
  spec.k_heads = 16; spec.head_dim = 128; spec.keep = 1;
  const soldered = __t.bxJSONNodeFrom(__t.bxJSONNode(spec));
  __t.bxRemove(spec.id);
  ok(soldered && soldered.k_heads === 16 && soldered.head_dim === 128 &&
     soldered.keep === 1, "spec op attrs survive export/import: " + JSON.stringify(soldered));
  const foreign = __t.BX.nodes.map(n => n.id);      // earlier checks may leave boxes
  const mine = () => __t.BX.nodes.filter(n => foreign.indexOf(n.id) < 0);
  const i = __t.bxAdd("input", 0, 0, {});
  const a = __t.bxAdd("act", 200, 0, {});
  const c = __t.bxAdd("custom", 400, 0, {});
  const o = __t.bxAdd("output", 600, 0, {});
  ok(__t.bxEnter(c.id), "inside the nested box");
  const ii = __t.bxAdd("input", 0, 0, {});
  const m = __t.bxAdd("layer", 200, 0, {});
  const io = __t.bxAdd("output", 400, 0, {});
  __t.bxConnect(ii.id, m.id);
  __t.bxConnect(m.id, io.id);
  __t.bxLeave();
  __t.bxConnect(i.id, a.id);
  __t.bxConnect(a.id, c.id);
  __t.bxConnect(c.id, o.id);
  __t.bxSetVia(edgeOf(i.id, a.id), 240, 40);
  __t.bxConfigure(i.id, { dout: 5120 });
  __t.bxConfigure(o.id, { din: 2560 });
  const before = JSON.stringify(__t.bxExportCanvas());
  ok(__t.bxJSONRead(before, {}).replaces, "loading it back would replace work — the form warns first");
  __t.bxEnter(c.id);
  eql(__t.bxExportCanvas(), null, "no canvas export from inside a nested box");
  eql(__t.bxImportJSON(before, {}).ok, false, "…and no canvas load either");
  eql(__t.BX.nodes.length, 3, "…so the nested box's own boxes never moved");
  __t.bxLeave();
  const res = __t.bxImportJSON(before, {});
  eql(res.ok, true, "the canvas loads back at the top level");
  eql(res.kind, "canvas", "…as a canvas");
  eql(JSON.stringify(__t.bxExportCanvas()), before, "the design survives the round trip untouched");
  eql(mine().length, 4, "…with all four of our boxes");
  const nest = mine().find(n => n.kind === "custom");
  eql(nest.inner.nodes.length, 3, "the nested box kept its own boxes");
  eql(nest.inner.edges.length, 2, "…and its wires");
  eql(__t.BX.nodes.find(n => n.kind === "input").dout, 5120, "the sizes round-trip");
  ok(edgeOf(i.id, a.id).via && edgeOf(i.id, a.id).via.x === 240, "…and a bent wire's shape");
  [i, a, c, o].forEach(n => __t.bxRemove(n.id));
});

check("a nested box takes nested JSON and never a canvas", () => {
  const c = __t.bxAdd("custom", 0, 0, {});
  ok(__t.bxEnter(c.id), "inside");
  const ii = __t.bxAdd("input", 0, 0, {});
  const io = __t.bxAdd("output", 200, 0, {});
  __t.bxConnect(ii.id, io.id);
  __t.bxLeave();
  const nested = __t.bxExportNested(c.id);
  eql(nested.type, "nested", "a nested export says nested");
  eql(nested.name, c.name, "…and carries the box's name");
  eql(JSON.stringify(nested).indexOf('"canvas"'), -1, "…with no canvas hidden in it");
  const box2 = __t.bxAdd("custom", 400, 0, {});
  eql(__t.bxImportJSON(JSON.stringify(nested), { box: box2.id }).ok, true, "a nested design loads into a box");
  eql(box2.inner.nodes.length, 2, "…its inside is there");
  const canvas = __t.bxExportCanvas();
  const res = __t.bxImportJSON(JSON.stringify(canvas), { box: box2.id });
  eql(res.ok, false, "a canvas is refused inside a nested box");
  eql(box2.inner.nodes.length, 2, "…and the box is untouched");
  eql(__t.bxImportJSON(JSON.stringify(nested), { x: 100, y: 100 }).ok, true, "a canvas can load a nested design");
  const fresh = __t.BX.nodes.filter(n => n.kind === "custom").filter(n => n.id !== c.id && n.id !== box2.id)[0];
  ok(fresh && fresh.inner.nodes.length === 2, "…as a new nested box with its own network");
  const bad = { type: "nested", version: 1, nodes: [{ kind: "custom", x: 0, y: 0,
    inner: { type: "canvas", version: 1, nodes: [], edges: [] } }], edges: [] };
  eql(__t.bxImportJSON(JSON.stringify(bad), { box: c.id }).ok, false, "a canvas hidden inside a nested payload is refused");
  eql(__t.bxImportJSON(JSON.stringify(bad), {}).ok, false, "…at the canvas level too");
  // inside a nested box the canvas items are gone; a nested design loads anywhere
  ok(__t.bxEnter(c.id), "inside a nested box");
  __t.bxMenuBuild("canvas", null, { clientX: 10, clientY: 10 });
  const nmenu = getEl("bxmenu");
  const nheads = () => nmenu.children.filter(x => String(x.className).split(/\s+/).includes("drop"));
  const nopen = () => nmenu.children.filter(x => x.className === "mi").map(x => x.textContent);
  eql(nmenu.children.map(x => x.textContent),
    ["boundaries", "weights", "text", "tensor ops", "blocks", "containers", "the design"],
    "inside a nested box the categories are all there, collapsed");
  nheads().find(x => x.textContent === "containers").click();
  eql(nopen(), ["nested box"], "containers is just the nested box, any depth");
  nheads().find(x => x.textContent === "the design").click();
  eql(nopen(), ["import a nested box …"],
    "the design has no canvas to save or load here");
  __t.bxMenuClose();
  const inner = __t.BX.nodes.length;
  eql(__t.bxImportJSON(JSON.stringify(nested), { x: 10, y: 10 }).ok, true,
    "a nested design loads inside a nested box too");
  eql(__t.BX.nodes.length, inner + 1, "…as a box in that nested box");
  eql(__t.BX.nodes[inner].kind, "custom", "…a nested box");
  eql(__t.BX.nodes[inner].inner.nodes.length, 2, "…holding the design");
  eql(__t.bxImportJSON(JSON.stringify(canvas), {}).ok, false, "a canvas pasted in there is refused");
  __t.bxLeave();
  [c, box2, fresh].forEach(n => n && __t.bxRemove(n.id));
});

check("bad designs are refused and change nothing", () => {
  const before = JSON.stringify(__t.bxExportCanvas());
  eql(__t.bxImportJSON("{oops", {}).ok, false, "broken JSON");
  eql(__t.bxImportJSON(JSON.stringify({ type: "canvas", nodes: [{ kind: "banana", x: 0, y: 0 }], edges: [] }), {}).ok, false, "an unknown kind");
  eql(__t.bxImportJSON(JSON.stringify({ type: "canvas", nodes: [], edges: [{ from: "nope", to: "nope" }] }), {}).ok, false, "a wire to nowhere");
  eql(__t.bxImportJSON(JSON.stringify({ type: "canvas", version: 1, nodes: [
        { id: "x", kind: "input", x: 0, y: 0 }, { id: "x", kind: "output", x: 0, y: 0 }], edges: [] }), {}).ok, false, "duplicate ids");
  eql(__t.bxImportJSON(JSON.stringify({ type: "nested", version: 1, nodes: [], edges: [] }), { box: "nope" }).ok, false, "a box that is not there");
  eql(JSON.stringify(__t.bxExportCanvas()), before, "…and the canvas never moved");
  eql(__t.bxImportJSON(JSON.stringify({ type: "canvas", version: 1, nodes: [], edges: [] }), {}).ok, true, "an empty canvas is a design too");
  eql(__t.BX.nodes.length, 0, "…it clears the place");
  eql(__t.bxJSONRead(before, {}).replaces, false, "…and loading into an empty canvas needs no warning");
  eql(__t.bxImportJSON(before, {}).ok, true, "…and the old design loads right back");
  const two = { type: "nested", version: 1, nodes: [
        { id: "a", kind: "input", x: 0, y: 0 }, { id: "b", kind: "output", x: 0, y: 0 }],
        edges: [{ from: "a", to: "b" }, { from: "a", to: "b" }] };
  eql(__t.bxImportJSON(JSON.stringify(two), {}).ok, true, "a repeated wire loads");
  const box = __t.BX.nodes[__t.BX.nodes.length - 1];
  eql(box.inner.edges.length, 1, "…as one wire (the last one wins, like a rewire)");
  __t.bxRemove(box.id);
});

check("the size rule covers every kind, and ?/? wires are grey", () => {
  const mk = (kind, y, vals) => __t.bxAdd(kind, 0, y, vals || {});
  const in1 = mk("input", 0), fc1 = mk("layer", 100), out1 = mk("output", 200), fc2 = mk("layer", 300);
  // input → fully connected: the FC's unknown in takes the input's out
  __t.bxConfigure(in1.id, { dout: 5120 });
  eql(__t.bxConnect(in1.id, fc1.id), null, "input → fully connected");
  eql(fc1.din, 5120, "…the FC's in follows the input's out");
  eql(__t.bxWireState(edgeOf(in1.id, fc1.id)), "match", "known and equal");
  // fully connected → output: the output's unknown in takes the FC's out
  __t.bxConfigure(fc1.id, { dout: 256 });
  eql(__t.bxConnect(fc1.id, out1.id), null, "fully connected → output");
  eql(out1.din, 256, "…the output's in follows the FC's out");
  // fully connected → fully connected: two known sizes that disagree are refused
  __t.bxConfigure(fc2.id, { din: 256, dout: 999 });
  ok(__t.bxConnect(fc2.id, fc1.id), "999 out vs 5120 in is refused (all kinds check): " +
     __t.bxConnect(fc2.id, fc1.id));
  __t.bxConfigure(fc2.id, { dout: 5120 });
  eql(__t.bxConnect(fc2.id, fc1.id), null, "…and connects once they agree");
  // a source with an unknown out simply takes the target's in
  const fc3 = __t.bxAdd("layer", 700, 0, {});
  eql(__t.bxConnect(fc3.id, fc1.id), null, "an unknown out wires into a known in");
  eql(fc3.dout, 5120, "…and the source's out follows");
  __t.bxRemove(fc3.id);
  // the three states of a settled wire
  const u1 = mk("input", 400), u2 = mk("layer", 500);
  eql(__t.bxConnect(u1.id, u2.id), null, "two unknown ends still wire");
  eql(__t.bxWireState(edgeOf(u1.id, u2.id)), "unknown", "…and the wire is 'unknown' (grey)");
  eql(__t.bxConnect(u1.id, u2.id), null, "re-wiring is fine too");
  __t.bxConfigure(u2.id, { din: 128 });              // the check runs on a size change
  eql(u1.dout, 128, "…configure one end and the unknown end fills in");
  eql(__t.bxWireState(edgeOf(u1.id, u2.id)), "match", "…turning the grey wire into a match");
  __t.bxConfigure(u2.id, { din: 999 });
  eql([u1.dout, __t.bxWireState(edgeOf(u1.id, u2.id))], [999, "match"],
    "…and the filled end follows when the size moves again");
  __t.bxConfigure(u1.id, { dout: 512 });             // now both ends are typed
  eql(__t.bxWireState(edgeOf(u1.id, u2.id)), "bad",
    "…but two typed sizes that differ break it");
  [in1, fc1, out1, fc2, u1, u2].forEach(n => __t.bxRemove(n.id));
});

check("the activation box: its own kind, its function list, transparent to sizes", () => {
  const fc = __t.bxAdd("layer", 0, 0, {});
  __t.bxConfigure(fc.id, { din: 5120, dout: 5120 });
  const a = __t.bxAdd("act", 200, 0, {});
  ok(/^activation \d+$/.test(a.name), "a new one is named like: " + a.name);
  eql([a.din, a.dout], [null, null], "…and stores no sizes");
  eql(__t.bxShape(a), "silu", "it starts on silu");
  ok(__t.BX_ACTS.indexOf("gelu") >= 0 && __t.BX_ACTS.indexOf("none") >= 0,
    "the list: " + __t.BX_ACTS.join(", "));
  __t.bxConfigure(a.id, { act: "gelu" });
  eql(__t.bxShape(a), "gelu", "configuring switches the function");
  __t.bxConfigure(a.id, { act: "nonsense" });
  eql(__t.bxShape(a), "gelu", "…and junk leaves it alone");
  // its menu offers the function instead of a size
  __t.bxMenuBuild("box", a.id, {});
  eql(getEl("bxmenu").children.map(c => c.textContent),
    ["this box", "flow", "remove this box"], "an activation box's menu categories");
  openCat(getEl("bxmenu"), "this box");
  eql(catItems(getEl("bxmenu")), ["rename this box", "copy this box", "set the activation"],
    "…this box holds the function");
  openCat(getEl("bxmenu"), "flow");
  eql(catItems(getEl("bxmenu")), ["deactivate this box", "freeze this box"],
    "…flow holds the switches");
  __t.bxMenuClose();
  // it passes the size through: fc(5120) → act → output gets 5120
  const o = __t.bxAdd("output", 400, 0, {});
  eql(__t.bxConnect(fc.id, a.id), null, "a sized box wires into it");
  eql(__t.bxWireState(edgeOf(fc.id, a.id)), "match", "…and the wire matches through it");
  eql(__t.bxConnect(a.id, o.id), null, "…and it wires on to an output");
  eql(o.din, 5120, "the output takes the size the activation passes through");
  eql(__t.bxWireState(edgeOf(a.id, o.id)), "match", "…so that wire matches too");
  __t.bxConfigure(fc.id, { dout: 256 });
  eql([o.din, __t.bxWireState(edgeOf(a.id, o.id))], [256, "match"],
    "…and a filled chain end follows the new size");
  __t.bxConfigure(o.id, { din: 777 });               // typed now: it holds
  __t.bxConfigure(fc.id, { dout: 512 });
  eql([o.din, __t.bxWireState(edgeOf(a.id, o.id))], [777, "bad"],
    "…until the far end is typed, when the mismatch shows downstream");
  [fc, a, o].forEach(n => __t.bxRemove(n.id));
});

check("static operation boxes: transpose, dot product, arithmetic, reshape", () => {
  const i = __t.bxAdd("input", 0, 0, {});
  const t = __t.bxAdd("op", 200, 0, {});
  const o = __t.bxAdd("output", 400, 0, {});
  eql(t.op, "transpose", "a new operation box starts as transpose");
  eql(__t.bxKindLabel("op"), "Operation", "the kind is called Operation");
  ok(/^operation \d+$/.test(t.name), "…and is named like: " + t.name);
  eql(__t.bxShape(t), "transpose", "the card shows the operation");
  eql(__t.bxPortDefs(t, "in").length, 1, "transpose takes one wire");
  eql(__t.bxPortDefs(t, "out").length, 1, "…and sends one");
  eql(__t.bxConnect(i.id, t.id), null, "input → transpose");
  eql(__t.bxConnect(t.id, o.id), null, "transpose → output");
  __t.bxConfigure(o.id, { din: 5120 });
  eql(i.dout, 5120, "the output's size runs back through the transpose to the input");
  eql(__t.bxWireState(edgeOf(i.id, t.id)), "match", "…both wires match");
  eql(__t.bxWireState(edgeOf(t.id, o.id)), "match", "…straight through the operation");
  // arithmetic: two in-dots of one size, the out-dot the same size
  const l = __t.bxAdd("layer", 0, 300, {});
  const a = __t.bxAdd("op", 200, 300, {});
  __t.bxConfigure(a.id, { op: "multiply" });
  eql(a.op, "multiply", "the operation can be set");
  eql(__t.bxPortDefs(a, "in").length, 2, "multiply takes two wires");
  __t.bxConfigure(l.id, { din: 5120, dout: 5120 });
  eql(__t.bxConnect(l.id, a.id), null, "the first wire lands");
  eql(a.din, 5120, "…and fills the operation's size");
  eql(__t.bxOutSize(a), 5120, "the out-dot follows element-wise");
  const l2 = __t.bxAdd("layer", 0, 500, {});
  __t.bxConfigure(l2.id, { din: 5120, dout: 5120 });
  eql(__t.bxConnect(l2.id, a.id, 0, 1), null, "a same-size second wire lands on the second dot");
  const l3 = __t.bxAdd("layer", 0, 700, {});
  __t.bxConfigure(l3.id, { din: 5120, dout: 256 });
  ok(__t.bxConnect(l3.id, a.id, 0, 1), "256 into the 5120 operation is refused");
  // the dot product: two in-dots, one scalar out
  const d = __t.bxAdd("op", 600, 300, {});
  __t.bxConfigure(d.id, { op: "dot product" });
  const o2 = __t.bxAdd("output", 800, 300, {});
  eql(__t.bxConnect(d.id, o2.id), null, "dot product → output");
  eql(o2.din, 1, "the dot product sends a scalar: the output takes size 1");
  const big = __t.bxAdd("output", 800, 500, {});
  __t.bxConfigure(big.id, { din: 5120 });
  ok(__t.bxConnect(d.id, big.id), "a size-1 dot output cannot feed a 5120 input");
  // its menu offers the operation instead of a size
  __t.bxMenuBuild("box", a.id, {});
  openCat(getEl("bxmenu"), "this box");
  eql(catItems(getEl("bxmenu")), ["rename this box", "copy this box", "set the operation"],
    "an operation box's menu");
  __t.bxMenuClose();
  // a smaller operation drops the wire that lost its dot
  eql(__t.BX.edges.filter(e => e.to === a.id).length, 2, "two wires into the multiply");
  __t.bxConfigure(a.id, { op: "reshape" });
  eql(__t.BX.edges.filter(e => e.to === a.id).length, 1, "reshape keeps one: the extra wire is dropped");
  eql(__t.BX.edges.filter(e => e.to === a.id)[0].toPort, 0, "…the one on the first dot");
  eql(__t.bxOutSize(a), 5120, "reshape passes the size through, like transpose");
  // …and the operation is what the JSON records
  const json = JSON.parse(JSON.stringify(__t.bxExportCanvas()));
  const rec = json.nodes.find(n => n.id === a.id);
  eql(rec.op, "reshape", "the JSON records the operation");
  eql(__t.bxImportJSON(JSON.stringify(json), {}).ok, true, "…and loads back");
  eql(__t.bxNode(a.id).op, "reshape", "…with its operation");
  [i, t, o, l, a, l2, l3, d, o2, big].forEach(n => __t.bxRemove(n.id));
});

check("a slice can be told which piece to take (axis, index, start/stop)", () => {
  const s = __t.bxAdd("op", 200, 0, {});
  const live = () => __t.bxNode(s.id);         // an import re-makes the node objects
  __t.bxConfigure(s.id, { op: "slice", axis: 0, index: -1 });
  eql([live().op, live().axis, live().index], ["slice", 0, -1],
    "axis 0, index -1: the last slot along the first axis");
  eql(__t.bxShape(live()), "axis 0 \u00b7 index -1", "the card reads it: " + __t.bxShape(live()));
  const rec = __t.bxJSONNode(live());
  eql([rec.op, rec.axis, rec.index], ["slice", 0, -1], "\u2026and the design carries them");
  eql(__t.bxImportJSON(JSON.stringify(__t.bxExportCanvas()), {}).ok, true, "\u2026and loads back");
  eql([live().axis, live().index], [0, -1], "\u2026with its slice intact");
  // start/stop is the other half: a range along that axis
  __t.bxConfigure(s.id, { index: "", start: "3", stop: "5" });
  eql([live().index, live().start, live().stop],
    [null, 3, 5], "clearing index hands it to start/stop");
  eql(__t.bxShape(live()), "axis 0 \u00b7 3:5", "\u2026which the card reads too");
  __t.bxConfigure(s.id, { op: "slice", axis: 0, index: -1 });
  // the form asks for the four \u2014 and only for a slice
  __t.bxForm("op", null, __t.bxNode(s.id), "operation");
  const wrap = getEl("bxmenu").children[0];
  const lab = (t) => wrap.children.find(c => String(c.textContent) === t);
  eql([lab("axis").hidden, lab("index").hidden, lab("start").hidden, lab("stop").hidden],
    [false, false, false, false], "a slice's form asks for its four fields");
  ok(String(lab("index").children[0].value) === "-1", "\u2026with the slice it has");
  __t.bxMenuClose();
  __t.bxConfigure(s.id, { op: "reshape", dims: "2x8" });
  __t.bxForm("op", null, __t.bxNode(s.id), "operation");
  const wrap2 = getEl("bxmenu").children[0];
  const lab2 = (t) => wrap2.children.find(c => String(c.textContent) === t);
  eql([lab2("axis").hidden, lab2("output dim").hidden], [true, false],
    "\u2026hidden for a reshape, which asks for its output dim instead");
  __t.bxMenuClose();
  __t.bxRemove(s.id);
});

check("reshape can be told the shape it makes (its output dim)", () => {
  const src = __t.bxAdd("layer", 0, 0, {});
  const r = __t.bxAdd("op", 200, 0, {});
  __t.bxConfigure(r.id, { op: "reshape" });
  eql(__t.bxShape(r), "reshape", "a bare reshape passes through, as before");
  __t.bxConfigure(src.id, { din: 4096, dout: 5120 });
  eql(__t.bxConnect(src.id, r.id), null, "a 5120 source wires in");
  eql(__t.bxReshapeSize(r), null, "…and nothing is shaped yet");
  __t.bxConfigure(r.id, { dims: "128x40" });
  eql(__t.bxDims(r.dims), [128, 40], "the shape is kept as dims");
  eql(__t.bxReshapeSize(r), 5120, "…its element count is the product");
  eql(__t.bxShape(r), "5120 \u2192 128\u00d740", "the card shows the arrangement");
  eql(__t.bxPortSizes(r, "in"), [5120], "the input dot asks for 5120");
  eql(__t.bxPortSizes(r, "out"), [5120], "…and the output dot carries it");
  const out = __t.bxAdd("output", 400, 0, {});
  eql(__t.bxConnect(r.id, out.id), null, "reshape → output");
  eql(out.din, 5120, "…the output takes the element count");
  const bad = __t.bxAdd("layer", 0, 300, {});
  __t.bxConfigure(bad.id, { din: 4096, dout: 2560 });
  ok(__t.bxConnect(bad.id, r.id), "2560 into a 5120-element reshape is refused");
  __t.bxConfigure(bad.id, { dout: 5120 });
  eql(__t.bxConnect(bad.id, r.id), null, "…5120 is fine");
  // junk keeps the shape; clearing the field turns reshape back into a pass-through
  __t.bxConfigure(r.id, { dims: "junk" });
  eql(__t.bxReshapeSize(r), 5120, "junk leaves the shape alone");
  __t.bxConfigure(r.id, { dims: "" });
  eql(__t.bxReshapeSize(r), null, "clearing it drops the shape");
  __t.bxConfigure(r.id, { din: "" });
  eql(__t.bxPortSizes(r, "in"), [5120], "…back to a pass-through, so the wire's size fills it again");
  // one number is a shape of one dim, and any separator parses
  __t.bxConfigure(r.id, { dims: "64 x 64" });
  eql(__t.bxDims(r.dims), [64, 64], "spaces and x both parse");
  eql(__t.bxReshapeSize(r), 4096, "…4096 elements");
  eql(__t.bxShape(r), "4096 \u2192 64\u00d764", "…shown normalised");
  // …and it rides in the JSON
  const json = JSON.parse(JSON.stringify(__t.bxExportCanvas()));
  eql(json.nodes.find(n => n.id === r.id).dims, "64x64", "the shape is in the JSON");
  eql(__t.bxImportJSON(JSON.stringify(json), {}).ok, true, "…and loads back");
  eql(__t.bxNode(r.id).dims, "64x64", "…with its shape");
  [src, r, out, bad].forEach(n => __t.bxRemove(n.id));
});

check("the atomic boxes: norm stays a kind, the op atoms are all there", () => {
  const src = __t.bxAdd("layer", 0, 0, {});
  __t.bxConfigure(src.id, { din: 4096, dout: 5120 });
  const norm = __t.bxAdd("norm", 200, 0, {});
  ok(/^norm \d+$/.test(norm.name), "a new norm is named like: " + norm.name);
  eql(__t.bxShape(norm), "rms", "it starts on rms");
  __t.bxConfigure(norm.id, { eps: "1e-6" });
  eql(__t.bxShape(norm), "rms \u00b7 1e-6", "…and shows its epsilon");
  __t.bxConfigure(norm.id, { norm: "layer" });
  eql(__t.bxShape(norm), "layer \u00b7 1e-6", "layer norm is the other choice");
  eql(__t.bxPortDefs(norm, "in").length, 1, "one wire in");
  eql(__t.bxConnect(src.id, norm.id), null, "a sized box wires in");
  eql(__t.bxOutSize(norm), 5120, "the size passes straight through the norm");
  __t.bxMenuBuild("box", norm.id, {});
  const nmenu2 = getEl("bxmenu");
  openCat(nmenu2, "this box");
  eql(catItems(nmenu2), ["rename this box", "copy this box", "set the norm"],
    "a norm box's menu: this box first");
  openCat(nmenu2, "weights");
  eql(catItems(nmenu2), ["set the dtype", "\u25cf load on the gpu \u2014 video memory",
                         "\u00a0\u00a0 load on the system \u2014 this machine's ram",
                         "load external weights …"], "…then its weights");
  __t.bxMenuClose();
  // the operation atoms a model needs
  for (const op of ["matmul", "concat", "rope", "scale", "topk", "conv1d", "scan"]) {
    ok(__t.BX_OPS.indexOf(op) >= 0, "the operation list has `" + op + "`");
  }
  const cc = __t.bxAdd("op", 0, 620, {});
  __t.bxConfigure(cc.id, { op: "concat" });
  eql([__t.bxPortDefs(cc, "in").length, __t.bxPortDefs(cc, "out").length], [2, 1],
    "a concat takes one wire per part and sends one");
  const cj = JSON.parse(JSON.stringify(__t.bxExportCanvas()));
  const ccRec = cj.nodes.find(n => n.id === cc.id);
  ccRec.din = 5120; ccRec.dout = 10240;                  // as a GGUF builder would stamp it
  eql(__t.bxImportJSON(JSON.stringify(cj), {}).ok, true, "…and its widths ride in the JSON");
  eql(__t.bxPortSizes(__t.bxNode(cc.id), "out"), [10240], "the concat's out reads its own width");
  const sc = __t.bxAdd("op", 0, 500, {});
  const src2 = __t.bxAdd("layer", 200, 500, {});
  __t.bxConfigure(sc.id, { op: "scale" });
  __t.bxConfigure(src2.id, { din: 64, dout: 512 });
  eql([__t.bxPortDefs(sc, "in").length, __t.bxPortDefs(sc, "out").length], [1, 1],
    "a scale takes one wire and sends one");
  eql(__t.bxConnect(src2.id, sc.id), null, "a sized box feeds it");
  eql(__t.bxOutSize(sc), 512, "…and it passes the size through (it is a bound scalar)");
  [sc, src2].forEach(n => __t.bxRemove(n.id));
  const mm = __t.bxAdd("op", 0, 300, {});
  __t.bxConfigure(mm.id, { op: "matmul" });
  eql(__t.bxPortDefs(mm, "in").length, 2, "a matmul takes two wires");
  eql(__t.bxPortDefs(mm, "out").length, 1, "…and sends one");
  const big = __t.bxAdd("layer", 200, 300, {});
  __t.bxConfigure(big.id, { din: 64, dout: 5120 });
  const small = __t.bxAdd("layer", 200, 420, {});
  __t.bxConfigure(small.id, { din: 64, dout: 1024 });
  eql(__t.bxConnect(big.id, mm.id, 0, 0), null, "a 5120 source into the contraction");
  eql(__t.bxConnect(small.id, mm.id, 0, 1), null, "…and a 1024 one on the other dot");
  eql(__t.bxWireState(edgeOf(big.id, mm.id)), "match", "a contraction does not judge shapes");
  eql(__t.bxWireState(edgeOf(small.id, mm.id, 0, 1)), "match", "…neither dot is called wrong");
  [src, norm, mm, big, small].forEach(n => __t.bxRemove(n.id));
});

check("attention, state space and MoE are nested boxes full of atomic boxes", () => {
  const keep = __t.BX.nodes.length;
  const attn = __t.bxAddTemplate("attn", 0, 0, "attention 1");
  eql(attn.kind, "custom", "an attention box is an ordinary nested box");
  eql([__t.bxPortDefs(attn, "in").length, __t.bxPortDefs(attn, "out").length], [1, 1],
    "one wire in, one out — the same dots any nested box has");
  const kinds = attn.inner.nodes.map(n => n.kind);
  for (const k of ["input", "output", "layer", "act", "op"]) {
    ok(kinds.indexOf(k) >= 0, "its inside is made of atomic boxes: " + kinds.join(","));
  }
  const names = attn.inner.nodes.map(n => n.name).join(",");
  ok(/q proj/.test(names) && /k proj/.test(names) && /v proj/.test(names) && /out proj/.test(names),
    "the projections are FC boxes: " + names);
  const ops = attn.inner.nodes.filter(n => n.kind === "op").map(n => __t.bxOpName(n));
  ok(ops.indexOf("matmul") >= 0 && ops.indexOf("rope") >= 0, "the contractions and rope are ops: " + ops.join(","));
  ok(attn.inner.nodes.some(n => n.kind === "act" && n.act === "softmax"), "softmax is an activation");
  // it opens like any nested box
  ok(__t.bxEnter(attn.id), "go inside it");
  eql(__t.BX.nodes.length, attn.inner.nodes.length, "its inside is the canvas now");
  const bad = __t.BX.edges.filter(e => __t.bxWireState(e) === "bad").length;
  eql(bad, 0, "nothing inside reads as mismatched");
  __t.bxLeave();
  // nothing inside a template may float: every box is fed and feeds something
  const orphans = (doc) => {
    const ins = new Set(doc.edges.map(e => e.to)), outs = new Set(doc.edges.map(e => e.from));
    const loose = [];
    for (const n of doc.nodes) {
      if (n.kind !== "input" && !ins.has(n.id)) loose.push(n.name + " has no input");
      if (n.kind !== "output" && !outs.has(n.id)) loose.push(n.name + " has no output");
    }
    return loose;
  };
  eql(orphans(attn.inner), [], "every atom in an attention is wired");
  // the same for state space and MoE
  const ssm = __t.bxAddTemplate("ssm", 600, 0, "state space 1");
  const sops = ssm.inner.nodes.filter(n => n.kind === "op").map(n => __t.bxOpName(n));
  ok(sops.indexOf("scan") >= 0 && sops.indexOf("conv1d") >= 0, "a state space has conv + scan atoms: " + sops.join(","));
  const moe = __t.bxAddTemplate("moe", 1200, 0, "mixture 1");
  const mnames = moe.inner.nodes.map(n => n.name).join(",");
  ok(/router/.test(mnames) && /expert/.test(mnames), "a MoE has a router and an expert path: " + mnames);
  ok(moe.inner.nodes.filter(n => n.kind === "op").map(n => __t.bxOpName(n)).indexOf("topk") >= 0,
    "…with a top-k pick");
  eql(orphans(ssm.inner), [], "every atom in a state space is wired");
  eql(orphans(moe.inner), [], "every atom in a MoE is wired");
  eql(__t.BX.nodes.length, keep + 3, "three boxes were added");
  // a design from before the change still loads: the box is rebuilt as nested
  const legacy = { type: "canvas", version: 1, nodes: [
      { id: "L1", kind: "input", name: "in", x: 0, y: 0, dout: 5120 },
      { id: "L2", kind: "attn", name: "attention old", x: 300, y: 0, heads: 24, kv: 4,
        t: "blk.0.attn_q.weight, blk.0.attn_k.weight, blk.0.attn_output.weight, blk.0.attn_gate.weight" },
      { id: "L3", kind: "output", name: "out", x: 600, y: 0, din: 5120 }],
    edges: [{ from: "L1", to: "L2" }, { from: "L2", to: "L3" }] };
  const res = __t.bxImportJSON(JSON.stringify(legacy), {});
  eql(res.ok, true, "an old design with an attention box loads");
  const l2 = __t.bxNode("L2");
  eql(l2.kind, "custom", "…the old box is a nested box now");
  eql(l2.name, "attention old", "…keeping its name");
  const lk = l2.inner.nodes.map(n => n.name + ":" + (n.t || "")).join(" | ");
  ok(/q proj:blk\.0\.attn_q\.weight/.test(lk) && /out proj:blk\.0\.attn_output\.weight/.test(lk),
    "…its tensors now bound to the atoms: " + lk);
  const gate = l2.inner.nodes.find(n => n.name === "gate proj");
  ok(gate && (gate.t || "") === "blk.0.attn_gate.weight", "the legacy gate projection comes back");
  ok(l2.inner.edges.some(e => e.to === gate.id), "…fed from the attention input, not left floating");
  eql(orphans(l2.inner), [], "no floating atoms in the rebuilt attention");
  eql(__t.bxWireState(edgeOf("L1", "L2")), "match", "the outer wires settle");
  ["L1", "L2", "L3"].forEach(id => __t.bxRemove(id));
  [attn, ssm, moe].forEach(n => __t.bxRemove(n.id));
});

check("dtype and the VRAM estimate", () => {
  eql([__t.bxBytes("f16"), __t.bxBytes("f32")], [2, 4], "bytes per weight: the simple ones");
  // a bound weight block shows the dtype its file actually stores
  const lb = __t.bxAdd("layer", 0, 0, {});
  __t.bxConfigure(lb.id, { din: 64, dout: 128, dtype: "u8" });
  const lower = Array.from(__t.bxCard(lb).querySelectorAll(".lower"))
    .map(e => e.textContent).join("");
  __t.bxRemove(lb.id);
  ok(lower.indexOf("u8") >= 0, "the block's own line carries its dtype: " + lower);
  eql([__t.bxBytes("q4_k"), __t.bxBytes("iq4_xs")], [0.5625, 0.515625],
    "…and the k-quants carry their block overhead");
  eql(__t.bxBytes("banana"), 2, "an unknown dtype estimates as 2 bytes");
  eql([__t.bxBytes("f64"), __t.bxBytes("u8"), __t.bxBytes("f8_e4m3"),
       __t.bxBytes("bool")], [8, 1, 1, 1],
    "the torch family carries its own byte sizes");
  eql(__t.BX_DTYPE_GROUPS.map(g => g[0]), ["torch", "gguf"],
    "the picker's dtypes are grouped torch then gguf");
  eql([__t.bxTensorDtype("F8_E4M3"), __t.bxTensorDtype("U8"), __t.bxTensorDtype("I32")],
    ["f8_e4m3", "u8", "i32"], "safetensors' own names map to the picker");
  // a small network with known sizes
  const before = __t.bxEstimate().params;
  const bd0 = __t.bxEstimate();                     // dtypes that were already there
  const i = __t.bxAdd("input", 0, 0, {});
  const fc = __t.bxAdd("layer", 200, 0, {});
  const nm = __t.bxAdd("norm", 400, 0, {});
  const o = __t.bxAdd("output", 600, 0, {});
  __t.bxConfigure(i.id, { dout: 64 });
  __t.bxConfigure(fc.id, { din: 64, dout: 128 });
  eql(fc.dtype, "f16", "an FC starts on f16");
  eql(nm.dtype, "f32", "a norm starts on f32");
  __t.bxConnect(i.id, fc.id);
  __t.bxConnect(fc.id, nm.id);
  __t.bxConnect(nm.id, o.id);
  const est = __t.bxEstimate();
  eql(est.params - before, 64 * 128 + 128, "params: the FC's in × out, plus one weight per norm channel");
  eql(est.weights, 64 * 128 * 2 + 128 * 4, "bytes: f16 for the FC, f32 for the norm");
  eql([(est.byDtypeParams.f16 || 0) - (bd0.byDtypeParams.f16 || 0),
       (est.byDtypeParams.f32 || 0) - (bd0.byDtypeParams.f32 || 0)], [64 * 128, 128],
    "the dtype breakdown reads every box's params, f16 and f32 apart");
  eql([(est.byDtype.f16 || 0) - (bd0.byDtype.f16 || 0),
       (est.byDtype.f32 || 0) - (bd0.byDtype.f32 || 0)], [64 * 128 * 2, 128 * 4],
    "…each with its own byte count");
  __t.bxConfigure(fc.id, { dtype: "q4_k" });
  eql(__t.bxEstimate().weights, 64 * 128 * 0.5625 + 128 * 4, "a dtype change moves the estimate");
  __t.bxConfigure(fc.id, { dtype: "f32" });
  eql(__t.bxEstimate().weights, 64 * 128 * 4 + 128 * 4, "…and f32 counts four bytes");
  __t.bxConfigure(fc.id, { dtype: "banana" });
  eql(fc.dtype, "f32", "junk keeps the dtype that was there");
  __t.bxConfigure(fc.id, { dtype: "f16" });
  // a box from a GGUF carries its exact numbers
  const sc = __t.bxAdd("op", 200, 300, {});
  __t.bxConfigure(sc.id, { op: "scale" });
  __t.bxConfigure(sc.id, { t: "blk.0.layer_output_scale.weight" });
  eql(__t.bxEstimate().params - before, 64 * 128 + 128 + 1, "a bound scalar counts one weight");
  // nested boxes add their whole inside up
  const nest = __t.bxAdd("custom", 600, 0, {});
  ok(__t.bxEnter(nest.id), "inside");
  const deep = __t.bxAdd("layer", 0, 0, {});
  __t.bxConfigure(deep.id, { din: 32, dout: 32, dtype: "f16" });
  __t.bxLeave();
  eql(__t.bxEstimate().params - before, 64 * 128 + 128 + 1 + 32 * 32,
    "a nested box weighs whatever its inside weighs");
  // boxes with no sizes are reported, never guessed
  const blank = __t.bxAdd("layer", 200, 500, {});
  ok(__t.bxEstimate().unknown >= 1, "a box without sizes is reported as unconfigured");
  __t.bxRemove(blank.id);
  // "estimate all as" and explicit dtype choices both re-estimate
  const exact = __t.bxAdd("layer", 200, 700, {});
  __t.bxConfigure(exact.id, { din: 100, dout: 100 });
  exact.params = 10000; exact.wbytes = 10000; exact.dtype = "iq4_xs";   // as a GGUF would stamp it
  const e0 = __t.bxEstimate().weights;
  __t.BX.model = { allDtype: "f16" };
  const e1 = __t.bxEstimate().weights;
  __t.bxConfigure(exact.id, { dtype: "f32" });
  eql(exact.wbytes, null, "an explicit dtype drops the file's exact byte count");
  // a shared weight (tied head, an output scale applied twice) weighs nothing itself
  const sharedBox = __t.bxAdd("layer", 600, 700, {});
  __t.bxConfigure(sharedBox.id, { din: 100, dout: 100 });
  sharedBox.params = 0; sharedBox.wbytes = 0; sharedBox.t = "token_embd.weight"; sharedBox.shared = true;
  const withShared = __t.bxEstimate().params;
  __t.bxRemove(sharedBox.id);
  eql(withShared, __t.bxEstimate().params, "a shared box adds no params of its own");
  const json2 = JSON.parse(JSON.stringify(__t.bxExportCanvas()));
  ok(true, "…");
  __t.BX.model = null;
  ok(e1 > e0, "estimating everything as f16 costs more than the quantized file did");
  // the model settings drive the KV cache and the training figure
  __t.BX.model = { layers: 3, kv_heads: 2, head_dim: 64, ctx: 1024, batch: 1,
                   opt: "adamw", master: false };
  const st = __t.bxEstimate();
  const run = __t.bxEstimateRuns(st);
  eql(run.kv, 2 * 3 * 2 * 64 * 1024 * 2, "the KV cache follows layers · kv heads · head dim · ctx");
  // only attention layers cache K/V, and the run context defaults to a sane 8k
  __t.BX.model.attn_layers = 1;
  eql(__t.bxEstimateRuns(st).kv, 2 * 1 * 2 * 64 * 1024 * 2,
    "a hybrid model caches only its attention layers");
  __t.BX.model.attn_layers = null;
  __t.BX.model.ctx = 262144;
  const big = __t.bxEstimateRuns(st);
  eql([big.kvCtx, big.kv], [8192, 2 * 3 * 2 * 64 * 8192 * 2],
    "a model that allows 256k runs at a sane context by default");
  eql(big.kvAtMax, 2 * 3 * 2 * 64 * 262144 * 2, "…and the ceiling's KV is reported alongside");
  __t.BX.model.kv_ctx = 4096;
  eql(__t.bxEstimateRuns(st).kv, 2 * 3 * 2 * 64 * 4096 * 2,
    "…while the context knob moves the figure");
  __t.BX.model.kv_ctx = null;
  __t.BX.model.ctx = 1024;
  eql(run.train, st.params * 12, "AdamW is grads + two fp32 moments per parameter");
  // frozen weights: counted in the model, skipped in the training bill
  nm.frozen = true;
  const fz = __t.bxEstimate();
  const fzRun = __t.bxEstimateRuns(fz);
  eql([fz.frozenParams, fz.frozenWeights], [128, 512],
    "a frozen box still weighs in the model (f32 × 128)");
  eql([fz.byDtypeFrozen.f32, fz.byDtypeFrozen.f16 || 0], [128, 0],
    "…and each dtype knows how much of it is frozen");
  eql([fzRun.train, fzRun.frozenTrain], [(st.params - 128) * 12, 128 * 12],
    "…while AdamW is priced over the trainable part and the skipped bill is reported");
  nm.frozen = false;
  __t.BX.model.master = true;
  const f32p = st.byDtypeParams.f32 || 0;
  const mr = __t.bxEstimateRuns(st);
  eql([mr.train, mr.masterBytes], [st.params * 12 + (st.params - f32p) * 4, (st.params - f32p) * 4],
    "…plus an fp32 master copy — but not for the weights already f32 (" + f32p + " params)");
  __t.BX.model.master = false;
  __t.BX.model.opt = "none";
  eql(__t.bxEstimateRuns(st).train, null, "…and no training figure at all when training is off");
  __t.BX.model.master = true;
  eql(__t.bxEstimateRuns(st).train, st.params * 4 + (st.params - f32p) * 4,
    "…while master weights alone still weigh (grads + master)");
  __t.BX.model = null;
  // the panel lives in the VRAM menu and wears the numbers
  __t.BX.model = { layers: 3, kv_heads: 2, head_dim: 64, ctx: 1024, batch: 1,
                   opt: "adamw", master: true };
  nm.frozen = true;
  __t.bxRenderPalette();
  const panel = getEl("bxvram");
  ok(panel, "the VRAM menu carries the estimate panel");
  __t.bxRenderEstimate();
  const txt = (panel.children || []).map(c => c.textContent || "").join(" | ");
  ok(txt.indexOf("weights") >= 0 && /(GiB|MiB|KiB| B)/.test(txt),
    "…with the weight figure: " + txt.slice(0, 90));
  ok(txt.indexOf("frozen") >= 0 && txt.indexOf("training skipped") >= 0 &&
     txt.indexOf("to run") >= 0 && txt.indexOf("to train") >= 0,
    "…and the frozen accounting with both totals: " + txt.slice(0, 170));
  nm.frozen = false;
  __t.BX.model = null;
  // …and it rides in the JSON
  __t.BX.model = { arch: "t", layers: 2, attn_layers: 1, kv_heads: 1, head_dim: 8,
                   ctx: 4096, kv_ctx: 2048, batch: 1, opt: "sgd", master: true };
  const json = JSON.parse(JSON.stringify(__t.bxExportCanvas()));
  eql(json.model.opt, "sgd", "the estimate knobs are exported");
  const rec = json.nodes.find(n => n.id === fc.id);
  eql(rec.dtype, "f16", "…and the box dtype with them");
  eql(__t.bxImportJSON(JSON.stringify(json), {}).ok, true, "the design loads back");
  eql(__t.BX.model.opt, "sgd", "…with its model block");
  eql([__t.BX.model.attn_layers, __t.BX.model.kv_ctx], [1, 2048],
    "…and how many layers cache K/V, at which context");
  eql(__t.bxNode(fc.id).dtype, "f16", "…and its dtypes");
  __t.BX.model = null;
  [i, fc, nm, o, sc, nest].forEach(n => __t.bxRemove(n.id));
});

check("error boxes: isolation, dead ends, bad wires, and nesting", () => {
  const errs = () => __t.bxErrorSet({ nodes: __t.BX.nodes, edges: __t.BX.edges });
  const i = __t.bxAdd("input", 0, 0, {});
  const fc = __t.bxAdd("layer", 200, 0, {});
  const o = __t.bxAdd("output", 400, 0, {});
  __t.bxConfigure(i.id, { dout: 512 });
  eql(__t.bxErrorSet({ nodes: [{ id: fc.id, kind: "layer", x: 0, y: 0 }], edges: [] }).has(fc.id), false,
    "a box with no wires at all is discarded, not an error");
  __t.bxConnect(i.id, fc.id);
  eql(errs().has(fc.id), false, "…and while it leads nowhere it stays discarded");
  eql(errs().has(i.id), false, "an input box that feeds something is fine");
  __t.bxConnect(fc.id, o.id);
  eql(errs().has(fc.id), false, "wiring it to an output puts it on the run — and it is whole");
  eql(errs().has(o.id), false, "…and the output is fed");
  // every input dot counts: a two-input operation with one dot empty is in error
  const i2 = __t.bxAdd("input", 0, 300, {});
  const mm = __t.bxAdd("op", 200, 300, {});
  const o2 = __t.bxAdd("output", 400, 300, {});
  __t.bxConfigure(i2.id, { dout: 512 });
  __t.bxConfigure(mm.id, { op: "add" });
  __t.bxConnect(i2.id, mm.id, 0, 0);
  eql(errs().has(mm.id), false, "off the run's path, an empty dot is fine: discarded");
  __t.bxConnect(i.id, mm.id, 0, 1);
  eql(errs().has(mm.id), false, "…still off it while nothing leaves the box");
  __t.bxConnect(mm.id, o2.id);
  eql(errs().has(mm.id), false, "on the run: both dots fed and one wire out, valid");
  __t.bxRemoveEdge(i.id, mm.id, 0, 1);
  eql(errs().has(mm.id), true, "…but an empty dot on the run is an error");
  __t.bxConnect(i.id, mm.id, 0, 1);
  eql(errs().has(mm.id), false, "…and feeding it clears it");
  // a mismatched wire puts both of its boxes in error
  const o3 = __t.bxAdd("output", 400, 500, {});
  __t.bxConnect(mm.id, o3.id);
  eql(errs().has(o3.id), false, "a fresh wire is fine");
  __t.bxConfigure(o3.id, { din: 999 });
  eql([errs().has(o3.id), errs().has(mm.id)], [true, true],
    "a resize that breaks the wire marks both ends as error boxes");
  __t.bxRemove(o3.id);
  // an error on the run's path deep inside marks every box above it
  const outer = __t.bxAdd("custom", 700, 0, {});
  ok(__t.bxEnter(outer.id), "into the outer nested box");
  const oin = __t.bxAdd("input", -200, 0, { dout: 512 });      // the block's own in dot
  const oadd = __t.bxAdd("op", 0, 0, { op: "add", din: 512 });  // two dots, one left empty
  const oout = __t.bxAdd("output", 200, 0, { din: 512 });      // …and its out dot
  __t.bxConnect(oin.id, oadd.id, 0, 0);
  __t.bxConnect(oadd.id, oout.id);
  __t.bxLeave();
  const link = __t.bxAdd("input", 500, 0, { dout: 512 });       // the block joins the run
  const sink = __t.bxAdd("output", 1000, 0, { din: 512 });
  __t.bxConnect(link.id, outer.id);
  __t.bxConnect(outer.id, sink.id);
  eql([__t.bxErrorSet(__t.bxNode(outer.id).inner).has(oadd.id), errs().has(outer.id)],
    [true, true], "an empty dot inside marks its box and every box above it");
  eql(errs().has(link.id), false, "…while clean boxes stay clean");
  // the card wears it: the class and the alert mark
  const host = getEl("bxcanvas");
  ok(host.querySelectorAll(".dnode.err").length >= 1, "the block is drawn with the err class");
  ok(host.querySelectorAll(".dwarn").length >= 1, "…and carries its alert mark");
  // feed the empty dot from the inside: the marks clear
  ok(__t.bxEnter(outer.id), "back inside");
  __t.bxConnect(oin.id, oadd.id, 0, 1);
  eql(__t.bxErrorSet({ nodes: __t.BX.nodes, edges: __t.BX.edges }).has(oadd.id), false,
    "the inner box is fixed");
  __t.bxLeave();
  eql(errs().has(outer.id), false, "…and the block clears with it");
  __t.bxRemove(outer.id);
  __t.bxRemove(link.id);
  __t.bxRemove(sink.id);
  const clean = __t.bxErrorSet({ nodes: [{ id: "x1", kind: "input", x: 0, y: 0, dout: 8 },
                                         { id: "x2", kind: "output", x: 200, y: 0, din: 8 }],
                                 edges: [{ from: "x1", to: "x2", fromPort: 0, toPort: 0 }] }).size;
  eql(clean, 0, "a well-formed little network has no error boxes");
});

check("a GGUF-built canvas loads: the model is one dedicated nested box", () => {
  // the shape the server's builder emits (see tests/gguf_canvas_test.py)
  const built = { type: "canvas", version: 1,
    nodes: [
      { id: "g1", kind: "input", name: "tokens", x: 0, y: 0, dout: 1 },
      { id: "g2", kind: "custom", name: "Qwen-test", x: 320, y: 0, inner: { type: "nested",
        version: 1,
        nodes: [
          { id: "n1", kind: "input", name: "tokens", x: 0, y: 0, dout: 1 },
          { id: "n2", kind: "embed", name: "token embedding", x: 320, y: 0, vocab: 4096,
            din: 1, dout: 512, t: "token_embd.weight" },
          { id: "n3", kind: "custom", name: "blk 0 \u00b7 attn mlp", x: 0, y: 240,
            inner: { type: "nested", version: 1,
            nodes: [
              { id: "i1", kind: "input", name: "in", x: 0, y: 0, dout: 512 },
              { id: "i2", kind: "norm", name: "attn_norm", x: 260, y: 0, norm: "rms", eps: 1e-6,
                t: "blk.0.attn_norm.weight" },
              { id: "i3", kind: "custom", name: "attention", x: 520, y: 0, inner: { type: "nested",
                version: 1,
                nodes: [
                  { id: "a1", kind: "input", name: "in", x: 0, y: 0, dout: 512 },
                  { id: "a2", kind: "layer", name: "q proj", x: 200, y: -60, din: 512, dout: 512,
                    t: "blk.0.attn_q.weight" },
                  { id: "a3", kind: "op", name: "scores", x: 400, y: 0, op: "matmul" },
                  { id: "a4", kind: "act", name: "softmax", x: 600, y: 0, act: "softmax" },
                  { id: "a5", kind: "layer", name: "out proj", x: 800, y: 0, din: 512, dout: 512,
                    t: "blk.0.attn_output.weight" },
                  { id: "a6", kind: "output", name: "out", x: 1000, y: 0, din: 512 }],
                edges: [{ from: "a1", to: "a2" }, { from: "a2", to: "a3" }, { from: "a3", to: "a4" },
                        { from: "a4", to: "a5" }, { from: "a5", to: "a6" }] } },
              { id: "i4", kind: "op", name: "add", x: 820, y: 0, op: "add" },
              { id: "i5", kind: "output", name: "out", x: 1080, y: 0, din: 512 }],
            edges: [{ from: "i1", to: "i2" }, { from: "i2", to: "i3" },
                    { from: "i3", to: "i4", toPort: 0 }, { from: "i1", to: "i4", toPort: 1 },
                    { from: "i4", to: "i5" }] } },
          { id: "n4", kind: "layer", name: "lm head", x: 320, y: 440, din: 512, dout: 4096,
            t: "output.weight", params: 512 * 4096, wbytes: 512 * 4096 * 2, dtype: "f16" },
          { id: "n5", kind: "output", name: "logits", x: 640, y: 440, din: 4096 }],
        edges: [{ from: "n1", to: "n2" }, { from: "n2", to: "n3" }, { from: "n3", to: "n4" },
                { from: "n4", to: "n5" }] } },
      { id: "g3", kind: "output", name: "logits", x: 640, y: 0, din: 4096 }],
    edges: [{ from: "g1", to: "g2" }, { from: "g2", to: "g3" }] };
  const res = __t.bxImportJSON(JSON.stringify(built), {});
  eql(res.ok, true, "the built canvas loads");
  eql(res.kind, "canvas", "\u2026as a canvas");
  eql(__t.BX.nodes.length, 3, "three top-level boxes: tokens, the model, logits");
  const model = __t.BX.nodes.find(n => n.kind === "custom");
  eql(model.name, "Qwen-test", "the box wears the model's name");
  eql(__t.bxPortSizes(model, "in"), [1], "one id goes into it");
  eql(__t.bxPortSizes(model, "out"), [4096], "the vocab comes out");
  const bad = [];
  const walk = (edges) => edges.forEach(e => { if (__t.bxWireState(e) === "bad") bad.push(e); });
  const blk = model.inner.nodes.find(n => n.kind === "custom");
  walk(__t.BX.edges);
  walk(model.inner.edges);
  walk(blk.inner.edges);
  walk(blk.inner.nodes.find(n => n.name === "attention").inner.edges);
  eql(bad.length, 0, "no mismatched wires at any level");
  eql(__t.bxEstimate().params, 4096 * 512 * 2 + 512 * 512 * 2 + 512,
    "the estimate reads through the model box (the embedding, the head, the two 512s, the norm)");
  eql(blk.inner.nodes.length, 5, "the block keeps its five boxes");
  eql(blk.inner.edges.length, 5, "\u2026and its wires");
  const att = blk.inner.nodes.find(n => n.name === "attention");
  eql(att.inner.nodes.length, 6, "its attention box holds its own atomic network");
  [ "g1", "g2", "g3" ].forEach(id => __t.bxRemove(id));
});

check("a loaded model: a nested box with the model structured inside", () => {
  const modelBefore = __t.BX.model;
  __t.BX.model = { opt: "sgd", batch: 2 };           // the panel's own settings
  __t.BX.nodes.slice().forEach(x => __t.bxRemove(x.id));
  const n = __t.bxAdd("custom", 100, 100, {});
  eql(n.kind, "custom", "a plain nested box to load into");
  eql(__t.bxShape(n), "0 in · 0 out", "empty until a model lands in it");
  __t.bxMenuBuild("box", n.id, {});
  const gmenu = getEl("bxmenu");
  eql(gmenu.children.map(c => c.textContent),
    ["this box", "inside", "flow", "remove this box"],
    "a nested box: its categories");
  openCat(gmenu, "inside");
  eql(catItems(gmenu), ["go inside", "export this nested box as JSON",
                        "import into this nested box …"],
    "…and no loader in it — Import ▾ is the only door");
  __t.bxMenuClose();
  eql(__t.bxGGUFLoad(n.id, GGUF_CANVAS, "models/test-model.gguf"), true,
    "the model loads into the box, with the file it reads from");
  eql(n.name, "Test-Model", "…which takes the model's name");
  eql(n.file, "models/test-model.gguf", "…and remembers where its tensors live");
  eql(__t.bxJSONNode(n).file, "models/test-model.gguf", "…and the design carries it");
  eql(__t.bxShape(n), "1 in · 1 out", "…and wears the model's dots");
  eql(__t.BX.nodes.map(x => x.kind), ["custom", "input", "output"],
    "alone on the canvas, the load brings the model's boundaries too");
  eql(__t.BX.edges.length, 2, "…wired in and out");
  eql(__t.bxErrorSet().size, 0, "…so the design is whole");
  eql(__t.bxPortSizes(n, "out"), [128], "the vocab comes out");
  eql(__t.bxEstimate().params, 8192, "the estimate reads through it");
  eql([__t.BX.model.ctx, __t.BX.model.layers, __t.BX.model.opt, __t.BX.model.batch],
    [4096, 2, "sgd", 2], "the model's numbers land, the panel's own settings stay");
  __t.bxUndoStep();
  eql([__t.BX.nodes.length, __t.BX.nodes.map(x => x.kind)], [1, ["custom"]],
    "one undo takes the whole load back");
  eql([__t.bxNode(n.id).name, __t.bxShape(__t.bxNode(n.id))], ["nested 1", "0 in · 0 out"],
    "…the box goes back to empty");
  __t.bxRedoStep();
  eql([__t.BX.nodes.length, __t.bxNode(n.id).name, __t.bxShape(__t.bxNode(n.id))],
    [3, "Test-Model", "1 in · 1 out"], "…and redo lands it again");
  eql(__t.bxGGUFLoad(n.id, GGUF_CANVAS2), true,
    "loading another model re-crops the boundaries");
  eql([__t.BX.nodes.length, __t.bxPortSizes(__t.bxNode(n.id), "out"), __t.BX.edges.length],
    [4, [128, 128], 3], "…the new dot arrives wired, the old ones stay");
  eql(__t.bxErrorSet().size, 0, "…still whole");
  eql(__t.bxGGUFLoad(n.id, GGUF_CANVAS), true, "and loading the first model back");
  eql([__t.BX.nodes.length, __t.bxPortSizes(__t.bxNode(n.id), "out")], [3, [128]],
    "…takes the extra boundary away, no duplicate stays");
  const saved = JSON.parse(JSON.stringify(__t.bxExportCanvas()));
  const rec = saved.nodes.find(x => x.id === n.id);
  eql([rec.kind, rec.gguf, rec.name], ["custom", undefined, "Test-Model"],
    "the JSON is just the nested box and where it reads from — no GGUF flag");
  eql(__t.bxImportJSON(JSON.stringify(saved), {}).ok, true, "…and it loads back");
  eql([__t.bxNode(n.id).gguf, __t.bxShape(__t.bxNode(n.id))],
    [undefined, "1 in · 1 out"], "…still the same loaded box");
  __t.BX.nodes.slice().forEach(x => __t.bxRemove(x.id));
  __t.BX.model = modelBefore;
});

check("loading a model into the blank skeleton replaces it, wired", () => {
  __t.BX.nodes.slice().forEach(x => __t.bxRemove(x.id));
  __t.bxSeedBlank();                            // input wired straight into output
  const n = __t.bxAdd("custom", 400, 240, {});
  eql(__t.bxGGUFLoad(n.id, GGUF_CANVAS, "models/test-model.gguf"), true, "the model loads");
  eql(__t.BX.nodes.map(x => x.kind).sort(), ["custom", "input", "output"],
    "the skeleton goes, the model's own boundaries land");
  eql(__t.BX.edges.length, 2, "an input feeds it, it feeds an output");
  ok(__t.BX.edges.some(e => e.to === n.id) && __t.BX.edges.some(e => e.from === n.id),
    "\u2026wired both ways");
});

check("spec op attrs survive two levels of nesting", () => {
  const outer = __t.bxAdd("custom", 0, 0, {});
  outer.inner = { nodes: [{ id: "blk", kind: "custom", name: "blk", x: 0, y: 0, inner: {
    nodes: [{ id: "n1", kind: "input", name: "x", x: -200, y: 0, dout: 64 },
            { id: "rz", kind: "op", name: "z heads", x: 0, y: 0, op: "reshape",
              dims: "4x16", keep: 1, k_heads: 4, head_dim: 16, v_tiled: 1 },
            { id: "sm", kind: "act", name: "softmax", x: 100, y: 0, act: "softmax",
              gain: 0.0625, causal: true },
            { id: "n2", kind: "output", name: "y", x: 200, y: 0, din: 64 }],
    edges: [{ from: "n1", to: "rz" }, { from: "rz", to: "sm" }, { from: "sm", to: "n2" }] } }], edges: [] };
  const back = __t.bxJSONNodeFrom(__t.bxJSONNode(outer));
  const rz = back.inner.nodes[0].inner.nodes.find(n => n.id === "rz");
  ok(rz && rz.keep === 1 && rz.k_heads === 4 && rz.head_dim === 16 && rz.v_tiled === 1,
    "the scan attrs travel through nested levels: " + JSON.stringify(rz));
  const sm = back.inner.nodes[0].inner.nodes.find(n => n.id === "sm");
  ok(sm && sm.gain === 0.0625 && sm.causal === true,
    "an act's gain and causal flag survive: " + JSON.stringify(sm));
  __t.bxRemove(outer.id);
});

check("a reshape's inferred dim keeps its sign through the UI", () => {
  const q = __t.bxAdd("op", 0, 0, {});
  __t.bxConfigure(q.id, { op: "reshape", dims: "-1, 24, 2, 256" });
  eql(q.dims, "-1x24x2x256", "configure keeps the sign");
  ok(__t.bxReshapeSize(q) === null, "an inferred dim has no fixed count");
  const back = __t.bxJSONNodeFrom(__t.bxJSONNode(q));
  eql(back.dims, "-1x24x2x256", "\u2026and a JSON round-trip too: " + back.dims);
  eql(__t.bxDims(back.dims), [-1, 24, 2, 256], "\u2026parsed back as it was");
  __t.bxRemove(q.id);
});

check("a served text pipeline lands whole, prompt to de-tokenizer", () => {
  __t.BX.nodes.slice().forEach(x => __t.bxRemove(x.id));
  const box = __t.bxAdd("custom", 500, 300, {});
  const b0 = [box.x, box.y];
  const served = { type: "canvas", version: 1,
    nodes: [
      { id: "t1", kind: "tokenize", name: "tokenize", x: -400, y: 0, repo: "org/repo" },
      { id: "m1", kind: "custom", name: "Model", x: 0, y: 0,
        inner: { type: "nested", version: 1,
        nodes: [{ id: "i", kind: "input", name: "tokens", x: -200, y: 0, dout: 1 },
                { id: "l", kind: "layer", name: "fc", x: 0, y: 0, din: 1, dout: 4 },
                { id: "o", kind: "output", name: "logits", x: 200, y: 0, din: 4 }],
        edges: [{ from: "i", to: "l" }, { from: "l", to: "o" }] } },
      { id: "o1", kind: "output", name: "logits", x: 400, y: 0, din: 4 },
      { id: "d1", kind: "detokenize", name: "text", x: 400, y: 240, repo: "org/repo" }],
    edges: [{ from: "t1", to: "m1" }, { from: "m1", to: "o1" }, { from: "t1", to: "d1" }] };
  eql(__t.bxGGUFLoad(box.id, served, null), true, "the pipeline loads");
  eql(__t.BX.nodes.map(x => x.kind), ["tokenize", "custom", "output", "detokenize"],
    "the whole served design lands — no plain tokens input to redo: "
    + JSON.stringify(__t.BX.nodes.map(x => [x.id, x.kind, x.name])));
  const model = __t.BX.nodes.find(x => x.kind === "custom");
  eql([model.x, model.y], b0, "the model lands where the box was, its wire crop with it");
  eql(model.inner.nodes.find(x => x.id === "l").dout, 4, "the model's inside is intact");
  ok(__t.BX.edges.some(e => e.from === "t1" && e.to === model.id) &&
     __t.BX.edges.some(e => e.from === "t1" && e.to === "d1"),
    "…the tokenizer feeds the model and the de-tokenizer");
  const before = __t.BXRUN;
  __t.BXRUN = { local: { connected: true, agent: { name: "pc", specs: {} } },
                inputs: {}, texts: {}, last: null, loaded: null, loading: false,
                loadKey: null, msg: "", running: false, gen: 0, temp: 0 };
  const pane = document.createElement("div");
  __t.bxRunFeeds(pane);
  ok(pane.querySelectorAll(".bxbin").some(f => f.placeholder === "text to tokenize"),
    "the Run box asks for the prompt");
  ok(pane.querySelectorAll(".bxbin").some(f => f.type === "number"),
    "…and offers tokens and temperature");
  __t.BX.nodes.slice().forEach(x => __t.bxRemove(x.id));
  __t.bxSeedBlank();
  __t.BXRUN = before;
});

check("a model box that lost its file binds back by name", () => {
  const models = [{ kind: "gguf", name: "test-model.gguf", path: "models/test-model.gguf" },
                  { kind: "hf", name: "not-a-gguf", path: "hf/not-a-gguf" }];
  const mkbox = (name) => ({ id: "b", kind: "custom", name, x: 0, y: 0,
    inner: { nodes: [{ id: "t", kind: "layer", name: "w", din: 2, dout: 2,
                       t: "some.weight" }], edges: [] } });
  const hit = mkbox("Test-Model");
  const r1 = __t.bxBindModels([hit], models, null);
  eql([r1.bound.length, r1.missing.length, hit.file],
      [1, 0, "models/test-model.gguf"], "a name that names one model binds it");
  const miss = mkbox("Unknown");
  const r2 = __t.bxBindModels([miss], models, null);
  eql([r2.bound.length, r2.missing.length, miss.file || null], [0, 1, null],
    "an unknown name is reported, never guessed");
});

check("a run binds a model box that lost its file, then goes on", () => {
  __t.BX.nodes.slice().forEach(x => __t.bxRemove(x.id));
  const inp = __t.bxAdd("input", 0, 0, { dout: 1 });
  const box = __t.bxAdd("custom", 240, 0, {});
  const out = __t.bxAdd("output", 480, 0, { din: 128 });
  box.name = "Test-Model";
  box.file = null;
  box.inner = { nodes: [
      { id: "i2", kind: "input", name: "tokens", x: 0, y: 0, dout: 1 },
      { id: "e2", kind: "embed", name: "emb", x: 0, y: 80, vocab: 128, din: 1,
        dout: 128, t: "token_embd.weight" },
      { id: "o2", kind: "output", name: "out", x: 0, y: 160, din: 128 }],
    edges: [{ from: "i2", to: "e2" }, { from: "e2", to: "o2" }] };
  __t.bxConnect(inp.id, box.id);
  __t.bxConnect(box.id, out.id);
  __t.bxModels = [{ kind: "gguf", name: "test-model.gguf", path: "models/test-model.gguf" }];
  __t.BXRUN = { local: { connected: true, agent: { name: "pc", specs: {} } }, inputs: {},
                last: null, loaded: null, loading: false, loadKey: null, msg: "",
                running: false };
  inp.values = "1";                            // the box's own matrix feeds the model
  __t.bxtRunGo();
  eql(__t.bxNode(box.id).file, "models/test-model.gguf", "the file binds by name");
  ok(__t.BXRUN.running === true, "\u2026and the run goes on");
  __t.BXRUN.running = false;
  __t.bxModels = null;
});

check("copy and paste: a selection leaves as boxes and lands with fresh ids", () => {
  const a = __t.bxAdd("input", 0, 0, { dout: 8 });
  const b = __t.bxAdd("layer", 200, 0, { din: 8, dout: 4 });
  __t.bxConnect(a.id, b.id);
  __t.bxSelectBoxes([a.id, b.id]);
  eql(__t.bxCopySelection(), true, "the selection copies");
  eql([__t.bxCopied.type, __t.bxCopied.nodes.length, __t.bxCopied.edges.length],
    ["boxes", 2, 1], "\u2026as a little boxes design, the wire between them included");
  __t.bxMenuBuild("box", a.id, {});
  openCat(getEl("bxmenu"), "this box");
  ok(catItems(getEl("bxmenu")).indexOf("copy these 2 boxes") >= 0,
    "a box in a multi-selection copies the whole group");
  __t.bxMenuClose();
  const before = __t.BX.nodes.length, edges0 = __t.BX.edges.length;
  eql(__t.bxPasteBoxes(), true, "\u2026and pastes");
  eql([__t.BX.nodes.length, __t.BX.edges.length], [before + 2, edges0 + 1],
    "two boxes and their wire arrive");
  const pasted = __t.bxSelectedIds();
  eql(pasted.length, 2, "the pasted boxes are selected, ready to move");
  ok(pasted.every(id => id !== a.id && id !== b.id), "with fresh ids");
  const pa = __t.bxNode(pasted.find(id => __t.bxNode(id).kind === "input"));
  const pb = __t.bxNode(pasted.find(id => __t.bxNode(id).kind === "layer"));
  eql([pa.dout, pb.din, pb.dout], [8, 8, 4], "\u2026and the same sizes");
  ok(__t.BX.edges.some(e => e.from === pa.id && e.to === pb.id),
    "wired to each other, not to the originals");
  ok(pa.x !== a.x || pa.y !== a.y, "placed beside them, not stacked");
  __t.bxMenuBuild("canvas", null, { clientX: 200, clientY: 150 });
  const pmenu = getEl("bxmenu");
  pmenu.children.filter(c => String(c.className).split(/\s+/).includes("drop") &&
                             c.textContent === "the design")[0].click();
  ok(pmenu.children.map(c => c.textContent).indexOf("paste") >= 0,
    "the empty canvas offers paste while the clipboard holds boxes");
  __t.bxMenuClose();
  __t.bxUndoStep();
  eql([__t.BX.nodes.length, __t.BX.edges.length], [before, edges0],
    "one undo takes the whole paste back");
  __t.bxRedoStep();
  eql(__t.BX.nodes.length, before + 2, "\u2026and redo brings it back");
  // the real event handlers: a copy event fills the clipboard, paste lands it
  let clip = "";
  const fakeEv = (tag) => ({ target: { tagName: tag || "DIV" }, _p: false,
    clipboardData: { setData: (t, v) => { clip = v; }, getData: () => clip },
    preventDefault() { this._p = true; } });
  __t.bxSelectBoxes([a.id, b.id]);
  const cev = fakeEv();
  eql(__t.bxOnCopy(cev), true, "the copy event takes the selection");
  ok(cev._p && JSON.parse(clip).type === "boxes", "\u2026puts its boxes JSON on the clipboard");
  const evBefore = __t.BX.nodes.length;
  const pev = fakeEv();
  eql(__t.bxOnPaste(pev), true, "the paste event lands the clipboard");
  ok(pev._p && __t.BX.nodes.length === evBefore + 2, "\u2026two boxes richer");
  eql(__t.bxOnCopy(fakeEv("TEXTAREA")), false, "a field keeps its own copy");
  eql(__t.bxOnPaste(fakeEv("INPUT")), false, "\u2026and its own paste");
  ok(__t.bxTyping({ tagName: "TEXTAREA" }) && !__t.bxTyping({ tagName: "DIV" }),
    "the typing guard knows a field");
  __t.BX.nodes.slice().forEach(n => __t.bxRemove(n.id));
  const c = __t.bxAdd("custom", 500, 0, { name: "packet" });
  c.inner = { nodes: [{ id: "i1", kind: "input", name: "in", x: 0, y: 0, dout: 4 },
                      { id: "i2", kind: "output", name: "out", x: 300, y: 0, din: 4 }],
              edges: [{ from: "i1", to: "i2" }] };
  __t.bxSelectBoxes([c.id]);
  __t.bxCopySelection();
  __t.bxPasteBoxes();
  const packet = __t.bxNode(__t.bxSelectedIds()[0]);
  eql([packet.name, packet.inner.nodes.length, packet.inner.edges.length], ["packet", 2, 1],
    "a nested box travels with its own inside");
  eql(__t.bxPasteBoxes('{"type":"canvas","version":1,"nodes":[],"edges":[]}'), false,
    "a canvas is not pasted into a level");
  const nest = { type: "nested", version: 1, name: "incoming",
                 nodes: [{ id: "x1", kind: "input", name: "in", x: 0, y: 0, dout: 2 }],
                 edges: [] };
  eql(__t.bxPasteBoxes(JSON.stringify(nest)), true, "a pasted nested design");
  const incoming = __t.bxNode(__t.bxSelectedIds()[0]);
  eql([incoming.name, incoming.inner.nodes.length], ["incoming", 1],
    "\u2026lands as one box holding it");
  __t.BX.nodes.slice().forEach(n => __t.bxRemove(n.id));
});

check("deactivate & freeze: no data in, no gradients out — and it reaches inside", () => {
  const modelBefore = __t.BX.model;
  __t.BX.model = { opt: "adamw", master: true };
  const a = __t.bxAdd("input", 0, 0, { dout: 8 });
  const b = __t.bxAdd("layer", 200, 0, { din: 8, dout: 4 });
  const o = __t.bxAdd("output", 400, 0, { din: 4 });
  __t.bxConnect(a.id, b.id);
  __t.bxConnect(b.id, o.id);
  eql([__t.bxEstimate().params, __t.bxErrorSet().size], [32, 0],
    "the layer is in the model, and whole");
  // a multi-selection offers both toggles for the whole group
  __t.bxSelectBoxes([a.id, b.id]);
  __t.bxMenuBuild("box", b.id, {});
  openCat(getEl("bxmenu"), "flow");
  const groupItems = catItems(getEl("bxmenu"));
  ok(groupItems.indexOf("deactivate these 2 boxes") >= 0 &&
     groupItems.indexOf("freeze these 2 boxes") >= 0,
    "a multi-selection toggles as a group: " + groupItems.join(" / "));
  __t.bxMenuClose();
  __t.bxSelect(null);
  // freeze: the weight stays in the model, the training step skips it
  __t.bxMenuBuild("box", b.id, {});
  openCat(getEl("bxmenu"), "flow");
  getEl("bxmenu").children.find(c => c.textContent === "freeze this box").click();
  eql([!!b.frozen, __t.bxEstimate().params, __t.bxEstimate().frozenParams], [true, 32, 32],
    "frozen: still counted, flagged as untrainable");
  eql(__t.bxEstimateRuns(__t.bxEstimate()).train, 0, "…so nothing is left to train");
  __t.bxMenuBuild("box", b.id, {});
  openCat(getEl("bxmenu"), "flow");
  getEl("bxmenu").children.find(c => c.textContent === "unfreeze this box").click();
  eql(__t.bxEstimateRuns(__t.bxEstimate()).train, 512,
    "…and unfreezing brings the training cost back");
  // deactivate: no data flows, no errors raised, weights out of the model
  __t.bxMenuBuild("box", b.id, {});
  openCat(getEl("bxmenu"), "flow");
  getEl("bxmenu").children.find(c => c.textContent === "deactivate this box").click();
  eql(__t.BX.edges.map(e => __t.bxWireState(e)), ["off", "off"], "its wires carry nothing");
  eql(__t.bxErrorSet().size, 0, "…and nothing is in error");
  const w = __t.bxWarnSet();
  eql([w.has(o.id), w.has(b.id), w.has(a.id)], [true, false, false],
    "the box that consumed the dead output is warned — and only it");
  eql(getEl("bxcanvas").querySelectorAll(".doff").length, 2,
    "the off box's two dots wear the deactivated mark");
  eql(getEl("bxcanvas").querySelectorAll(".dstarved").length, 1,
    "…and its consumer's feed dot goes amber");
  eql([__t.bxEstimate().params, __t.bxEstimate().offParams], [0, 32],
    "an off box leaves the model (its 32 params are out)");
  b.din = 999;
  eql(__t.bxWireState(__t.BX.edges[0]), "off", "an off box takes any size without complaint");
  b.din = 8;
  // a nested box carries the flag to everything inside it
  const nest = __t.bxAdd("custom", 600, 0, { name: "block" });
  nest.inner = { nodes: [{ id: "i1", kind: "input", name: "in", x: 0, y: 0, dout: 4 },
                         { id: "i2", kind: "norm", name: "n", x: 200, y: 0, norm: "rms",
                           eps: 1e-6, t: "w", params: 8, wbytes: 32, dtype: "f32" },
                         { id: "i3", kind: "op", name: "add", x: 380, y: 0, op: "add",
                           din: 4, dout: 4 },
                         { id: "i4", kind: "output", name: "out", x: 560, y: 0, din: 4 }],
                 edges: [{ from: "i1", to: "i2" },
                         { from: "i2", to: "i3", toPort: 0 },
                         { from: "i3", to: "i4" }] };   // one dot of the add is empty
  const o2 = __t.bxAdd("output", 800, 0, { din: 4 });
  __t.bxConnect(b.id, nest.id);                         // the block joins the run
  __t.bxConnect(nest.id, o2.id);
  eql(__t.bxErrorSet().has(nest.id), true, "an unfinished inside marks the box");
  __t.bxMenuBuild("box", nest.id, {});
  openCat(getEl("bxmenu"), "flow");
  getEl("bxmenu").children.find(c => c.textContent === "deactivate this box").click();
  eql([__t.bxErrorSet().has(nest.id), __t.bxEstimate().offParams], [false, 40],
    "deactivating it switches the whole inside off, its 8 params too");
  eql(__t.bxEnter(nest.id), true, "go inside the off box");
  eql([__t.bxErrorSet().size, __t.bxEstimate().params], [0, 0],
    "everything in here is off: no error marks, nothing counted");
  __t.bxLeave();
  // a dead output inside a nested box warns its user — and bubbles up
  const inner2 = { nodes: [{ id: "j1", kind: "input", name: "in", x: 0, y: 0, dout: 8 },
                           { id: "j2", kind: "layer", name: "w", x: 200, y: 0, din: 8, dout: 8 },
                           { id: "j3", kind: "output", name: "out", x: 400, y: 0, din: 8 }],
                   edges: [{ from: "j1", to: "j2" }, { from: "j2", to: "j3" }] };
  const packet2 = __t.bxAdd("custom", 800, 0, { name: "packet2" });
  packet2.inner = inner2;
  const pin = __t.bxAdd("input", 600, 0, { dout: 8 });    // the packet joins the run
  const pout = __t.bxAdd("output", 1000, 0, { din: 8 });
  __t.bxConnect(pin.id, packet2.id);
  __t.bxConnect(packet2.id, pout.id);
  eql(__t.bxWarnSet().has(packet2.id), false, "a whole nested box starts unwarned");
  inner2.nodes[1].off = true;
  eql([__t.bxWarnSet(packet2.inner).has("j3"), __t.bxWarnSet().has(packet2.id),
       __t.bxErrorSet(packet2.inner).size], [true, true, 0],
    "the dead output warns its user, the box above it — with no error raised");
  inner2.nodes[1].off = false;
  // JSON keeps both flags
  const saved = JSON.parse(JSON.stringify(__t.bxExportCanvas()));
  const rec = saved.nodes.find(x => x.id === nest.id);
  eql([rec.off, rec.frozen === true], [true, false], "the JSON remembers deactivated, not frozen");
  eql(__t.bxImportJSON(JSON.stringify(saved), {}).ok, true, "…and loads back");
  eql(__t.bxNode(nest.id).off, true, "…still off");
  __t.BX.nodes.slice().forEach(n => __t.bxRemove(n.id));
  __t.BX.model = modelBefore;
});

check("an inactive output marks the block that holds it inactive too", () => {
  const src = __t.bxAdd("input", 0, 0, { dout: 64 });
  const nest = __t.bxAdd("custom", 200, 0, {});
  nest.inner = { nodes: [{ id: "i1", kind: "input", name: "in", x: 0, y: 0, dout: 64 },
                         { id: "l1", kind: "layer", name: "w", x: 200, y: 0, din: 64, dout: 64 },
                         { id: "o1", kind: "output", name: "out", x: 400, y: 0, din: 64 }],
                 edges: [{ from: "i1", to: "l1" }, { from: "l1", to: "o1" }] };
  const dst = __t.bxAdd("output", 600, 0, { din: 64 });
  __t.bxConnect(src.id, nest.id);
  __t.bxConnect(nest.id, dst.id);
  eql([__t.bxBoxOff(nest), __t.bxWarnSet().has(dst.id), __t.bxEstimate().params],
    [false, false, 4096], "a whole block starts active, its 64\u00d764 inside counted");
  nest.inner.nodes[2].off = true;            // the block's output boundary goes inactive
  __t.bxRender();
  eql([__t.bxInnerOutOff(nest), __t.bxBoxOff(nest), __t.bxWireState(__t.BX.edges[1])],
    [true, true, "off"],
    "\u2026the block that holds it counts as inactive and its wire carries nothing");
  eql([__t.bxWarnSet().has(dst.id), __t.bxErrorSet().size], [true, 0],
    "\u2026its user is warned, not errored");
  eql([__t.bxEstimate().params, __t.bxEstimate().offParams], [0, 4096],
    "\u2026and the inside leaves the model");
  nest.inner.nodes[2].off = false;
  __t.bxRender();
  eql([__t.bxBoxOff(nest), __t.bxWireState(__t.BX.edges[1]), __t.bxWarnSet().has(dst.id)],
    [false, "match", false], "\u2026and activating the output brings it back");
  // two output dots: one dead only kills that dot, not the block
  const two = __t.bxAdd("custom", 0, 300, {});
  two.inner = { nodes: [{ id: "s1", kind: "input", name: "in", x: 0, y: 0, dout: 8 },
                        { id: "s2", kind: "output", name: "logits", x: 300, y: 0, din: 8 },
                        { id: "s3", kind: "output", name: "mtp logits", x: 300, y: 200, din: 8 }],
                edges: [{ from: "s1", to: "s2" }, { from: "s1", to: "s3" }] };
  const dstA = __t.bxAdd("output", 600, 300, { din: 8 });
  const dstB = __t.bxAdd("output", 600, 500, { din: 8 });
  const feed2 = __t.bxAdd("input", -200, 300, { dout: 8 });   // the block joins the run
  __t.bxConnect(feed2.id, two.id);
  __t.bxConnect(two.id, dstA.id, 0, 0);
  __t.bxConnect(two.id, dstB.id, 1, 0);
  const wireTo = (id) => __t.BX.edges.find(e => e.to === id);
  two.inner.nodes[2].off = true;             // only the second output goes inactive
  __t.bxRender();
  eql([__t.bxOffHere(two), __t.bxInnerOutOff(two)], [false, false],
    "one dead output does not deactivate its block");
  eql([__t.bxWireState(wireTo(dstA.id)), __t.bxWireState(wireTo(dstB.id))], ["match", "off"],
    "\u2026only that dot's wire carries nothing");
  eql([__t.bxWarnSet().has(dstA.id), __t.bxWarnSet().has(dstB.id)], [false, true],
    "\u2026and only that dot's user is warned");
  two.inner.nodes[1].off = true;             // now every output is dead
  __t.bxRender();
  eql([__t.bxOffHere(two), __t.bxBoxOff(two)], [true, true],
    "\u2026and only when every output is dead does the block go inactive");
  // "activate this box" means activate: it clears the dead outputs that hold it off
  __t.bxMenuBuild("box", two.id, {});
  openCat(getEl("bxmenu"), "flow");
  const twoMenu = catItems(getEl("bxmenu"));
  ok(twoMenu.indexOf("activate this box") >= 0,
    "an inactive block offers activate: " + twoMenu.join(" / "));
  getEl("bxmenu").children.find(c => c.textContent === "activate this box").click();
  eql([__t.bxOffHere(two), __t.bxInnerOutOff(two),
       !!(two.inner.nodes[1].off || two.inner.nodes[2].off)], [false, false, false],
    "\u2026and activating it clears the dead outputs it was blocked by");
  two.inner.nodes[1].off = true;             // its own flag plus every output dead
  two.inner.nodes[2].off = true;
  two.off = true;
  __t.bxRender();
  __t.bxMenuBuild("box", two.id, {});
  openCat(getEl("bxmenu"), "flow");
  getEl("bxmenu").children.find(c => c.textContent === "activate this box").click();
  eql([__t.bxOffHere(two), !!two.off,
       !!(two.inner.nodes[1].off || two.inner.nodes[2].off)], [false, false, false],
    "\u2026even when the box itself was switched off too");
  [two, dstA, dstB].forEach(n => __t.bxRemove(n.id));
  [src, nest, dst].forEach(n => __t.bxRemove(n.id));
});

check("the nested box: go inside; its input/output boxes are its dots", () => {
  const c = __t.bxAdd("custom", 0, 0, {});
  ok(/^nested \d+$/.test(c.name), "a new one is named like: " + c.name);
  eql(__t.bxShape(c), "0 in · 0 out", "it starts with no ports at all");
  eql(__t.bxEnter(c.id), true, "…going inside");
  const i1 = __t.bxAdd("input", 0, 0, {}); i1.dout = 5120;
  const i2 = __t.bxAdd("input", 0, 200, {}); i2.dout = 256;
  const l = __t.bxAdd("layer", 200, 100, {});
  const o1 = __t.bxAdd("output", 400, 100, {});
  __t.bxConnect(i1.id, l.id);
  __t.bxConnect(l.id, o1.id);
  eql(__t.bxPortDefs(c, "in").length, 2, "two input boxes inside → two in-dots");
  eql(__t.bxPortDefs(c, "in").map(p => p.size), [5120, 256], "…top to bottom");
  eql(__t.bxPortDefs(c, "out").length, 1, "…and one out-dot");
  eql(__t.bxShape(c), "2 in · 1 out", "the card says so");
  eql(__t.bxLeave(), true, "back outside");
  ok(!__t.BX.nodes.some(n => n.id === l.id), "…the inner boxes are not on the outer canvas");
  ok(__t.BX.nodes.some(n => n.id === c.id), "…and the custom box is");
  const src = __t.bxAdd("layer", 0, 400, {});
  __t.bxConfigure(src.id, { dout: 999 });
  ok(__t.bxCheckLink(src.id, c.id, 0, 1).error, "999 into the 256 port is refused: " +
     __t.bxCheckLink(src.id, c.id, 0, 1).error);
  __t.bxConfigure(src.id, { dout: 256 });
  eql(__t.bxConnect(src.id, c.id, 0, 1), null, "…and fits the 256 port");
  const c2 = __t.bxAdd("custom", 600, 0, {});
  __t.bxEnter(c2.id);
  const ci = __t.bxAdd("input", 0, 0, {});            // its size is still ?
  __t.bxLeave();
  eql(__t.bxConnect(src.id, c2.id, 0, 0), null, "a wire into an unknown custom port");
  eql(ci.dout, 256, "…fills the input box *inside* the custom box");
  __t.bxMenuBuild("box", c.id, {});
  const cm = getEl("bxmenu");
  eql(cm.children.map(x => x.textContent),
    ["this box", "weights", "inside", "flow", "remove this box"],
    "a nested box's menu categories");
  openCat(cm, "inside");
  eql(catItems(cm), ["go inside", "export this nested box as JSON",
                     "import into this nested box …"],
    "…inside holds its network and JSON");
  openCat(cm, "weights");
  eql(catItems(cm), ["\u25cf load on the gpu \u2014 video memory",
                     "\u00a0\u00a0 load on the system \u2014 this machine's ram",
                     "load external weights …"], "…weights the placement and the files");
  __t.bxMenuClose();
  // leaving and coming back must keep the inside network — wires included
  __t.bxEnter(c.id);
  const before = __t.BX.edges.length;
  ok(before > 0, "the inside has wires: " + before);
  __t.bxLeave();
  __t.bxEnter(c.id);
  eql(__t.BX.edges.length, before, "…and they are all still there on re-entry");
  eql(__t.BX.edges.length, c.inner.edges.length, "…the box's own copy included");
  __t.bxLeave();
  [c, c2, src].forEach(n => __t.bxRemove(n.id));
});

check("dropping a box on a nested box moves it inside, wires and all", () => {
  __t.bxImportJSON(JSON.stringify({ type: "canvas", version: 1, nodes: [], edges: [] }), {});
  const i = __t.bxAdd("input", 0, 0, {});
  const fc = __t.bxAdd("layer", 240, 0, {});
  const o = __t.bxAdd("output", 480, 0, {});
  const nest = __t.bxAdd("custom", 240, 300, {});
  __t.bxConfigure(i.id, { dout: 512 });
  __t.bxConfigure(fc.id, { din: 512, dout: 512 });
  __t.bxConnect(i.id, fc.id);
  __t.bxConnect(fc.id, o.id);
  eql(__t.bxMoveInto(fc.id, nest.id), true, "the box moves into the nested box");
  eql(__t.bxNode(fc.id), null, "…it is gone from this level");
  ok(nest.inner.nodes.some(n => n.id === fc.id), "…and lives inside the nested box");
  eql(__t.BX.edges.map(e => [e.from, e.to]).length, 2, "the two outer wires are still there");
  ok(__t.BX.edges.some(e => e.from === i.id && e.to === nest.id), "the input now feeds the nested box");
  ok(__t.BX.edges.some(e => e.from === nest.id && e.to === o.id), "…and the nested box feeds the output");
  const inPorts = nest.inner.nodes.filter(n => n.kind === "input");
  const outPorts = nest.inner.nodes.filter(n => n.kind === "output");
  eql([inPorts.length, outPorts.length], [1, 1], "a port was grown on each side for the connection");
  eql(nest.inner.edges.length, 2, "two wires inside");
  ok(nest.inner.edges.some(e => e.from === inPorts[0].id && e.to === fc.id), "the input port feeds the box");
  ok(nest.inner.edges.some(e => e.from === fc.id && e.to === outPorts[0].id), "…and the box feeds the output port");
  eql(__t.BX.edges.concat(nest.inner.edges).filter(e => __t.bxWireState(e) === "bad").length, 0,
    "no wire went wrong in the move");
  eql(__t.bxOutSize(nest), 512, "the nested box sends the size through its new port");
  // a fan of wires out of one side shares one port; an empty input dot needs none
  const src = __t.bxAdd("input", 0, 600, {});
  const mm = __t.bxAdd("op", 240, 600, {});
  const d1 = __t.bxAdd("output", 480, 560, {});
  const d2 = __t.bxAdd("output", 480, 700, {});
  const nest2 = __t.bxAdd("custom", 240, 820, {});
  __t.bxConfigure(src.id, { dout: 64 });
  __t.bxConfigure(mm.id, { op: "add" });
  __t.bxConnect(src.id, mm.id, 0, 0);
  __t.bxConnect(mm.id, d1.id);
  __t.bxConnect(mm.id, d2.id);
  eql(__t.bxMoveInto(mm.id, nest2.id), true, "a two-input, two-output-fan box moves in too");
  eql([nest2.inner.nodes.filter(n => n.kind === "input").length,
       nest2.inner.nodes.filter(n => n.kind === "output").length], [1, 1],
    "one input port (its second dot was empty) and one output port for both wires");
  eql(__t.BX.edges.filter(e => e.from === nest2.id).length, 2, "both fan wires leave the same port");
  // never into itself, or into something it holds
  const A = __t.bxAdd("custom", 700, 1000, {});
  ok(__t.bxEnter(A.id), "inside A");
  const B = __t.bxAdd("custom", 0, 0, {});
  __t.bxLeave();
  eql(__t.bxMoveInto(A.id, B.id), false, "…but a box cannot move into something it holds");
  eql(__t.bxNode(A.id).id, A.id, "…and it stays where it is");
  [i, o, nest, src, d1, d2, nest2, A].forEach(n => __t.bxRemove(n.id));
});

check("moving a chain in one box at a time reuses the ports", () => {
  __t.bxImportJSON(JSON.stringify({ type: "canvas", version: 1, nodes: [], edges: [] }), {});
  const i = __t.bxAdd("input", 0, 0, {});
  const f1 = __t.bxAdd("layer", 240, 0, {});
  const f2 = __t.bxAdd("layer", 480, 0, {});
  const o = __t.bxAdd("output", 720, 0, {});
  const n = __t.bxAdd("custom", 240, 300, {});
  __t.bxConfigure(i.id, { dout: 512 });
  __t.bxConfigure(f1.id, { din: 512, dout: 512 });
  __t.bxConfigure(f2.id, { din: 512, dout: 512 });
  __t.bxConnect(i.id, f1.id);
  __t.bxConnect(f1.id, f2.id);
  __t.bxConnect(f2.id, o.id);
  const ports = (dir) => n.inner.nodes.filter(x => x.kind === (dir === "in" ? "input" : "output")).length;
  eql(__t.bxMoveInto(f2.id, n.id), true, "the box at the end of the chain moves in");
  eql([ports("in"), ports("out")], [1, 1], "…growing a port on each side");
  eql(__t.bxMoveInto(f1.id, n.id), true, "then the box in front of it moves in too");
  eql([ports("in"), ports("out")], [1, 1], "…and it rides the ports already there");
  eql(__t.BX.edges.filter(e => e.to === n.id).length, 1, "outside: one wire into the nested box…");
  eql(__t.BX.edges.filter(e => e.from === n.id).length, 1, "…and one out of it (no fan grew)");
  eql(n.inner.edges.length, 3, "inside: three wires: " + JSON.stringify(n.inner.edges.map(e => [e.from, e.to])));
  const inBox = n.inner.nodes.find(x => x.kind === "input");
  const outBox = n.inner.nodes.find(x => x.kind === "output");
  const wire = (a, b) => n.inner.edges.some(e => e.from === a.id && e.to === b.id);
  ok(wire(inBox, f1) && wire(f1, f2) && wire(f2, outBox), "the chain inside is in → f1 → f2 → out");
  eql(__t.BX.edges.concat(n.inner.edges).filter(e => __t.bxWireState(e) === "bad").length, 0,
    "no wire reads mismatched");
  eql(__t.bxErrorSet({ nodes: __t.BX.nodes, edges: __t.BX.edges }).size, 0, "no error boxes");
  eql(__t.bxOutSize(n), 512, "…and the nested box still sends 512");
  // one more box after it reuses the output side, never doubling it
  const f3 = __t.bxAdd("layer", 960, 0, {});
  __t.bxConfigure(f3.id, { din: 512, dout: 512 });
  __t.bxConnect(n.id, f3.id);
  __t.bxConnect(f3.id, o.id);
  eql(__t.bxMoveInto(f3.id, n.id), true, "a third box goes in as well");
  eql([ports("in"), ports("out")], [1, 1], "…still one port each");
  eql(n.inner.edges.length, 4, "…and the inside is in → f1 → f2 → f3 → out");
  eql(__t.bxErrorSet({ nodes: __t.BX.nodes, edges: __t.BX.edges }).size, 0, "still no error boxes");
  [i, o, n].forEach(x => __t.bxRemove(x.id));
});

check("Ctrl+Z walks back through the actions, and redo walks forward", () => {
  __t.bxImportJSON(JSON.stringify({ type: "canvas", version: 1, nodes: [], edges: [] }), {});
  const start = __t.bxUndoDepth().undo;
  eql(__t.bxCommit(), undefined, "committing is quiet");
  eql(__t.bxUndoDepth().undo, start, "…and an unchanged canvas adds no step");
  const a = __t.bxAdd("input", 0, 0, {});
  const b = __t.bxAdd("layer", 240, 0, {});
  __t.bxConfigure(b.id, { din: 64, dout: 64 });
  __t.bxConnect(a.id, b.id);
  eql(__t.bxUndoDepth().redo, 0, "a fresh action leaves the redo queue empty");
  eql(__t.bxUndoStep(), true, "Ctrl+Z once");
  eql(__t.BX.edges.length, 0, "…undoes the wire");
  __t.bxUndoStep();
  eql([__t.bxNode(b.id).din, __t.bxNode(b.id).dout], [null, null], "…undoes the sizes");
  __t.bxUndoStep();
  eql(__t.bxNode(b.id), null, "…undoes the second box");
  __t.bxUndoStep();
  eql(__t.bxNode(a.id), null, "…and the first");
  eql(__t.BX.nodes.length, 0, "four undos brought the canvas back to empty");
  eql(__t.bxUndoDepth().redo, 4, "the four steps wait to be redone");
  eql(__t.bxRedoStep(), true, "Ctrl+Shift+Z once");
  eql(__t.bxNode(a.id).id, a.id, "…puts the first box back");
  __t.bxRedoStep(); __t.bxRedoStep(); __t.bxRedoStep();
  eql(__t.bxNode(b.id).din, 64, "redoing all the way restores the sizes: " + __t.bxNode(b.id).din);
  ok(__t.BX.edges.some(e => e.from === a.id && e.to === b.id), "…and the wire");
  eql(__t.bxUndoDepth().redo, 0, "nothing left to redo");
  // a new action drops whatever was waiting to be redone
  __t.bxUndoStep();
  eql(__t.bxUndoDepth().redo, 1, "one step parked in redo");
  const fresh = __t.bxAdd("layer", 500, 300, {});
  eql(__t.bxUndoDepth().redo, 0, "a new action clears the redo queue");
  __t.bxUndoStep();
  eql(__t.bxNode(fresh.id), null, "…and that action undoes like any other");
  // a change made inside a nested box undoes back inside that box
  const nest = __t.bxAdd("custom", 600, 0, {});
  ok(__t.bxEnter(nest.id), "go inside");
  const x = __t.bxAdd("layer", 0, 0, {});
  __t.bxLeave();
  eql(__t.BX.up, null, "back at the top level");
  __t.bxUndoStep();
  eql(__t.BX.up, nest.id, "undo returned inside the box where it happened");
  eql(__t.bxNode(x.id), null, "…with the box gone");
  __t.bxRedoStep();
  eql(__t.bxNode(x.id).id, x.id, "redo brings it back");
  __t.bxLeave();
  // moving a box into a nested box is a single step
  const src = __t.bxAdd("input", 0, 400, {});
  const fc = __t.bxAdd("layer", 240, 400, {});
  __t.bxConnect(src.id, fc.id);
  const n2 = __t.bxAdd("custom", 480, 400, {});
  const parked = __t.bxUndoDepth().redo;
  __t.bxMoveInto(fc.id, n2.id);
  __t.bxUndoStep();
  eql(__t.bxUndoDepth().redo, parked + 1, "a move-into is one step, not several");
  eql(__t.bxNode(fc.id).id, fc.id, "undo brings the box back out");
  ok(__t.BX.edges.some(e => e.from === src.id && e.to === fc.id), "…with its wire");
  eql(__t.bxNode(n2.id).inner.nodes.filter(n => n.kind === "input" || n.kind === "output").length, 0,
    "…and the ports it grew are gone again");
  // undoing must not move the canvas
  __t.BX.view = { cx: 321, cy: 654, k: 1.75 };
  const vbox = __t.bxAdd("layer", 0, 0, {});
  __t.bxUndoStep();
  eql([__t.BX.view.cx, __t.BX.view.cy, __t.BX.view.k], [321, 654, 1.75],
    "undo leaves the camera exactly where it was");
  __t.bxRedoStep();
  eql([__t.BX.view.cx, __t.BX.view.cy, __t.BX.view.k], [321, 654, 1.75],
    "…and redo does too");
  __t.bxUndoStep();
  eql(__t.bxNode(vbox.id), null, "…while still undoing the action");
  [a, b, nest, src, fc, n2].forEach(n => __t.bxRemove(n.id));
});

check("breadboard: boxes snap to the grid, siblings show guides", () => {
  __t.bxImportJSON(JSON.stringify({ type: "canvas", version: 1, nodes: [], edges: [] }), {});
  // a fresh box is born on the grid, and the step is world-space
  const born = __t.bxAdd("layer", 95, 67, {});
  eql([born.x, born.y], [80, 80], "a near miss lands on the nearest crossing");
  eql(__t.bxSnapGrid(303), 320, "the step is 40 units, zoom-independent");
  const a = __t.bxAdd("layer", 320, 200, {});
  const b = __t.bxAdd("layer", 120, 440, {});
  eql([a.x, a.y, b.x, b.y], [320, 200, 120, 440], "multiples of the step stay put");
  // a near miss on both axes snaps to the grid, and shared lines get guides
  const s1 = __t.bxSnapMove(b, 305, 195);
  eql([s1.x, s1.y], [320, 200], "the grid snaps both axes");
  eql(s1.guides.map(g => g.dir).sort().join(","), "h,v", "…and both shared lines are guided");
  const v = s1.guides.find(g => g.dir === "v");
  eql(v.at, 320, "the column guide sits on the shared grid line");
  ok(v.a <= 200 && v.b >= 260, "…and spans both boxes: " + JSON.stringify([v.a, v.b]));
  // the grid is the only law: far boxes still land on it, with no guides
  const far = __t.bxSnapMove(b, 600, 520);
  eql([far.x, far.y], [600, 520], "600 and 520 are exact crossings");
  eql(far.guides.length, 0, "…and nothing else lines up");
  // the lattice is world-space: the same miss snaps the same way at any zoom
  __t.BX.view = { cx: 0, cy: 0, k: 0.5 };
  eql([__t.bxSnapMove(b, 305, 195).x, __t.bxSnapMove(b, 305, 195).y], [320, 200],
    "at 50% zoom the same miss snaps to the same crossing");
  __t.BX.view = { cx: 0, cy: 0, k: 1 };
  // the guides are drawn while dragging and gone once it ends
  const host = getEl("bxcanvas");
  __t.bxGuides = s1.guides;
  __t.bxRender();
  eql(host.querySelectorAll(".bxguide").length, 2, "two guide lines are drawn while dragging");
  __t.bxGuides = [];
  __t.bxRender();
  eql(host.querySelectorAll(".bxguide").length, 0, "…and none once the drag ends");
  [born, a, b].forEach(n => __t.bxRemove(n.id));
});

check("shift+drag grabs several boxes, and they move as one", () => {
  __t.bxImportJSON(JSON.stringify({ type: "canvas", version: 1, nodes: [], edges: [] }), {});
  const a = __t.bxAdd("layer", 1000, 1000, {});
  const b = __t.bxAdd("layer", 1400, 1300, {});
  const c = __t.bxAdd("layer", 1800, 1600, {});
  eql(__t.bxSelectedIds(), [], "nothing is selected to begin with");
  __t.bxMarqueeSelect({ x0: 950, y0: 950, x1: 1480, y1: 1420 }, false);
  eql(__t.bxSelectedIds().sort(), [a.id, b.id].sort(), "the band picks up the boxes it touches");
  eql([__t.bxSelBox(a.id), __t.bxSelBox(b.id), __t.bxSelBox(c.id)], [true, true, false],
    "…and leaves the rest alone");
  __t.bxToggleSelect(c.id);
  eql(__t.bxSelectedIds().length, 3, "shift+click adds a box to the group");
  __t.bxToggleSelect(c.id);
  eql(__t.bxSelectedIds().length, 2, "…and takes it out again");
  __t.bxMarqueeSelect({ x0: 1750, y0: 1550, x1: 2000, y1: 1750 }, false);
  eql(__t.bxSelectedIds(), [c.id], "a fresh band replaces the selection");
  __t.bxMarqueeSelect({ x0: 950, y0: 950, x1: 2000, y1: 1750 }, false);
  eql(__t.bxSelectedIds().length, 3, "all three are grabbed");
  // the group moves together, offsets intact — the grabbed box snaps to the grid
  __t.bxStartMove(__t.bxSelectedIds(), a.id, 0, 0);
  __t.bxMove({ clientX: 40, clientY: 20 });
  eql([a.x, a.y, b.x, b.y, c.x, c.y], [1040, 1040, 1440, 1360, 1840, 1640],
    "the primary snaps to the grid and the group keeps its offsets");
  __t.bxMove({ clientX: 80, clientY: 40 });
  eql([a.x, b.x, c.x], [1080, 1480, 1880], "…and again on the next move");
  __t.bxUp({});
  eql([a.x, a.y], [1080, 1040], "the drop keeps the group where it stands");
  // one undo takes the whole group move back
  const parked = __t.bxUndoDepth().redo;
  __t.bxUndoStep();
  eql([__t.bxNode(a.id).x, __t.bxNode(b.id).x, __t.bxNode(c.id).x], [1000, 1400, 1800],
    "one undo puts the whole group back");
  eql(__t.bxUndoDepth().redo, parked + 1, "…as a single step");
  ok(__t.bxSelectedIds().every(id => !!__t.bxNode(id)),
    "…and any selection it restores only names boxes that still exist");
  // Delete removes the group at once
  __t.bxSelectBoxes([a.id, b.id]);
  eql(__t.bxDeleteSelection(), true, "Delete takes the selection");
  eql([__t.bxNode(a.id), __t.bxNode(b.id)], [null, null], "…both boxes gone");
  eql(__t.bxNode(c.id).id, c.id, "…and the unselected one stays");
  __t.bxRemove(c.id);
});

check("the embedding box: an index in, a vector out — not a matmul", () => {
  const e = __t.bxAdd("embed", 0, 0, {});
  const before = __t.bxEstimate().params;
  ok(/^embedding \d+$/.test(e.name), "a new one is named like: " + e.name);
  eql([e.din, e.dout, e.vocab], [1, null, null], "it starts with one id in and nothing out");
  eql(__t.bxShape(e), "embedding", "…and says so on the card");
  __t.bxConfigure(e.id, { vocab: 32000, dout: 512 });
  eql(__t.bxShape(e), "32000 × 512", "the card shows the table it indexes");
  eql([__t.bxPortSizes(e, "in"), __t.bxPortSizes(e, "out")], [[1], [512]],
    "one id comes in, a 512-wide vector goes out");
  eql(__t.bxEstimate().params - before, 32000 * 512, "the table weighs vocab × out");
  eql(e.dtype, "f16", "it is a weight box, so it carries a dtype");
  __t.bxMenuBuild("box", e.id, {});
  const em = getEl("bxmenu");
  eql(em.children.map(c => c.textContent),
    ["this box", "weights", "flow", "remove this box"], "an embedding box's menu categories");
  openCat(em, "this box");
  eql(catItems(em), ["rename this box", "copy this box", "set the embedding"],
    "…this box holds the table");
  openCat(em, "weights");
  eql(catItems(em), ["set the dtype", "\u25cf load on the gpu \u2014 video memory",
                     "\u00a0\u00a0 load on the system \u2014 this machine's ram",
                     "load external weights …"], "…weights the dtype and the binding");
  __t.bxMenuClose();
  const json = JSON.parse(JSON.stringify(__t.bxExportCanvas()));
  const rec = json.nodes.find(n => n.id === e.id);
  eql([rec.kind, rec.vocab, rec.din, rec.dout], ["embed", 32000, 1, 512], "the JSON records the table");
  eql(__t.bxImportJSON(JSON.stringify(json), {}).ok, true, "…and loads back");
  eql([__t.bxNode(e.id).vocab, __t.bxNode(e.id).dout], [32000, 512], "…with it intact");
  eql(__t.bxShape(__t.bxNode(e.id)), "32000 × 512", "…and the card still reads right");
  __t.bxRemove(e.id);
});

check("external weights: a file's tensor points a block at real weights", () => {
  ok(__t.bxWeightScore("q proj", "blk.0.attn_q.weight") > 0, "a name match is found");
  ok(__t.bxWeightScore("out proj", "blk.0.attn_q.weight") === 0, "…and a stranger is not");
  const l = __t.bxAdd("layer", 0, 0, {});
  const t = { name: "model.layers.0.fc.weight", shape: [3, 4], dtype: "F32" };
  __t.bxWeightsApply(l, "weights/x.safetensors", t);
  const bound = __t.bxNode(l.id);
  eql([bound.din, bound.dout], [4, 3], "the tensor's shape sizes the layer");
  eql([bound.file, bound.t], ["weights/x.safetensors", t.name],
    "…and the file + tensor are bound");
  eql(bound.dtype, "f32", "…and the dtype follows the tensor");
  __t.bxMenuBuild("box", l.id, {});
  openCat(getEl("bxmenu"), "weights");
  ok(catItems(getEl("bxmenu")).indexOf("load external weights …") >= 0,
    "the menu offers external weights");
  __t.bxMenuClose();
  const rec = __t.bxJSONNode(bound);
  eql([rec.file, rec.t], ["weights/x.safetensors", t.name], "…and the JSON records it");
  const n = __t.bxApplyManifest("weights/out.safetensors", [
    { id: l.id, key: "fc 1.weight" }, { id: l.id, key: "fc 1.bias" }]);
  eql(n, 1, "an export's manifest rebinds the block");
  eql([bound.file, bound.t], ["weights/out.safetensors", "fc 1.weight,fc 1.bias"],
    "…to the exported keys");
  __t.bxWeightsApply(bound, "weights/x.safetensors", { name: "b", shape: [3], dtype: "F32" });
  eql([bound.din, bound.dout], [4, 3], "a 1-D tensor leaves the sizes alone");
  __t.bxWeightsForm(l.id);
  ok(getEl("bxmenu").children.length > 0, "the weights form opens");
  __t.bxMenuClose();
  __t.bxRemove(l.id);
});

check("the Script box: python on the card, pass-through sizes, no weights", () => {
  const before = __t.bxEstimate().params;
  const s = __t.bxAdd("script", 300, 200, { script: "def run(x):\n    return x * 2" });
  eql(s.kind, "script", "a script box is its own kind");
  ok(/^script \d+$/.test(s.name), "named like: " + s.name);
  eql(__t.bxShape(s), "run(x)", "the card shows the function it runs");
  eql([__t.bxPortDefs(s, "in").length, __t.bxPortDefs(s, "out").length], [1, 1],
    "one dot each side");
  eql(__t.bxEstimate().params - before, 0, "it holds no weights");
  // the wire's size passes straight through it
  const src = __t.bxAdd("input", 100, 200, { dout: 8 });
  __t.bxConnect(src.id, s.id);
  eql(__t.bxPortSizes(s, "in"), [8], "a sized source fills its input");
  eql(__t.bxPortSizes(s, "out"), [8], "…which passes straight through");
  // its menu offers the editor, and the editor carries the code
  __t.bxMenuBuild("box", s.id, {});
  openCat(getEl("bxmenu"), "this box");
  ok(catItems(getEl("bxmenu")).indexOf("edit the script …") >= 0,
    "its menu offers the script");
  __t.bxMenuClose();
  __t.bxForm("script", null, s, "script");
  const wrap = getEl("bxmenu").children[0];
  const ta = (wrap.children || []).find(c => String(c.className).indexOf("bxjson") >= 0);
  ok(ta && String(ta.value).indexOf("def run") >= 0, "…holding the code");
  ta.value = "def pre(x):\n    return x + 1";
  wrap.children.find(c => c.className === "row").children
    .find(c => c.textContent === "save").click();
  eql(__t.bxNode(s.id).script, "def pre(x):\n    return x + 1", "editing the form saves the code");
  // the code travels with the design as JSON
  const json = JSON.parse(JSON.stringify(__t.bxExportCanvas()));
  eql(json.nodes.find(n => n.id === s.id).script, "def pre(x):\n    return x + 1",
    "the JSON carries the script");
  eql(__t.bxImportJSON(JSON.stringify(json), {}).ok, true, "…and loads back");
  eql(__t.bxNode(s.id).script, "def pre(x):\n    return x + 1", "…with the code intact");
  __t.bxRemove(src.id);
  __t.bxRemove(s.id);
});

check("the toolbar's Files menu exports/imports the design, top level only", () => {
  const i = __t.bxAdd("input", 100, 80, { dout: 4 });
  const fm = getEl("bxtFilesMenu");
  ok(fm.hidden, "the Files menu starts closed");
  ok(__t.bxtFilesToggle() === true, "Files opens it");
  ok(!fm.hidden, "…and the menu is visible");
  eql((fm.children || []).map(c => c.textContent), ["export", "import", "export weights …"],
    "…holding export, import and the weights export");
  fm.children.find(c => c.textContent === "export").click();
  const m = getEl("bxmenu");
  ok(!m.hidden, "export opens the JSON form");
  const ta = (m.children[0].children || []).find(c => c.readOnly === true);
  ok(ta && String(ta.value).indexOf('"type": "canvas"') >= 0, "…holding the design JSON");
  ok(fm.hidden, "…and the Files menu closed behind it");
  __t.bxMenuClose();
  __t.bxtFilesToggle();
  fm.children.find(c => c.textContent === "import").click();
  ok(!getEl("bxmenu").hidden, "import opens its form");
  ok(fm.hidden, "…and the Files menu closes");
  __t.bxMenuClose();
  // inside a nested box there is no canvas: Files dims and will not open
  const nest = __t.bxAdd("custom", 400, 80, {});
  __t.bxEnter(nest.id);
  ok(String(getEl("bxtFiles").className).indexOf("off") >= 0, "inside a box Files dims");
  ok(__t.bxtFilesToggle() === false && fm.hidden, "…and its menu will not open");
  __t.bxLeave();
  ok(String(getEl("bxtFiles").className).indexOf("off") < 0, "…and comes back at the top");
  __t.bxRemove(nest.id);
  __t.bxRemove(i.id);
});

check("the view cannot be lost: a tight world around origin and design", () => {
  __t.bxImportJSON(JSON.stringify({ type: "canvas", version: 1, nodes: [], edges: [] }), {});
  __t.BX.view = { cx: 999999, cy: -999999, k: 1 };
  __t.bxClampView();
  ok(__t.BX.view.cx <= 600 && __t.BX.view.cy >= -200,
    "a run-away view is pulled back a few cells past the origin: " +
    [__t.BX.view.cx, __t.BX.view.cy].join(", "));
  // …and the little world grows on its own with the design
  const far = __t.bxAdd("layer", 20000, 0, {});
  __t.BX.view = { cx: 999999, cy: 0, k: 1 };
  __t.bxClampView();
  ok(__t.BX.view.cx > 15000, "…but follows a box placed far from the origin");
  // the furthest zoom-out is 50 %
  __t.bxZoomAt({ clientX: 300, clientY: 200 }, 0.01);
  eql(__t.BX.view.k, 0.5, "zooming out stops at 50 %");
  __t.bxRemove(far.id);
  __t.BX.view = { cx: 0, cy: 0, k: 1 };
});

check("an inf/nan stat arrives as null and still reads clean", () => {
  eql(__t.bxNumText(null), "?", "a null stat is a question mark");
  eql(__t.bxNumText(NaN), "?", "\u2026and so is NaN");
  eql(__t.bxStateText({ shape: [4], peek: [null, 2.5, null, -1], mean: null }),
    "?, 2.5, ?, -1", "a peek with holes still reads");
});

check("a blank network is one input wired straight into one output", () => {
  for (const n of __t.BX.nodes.slice()) __t.bxRemove(n.id);      // start empty
  ok(__t.BX.nodes.length === 0 && __t.BX.edges.length === 0, "a truly empty canvas");
  ok(__t.bxSeedBlank() === true, "the blank gets its skeleton");
  const ins = __t.BX.nodes.filter(n => n.kind === "input");
  const outs = __t.BX.nodes.filter(n => n.kind === "output");
  eql([ins.length, outs.length], [1, 1], "exactly one input and one output");
  eql(__t.BX.edges.length, 1, "…with one wire");
  const e = __t.BX.edges[0];
  eql([e.from, e.to], [ins[0].id, outs[0].id], "…running input → output");
  eql([ins[0].x % 40, ins[0].y % 40, outs[0].x % 40, outs[0].y % 40], [0, 0, 0, 0],
    "…and both sit on the grid");
  ok(__t.bxSeedBlank() === false, "a used canvas is left alone");
});

if (process.env.UI_BOOT_CANVAS) {
  // opt-in: a real canvas (a served repo design) must round-trip losslessly —
  // every op keeps its dims (a -1 stays -1) and its spec extras
  check("a real canvas round-trips with its ops intact", () => {
    const fs2 = require("fs");
    const doc = JSON.parse(fs2.readFileSync(process.env.UI_BOOT_CANVAS, "utf8"));
    const back = __t.bxJSONDocFrom(JSON.parse(JSON.stringify(doc)), "canvas");
    ok(!!back, "the served canvas normalizes");
    let ops = 0, lost = [];
    const walk = (a, b) => {
      for (let i = 0; i < (a || []).length; i++) {
        const x = a[i], y = (b || [])[i];
        if (x.kind === "op") {
          ops++;
          const want = __t.bxDims(x.dims), got = __t.bxDims(y && y.dims);
          if (x.dims != null && JSON.stringify(want) !== JSON.stringify(got))
            lost.push(x.name + " dims " + JSON.stringify(want) + " -> " + JSON.stringify(got));
          if (x.keep != null && (!y || y.keep !== x.keep))
            lost.push(x.name + " keep " + x.keep + " -> " + (y && y.keep));
        }
        if (x.inner && x.inner.nodes) walk(x.inner.nodes, y && y.inner && y.inner.nodes);
      }
    };
    walk(doc.nodes, back.nodes);
    ok(ops > 700, "saw the ops: " + ops);
    ok(!lost.length, "not one dim or keep was lost: " + lost.slice(0, 3).join("; "));
  });
  check("the real canvas lands whole, the failing box intact", () => {
    const doc = JSON.parse(require("fs").readFileSync(process.env.UI_BOOT_CANVAS, "utf8"));
    for (const n of __t.BX.nodes.slice()) __t.bxRemove(n.id);
    const box = __t.bxAdd("custom", 0, 0, {});
    ok(__t.bxGGUFLoad(box.id, doc, null), "the repo load lands");
    ok(__t.BX.nodes.some(n => n.kind === "tokenize") &&
       __t.BX.nodes.some(n => n.kind === "detokenize"),
       "the whole pipeline: tokenizer, model, logits, de-tokenizer");
    const model = __t.BX.nodes.find(n => n.kind === "custom");
    let rz = null, q = null;
    const find = (nodes) => {
      for (const n of nodes) {
        if (n.kind === "op" && n.name === "z heads" && !rz) rz = n;
        if (n.kind === "op" && n.name === "q heads [q|gate]" && !q) q = n;
        if (n.inner && n.inner.nodes) find(n.inner.nodes);
      }
    };
    find(model.inner.nodes);
    ok(rz && rz.keep === 1, "the z-heads reshape keeps its keep: " + JSON.stringify(rz && rz.keep));
    ok(q && JSON.stringify(__t.bxDims(q.dims)) === JSON.stringify([-1, 24, 2, 256]),
       "the q-heads reshape keeps its -1: " + JSON.stringify(q && q.dims));
    for (const n of __t.BX.nodes.slice()) __t.bxRemove(n.id);
    __t.bxSeedBlank();
  });
}

check("the weight registry travels with the design", () => {
  __t.BX.weights = { "m.gguf::blk.0.w": { tensor: "blk.0.w", shape: [4, 8],
                                          dtype: "IQ4_XS", bytes: 16 } };
  const doc = __t.bxExportCanvas();
  ok(doc.weights && doc.weights["m.gguf::blk.0.w"] &&
     doc.weights["m.gguf::blk.0.w"].dtype === "IQ4_XS",
     "the export carries the registry: " + JSON.stringify(doc.weights || {}));
  const back = __t.bxJSONDocFrom(JSON.parse(JSON.stringify(doc)), "canvas");
  ok(back && back.weights && back.weights["m.gguf::blk.0.w"],
     "…and an import brings it back");
  __t.BX.weights = null;
});

check("the dataset table: csv rows, columns, dtypes — and the cells edit", () => {
  eql(__t.bxRowsParse("a\tc\nb").map(r => r.p), ["a", "b"],
     "legacy tab pairs still read");

  // csv round trip: quotes, commas and newlines inside a cell
  const csv = __t.bxCsvWrite([["prompt", "completion"], ["a, b", 'say "hi"'], ["x", "y\nz"]]);
  const grid = __t.bxCsvParse(csv);
  eql(grid[0], ["prompt", "completion"], "the header names the columns");
  eql(grid[1], ["a, b", 'say "hi"'], "commas and quotes survive");
  eql(grid[2], ["x", "y\nz"], "a newline in a cell survives too");

  // jsonl keys become columns — extras are not lost
  const n = __t.bxAdd("dataset", 0, 0, {});
  n.text = '{"prompt":"a","completion":"c","chosen":"x"}\n{"prompt":"b","completion":"d"}';
  let t = __t.bxTableRead(n);
  eql(t.cols, ["prompt", "completion", "chosen"], "jsonl keys become columns");
  eql(t.rows[0], ["a", "c", "x"], "…and the values line up under them");

  // an edit writes csv, headed by the names
  __t.bxTableSetCell(n, 1, 2, "y");
  eql(__t.bxTableRead(n).rows, [["a", "c", "x"], ["b", "d", "y"]], "cells write through");
  ok(n.format === "csv" &&
     String(n.text).split("\n")[0] === "prompt,completion,chosen",
     "…as csv, headed by the names");

  // columns come and go; a column's default name is its letter
  __t.bxTableInsertCol(n, 1);
  t = __t.bxTableRead(n);
  eql(t.cols.length, 4, "insert adds a column");
  eql(t.cols[1], "B", "…and a new column takes its letter");
  eql(t.rows[0], ["a", "", "c", "x"], "…and every row makes room");
  __t.bxTableRenameCol(n, 1, "middle");
  eql(__t.bxTableRead(n).cols[1], "middle", "a typed name replaces the letter");
  __t.bxTableRenameCol(n, 1, "");
  eql(__t.bxTableRead(n).cols[1], "B", "an emptied name goes back to its letter");
  __t.bxTableDeleteCol(n, 1);
  eql(__t.bxTableRead(n).cols, ["prompt", "completion", "chosen"], "delete puts it back");

  // an unnamed csv column reads as its letter — the default name
  const m = __t.bxAdd("dataset", 0, 0, {});
  m.format = "csv";
  m.text = "prompt,,note\nhi,there,1";
  eql(__t.bxTableRead(m).cols, ["prompt", "B", "note"],
     "an unnamed column reads as its letter");

  // dtypes: inferred like read_csv, overridable per column
  eql(__t.bxDtypeInfer(["1", "2"]), "int64", "whole numbers read as int64");
  eql(__t.bxDtypeInfer(["1.5", "2"]), "float64", "…decimals as float64");
  eql(__t.bxDtypeInfer(["true", "false"]), "bool", "…true/false as bool");
  eql(__t.bxDtypeInfer(["x"]), "object", "…text as object");
  eql(__t.bxDtypeInfer(["blob:" + "a".repeat(64) + ".png"]), "image", "…media ids as image");
  eql(__t.bxDtypeInfer(["blob:" + "b".repeat(64) + ".mp4"]), "video", "…and as video");
  eql(__t.bxTableType(n, 2), "object", "the chosen column reads as object");
  n.colt = ["", "", "int64"];
  eql(__t.bxTableType(n, 2), "int64", "an override wins");

  // the grid renders text in the cells — no boxes standing around
  const pane = document.createElement("div");
  __t.bxDatasetFill(pane, n);
  const vals = pane.querySelectorAll(".bxval");
  eql(vals.length, 6, "two rows by three columns of text");
  eql(vals.map(v => v.textContent).slice(0, 3), ["a", "c", "x"],
     "…as plain text in the cells");
  eql(pane.querySelectorAll(".bxnameval").map(i => i.textContent),
    ["prompt", "completion", "chosen"], "…under their names");
  eql(pane.querySelectorAll(".bxcell").length, 0, "and no text boxes anywhere");
  ok(pane.querySelectorAll(".bxthumb").length === 0, "no media, no thumbnails");

  // clicking a cell brings up the editor; the edit lands and text returns
  vals[0]._parent.dispatchEvent({ type: "click" });
  const editor = pane.querySelectorAll(".bxcell")[0];
  ok(editor, "clicking a cell brings up the editor");
  editor.value = "A";
  editor.dispatchEvent({ type: "change" });
  eql(__t.bxTableRead(n).rows[0], ["A", "c", "x"], "the edit lands in the rows");

  // bookkeeping travels with the design
  n.colw = [80, 0, 120];
  const back = __t.bxJSONNodeFrom(__t.bxJSONNode(n));
  eql(back.colw, [80, 0, 120], "column widths survive the JSON round trip");
  eql(back.colt, ["", "", "int64"], "…and so do the type overrides");

  // a media cell shows its media, referenced by id
  const id = "c".repeat(64) + ".png";
  __t.bxTableSetCell(n, 0, 1, "blob:" + id);
  const pane2 = document.createElement("div");
  __t.bxDatasetFill(pane2, n);
  const thumbs = pane2.querySelectorAll(".bxthumb");
  eql(thumbs.length, 1, "a blob cell renders a thumbnail");
  ok(String(thumbs[0].src).indexOf(id) >= 0, "…served by its id");
});

check("a vision tower and its image input survive the design round trip", () => {
  const tow = { id: "vision", kind: "vision", name: "Qwen vision", x: 40, y: 40,
                file: "models/mmproj/mmproj-x.gguf", projector: "qwen3vl_merger",
                d_model: 1152, heads: 16, blocks: 27, patch: 16, merge: 2,
                image_size: 768, proj_dim: 5120, eps: 1e-6,
                params: 460730096, wbytes: 927587264, dtype: "f16",
                din: 768, dout: 5120, place: "gpu" };
  const img = { id: "image", kind: "input", name: "image", x: 0, y: 40,
                input_mode: "image", image: "blob:abc.png", image_size: 768,
                patch_size: 16, image_mean: "0.5 0.5 0.5", image_std: "0.5 0.5 0.5",
                dout: 768 };
  const back = __t.bxJSONNodeFrom(__t.bxJSONNode(tow));
  eql([back.kind, back.projector, back.d_model, back.blocks, back.proj_dim,
       back.patch, back.image_size, back.eps, back.file, back.dtype, back.place,
       back.din, back.dout],
      ["vision", "qwen3vl_merger", 1152, 27, 5120, 16, 768, 1e-6,
       "models/mmproj/mmproj-x.gguf", "f16", "gpu", 768, 5120],
      "…the tower's geometry, file and placement");
  const bi = __t.bxJSONNodeFrom(__t.bxJSONNode(img));
  eql([bi.kind, bi.input_mode, bi.image, bi.image_size, bi.patch_size,
       bi.image_mean, bi.image_std, bi.dout],
      ["input", "image", "blob:abc.png", 768, 16, "0.5 0.5 0.5", "0.5 0.5 0.5", 768],
      "…and the image input's picture and geometry");
  const d = document.createElement("div");
  __t.bxPaneFill(d, tow);
  const rows = d.querySelectorAll(".bxbhint").map(x => String(x.textContent)).join(" ");
  ok(rows.indexOf("qwen3vl_merger") >= 0 && rows.indexOf("5120") >= 0,
     "…and the box's window says what it is: " + rows.slice(0, 60));
});

check("the Train box: the hub of the training — data, columns, signal, parameters", () => {
  const tr = __t.bxAdd("train", 0, 0, {});
  const ds = __t.bxAdd("dataset", 0, 120, {});
  const out = __t.bxAdd("output", 0, 240, { din: 8 });
  // the form: every hub field, in order
  __t.bxForm("train", null, tr, "train");
  const labels = Array.from(getEl("bxmenu").children[0].children)
    .filter(c => c.textContent);   // the field labels (the chip row carries no text)
  eql(labels.map(c => c.textContent),
      ["data", "test data", "x col", "y col", "classes", "paradigm",
       "target", "reward", "learning rate", "optimizer", "weight decay",
       "batch", "steps", "grad clip", "save every", "eval every",
       "temperature (rl)", "max tokens (rl)", "n samples (rl)"],
      "the form names the data, the columns, the signal and every parameter");
  const lab = (t) => labels.find(c => c.textContent === t);
  ok(lab("reward").hidden, "the reward picker stays hidden while supervised");
  ok(lab("temperature (rl)").hidden, "…and so do the rl sampling fields");
  lab("paradigm").children[0].value = "rl";
  lab("paradigm").children[0].dispatchEvent({ type: "change" });
  ok(!lab("reward").hidden, "rl brings the reward picker back");
  ok(!lab("n samples (rl)").hidden, "…and its sample fields");
  __t.bxMenuClose();
  // configuring lands the hub in the node
  ok(__t.bxConfigure(tr.id, { name: "hub", dataset: ds.id, x_col: "a", y_col: "b",
                                classes: "cat dog", paradigm: "supervised", target: out.id,
                                lr: 0.01, batch: 8, steps: 40 }), "configuring lands");
  eql([tr.name, tr.dataset, tr.x_col, tr.y_col, tr.paradigm, tr.target],
      ["train 1", ds.id, "A", "B", "supervised", out.id],
      "the node binds the data, the columns and the signal (the name is not ours to set)");
  eql(tr.classes, ["cat", "dog"], "…and the classes land as a list");
  eql([tr.lr, tr.batch, tr.steps], [0.01, 8, 40], "…and the parameters as numbers");
  // and it all travels with the design
  const back = __t.bxJSONNodeFrom(__t.bxJSONNode(tr));
  eql([back.dataset, back.x_col, back.y_col, back.paradigm, back.target],
      [ds.id, "A", "B", "supervised", out.id], "the hub survives the JSON round trip");
  eql(back.classes, ["cat", "dog"], "…so do the classes");
  eql([back.lr, back.batch, back.steps], [0.01, 8, 40], "…and so do the parameters");
  // the window opens on Execute: the training, with no mode switch
  const tp = document.createElement("div");
  __t.BXRUN = { local: { connected: false, agent: null }, inputs: {}, last: null,
                loaded: null, loading: false, loadKey: null, msg: "", running: false };
  __t.bxTrainFill(tp, tr);
  eql(tp.querySelectorAll(".runmi").map(e => e.textContent), ["Execute", "Spec"],
     "the Train box's own menu: Execute and Spec");
  eql(tp.querySelectorAll(".mseg").length, 0, "…and no mode switch anywhere");
  ok(tp.querySelectorAll(".chip").map(e => e.textContent).join(" | ")
       .indexOf("set up the training") >= 0, "…just the training and its controls");
  __t.bxRemove(tr.id);
  __t.bxRemove(ds.id);
  __t.bxRemove(out.id);
});

check("the Train box: one only, at the top level, and no rename", () => {
  const tr = __t.bxAdd("train", 0, 0, {});
  ok(!!tr && tr.name === "train 1", "a Train box lands, named train 1");
  ok(!__t.bxErrorSet().has(tr.id),
     "…and a lone Train box is not an error — it takes no wires");
  const ag = __t.bxAdd("agent", 0, 300, {});
  ok(!__t.bxErrorSet().has(ag.id), "…nor is a lone agent box");
  __t.bxRemove(ag.id);
  ok(__t.bxAdd("train", 200, 0, {}) === null, "…a second one is refused");
  ok(__t.bxDropPalette("train", null) === null, "…however it is dropped");
  // the right-click menu: no rename; the hub's own setup form has no name field
  __t.bxMenuBuild("box", tr.id, {});
  openCat(getEl("bxmenu"), "this box");
  const items = catItems(getEl("bxmenu"));
  ok(items.indexOf("rename this box") < 0, "…and it cannot be renamed");
  ok(items.some(t => t.indexOf("set up the training") === 0),
     "…while setting up the training stays one click away");
  __t.bxForm("train", null, tr, "train");
  const labels = Array.from(getEl("bxmenu").children[0].children).filter(c => c.textContent);
  ok(labels.map(c => c.textContent).indexOf("name") < 0, "…its setup form has no name field");
  __t.bxMenuClose();
  // a nested box never holds one
  const nest = __t.bxAdd("custom", 0, 300, {});
  __t.bxEnter(nest.id);
  ok(__t.bxAdd("train", 0, 0, {}) === null, "inside a nested box a Train box is refused");
  ok(__t.bxtRunToggle() === false, "…and the Train button shows no box here");
  __t.bxtRunRefresh();
  ok(getEl("bxtTrainBtn").title.indexOf("top level") >= 0,
     "…and says the Train box lives at the top level");
  __t.bxLeave();
  // it never moves into one, and a paste never makes a second
  ok(__t.bxMoveInto(tr.id, nest.id) === false, "it never moves into a nested box");
  ok(__t.bxCopyBoxes([tr.id]), "the hub copies");
  const before = __t.BX.nodes.length;
  ok(__t.bxPasteBoxes(null, { x: 320, y: 320 }) === false, "…but pasting it makes no second");
  ok(__t.BX.nodes.length === before, "…and nothing lands");
  // a canvas with stray ones loads with the extras stripped
  const doc = { type: "canvas", version: 1, nodes: [
      { id: "t1", kind: "train", x: 0, y: 0, name: "one" },
      { id: "t2", kind: "train", x: 260, y: 0, name: "two" },
      { id: "cc", kind: "custom", x: 0, y: 260,
        inner: { type: "nested", version: 1,
                 nodes: [{ id: "t3", kind: "train", x: 0, y: 0 }], edges: [] } }],
      edges: [] };
  const res = __t.bxImportJSON(JSON.stringify(doc), {});
  eql(res.ok, true, "a canvas with stray Run boxes loads: " + (res.error || ""));
  const kept = __t.BX.nodes.filter(n => n.kind === "train");
  eql([kept.length, kept[0].name], [1, "one"], "…one Run box kept (the first)");
  const held = __t.BX.nodes.find(n => n.kind === "custom");
  eql((held.inner.nodes || []).filter(n => n.kind === "train").length, 0,
      "…and none inside the nested box");
  for (const n of __t.BX.nodes.slice()) __t.bxRemove(n.id);
});


check("the account: guests by default, a bot check to sign up, sealed designs", async () => {
  const nodes0 = __t.BX.nodes.length;
  const flush = async () => {                  // let a UI chain land: the crypto settles
    for (let i = 0; i < 12; i++) await new Promise(r => setImmediate(r));
  };
  // everyone starts a guest — nothing is saved for one
  eql(__t.BXACCT.user, null, "the account starts empty: a guest");
  eql(getEl("bxtAccount").textContent, "guest ▾", "the toolbar says guest");
  eql(__t.bxAccountSaveForm(), false, "a guest has no save dialog");
  ok(__t.bxtAccountToggle() === true && !getEl("bxtAccountMenu").hidden,
     "the account menu opens");
  const guestItems = getEl("bxtAccountMenu").children.map(c => c.textContent);
  ok(guestItems.some(t => t.indexOf("create an account") === 0) &&
     guestItems.some(t => t.indexOf("log in with a token") === 0) &&
     !guestItems.some(t => t.indexOf("save this canvas") === 0),
     "a guest is offered an account, not saving: " + guestItems.join(" | "));
  __t.bxtDropsClose();
  // the token the client makes: 32 random bytes, base64url, never a password
  const tok = __t.bxSecret();
  ok(/^[A-Za-z0-9_-]{43}$/.test(tok) && tok !== __t.bxSecret(),
     "the browser makes its own 43-character token: " + tok.slice(0, 8) + "…");
  // the derivations: what the server may see, and what seals the design
  const auth = await __t.bxAuthCred(tok);
  ok(/^[0-9a-f]{64}$/.test(auth) && auth !== tok,
     "auth is a sha256 of the token, never the token: " + auth.slice(0, 12) + "…");
  eql(await __t.bxNameTag(tok, " My Canvas "), await __t.bxNameTag(tok, "my canvas"),
      "the name tag is keyed and case-blind");
  const sealed = await __t.bxSeal(tok, { hello: "world" });
  ok(!JSON.stringify(sealed).includes("world"), "sealing hides the plaintext");
  eql(await __t.bxUnseal(tok, sealed), { hello: "world" }, "…the same token opens it");
  let refused = false;
  try { await __t.bxUnseal(tok + "x", sealed); } catch (e) { refused = true; }
  ok(refused, "no other token can open it");
  // creating one: the widget runs, and only sealed values leave the browser
  const realFetch = global.fetch, realRand = crypto.getRandomValues;
  crypto.getRandomValues = (arr) => { for (let i = 0; i < arr.length; i++) arr[i] = i + 1; return arr; };
  const made = __t.bxSecret();                 // the dialog makes exactly this token
  let renders = 0, sent = null, hit = null;
  const seen = new Promise(r => { hit = r; });
  global.turnstile = { render: (_mount, opts) => { renders++; opts.callback("cf-test"); return "w1"; },
                       reset: () => {} };
  global.fetch = async (url, opts) => {
    if (String(url) === "/api/account/register") {
      sent = JSON.parse(opts.body);
      hit();
      return { ok: true, status: 200,
               json: async () => ({ ok: true, user: { id: "u_1", name_ct: sent.name_ct, created: 0 } }) };
    }
    return { ok: true, status: 200, json: async () => ({ ok: true }) };
  };
  __t.bxAccountCreate();
  const form = getEl("bxmenu");
  ok(!form.hidden, "the create dialog is up");
  eql(renders, 1, "…with Cloudflare's widget rendered in it");
  form.querySelectorAll(".bxinp")[0].value = "tester";
  form.querySelectorAll(".chip").find(c => String(c.textContent) === "create").click();
  await seen;
  eql(sent.auth, await __t.bxAuthCred(made), "register carries the derived auth");
  eql(sent.challenge, "cf-test", "…and the bot check's challenge");
  eql(sent.name_tag, await __t.bxNameTag(made, "tester"), "…and the keyed name tag");
  eql(await __t.bxUnseal(made, sent.name_ct), "tester",
      "…and the sealed name, which only the token opens");
  ok(!JSON.stringify(sent).includes("tester") && !JSON.stringify(sent).includes(made),
     "no plaintext name goes out — and never the token itself");
  await flush();                               // the stubbed answer lands: signed in + revealed
  eql(getEl("bxtAccount").textContent, "tester ▾", "signed in, the toolbar reads the name");
  eql(__t.BXACCT.token, made, "the token is kept in this browser");
  eql(__t.bxTokenRecall(), made, "…remembered for the next reload");
  eql(getEl("bxmenu").querySelectorAll(".bxtok")[0].value, made,
      "the token is shown once, in full — the login and the key");
  __t.bxMenuClose();
  // saving: the design travels sealed
  let save = null, saveHit = null;
  const saved = new Promise(r => { saveHit = r; });
  global.fetch = async (url, opts) => {
    if (String(url) === "/api/account/save") {
      save = JSON.parse(opts.body);
      saveHit();
      return { ok: true, status: 200, json: async () => ({ ok: true, saved: { name_tag: save.name_tag } }) };
    }
    return { ok: true, status: 200, json: async () => ({ ok: true }) };
  };
  __t.bxAdd("input", 40, 40, { dout: 8 });     // something worth keeping
  __t.bxAccountSaveForm();
  const f2 = getEl("bxmenu");
  f2.querySelectorAll(".bxinp")[0].value = "my canvas";
  f2.querySelectorAll(".chip").find(c => String(c.textContent) === "save").click();
  await saved;
  ok(save && save.design === undefined && !JSON.stringify(save).includes("my canvas"),
     "saving posts ciphertext only");
  eql(save.name_tag, await __t.bxNameTag(made, "my canvas"), "…under the keyed name tag");
  eql(await __t.bxUnseal(made, save.name_ct), "my canvas", "…with the name sealed");
  eql(((await __t.bxUnseal(made, save.blob)).nodes || []).length, nodes0 + 1,
      "…and the whole design sealed");
  await flush();
  __t.bxMenuClose();
  while (__t.BX.nodes.length > nodes0) __t.bxRemove(__t.BX.nodes[__t.BX.nodes.length - 1].id);
  // locked: the cookie says who, but without the token there is nothing to read
  __t.bxAccountLocked(__t.BXACCT.user);
  eql(__t.BXACCT.locked, true, "without the key the account is locked");
  eql(getEl("bxtAccount").textContent, "account 🔒", "…and the toolbar says so");
  eql(__t.bxAccountSaveForm(), false, "…saving waits for the unlock");
  __t.bxAccountUnlockForm();
  const f3 = getEl("bxmenu");
  f3.querySelectorAll(".bxinp")[0].value = "not-the-token";
  f3.querySelectorAll(".chip").find(c => String(c.textContent) === "unlock").click();
  await flush();
  const hint = f3.querySelectorAll(".panehint");
  ok(String(hint[hint.length - 1].textContent).indexOf("cannot open") >= 0,
     "a wrong token cannot unlock: " + hint[hint.length - 1].textContent);
  f3.querySelectorAll(".bxinp")[0].value = made;
  f3.querySelectorAll(".chip").find(c => String(c.textContent) === "unlock").click();
  await flush();
  eql(getEl("bxtAccount").textContent, "tester ▾", "the real token unlocks it");
  __t.bxMenuClose();
  // opening: rows carry sealed names; a click asks for the blob by its tag
  __t.bxAccountOpenForm();
  const f4 = getEl("bxmenu");
  const hints = f4.querySelectorAll(".panehint");
  const list = f4.querySelectorAll(".bxlist")[0];
  const tagX = "ab".repeat(32);
  await __t.bxAccountList(list, hints[hints.length - 1],
                          [{ name_tag: tagX, name_ct: await __t.bxSeal(made, "demo"),
                             updated: 1791986884, bytes: 40 }]);
  const rows = f4.querySelectorAll(".rowl");
  eql(rows.length, 1, "one row per saved design");
  eql(rows[0].querySelectorAll(".nm")[0].textContent, "demo", "…named from the sealed name");
  let asked = "";
  global.fetch = async (url) => {
    asked = String(url);
    return { ok: true, status: 200, json: async () => ({ ok: true, name_ct: null,
             blob: await __t.bxSeal(made, { type: "canvas", version: 1, nodes: [], edges: [] }),
             updated: 0 }) };
  };
  const box = __t.bxAdd("output", 480, 40, {});
  rows[0].click();
  eql(asked, "", "with work on the canvas, the first click only warns");
  ok(String(hints[hints.length - 1].textContent).indexOf("again") >= 0,
     "…and says a second click replaces it");
  rows[0].click();
  eql(asked, "/api/account/canvas?tag=" + tagX, "the second click asks the server by tag");
  global.fetch = realFetch;
  __t.bxRemove(box.id);
  const keepDoc = JSON.stringify(__t.bxExportCanvas());
  const blob = await __t.bxSeal(made, { type: "canvas", version: 1, nodes: [], edges: [] });
  eql(__t.bxAccountOpenApply("demo", await __t.bxUnseal(made, blob)), true,
      "…and the sealed design opens locally");
  eql(__t.BX.nodes.length, 0, "…the saved design is what is on the canvas now");
  __t.bxImportJSON(keepDoc, {});
  // logging out: a guest again, and the browser forgets the key
  __t.bxAccountOut();
  eql(__t.BXACCT.user, null, "logging out is a guest again");
  eql(getEl("bxtAccount").textContent, "guest ▾", "…and the toolbar says so");
  eql(__t.bxTokenRecall(), "", "…and the remembered key is gone");
  __t.bxMenuClose();
  delete global.turnstile;
  crypto.getRandomValues = realRand;
  eql(__t.BX.nodes.length, nodes0, "the canvas is left as it was found");
});

Promise.all(pending).then(() => {            // async checks report with the rest
  if (failed) { console.log("ui_boot: " + failed + " failure(s)"); process.exit(1); }
  console.log("ui_boot: all assertions passed (" + __t.BX.nodes.length + " boxes on the canvas)");
  process.exit(0);
});
