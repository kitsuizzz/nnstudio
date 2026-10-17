/**
 * The component registry: the single source of truth for every component an
 * AI-built app can use. Each entry sanitizes raw model output into a safe
 * block and documents the component in the prompt vocabulary — so adding a
 * future component is one entry here plus one renderer.
 *
 * The registry is deliberately pure (no React): the spec parser, the prompt,
 * the AppFrame renderers and the sandboxed-logic bridge all share it.
 */

import type { Align, Emph, Face, Justify, StudioAction, StudioBlock, StudioValue, Tone } from "../types";

export const MAX_BLOCKS = 80;
export const MAX_NESTING = 4;
/** Hard ceiling across a whole spec, so nested lists can't multiply without bound. */
export const MAX_TOTAL_BLOCKS = 400;

/** Options for {@link sanitizeBlocks}: what a brain is allowed to emit. */
export interface SanitizeOpts {
  /** Trusted brains (the product's own apps) may emit app-only components. */
  allowAppOnly?: boolean;
  /** Override the per-list block cap (trusted brains render longer lists). */
  maxBlocks?: number;
  /** A shared per-spec budget, decremented for every accepted block. */
  budget?: { left: number };
}

// Long-form content (prose, reasoning, raw output) is capped tightly for model
// output but generously for trusted brains, which render full conversations.
const MAX_CONTENT_AI = 40_000;
const MAX_CONTENT_TRUSTED = 200_000;

function contentMax(opts?: SanitizeOpts): number {
  return opts?.allowAppOnly ? MAX_CONTENT_TRUSTED : MAX_CONTENT_AI;
}

export interface BlockDef {
  /** Reduce raw model output to a safe block, or null to drop it. */
  sanitize: (raw: unknown, depth?: number, opts?: SanitizeOpts) => StudioBlock | null;
  /** One prompt line describing the component to the model. */
  doc: string;
  /**
   * App-only components (ai: false) are the product's own surfaces, composed
   * by hosted apps. They are excluded from the model's prompt vocabulary and
   * dropped by the sanitizer, so neither the AI nor an untrusted sandboxed
   * brain can emit them — only a trusted brain may.
   */
  ai?: boolean;
}

function isRecord(x: unknown): x is Record<string, unknown> {
  return !!x && typeof x === "object" && !Array.isArray(x);
}

function str(x: unknown, max: number): string {
  return typeof x === "string" ? x.slice(0, max) : "";
}

function num(x: unknown, min: number, max: number): number | null {
  return typeof x === "number" && Number.isFinite(x) && x >= min && x <= max ? Math.trunc(x) : null;
}

function bool(x: unknown): boolean {
  return x === true;
}

function cleanStringArray(x: unknown, maxLen: number, itemMax: number): string[] | null {
  if (!Array.isArray(x)) return null;
  const out = x
    .filter((s): s is string => typeof s === "string")
    .map((s) => s.slice(0, itemMax))
    .slice(0, maxLen);
  return out.length ? out : null;
}

/** Sanitize one action value: primitives, arrays, and plain records (depth-capped). */
function cleanValue(v: unknown, depth: number): StudioValue | undefined {
  if (typeof v === "string") return v.slice(0, 200);
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "boolean" || v === null) return v;
  if (Array.isArray(v)) {
    if (depth >= 3) return undefined;
    const arr: StudioValue[] = [];
    for (const item of v.slice(0, 512)) {
      const c = cleanValue(item, depth + 1);
      if (c !== undefined) arr.push(c);
    }
    return arr;
  }
  if (isRecord(v) && depth < 3) {
    const out: Record<string, StudioValue> = {};
    for (const [k, item] of Object.entries(v).slice(0, 32)) {
      const c = cleanValue(item, depth + 1);
      if (c !== undefined) out[k.slice(0, 64)] = c;
    }
    return Object.keys(out).length ? out : undefined;
  }
  return undefined;
}

/** Sanitize a path→value map (set / push). */
function cleanValueMap(x: unknown): Record<string, StudioValue> | undefined {
  if (!isRecord(x)) return undefined;
  const out: Record<string, StudioValue> = {};
  for (const [k, v] of Object.entries(x)) {
    if (Object.keys(out).length >= 8) break;
    const value = cleanValue(v, 0);
    if (value !== undefined) out[k.slice(0, 64)] = value;
  }
  return Object.keys(out).length ? out : undefined;
}

function cleanAction(x: unknown, depth = 0): StudioAction | null {
  if (!isRecord(x) || depth > 2) return null;
  const action: StudioAction = {};
  const set = cleanValueMap(x.set);
  if (set) action.set = set;
  if (isRecord(x.inc)) {
    const inc: Record<string, number> = {};
    for (const [k, v] of Object.entries(x.inc)) {
      if (Object.keys(inc).length >= 8) break;
      if (typeof v === "number" && Number.isFinite(v)) inc[k.slice(0, 64)] = v;
    }
    if (Object.keys(inc).length) action.inc = inc;
  }
  const push = cleanValueMap(x.push);
  if (push) action.push = push;
  if (isRecord(x.remove)) {
    const remove: Record<string, string | number> = {};
    for (const [k, v] of Object.entries(x.remove)) {
      if (Object.keys(remove).length >= 8) break;
      if (typeof v === "string") remove[k.slice(0, 64)] = v.slice(0, 200);
      else if (typeof v === "number" && Number.isFinite(v)) remove[k.slice(0, 64)] = Math.trunc(v);
    }
    if (Object.keys(remove).length) action.remove = remove;
  }
  if (Array.isArray(x.toggle)) {
    const toggle = x.toggle
      .filter((p): p is string => typeof p === "string")
      .map((p) => p.slice(0, 64))
      .slice(0, 8);
    if (toggle.length) action.toggle = toggle;
  }
  if (Array.isArray(x.reset)) {
    const reset = x.reset.filter((p): p is string => typeof p === "string").map((p) => p.slice(0, 64)).slice(0, 8);
    if (reset.length) action.reset = reset;
  }
  const to = str(x.to, 64);
  if (to) action.to = to;
  if (Array.isArray(x.actions)) {
    const chain: StudioAction[] = [];
    for (const nested of x.actions.slice(0, 8)) {
      const a = cleanAction(nested, depth + 1);
      if (a) chain.push(a);
    }
    if (chain.length) action.actions = chain;
  }
  return action.set ||
    action.inc ||
    action.push ||
    action.remove ||
    action.toggle ||
    action.reset ||
    action.to ||
    action.actions
    ? action
    : null;
}

/** A "{{expression}}" string used where a value (not text) is expected. */
function bindString(x: unknown): `{{${string}}}` | null {
  if (typeof x !== "string") return null;
  const t = x.trim();
  if (t.length > 200 || !/^\{\{[\s\S]+\}\}$/.test(t)) return null;
  return t as `{{${string}}}`;
}

/** Sanitize the wallabel shape (shared by the `wallabel` and `work` blocks). */
function sanitizeWallabel(x: unknown): StudioBlock | null {
  if (!isRecord(x)) return null;
  const kicker = str(x.kicker, 40);
  const title = str(x.title, 600);
  const artist = str(x.artist, 80);
  const dims = str(x.dims, 120);
  if (!kicker || !title || !artist) return null;
  const block: StudioBlock = { type: "wallabel", kicker, title, artist, dims };
  const no = num(x.no, 1, 99999);
  if (no != null) (block as { no?: number }).no = no;
  if (bool(x.live)) (block as { live?: boolean }).live = true;
  if (bool(x.long)) (block as { long?: boolean }).long = true;
  const time = str(x.time, 40);
  if (time) (block as { time?: string }).time = time;
  const medium = str(x.medium, 120);
  if (medium) (block as { medium?: string }).medium = medium;
  const images = cleanStringArray(x.images, 8, 2000);
  if (images) (block as { images?: string[] }).images = images;
  if (isRecord(x.study) && typeof x.study.content === "string") {
    const study: { content: string; active: boolean; ms?: number } = {
      content: (x.study.content as string).slice(0, 20_000),
      active: bool(x.study.active),
    };
    const ms = num(x.study.ms, 0, 10_000_000);
    if (ms != null) study.ms = ms;
    (block as { study?: { content: string; active: boolean; ms?: number } }).study = study;
  }
  return block;
}

/** A gap hint for layout containers ("sm" | "md" | "lg"), when valid. */
function gapOf(x: unknown): "sm" | "md" | "lg" | undefined {
  if (isRecord(x) && (x.gap === "sm" || x.gap === "md" || x.gap === "lg")) return x.gap;
  return undefined;
}

function alignOf(x: unknown): Align | undefined {
  if (isRecord(x) && (x.align === "start" || x.align === "center" || x.align === "end" || x.align === "stretch" || x.align === "baseline")) return x.align;
  return undefined;
}

function justifyOf(x: unknown): Justify | undefined {
  if (isRecord(x) && (x.justify === "start" || x.justify === "center" || x.justify === "end" || x.justify === "between" || x.justify === "around")) return x.justify;
  return undefined;
}

function emphOf(x: unknown): Emph | undefined {
  if (isRecord(x) && (x.emph === "display" || x.emph === "h1" || x.emph === "h2" || x.emph === "body" || x.emph === "small" || x.emph === "caption")) return x.emph;
  return undefined;
}

function faceOf(x: unknown): Face | undefined {
  if (isRecord(x) && (x.face === "serif" || x.face === "sans" || x.face === "mono")) return x.face;
  return undefined;
}

function toneOf(x: unknown): Tone | undefined {
  if (isRecord(x) && (x.tone === "accent" || x.tone === "danger" || x.tone === "ok" || x.tone === "muted")) return x.tone;
  return undefined;
}

function sizesOf(x: unknown): number[] | undefined {
  if (!isRecord(x) || !Array.isArray(x.sizes)) return undefined;
  const out = x.sizes
    .filter((n): n is number => typeof n === "number" && Number.isFinite(n) && n >= 1 && n <= 100)
    .slice(0, 8)
    .map((n) => Math.trunc(n));
  return out.length >= 2 ? out : undefined;
}

/** Sanitize a container's `blocks` child list, respecting the nesting budget. */
function cleanBlocks(x: unknown, depth: number | undefined, opts?: SanitizeOpts): StudioBlock[] | null {
  if (depth != null && depth > MAX_NESTING) return null;
  return sanitizeBlocks(x, (depth ?? 1) + 1, opts);
}

/** The registry, keyed by block type. */
export const BLOCKS: Record<string, BlockDef> = {
  text: {
    sanitize: (x, _depth, opts) => {
      if (!isRecord(x)) return null;
      const content = str(x.content, contentMax(opts));
      if (!content) return null;
      const block: StudioBlock = { type: "text", content };
      if (bool(x.error)) block.error = true;
      if (bool(x.streaming)) block.streaming = true;
      const emph = emphOf(x);
      if (emph) block.emph = emph;
      const face = faceOf(x);
      if (face) block.face = face;
      return block;
    },
    doc: '{"type":"text","content":"…","emph":"h1","face":"sans"} — prose; Markdown is rendered in the app serif body style, and LaTeX math is typeset — write formulas as $inline$ or $$display$$ with SINGLE braces ({{ }} is the data-binding syntax). Optional "emph":"display"|"h1"|"h2"|"body"|"small"|"caption" and "face":"serif"|"sans"|"mono".',
  },
  heading: {
    sanitize: (x) => {
      if (!isRecord(x)) return null;
      const content = str(x.content, 200);
      if (!content) return null;
      const block: StudioBlock = { type: "heading", content };
      const emph = emphOf(x);
      if (emph) block.emph = emph;
      const face = faceOf(x);
      if (face) block.face = face;
      return block;
    },
    doc: '{"type":"heading","content":"…","emph":"h1"} — a section heading. Optional "emph":"display"|"h1"|"h2"|"body"|"small"|"caption" and "face":"serif"|"sans"|"mono".',
  },
  code: {
    sanitize: (x) => {
      if (!isRecord(x)) return null;
      const content = str(x.content, 40_000);
      if (!content) return null;
      const block: StudioBlock = { type: "code", content };
      if (typeof x.language === "string" && x.language) (block as { language: string }).language = x.language.slice(0, 24);
      if (Array.isArray(x.highlight)) {
        const hl = x.highlight.filter((n): n is number => typeof n === "number" && n > 0).slice(0, 12);
        if (hl.length) (block as { highlight: number[] }).highlight = hl;
      }
      return block;
    },
    doc: '{"type":"code","content":"…","language":"python","highlight":[5]} — a code panel; "highlight" is 1-based line numbers to emphasize.',
  },
  output: {
    sanitize: (x) => {
      if (!isRecord(x)) return null;
      const bind = bindString(x.lines);
      if (bind) return { type: "output", lines: bind };
      if (!Array.isArray(x.lines)) return { type: "output" };
      const lines = x.lines.filter((l): l is string => typeof l === "string").slice(0, 200);
      return { type: "output", lines };
    },
    doc: '{"type":"output","lines":["…"] or "{{log}}"} — a monospace output pane (logs, results); "lines" may bind to an array.',
  },
  button: {
    sanitize: (x) => {
      if (!isRecord(x)) return null;
      const label = str(x.label, 60);
      const to = str(x.to, 64);
      if (!label || !to) return null;
      const block: StudioBlock = { type: "button", label, to };
      if (x.variant === "ghost") (block as { variant: "ghost" }).variant = "ghost";
      if (bool(x.disabled)) (block as { disabled?: boolean }).disabled = true;
      const tone = toneOf(x);
      if (tone) block.tone = tone;
      const action = cleanAction(x.action);
      if (action) (block as { action: StudioAction }).action = action;
      return block;
    },
    doc: '{"type":"button","label":"Run","to":"t1","tone":"accent","action":{"set":{"x":"{{score + 1}}"}}} — navigate to a state (and optionally mutate data); "variant":"ghost" for a quiet button; "tone":"accent"|"danger"|"ok"|"muted"; action values may be {{expressions}}.',
  },
  note: {
    sanitize: (x) => {
      if (!isRecord(x)) return null;
      const content = str(x.content, 1_000);
      if (!content) return null;
      const block: StudioBlock = { type: "note", content };
      if (x.tone === "hint" || x.tone === "error") block.tone = x.tone;
      const emph = emphOf(x);
      if (emph) block.emph = emph;
      const face = faceOf(x);
      if (face) block.face = face;
      return block;
    },
    doc: '{"type":"note","content":"…","tone":"hint"} — a small muted caption; optional "tone":"hint"|"error", "emph", "face".',
  },
  row: {
    sanitize: (x, depth, opts) => sanitizeContainer(x, "row", depth, opts),
    doc: '{"type":"row","blocks":[ … ],"align":"center","justify":"between"} — components side by side; optional "align":"start"|"center"|"end"|"stretch"|"baseline" and "justify":"start"|"center"|"end"|"between"|"around".',
  },
  spacer: {
    sanitize: (x) => {
      const size = isRecord(x) && (x.size === "sm" || x.size === "lg") ? x.size : "md";
      return { type: "spacer", size };
    },
    doc: '{"type":"spacer","size":"sm"|"md"|"lg"} — vertical gap.',
  },
  divider: {
    sanitize: () => ({ type: "divider" }),
    doc: '{"type":"divider"} — a hairline separator.',
  },
  card: {
    sanitize: (x, depth, opts) => sanitizeContainer(x, "card", depth, opts),
    doc: '{"type":"card","blocks":[ … ]} — a bordered panel grouping components.',
  },
  grid: {
    sanitize: (x) => {
      if (!isRecord(x)) return null;
      const cols = num(x.cols, 1, 32);
      if (cols == null) return null;
      let cells: string[] | `{{${string}}}` | null = null;
      if (Array.isArray(x.cells)) {
        const list = x.cells
          .filter((c): c is string => typeof c === "string")
          .map((c) => c.slice(0, 8))
          .slice(0, cols * 64);
        if (list.length) cells = list;
      } else {
        const bind = bindString(x.cells);
        if (bind) cells = bind;
      }
      if (!cells) return null;
      const block: StudioBlock = { type: "grid", cols, cells };
      if (bool(x.selectable)) (block as { selectable: boolean }).selectable = true;
      const onCell = cleanAction(x.onCell);
      if (onCell) (block as { onCell: StudioAction }).onCell = onCell;
      return block;
    },
    doc: '{"type":"grid","cols":8,"cells":["♜","♞","…"] or "cells":"{{board}}","selectable":true,"onCell":{"set":{"sel":"{{arg}}"}}} — a tappable grid of cells (boards, keypads, maps); the argument of a cell tap is its index.',
  },
  choice: {
    sanitize: (x) => {
      if (!isRecord(x)) return null;
      const options: (string | { label: string; value: string })[] = [];
      if (Array.isArray(x.options)) {
        for (const o of x.options.slice(0, 24)) {
          if (typeof o === "string" && o) options.push(o.slice(0, 60));
          else if (isRecord(o) && typeof o.label === "string" && typeof o.value === "string")
            options.push({ label: o.label.slice(0, 60), value: o.value.slice(0, 60) });
          if (options.length >= 24) break;
        }
      }
      if (options.length < 2) return null;
      const block: StudioBlock = { type: "choice", options };
      if (typeof x.value === "string") block.value = x.value.slice(0, 64);
      const tone = toneOf(x);
      if (tone) block.tone = tone;
      const onChoose = cleanAction(x.onChoose);
      if (onChoose) block.onChoose = onChoose;
      return block;
    },
    doc: '{"type":"choice","options":["A","B"],"value":"{{pick}}","tone":"accent","onChoose":{"set":{"pick":"{{arg}}"}}} — a pill/radio group; "tone":"accent"|"danger"|"ok"|"muted"; the argument of a pick is the option value.',
  },
  input: {
    sanitize: (x) => {
      if (!isRecord(x)) return null;
      const block: StudioBlock = { type: "input" };
      if (typeof x.label === "string" && x.label) block.label = x.label.slice(0, 80);
      if (typeof x.placeholder === "string" && x.placeholder) block.placeholder = x.placeholder.slice(0, 80);
      if (typeof x.value === "string") block.value = x.value.slice(0, 64);
      if (x.kind === "number" || x.kind === "email" || x.kind === "url" || x.kind === "password") block.kind = x.kind;
      const onSubmit = cleanAction(x.onSubmit);
      if (onSubmit) block.onSubmit = onSubmit;
      return block;
    },
    doc: '{"type":"input","label":"Name","placeholder":"…","value":"{{name}}","kind":"text","onSubmit":{"set":{"name":"{{arg}}"}}} — a field that commits on OK/Enter; kind is text/number/email/url/password; the submit argument is the entered value.',
  },
  select: {
    sanitize: (x, _depth, opts) => {
      if (!isRecord(x)) return null;
      const options: (string | { label: string; value: string })[] = [];
      if (Array.isArray(x.options)) {
        for (const o of x.options.slice(0, 24)) {
          if (typeof o === "string" && o) options.push(o.slice(0, 60));
          else if (isRecord(o) && typeof o.label === "string" && typeof o.value === "string")
            options.push({ label: o.label.slice(0, 60), value: o.value.slice(0, 60) });
          if (options.length >= 24) break;
        }
      }
      if (options.length < 1) return null;
      const block: StudioBlock = { type: "select", options };
      if (typeof x.label === "string" && x.label) block.label = x.label.slice(0, 80);
      if (typeof x.value === "string") block.value = x.value.slice(0, 64);
      const onChoose = cleanAction(x.onChoose);
      if (onChoose) block.onChoose = onChoose;
      if (opts?.allowAppOnly) {
        const chip = str(x.chip, 40);
        if (chip) block.chip = chip;
        const ariaLabel = str(x.ariaLabel, 80);
        if (ariaLabel) block.ariaLabel = ariaLabel;
      }
      return block;
    },
    doc: '{"type":"select","label":"Model","options":["a","b"],"value":"{{pick}}","onChoose":{"set":{"pick":"{{arg}}"}}} — a dropdown picker; it commits on selection and the argument is the chosen value.',
  },
  stat: {
    sanitize: (x) => {
      if (!isRecord(x)) return null;
      const label = str(x.label, 40);
      const value = str(x.value, 64);
      if (!label || value === undefined) return null;
      const block: StudioBlock = { type: "stat", label, value };
      const emph = emphOf(x);
      if (emph) block.emph = emph;
      const face = faceOf(x);
      if (face) block.face = face;
      const tone = toneOf(x);
      if (tone) block.tone = tone;
      return block;
    },
    doc: '{"type":"stat","label":"Score","value":"{{score}}","emph":"h1","tone":"accent"} — a label+value chip for live numbers; optional "emph", "face", "tone":"accent"|"danger"|"ok"|"muted".',
  },
  table: {
    sanitize: (x) => {
      if (!isRecord(x)) return null;
      const colsBind = bindString(x.cols);
      const cols = cleanStringArray(x.cols, 12, 40);
      if (!cols && !colsBind) return null;
      const rowsBind = bindString(x.rows);
      let rows: (string | number)[][] | null = null;
      if (Array.isArray(x.rows)) {
        const cleaned = x.rows
          .slice(0, 40)
          .map((r) =>
            Array.isArray(r)
              ? r.slice(0, 12).map((c) => (typeof c === "number" ? c : str(c, 80)))
              : null
          )
          .filter((r): r is (string | number)[] => r != null);
        if (cleaned.length) rows = cleaned;
      }
      if (!rows && !rowsBind) return null;
      return { type: "table", cols: cols ?? colsBind!, rows: rows ?? rowsBind! };
    },
    doc: '{"type":"table","cols":["Move","Score"],"rows":[["1","0.0"],[2,"1.5"]]} — a data table; "cols" and "rows" may each bind to arrays ("{{table.cols}}").',
  },
  image: {
    sanitize: (x) => {
      if (!isRecord(x)) return null;
      const src = str(x.src, 2_000);
      if (!/^(https?:|data:image\/)/.test(src)) return null;
      const block: StudioBlock = { type: "image", src };
      if (typeof x.alt === "string" && x.alt) block.alt = x.alt.slice(0, 120);
      const width = num(x.width, 32, 1200);
      if (width != null) block.width = width;
      return block;
    },
    doc: '{"type":"image","src":"https://…","alt":"…","width":420} — an image (http/https/data URI only).',
  },
  chat: {
    sanitize: () => ({ type: "chat" }),
    doc: '{"type":"chat"} — the live conversation itself: recent history plus a composer that sends into this conversation.',
  },
  raw: {
    sanitize: (x) => {
      if (!isRecord(x)) return null;
      const src = typeof x.src === "string" ? x.src.slice(0, 262_144) : "";
      if (!src.trim()) return null;
      const id = typeof x.id === "string" && /^[A-Za-z0-9_-]{1,40}$/.test(x.id.trim()) ? x.id.trim() : undefined;
      const rawEvents = cleanStringArray(x.events, 16, 40);
      const events = rawEvents ? rawEvents.filter((ev) => /^[A-Za-z0-9_-]{1,40}$/.test(ev)) : undefined;
      const width = num(x.width, 32, 2000);
      const onEvent = cleanAction(x.onEvent);
      const block: StudioBlock = { type: "raw", src };
      if (id) (block as { id?: string }).id = id;
      if (events?.length) (block as { events?: string[] }).events = events;
      if (width != null) (block as { width?: number }).width = width;
      if (bool(x.fill)) (block as { fill?: boolean }).fill = true;
      if (onEvent) (block as { onEvent?: StudioAction }).onEvent = onEvent;
      return block;
    },
    doc: '{"type":"raw","src":"<!doctype html>…","id":"fx1","events":["tick"],"width":400,"fill":true} — a sandboxed HTML/canvas/SVG frame (up to 128 KB). Use for custom visualizers, animations, and mini-games.',
  },
  // --- layout: full-window composition (the app owns the whole window) ---
  stack: {
    sanitize: (x, depth, opts) => {
      if (!isRecord(x)) return null;
      const blocks = cleanBlocks(x.blocks, depth, opts);
      if (!blocks) return null;
      const block: StudioBlock = { type: "stack", blocks };
      if (bool(x.fill)) block.fill = true;
      const gap = gapOf(x);
      if (gap) block.gap = gap;
      const align = alignOf(x);
      if (align) block.align = align;
      const justify = justifyOf(x);
      if (justify) block.justify = justify;
      return block;
    },
    doc: '{"type":"stack","blocks":[ … ],"fill":true,"gap":"md","align":"center","justify":"center"} — a vertical layout; "fill" grows it to fill the window height (its content then scrolls); "align"/"justify" control child placement.',
  },
  columns: {
    sanitize: (x, depth, opts) => {
      if (!isRecord(x)) return null;
      const blocks = cleanBlocks(x.blocks, depth, opts);
      if (!blocks) return null;
      const block: StudioBlock = { type: "columns", blocks };
      if (bool(x.fill)) block.fill = true;
      const gap = gapOf(x);
      if (gap) block.gap = gap;
      const align = alignOf(x);
      if (align) block.align = align;
      const justify = justifyOf(x);
      if (justify) block.justify = justify;
      const sizes = sizesOf(x);
      if (sizes) block.sizes = sizes;
      return block;
    },
    doc: '{"type":"columns","blocks":[ … ],"fill":true,"gap":"md","sizes":[2,1],"align":"stretch","justify":"start"} — components side by side (a two- or three-panel layout); "sizes":[2,1] sets relative column widths (default equal); "align"/"justify" control child placement.',
  },
  panel: {
    sanitize: (x, depth, opts) => {
      if (!isRecord(x)) return null;
      const blocks = cleanBlocks(x.blocks, depth, opts);
      if (!blocks) return null;
      const block: StudioBlock = { type: "panel", blocks };
      const title = str(x.title, 120);
      if (title) block.title = title;
      if (bool(x.fill)) block.fill = true;
      const width = num(x.width, 64, 2000);
      if (width != null) block.width = width;
      const gap = gapOf(x);
      if (gap) block.gap = gap;
      const tone = toneOf(x);
      if (tone) block.tone = tone;
      return block;
    },
    doc: '{"type":"panel","title":"…","blocks":[ … ],"fill":true,"width":260,"tone":"danger"} — a titled box grouping components; "fill" fills the height and scrolls; "width" fixes its width in px; "tone":"accent"|"danger"|"ok"|"muted" tints the border.',
  },
  toolbar: {
    sanitize: (x, depth, opts) => {
      if (!isRecord(x)) return null;
      const blocks = cleanBlocks(x.blocks, depth, opts);
      if (!blocks) return null;
      const block: StudioBlock = { type: "toolbar", blocks };
      if (bool(x.sticky)) block.sticky = true;
      const align = alignOf(x);
      if (align) block.align = align;
      const justify = justifyOf(x);
      if (justify) block.justify = justify;
      return block;
    },
    doc: '{"type":"toolbar","blocks":[ … ],"sticky":true,"align":"center","justify":"between"} — a top bar of controls (buttons, selects, stats, text, spacers); "align"/"justify" control child placement.',
  },
  list: {
    sanitize: (x, depth, opts) => {
      if (!isRecord(x)) return null;
      const each = str(x.each, 200).trim();
      if (!each) return null;
      const blocks = cleanBlocks(x.blocks, depth, opts);
      if (!blocks) return null;
      const block: StudioBlock = { type: "list", each, blocks };
      const as = str(x.as, 32).trim();
      if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(as)) block.as = as;
      const gap = gapOf(x);
      if (gap) block.gap = gap;
      return block;
    },
    doc: '{"type":"list","each":"cards","as":"card","blocks":[ … ],"gap":"md"} — repeats the template for every item of the array the expression resolves to; inside, bind with {{card.title}} ("as" defaults to "item").',
  },
  // --- app-only components: the product's own surfaces (excluded from the AI
  //     prompt; composed by hosted apps, never emitted by the model) ---
  field: {
    ai: false,
    sanitize: (x) => {
      if (!isRecord(x)) return null;
      const block: StudioBlock = { type: "field" };
      if (typeof x.label === "string" && x.label) block.label = x.label.slice(0, 80);
      if (typeof x.placeholder === "string" && x.placeholder) block.placeholder = x.placeholder.slice(0, 80);
      if (typeof x.value === "string") block.value = x.value.slice(0, 64);
      if (x.kind === "email" || x.kind === "url" || x.kind === "password") block.kind = x.kind;
      if (typeof x.key === "string" && x.key) block.key = x.key.slice(0, 64);
      if (typeof x.className === "string" && x.className) block.className = x.className.slice(0, 120);
      if (x.autoFocus === true) block.autoFocus = true;
      if (typeof x.ariaLabel === "string" && x.ariaLabel) block.ariaLabel = x.ariaLabel.slice(0, 80);
      return block;
    },
    doc: "app-only: a live (change-on-keystroke) form field.",
  },
  wallhead: {
    ai: false,
    sanitize: (x) => {
      if (!isRecord(x)) return null;
      const title = str(x.title, 200);
      if (!title) return null;
      const meta: { text: string; model?: boolean }[] = [];
      if (Array.isArray(x.meta)) {
        for (const m of x.meta.slice(0, 8)) {
          if (typeof m === "string" && m) meta.push({ text: m.slice(0, 80) });
          else if (isRecord(m) && typeof m.text === "string" && m.text) {
            const item: { text: string; model?: boolean } = { text: (m.text as string).slice(0, 80) };
            if (bool(m.model)) item.model = true;
            meta.push(item);
          }
        }
      }
      return { type: "wallhead", title, meta };
    },
    doc: "app-only: the wall header (title + meta line).",
  },
  wallabel: {
    ai: false,
    sanitize: sanitizeWallabel,
    doc: "app-only: an exchange's right-column label.",
  },
  work: {
    ai: false,
    sanitize: (x, depth, opts) => {
      if (!isRecord(x)) return null;
      const label = sanitizeWallabel(x.label);
      if (!label || label.type !== "wallabel") return null;
      const piece = sanitizeBlocks(x.piece, (depth ?? 1) + 1, opts);
      if (!piece) return null;
      const block: StudioBlock = { type: "work", label, piece };
      if (bool(x.streaming)) block.streaming = true;
      return block;
    },
    doc: "app-only: one exchange (label column + piece column).",
  },
  study: {
    ai: false,
    sanitize: (x, _depth, opts) => {
      if (!isRecord(x)) return null;
      const content = str(x.content, contentMax(opts));
      if (!content) return null;
      const block: StudioBlock = { type: "study", content, active: bool(x.active) };
      const ms = num(x.ms, 0, 10_000_000);
      if (ms != null) (block as { ms?: number }).ms = ms;
      return block;
    },
    doc: "app-only: the model's reasoning panel.",
  },
  preparing: {
    ai: false,
    sanitize: (x) => {
      const block: StudioBlock = { type: "preparing" };
      if (isRecord(x)) {
        const text = str(x.text, 120);
        if (text) block.text = text;
      }
      return block;
    },
    doc: "app-only: a 'preparing…' status line.",
  },
  status: {
    ai: false,
    sanitize: (x) => {
      if (!isRecord(x)) return null;
      const text = str(x.text, 120);
      return text ? ({ type: "status", text } as StudioBlock) : null;
    },
    doc: "app-only: the streaming reply's live status line.",
  },
  plan: {
    ai: false,
    sanitize: (x) => {
      if (!isRecord(x)) return null;
      // New shape: {todos: [{title, cot?}], active, done, pct}
      if (Array.isArray(x.todos) && x.todos.length > 0) {
        const todos: { title: string; cot?: string }[] = [];
        for (const t of x.todos as unknown[]) {
          if (t && typeof t === "object" && typeof (t as Record<string, unknown>).title === "string") {
            const td = t as Record<string, unknown>;
            const item: { title: string; cot?: string } = { title: String(td.title).slice(0, 80) };
            if (typeof td.cot === "string" && td.cot) item.cot = td.cot.slice(0, 200);
            todos.push(item);
          }
        }
        if (!todos.length) return null;
        const active = num(x.active, 0, 12) ?? 0;
        const done = num(x.done, 0, 12) ?? 0;
        const pct = typeof x.pct === "number" && x.pct >= 0 ? Math.min(x.pct, 100) : null;
        const block: StudioBlock = { type: "plan", todos, active: Math.min(active, todos.length), done: Math.min(done, todos.length), pct };
        if (bool(x.confirm)) block.confirm = true;
        if (typeof x.thinking === "string" && x.thinking) block.thinking = x.thinking.slice(0, 20_000);
        if (bool(x.thinkingActive)) block.thinkingActive = true;
        return block;
      }
      // Legacy shape: {steps: ["…"], done}
      const steps = cleanStringArray(x.steps, 12, 80);
      if (!steps) return null;
      const done = num(x.done, 0, 12) ?? 0;
      const todos = steps.map((s) => ({ title: s }));
      return { type: "plan", todos, active: Math.min(done, todos.length), done: Math.min(done, todos.length), pct: null };
    },
    doc: "app-only: a build plan with progress (todos + active + pct).",
  },
  question: {
    ai: false,
    sanitize: (x) => {
      if (!isRecord(x)) return null;
      const q = str(x.question, 500);
      if (!q) return null;
      const block: StudioBlock = { type: "question" as const, question: q };
      const options = cleanStringArray(x.options, 6, 120);
      if (options) (block as { options?: string[] }).options = options;
      const assumptions = cleanStringArray(x.assumptions, 8, 200);
      if (assumptions) (block as { assumptions?: string[] }).assumptions = assumptions;
      return block;
    },
    doc: "app-only: a one-question card with optional choices (the builder's triage).",
  },
  interrupted: {
    ai: false,
    sanitize: (x, _depth, opts) => {
      if (!isRecord(x)) return null;
      const kind = x.kind === "truncated" || x.kind === "unparseable" || x.kind === "planning" ? x.kind : null;
      if (!kind) return null;
      const block: StudioBlock = { type: "interrupted", kind };
      const raw = str(x.raw, contentMax(opts));
      if (raw) block.raw = raw;
      if (bool(x.canRetry)) (block as { canRetry?: boolean }).canRetry = true;
      return block;
    },
    doc: "app-only: a workspace-failure card (retry / show raw).",
  },
  door: {
    ai: false,
    sanitize: (x) => {
      if (!isRecord(x)) return null;
      const title = str(x.title, 200);
      const appId = str(x.appId, 64);
      if (!title || !appId) return null;
      return { type: "door", title, appId };
    },
    doc: "app-only: a door that opens another app.",
  },
  critic: {
    ai: false,
    sanitize: (x) => {
      if (!isRecord(x)) return null;
      const status =
        x.status === "reviewing" || x.status === "refining" || x.status === "refined" ? x.status : null;
      if (!status) return null;
      return { type: "critic", status };
    },
    doc: "app-only: the design critic's status line.",
  },
  activity: {
    ai: false,
    sanitize: (x, _depth, opts) => {
      if (!isRecord(x)) return null;
      const phase = str(x.phase, 60) || "Working";
      const block: StudioBlock = { type: "activity", phase, active: bool(x.active) };
      const thinking = str(x.thinking, contentMax(opts));
      if (thinking) block.thinking = thinking;
      const writing = str(x.writing, contentMax(opts));
      if (writing) block.writing = writing;
      return block;
    },
    doc: "app-only: the inline AI activity window (phase + reasoning + raw writing).",
  },
  thread: {
    ai: false,
    sanitize: (x) => {
      if (!isRecord(x)) return null;
      const title = str(x.title, 200);
      const appId = str(x.appId, 64);
      if (!title || !appId) return null;
      return { type: "thread", appId, title, status: str(x.status, 120) || "Working", active: bool(x.active) };
    },
    doc: "app-only: a build-thread card (title + live status; opens the thread).",
  },
  none: {
    ai: false,
    sanitize: (x) => {
      const text = str(isRecord(x) ? x.text : null, 60);
      return { type: "none", ...(text ? { text } : {}) };
    },
    doc: "app-only: an empty-piece placeholder.",
  },
  caret: {
    ai: false,
    sanitize: () => ({ type: "caret" }),
    doc: "app-only: a blinking text caret.",
  },
  composer: {
    ai: false,
    sanitize: () => ({ type: "composer" }),
    doc: "app-only: the docked message box (host-wired).",
  },
  chip: {
    ai: false,
    sanitize: (x) => {
      const label = str(isRecord(x) ? x.label : null, 200);
      if (!label) return null;
      const block: StudioBlock = { type: "chip", label };
      if (isRecord(x)) {
        if (bool(x.thinking)) block.thinking = true;
        const delay = num(x.delay, 0, 10_000);
        if (delay != null) (block as { delay?: number }).delay = delay;
      }
      return block;
    },
    doc: "app-only: a suggestion pill.",
  },
  dot: {
    ai: false,
    sanitize: (x) => {
      const block: StudioBlock = { type: "dot" };
      if (isRecord(x) && (x.variant === "live" || x.variant === "static" || x.variant === "replying"))
        block.variant = x.variant;
      return block;
    },
    doc: "app-only: a status dot.",
  },
  greetinghead: {
    ai: false,
    sanitize: (x, _depth, opts) => {
      if (!isRecord(x)) return null;
      const title = str(x.title, contentMax(opts));
      const sub = str(x.sub, 400);
      if (!title && !sub) return null;
      return { type: "greetinghead", title, sub };
    },
    doc: "app-only: the welcome header (an animated title line + a subtitle).",
  },
  group: {
    ai: false,
    sanitize: (x, depth, opts) => {
      if (!isRecord(x)) return null;
      const className = str(x.className, 120);
      if (!className) return null;
      const blocks = cleanBlocks(x.blocks, depth, opts);
      if (!blocks) return null;
      const block: StudioBlock = { type: "group", className, blocks };
      if (typeof x.key === "string" && x.key) block.key = x.key.slice(0, 64);
      return block;
    },
    doc: "app-only: a named group — a <div className> wrapping child blocks.",
  },
};

/** Sanitize a list of blocks (used for state bodies, docks, rows, cards). */
export function sanitizeBlocks(x: unknown, depth = 1, opts?: SanitizeOpts): StudioBlock[] | null {
  if (!Array.isArray(x)) return null;
  const cap = opts?.maxBlocks ?? MAX_BLOCKS;
  const out: StudioBlock[] = [];
  for (const raw of x.slice(0, cap)) {
    if (opts?.budget && opts.budget.left <= 0) break;
    if (!isRecord(raw) || typeof raw.type !== "string") continue;
    const def = BLOCKS[raw.type];
    if (!def || (def.ai === false && !opts?.allowAppOnly)) continue;
    const block = def.sanitize(raw, depth, opts);
    if (block) {
      // Any block may carry an `if` condition (declarative apps evaluate it;
      // other brains ignore it).
      const cond = str(raw.if, 200);
      if (cond) block.if = cond;
      // A shared budget spans every list in the spec (states, docks, nesting).
      if (opts?.budget) opts.budget.left -= 1;
      out.push(block);
    }
  }
  return out.length ? out : null;
}

function sanitizeContainer(x: unknown, type: "row" | "card", depth: number | undefined, opts?: SanitizeOpts): StudioBlock | null {
  if (!isRecord(x)) return null;
  if (depth != null && depth > MAX_NESTING) return null;
  const blocks = sanitizeBlocks(x.blocks, (depth ?? 1) + 1, opts);
  if (!blocks) return null;
  if (type === "row") {
    const block: StudioBlock = { type: "row", blocks };
    const align = alignOf(x);
    if (align) block.align = align;
    const justify = justifyOf(x);
    if (justify) block.justify = justify;
    return block;
  }
  return { type, blocks } as StudioBlock;
}

/** The prompt's component vocabulary, generated from the registry. */
export function blockDocs(): string[] {
  return Object.values(BLOCKS)
    .filter((d) => d.ai !== false)
    .map((d) => "- " + d.doc);
}
