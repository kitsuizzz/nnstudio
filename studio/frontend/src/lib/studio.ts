/**
 * The studio protocol: when the AI decides a request is better served by a
 * purpose-built environment than by text, it emits one studio block in its
 * reply. The client strips the block from the visible text, buffers it while
 * it streams in, and turns it into a navigable app — a declarative state
 * machine built from the shared component library, so every AI-built app
 * looks and feels like the product itself.
 *
 * Models are sloppy: they emit the spec bare, under a different fence label
 * (```json), with reordered keys, or with "states" as an array. The resolver
 * below is shape-agnostic — it hunts for a spec-shaped JSON object anywhere
 * in the reply and buffers it while it streams, so raw JSON never reaches
 * the wall.
 */

import type { StudioLogEntry, StudioSpec, StudioState } from "../types";
import { blockDocs, sanitizeBlocks, MAX_TOTAL_BLOCKS, type SanitizeOpts } from "./studio-blocks";
import { formatLogLine } from "./studio-log";

export const STUDIO_FENCE = "```studio";

export interface StudioSplit {
  /** Reply text with any studio block removed. */
  visible: string;
  /** Complete block body (JSON text), or null while it is still streaming. */
  block: string | null;
  /** A spec is mid-stream; the caller should keep buffering. */
  open: boolean;
  /** Set when a spec-shaped object is present but could not be read. */
  failed?: "truncated" | "unparseable";
}

interface Fence {
  openAt: number;
  bodyStart: number;
  closeAt: number; // -1 while still streaming
  label: string;
}

type SpecHunt =
  | { kind: "none" }
  | { kind: "open"; start: number }
  | { kind: "parsed"; start: number; end: number; block: string }
  | { kind: "failed"; start: number; end: number };

export function splitStudioBlock(text: string): StudioSplit {
  // A spec may live inside any code fence (the prompt says ```studio, but
  // models say ```json or nothing). Check each fence first so the fence
  // markers themselves never leak onto the wall.
  for (const f of findFences(text)) {
    const body = text.slice(f.bodyStart, f.closeAt === -1 ? text.length : f.closeAt);
    const hunt = huntSpec(body);
    if (hunt.kind === "parsed") {
      const before = text.slice(0, f.openAt).trimEnd();
      // The fence may never close (the model stopped before the closing
      // marker) — the block is the rest of the text, nothing follows it.
      const after = f.closeAt === -1 ? "" : text.slice(f.closeAt + 3).trimStart();
      let visible = [before, after].filter(Boolean).join("\n");
      // The model sometimes repeats the spec outside the fence — strip it.
      const leftover = huntSpec(visible);
      if (leftover.kind === "parsed") visible = stripRange(visible, leftover).trim();
      return { visible, block: hunt.block, open: false };
    }
    if (f.closeAt === -1 && (hunt.kind === "open" || f.label === "studio")) {
      // The block is still streaming in; buffer instead of showing JSON.
      return { visible: text.slice(0, f.openAt).trimEnd(), block: null, open: true };
    }
  }
  const hunt = huntSpec(text);
  if (hunt.kind === "parsed") {
    return { visible: stripRange(text, hunt).trim(), block: hunt.block, open: false };
  }
  if (hunt.kind === "open") {
    return { visible: text.slice(0, hunt.start).trimEnd(), block: null, open: true };
  }
  if (hunt.kind === "failed") {
    // A spec is there but unreadable: surface the model's words, flagged.
    return { visible: text, block: null, open: false, failed: "unparseable" };
  }
  return { visible: text, block: null, open: false };
}

function stripRange(text: string, hunt: Extract<SpecHunt, { kind: "parsed" }>): string {
  let after = text.slice(hunt.end + 1);
  // A lone stray quote after the spec (the model "closing" something it
  // never opened) is dropped with the block.
  if (/^"+$/.test(after.trim())) after = "";
  return `${text.slice(0, hunt.start)}${after}`;
}

/** All code fences in the text, paired in order; the last one may be open. */
function findFences(text: string): Fence[] {
  const fences: Fence[] = [];
  let i = 0;
  for (;;) {
    const at = text.indexOf("```", i);
    if (at === -1) return fences;
    const lineEnd = text.indexOf("\n", at);
    if (lineEnd === -1) {
      fences.push({ openAt: at, bodyStart: at + 3, closeAt: -1, label: text.slice(at + 3).trim() });
      return fences;
    }
    const label = text.slice(at + 3, lineEnd).trim();
    const bodyStart = lineEnd + 1;
    const close = /^[ \t]*```/m.exec(text.slice(bodyStart));
    const closeAt = close ? bodyStart + close.index : -1;
    fences.push({ openAt: at, bodyStart, closeAt, label });
    if (closeAt === -1) return fences;
    i = closeAt + 3;
  }
}

/**
 * Hunts for a spec-shaped JSON object anywhere in the text. The anchor is any
 * top-level spec key ("title", "initial" or "states" — in whichever order the
 * model wrote them); the candidate is the nearest enclosing object, found by
 * depth-tracking backwards. A balanced candidate that parses wins; a balanced
 * one that does not parse (or is a legitimate JSON answer, not a spec) is
 * skipped; an unbalanced one means the spec is mid-stream, so it is buffered.
 */
function huntSpec(text: string): SpecHunt {
  const re = /"(title|initial|states)"\s*:/g;
  let failed: { start: number; end: number } | null = null;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    const start = findEnclosingOpen(text, m.index);
    if (start === -1) {
      re.lastIndex = m.index + 1;
      continue;
    }
    const end = matchBrace(text, start);
    if (end === -1) return { kind: "open", start };
    const candidate = text.slice(start, end + 1);
    if (parseStudioBlock(candidate)) {
      return { kind: "parsed", start, end, block: candidate };
    }
    // A spec-shaped object that will not parse: remember it, keep looking
    // (the model sometimes drafts, breaks, then writes it right).
    if (candidate.includes('"states"')) failed = { start, end };
    re.lastIndex = end + 1;
  }
  return failed ? { kind: "failed", ...failed } : { kind: "none" };
}

/**
 * Nearest `{` enclosing position `from`, found by scanning left with depth
 * tracking — correct no matter which key the model happened to write first.
 */
function findEnclosingOpen(text: string, from: number): number {
  let depth = 0;
  for (let i = from - 1; i >= 0; i--) {
    const ch = text[i];
    if (ch === "}" || ch === "]") depth += 1;
    else if (ch === "{") {
      if (depth === 0) return i;
      depth -= 1;
    } else if (ch === "[" && depth > 0) depth -= 1;
  }
  return -1;
}

/**
 * One string-aware repair pass for the slips small models make: raw control
 * characters inside string literals (real line breaks in a board or code
 * block) and trailing commas. Returns text that JSON.parse will accept if
 * the JSON is salvageable.
 */
function repairJsonSlips(json: string): string {
  let out = "";
  let inString = false;
  let escaped = false;
  for (let i = 0; i < json.length; i += 1) {
    const ch = json[i];
    if (inString) {
      if (escaped) {
        escaped = false;
        out += ch;
      } else if (ch === "\\") {
        escaped = true;
        out += ch;
      } else if (ch === '"') {
        inString = false;
        out += ch;
      } else if (ch === "\n") out += "\\n";
      else if (ch === "\t") out += "\\t";
      else if (ch === "\r") out += "\\r";
      else if (ch.charCodeAt(0) < 0x20) out += "\\u" + ch.charCodeAt(0).toString(16).padStart(4, "0");
      else out += ch;
    } else if (ch === '"') {
      inString = true;
      out += ch;
    } else out += ch;
  }
  return out.replace(/,(\s*[}\]])/g, "$1");
}

/** Last `}` matching the `{` at `start`, respecting JSON string literals. */
function matchBrace(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

const MAX_SPEC_CHARS = 260_000;
const MAX_STATES = 32;

function isRecord(x: unknown): x is Record<string, unknown> {
  return !!x && typeof x === "object" && !Array.isArray(x);
}

function cleanString(x: unknown, max: number): string {
  return typeof x === "string" ? x.slice(0, max) : "";
}

const MAX_DATA_KEYS = 512;
const MAX_LOGIC_CHARS = 256_000;

function sanitizeState(x: unknown, opts?: SanitizeOpts): StudioState | null {
  if (!isRecord(x)) return null;
  const blocks = sanitizeBlocks(x.blocks, 1, opts);
  if (!blocks) return null;
  const state: StudioState = { blocks };
  if (typeof x.after === "number" && x.after >= 0) state.after = Math.min(x.after, 60_000);
  if (typeof x.next === "string") state.next = x.next.slice(0, 64);
  const dock = sanitizeBlocks(x.dock, 1, opts);
  if (dock) state.dock = dock;
  return state;
}

/** Deep-clone the app's initial data with hard caps (no functions, no depth bombs). */
/** Deep-clone an array of data with hard caps (records and nested arrays included). */
function cloneArray(x: unknown[], depth: number): unknown[] {
  const arr: unknown[] = [];
  for (const item of x.slice(0, 512)) {
    if (typeof item === "string") arr.push(item.slice(0, 1_000));
    else if (typeof item === "number" && Number.isFinite(item)) arr.push(item);
    else if (typeof item === "boolean" || item === null) arr.push(item);
    else if (Array.isArray(item)) {
      if (depth < 8) arr.push(cloneArray(item, depth + 1));
    } else if (isRecord(item)) {
      const sub = cloneData(item, depth + 1);
      if (sub) arr.push(sub);
    }
  }
  return arr;
}

function cloneData(x: unknown, depth = 0): Record<string, unknown> | null {
  if (!isRecord(x) || depth > 8) return null;
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(x)) {
    if (Object.keys(out).length >= MAX_DATA_KEYS) break;
    const key = k.slice(0, 64);
    if (typeof v === "string") out[key] = v.slice(0, 10_000);
    else if (typeof v === "number" && Number.isFinite(v)) out[key] = v;
    else if (typeof v === "boolean") out[key] = v;
    else if (v === null) out[key] = null;
    else if (Array.isArray(v)) out[key] = cloneArray(v, depth);
    else if (isRecord(v)) {
      const sub = cloneData(v, depth + 1);
      if (sub) out[key] = sub;
    }
  }
  return out;
}

export interface ParsedStudio {
  title: string;
  spec: StudioSpec;
}

/**
 * Lenient by design: unknown blocks are dropped, not fatal. The app is
 * rejected only when nothing renderable survives — or when it is a logic
 * workspace (a sandboxed JS brain driving the same components).
 */
export function parseStudioBlock(json: string): ParsedStudio | null {
  if (json.length > MAX_SPEC_CHARS) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch {
    // Small models slip: raw line breaks inside strings, trailing commas.
    // One repair pass, then give up.
    try {
      parsed = JSON.parse(repairJsonSlips(json));
    } catch {
      return null;
    }
  }
  if (!isRecord(parsed)) return null;
  const title = cleanString(parsed.title, 120);
  if (!title) return null;

  // Logic mode: the model's JS runs in a sandboxed iframe and drives the
  // shared component library through a postMessage bridge.
  if (typeof parsed.logic === "string" && parsed.logic.trim()) {
    const spec: StudioSpec = { title, logic: parsed.logic.slice(0, MAX_LOGIC_CHARS) };
    // The optional in-app AI role: enables ai.ask and drives the user's
    // one-time consent. Both halves are required for it to mean anything.
    if (isRecord(parsed.ai)) {
      const role = cleanString(parsed.ai.role, 60);
      const system = cleanString(parsed.ai.system, 2000);
      if (role && system) spec.ai = { role, system };
    }
    return { title, spec };
  }

  const states: Record<string, StudioState> = {};
  // One shared budget for the whole spec: every accepted block (nested lists
  // included) spends from it, so a small spec can't balloon at render time.
  const opts: SanitizeOpts = { budget: { left: MAX_TOTAL_BLOCKS } };
  if (isRecord(parsed.states)) {
    for (const [key, value] of Object.entries(parsed.states)) {
      if (Object.keys(states).length >= MAX_STATES) break;
      const state = sanitizeState(value, opts);
      if (state) states[key.slice(0, 64)] = state;
    }
  } else if (Array.isArray(parsed.states)) {
    // Some models emit states as an array; map the entries to s0..sN.
    parsed.states.slice(0, MAX_STATES).forEach((value, i) => {
      const state = sanitizeState(value, opts);
      if (state) states[`s${i}`] = state;
    });
  }
  if (!Object.keys(states).length) return null;

  let initial = typeof parsed.initial === "string" ? parsed.initial.slice(0, 64) : "";
  if (!states[initial]) initial = Object.keys(states)[0];

  const spec: StudioSpec = { title, initial, states };
  if (isRecord(parsed.data)) {
    const data = cloneData(parsed.data);
    if (data) spec.data = data;
  }
  return { title, spec };
}

const FENCE = "```studio";

/** The seven non-negotiable design rules, shared by the builder and the critic. */
function designFloorLines(): string[] {
  return [
    "1. Focal point — one element per screen carries the most visual weight (largest emph, accent tone, or dominant size). The eye must land somewhere specific.",
    "2. Aligned — never default to a 50/50 split. Use \"sizes\" on columns to express the actual relationship (2:1, 3:1, 7:3). Equal widths only when the content is truly equal.",
    "3. Type hierarchy — vary \"emph\" across the screen: at most one display or h1, then h2 for sections, body for reading, small/caption for metadata. Flat type (everything the same size) is a failure.",
    '4. Meaningful accent — the "accent" tone is the single loudest color in the palette (a warm red dot on off-white). Use it on 1–2 elements per screen max: the focal stat, the primary button, or a key panel border. If everything is accented, nothing is.',
    "5. Fills the window — the root layout (stack or columns) must have \"fill\": true. The app owns the full viewport; never emit a small centered card floating in empty space.",
    "6. Grouped — related blocks live inside a panel or card with a short title. Loose, ungrouped blocks floating in a stack read as an accident, not a layout.",
    '7. Legible — body text stays at emph "body" (the default 17px serif). Keep line lengths under ~70 chars (a panel width of 480–680px is the sweet spot). Use face "mono" only for data, code, or numbers — never for paragraphs.',
  ];
}

/** The planner's system prompt. Marked so the mock provider can route on it. */
export function planPrompt(editContext?: { title: string; change: string }): string {
  const isEdit = editContext != null;
  const countRange = isEdit ? "2–4" : "3–5";
  return [
    "===PLANNER===",
    "",
    isEdit
      ? `You are a planning specialist. You are planning an EDIT to an existing workspace called "${editContext.title}". The requested change: "${editContext.change}". Do NOT build it. Do NOT emit a studio block. Do NOT ask questions.`
      : "You are a planning specialist. You receive a brief — a one-line description of a workspace to build. Do NOT build it. Do NOT emit a studio block. Do NOT ask questions.",
    "",
    `Your job: break the ${isEdit ? "edit" : "brief"} into ${countRange} concrete, ordered TODOs that a builder will execute. Each TODO has:`,
    '- "title": a short phrase (3–8 words) naming the step — e.g. "Pick the layout", "Write the questions", "Wire the scoring".',
    '- "cot": one sentence of chain-of-thought explaining what the builder will actually do for this step (the user sees this while the step is active).',
    "",
    "The LAST todo MUST be titled \"Visual finalize\" with a cot like \"Review every screen for layout balance, spacing, alignment, and contrast; fix any issues before showing the user.\" This step is mandatory — the builder reviews and polishes the spec before it is shown.",
    "",
    "Emit your output as a single ```plan fence and nothing else:",
    "```plan",
    '{"todos": [{"title": "…", "cot": "…"}, {"title": "…", "cot": "…"}]}',
    "```",
    "",
    "No other fences, no prose, no studio block. Just the plan fence.",
  ].join("\n");
}

/** The builder's system prompt. Marked so the mock provider can route on it. */
export function builderPrompt(todos?: { title: string; cot?: string }[]): string {
  const todoSection = todos && todos.length
    ? [
        "Your plan (already shown to the user — do NOT re-emit a plan fence):",
        "```plan",
        JSON.stringify({ todos }, null, 2),
        "```",
        "",
      ]
    : [];

  return [
    "===BUILDER===",
    "",
    "You are a focused workspace builder. You receive a brief — a one-line description of what to build. Do not ask questions. Do not emit questions or build fences. Do not emit a plan fence (the plan is already shown).",
    "",
    "You build a workspace: a real, interactive APP that takes over the entire window. The host gives it its own top bar (title + exit); you design the body, which should fill the whole window — full width AND full height. It is assembled from the studio's own shared components — the same components the product itself is made of.",
    "",
    "You also have web tools (web_search / web_fetch). Use them only when the app genuinely needs a real fact or URL you cannot invent; each call pauses for the user's approval, so never plan a build around one — if it is declined, continue with sensible local defaults.",
    "",
    ...todoSection,
    "Your output protocol (follow exactly):",
    "",
    "Progress (one fence per TODO — drives the live progress bar):",
    '```progress\n{"active": N, "pct": P}\n```\n',
    "Where N is the 0-based index of the TODO you are NOW working on and P is your best estimate of overall completion (0–100).",
    "After you finish each TODO you MUST emit a progress fence marking it done and announcing the next one. For a plan of N todos you will emit exactly N+1 progress fences: one before you start (active=0), one after each completion (active=1, active=2, …, active=N-1), and one final (active=N, pct=100) immediately before the studio block. Skipping a TODO or jumping more than one index forward in a single fence is a protocol violation.",
    "",
    "Visual finalize (your last TODO — mandatory): Before emitting the studio block you MUST work through the \"Visual finalize\" step. Inspect every screen in your spec: check layout balance, spacing consistency, alignment, type hierarchy, and that no element overflows or is orphaned. Fix any issues directly in the spec. The host renders every screen and sends the screenshots back for a review pass after you emit — so make this first pass your best, complete composition; the review exists to catch what you missed, not to design for you. Only after this step is complete (and you have emitted its progress fence) may you emit the studio fence.",
    "",
    "Studio block (the actual spec — must be the last thing you emit):",
    FENCE,
    '{"title": "3 to 8 words", "initial": "s0", "states": { … }, "data": { … }}',
    "```",
    "",
    "Two ways to build one — choose by how much logic the app needs:",
    "",
    "A) DECLARATIVE (default): a state machine plus data and actions. Use for guided flows, demos, tutorials, and visualizers where no computation is needed.",
    '   - "initial": the id of the first state. "states": an object of states; each state is a complete screen: "blocks", optional "after" (ms before auto-advancing) + "next", and optional "dock" (primary actions pinned in a footer bar).',
    '   - "data": the app\'s mutable state (plain JSON). Bind it anywhere a string appears with {{expression}}.',
    "   - Expressions: paths plus arithmetic (+ - * / %), comparisons (== != < <= > >=), logic (&& || !), parentheses, and functions (len, min, max, abs, round, floor, ceil, number, str, upper, lower, join, contains). Examples: \"{{score + 1}}\", \"{{len(cards)}} left\", \"{{score >= 3}}\". A string that is only one {{expression}} keeps its type (numbers stay numbers).",
    '   - Conditions: any block may carry "if": "expression" — it renders only when the expression is truthy (0, "", empty arrays and missing values are false). Use it instead of pre-baking near-duplicate states.',
    '   - Lists: {"type":"list","each":"cards","as":"card","blocks":[ … ]} repeats its template for every item of the array the expression resolves to; inside, bind with {{card.title}} ("as" defaults to "item").',
    "   - Actions (closed vocabulary): {\"set\":{\"path\":\"{{expression}} or literal\"},\"inc\":{\"path\":1},\"push\":{\"path\":\"{{value}}\"},\"remove\":{\"path\":\"index or value\"},\"toggle\":[\"path\"],\"reset\":[\"path\"],\"to\":\"state\"}. Chain several with {\"actions\":[ … ]}. Event arguments: a grid cell tap passes its index, a choice passes the option value, an input submit passes the entered value (bind with {{arg}}).",
    "   - Transitions: a button's \"to\"; a state's \"after\"+\"next\" timer. Never reference a state id you did not define.",
    "",
    "B) LOGIC: a sandboxed JavaScript brain. Use when the app needs real computation, user-driven state, or game rules (e.g. a playable game).",
    '   - Spec shape: {"title":"…","logic":"<js>"} — the JS is a JSON string. Add an optional "ai" field only when the app puts the model inside itself (see below): {"ai":{"role":"chess opponent","system":"You are a strong chess opponent…"}}.',
    "   - Bridge API: ai.render({ blocks: [ … ] }) shows a screen (any components below); ai.on(\"cell\"|\"choose\"|\"click\"|\"submit\", (arg) => …) receives user input (arg = cell index / option value / button label / entered value).",
    "   - You own the app's state in your own variables; call ai.render after every change. No DOM, no storage; the only output is ai.render.",
    '   - The declarative helpers ("if", "list", {{expressions}}) are ignored in logic mode: compute everything in JS and render final strings.',
    '   - AI inside the app: ai.ask({ input: "…", json?: true }) returns a Promise resolving to the model\'s reply — the parsed object when "json": true. The model answers as the "ai.role" declared in the spec, using its "system" as the default prompt. Pass the current state in "input" (keep it under ~8 KB). The user is asked once per workspace to allow the AI, and no provider may be configured — ALWAYS catch the rejection and continue with a local fallback.',
    '   - AI opponent shortcut: ai.fetch("move", { board: [ … ] }) resolves to a valid cell index for a 9-cell board game. Prefer ai.ask for anything richer (chess, judges, quizmasters, coaches, NPCs).',
    '   - When the user asks for the AI to play with or against them, build it in: declare "ai" and call ai.ask/ai.fetch from your handlers — never fake the opponent with a static heuristic when the user asked for the real model.',
    "",
    "Component vocabulary (the ONLY components that exist):",
    ...blockDocs(),
    "",
    "Design floor (non-negotiable — every screen must satisfy all seven):",
    ...designFloorLines(),
    "",
    "Layout patterns (pick the one that fits the app — do not invent a structure):",
    "- Workbench: toolbar on top, then a fill stack for the working area. Tools, editors, simulators.",
    "- Sidebar + detail: a narrow 1–2 column beside a 3–4 detail pane. Lists, logs, settings, pickers.",
    "- Hero + grid: one accented focal panel or stat, secondary panels in a grid below. Summaries, dashboards.",
    "- Guided flow: one 520–680px column that changes state, centered horizontally. Quizzes, wizards, tutorials.",
    "- Board + HUD: the board fills the majority, a narrow side panel holds scores and controls. Games.",
    "",
    "Palette (fixed — you cannot invent colors):",
    "- Wall (background): warm off-white #fafaf7 (dark mode: #1a1a19).",
    "- Ink (text): near-black #161616 (dark: #ebebe7).",
    "- Accent: a single warm red #e5321b (dark: #ff5a45) — used via tone:\"accent\".",
    "- Allies: green #3f7a3a (tone:\"ok\"), red-orange #c8321c (tone:\"danger\"), warm gray #8a8a86 (tone:\"muted\").",
    "- Hairlines: #e3e3de / #cfcfc9 for borders and dividers.",
    "You assign tone to components; you never write hex colors.",
    "",
    "Rules:",
    "- Build real UI, not documents: a board is a grid, a pick is a choice, a live number is a stat, a user value is an input, data is a table. Mutable things live in \"data\" and are mutated by actions — only fall back to a sequence of pre-baked states when nothing is user-driven.",
    "- Prefer live bindings over pre-baked states: derive values with {{expressions}}, hide blocks with \"if\", and repeat templates with \"list\". States are for navigation, not for data variants.",
    "- Shape every screen: open with a toolbar (its label/actions), then a main area that fills the rest — a stack (fill:true) for one scrolling column, or columns (fill:true) holding two or three side-by-side panels. Group related things in a panel with a short title. Fill the width and the height of the window; never emit one narrow centered column of bare blocks.",
    "- The block body is strict JSON: one object, nothing else. Escape strings correctly (\\\" for quotes, \\n for newlines). No trailing commas.",
    "- Multi-line content (boards, code, lists) must use \\n escapes inside its string value — never a raw line break inside a JSON string.",
    '- Emit the top-level keys in exactly this order: "title", "initial", "states", "data" (or "title", "logic").',
    "- The studio block is the ONLY place the JSON may appear. Never write, repeat, or preview the spec outside the ```studio fence — not before it, not after it, not as an example.",
    "- Progress fences come first (one per stage boundary), then the studio block, then stop.",
    '- Use "chat" when the app should let the user ask the AI follow-up questions in context.',
    "- Teach in person: narrate what is happening in short text, and use timers (\"after\" around 900–1500ms) to pace a demonstration.",
    "- Keep it tight: declarative — at most 16 states, at most 80 blocks per screen, at most 30 KB; logic — at most 256 KB of JS.",
    "- Do not use components, fields, or JSON keys outside the vocabulary above.",
  ].join("\n");
}

/** The builder's system prompt for EDITING an existing workspace. Marked so the mock provider can route on it. */
export function builderEditPrompt(
  existingSpec: StudioSpec,
  change: string,
  todos?: { title: string; cot?: string }[],
  logs?: StudioLogEntry[]
): string {
  const specJson = JSON.stringify(existingSpec, null, 2);
  const todoSection = todos && todos.length
    ? [
        "Your plan (already shown to the user — do NOT re-emit a plan fence):",
        "```plan",
        JSON.stringify({ todos }, null, 2),
        "```",
        "",
      ]
    : [];
  // The workspace's captured runtime output: errors and console lines the app
  // produced while it ran. The AI is the only one who can fix them.
  const errorLines = (logs ?? []).slice(-20).map(formatLogLine);
  let errorText = errorLines.join("\n");
  if (errorText.length > 4000) errorText = errorText.slice(-4000);
  const errorSection = errorLines.length
    ? [
        "",
        "Runtime output observed while this app was running (most recent last) — if any of these are real errors, fix them as part of the edit:",
        errorText,
      ]
    : [];

  return [
    "===BUILDER-EDIT===",
    "",
    "You are editing an existing workspace. The current spec is between the markers below. Do not ask questions. Do not emit questions or build fences. Do not emit a plan fence (the plan is already shown).",
    "",
    "You also have web tools (web_search / web_fetch). Use them only when the edit genuinely needs a real fact or URL you cannot invent; each call pauses for the user's approval, so never depend on one — if it is declined, keep the current behaviour.",
    "",
    ...todoSection,
    "The user's request: \"" + change + "\".",
    "The user may have attached a screenshot showing the current visual state — use it to understand what they see and what is wrong.",
    ...errorSection,
    "",
    "===SPEC-BEGIN===",
    specJson,
    "===SPEC-END===",
    "",
    "Produce the complete updated spec in a single ```studio fence. Keep everything that does not need changing. Do not restructure the whole app unless the request demands it.",
    "",
    "Your output protocol (follow exactly):",
    "",
    "Progress (one fence per TODO — drives the live progress bar):",
    '```progress\n{"active": N, "pct": P}\n```\n',
    "Where N is the 0-based index of the TODO you are NOW working on and P is your best estimate of overall completion (0–100).",
    "After you finish each TODO you MUST emit a progress fence marking it done and announcing the next one. Skipping a TODO or jumping more than one index forward is a protocol violation.",
    "",
    "Visual finalize (your last TODO — mandatory): Before emitting the studio block you MUST work through the \"Visual finalize\" step. Inspect every screen in your updated spec: check layout balance, spacing consistency, alignment, type hierarchy, and that no element overflows or is orphaned. Fix any issues directly in the spec. Only after this step is complete may you emit the studio fence.",
    "",
    "Studio block (the updated spec — must be the last thing you emit):",
    FENCE,
    '{"title": "…", "initial": "s0", "states": { … }, "data": { … }}',
    "```",
    "",
    "Rules (same as normal building):",
    '- Logic apps may declare an "ai" role and call ai.ask / ai.fetch("move") — keep both working when you edit them (the user has already been asked to allow the AI).',
    "- Strings may bind data with {{expressions}} (arithmetic, comparisons, logic, len/join/contains…); any block may carry \"if\" to render conditionally; \"list\" repeats a template over an array. Keep existing bindings working.",
    "- The block body is strict JSON: one object, nothing else. Escape strings correctly (\\\" for quotes, \\n for newlines). No trailing commas.",
    "- Multi-line content must use \\n escapes inside its string value — never a raw line break inside a JSON string.",
    '- Emit the top-level keys in exactly this order: "title", "initial", "states", "data" (or "title", "logic").',
    "- The studio block is the ONLY place the JSON may appear.",
    "- Do not use components, fields, or JSON keys outside the vocabulary above.",
    ...blockDocs(),
  ].join("\n");
}

/**
 * The App Builder's self-review: after the spec lands, the host renders every
 * screen and hands the builder the screenshots + measured layout facts. The
 * builder answers APPROVED, or rewrites the spec to fix what it actually sees.
 * Marked so the mock provider can route on it.
 */
export function workspaceReviewPrompt(
  spec: StudioSpec,
  pass: number,
  passes: number,
  diagnostics: string[]
): string {
  const specJson = JSON.stringify(spec, null, 2);
  return [
    "===APP-REVIEW===",
    "",
    `You are the App Builder reviewing your own finished app as it actually renders (visual review pass ${pass} of ${passes}). You receive one screenshot per captured screen, plus layout diagnostics measured from the live DOM.`,
    "",
    "The design floor every screen must satisfy:",
    ...designFloorLines(),
    "",
    "How to review:",
    "- Judge only what the screenshots and diagnostics show: layout, hierarchy, spacing, alignment, contrast, overflow, crowding. No feature requests, no praise, no restating the app.",
    "- An overflowing, clipped, squashed, unreadable or empty-looking screen is a failure.",
    "- If every screen satisfies the floor, reply with exactly APPROVED and nothing else.",
    "- Otherwise reply with the complete corrected spec in a single ```studio fence and nothing else. Fix the problems in place: never add features, never change the app's purpose, never drop working states.",
    diagnostics.length ? `\nLayout diagnostics from this render:\n- ${diagnostics.join("\n- ")}` : "",
    "",
    "The exact spec you are reviewing (your correction is a replacement for it):",
    "===SPEC-BEGIN===",
    specJson,
    "===SPEC-END===",
  ].join("\n");
}

/**
 * The design critic's system prompt. Marked so the mock provider can route on
 * it. The critic sees the contact sheet of the finished workspace — one image
 * per captured screen — and answers APPROVED, or a bounded list of concrete
 * in-vocabulary defects that a polish edit can act on.
 */
export function criticPrompt(): string {
  return [
    "===CRITIC===",
    "",
    "You are a senior product designer reviewing one or more screenshots of a finished workspace (one image per captured screen): an app assembled from a fixed component library (panels, rows, columns, stacks, headings, text, stats, tables, buttons…). Judge only what is visible.",
    "",
    "The design floor every screen must satisfy:",
    ...designFloorLines(),
    "",
    "Reply with exactly APPROVED when the design floor is met and nothing is worth fixing.",
    "Otherwise reply with a numbered list of at most 5 concrete defects. For each: name the visible element, say what is wrong, and name the fix in the component vocabulary (\"fill\": true, \"sizes\": [2,1], emph, face, tone, align, justify, gap, panel, toolbar, stack, columns).",
    "Rules: no praise, no restating the app, no feature requests, no questions. Review layout, hierarchy, spacing, alignment, contrast, and overflow only.",
  ].join("\n");
}
