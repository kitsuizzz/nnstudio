/**
 * Parses ```plan, ```progress, ```questions, and ```build fences from the AI's
 * streamed reply. The AI emits these to report its design TODOs, track
 * completion, ask clarifying questions, or delegate to the builder.
 *
 * New protocol (two-phase builder):
 *
 * ```plan
 * {"todos": [{"title": "Pick layout", "cot": "…"}, …]}
 * ```
 *
 * ```progress
 * {"active": 2, "pct": 67}
 * ```
 *
 * Legacy protocol (still accepted for backward compatibility):
 *
 * ```plan
 * {"steps": ["Analyze the request", "Design the layout"]}
 * ```
 *
 * ```progress
 * {"done": 2}
 * ```
 *
 * ```questions
 * {"question":"…","options":["…"],"assumptions":["…"]}
 * ```
 *
 * ```build
 * {"topic":"…"}
 * ```
 *
 * The frontend strips these fences from the visible text (same as studio
 * blocks) and renders a two-sided TODO list + progress on the wall.
 */

import type { EditFence, PlanTodo } from "../types";

export interface PlanState {
  todos: PlanTodo[];
  /** 0-based index of the currently active todo. */
  active: number;
  /** Number of completed todos (= active in the new protocol). */
  done: number;
  /** Self-reported percentage 0–100, or null when not yet reported. */
  pct: number | null;
}

export interface QuestionState {
  question: string;
  options?: string[];
  assumptions?: string[];
}

export interface ParsedFences {
  visible: string;
  plan: PlanState | null;
  /** A progress-only report: the builder streams progress fences without
      re-emitting the plan, so the caller merges these into the plan it has. */
  progress: { active: number | null; done: number | null; pct: number | null } | null;
  question: QuestionState | null;
  buildTopic: string | null;
  edit: EditFence | null;
  /** A fence never closed (the model stopped mid-fence). */
  truncated: boolean;
  /** A fence closed but its body did not parse — it was left visible. */
  failed: boolean;
}

interface FenceSpan {
  label: string;
  openAt: number;
  bodyStart: number;
  closeAt: number; // -1 if still open
}

export function splitPlanBlocks(text: string): ParsedFences {
  const fences = findPlanFences(text);
  if (fences.length === 0) {
    return { visible: text, plan: null, progress: null, question: null, buildTopic: null, edit: null, truncated: false, failed: false };
  }

  let todos: PlanTodo[] | null = null;
  let active = 0;
  let done = 0;
  let pct: number | null = null;
  let progressActive: number | null = null;
  let progressDone: number | null = null;
  let progressPct: number | null = null;
  let sawProgress = false;
  let question: QuestionState | null = null;
  let buildTopic: string | null = null;
  let edit: EditFence | null = null;
  let visible = text;
  let truncated = false;
  let failed = false;

  // Process in reverse so earlier indices remain valid after removal.
  for (let i = fences.length - 1; i >= 0; i--) {
    const f = fences[i];
    const body = visible.slice(f.bodyStart, f.closeAt === -1 ? visible.length : f.closeAt);
    let parsedOk = false;

    if (f.label === "plan") {
      try {
        const parsed = JSON.parse(body.trim()) as Record<string, unknown>;
        if (Array.isArray(parsed.todos) && parsed.todos.length > 0) {
          // New format: {todos: [{title, cot?}]}
          const items: PlanTodo[] = [];
          for (const t of parsed.todos as unknown[]) {
            if (t && typeof t === "object" && typeof (t as PlanTodo).title === "string") {
              const td = t as PlanTodo;
              items.push(td.cot ? { title: td.title, cot: td.cot } : { title: td.title });
            }
          }
          if (items.length > 0) {
            todos = items;
            parsedOk = true;
          }
        } else if (Array.isArray(parsed.steps) && parsed.steps.length > 0) {
          // Legacy format: {steps: ["…"]}
          const items: PlanTodo[] = [];
          for (const s of parsed.steps as unknown[]) {
            if (typeof s === "string") items.push({ title: s });
          }
          if (items.length > 0) {
            todos = items;
            parsedOk = true;
          }
        }
      } catch {
        // Malformed plan fence — left visible below.
      }
    } else if (f.label === "progress") {
      try {
        const parsed = JSON.parse(body.trim()) as Record<string, unknown>;
        if (typeof parsed.active === "number" && parsed.active >= 0) {
          active = parsed.active;
          progressActive = parsed.active;
          sawProgress = true;
          parsedOk = true;
        }
        if (typeof parsed.pct === "number" && parsed.pct >= 0) {
          pct = Math.min(parsed.pct, 100);
          progressPct = pct;
          sawProgress = true;
          parsedOk = true;
        }
        // Legacy: {done: N} — maps to both active and done.
        if (typeof parsed.done === "number" && parsed.done >= 0) {
          done = parsed.done;
          progressDone = parsed.done;
          sawProgress = true;
          parsedOk = true;
          if (typeof parsed.active !== "number") active = parsed.done;
        }
      } catch {
        // Malformed progress fence — left visible below.
      }
    } else if (f.label === "questions") {
      try {
        const parsed = JSON.parse(body.trim()) as { question?: string; options?: string[]; assumptions?: string[] };
        if (typeof parsed.question === "string" && parsed.question) {
          const q: QuestionState = { question: parsed.question };
          if (Array.isArray(parsed.options) && parsed.options.length > 0) q.options = parsed.options;
          if (Array.isArray(parsed.assumptions) && parsed.assumptions.length > 0) q.assumptions = parsed.assumptions;
          question = q;
          parsedOk = true;
        }
      } catch {
        // Malformed questions fence — left visible below.
      }
    } else if (f.label === "build") {
      try {
        const parsed = JSON.parse(body.trim()) as { topic?: string };
        if (typeof parsed.topic === "string" && parsed.topic) {
          buildTopic = parsed.topic;
          parsedOk = true;
        }
      } catch {
        // Malformed build fence — left visible below.
      }
    } else if (f.label === "edit") {
      try {
        const parsed = JSON.parse(body.trim()) as { studio?: string; studioId?: string; change?: string };
        if (typeof parsed.studio === "string" && parsed.studio && typeof parsed.change === "string" && parsed.change) {
          edit = { studio: parsed.studio, change: parsed.change };
          if (typeof parsed.studioId === "string" && parsed.studioId) edit.studioId = parsed.studioId;
          parsedOk = true;
        }
      } catch {
        // Malformed edit fence — left visible below.
      }
    }

    // Never silently eat the model's words: an unclosed fence keeps its tail
    // (mid-stream buffering is the caller's job), and a closed-but-malformed
    // fence stays visible so the caller can fall back to the raw reply.
    if (f.closeAt === -1) {
      truncated = true;
      continue;
    }
    if (!parsedOk) {
      failed = true;
      continue;
    }

    // Remove the parsed fence from the visible text.
    const before = visible.slice(0, f.openAt);
    const after = visible.slice(f.closeAt + 3);
    visible = (before + after).replace(/\n{3,}/g, "\n\n").trim();
  }

  const plan = todos
    ? {
        todos,
        active: Math.min(active, todos.length),
        done: Math.min(done, todos.length),
        pct,
      }
    : null;
  const progress = sawProgress
    ? { active: progressActive, done: progressDone, pct: progressPct }
    : null;
  return { visible, plan, progress, question, buildTopic, edit, truncated, failed };
}

/** Find all ```plan, ```progress, ```questions, ```build, ```edit fences (paired or open). */
function findPlanFences(text: string): FenceSpan[] {
  const results: FenceSpan[] = [];
  const regex = /```(plan|progress|questions|build|edit)\n/;
  let searchFrom = 0;

  while (true) {
    const match = regex.exec(text.slice(searchFrom));
    if (!match) break;
    const openAt = searchFrom + match.index;
    const bodyStart = openAt + match[0].length;
    const label = match[1];

    const closeIdx = text.indexOf("```", bodyStart);
    const closeAt = closeIdx === -1 ? -1 : closeIdx;

    results.push({ label, openAt, bodyStart, closeAt });
    searchFrom = closeAt === -1 ? text.length : closeAt + 3;
  }

  return results;
}
