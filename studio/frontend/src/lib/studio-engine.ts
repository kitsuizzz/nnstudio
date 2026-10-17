/**
 * The declarative workspace engine: {{expression}} bindings and the closed
 * action vocabulary (set / inc / push / remove / toggle / reset / to, plus
 * chained actions) over dotted data paths. Pure and side-effect free so it
 * can be unit-tested without React.
 */

import type { StudioAction, StudioValue } from "../types";
import { resolveText, resolveValue, truthy } from "./expr";

export { evalBinding, evalExpr, resolveText, resolveValue, truthy } from "./expr";

function isRecord(x: unknown): x is Record<string, unknown> {
  return !!x && typeof x === "object" && !Array.isArray(x);
}

/** Read a dotted path ("board.42" or "move.n") from a plain object. */
export function getPath(obj: unknown, path: string): unknown {
  let cur: unknown = obj;
  for (const key of path.split(".")) {
    if (Array.isArray(cur)) cur = cur[Number(key)];
    else if (isRecord(cur)) cur = cur[key];
    else return undefined;
  }
  return cur;
}

/** Write a dotted path, immutably; missing intermediates are created. */
export function setPath(obj: Record<string, unknown>, path: string, value: unknown): Record<string, unknown> {
  const keys = path.split(".");
  const walk = (node: unknown, i: number): unknown => {
    if (i === keys.length) return value;
    const key = keys[i];
    if (Array.isArray(node)) {
      const idx = Number(key);
      if (!Number.isInteger(idx) || idx < 0 || idx >= node.length) return node;
      const arr = [...node];
      arr[idx] = walk(arr[idx], i + 1);
      return arr;
    }
    if (isRecord(node)) {
      const rec = { ...node };
      rec[key] = key in rec ? walk(rec[key], i + 1) : walk({}, i + 1);
      return rec;
    }
    return node;
  };
  return walk(obj, 0) as Record<string, unknown>;
}

/** Resolve {{expression}} tokens in a string to display text. */
export function interpolate(text: string, data: Record<string, unknown>, arg?: unknown): string {
  return resolveText(text, data, arg);
}

/** True when the action mutates data (navigation alone does not). */
export function hasMutation(action: StudioAction): boolean {
  return !!(
    action.set ||
    action.inc ||
    action.push ||
    action.remove ||
    action.toggle ||
    action.reset ||
    action.actions?.length
  );
}

/** The state to navigate to after an action; the last nested `to` wins. */
export function actionTarget(action: StudioAction): string | undefined {
  let target = action.to;
  for (const nested of action.actions ?? []) {
    const t = actionTarget(nested);
    if (t) target = t;
  }
  return target;
}

/** Coerce an evaluated expression into a storable plain value (records included). */
function toStudioValue(v: unknown, depth = 0): StudioValue {
  if (v == null) return "";
  if (typeof v === "boolean") return v;
  if (typeof v === "number") return Number.isFinite(v) ? v : 0;
  if (typeof v === "string") return v.slice(0, 2_000);
  if (Array.isArray(v)) {
    if (depth >= 3) return [];
    const arr: StudioValue[] = [];
    for (const item of v.slice(0, 512)) arr.push(toStudioValue(item, depth + 1));
    return arr;
  }
  if (typeof v === "object" && depth < 3) {
    const out: Record<string, StudioValue> = {};
    for (const [k, item] of Object.entries(v as Record<string, unknown>).slice(0, 64)) {
      out[k.slice(0, 64)] = toStudioValue(item, depth + 1);
    }
    return out;
  }
  return "";
}

function sameValue(a: unknown, b: unknown): boolean {
  if (typeof a === "number" && typeof b === "number") return a === b;
  return String(a) === String(b);
}

/** Apply one action (and any chained actions) to a data snapshot (pure). */
export function applyAction(
  data: Record<string, unknown>,
  action: StudioAction,
  arg: unknown,
  initial: Record<string, unknown>
): Record<string, unknown> {
  let d = data;
  for (const [path, raw] of Object.entries(action.set ?? {})) {
    const value = typeof raw === "string" ? resolveValue(raw, d, arg) : raw;
    d = setPath(d, path, toStudioValue(value));
  }
  for (const [path, delta] of Object.entries(action.inc ?? {})) {
    const cur = getPath(d, path);
    d = setPath(d, path, (typeof cur === "number" ? cur : Number(cur) || 0) + delta);
  }
  for (const [path, raw] of Object.entries(action.push ?? {})) {
    const value = typeof raw === "string" ? resolveValue(raw, d, arg) : raw;
    const cur = getPath(d, path);
    const arr = Array.isArray(cur) ? cur : [];
    d = setPath(d, path, [...arr, toStudioValue(value)]);
  }
  for (const [path, raw] of Object.entries(action.remove ?? {})) {
    const cur = getPath(d, path);
    if (!Array.isArray(cur)) continue;
    const value = typeof raw === "string" ? resolveValue(raw, d, arg) : raw;
    const next =
      typeof value === "number"
        ? cur.filter((_, i) => i !== value)
        : (() => {
            const i = cur.findIndex((item) => sameValue(item, value));
            return i === -1 ? cur : [...cur.slice(0, i), ...cur.slice(i + 1)];
          })();
    d = setPath(d, path, next);
  }
  for (const path of action.toggle ?? []) {
    d = setPath(d, path, !truthy(getPath(d, path)));
  }
  for (const path of action.reset ?? []) {
    d = setPath(d, path, getPath(initial, path));
  }
  for (const nested of action.actions ?? []) {
    d = applyAction(d, nested, arg, initial);
  }
  return d;
}
