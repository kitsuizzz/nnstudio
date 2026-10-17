/**
 * The declarative expression language: a tiny, safe evaluator behind the
 * {{…}} bindings AI-built workspaces use. Pure and dependency-free — no
 * eval, no calls beyond the whitelist below, no prototype access — so a
 * model-authored expression can never run code.
 *
 * Grammar (loosest to tightest):
 *   or      := and ("||" and)*
 *   and     := cmp ("&&" cmp)*
 *   cmp     := add (("==" | "!=" | "<" | "<=" | ">" | ">=") add)*
 *   add     := mul (("+" | "-") mul)*
 *   mul     := unary (("*" | "/" | "%") unary)*
 *   unary   := ("!" | "-") unary | primary
 *   primary := number | string | true | false | null
 *            | path | path "(" args ")" | "(" or ")"
 *
 * Paths read the app's data (dotted, numeric indices allowed: cards.3.title)
 * and "arg" — the event argument a tap/choice/submit passed in.
 */

const MAX_EXPR_CHARS = 200;
const MAX_TOKENS = 200;
// One grammar level costs a few frames (parseOr → … → parsePrimary), so the
// ceiling must comfortably clear normal parenthesized / function expressions.
const MAX_DEPTH = 64;

const BLOCKED_KEYS = new Set(["__proto__", "constructor", "prototype"]);

type Token =
  | { k: "num"; v: number }
  | { k: "str"; v: string }
  | { k: "id"; v: string }
  | { k: "op"; v: string }
  | { k: "(" }
  | { k: ")" }
  | { k: "," };

const IDENT_START = /[A-Za-z_]/;
const IDENT_PART = /[A-Za-z0-9_]/;

/** Tokenize, or null on anything the grammar does not admit. */
function tokenize(src: string): Token[] | null {
  const toks: Token[] = [];
  let i = 0;
  while (i < src.length) {
    const ch = src[i];
    if (ch === " " || ch === "\t" || ch === "\n" || ch === "\r") {
      i += 1;
      continue;
    }
    if (toks.length > MAX_TOKENS) return null;
    if (ch >= "0" && ch <= "9") {
      let j = i;
      while (j < src.length && ((src[j] >= "0" && src[j] <= "9") || src[j] === ".")) j += 1;
      const n = Number(src.slice(i, j));
      if (!Number.isFinite(n)) return null;
      toks.push({ k: "num", v: n });
      i = j;
      continue;
    }
    if (ch === '"' || ch === "'") {
      const quote = ch;
      let out = "";
      let j = i + 1;
      let closed = false;
      while (j < src.length) {
        const c = src[j];
        if (c === "\\") {
          const next = src[j + 1];
          if (next === "n") out += "\n";
          else if (next === "t") out += "\t";
          else if (next === "r") out += "\r";
          else if (next != null) out += next;
          j += 2;
          continue;
        }
        if (c === quote) {
          closed = true;
          j += 1;
          break;
        }
        out += c;
        j += 1;
      }
      if (!closed) return null;
      toks.push({ k: "str", v: out.slice(0, 500) });
      i = j;
      continue;
    }
    if (IDENT_START.test(ch)) {
      let j = i + 1;
      while (j < src.length && IDENT_PART.test(src[j])) j += 1;
      let path = src.slice(i, j);
      // Dotted segments (cards.3.title) extend the same path token.
      while (src[j] === "." && IDENT_PART.test(src[j + 1] ?? "")) {
        let k = j + 1;
        while (k < src.length && IDENT_PART.test(src[k])) k += 1;
        path += "." + src.slice(j + 1, k);
        j = k;
      }
      toks.push({ k: "id", v: path.slice(0, 120) });
      i = j;
      continue;
    }
    const two = src.slice(i, i + 2);
    if (two === "==" || two === "!=" || two === "<=" || two === ">=" || two === "&&" || two === "||") {
      toks.push({ k: "op", v: two });
      i += 2;
      continue;
    }
    if ("+-*/%!<>".includes(ch)) {
      toks.push({ k: "op", v: ch });
      i += 1;
      continue;
    }
    if (ch === "(") {
      toks.push({ k: "(" });
      i += 1;
      continue;
    }
    if (ch === ")") {
      toks.push({ k: ")" });
      i += 1;
      continue;
    }
    if (ch === ",") {
      toks.push({ k: "," });
      i += 1;
      continue;
    }
    return null;
  }
  return toks.length && toks.length <= MAX_TOKENS ? toks : null;
}

function toNum(v: unknown): number {
  if (typeof v === "number") return Number.isFinite(v) ? v : 0;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "string") {
    const n = Number(v.trim());
    return Number.isFinite(n) ? n : 0;
  }
  return 0;
}

function isNumeric(v: unknown): boolean {
  if (typeof v === "number") return Number.isFinite(v);
  if (typeof v === "string") return v.trim() !== "" && Number.isFinite(Number(v));
  return false;
}

/** Truthiness: empty / zero / "0" / "false"-as-number are false; a non-empty array is true. */
export function truthy(v: unknown): boolean {
  if (v == null || v === false) return false;
  if (typeof v === "number") return Number.isFinite(v) && v !== 0;
  if (typeof v === "string") {
    const s = v.trim();
    if (s === "") return false;
    const n = Number(s);
    if (Number.isFinite(n)) return n !== 0;
    return true;
  }
  if (Array.isArray(v)) return v.length > 0;
  return true;
}

function looseEq(a: unknown, b: unknown): boolean {
  if (a == null && b == null) return true;
  if (a == null || b == null) return String(a ?? "") === String(b ?? "");
  if (typeof a === "boolean" || typeof b === "boolean") return truthy(a) === truthy(b);
  if (isNumeric(a) && isNumeric(b)) return toNum(a) === toNum(b);
  if (Array.isArray(a) || Array.isArray(b)) return JSON.stringify(a) === JSON.stringify(b);
  return String(a) === String(b);
}

function compare(a: unknown, b: unknown): number {
  if (isNumeric(a) && isNumeric(b)) return toNum(a) - toNum(b);
  return String(a ?? "").localeCompare(String(b ?? ""));
}

function readPath(data: Record<string, unknown>, path: string, arg: unknown): unknown {
  if (path === "arg") return arg;
  const keys = path.split(".");
  for (const key of keys) if (BLOCKED_KEYS.has(key)) return undefined;
  let cur: unknown = data;
  for (const key of keys) {
    if (Array.isArray(cur)) cur = cur[Number(key)];
    else if (cur && typeof cur === "object") cur = (cur as Record<string, unknown>)[key];
    else return undefined;
  }
  return cur;
}

function flattenNums(args: unknown[]): number[] {
  const out: number[] = [];
  for (const a of args) {
    if (Array.isArray(a)) out.push(...a.map(toNum));
    else out.push(toNum(a));
  }
  return out;
}

/** The whole function vocabulary — fixed names, fixed arity, no code. */
const FUNCS: Record<string, { min: number; max: number; fn: (args: unknown[]) => unknown }> = {
  len: {
    min: 1,
    max: 1,
    fn: ([v]) =>
      Array.isArray(v)
        ? v.length
        : typeof v === "string"
          ? v.length
          : v && typeof v === "object"
            ? Object.keys(v).length
            : 0,
  },
  min: { min: 1, max: 8, fn: (a) => Math.min(...flattenNums(a)) },
  max: { min: 1, max: 8, fn: (a) => Math.max(...flattenNums(a)) },
  abs: { min: 1, max: 1, fn: ([v]) => Math.abs(toNum(v)) },
  round: { min: 1, max: 1, fn: ([v]) => Math.round(toNum(v)) },
  floor: { min: 1, max: 1, fn: ([v]) => Math.floor(toNum(v)) },
  ceil: { min: 1, max: 1, fn: ([v]) => Math.ceil(toNum(v)) },
  number: { min: 1, max: 1, fn: ([v]) => toNum(v) },
  str: {
    min: 1,
    max: 1,
    fn: ([v]) => (v == null ? "" : Array.isArray(v) ? v.map(String).join(", ") : String(v)),
  },
  upper: { min: 1, max: 1, fn: ([v]) => String(v ?? "").toUpperCase() },
  lower: { min: 1, max: 1, fn: ([v]) => String(v ?? "").toLowerCase() },
  join: {
    min: 1,
    max: 2,
    fn: ([v, sep]) =>
      Array.isArray(v) ? v.map((x) => (x == null ? "" : String(x))).join(sep == null ? ", " : String(sep)) : String(v ?? ""),
  },
  contains: {
    min: 2,
    max: 2,
    fn: ([a, b]) =>
      Array.isArray(a) ? a.some((x) => looseEq(x, b)) : String(a ?? "").includes(String(b ?? "")),
  },
};

/** Evaluate one expression against the data (and the event argument). */
export function evalExpr(src: string, data: Record<string, unknown>, arg?: unknown): unknown {
  const text = src.trim();
  if (!text || text.length > MAX_EXPR_CHARS) return undefined;
  const toks = tokenize(text);
  if (!toks) return undefined;

  let pos = 0;
  let depth = 0;

  const peek = (): Token | undefined => toks[pos];
  const next = (): Token => {
    const t = toks[pos];
    if (!t) throw new Error("unexpected end");
    pos += 1;
    return t;
  };
  const isOp = (v: string): boolean => {
    const t = peek();
    return !!t && t.k === "op" && t.v === v;
  };
  const guard = (): void => {
    depth += 1;
    if (depth > MAX_DEPTH) throw new Error("too deep");
  };

  const parseOr = (): unknown => {
    guard();
    let left = parseAnd();
    while (isOp("||")) {
      next();
      const right = parseAnd();
      left = truthy(left) ? left : right;
    }
    depth -= 1;
    return left;
  };

  const parseAnd = (): unknown => {
    guard();
    let left = parseCmp();
    while (isOp("&&")) {
      next();
      const right = parseCmp();
      left = truthy(left) ? right : left;
    }
    depth -= 1;
    return left;
  };

  const parseCmp = (): unknown => {
    guard();
    let left = parseAdd();
    for (;;) {
      const t = peek();
      if (!t || t.k !== "op") break;
      if (t.v === "==" || t.v === "!=" || t.v === "<" || t.v === "<=" || t.v === ">" || t.v === ">=") {
        next();
        const right = parseAdd();
        if (t.v === "==") left = looseEq(left, right);
        else if (t.v === "!=") left = !looseEq(left, right);
        else if (t.v === "<") left = compare(left, right) < 0;
        else if (t.v === "<=") left = compare(left, right) <= 0;
        else if (t.v === ">") left = compare(left, right) > 0;
        else left = compare(left, right) >= 0;
      } else break;
    }
    depth -= 1;
    return left;
  };

  const parseAdd = (): unknown => {
    guard();
    let left = parseMul();
    for (;;) {
      if (isOp("+")) {
        next();
        const right = parseMul();
        left =
          typeof left === "string" || typeof right === "string"
            ? String(left ?? "") + String(right ?? "")
            : toNum(left) + toNum(right);
      } else if (isOp("-")) {
        next();
        left = toNum(left) - toNum(parseMul());
      } else break;
    }
    depth -= 1;
    return left;
  };

  const parseMul = (): unknown => {
    guard();
    let left = parseUnary();
    for (;;) {
      if (isOp("*")) {
        next();
        left = toNum(left) * toNum(parseUnary());
      } else if (isOp("/")) {
        next();
        const right = toNum(parseUnary());
        left = right === 0 ? 0 : toNum(left) / right;
      } else if (isOp("%")) {
        next();
        const right = toNum(parseUnary());
        left = right === 0 ? 0 : toNum(left) % right;
      } else break;
    }
    depth -= 1;
    return left;
  };

  const parseUnary = (): unknown => {
    guard();
    if (isOp("!")) {
      next();
      const v = parseUnary();
      depth -= 1;
      return !truthy(v);
    }
    if (isOp("-")) {
      next();
      const v = toNum(parseUnary());
      depth -= 1;
      return -v;
    }
    const v = parsePrimary();
    depth -= 1;
    return v;
  };

  const parsePrimary = (): unknown => {
    guard();
    const t = peek();
    if (!t) throw new Error("unexpected end");
    if (t.k === "(") {
      next();
      const v = parseOr();
      const close = next();
      if (close.k !== ")") throw new Error("expected )");
      depth -= 1;
      return v;
    }
    if (t.k === "num" || t.k === "str") {
      next();
      depth -= 1;
      return t.v;
    }
    if (t.k === "id") {
      next();
      if (t.v === "true") {
        depth -= 1;
        return true;
      }
      if (t.v === "false") {
        depth -= 1;
        return false;
      }
      if (t.v === "null") {
        depth -= 1;
        return null;
      }
      // A function call: name "(" args ")".
      if (peek()?.k === "(") {
        next();
        const args: unknown[] = [];
        if (peek()?.k !== ")") {
          for (;;) {
            args.push(parseOr());
            const sep = peek();
            if (sep?.k === ",") {
              next();
              continue;
            }
            break;
          }
        }
        const close = next();
        if (close.k !== ")") throw new Error("expected )");
        const def = FUNCS[t.v];
        if (!def) throw new Error("unknown function");
        if (args.length < def.min || args.length > def.max) throw new Error("bad arity");
        depth -= 1;
        return def.fn(args);
      }
      depth -= 1;
      return readPath(data, t.v, arg);
    }
    throw new Error("unexpected token");
  };

  try {
    const value = parseOr();
    if (pos !== toks.length) return undefined;
    return value;
  } catch {
    return undefined;
  }
}

/** Evaluate an `if` / `each` binding: braces optional, result truthy-checked by the caller. */
export function evalBinding(src: string, data: Record<string, unknown>, arg?: unknown): unknown {
  const text = src.trim();
  const m = text.match(/^\{\{([\s\S]*)\}\}$/);
  return evalExpr(m ? m[1] : text, data, arg);
}

const BIND = /\{\{([\s\S]*?)\}\}/g;

/** A value as display text: null/undefined empty, arrays joined with spaces, objects hidden. */
function stringify(v: unknown): string {
  if (v == null) return "";
  if (typeof v === "object") return Array.isArray(v) ? v.map(stringify).filter(Boolean).join(" ") : "";
  return String(v);
}

/** Resolve every {{expression}} in a string to display text. */
export function resolveText(text: string, data: Record<string, unknown>, arg?: unknown): string {
  return text.replace(BIND, (_m, src: string) => stringify(evalExpr(String(src), data, arg)));
}

/** A binding that is the entire string keeps its typed value; mixed text stays text. */
export function resolveValue(raw: string, data: Record<string, unknown>, arg?: unknown): unknown {
  const m = raw.match(/^\s*\{\{([\s\S]*)\}\}\s*$/);
  if (m) return evalExpr(m[1], data, arg);
  return resolveText(raw, data, arg);
}
