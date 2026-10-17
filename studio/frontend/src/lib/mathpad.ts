/**
 * KaTeX hangs an \underset label directly off its base, so a tall base — the
 * fractions models stack annotations under all the time — leaves the label
 * touching (even overlapping) the denominator. An invisible `\big|` phantom on
 * the label deepens its box, which is exactly what KaTeX's under-limit gap is
 * measured against, so the label drops clear of the base.
 *
 * Scoped to fraction bases: a short base (`\underset{x}{=}`) keeps TeX's own
 * tight placement. Fail-open: anything unparsable is left untouched.
 */

const UNDERSET = "\\underset";
const FRACTION = /\\[dt]?frac\b/;
/** The invisible depth the label gains (see tmp/shot-math.mjs, port 9360). */
const CLEARANCE = "\\vphantom{\\big|}";

/** Read one balanced `{...}` group at `start` (escaped braces never count). */
function readGroup(tex: string, start: number): { text: string; end: number } | null {
  if (tex[start] !== "{") return null;
  let depth = 0;
  for (let i = start; i < tex.length; i++) {
    const ch = tex[i];
    if (ch === "\\") {
      i++; // an escaped char (including \{ and \}) never affects the depth
      continue;
    }
    if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) return { text: tex.slice(start + 1, i), end: i + 1 };
    }
  }
  return null;
}

/** Give every \underset label over a fraction its own clearance. */
export function padUnderset(tex: string): string {
  if (!tex.includes(UNDERSET)) return tex;
  let out = "";
  let i = 0;
  for (;;) {
    const at = tex.indexOf(UNDERSET, i);
    if (at === -1) return out + tex.slice(i);
    out += tex.slice(i, at);
    let j = at + UNDERSET.length;
    while (tex[j] === " ") j++;
    const label = readGroup(tex, j);
    if (!label) return out + tex.slice(at);
    let k = label.end;
    while (tex[k] === " ") k++;
    const base = readGroup(tex, k);
    if (!base) return out + tex.slice(at);
    const pad = FRACTION.test(base.text) ? CLEARANCE : "";
    out += `${UNDERSET}{${label.text}${pad}}{${base.text}}`;
    i = base.end;
  }
}

/** Minimal mdast shape the walk touches. */
interface MathNode {
  type: string;
  value?: string;
  /** mdast-util-math pre-builds the hast text child here; it wins at
      conversion, so the padding must overwrite it as well as `value`. */
  data?: { hChildren?: { type: string; value?: string }[] };
  children?: MathNode[];
}

/** The remark plugin: pad every math node's source before KaTeX sees it. */
export function padMathAnnotations() {
  return (tree: MathNode): void => {
    const walk = (node: MathNode | undefined) => {
      for (const child of node?.children ?? []) {
        if ((child.type === "inlineMath" || child.type === "math") && typeof child.value === "string") {
          const next = padUnderset(child.value);
          if (next !== child.value) {
            child.value = next;
            for (const kid of child.data?.hChildren ?? []) {
              if (kid.type === "text") kid.value = next;
            }
          }
        }
        walk(child);
      }
    };
    walk(tree);
  };
}
