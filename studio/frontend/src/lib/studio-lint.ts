/**
 * Deterministic design review: the mechanical half of the critic. It reads a
 * declarative spec (no rendering) and reports the design-floor violations a
 * text-only model always makes — a root that does not fill the window, a
 * default 50/50 split, accent spam, flat type, overlong prose. When the model
 * has no vision (or the screenshot fails), these findings stand in for the
 * critic's review and drive the same polish pass.
 */

import type { StudioBlock, StudioSpec } from "../types";

const MAX_SCREENS = 6;
const MAX_ISSUES = 6;
const MAX_PROSE_CHARS = 320;
const MAX_DEPTH = 4;

function childrenOf(block: StudioBlock): StudioBlock[] | null {
  const kids = (block as { blocks?: unknown }).blocks;
  return Array.isArray(kids) ? (kids as StudioBlock[]) : null;
}

function walk(blocks: StudioBlock[], visit: (block: StudioBlock) => void, depth = 0): void {
  if (depth > MAX_DEPTH) return;
  for (const block of blocks) {
    visit(block);
    const kids = childrenOf(block);
    if (kids) walk(kids, visit, depth + 1);
  }
}

/** Every design-floor violation an unrendered spec can be checked for. */
export function lintSpec(spec: StudioSpec): string[] {
  if (spec.logic) return [];
  const states = spec.states ?? {};
  const issues: string[] = [];

  for (const id of Object.keys(states).slice(0, MAX_SCREENS)) {
    if (issues.length >= MAX_ISSUES) break;
    const blocks = states[id]?.blocks ?? [];
    const screen = `screen "${id}"`;

    const fills =
      blocks.length <= 2 ||
      blocks.some((b) => (b.type === "stack" || b.type === "columns") && (b as { fill?: boolean }).fill === true);
    if (!fills) issues.push(`${screen}: the root layout is missing "fill": true — the app must fill the window.`);

    let accents = 0;
    let headings = 0;
    let texts = 0;
    walk(blocks, (block) => {
      const tone = (block as { tone?: string }).tone;
      if (tone === "accent") accents += 1;

      if (block.type === "heading") headings += 1;
      if (block.type === "text") {
        texts += 1;
        const content = (block as { content?: string }).content ?? "";
        if (content.length > MAX_PROSE_CHARS && !content.includes("\n")) {
          issues.push(
            `${screen}: a text block runs ${content.length} characters unbroken — keep prose under ~${MAX_PROSE_CHARS} chars or split it.`
          );
        }
      }

      if (block.type === "columns") {
        const sizes = (block as { sizes?: number[] }).sizes;
        const columns = childrenOf(block) ?? [];
        const equalSplit = !sizes || new Set(sizes.slice(0, columns.length)).size <= 1;
        if (columns.length >= 2 && equalSplit) {
          issues.push(
            `${screen}: columns are an equal ${columns.length}-way split — use "sizes" (e.g. [2,1]) to express the real relationship.`
          );
        }
      }
    });

    if (accents > 2) {
      issues.push(`${screen}: ${accents} elements use tone "accent" — it is the loudest color; use it on at most 2.`);
    }
    if (texts + headings >= 4 && headings === 0) {
      issues.push(`${screen}: no heading sets the type hierarchy — add one display/h1 (and h2 for sections).`);
    }
  }

  return issues.slice(0, MAX_ISSUES);
}