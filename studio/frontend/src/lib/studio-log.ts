/**
 * Formatting for the debug log: one-line entries for the builder-edit prompt
 * and a fuller report for the Debug panel (copy / "Fix with AI").
 */

import type { StudioLogEntry } from "../types";

const MAX_MESSAGE = 500;
const MAX_STACK_LINE = 200;

/** One prompt-friendly line: `- [error] (logic) message — first stack frame`. */
export function formatLogLine(entry: StudioLogEntry): string {
  const count = entry.count && entry.count > 1 ? ` (x${entry.count})` : "";
  const stack = entry.stack ? ` — ${entry.stack.split("\n")[0].trim().slice(0, MAX_STACK_LINE)}` : "";
  const message = entry.message.replace(/\s+/g, " ").slice(0, MAX_MESSAGE);
  return `- [${entry.level}] (${entry.source}) ${message}${count}${stack}`;
}

/** The report the Debug panel copies and hands to the AI. */
export function formatLogReport(logs: StudioLogEntry[], limit = 40): string {
  return logs
    .slice(-limit)
    .map((entry) => {
      const count = entry.count && entry.count > 1 ? ` (x${entry.count})` : "";
      const time = new Date(entry.at).toISOString().slice(11, 19);
      const head = `${time} [${entry.level}] (${entry.source})${count} ${entry.message}`;
      const stack = entry.stack
        ? "\n    " + entry.stack.split("\n").slice(0, 4).join("\n    ").slice(0, 800)
        : "";
      return head + stack;
    })
    .join("\n");
}
