/**
 * The App Builder agent's handoff: what the main chat agent passes down when it
 * delegates a build (or an edit) to a build thread. The builder gets the tail
 * of the conversation that produced the request — the ask, the user's answers
 * to the triage questions, and any constraints — instead of just the one-line
 * brief. Text only (re-sending attachments would double their cost; the newest
 * attachment rides on the thread's own seed message).
 */

import type { ChatMessage } from "../types";

const MAX_HANDOFF_MESSAGES = 8;
const MAX_HANDOFF_CHARS = 1500;

/** The tail of a conversation, trimmed and capped, for the builder's context. */
export function handoffContext(messages: ChatMessage[]): ChatMessage[] {
  const out: ChatMessage[] = [];
  for (const m of messages) {
    const content = m.content.trim();
    if (!content) continue;
    out.push({
      role: m.role,
      content: content.length > MAX_HANDOFF_CHARS ? content.slice(0, MAX_HANDOFF_CHARS) + "…" : content,
      ...(m.at != null ? { at: m.at } : {}),
    });
  }
  return out.slice(-MAX_HANDOFF_MESSAGES);
}
