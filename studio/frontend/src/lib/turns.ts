import type { ChatMessage } from "../types";

/** One exchange: the prompt (a "title") and the work it produced. */
export interface Turn {
  /** 1-based number within the conversation. */
  no: number;
  prompt: ChatMessage;
  response?: ChatMessage;
}

/**
 * Pairs the flat message list into turns. A user message opens a turn; the
 * assistant message that follows closes it. A user message with no reply
 * (stream stopped before any token arrived) yields a turn without a response.
 * Stray leading assistant messages are attached to a synthetic empty prompt so
 * nothing is silently dropped.
 */
export function pairTurns(messages: ChatMessage[]): Turn[] {
  const turns: Turn[] = [];
  for (const m of messages) {
    if (m.role === "user") {
      turns.push({ no: turns.length + 1, prompt: m });
      continue;
    }
    const last = turns[turns.length - 1];
    if (last && !last.response) {
      last.response = m;
    } else {
      turns.push({ no: turns.length + 1, prompt: { role: "user", content: "" }, response: m });
    }
  }
  return turns;
}
