/**
 * The web tools the chat model may call — `web_search` and `web_fetch` —
 * executed by the platform gateway (keyless) and approved by the user in-chat.
 *
 * This module owns the tool schemas, the gateway calls, and the string result
 * the model receives. The streaming loop that drives them lives in api.ts; the
 * consent card and the message record live in App.tsx.
 */

import { PLATFORM_BASE_URL } from "./platform";
import type { ChatToolDef, WebToolEvent, WebToolSettings } from "../types";

export const WEB_SEARCH_TOOL: ChatToolDef = {
  type: "function",
  function: {
    name: "web_search",
    description:
      "Search the web. Use for current events, facts you are unsure about, or anything that may have changed since your training. Returns titles, URLs and snippets.",
    parameters: {
      type: "object",
      properties: {
        query: { type: "string", description: "The search query." },
      },
      required: ["query"],
    },
  },
};

export const WEB_FETCH_TOOL: ChatToolDef = {
  type: "function",
  function: {
    name: "web_fetch",
    description:
      "Fetch one web page and return its readable content as markdown. Use after web_search to read a promising result, or when the user gives you a URL.",
    parameters: {
      type: "object",
      properties: {
        url: { type: "string", description: "The absolute http(s) URL to fetch." },
        format: {
          type: "string",
          enum: ["markdown", "text", "html"],
          description: "Output format; defaults to markdown.",
        },
      },
      required: ["url"],
    },
  },
};

/** The schemas to advertise, given the user's Settings toggles. */
export function webToolDefs(enabled: WebToolSettings): ChatToolDef[] {
  const defs: ChatToolDef[] = [];
  if (enabled.search) defs.push(WEB_SEARCH_TOOL);
  if (enabled.fetch) defs.push(WEB_FETCH_TOOL);
  return defs;
}

/** What running one tool produced: the model's result + the card's fields. */
export interface ToolRun {
  content: string;
  patch: Partial<Pick<WebToolEvent, "sources" | "title" | "detail">>;
}

async function postJson(
  path: string,
  body: unknown,
  signal?: AbortSignal
): Promise<Record<string, unknown>> {
  const base = PLATFORM_BASE_URL.replace(/\/+$/, "");
  const res = await fetch(`${base}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    let detail = `Request failed with status ${res.status}`;
    try {
      const payload: unknown = await res.json();
      const msg = (payload as { error?: { message?: string } })?.error?.message;
      if (msg) detail = msg;
    } catch {
      // non-JSON error body
    }
    throw new Error(detail);
  }
  return (await res.json()) as Record<string, unknown>;
}

/**
 * Run one tool call against the platform gateway. Throws on failure; the
 * caller turns the message into the tool result so the model can react.
 */
export async function runWebTool(
  name: string,
  args: Record<string, unknown>,
  signal?: AbortSignal
): Promise<ToolRun> {
  if (name === "web_search") {
    const query = typeof args.query === "string" ? args.query.trim() : "";
    if (!query) throw new Error("The web_search call had no query.");
    const data = await postJson("/websearch", { query }, signal);
    const raw = Array.isArray(data.results) ? (data.results as Record<string, unknown>[]) : [];
    const results = raw
      .filter((r) => typeof r.url === "string" && /^https?:\/\//i.test(r.url))
      .slice(0, 8)
      .map((r) => ({
        title: typeof r.title === "string" && r.title ? r.title : String(r.url),
        url: String(r.url),
        content: typeof r.content === "string" ? r.content : "",
      }));
    if (!results.length) throw new Error(`The web search for "${query}" returned no results.`);
    const lines = results.map(
      (r, i) => `${i + 1}. ${r.title} — ${r.url}${r.content ? `\n   ${r.content}` : ""}`
    );
    const today = new Date().toISOString().slice(0, 10);
    return {
      content: `Search results for "${query}" (${today}):\n\n${lines.join("\n")}\n\nCite the sources you use as markdown links.`,
      patch: {
        sources: results.map((r) => ({ title: r.title, url: r.url })),
        detail: `${results.length} source${results.length === 1 ? "" : "s"}`,
      },
    };
  }

  if (name === "web_fetch") {
    const url = typeof args.url === "string" ? args.url.trim() : "";
    if (!/^https?:\/\//i.test(url)) throw new Error("The web_fetch call needs an absolute http(s) URL.");
    const format = args.format === "text" || args.format === "html" ? args.format : "markdown";
    const data = await postJson("/webfetch", { url, format }, signal);
    const content = typeof data.content === "string" ? data.content : "";
    if (!content.trim()) throw new Error(`The page ${url} returned nothing readable.`);
    const finalUrl = typeof data.finalUrl === "string" && data.finalUrl ? data.finalUrl : url;
    const title = typeof data.title === "string" ? data.title : "";
    const truncated = data.truncated === true;
    return {
      content: [
        `Fetched ${finalUrl}${title ? ` — "${title}"` : ""}${truncated ? " (truncated)" : ""}:`,
        "",
        content,
      ].join("\n"),
      patch: {
        title,
        detail: truncated ? "truncated" : finalUrl.replace(/^https?:\/\//, "").replace(/\/$/, ""),
      },
    };
  }

  throw new Error(`Unknown tool "${name}".`);
}
