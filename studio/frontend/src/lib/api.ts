import type { ChatMessage, ChatToolDef, LlmConfig } from "../types";

const TITLE_PROMPT = [
  "Based on the conversation above, create a title for this session.",
  "Rules:",
  "- 3 to 10 words",
  "- Capture the core topic of the conversation, not a summary of the exchange",
  "- Start with a capital letter",
  "- No trailing punctuation, no quotation marks",
  '- Do not include prefixes like "Title:"',
  "- Output only the title and nothing else",
].join("\n");

const TITLE_SYSTEM =
  "You generate short conversation titles. Do not think, do not reason, do not plan. Respond immediately with only the title.";

async function postCompletions(
  config: LlmConfig,
  body: Record<string, unknown>,
  signal: AbortSignal
): Promise<Response> {
  const url = config.baseUrl.replace(/\/+$/, "") + "/chat/completions";
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  // The platform gateway takes no key; an empty bearer is simply omitted.
  if (config.apiKey) headers.Authorization = `Bearer ${config.apiKey}`;
  return fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal,
  });
}

export interface ProviderTestResult {
  ok: boolean;
  message: string;
}

/** One model as exposed by an OpenAI-compatible /models listing. */
export interface ModelInfo {
  id: string;
  /** Whether the provider's metadata says the model accepts images.
      Undefined when the provider doesn't advertise modality information. */
  vision?: boolean;
}

/** Best-effort vision detection across the shapes providers use. */
function modelVision(m: Record<string, unknown>): boolean | undefined {
  const mods =
    (m.architecture as { input_modalities?: unknown } | undefined)?.input_modalities ??
    m.input_modalities ??
    m.modalities;
  if (Array.isArray(mods) && mods.length && mods.every((x) => typeof x === "string")) {
    const hasImage = mods.some((x) => /image|vision/i.test(x as string));
    const hasText = mods.some((x) => /text/i.test(x as string));
    if (hasImage || hasText) return hasImage;
  }
  const caps = m.capabilities;
  if (caps && typeof caps === "object" && !Array.isArray(caps)) {
    const keys = Object.keys(caps).filter((k) => /vision|image/i.test(k));
    if (keys.length) {
      return keys.some((k) => (caps as Record<string, unknown>)[k] === true || (caps as Record<string, unknown>)[k] === "true");
    }
  }
  if (typeof m.vision === "boolean") return m.vision;
  return undefined;
}

/** A /models listing, or why it couldn't be read. */
export type ModelListing =
  | { ok: true; models: ModelInfo[] }
  | { ok: false; error: string };

/** The provider's model list. Unlike `fetchModelInfos` (which quietly degrades
    to an empty list) the failure is reported: the Agent Settings picker's job
    is to show exactly that — a URL or key the provider won't accept. */
export async function listModels(
  baseUrl: string,
  apiKey: string,
  signal?: AbortSignal
): Promise<ModelListing> {
  try {
    const url = baseUrl.replace(/\/+$/, "") + "/models";
    const headers: Record<string, string> = {};
    if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
    const res = await fetch(url, {
      headers,
      signal,
    });
    if (!res.ok) {
      let detail = `Request failed with status ${res.status}`;
      try {
        const body: unknown = await res.json();
        const msg = (body as { error?: { message?: string } })?.error?.message;
        if (msg) detail = msg;
      } catch {
        // non-JSON error body
      }
      return { ok: false, error: detail };
    }
    const data: { data?: ({ id?: string } & Record<string, unknown>)[] } = await res.json();
    const models = (data.data ?? [])
      .map((m) => {
        const id = m.id ?? "";
        if (!id) return null;
        const info: ModelInfo = { id };
        const vision = modelVision(m);
        if (vision !== undefined) info.vision = vision;
        return info;
      })
      .filter((m): m is ModelInfo => m !== null);
    return { ok: true, models };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
}

/** Lists models with whatever capability metadata the provider exposes. */
export async function fetchModelInfos(
  baseUrl: string,
  apiKey: string,
  signal?: AbortSignal
): Promise<ModelInfo[]> {
  const listing = await listModels(baseUrl, apiKey, signal);
  return listing.ok ? listing.models : [];
}

/** Lists model ids exposed by an OpenAI-compatible /models endpoint. */
export async function fetchModels(
  baseUrl: string,
  apiKey: string,
  signal?: AbortSignal
): Promise<string[]> {
  return (await fetchModelInfos(baseUrl, apiKey, signal)).map((m) => m.id);
}

/** Successful lookups, keyed by baseUrl+model, so switching conversations back
 *  and forth doesn't re-hit the endpoint. Failures are not cached, so a
 * transient error can still be retried on the next agent change. */
const maxTokenCache = new Map<string, number>();

/** The model's context window, from the provider's single-model endpoint
 *  (max_context_tokens). Null when the endpoint doesn't expose it. */
export async function fetchModelMaxTokens(
  baseUrl: string,
  apiKey: string,
  model: string,
  signal?: AbortSignal
): Promise<number | null> {
  const key = `${baseUrl}::${model}`;
  const cached = maxTokenCache.get(key);
  if (cached != null) return cached;
  try {
    const url = `${baseUrl.replace(/\/+$/, "")}/models/${encodeURIComponent(model)}`;
    const headers: Record<string, string> = {};
    if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
    const res = await fetch(url, {
      headers,
      signal,
    });
    if (!res.ok) return null;
    const data: { max_context_tokens?: number } = await res.json();
    if (typeof data.max_context_tokens === "number") {
      maxTokenCache.set(key, data.max_context_tokens);
      return data.max_context_tokens;
    }
  } catch {
    // endpoint unreachable or non-JSON: treat as unknown
  }
  return null;
}

export async function testProvider(
  baseUrl: string,
  apiKey: string
): Promise<ProviderTestResult> {
  try {
    const url = baseUrl.replace(/\/+$/, "") + "/models";
    const res = await fetch(url, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
    });
    if (!res.ok) {
      let detail = `Request failed with status ${res.status}`;
      try {
        const body: unknown = await res.json();
        const msg = (body as { error?: { message?: string } })?.error?.message;
        if (msg) detail = msg;
      } catch {
        // non-JSON error body
      }
      return { ok: false, message: detail };
    }
    const data: { data?: { id?: string }[] } = await res.json();
    const models = data.data ?? [];
    return {
      ok: true,
      message:
        models.length > 0
          ? `Connection successful — ${models.length} model${models.length === 1 ? "" : "s"} available.`
          : "Connection successful.",
    };
  } catch (err) {
    return {
      ok: false,
      message: err instanceof Error ? err.message : String(err),
    };
  }
}

export async function generateTitle(
  config: LlmConfig,
  history: ChatMessage[],
  signal: AbortSignal
): Promise<string | null> {
  try {
    const messages = [
      { role: "system", content: TITLE_SYSTEM },
      ...history.map((m) => ({
        role: m.role,
        content: m.content || (m.images && m.images.length > 0 ? "[image]" : ""),
      })),
      { role: "user", content: TITLE_PROMPT },
    ];
    const plain: Record<string, unknown> = {
      model: config.model,
      messages,
      stream: false,
    };
    const full: Record<string, unknown> = {
      ...plain,
      reasoning_effort: "low",
    };
    if (config.baseUrl.includes("deepseek")) {
      full.chat_template_kwargs = { enable_thinking: false };
    }

    let res = await postCompletions(config, full, signal);
    if (!res.ok) {
      // provider may reject the thinking-effort params, retry without them
      res = await postCompletions(config, plain, signal);
    }
    if (!res.ok) return null;
    const data: { choices?: { message?: { content?: string } }[] } = await res.json();
    const text = data.choices?.[0]?.message?.content;
    const title = typeof text === "string" ? text.trim() : "";
    return title || null;
  } catch {
    return null;
  }
}

const IDEAS_PROMPT =
  "Come up with 3 questions Users can ask to an LLM. Give in JSON format as list of string. No other texts.";

/** One-shot inference: candidate first moves, rendered as clickable chips. */
export async function suggestIdeas(
  baseUrl: string,
  apiKey: string,
  model: string,
  signal?: AbortSignal
): Promise<string[]> {
  try {
    const config: LlmConfig = { baseUrl, apiKey, model, reasoningEffort: "low" };
    const res = await postCompletions(
      config,
      {
        model,
        stream: false,
        temperature: 0.9,
        max_tokens: 240,
        messages: [{ role: "user", content: IDEAS_PROMPT }],
      },
      signal ?? new AbortController().signal
    );
    if (!res.ok) return [];
    const data: { choices?: { message?: { content?: string } }[] } = await res.json();
    const content = data.choices?.[0]?.message?.content;
    if (typeof content !== "string") return [];
    const text = content.replace(/```(?:json)?/gi, "").trim();
    const start = text.indexOf("[");
    const end = text.lastIndexOf("]");
    if (start === -1 || end <= start) return [];
    let parsed: unknown;
    try {
      parsed = JSON.parse(text.slice(start, end + 1));
    } catch {
      return [];
    }
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((s): s is string => typeof s === "string" && s.trim().length > 0)
      .map((s) => s.trim())
      .slice(0, 3);
  } catch {
    return [];
  }
}

/**
 * One-shot, non-streamed completion. Throws on transport or HTTP failure —
 * the caller (the App Builder's planner/reviewer, the design critic, the Code
 * pane's AI assist) surfaces the message. Optional `images` are sent as
 * image_url parts so a vision-capable provider sees the attached screenshots;
 * optional `history` lets a call carry the conversation context it was handed.
 */
export async function completeChat(
  config: LlmConfig,
  systemPrompt: string,
  userContent: string,
  signal?: AbortSignal,
  images?: string[],
  history?: ChatMessage[]
): Promise<string> {
  const userContentParts: unknown =
    images && images.length
      ? [
          ...(userContent ? [{ type: "text", text: userContent }] : []),
          ...images.map((url) => ({ type: "image_url", image_url: { url } })),
        ]
      : userContent;
  const messages: { role: string; content: unknown }[] = [];
  if (systemPrompt) messages.push({ role: "system", content: systemPrompt });
  for (const m of history ?? []) messages.push({ role: m.role, content: toApiContent(m) });
  messages.push({ role: "user", content: userContentParts });
  const res = await postCompletions(
    config,
    {
      model: config.model,
      stream: false,
      messages,
    },
    signal ?? new AbortController().signal
  );
  if (!res.ok) {
    let detail = `Request failed with status ${res.status}`;
    try {
      const body: unknown = await res.json();
      const msg = (body as { error?: { message?: string } })?.error?.message;
      if (msg) detail = msg;
    } catch {
      // non-JSON error body
    }
    throw new Error(detail);
  }
  const data: { choices?: { message?: { content?: string } }[] } = await res.json();
  const content = data.choices?.[0]?.message?.content;
  if (typeof content !== "string" || !content.trim()) {
    throw new Error("The model returned an empty reply.");
  }
  return content;
}

export interface Usage {
  promptTokens: number;
  completionTokens: number;
}

/** One tool call the model requested (accumulated from streamed deltas). */
export interface ToolCallRequest {
  id: string;
  name: string;
  args: Record<string, unknown>;
  /** The exact arguments string the model sent (echoed back to the provider). */
  raw?: string;
}

/** The client-side tool loop's hooks: surface the call, ask the user, run it. */
export interface ChatToolHooks {
  defs: ChatToolDef[];
  /** The model requested a call, before approval — surface it as pending. */
  onCall: (call: ToolCallRequest) => void;
  /** Ask the user. Resolves true to run the call. */
  approve: (call: ToolCallRequest) => Promise<boolean>;
  /** The user declined; the model gets an error result and carries on. */
  onDenied: (call: ToolCallRequest) => void;
  /** Run an approved call; return the string result the model receives. */
  execute: (call: ToolCallRequest) => Promise<string>;
  /** Tool rounds before tools are dropped for a final answer (default 4). */
  maxRounds?: number;
}

interface RoundResult {
  text: string;
  calls: ToolCallRequest[];
  toolsRejected: boolean;
  failed: boolean;
  /** This round's streaming window: first chunk → stream end, in ms. */
  streamMs: number;
}

/** What one streamed exchange measured.

    `streamMs` is the time tokens were actually arriving, summed over every
    tool round. Consent waits, tool runs and the follow-up request round-trips
    are waiting — not generation — so they stay out of it; a decode rate
    divided by them would report how patient the user is, not how fast the
    model writes. */
export interface StreamStats {
  streamMs: number;
}

/** Turn accumulated streaming deltas into calls in their original order. */
function toToolCalls(
  accumulated: Map<number, { id: string; name: string; args: string }>
): ToolCallRequest[] {
  const calls: ToolCallRequest[] = [];
  for (const [index, entry] of [...accumulated.entries()].sort((a, b) => a[0] - b[0])) {
    if (!entry.name) continue;
    let args: Record<string, unknown> = {};
    try {
      const parsed: unknown = JSON.parse(entry.args || "{}");
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        args = parsed as Record<string, unknown>;
      }
    } catch {
      // unparseable arguments: run with an empty object; the executor reports it
    }
    calls.push({ id: entry.id || `call_${index}`, name: entry.name, args, raw: entry.args || undefined });
  }
  return calls;
}

function toApiContent(message: ChatMessage): unknown {
  if (message.role === "user" && message.images && message.images.length > 0) {
    const parts: Record<string, unknown>[] = [];
    if (message.content) parts.push({ type: "text", text: message.content });
    for (const url of message.images) {
      parts.push({ type: "image_url", image_url: { url } });
    }
    return parts;
  }
  return message.content;
}

function parseUsage(raw: {
  prompt_tokens?: number;
  completion_tokens?: number;
} | undefined): Usage | null {
  if (!raw) return null;
  return {
    promptTokens: raw.prompt_tokens ?? 0,
    completionTokens: raw.completion_tokens ?? 0,
  };
}

export async function streamChat(
  config: LlmConfig,
  history: ChatMessage[],
  systemPrompt: string,
  onDelta: (delta: string) => void,
  onThinking: ((delta: string) => void) | undefined,
  onUsage: (usage: Usage) => void,
  onError: (message: string) => void,
  signal: AbortSignal,
  tools?: ChatToolHooks
): Promise<StreamStats> {
  const messages: Record<string, unknown>[] = [];
  if (systemPrompt) messages.push({ role: "system", content: systemPrompt });
  for (const m of history) {
    messages.push({ role: m.role, content: toApiContent(m) });
  }

  let toolsEnabled = !!tools && tools.defs.length > 0;
  const maxRounds = tools?.maxRounds ?? 4;

  /** One assistant round: send (retry ladder), stream, collect any tool calls. */
  const runRound = async (withTools: boolean): Promise<RoundResult> => {
    const defs = withTools && tools ? tools.defs : null;
    const payload = (extra: Record<string, unknown>): Record<string, unknown> => ({
      model: config.model,
      messages,
      stream: true,
      ...(defs ? { tools: defs, tool_choice: "auto" } : {}),
      ...extra,
    });
    const attempts = [
      // stream_options may be rejected (no usage report then)…
      payload({ reasoning_effort: config.reasoningEffort, stream_options: { include_usage: true } }),
      // …as may reasoning_effort.
      payload({ reasoning_effort: config.reasoningEffort }),
      payload({}),
    ];
    let res: Response | null = null;
    let toolsRejected = false;
    for (const body of attempts) {
      res = await postCompletions(config, body, signal);
      if (res.ok) break;
    }
    if (res && !res.ok && defs) {
      // The provider may reject the tools themselves: try once without them.
      const plain = await postCompletions(
        config,
        { model: config.model, messages, stream: true },
        signal
      );
      if (plain.ok) {
        res = plain;
        toolsRejected = true;
      }
    }
    if (!res || !res.ok) {
      let detail = res ? `Request failed with status ${res.status}` : "Request failed";
      if (res) {
        try {
          const body: unknown = await res.json();
          const msg = (body as { error?: { message?: string } })?.error?.message;
          if (msg) detail = msg;
        } catch {
          // non-JSON error body
        }
      }
      onError(detail);
      return { text: "", calls: [], toolsRejected: false, failed: true, streamMs: 0 };
    }

    let roundText = "";
    const accumulated = new Map<number, { id: string; name: string; args: string }>();
    const absorb = (
      list: { index?: number; id?: string; function?: { name?: string; arguments?: string } }[] | undefined
    ) => {
      if (!list) return;
      for (const tc of list) {
        const index = tc.index ?? 0;
        const cur = accumulated.get(index) ?? { id: "", name: "", args: "" };
        if (tc.id) cur.id = tc.id;
        if (tc.function?.name) cur.name = tc.function.name;
        if (tc.function?.arguments) cur.args += tc.function.arguments;
        accumulated.set(index, cur);
      }
    };

    const reader = res.body?.getReader();
    const isSse = (res.headers.get("content-type") ?? "").includes("event-stream");
    if (!reader || !isSse) {
      // Some providers answer a stream request with a single JSON body.
      const data: {
        choices?: {
          message?: {
            content?: string;
            reasoning?: unknown;
            reasoning_content?: unknown;
            tool_calls?: { id?: string; function?: { name?: string; arguments?: string } }[];
          };
        }[];
        usage?: { prompt_tokens?: number; completion_tokens?: number };
      } = await res.json();
      const msg = data.choices?.[0]?.message;
      const reasoning =
        typeof msg?.reasoning === "string" ? msg.reasoning : typeof msg?.reasoning_content === "string" ? msg.reasoning_content : "";
      if (reasoning) onThinking?.(reasoning);
      if (msg?.content) {
        roundText += msg.content;
        onDelta(msg.content);
      }
      absorb(msg?.tool_calls?.map((tc) => ({ ...tc, index: 0 })));
      const usage = parseUsage(data.usage);
      if (usage) onUsage(usage);
      // One JSON body, delivered whole: there is no streaming window to time.
      return { text: roundText, calls: toToolCalls(accumulated), toolsRejected, failed: false, streamMs: 0 };
    }

    const decoder = new TextDecoder();
    let buffer = "";
    // The round's streaming window (first chunk → stream end) is the time the
    // model was actually writing. It is measured per chunk arrival, so the
    // request's own latency — and any tool wait between rounds — stays out.
    let streamStart = 0;
    let streamEnd = 0;
    outer: for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      const now = performance.now();
      if (!streamStart) streamStart = now;
      streamEnd = now;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice(5).trim();
        if (payload === "[DONE]") break outer;
        try {
          const json: {
            choices?: {
              delta?: {
                content?: string;
                reasoning?: unknown;
                reasoning_content?: unknown;
                tool_calls?: { index?: number; id?: string; function?: { name?: string; arguments?: string } }[];
              };
            }[];
            usage?: { prompt_tokens?: number; completion_tokens?: number };
          } = JSON.parse(payload);
          const usage = parseUsage(json.usage);
          if (usage) onUsage(usage);
          const d = json.choices?.[0]?.delta;
          if (!d) continue;
          const reasoning =
            typeof d.reasoning === "string"
              ? d.reasoning
              : typeof d.reasoning_content === "string"
                ? d.reasoning_content
                  : "";
          if (reasoning) onThinking?.(reasoning);
          if (typeof d.content === "string" && d.content) {
            roundText += d.content;
            onDelta(d.content);
          }
          absorb(d.tool_calls);
        } catch {
          // partial JSON line, wait for more data
        }
      }
    }
    return {
      text: roundText,
      calls: toToolCalls(accumulated),
      toolsRejected,
      failed: false,
      streamMs: streamStart ? streamEnd - streamStart : 0,
    };
  };

  const stats: StreamStats = { streamMs: 0 };
  try {
    for (let round = 0; ; round++) {
      const outcome = await runRound(toolsEnabled);
      stats.streamMs += outcome.streamMs;
      if (outcome.failed) return stats;
      if (outcome.toolsRejected) toolsEnabled = false;
      if (!outcome.calls.length) return stats;
      // Echo the call back, then run it (or report the denial) and hand the
      // model the tool result: the next round continues the same reply.
      messages.push({
        role: "assistant",
        content: outcome.text || null,
        tool_calls: outcome.calls.map((call) => ({
          id: call.id,
          type: "function",
          function: { name: call.name, arguments: call.raw ?? JSON.stringify(call.args) },
        })),
      });
      for (const call of outcome.calls) {
        if (!tools || signal.aborted) return stats;
        tools.onCall(call);
        let allowed = false;
        try {
          allowed = await tools.approve(call);
        } catch {
          allowed = false;
        }
        if (signal.aborted) return stats;
        let result: string;
        if (!allowed) {
          tools.onDenied(call);
          result = "The user denied this tool call. Do not repeat it; answer with what you already know.";
        } else {
          try {
            result = await tools.execute(call);
          } catch (err) {
            result = `Error: ${err instanceof Error ? err.message : String(err)}`;
          }
        }
        messages.push({ role: "tool", tool_call_id: call.id, content: result });
      }
      // Out of rounds: drop the tools so the model must write its answer now.
      if (round + 1 >= maxRounds) toolsEnabled = false;
    }
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") return stats;
    onError(err instanceof Error ? err.message : String(err));
  }
  return stats;
}
