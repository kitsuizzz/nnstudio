import type { Agent } from "../types";

/**
 * The one-shot prompt behind the Code pane's AI assist. The brain's full
 * source rides in the system message between markers (the mock provider keys
 * its edit route off them); the user message is the change instruction. The
 * model must answer with the complete updated source in a single fence.
 */
export function codeEditSystem(appTitle: string, source: string): string {
  return [
    `You maintain the source code of the built-in "${appTitle}" app.`,
    "The full current source of the app's brain is between the markers below.",
    "A brain is a self-contained JavaScript program with no DOM access and no imports; it runs in a sandbox and talks to the host only through window.ai: ai.render({blocks}) emits UI blocks, ai.on(event, handler) receives events, ai.host(action, arg) asks the host to act, ai.fetch(kind, args) calls a host capability, ai.ready() signals readiness.",
    "The source must never use backticks or ${ sequences (it is embedded in an HTML script tag).",
    "Apply the user's requested change to the source, keeping every other line and the overall structure intact.",
    "Reply with the complete updated source inside a single ```js code fence and nothing else.",
    "",
    "===BRAIN-SOURCE-BEGIN===",
    source,
    "===BRAIN-SOURCE-END===",
  ].join("\n");
}

/**
 * The triage protocol: the main AI decides whether a workspace is warranted,
 * asks at most ONE clarifying question per turn, and delegates to the builder
 * once it has enough information. It never builds the spec itself.
 */
export function studioTriagePrompt(studios?: { title: string; id: string }[]): string {
  const studioList =
    studios && studios.length > 0
      ? [
          "",
          "Workspaces already built in this conversation:",
          ...studios.map((s) => `- "${s.title}" (id: ${s.id})`),
        ]
      : [];

  return [
    "Workspaces",
    "",
    "You can build a workspace: a real, interactive APP that takes over the entire window. Use one whenever the request is better served by seeing and doing than by reading: games, demos, tutorials, visualizers, simulators, trainers, calculators, dashboards, forms.",
    "For ordinary questions, answer in plain chat — no fences.",
    ...studioList,
    "",
    "Triage protocol (you do NOT build the spec yourself):",
    "",
    "1. If the request is under-specified (vague goal, missing key parameters, multiple valid interpretations), ask ONE clarifying question and stop. Use:",
    "```questions",
    '{"question":"…","options":["…","…"],"assumptions":["…"]}',
    "```",
    "   - Exactly ONE question per reply. Wait for the user's answer before asking the next.",
    "   - Options: 2–4 short choices the user can click. Omit if open-ended.",
    "   - Assumptions: what you'd default to if the user says \"just build it\".",
    "",
    "2. If the request is clear enough to build a NEW workspace, emit:",
    "```build",
    '{"topic":"one-line description of what to build"}',
    "```",
    "   Then stop. Do NOT emit plan, progress, or studio fences — those belong to the builder.",
    "",
    "3. If the user is asking to change, fix, or improve a workspace that was already built in this conversation, emit:",
    "```edit",
    '{"studio":"<title of the existing workspace>","change":"one-line description of the requested change"}',
    "```",
    "   The user may attach a screenshot of the workspace showing what they see. Then stop.",
    "",
    "Never emit more than one fence type per reply (either questions OR build OR edit, not multiple).",
  ].join("\n");
}

/**
 * Replies render as Markdown with typeset math: the model should write LaTeX
 * for formulas ($…$ inline, $$…$$ display) instead of Unicode approximations.
 */
const MATH_PROMPT =
  "Formatting: replies render as Markdown and LaTeX math is typeset — write inline math as $…$ and display equations as $$…$$ on their own lines.";

/**
 * How the agent uses the web tools. The schemas themselves ride in the
 * request (native tool calling); this is the usage guidance: when to reach
 * for search vs. fetch, and how to cite what it used.
 */
export function webToolsPrompt(enabled: { search: boolean; fetch: boolean }): string {
  const tools: string[] = [];
  if (enabled.search) tools.push("web_search");
  if (enabled.fetch) tools.push("web_fetch");
  if (tools.length === 0) return "";
  return [
    "Web tools: " + tools.join(" and ") + ".",
    "- Search before answering anything about current events, releases, prices, or facts that may have changed after your training cutoff — and when the user asks you to look something up.",
    "- Fetch a page when the user gives you a URL or when a search result looks worth reading in full.",
    "- If a call is declined or fails, continue with what you know and say so briefly. Never invent a result a declined call might have produced.",
    "- The chat shows no list of searches or sources, so name the sources you actually used in your answer as markdown links, and never make up URLs.",
  ].join("\n");
}

export function buildAgentSystemPrompt(
  agent: Agent,
  appCode?: string,
  studios?: { title: string; id: string }[],
  webTools?: { search: boolean; fetch: boolean }
): string {
  const parts: string[] = [];
  const prompt = agent.systemPrompt.trim();
  if (prompt) parts.push(prompt);
  parts.push(MATH_PROMPT);
  // The built-in capabilities: the app builder's triage, the web tools' guidance
  // and the built-in apps' source. A custom agent without them is a plain chat.
  if (agent.builtins) {
    parts.push(studioTriagePrompt(studios));
    if (webTools) {
      const web = webToolsPrompt(webTools);
      if (web) parts.push(web);
    }
    if (appCode) parts.push(appCode);
  }
  return parts.join("\n\n");
}
