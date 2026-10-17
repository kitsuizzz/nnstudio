/**
 * Agents: the unit the chat talks to. Chat Kitsui is the built-in agent every
 * workspace has: the app's own prompts and capabilities, powered by whatever
 * provider and model the user set up. Custom agents are the user's own: name,
 * provider, model, reasoning effort, system prompt, and whether they keep the
 * built-in capabilities.
 */

import type { Agent, ReasoningEffort, Workspace } from "../types";
import type { ModelCategory } from "./platform";
import { openSecret, sealSecret } from "./crypto";

export const CHAT_KITSUI_ID = "chat-kitsui";
export const CHAT_KITSUI_NAME = "Chat Kitsui";
export const REASONING_EFFORTS: ReasoningEffort[] = ["low", "medium", "xhigh"];

/** One row of the agent picker. */
export interface AgentOption {
  id: string;
  name: string;
  model: string;
  providerName: string;
  category: ModelCategory;
  usable: boolean;
}

/** The built-in agent: first, never removed. Its provider and model are the user's; its prompts and capabilities are the app's. */
export function chatKitsui(providerId: string, model: string, reasoningEffort: ReasoningEffort = "low"): Agent {
  return {
    id: CHAT_KITSUI_ID,
    name: CHAT_KITSUI_NAME,
    providerId,
    model,
    reasoningEffort,
    systemPrompt: "",
    builtins: true,
    builtIn: true,
  };
}

/** Repair an agent read from storage: an unknown effort falls back to low, and the built-in keeps its fixed name and capabilities. */
export function normalizeAgent(raw: Partial<Agent>): Agent | null {
  if (typeof raw.id !== "string" || !raw.id) return null;
  const effort = REASONING_EFFORTS.includes(raw.reasoningEffort as ReasoningEffort)
    ? (raw.reasoningEffort as ReasoningEffort)
    : "low";
  const providerId = typeof raw.providerId === "string" ? raw.providerId : "";
  const model = typeof raw.model === "string" ? raw.model : "";
  if (raw.id === CHAT_KITSUI_ID || raw.builtIn === true) return chatKitsui(providerId, model, effort);
  return {
    id: raw.id,
    name: typeof raw.name === "string" && raw.name.trim() ? raw.name.trim() : "Agent",
    providerId,
    model,
    reasoningEffort: effort,
    systemPrompt: typeof raw.systemPrompt === "string" ? raw.systemPrompt : "",
    builtins: raw.builtins !== false,
    vision: raw.vision,
  };
}

/** Make sure Chat Kitsui is present and first. `fallback` applies only when it is missing. */
export function ensureChatKitsui(agents: Agent[], fallback: { providerId: string; model: string }): Agent[] {
  const existing = agents.find((a) => a.builtIn);
  const chat = existing
    ? chatKitsui(existing.providerId, existing.model, existing.reasoningEffort)
    : chatKitsui(fallback.providerId, fallback.model);
  return [chat, ...agents.filter((a) => !a.builtIn)];
}

/** Point Chat Kitsui at a provider and model. It becomes the default agent unless `makeDefault` is false. */
export function pointChatKitsui(ws: Workspace, providerId: string, model: string, makeDefault = true): Workspace {
  const hasChat = ws.agents.some((a) => a.builtIn);
  const agents = hasChat
    ? ws.agents.map((a) => (a.builtIn ? { ...a, providerId, model } : a))
    : [chatKitsui(providerId, model), ...ws.agents];
  return { ...ws, agents, activeAgentId: makeDefault ? CHAT_KITSUI_ID : ws.activeAgentId };
}

/** The agent a conversation rides: its own pick, else the default agent, else Chat Kitsui. */
export function effectiveAgent(ws: Workspace, conversation: { agentId?: string } | null): Agent | null {
  const picked = conversation?.agentId ? ws.agents.find((a) => a.id === conversation.agentId) : undefined;
  return picked ?? ws.agents.find((a) => a.id === ws.activeAgentId) ?? ws.agents.find((a) => a.builtIn) ?? ws.agents[0] ?? null;
}

/** A profile's sealed agents, opened with the account key: the list and the default. */
export async function openAgentsRecord(
  key: CryptoKey | null,
  sealed: string | null
): Promise<{ agents: Agent[]; defaultAgentId: string } | null> {
  if (!sealed || !key) return null;
  try {
    const data = JSON.parse(await openSecret(key, sealed)) as { agents?: Partial<Agent>[]; defaultAgentId?: string };
    const agents = (data.agents ?? []).map((a) => normalizeAgent(a)).filter((a): a is Agent => a !== null);
    return { agents, defaultAgentId: data.defaultAgentId ?? CHAT_KITSUI_ID };
  } catch {
    return null;
  }
}

/** The agents sealed under the account key, ready for the profile (opaque to the server). */
export async function sealAgentsRecord(
  key: CryptoKey | null,
  agents: Agent[],
  defaultAgentId: string | null
): Promise<string | null> {
  if (!key) return null;
  return sealSecret(key, JSON.stringify({ agents, defaultAgentId: defaultAgentId ?? CHAT_KITSUI_ID }));
}
