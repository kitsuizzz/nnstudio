/**
 * The provider and model an agent rides. Every chat turn resolves its model
 * through the conversation's agent: the default agent when none is picked.
 */

import type { Conversation, LlmProvider, ModelChoice, Workspace } from "../types";
import { CHAT_KITSUI_ID, effectiveAgent } from "./agents";

export function providerFor(workspace: Workspace, choice: ModelChoice | null): LlmProvider | null {
  if (!choice) return null;
  return workspace.providers.find((p) => p.id === choice.providerId) ?? null;
}

/** Chat Kitsui's provider and model: the default for new conversations. */
export function defaultChoice(workspace: Workspace): ModelChoice | null {
  const chat = workspace.agents.find((a) => a.id === CHAT_KITSUI_ID) ?? workspace.agents.find((a) => a.builtIn);
  return chat && chat.providerId ? { providerId: chat.providerId, model: chat.model } : null;
}

/** What a conversation rides: its agent's provider and model. */
export function effectiveChoice(workspace: Workspace, conversation: Pick<Conversation, "agentId"> | null): ModelChoice | null {
  const agent = effectiveAgent(workspace, conversation);
  return agent && agent.providerId ? { providerId: agent.providerId, model: agent.model } : null;
}
