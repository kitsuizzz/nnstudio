/**
 * The platform-provided model: the deployment's gateway (same-origin by
 * default) holds the upstream key, so guests and fresh accounts need no
 * provider form. A synthetic "platform" provider points at that gateway with
 * no user key; a BYOK provider (an account's own key) takes precedence
 * whenever one is saved.
 *
 * `VITE_PLATFORM_BASE_URL` overrides the gateway URL (dev and the E2E gates
 * point it at the mock provider); `VITE_PLATFORM_MODEL` optionally pins the
 * default model — otherwise the first /v1/models entry is used.
 */

import type { LlmConfig, LlmProvider, ReasoningEffort } from "../types";

export const PLATFORM_NAME = "Studio Kitsui";
export const PLATFORM_BASE_URL =
  (import.meta.env.VITE_PLATFORM_BASE_URL as string | undefined)?.trim() ||
  // The gateway under the app's own mount: /v1, or /dev/v1 at /dev/.
  `${import.meta.env.BASE_URL}v1`;
export const PLATFORM_MODEL =
  (import.meta.env.VITE_PLATFORM_MODEL as string | undefined)?.trim() || "";

/**
 * The platform provider's stable id: a Server model choice must survive a
 * reload and a move to another device, so it cannot reference a random uid.
 */
export const PLATFORM_PROVIDER_ID = "platform";

/** Which side of the model picker a provider's models belong to. */
export type ModelCategory = "server" | "byok";

export function isPlatform(provider: LlmProvider | null | undefined): boolean {
  return provider?.mode === "platform";
}

/** Server = the deployment's gateway; every other provider is the user's own. */
export function providerCategory(provider: LlmProvider | null | undefined): ModelCategory {
  return isPlatform(provider) ? "server" : "byok";
}

/** A provider is usable when it carries its own key, needs none, or is the
    platform gateway. */
export function providerUsable(provider: LlmProvider | null | undefined): boolean {
  if (!provider) return false;
  return isPlatform(provider) || provider.noKey === true || provider.apiKey.trim().length > 0;
}

/** The LLM config for a provider — platform credentials resolved here. */
export function llmFor(provider: LlmProvider, model: string, effort: ReasoningEffort): LlmConfig {
  return {
    baseUrl: isPlatform(provider) ? provider.baseUrl || PLATFORM_BASE_URL : provider.baseUrl,
    apiKey: isPlatform(provider) ? "" : provider.apiKey,
    model,
    reasoningEffort: effort,
  };
}
