/**
 * Provider presets for the settings page's BYOK editor. The greeting brain
 * keeps its own copy (a sandboxed brain source cannot import), so these two
 * lists are deliberately parallel — update both when a provider is added.
 */

export interface ProviderPreset {
  name: string;
  baseUrl: string;
  models: string[];
}

export const PROVIDER_PRESETS: ProviderPreset[] = [
  { name: "OpenAI", baseUrl: "https://api.openai.com/v1", models: ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"] },
  { name: "Anthropic", baseUrl: "https://api.anthropic.com/v1", models: ["claude-sonnet-4-5", "claude-opus-4-1", "claude-haiku-4-5"] },
  { name: "Google", baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai", models: ["gemini-2.5-pro", "gemini-2.5-flash"] },
  { name: "OpenRouter", baseUrl: "https://openrouter.ai/api/v1", models: ["anthropic/claude-sonnet-4.5", "openai/gpt-4o"] },
  { name: "Mistral", baseUrl: "https://api.mistral.ai/v1", models: ["mistral-large-latest", "mistral-small-latest"] },
  { name: "Groq", baseUrl: "https://api.groq.com/openai/v1", models: ["llama-3.3-70b-versatile"] },
  { name: "Custom", baseUrl: "", models: [] },
];
