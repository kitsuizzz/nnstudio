import { useMemo } from "react";
import { greetingSource } from "./studio/brains/greeting";
import BrainHost from "./studio/BrainHost";
import { Block, type BlockCtx, type ComposerHost } from "./studio/blocks";
import { fetchModels, suggestIdeas, testProvider } from "../lib/api";
import type { GuestSession, StudioBlock } from "../types";

export interface GuestStart {
  providerName: string;
  baseUrl: string;
  apiKey: string;
  model: string;
}

interface GreetingFrameProps {
  /** Provider settings confirmed; the wall switches to the message box. */
  onStart: (config: GuestStart) => void;
  /** The guest overrides the model the AI picked. */
  onModelChange?: (model: string) => void;
  /** A suggested first move was picked; send it as the first message. */
  onFirstMessage?: (text: string) => void;
  /** The host-wired message box, docked once settings are in. */
  composer: ComposerHost;
  /** The app's effective brain source (override-aware, from the App). */
  source?: string;
  /** The guest's saved onboarding (cookie), so a reload can resume it. */
  guest?: GuestSession | null;
  /** The platform-provided models (the deployment's gateway), when available. */
  platform?: { baseUrl: string; model: string; models?: unknown[] } | null;
  /** True when a usable provider already exists (the platform gateway or a
      saved BYOK workspace): "Continue as guest" can start a session directly
      instead of opening the provider form. */
  canStart?: boolean;
  /** The brain asked to start a fresh conversation (platform mode's guest
      path, or the end of the BYOK form) — the host routes to a new session. */
  onNewSession?: () => void;
  /** Open the auth dialog (the brain's "Create an account" / "Sign up" buttons). */
  onOpenAuth?: (mode: "signin" | "signup") => void;
  /** The host's account state; the brain uses it to skip the choice step. */
  signedIn?: boolean;
}

/**
 * The welcome screen, hung as the "greeting" default app. The greeting brain
 * (a sandbox) owns the whole onboarding flow and emits registry blocks; this
 * frame owns only the stage — splitting the brain's flat block list back into
 * the persistent head and the per-step body — and relays the brain's
 * `ai.host` / `ai.fetch` calls to the app.
 */
export default function GreetingFrame({ onStart, onModelChange, onNewSession, onFirstMessage, composer, source, guest, platform, canStart, onOpenAuth, signedIn }: GreetingFrameProps) {
  const onHost = (action: string, arg?: unknown) => {
    if (action === "start") onStart(arg as GuestStart);
    else if (action === "model-change") onModelChange?.(arg as string);
    else if (action === "new-session") onNewSession?.();
    else if (action === "first-message") onFirstMessage?.(arg as string);
    else if (action === "auth") onOpenAuth?.(arg as "signin" | "signup");
  };

  // The whitelisted host capabilities the brain may call across the bridge.
  const onFetch = (kind: string, args?: unknown) => {
    const a = (Array.isArray(args) ? args : []) as string[];
    if (kind === "models") return fetchModels(a[0], a[1]);
    if (kind === "test") return testProvider(a[0], a[1]);
    if (kind === "ideas") return suggestIdeas(a[0], a[1], a[2]);
    return Promise.reject(new Error("unknown fetch kind: " + kind));
  };

  const ctxData = useMemo(
    () => ({ guest: guest ?? null, signedIn: !!signedIn, platform: platform ?? null, canStart: !!canStart }),
    [guest, signedIn, platform, canStart]
  );

  return (
    <BrainHost
      source={source ?? greetingSource}
      title="Welcome"
      trusted
      composer={composer}
      ctxData={ctxData}
      onHost={onHost}
      onFetch={onFetch}
    >
      {({ blocks, ctx }: { blocks: StudioBlock[] | null; ctx: BlockCtx }) => {
        // The head (typewriter title) stays mounted across steps; the body
        // group carries a per-step key so each step remounts cleanly.
        const head = blocks ? blocks.find((b) => b.type === "greetinghead") : undefined;
        const body = blocks ? blocks.filter((b) => b.type !== "greetinghead") : [];
        return (
          <main className="greeting">
            {head && <Block block={head} ctx={ctx} />}
            {body.map((b, i) => (
              <Block key={b.type === "group" ? (b.key ?? b.className) : String(i)} block={b} ctx={ctx} />
            ))}
          </main>
        );
      }}
    </BrainHost>
  );
}
