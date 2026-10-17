import { useEffect, useRef, useState } from "react";
import { wallSource } from "./studio/brains/wall";
import BrainHost from "./studio/BrainHost";
import { Block, type BlockCtx, type ComposerHost } from "./studio/blocks";
import type { Conversation, CurrentMeta, SendMode, Studio } from "../types";
import { useFollowScroll } from "./useFollowScroll";
import type { ReadingSpot } from "./readingSpot";

interface WallFrameProps {
  conversation: Conversation;
  /** Title as currently typed out by the typewriter. */
  title: string;
  current: CurrentMeta;
  streaming: boolean;
  firstTokenReceived: boolean;
  sendMode: SendMode;
  /** Why the composer can't send right now, if anything. */
  sendBlocker: string | null;
  /** A builder stream is in progress (the composer is locked). */
  building?: boolean;
  /** Speed counter (tok/s) for the streaming exchange; decode stays null until the stream settles. */
  speeds?: { prefill: number; decode: number | null } | null;
  /** The app's effective brain source (override-aware, from the App). */
  source?: string;
  /** Centre the works on the wall (sidebar collapsed). */
  centered?: boolean;
  /** AI-built workspaces the door cards can open. */
  studios: Studio[];
  /** Build threads spawned from this conversation, projected for their cards. */
  threads?: { id: number; title: string; status: string; active: boolean }[];
  /** "build" for a dedicated build thread (renders the full build pieces). */
  kind?: "chat" | "build";
  onOpenStudio: (id: string) => void;
  /** Open a build thread (the thread card). */
  onOpenThread?: (id: number) => void;
  /** Re-send the last prompt (offered on a workspace that failed to build). */
  onRetry?: () => void;
  /** The user confirmed a plan — start the build. */
  onBuild?: () => void;
  onSend: (text: string, images: string[]) => void;
  onStop: () => void;
  /** Reports whether the large wall text is on screen (drives the running head). */
  onWallTextVisible: (visible: boolean) => void;
  /** The reader's last spot in a conversation, restored on open. */
  readingSpotFor?: (conversationId: number) => ReadingSpot | null;
  /** The reader moved: remember where they are in the conversation. */
  onReadingSpot?: (conversationId: number, spot: ReadingSpot) => void;
}

/**
 * A conversation, hung on the wall — the "conversation" default app. The wall
 * brain (a sandbox) reads a live projection of the conversation and emits the
 * registry blocks; this frame owns only the window — the scroll container, the
 * running-head observer, and the pinned-to-bottom composer — and splits the
 * brain's flat block list back into head / works / dock.
 */
export default function WallFrame({
  conversation,
  title,
  current,
  streaming,
  firstTokenReceived,
  sendMode,
  sendBlocker,
  building,
  speeds,
  source,
  centered = false,
  studios,
  threads,
  kind,
  onOpenStudio,
  onOpenThread,
  onRetry,
  onBuild,
  onSend,
  onStop,
  onWallTextVisible,
  readingSpotFor,
  onReadingSpot,
}: WallFrameProps) {
  const { messages } = conversation;
  const scrollRef = useRef<HTMLDivElement>(null);
  const wallTextRef = useRef<HTMLDivElement>(null);
  const [brainReady, setBrainReady] = useState(false);
  /** The wall brain never rendered (a boot crash/hang): say so instead of a blank wall. */
  const [brainFailed, setBrainFailed] = useState(false);

  // The live snapshot the brain reads. Studios are projected down to the
  // id/title the doors need; the full specs stay host-side.
  const ctxData = {
    messages,
    createdAt: conversation.createdAt,
    conversationId: conversation.id,
    title,
    current,
    streaming,
    firstTokenReceived,
    studios: studios.map((s) => ({ id: s.id, title: s.title })),
    threads: threads ?? [],
    kind: kind ?? "chat",
    sendMode,
    sendBlocker,
    building,
    speeds,
  };

  useFollowScroll(scrollRef, conversation.id, messages, streaming, brainReady, {
    spotFor: readingSpotFor,
    onSpot: onReadingSpot ? (spot) => onReadingSpot(conversation.id, spot) : undefined,
  });

  // The running head reports when the large wall title leaves the viewport.
  // The observer waits for the brain to emit the head (the wall-text element).
  useEffect(() => {
    if (!brainReady) return;
    const root = scrollRef.current;
    const target = wallTextRef.current;
    if (!root || !target || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(
      (entries) => {
        const e = entries[0];
        if (e) onWallTextVisible(e.isIntersecting);
      },
      { root, threshold: 0 }
    );
    io.observe(target);
    return () => io.disconnect();
  }, [brainReady, onWallTextVisible, conversation.id]);

  const composer: ComposerHost = {
    onSend: (text, images) => onSend(text, images),
    onStop,
    disabled: streaming || !!building,
    sendMode,
    sendBlocker,
  };

  const onHost = (action: string, arg?: unknown) => {
    if (action === "open-studio") onOpenStudio(arg as string);
    else if (action === "open-thread" && typeof arg === "string") onOpenThread?.(Number(arg));
    else if (action === "retry") onRetry?.();
    else if (action === "build") onBuild?.();
    else if (action === "option" && typeof arg === "string" && arg) onSend(arg, []);
    else if (action === "skip-questions") onSend("Just build it", []);
  };

  return (
    <BrainHost
      source={source ?? wallSource}
      title="Conversation wall"
      trusted
      ctxData={ctxData}
      composer={composer}
      onHost={onHost}
      onReady={() => setBrainReady(true)}
      onBootFailure={() => setBrainFailed(true)}
    >
      {({ blocks, ctx }: { blocks: import("../types").StudioBlock[] | null; ctx: BlockCtx }) => {
        const head = blocks ? blocks.find((b) => b.type === "wallhead") : undefined;
        const works = blocks ? blocks.filter((b) => b.type === "work") : [];
        const dock = blocks ? blocks.find((b) => b.type === "composer") : undefined;
        return (
          <main className="exhibition">
            <div
              className={`wall${centered ? " centered" : ""}`}
              ref={scrollRef}
              role="log"
            >
              <div ref={wallTextRef}>
                {head && <Block block={head} ctx={ctx} />}
              </div>

              {conversation.messages.length === 0 && (
                <div className="wall-empty">
                  <p>Start a conversation below.</p>
                  <p className="wall-empty-hint">
                    Your message is hung on the right like a wall label; the reply is the work on the left.
                  </p>
                </div>
              )}

              {!blocks && (
                // The wall brain hasn't rendered (yet, or at all). Never show
                // an empty frame with no composer — a blank wall reads as
                // "the app is gone".
                <div className="wall-empty">
                  {brainFailed ? (
                    <>
                      <p>The conversation view didn't start.</p>
                      <p className="wall-empty-hint">
                        <button type="button" className="btn ghost" onClick={() => window.location.reload()}>
                          Reload
                        </button>
                      </p>
                    </>
                  ) : (
                    <p>Starting up…</p>
                  )}
                </div>
              )}

              {works.map((w, i) => (
                <Block key={i} block={w} ctx={ctx} />
              ))}

              {dock && <Block block={dock} ctx={ctx} />}
            </div>
          </main>
        );
      }}
    </BrainHost>
  );
}
