import { useEffect, useRef } from "react";
import { wallBrain } from "../lib/wallApp";
import { Block, type ComposerHost } from "./studio/blocks";
import { appCtx, type AppEventHandler } from "./studio/hosted";
import type { Conversation, CurrentMeta, SendMode, Studio } from "../types";

interface ExhibitionProps {
  conversation: Conversation;
  /** Title as currently typed out by the typewriter. */
  title: string;
  current: CurrentMeta;
  streaming: boolean;
  firstTokenReceived: boolean;
  sendMode: SendMode;
  /** Why the composer can't send right now, if anything. */
  sendBlocker: string | null;
  /** Centre the works on the wall (sidebar collapsed). */
  centered?: boolean;
  /** AI-built workspaces the door cards can open. */
  studios: Studio[];
  onOpenStudio: (id: string) => void;
  /** Re-send the last prompt (offered on a workspace that failed to build). */
  onRetry?: () => void;
  onSend: (text: string, images: string[]) => void;
  onStop: () => void;
  /** Reports whether the large wall text is on screen (drives the running head). */
  onWallTextVisible: (visible: boolean) => void;
}

const PIN_THRESHOLD_PX = 120;

/**
 * A conversation, hung on the wall — rendered as the "conversation" default
 * app: the wall brain emits registry blocks (header, one work per exchange, a
 * docked composer) and the shared component library renders them. This frame
 * owns only the window: the scroll container, the running-head observer, and
 * the pinned-to-bottom composer.
 */
export default function Exhibition({
  conversation,
  title,
  current,
  streaming,
  firstTokenReceived,
  sendMode,
  sendBlocker,
  centered = false,
  studios,
  onOpenStudio,
  onRetry,
  onSend,
  onStop,
  onWallTextVisible,
}: ExhibitionProps) {
  const { messages } = conversation;
  const scrollRef = useRef<HTMLDivElement>(null);
  const wallTextRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);
  const prevLen = useRef(messages.length);
  const prevId = useRef(conversation.id);

  // Follow the stream only while the reader is near the bottom; a new message
  // or a conversation switch always re-pins.
  useEffect(() => {
    if (messages.length !== prevLen.current || conversation.id !== prevId.current) {
      prevLen.current = messages.length;
      prevId.current = conversation.id;
      pinnedRef.current = true;
    }
    const el = scrollRef.current;
    if (el && pinnedRef.current) el.scrollTop = el.scrollHeight;
  }, [messages, streaming, conversation.id]);

  useEffect(() => {
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
  }, [onWallTextVisible, conversation.id]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < PIN_THRESHOLD_PX;
  };

  const content = wallBrain({
    conversation,
    title,
    current,
    streaming,
    firstTokenReceived,
    studios,
    sendMode,
    sendBlocker,
  });

  const onEvent: AppEventHandler = (block, event, arg) => {
    if (block.type === "door" && event === "click") onOpenStudio(arg as string);
    else if (block.type === "interrupted" && event === "click" && arg === "retry") onRetry?.();
  };
  const composer: ComposerHost = {
    onSend: (text, images) => onSend(text, images),
    onStop,
    disabled: streaming,
    sendMode,
    sendBlocker,
  };
  const ctx = appCtx(onEvent, composer);

  return (
    <main className="exhibition">
      <div
        className={`wall${centered ? " centered" : ""}`}
        ref={scrollRef}
        onScroll={onScroll}
        role="log"
      >
        <div ref={wallTextRef}>
          <Block block={content.head} ctx={ctx} />
        </div>

        {conversation.messages.length === 0 && (
          <div className="wall-empty">
            <p>Start a conversation below.</p>
            <p className="wall-empty-hint">
              Your message is hung on the right like a wall label; the reply is the work on the left.
            </p>
          </div>
        )}

        {content.works.map((w, i) => (
          <Block key={i} block={w} ctx={ctx} />
        ))}

        <Block block={content.dock[0]} ctx={ctx} />
      </div>
    </main>
  );
}
