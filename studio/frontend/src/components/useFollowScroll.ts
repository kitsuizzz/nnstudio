import { useEffect, useRef, type RefObject } from "react";
import { RESUME_PX, spotAt, targetFor, type ReadingSpot } from "./readingSpot";

/** How long a restored position is held while the wall settles (fonts, math, the first blocks). */
const RESTORE_MS = 1500;

/** The wall's scroll state, kept off-render so scrolling costs no re-renders. */
interface FollowState {
  pinned: boolean;
  lastTop: number;
  count: number;
  conversation: number;
  restoring: boolean;
  frame: number | null;
  restoredFor: number | null;
  lastSpot: string;
}

/** The conversation's reading position: where to restore it from, and where to report it. */
export interface ReadingHooks {
  spotFor?: (conversationId: number) => ReadingSpot | null;
  onSpot?: (spot: ReadingSpot) => void;
}

/**
 * Keeps a conversation pinned to its newest text while a reply streams, lets the
 * reader step away without a fight, and remembers where they were.
 *
 * Following is a state, not a distance: any upward scroll (the wheel, a touch
 * drag, the keys, the scrollbar) releases it at once, so the next token cannot
 * pull the reader back. Following resumes only when the reader scrolls back to
 * the bottom. A new message always pins. On open, the saved position is restored;
 * the reader's first scroll or key stops that.
 *
 * A vertical wheel over a box inside the scroller that clips its own overflow but
 * has nothing to scroll vertically (a display formula, a code block) is handed to
 * the scroller, since such a box would otherwise swallow the wheel.
 */
export function useFollowScroll(
  scrollRef: RefObject<HTMLElement | null>,
  conversationId: number,
  messages: readonly unknown[],
  streaming: unknown,
  ready: boolean,
  reading?: ReadingHooks
): void {
  const stateRef = useRef<FollowState>({
    pinned: true,
    lastTop: 0,
    count: messages.length,
    conversation: conversationId,
    restoring: false,
    frame: null,
    restoredFor: null,
    lastSpot: "",
  });
  const readingRef = useRef(reading);

  useEffect(() => {
    readingRef.current = reading;
  });

  // A new message, or a new conversation, pins to the bottom; otherwise the
  // newest text is followed only while pinned. The messages array changes on every token.
  useEffect(() => {
    const st = stateRef.current;
    if (messages.length !== st.count || conversationId !== st.conversation) {
      cancelRestore(st);
      st.count = messages.length;
      st.conversation = conversationId;
      st.pinned = true;
    }
    const el = scrollRef.current;
    if (el && st.pinned) {
      el.scrollTop = el.scrollHeight;
      st.lastTop = el.scrollTop;
    }
  }, [scrollRef, messages, streaming, conversationId]);

  // Open the conversation where the reader left it, once the wall is ready for it.
  useEffect(() => {
    const st = stateRef.current;
    if (!ready) {
      st.restoredFor = null;
      return;
    }
    if (st.restoredFor === conversationId) return;
    st.restoredFor = conversationId;
    restoreSpot(scrollRef, st, readingRef.current?.spotFor?.(conversationId) ?? null, readingRef.current?.onSpot);
  }, [scrollRef, ready, conversationId]);

  // The reader's intent, and the reading position as it changes.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const st = stateRef.current;
    st.lastTop = el.scrollTop;
    let lastY: number | null = null;

    const release = () => {
      st.pinned = false;
    };
    const onScroll = () => {
      const top = el.scrollTop;
      const gap = el.scrollHeight - top - el.clientHeight;
      if (top > st.lastTop && gap <= RESUME_PX) st.pinned = true;
      st.lastTop = top;
      if (!st.restoring) reportSpot(el, st, readingRef.current?.onSpot);
    };
    const onWheel = (e: WheelEvent) => {
      cancelRestore(st);
      if (e.deltaY < 0) release();
      if (e.ctrlKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;
      if (!swallowsVerticalWheel(e.target, el)) return;
      e.preventDefault();
      const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? el.clientHeight : 1;
      el.scrollTop += e.deltaY * unit;
    };
    const onTouchStart = (e: TouchEvent) => {
      cancelRestore(st);
      lastY = e.touches[0]?.clientY ?? null;
    };
    const onTouchMove = (e: TouchEvent) => {
      const y = e.touches[0]?.clientY;
      if (y === undefined || lastY === null) return;
      if (y > lastY + 2) release();
      lastY = y;
    };
    const onPointerDown = (e: PointerEvent) => {
      if (e.target === el && e.clientX > el.getBoundingClientRect().left + el.clientWidth) {
        cancelRestore(st);
        release();
      }
    };
    const onKey = (e: KeyboardEvent) => {
      const t = e.target;
      if (t instanceof HTMLElement && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
      if (e.key === "ArrowUp" || e.key === "PageUp" || e.key === "Home") release();
      if (["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "].includes(e.key)) cancelRestore(st);
    };

    el.addEventListener("scroll", onScroll, { passive: true });
    el.addEventListener("wheel", onWheel, { passive: false });
    el.addEventListener("touchstart", onTouchStart, { passive: true });
    el.addEventListener("touchmove", onTouchMove, { passive: true });
    el.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKey);
    return () => {
      el.removeEventListener("scroll", onScroll);
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("touchstart", onTouchStart);
      el.removeEventListener("touchmove", onTouchMove);
      el.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [scrollRef, ready, conversationId]);
}

/** Stop any restore in progress: the reader has taken over. */
function cancelRestore(st: FollowState): void {
  st.restoring = false;
  if (st.frame !== null) {
    cancelAnimationFrame(st.frame);
    st.frame = null;
  }
}

/** Report the spot when it changes, not on every scroll event. */
function reportSpot(el: HTMLElement, st: FollowState, onSpot?: (spot: ReadingSpot) => void): void {
  const spot = spotAt(el);
  const json = JSON.stringify(spot);
  if (json === st.lastSpot) return;
  st.lastSpot = json;
  onSpot?.(spot);
}

/** Put the reader back at a saved spot, holding it while the wall settles. */
function restoreSpot(
  scrollRef: RefObject<HTMLElement | null>,
  st: FollowState,
  spot: ReadingSpot | null,
  onSpot?: (spot: ReadingSpot) => void
): void {
  cancelRestore(st);
  if (!spot) {
    st.pinned = true;
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
    return;
  }
  st.pinned = spot.following;
  st.restoring = true;
  const deadline = performance.now() + RESTORE_MS;
  const step = (): void => {
    st.frame = null;
    if (!st.restoring) return;
    const el = scrollRef.current;
    if (el) {
      el.scrollTop = targetFor(el, spot);
      st.lastTop = el.scrollTop;
    }
    if (performance.now() < deadline) {
      st.frame = requestAnimationFrame(step);
      return;
    }
    st.restoring = false;
    if (el) reportSpot(el, st, onSpot);
  };
  step();
}

/** True when a box between the wheel's target and the scroller clips its overflow but cannot scroll vertically. */
function swallowsVerticalWheel(target: EventTarget | null, scroller: HTMLElement): boolean {
  let swallowed = false;
  for (let node = target instanceof Element ? target : null; node && node !== scroller; node = node.parentElement) {
    const cs = getComputedStyle(node);
    const clips = [cs.overflowX, cs.overflowY].some((v) => v !== "visible" && v !== "clip");
    if (!clips) continue;
    if (node.scrollHeight > node.clientHeight + 1) return false;
    swallowed = true;
  }
  return swallowed;
}
