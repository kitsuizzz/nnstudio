import { useEffect, useRef, useState } from "react";
import { ChevronDownIcon } from "./Icon";
import { formatDuration } from "../lib/format";

interface StudyProps {
  thinking: string;
  active: boolean;
  thinkingMs?: number;
}

/**
 * The model's reasoning, shown in the label column in pencil gray.
 * Closed by default — even while the model is thinking the label just ticks
 * "Thinking… Xs". Click to expand (clamped + auto-scrolling while active);
 * it collapses back to a single line once the reply begins to arrive.
 */
export default function Study({ thinking, active, thinkingMs }: StudyProps) {
  const [open, setOpen] = useState(false);
  const [elapsedMs, setElapsedMs] = useState(0);
  const prevActive = useRef(active);
  const startRef = useRef<number | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const bodyRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (active !== prevActive.current) {
      prevActive.current = active;
      if (!active) setOpen(false);
    }
  }, [active]);

  useEffect(() => {
    if (active) {
      startRef.current = performance.now();
      setElapsedMs(0);
      timerRef.current = setInterval(() => {
        if (startRef.current != null) {
          setElapsedMs(performance.now() - startRef.current);
        }
      }, 100);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [active]);

  useEffect(() => {
    const el = bodyRef.current;
    if (active && el) el.scrollTop = el.scrollHeight;
  }, [thinking, active]);

  const shown = open;
  const finalMs = active ? elapsedMs : (thinkingMs ?? elapsedMs);

  return (
    <div className={`study${shown ? " open" : ""}${active ? " active" : ""}`}>
      <button
        className="study-toggle"
        type="button"
        aria-expanded={shown}
        onClick={() => setOpen((o) => !o)}
      >
        <span className="study-label">
          {active ? `Thinking… ${formatDuration(finalMs)}` : `Thought for ${formatDuration(finalMs)}`}
        </span>
        <span className="study-chevron" aria-hidden="true">
          <ChevronDownIcon size={11} />
        </span>
      </button>
      {shown && (
        <div className={`study-body${active ? " clamped" : ""}`} ref={bodyRef}>
          {thinking || (active ? "…" : "")}
        </div>
      )}
    </div>
  );
}
