import { useEffect, useRef, useState } from "react";

function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

export function useTypewriter(text: string, trigger: number, speed = 25): string {
  const [display, setDisplay] = useState(text);
  const prevTrigger = useRef(trigger);

  useEffect(() => {
    if (trigger === prevTrigger.current) {
      setDisplay(text);
      return;
    }
    prevTrigger.current = trigger;
    if (!text || prefersReducedMotion()) {
      setDisplay(text);
      return;
    }
    let i = 0;
    setDisplay("");
    const timer = setInterval(() => {
      i += 1;
      setDisplay(text.slice(0, i));
      if (i >= text.length) clearInterval(timer);
    }, speed);
    return () => clearInterval(timer);
  }, [trigger, text, speed]);

  return display;
}
