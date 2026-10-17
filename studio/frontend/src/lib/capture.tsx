/**
 * The App Builder's eyes: mount a finished workspace offscreen, let it render
 * (fonts, timers, the logic-brain bridge), and rasterize it with html-to-image.
 * For declarative apps every key screen is captured (its state forced, timers
 * frozen); for logic apps the booted first frame is. Alongside each image the
 * stage is inspected for the layout facts a screenshot can't state — overflow,
 * clipped text, broken images, an empty screen. Runs entirely off the visible
 * DOM — the user never sees the mount — and fails open (empty set) so a capture
 * problem never blocks a build.
 */

import { createRoot } from "react-dom/client";
import { toJpeg } from "html-to-image";
import StudioView from "../components/StudioView";
import type { Studio, StudioLogInput, StudioSpec, StudioState } from "../types";
import type { Theme } from "../App";

const CAPTURE_W = 1280;
const CAPTURE_H = 800;
const SETTLE_MS = 350;
const BRAIN_DEADLINE_MS = 3000;
const RASTER_TIMEOUT_MS = 8000;
/** How many screens a declarative app contributes to the contact sheet. */
const MAX_CAPTURE_STATES = 4;
/** Per-screen diagnostic cap (and a global one across screens). */
const MAX_DIAGNOSTICS_PER_SCREEN = 4;
const MAX_DIAGNOSTICS = 6;
/** DOM walk cap: a huge board never turns the inspection into a stall. */
const MAX_INSPECT_ELEMENTS = 800;

/** Everything the review loop and the critic see after a render. */
export interface StudioCapture {
  /** JPEG data URLs, one per captured screen. */
  images: string[];
  /** Human-readable layout facts measured from the live DOM. */
  diagnostics: string[];
  /** Runtime output collected while the screens booted. */
  logs: StudioLogInput[];
}

const nextFrame = () => new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));

/** A stalled rasterization must never leave the build locked in review. */
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) => setTimeout(() => reject(new Error("capture timed out")), ms)),
  ]);
}

async function waitForFonts(): Promise<void> {
  try {
    await Promise.race([document.fonts.ready, new Promise((r) => setTimeout(r, 1500))]);
  } catch {
    // A font failure must not block the capture.
  }
}

/** The layout facts a screenshot alone can't state. */
function inspectStage(stage: HTMLElement): string[] {
  const out: string[] = [];
  const width = stage.clientWidth;
  if (width > 0 && stage.scrollWidth > width + 8) {
    out.push(`the screen scrolls horizontally by ${stage.scrollWidth - width}px — content does not fit the width`);
  }
  if (!(stage.textContent ?? "").trim()) {
    out.push("the screen renders no visible text at all");
  }
  const elements = Array.from(stage.querySelectorAll<HTMLElement>("*")).slice(0, MAX_INSPECT_ELEMENTS);
  let clipped = 0;
  for (const el of elements) {
    if (el.scrollWidth <= el.clientWidth + 8 && el.scrollHeight <= el.clientHeight + 8) continue;
    const style = getComputedStyle(el);
    const clips = style.overflow === "hidden" || style.overflowX === "hidden" || style.overflowY === "hidden";
    if (clips && (el.textContent ?? "").trim()) clipped += 1;
  }
  if (clipped) out.push(`${clipped} element${clipped === 1 ? "" : "s"} clip their text (overflow: hidden)`);
  let broken = 0;
  for (const img of Array.from(stage.querySelectorAll("img"))) {
    if (img.complete && img.naturalWidth === 0) broken += 1;
  }
  if (broken) out.push(`${broken} image${broken === 1 ? "" : "s"} failed to load`);
  return out.slice(0, MAX_DIAGNOSTICS_PER_SCREEN);
}

/**
 * One offscreen mount: renders the spec, waits for it to settle, inspects the
 * stage, and rasterizes the whole studio view (not just the stage: its
 * data-surface/data-theme attributes scope the app's CSS, and the clone must
 * carry them or the app renders unstyled).
 */
async function captureVariant(
  spec: StudioSpec,
  onLog?: (entry: StudioLogInput) => void
): Promise<{ image: string | null; diagnostics: string[] }> {
  const theme: Theme = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
  const host = document.createElement("div");
  host.setAttribute("aria-hidden", "true");
  host.style.cssText = [
    "position:fixed",
    "left:-20000px",
    "top:0",
    `width:${CAPTURE_W}px`,
    `height:${CAPTURE_H}px`,
    // The studio frame is a flex child (flex: 1 1 0); a flex column host gives
    // it the full capture viewport to fill.
    "display:flex",
    "flex-direction:column",
    "overflow:hidden",
    "pointer-events:none",
    "z-index:-1",
  ].join(";");
  document.body.appendChild(host);

  const root = createRoot(host);
  const studio: Studio = { id: "capture", title: spec.title, spec, conversationId: -1, createdAt: Date.now() };
  try {
    root.render(<StudioView studio={studio} theme={theme} onBack={() => {}} onLog={onLog} />);
    await waitForFonts();
    await nextFrame();
    await nextFrame();

    // A logic brain boots asynchronously (iframe loads, bridge handshake,
    // first render); wait for its placeholder to give way, but never hang.
    const deadline = Date.now() + BRAIN_DEADLINE_MS;
    while (Date.now() < deadline) {
      const stage = host.querySelector(".app-stage");
      if (stage && !stage.querySelector(".studio-preparing")) break;
      await new Promise((r) => setTimeout(r, 100));
    }
    await new Promise((r) => setTimeout(r, SETTLE_MS));

    const node =
      (host.querySelector(".studio-view") as HTMLElement | null) ??
      (host.querySelector(".app-stage") as HTMLElement | null);
    if (!node) return { image: null, diagnostics: [] };
    const stage = host.querySelector(".app-stage") as HTMLElement | null;
    const diagnostics = stage ? inspectStage(stage) : [];

    const base = { quality: 0.82, pixelRatio: 1, backgroundColor: theme === "dark" ? "#1a1a19" : "#fafaf7" };
    try {
      return { image: await withTimeout(toJpeg(node, base), RASTER_TIMEOUT_MS), diagnostics };
    } catch {
      // Font inlining can fail on restricted networks; fall back to system faces.
      try {
        return { image: await withTimeout(toJpeg(node, { ...base, skipFonts: true }), RASTER_TIMEOUT_MS), diagnostics };
      } catch {
        return { image: null, diagnostics };
      }
    }
  } catch {
    return { image: null, diagnostics: [] };
  } finally {
    try {
      root.unmount();
    } catch {
      // Already torn down.
    }
    host.remove();
  }
}

/** The screens to capture: the initial state first, then the other states. */
function captureVariants(spec: StudioSpec): { label: string; variant: StudioSpec }[] {
  if (spec.logic) return [{ label: "", variant: spec }];
  const states = spec.states ?? {};
  const ids = Object.keys(states);
  if (!ids.length) return [{ label: "", variant: spec }];
  const initial = spec.initial && states[spec.initial] ? spec.initial : ids[0];
  const ordered = [initial, ...ids.filter((id) => id !== initial)].slice(0, MAX_CAPTURE_STATES);
  // Freeze timers so a state with `after` cannot auto-advance mid-capture.
  const frozen: Record<string, StudioState> = {};
  for (const [id, state] of Object.entries(states)) {
    frozen[id] = state.dock ? { blocks: state.blocks, dock: state.dock } : { blocks: state.blocks };
  }
  return ordered.map((id) => ({ label: id, variant: { ...spec, states: frozen, initial: id } }));
}

/**
 * The contact sheet: every captured screen plus the measured layout facts and
 * the runtime output of the offscreen boots. Empty images = no eyes (capture
 * failed); the caller falls back to the deterministic lint.
 */
export async function captureStudioSet(
  spec: StudioSpec,
  onLog?: (entry: StudioLogInput) => void
): Promise<StudioCapture> {
  const variants = captureVariants(spec);
  const images: string[] = [];
  const diagnostics: string[] = [];
  const logs: StudioLogInput[] = [];
  const seenLogs = new Set<string>();
  const collectLog = (entry: StudioLogInput) => {
    onLog?.(entry);
    const key = `${entry.source}:${entry.level}:${entry.message}`;
    if (!seenLogs.has(key) && logs.length < 20) {
      seenLogs.add(key);
      logs.push(entry);
    }
  };

  for (const { label, variant } of variants) {
    const { image, diagnostics: found } = await captureVariant(variant, collectLog);
    if (image) images.push(image);
    for (const line of found) {
      if (diagnostics.length >= MAX_DIAGNOSTICS) break;
      diagnostics.push(variants.length > 1 && label ? `screen "${label}": ${line}` : line);
    }
  }
  return { images, diagnostics, logs };
}
