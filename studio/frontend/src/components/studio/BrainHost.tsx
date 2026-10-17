/**
 * BrainHost — the shared sandbox host for "logic" brains.
 *
 * A brain is a self-contained JS program (no DOM, no origin) that runs in a
 * sandboxed iframe and talks to the host only through a postMessage bridge:
 *
 *   brain → host  {type:"render", blocks}   {type:"ready"}
 *                 {type:"host", action, arg} {type:"fetch", id, kind, args}
 *   host → brain  {type:"event", event, arg, blockType}
 *                 {type:"ctx", data}        {type:"fetch-result", id, ok, value, error}
 *
 * The host renders the brain's `blocks` with the shared component library and
 * hands them — plus a live `BlockCtx` — to `children`, so each parent can wrap
 * the brain in its own chrome (the app bar for AI apps, the wall scroll, the
 * welcome stage). Live data reaches the brain as a serializable `ctx`
 * projection, pushed on a short throttle; the brain re-renders in response.
 *
 * Inbound renders are throttled at ~30fps, but the throttle is trailing-edge:
 * a render that lands inside a closed window is held as the pending snapshot
 * (latest wins) and flushed the moment the window closes. Dropping it instead
 * would be fatal for sparse brains — the greeting only renders on discrete
 * state changes, so a dropped "final" frame (e.g. the one a fetch result
 * triggers a millisecond after its trigger) would leave the DOM stale forever.
 *
 * The ready handshake is belt-and-suspenders: the host pushes `ctx` both on
 * mount and the moment the brain reports ready, so it lands whichever way the
 * iframe load races React's effect. Content, however, gates on the first
 * `render` the host actually receives, which is always well-defined.
 */

import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { sanitizeBlocks, type SanitizeOpts } from "../../lib/studio-blocks";
import { type BlockCtx, type ChatEmbed, type ComposerHost } from "./blocks";
import type { StudioBlock } from "../../types";

const MIN_RENDER_INTERVAL_MS = 33;
const CTX_PUSH_MS = 100;
/** How long a brain may go without rendering before the host reports it. */
const BRAIN_BOOT_TIMEOUT_MS = 8000;
// AI brains are capped tightly (a run-away model); trusted brains (the
// product's own apps) may render much longer lists — a full conversation wall.
const MAX_BLOCKS_AI = 300;
const MAX_BYTES_AI = 64_000;
const MAX_BLOCKS_TRUSTED = 400;
const MAX_BYTES_TRUSTED = 1_000_000;

export interface BrainHostProps {
  /** The brain's JS source. Changing it reloads the iframe. */
  source: string;
  /** Accessible name for the (invisible) iframe. */
  title: string;
  /** Embedded conversation, when the brain renders a `chat` block. */
  chat?: ChatEmbed;
  /** The host-wired docked composer, when the brain renders a `composer` block. */
  composer?: ComposerHost;
  /** Trusted brains may emit app-only components and long lists. */
  trusted?: boolean;
  /** A serializable snapshot of live data, pushed to the brain as `ctx`. */
  ctxData?: Record<string, unknown>;
  /** The brain asks the host to do something (navigate, start, save…). */
  onHost?: (action: string, arg?: unknown) => void;
  /** The brain requests a whitelisted host capability (models/test/ideas…). */
  onFetch?: (kind: string, args?: unknown) => Promise<unknown> | unknown;
  /** The brain reports runtime output (errors, console) for the debug log. */
  onLog?: (entry: { level: "error" | "warn" | "log"; message: string; stack?: string }) => void;
  /** The brain produced no render in time (a boot hang); the host has already
      logged the failure — the parent may surface its Debug panel. */
  onBootFailure?: () => void;
  /** Raw frames rendered by the brain report through `ctx.log` (source: raw). */
  onRawLog?: (entry: { level: "error" | "warn" | "log"; message: string; stack?: string }) => void;
  onReady?: () => void;
  children: (r: { blocks: StudioBlock[] | null; ctx: BlockCtx; ready: boolean }) => ReactNode;
}

/**
 * Build the iframe document. The brain may only reach the frame through the
 * bridge — it has no DOM and no origin. The closing script tag is split so the
 * brain's own code can never terminate it early.
 *
 * The brain's source runs through `new Function(...)` at runtime instead of
 * being spliced into this script: a syntax error in the model's code (or a
 * `let`/`const` that collides with a shim name) then surfaces as a catchable
 * SyntaxError — reported to the Debug panel and rendered as a note — instead
 * of killing the entire bootstrap silently (the old "Starting up…" with an
 * empty log, which left nothing for the user or the AI to fix).
 */

/** A JS string literal — safe to splice into the bootstrap script, HTML-safe. */
function jsSourceLiteral(source: string): string {
  return (
    '"' +
    source
      .replace(/\\/g, "\\\\")
      .replace(/"/g, '\\"')
      .replace(/\n/g, "\\n")
      .replace(/\r/g, "\\r")
      .replace(/\u2028/g, "\\u2028")
      .replace(/\u2029/g, "\\u2029")
      .replace(/<\/script/gi, "<\\/script") +
    '"'
  );
}

function brainSrcdoc(source: string): string {
  const literal = jsSourceLiteral(source);
  return `<!doctype html><html><head><meta charset="utf-8"></head><body><script>
(function () {
  var handlers = {};
  var fetches = {};
  var fid = 0;
  function send(o) { parent.postMessage(o, "*"); }
  function report(level, message, stack) {
    try {
      send({ type: "log", level: level, message: String(message == null ? "" : message).slice(0, 2000), stack: stack ? String(stack).slice(0, 4000) : undefined });
    } catch (_) {}
  }
  // Runtime errors are reported to the host so the Debug panel (and the AI)
  // can see them; they would otherwise die silently inside the sandbox.
  window.onerror = function (msg, _src, _line, _col, err) { report("error", msg, err && err.stack); return false; };
  addEventListener("unhandledrejection", function (e) {
    var r = e && e.reason;
    report("error", "Unhandled rejection: " + ((r && r.message) || r), r && r.stack);
  });
  // Console output rides along (rate-limited); the original console still runs.
  var logTimes = [];
  function allowLog() {
    var now = Date.now();
    logTimes = logTimes.filter(function (t) { return now - t < 1000; });
    if (logTimes.length >= 10) return false;
    logTimes.push(now);
    return true;
  }
  function describe(a) {
    try { return typeof a === "string" ? a : JSON.stringify(a); } catch (_) { return String(a); }
  }
  ["log", "warn", "error"].forEach(function (level) {
    var original = console[level];
    console[level] = function () {
      var args = Array.prototype.slice.call(arguments);
      try { original.apply(console, args); } catch (_) {}
      if (allowLog()) report(level, args.map(describe).join(" "));
    };
  });
  window.ai = {
    render: function (r) {
      send({ type: "render", blocks: r && Array.isArray(r.blocks) ? r.blocks : null });
    },
    on: function (ev, fn) {
      if (typeof ev === "string" && typeof fn === "function") (handlers[ev] = handlers[ev] || []).push(fn);
    },
    ready: function () { send({ type: "ready" }); },
    show: function (r) {
      send({ type: "render", blocks: Array.isArray(r) ? r : (r && Array.isArray(r.blocks) ? r.blocks : [r]) });
    },
    fx: function (id, data) {
      send({ type: "fx-push", id: id, data: data });
    },
    host: function (action, arg) { send({ type: "host", action: action, arg: arg }); },
    fetch: function (kind, args) {
      return new Promise(function (resolve, reject) {
        fid += 1;
        var id = fid;
        fetches[id] = { resolve: resolve, reject: reject };
        send({ type: "fetch", id: id, kind: kind, args: args });
      });
    },
    ask: function (req) {
      return new Promise(function (resolve, reject) {
        fid += 1;
        var id = fid;
        fetches[id] = { resolve: resolve, reject: reject };
        send({ type: "fetch", id: id, kind: "ask", args: req });
      });
    }
  };
  addEventListener("message", function (e) {
    var d = e.data;
    if (!d || typeof d.type !== "string") return;
    if (d.type === "event" && typeof d.event === "string") {
      (handlers[d.event] || []).forEach(function (f) { try { f(d.arg, d.blockType); } catch (err) { report("error", "Handler error: " + ((err && err.message) || err), err && err.stack); } });
    } else if (d.type === "ctx") {
      (handlers["ctx"] || []).forEach(function (f) { try { f(d.data); } catch (err) { report("error", "Handler error: " + ((err && err.message) || err), err && err.stack); } });
    } else if (d.type === "fetch-result" && fetches[d.id]) {
      var p = fetches[d.id];
      delete fetches[d.id];
      if (d.ok) p.resolve(d.value); else p.reject(new Error(d.error || "fetch failed"));
    }
  });
  try {
    new Function(${literal}).call(window);
  } catch (err) {
    var startMsg = (err && err.name === "SyntaxError" ? "The app's brain failed to parse: " : "Workspace logic error: ") + ((err && err.message) || String(err));
    report("error", startMsg, err && err.stack);
    send({ type: "render", blocks: [{ type: "note", tone: "error", content: startMsg }] });
  }
  if (window.ai && window.ai.ready) window.ai.ready();
})();
</scr` + `ipt></body></html>`;
}

type BrainMsg = {
  type?: string;
  action?: string;
  arg?: unknown;
  kind?: string;
  args?: unknown;
} & Record<string, unknown>;

export default function BrainHost({
  source,
  title,
  chat,
  composer,
  trusted,
  ctxData,
  onHost,
  onFetch,
  onLog,
  onBootFailure,
  onRawLog,
  onReady,
  children,
}: BrainHostProps) {
  const [blocks, setBlocks] = useState<StudioBlock[] | null>(null);
  const [ready, setReady] = useState(false);
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const lastRender = useRef(0);
  const pendingRender = useRef<StudioBlock[] | null>(null);
  const renderFlush = useRef<ReturnType<typeof setTimeout> | null>(null);
  const ctxTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  /** True once the brain has produced any render (the boot watchdog's guard). */
  const renderedRef = useRef(false);

  // Everything the (mount-once) message handler needs lives in refs, so the
  // handler never goes stale and never has to re-subscribe on every render.
  const ctxRef = useRef(ctxData);
  ctxRef.current = ctxData;
  const onHostRef = useRef(onHost);
  onHostRef.current = onHost;
  const onFetchRef = useRef(onFetch);
  onFetchRef.current = onFetch;
  const onLogRef = useRef(onLog);
  onLogRef.current = onLog;
  const onBootFailureRef = useRef(onBootFailure);
  onBootFailureRef.current = onBootFailure;
  const onRawLogRef = useRef(onRawLog);
  onRawLogRef.current = onRawLog;
  const onReadyRef = useRef(onReady);
  onReadyRef.current = onReady;
  const trustedRef = useRef(trusted);
  trustedRef.current = trusted;
  const readyRef = useRef(false);
  readyRef.current = ready;
  /** In-flight host fetches: the watchdog holds off while one is pending (a
      brain may legitimately await ai.ask before its first render). */
  const inFlightRef = useRef(0);

  const post = (o: unknown) => {
    iframeRef.current?.contentWindow?.postMessage(o, "*");
  };

  useEffect(() => {
    const onMsg = (e: MessageEvent) => {
      if (e.source !== iframeRef.current?.contentWindow) return;
      const d = e.data as BrainMsg | null;
      if (!d || typeof d.type !== "string") return;
      if (d.type === "ready") {
        if (!readyRef.current) {
          readyRef.current = true;
          setReady(true);
          onReadyRef.current?.();
        }
        post({ type: "ctx", data: ctxRef.current });
      } else if (d.type === "render") {
        // Any render attempt counts as "the brain booted" for the watchdog.
        renderedRef.current = true;
        if (!Array.isArray(d.blocks)) {
          onLogRef.current?.({
            level: "error",
            message: "The app called ai.render() without a blocks array — nothing to show.",
          });
          return;
        }
        const isTrusted = trustedRef.current;
        const maxBlocks = isTrusted ? MAX_BLOCKS_TRUSTED : MAX_BLOCKS_AI;
        const maxBytes = isTrusted ? MAX_BYTES_TRUSTED : MAX_BYTES_AI;
        const trimmed = (d.blocks as unknown[]).slice(0, maxBlocks);
        if (trimmed.length < d.blocks.length) {
          onLogRef.current?.({
            level: "warn",
            message: `The app rendered ${d.blocks.length} blocks; only the first ${maxBlocks} were kept.`,
          });
        }
        let bytes = 0;
        try {
          bytes = JSON.stringify(trimmed, (key, value) => {
            if (key === "src" && typeof value === "string" && value.length > 500) return "";
            return value;
          }).length;
        } catch {
          onLogRef.current?.({
            level: "error",
            message: "The app rendered a frame that could not be serialized; it was dropped.",
          });
          return;
        }
        if (bytes > maxBytes) {
          onLogRef.current?.({
            level: "error",
            message: `The app rendered ${Math.round(bytes / 1024)} KB in one frame — over the ${Math.round(maxBytes / 1024)} KB cap; it was dropped.`,
          });
          return;
        }
        const opts: SanitizeOpts | undefined = isTrusted ? { allowAppOnly: true, maxBlocks: MAX_BLOCKS_TRUSTED } : undefined;
        const clean = sanitizeBlocks(trimmed, 1, opts);
        if (!clean?.length) {
          onLogRef.current?.({
            level: "error",
            message: "The app rendered a frame with no valid components — every block was rejected.",
          });
          return;
        }
        const now = Date.now();
        if (now - lastRender.current < MIN_RENDER_INTERVAL_MS) {
          // Window is closed: hold this snapshot (latest wins) and flush it the
          // moment the window reopens — dropping it would strand sparse brains
          // on a stale frame (their next render may be minutes away).
          pendingRender.current = clean;
          if (renderFlush.current == null) {
            renderFlush.current = setTimeout(() => {
              renderFlush.current = null;
              const pending = pendingRender.current;
              pendingRender.current = null;
              if (pending) {
                lastRender.current = Date.now();
                setBlocks(pending);
              }
            }, Math.max(0, MIN_RENDER_INTERVAL_MS - (now - lastRender.current)));
          }
          return;
        }
        if (renderFlush.current != null) {
          clearTimeout(renderFlush.current);
          renderFlush.current = null;
          pendingRender.current = null;
        }
        lastRender.current = now;
        if (!readyRef.current) {
          readyRef.current = true;
          setReady(true);
          onReadyRef.current?.();
        }
        setBlocks(clean);
      } else if (d.type === "log" && typeof d.message === "string") {
        onLogRef.current?.({
          level: d.level === "error" || d.level === "warn" ? d.level : "log",
          message: d.message,
          stack: typeof d.stack === "string" ? d.stack : undefined,
        });
      } else if (d.type === "host" && typeof d.action === "string") {
        onHostRef.current?.(d.action, d.arg);
      } else if (d.type === "fetch" && typeof d.kind === "string") {
        const id = d.id;
        inFlightRef.current += 1;
        Promise.resolve(onFetchRef.current?.(d.kind, d.args))
          .then((value) => {
            post({ type: "fetch-result", id, ok: true, value });
          })
          .catch((err) => {
            post({ type: "fetch-result", id, ok: false, error: String((err && err.message) || err) });
          })
          .finally(() => {
            inFlightRef.current = Math.max(0, inFlightRef.current - 1);
          });
      } else if (d.type === "fx-push" && typeof d.id === "string") {
        const iframes = document.querySelectorAll<HTMLIFrameElement>(`iframe[data-raw-id="${d.id}"]`);
        iframes.forEach((ifr) => ifr.contentWindow?.postMessage({ type: "fx-push", id: d.id, data: d.data }, "*"));
      }
    };
    window.addEventListener("message", onMsg);
    // Cover the race where the iframe loaded (and is now listening) before this
    // effect attached. The "ready" flush below covers the opposite ordering.
    post({ type: "ctx", data: ctxRef.current });
    return () => {
      window.removeEventListener("message", onMsg);
      if (ctxTimer.current != null) {
        clearTimeout(ctxTimer.current);
        ctxTimer.current = null;
      }
      if (renderFlush.current != null) {
        clearTimeout(renderFlush.current);
        renderFlush.current = null;
      }
      pendingRender.current = null;
    };
  }, [source]);

  // Boot watchdog: a brain that never renders — a crash before the bridge, an
  // endless loop, or logic that simply never calls ai.render() — would leave
  // "Starting up…" on screen with nothing in the Debug panel for the user (or
  // the AI) to act on. Waiting on an in-app AI call doesn't count as a hang:
  // the deadline is extended while a host fetch is in flight.
  useEffect(() => {
    renderedRef.current = false;
    let timer: ReturnType<typeof setTimeout>;
    const check = () => {
      if (renderedRef.current) return;
      if (inFlightRef.current > 0) {
        timer = setTimeout(check, BRAIN_BOOT_TIMEOUT_MS);
        return;
      }
      onLogRef.current?.({
        level: "error",
        message: `The app's brain hasn't rendered anything in ${Math.round(BRAIN_BOOT_TIMEOUT_MS / 1000)}s — it may have crashed on start, be stuck in an endless loop, or never call ai.render().`,
      });
      onBootFailureRef.current?.();
    };
    timer = setTimeout(check, BRAIN_BOOT_TIMEOUT_MS);
    return () => clearTimeout(timer);
  }, [source]);

  // Live data: coalesce rapid updates into one push per throttle window.
  useEffect(() => {
    if (!readyRef.current) return;
    if (ctxTimer.current != null) return;
    ctxTimer.current = setTimeout(() => {
      ctxTimer.current = null;
      post({ type: "ctx", data: ctxRef.current });
    }, CTX_PUSH_MS);
  }, [ctxData]);

  const ctx: BlockCtx = {
    data: ctxData ?? {},
    resolve: (s) => s, // the brain renders fully-resolved blocks
    do: (block, event, arg) => post({ type: "event", event, arg, blockType: block.type, id: (block as { id?: string }).id }),
    log: (entry) => onRawLogRef.current?.(entry),
    chat,
    composer,
  };

  return (
    <>
      {children({ blocks, ctx, ready })}
      <iframe ref={iframeRef} className="logic-frame" sandbox="allow-scripts" srcDoc={brainSrcdoc(source)} title={title} />
    </>
  );
}
