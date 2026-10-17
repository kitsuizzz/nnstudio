import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { applyAction, actionTarget, hasMutation, resolveText } from "../lib/studio-engine";
import { evalBinding } from "../lib/expr";
import BrainHost from "./studio/BrainHost";
import { Block, type BlockCtx, type ChatEmbed } from "./studio/blocks";
import StudioErrorBoundary from "./studio/StudioErrorBoundary";
import { completeChat } from "../lib/api";
import { formatLogReport } from "../lib/studio-log";
import type { LlmConfig, Studio, StudioBlock, StudioLogEntry, StudioLogInput, StudioSpec } from "../types";
import type { Theme } from "../App";

/** A runtime line before the host stamps its source. */
type LogInput = Omit<StudioLogInput, "source">;

/** The in-app AI call budget: one call may take this long, and this many per minute. */
const IN_APP_TIMEOUT_MS = 20_000;
const MAX_IN_APP_CALLS = 30;

/** The debug state the app bar + panel render from. */
interface StudioDebug {
  open: boolean;
  errors: number;
  logs: StudioLogEntry[];
  onToggle: () => void;
  onClear?: () => void;
  onFix?: (report: string) => void;
}

interface StudioViewProps {
  studio: Studio;
  theme: Theme;
  onBack: () => void;
  /** The active conversation, so a `chat` component can embed it. */
  chat?: ChatEmbed;
  /** The active LLM config, so a logic brain can call ai.fetch("move"). */
  llm?: LlmConfig;
  /** The captured debug log for this workspace (persisted with the studio). */
  logs?: StudioLogEntry[];
  /** Report a runtime line; the host stamps the time and caps the tail. */
  onLog?: (entry: StudioLogInput) => void;
  onClearLogs?: () => void;
  /** Hand the formatted log to the AI ("Fix with AI"). */
  onFixWithAI?: (report: string) => void;
  /** The user already allowed this app to call the AI in-app. */
  aiAllowed?: boolean;
  /** Persist the user's in-app AI allow decision. */
  onAllowAI?: () => void;
}

/**
 * An AI-built app: it owns the whole window (app bar + full stage), driven by
 * either the declarative engine (states + data + actions) or a sandboxed JS
 * brain (logic mode) over a postMessage bridge. Runtime output — errors,
 * rejections, console — is captured into the debug log so the user can see it
 * and the AI can fix it.
 */
export default function StudioView({
  studio,
  theme,
  onBack,
  chat,
  llm,
  logs,
  onLog,
  onClearLogs,
  onFixWithAI,
  aiAllowed,
  onAllowAI,
}: StudioViewProps) {
  const spec = studio.spec;
  const [debugOpen, setDebugOpen] = useState(false);

  const logLogic = (entry: LogInput) => onLog?.({ source: "logic", ...entry });
  const logRaw = (entry: LogInput) => onLog?.({ source: "raw", ...entry });
  const logInApp = (entry: LogInput) => onLog?.({ source: "in-app", ...entry });

  const handleCrash = (error: Error) => {
    onLog?.({ source: "render", level: "error", message: error.message || String(error), stack: error.stack });
    setDebugOpen(true);
  };

  const errors = (logs ?? []).reduce((n, entry) => n + (entry.level === "error" ? entry.count ?? 1 : 0), 0);
  const debug: StudioDebug = {
    open: debugOpen,
    errors,
    logs: logs ?? [],
    onToggle: () => setDebugOpen((o) => !o),
    onClear: onClearLogs,
    onFix: onFixWithAI,
  };

  return (
    <StudioErrorBoundary onError={handleCrash} onBack={onBack}>
      {spec.logic ? (
        <LogicFrame
          spec={spec}
          theme={theme}
          onBack={onBack}
          chat={chat}
          llm={llm}
          debug={debug}
          aiRole={spec.ai?.role}
          aiAllowed={aiAllowed}
          onAllowAI={onAllowAI}
          onLog={logLogic}
          onRawLog={logRaw}
          onInAppLog={logInApp}
          onBootFailure={() => setDebugOpen(true)}
        />
      ) : (
        <DeclarativeFrame spec={spec} theme={theme} onBack={onBack} chat={chat} debug={debug} onRawLog={logRaw} />
      )}
    </StudioErrorBoundary>
  );
}

function Shell({
  spec,
  theme,
  onBack,
  children,
  dock,
  debug,
  aiRole,
}: {
  spec: StudioSpec;
  theme: Theme;
  onBack: () => void;
  children: ReactNode;
  dock?: ReactNode;
  debug?: StudioDebug;
  aiRole?: string;
}) {
  return (
    <main className="studio-view" data-surface="app" data-theme={theme}>
      <header className="studio-appbar">
        <h1 className="studio-appbar-title">{spec.title}</h1>
        <div className="studio-appbar-actions">
          {aiRole && (
            <span className="appbar-ai-chip" title={`This app may use the AI as ${aiRole}`}>
              AI inside · {aiRole}
            </span>
          )}
          <button
            type="button"
            className={`btn ghost small appbar-debug${debug && debug.errors > 0 ? " has-errors" : ""}`}
            aria-expanded={debug?.open ?? false}
            onClick={debug?.onToggle}
          >
            Debug
            {debug && debug.errors > 0 ? <span className="appbar-debug-badge">{debug.errors}</span> : null}
          </button>
          <button type="button" className="btn ghost small appbar-exit" onClick={onBack}>
            Exit
          </button>
        </div>
      </header>
      <div className="app-stage">
        {children}
      </div>
      {debug?.open ? <DebugPanel logs={debug.logs} onClear={debug.onClear} onFix={debug.onFix} /> : null}
      {dock ? <footer className="studio-dock">{dock}</footer> : null}
    </main>
  );
}

/** The workspace's debug log: captured runtime output, copyable and AI-fixable. */
function DebugPanel({
  logs,
  onClear,
  onFix,
}: {
  logs: StudioLogEntry[];
  onClear?: () => void;
  onFix?: (report: string) => void;
}) {
  const report = formatLogReport(logs);
  return (
    <aside className="studio-debug" aria-label="Debug log">
      <header className="studio-debug-head">
        <span className="studio-debug-title">Debug log</span>
        <span className="studio-debug-count">
          {logs.length ? `${logs.length} line${logs.length === 1 ? "" : "s"}` : "empty"}
        </span>
        <div className="studio-debug-actions">
          <button
            type="button"
            className="btn ghost small"
            disabled={!logs.length}
            onClick={() => void navigator.clipboard?.writeText(report)}
          >
            Copy
          </button>
          <button type="button" className="btn ghost small" disabled={!logs.length} onClick={onClear}>
            Clear
          </button>
          <button type="button" className="btn small studio-debug-fix" disabled={!logs.length} onClick={() => onFix?.(report)}>
            Fix with AI
          </button>
        </div>
      </header>
      {logs.length ? (
        <ol className="studio-debug-list">
          {logs
            .slice()
            .reverse()
            .map((entry, i) => (
              <li key={i} className={`studio-debug-entry ${entry.level}`}>
                <span className="studio-debug-time">{new Date(entry.at).toLocaleTimeString()}</span>
                <span className="studio-debug-source">{entry.source}</span>
                <span className="studio-debug-level">{entry.level}</span>
                <span className="studio-debug-message">
                  {entry.message}
                  {entry.count && entry.count > 1 ? ` ×${entry.count}` : ""}
                </span>
                {entry.stack && (
                  <details className="studio-debug-stack">
                    <summary>stack</summary>
                    <pre>{entry.stack}</pre>
                  </details>
                )}
              </li>
            ))}
        </ol>
      ) : (
        <p className="studio-debug-empty">
          No runtime output yet. Errors and console output from this app show up here — and in the AI's next edit.
        </p>
      )}
    </aside>
  );
}

/* ---------------- declarative engine ---------------- */

function DeclarativeFrame({
  spec,
  theme,
  onBack,
  chat,
  debug,
  onRawLog,
}: {
  spec: StudioSpec;
  theme: Theme;
  onBack: () => void;
  chat?: ChatEmbed;
  debug?: StudioDebug;
  onRawLog?: (entry: LogInput) => void;
}) {
  const initialData = useRef<Record<string, unknown>>(spec.data ? JSON.parse(JSON.stringify(spec.data)) : {});
  const [stateId, setStateId] = useState(spec.initial ?? "");
  const [data, setData] = useState<Record<string, unknown>>(initialData.current);
  const state = spec.states?.[stateId] ?? Object.values(spec.states ?? {})[0];

  const ctx: BlockCtx = {
    data,
    resolve: (s) => resolveText(s, data),
    eval: (expr) => evalBinding(expr, data),
    do: (block, event, arg) => {
      const action =
        block.type === "button"
          ? block.action
          : event === "cell"
            ? (block as Extract<StudioBlock, { type: "grid" }>).onCell
            : event === "choose"
              ? (block as Extract<StudioBlock, { type: "choice" }>).onChoose
              : event === "submit"
                ? (block as Extract<StudioBlock, { type: "input" }>).onSubmit
                : event === "event" && block.type === "raw"
                  ? (block as Extract<StudioBlock, { type: "raw" }>).onEvent
                  : undefined;
      if (action && hasMutation(action)) {
        setData((d) => applyAction(d, action, arg, initialData.current));
      }
      if (block.type === "button") {
        if (spec.states?.[block.to]) setStateId(block.to);
        return;
      }
      const target = action ? actionTarget(action) : undefined;
      if (target && spec.states?.[target]) setStateId(target);
    },
    log: onRawLog,
    chat,
  };

  return (
    <Shell
      spec={spec}
      theme={theme}
      onBack={onBack}
      debug={debug}
      dock={
        state?.dock?.length ? (
          state.dock.map((b, i) => <Block key={i} block={b} ctx={ctx} />)
        ) : undefined
      }
    >
      <div className="app-settle" key={stateId} style={{ display: "contents" }}>
        {state?.blocks.map((block, i) => <Block key={i} block={block} ctx={ctx} />)}
      </div>
    </Shell>
  );
}

/* ---------------- logic mode (sandboxed JS brain) ---------------- */

function LogicFrame({
  spec,
  theme,
  onBack,
  chat,
  llm,
  debug,
  aiRole,
  aiAllowed,
  onAllowAI,
  onLog,
  onRawLog,
  onInAppLog,
  onBootFailure,
}: {
  spec: StudioSpec;
  theme: Theme;
  onBack: () => void;
  chat?: ChatEmbed;
  llm?: LlmConfig;
  debug?: StudioDebug;
  aiRole?: string;
  aiAllowed?: boolean;
  onAllowAI?: () => void;
  onLog?: (entry: LogInput) => void;
  onRawLog?: (entry: LogInput) => void;
  onInAppLog?: (entry: LogInput) => void;
  /** The brain never started: surface the Debug panel with the host's report. */
  onBootFailure?: () => void;
}) {
  const [asking, setAsking] = useState(false);
  const deniedRef = useRef(false);
  const consentRef = useRef<((ok: boolean) => void) | null>(null);
  const inAppAbortRef = useRef<AbortController | null>(null);
  const callTimesRef = useRef<number[]>([]);

  // Leaving the workspace cancels any in-flight in-app AI call.
  useEffect(() => () => inAppAbortRef.current?.abort(), []);

  /** Ask once per workspace; a decline is remembered for this session only. */
  const ensureConsent = () => {
    if (aiAllowed) return Promise.resolve(true);
    if (deniedRef.current) return Promise.resolve(false);
    setAsking(true);
    return new Promise<boolean>((resolve) => {
      consentRef.current = resolve;
    });
  };

  const decideConsent = (ok: boolean) => {
    setAsking(false);
    if (ok) onAllowAI?.();
    else deniedRef.current = true;
    consentRef.current?.(ok);
    consentRef.current = null;
  };

  /** One in-app model call: budgeted, timed out, aborted on exit. */
  const runInApp = async (role: string, system: string, input: string, json: boolean): Promise<string> => {
    if (!llm) throw new Error("No model is configured for this workspace.");
    const now = Date.now();
    callTimesRef.current = callTimesRef.current.filter((t) => now - t < 60_000);
    if (callTimesRef.current.length >= MAX_IN_APP_CALLS) {
      throw new Error("Too many AI calls from this app — try again in a minute.");
    }
    callTimesRef.current.push(now);

    const controller = new AbortController();
    inAppAbortRef.current = controller;
    const timer = setTimeout(() => controller.abort(), IN_APP_TIMEOUT_MS);
    try {
      const fullSystem = "===IN-APP===\n" + system + (json ? "\nReply with ONLY valid JSON. No prose, no markdown fences." : "");
      const reply = await completeChat(llm, fullSystem, input, controller.signal);
      onInAppLog?.({
        level: "log",
        message: `[in-app] ${role} → ${reply.replace(/\s+/g, " ").slice(0, 240)}`,
      });
      return reply;
    } catch (err) {
      onInAppLog?.({
        level: "error",
        message: `[in-app] ${role} failed: ${err instanceof Error ? err.message : String(err)}`,
      });
      throw err;
    } finally {
      clearTimeout(timer);
      if (inAppAbortRef.current === controller) inAppAbortRef.current = null;
    }
  };

  const onFetch = async (kind: string, args?: unknown) => {
    const allowed = await ensureConsent();
    if (!allowed) throw new Error("AI use was declined for this workspace.");

    // The board-game shortcut: resolve to a valid cell index.
    if (kind === "move") {
      const { board } = (args ?? {}) as { board?: string[] };
      const reply = await runInApp(
        spec.ai?.role ?? "board-game opponent",
        spec.ai?.system ??
          "You are the AI opponent in a board game. Reply with ONLY a single digit (0-8) indicating your next move. No explanation, no markdown, no punctuation.",
        `Board: ${JSON.stringify(board)}. Your turn.`,
        false
      );
      const match = reply.match(/\d+/);
      if (!match) return -1;
      const move = parseInt(match[0], 10);
      if (move < 0 || move >= (board?.length ?? 9) || board?.[move] !== "·") return -1;
      return move;
    }

    // The general call: the app passes its state; the spec declares the role.
    if (kind === "ask") {
      const req = (args ?? {}) as { input?: unknown; system?: string; json?: boolean };
      const system = typeof req.system === "string" && req.system.trim() ? req.system.slice(0, 2000) : spec.ai?.system;
      if (!system) throw new Error("This app did not declare an AI role.");
      const input =
        typeof req.input === "string" ? req.input.slice(0, 8000) : JSON.stringify(req.input ?? {}).slice(0, 8000);
      const reply = await runInApp(spec.ai?.role ?? "assistant", system, input, req.json === true);
      if (req.json) return JSON.parse(reply);
      return reply;
    }
    throw new Error(`Unknown fetch kind: ${kind}`);
  };

  // The brain runs in a sandboxed iframe (BrainHost) and drives the shared
  // component library over a postMessage bridge; the app bar is the host's.
  return (
    <>
      <Shell spec={spec} theme={theme} onBack={onBack} debug={debug} aiRole={aiRole}>
        <BrainHost
          source={spec.logic as string}
          title={spec.title}
          trusted={spec.trusted}
          chat={chat}
          onLog={onLog}
          onRawLog={onRawLog}
          onFetch={llm ? onFetch : undefined}
          onBootFailure={onBootFailure}
        >
          {({ blocks, ctx }) =>
            blocks ? (
              <div className="app-settle" style={{ display: "contents" }}>
                {blocks.map((b, i) => (
                  <Block key={i} block={b} ctx={ctx} />
                ))}
              </div>
            ) : (
              <div className="studio-preparing" style={{ margin: "auto" }}>
                <span className="dot" aria-hidden="true" />
                Starting up…
              </div>
            )
          }
        </BrainHost>
      </Shell>
      {asking && (
        <div className="ai-consent-backdrop">
          <div className="ai-consent" role="dialog" aria-label="Allow the AI in this app">
            <p className="ai-consent-text">
              This app wants to use the AI as <strong>{spec.ai?.role ?? "an assistant"}</strong>.
            </p>
            <p className="ai-consent-sub">
              It will send the app's state to {llm?.model ?? "your model"} with your provider key.
            </p>
            <div className="ai-consent-actions">
              <button type="button" className="btn small ai-consent-allow" onClick={() => decideConsent(true)}>
                Allow
              </button>
              <button type="button" className="btn ghost small" onClick={() => decideConsent(false)}>
                Not now
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
