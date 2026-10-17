/**
 * The React side of the component registry: one renderer per block type.
 * Adding a future component = one entry in lib/studio-blocks.ts + one
 * renderer here + its CSS. Everything the user touches in the product —
 * AI-built apps and the product's own surfaces alike — flows through this
 * library, so the workspaces stay visually the product.
 */

import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import WorkBody from "../WorkBody";
import Composer from "../Composer";
import Study from "../Study";
import { ChevronDownIcon } from "../Icon";
import { getPath, resolveText } from "../../lib/studio-engine";
import { evalBinding, truthy } from "../../lib/expr";
import type { Align, ChatMessage, Emph, Face, Justify, SendMode, StudioBlock, Tone, Wallabel } from "../../types";

export type BlockEvent = "cell" | "choose" | "click" | "submit" | "change" | "event";

/** Live conversation embedded via the `chat` component. */
export interface ChatEmbed {
  messages: ChatMessage[];
  streaming: boolean;
  onSend: (text: string, images?: string[]) => void;
  sendMode: SendMode;
}

/** The host-wired docked message box, mounted by a `composer` block. */
export interface ComposerHost {
  onSend: (text: string, images: string[]) => void;
  onStop: () => void;
  disabled: boolean;
  sendMode: SendMode;
  sendBlocker: string | null;
}

/** Everything a renderer may need from the frame driving it. */
export interface BlockCtx {
  /** The app's data snapshot (declarative mode). */
  data: Record<string, unknown>;
  /** Resolve {{expression}} bindings in a string (identity for hosted apps). */
  resolve: (s: string) => string;
  /** Evaluate an expression against the data (declarative apps only). */
  eval?: (expr: string) => unknown;
  /** Fire a user event for a block (applies its action / relays to the brain). */
  do: (block: StudioBlock, event: BlockEvent, arg?: unknown) => void;
  /** Report runtime output (errors, console) for the workspace debug log. */
  log?: (entry: { level: "error" | "warn" | "log"; message: string; stack?: string }) => void;
  /** The embedded conversation, when a `chat` component is rendered. */
  chat?: ChatEmbed;
  /** The host-wired composer, when a `composer` block is rendered. */
  composer?: ComposerHost;
}

/* ---------------- design dials: per-component class helpers ---------------- */

function dialCls(emph?: Emph, face?: Face): string {
  const p: string[] = [];
  if (emph) p.push(`emph-${emph}`);
  if (face) p.push(`face-${face}`);
  return p.join(" ");
}

function toneCls(tone?: Tone): string {
  return tone ? `tone-${tone}` : "";
}

function layoutCls(align?: Align, justify?: Justify): string {
  const p: string[] = [];
  if (align) p.push(`align-${align}`);
  if (justify) p.push(`justify-${justify}`);
  return p.join(" ");
}

/* ---------------- presentational atoms (shared by blocks + app forms) ------ */

interface FieldProps {
  label?: string;
  kind?: "text" | "number" | "email" | "url" | "password";
  value: string;
  placeholder?: string;
  onChange: (v: string) => void;
  autoFocus?: boolean;
  ariaLabel?: string;
  /** Extra classes appended to the `.field` root (e.g. "model-ask"). */
  className?: string;
}

/** A labeled single-line input — the product's canonical form field. */
export function Field({ label, kind = "text", value, placeholder, onChange, autoFocus, ariaLabel, className }: FieldProps) {
  return (
    <div className={className ? `field ${className}` : "field"}>
      {label && <span className="field-label">{label}</span>}
      <input
        type={kind}
        value={value}
        placeholder={placeholder}
        aria-label={ariaLabel ?? label}
        autoFocus={autoFocus}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  );
}

interface SelectFieldProps {
  label?: string;
  options: (string | { label: string; value: string })[];
  value: string;
  onChange: (v: string) => void;
  /** When set, render the compact "chip" variant (inline label + bare select). */
  chip?: string;
  ariaLabel?: string;
  /** Extra classes appended to the `.field` root (non-chip variant only). */
  className?: string;
}

/** A labeled dropdown — the product's canonical picker. */
export function SelectField({ label, options, value, onChange, chip, ariaLabel, className }: SelectFieldProps) {
  const select = (
    <select value={value} aria-label={ariaLabel ?? label ?? chip} onChange={(e) => onChange(e.target.value)}>
      {value && !options.some((o) => (typeof o === "string" ? o : o.value) === value) && (
        <option value={value}>{value}</option>
      )}
      {options.map((o) => {
        const optLabel = typeof o === "string" ? o : o.label;
        const optValue = typeof o === "string" ? o : o.value;
        return (
          <option key={optValue} value={optValue}>
            {optLabel}
          </option>
        );
      })}
    </select>
  );
  if (chip) {
    return (
      <label className="model-chip">
        <span className="model-chip-label">{chip}</span>
        {select}
      </label>
    );
  }
  return (
    <div className={className ? `field ${className}` : "field"}>
      {label && <span className="field-label">{label}</span>}
      {select}
    </div>
  );
}

function CodeBlock({
  content,
  language,
  highlight,
}: {
  content: string;
  language?: string;
  highlight?: number[];
}) {
  const lines = content.replace(/\n$/, "").split("\n");
  const hl = new Set(highlight ?? []);
  return (
    <div className="app-code" data-language={language || undefined}>
      <pre>
        {lines.map((line, i) => (
          <span
            key={i}
            className={hl.has(i + 1) ? "app-code-line app-code-hl" : "app-code-line"}
          >
            {line || " "}
            {"\n"}
          </span>
        ))}
      </pre>
    </div>
  );
}

/* ---------------- app-only composites ------------------------------------- */

/** The wall's right-column label (a work's "gallery tag"), with long-title clamp. */
function WallabelView({ data }: { data: Wallabel }) {
  const [expanded, setExpanded] = useState(false);
  const title = data.title || (data.images?.length ? "(images only)" : "(empty)");
  return (
    <aside className="label">
      <div className="label-no">
        {data.live ? <span className="dot" role="img" aria-label="Replying" /> : data.no}
      </div>
      <div className="label-kicker">{data.kicker}</div>
      <div className="label-title">
        <em className={data.long && !expanded ? "clamped" : undefined}>{title}</em>
        {data.time && <span className="label-time">, {data.time}</span>}
        {data.long && (
          <button className="label-more" type="button" onClick={() => setExpanded((e) => !e)}>
            {expanded ? "less" : "more"}
          </button>
        )}
      </div>
      {data.images && data.images.length > 0 && (
        <div className="label-refs">
          {data.images.map((src, i) => (
            <a key={i} href={src} target="_blank" rel="noreferrer" className="label-ref">
              <img src={src} alt={`Attachment ${i + 1}`} />
            </a>
          ))}
        </div>
      )}
      <div className="label-artist">
        {data.artist}
        {data.medium && (
          <span className="label-medium">
            {" "}
            {data.medium}
          </span>
        )}
      </div>
      <div className="label-dims">{data.dims}</div>
      {data.study && (
        <Study thinking={data.study.content} active={data.study.active} thinkingMs={data.study.ms} />
      )}
    </aside>
  );
}

const MAX_CHAT_MESSAGES = 12;

/**
 * The plan/progress card. The compact step list stays visible; the chevron on
 * the progress bar expands the detail: every step's chain-of-thought and the
 * builder's live reasoning (for providers that stream it). The disclosure is a
 * native <details>, so its open state survives the brain's re-renders.
 */
function PlanCard({ block, ctx }: { block: Extract<StudioBlock, { type: "plan" }>; ctx: BlockCtx }) {
  const thinkingRef = useRef<HTMLDivElement>(null);
  const thinking = block.thinking ?? "";
  const active = block.thinkingActive === true;

  useEffect(() => {
    const el = thinkingRef.current;
    if (el && active) el.scrollTop = el.scrollHeight;
  }, [thinking, active]);

  const total = block.todos.length;
  const pct = block.pct ?? (total ? Math.round((block.done / total) * 100) : 0);

  return (
    <div className="app-plan">
      <details className="app-plan-detail">
        <summary className="app-plan-head" aria-label="Expand what the AI is doing">
          <div className="app-plan-bar">
            <div className="app-plan-fill" style={{ width: `${pct}%` }} />
          </div>
          {block.pct != null && <span className="app-plan-pct">≈ {block.pct}%</span>}
          <span className="app-plan-chevron" aria-hidden="true">
            <ChevronDownIcon size={11} />
          </span>
        </summary>
        <div className="app-plan-detail-body">
          <ul className="app-plan-detail-rows">
            {block.todos.map((t, i) => {
              const cls = i < block.active ? "done" : i === block.active ? "active" : "pending";
              return (
                <li key={i} className={cls}>
                  <span className="app-plan-check" aria-hidden="true">
                    {i < block.active ? "✓" : i === block.active ? "●" : "○"}
                  </span>
                  <span className="app-plan-detail-text">
                    <span className="app-plan-title">{t.title}</span>
                    {t.cot && <span className="app-plan-cot">{t.cot}</span>}
                  </span>
                </li>
              );
            })}
          </ul>
          {thinking && (
            <div className="app-plan-thinking">
              <span className="app-plan-thinking-label">{active ? "Thinking…" : "Thought"}</span>
              <div className="app-plan-thinking-body" ref={thinkingRef}>
                {thinking}
              </div>
            </div>
          )}
        </div>
      </details>
      <ul className="app-plan-steps">
        {block.todos.map((t, i) => {
          const cls = i < block.active ? "done" : i === block.active ? "active" : "pending";
          return (
            <li key={i} className={cls}>
              <span className="app-plan-check" aria-hidden="true">
                {i < block.active ? "✓" : i === block.active ? "●" : "○"}
              </span>
              <span className="app-plan-title">{t.title}</span>
              {i === block.active && t.cot && <span className="app-plan-cot">{t.cot}</span>}
            </li>
          );
        })}
      </ul>
      {block.confirm && (
        <div className="app-plan-actions">
          <button type="button" className="app-plan-build" onClick={() => ctx.do(block, "click", "build")}>
            Build it
          </button>
        </div>
      )}
    </div>
  );
}

/** Dim the fence lines so the raw JSON body reads clearly. */
function writingLines(text: string) {
  return text.split("\n").map((line, i) => (
    <span key={i} className={line.trimStart().startsWith("```") ? "ai-fence" : undefined}>
      {line}
      {"\n"}
    </span>
  ));
}

/**
 * The inline AI activity window — part of the assistant's message, right where
 * the work happens ("Sure, let me build it" → this → the result). Collapsed it
 * is a status bar; expanding shows the reasoning and the raw writing stream
 * (fences included). It lives on the message, so it never resets or vanishes.
 */
function ActivityCard({ block }: { block: Extract<StudioBlock, { type: "activity" }> }) {
  const thinkingRef = useRef<HTMLPreElement>(null);
  const writingRef = useRef<HTMLPreElement>(null);
  const thinking = block.thinking ?? "";
  const writing = block.writing ?? "";

  useEffect(() => {
    if (!block.active) return;
    const t = thinkingRef.current;
    if (t) t.scrollTop = t.scrollHeight;
    const w = writingRef.current;
    if (w) w.scrollTop = w.scrollHeight;
  }, [thinking, writing, block.active]);

  return (
    <details className={`app-activity${block.active ? " live" : ""}`}>
      <summary className="app-activity-head" aria-label="Show what the AI is doing">
        <span className={`app-activity-dot${block.active ? " live" : ""}`} aria-hidden="true" />
        <span className="app-activity-phase">{block.phase}</span>
        <span className="app-activity-chevron" aria-hidden="true">
          <ChevronDownIcon size={11} />
        </span>
      </summary>
      <div className="app-activity-body">
        <section className="app-activity-pane">
          <h3 className="app-activity-pane-title">Reasoning</h3>
          {thinking ? (
            <pre className="app-activity-pre" ref={thinkingRef}>
              {thinking}
            </pre>
          ) : (
            <p className="app-activity-empty">This model isn't streaming reasoning.</p>
          )}
        </section>
        <section className="app-activity-pane">
          <h3 className="app-activity-pane-title">Writing</h3>
          {writing ? (
            <pre className="app-activity-pre" ref={writingRef}>
              {writingLines(writing)}
            </pre>
          ) : (
            <p className="app-activity-empty">Nothing written yet.</p>
          )}
        </section>
      </div>
    </details>
  );
}


/** Resolve a "{{expression}}" value binding through the frame's evaluator. */
function bindValue(ctx: BlockCtx, raw: string): unknown {
  if (ctx.eval) return ctx.eval(raw);
  const path = raw.replace(/^\{\{|\}\}$/g, "").trim();
  return getPath(ctx.data, path);
}

export function Block({ block, ctx, depth = 0 }: { block: StudioBlock; ctx: BlockCtx; depth?: number }): ReactNode {
  // Conditional blocks: only a declarative frame supplies `ctx.eval`; other
  // brains (logic, hosted) ignore `if` entirely.
  if (block.if && ctx.eval && !truthy(ctx.eval(block.if))) return null;
  switch (block.type) {
    case "text": {
      const inner = <WorkBody content={ctx.resolve(block.content)} error={block.error ?? false} streaming={block.streaming} />;
      const cls = dialCls(block.emph, block.face);
      return cls ? <div className={cls}>{inner}</div> : inner;
    }
    case "heading": {
      const cls = dialCls(block.emph, block.face);
      return <h3 className={`app-heading${cls ? ` ${cls}` : ""}`}>{ctx.resolve(block.content)}</h3>;
    }
    case "code":
      return <CodeBlock content={block.content} language={block.language} highlight={block.highlight} />;
    case "output": {
      const bound = typeof block.lines === "string" ? bindValue(ctx, block.lines) : block.lines;
      const lines = Array.isArray(bound) ? bound.map((l) => String(l)) : [];
      return <pre className="app-output">{lines.map((l) => ctx.resolve(l)).join("\n")}</pre>;
    }
    case "button": {
      const tone = toneCls(block.tone);
      const cls = `btn${block.variant === "ghost" ? " ghost" : ""}${tone ? ` ${tone}` : ""}`;
      return (
        <button
          type="button"
          className={cls}
          disabled={block.disabled || undefined}
          onClick={() => ctx.do(block, "click", block.label)}
        >
          {ctx.resolve(block.label)}
        </button>
      );
    }
    case "note": {
      const base = block.tone === "error" ? "form-error" : block.tone === "hint" ? "field-hint" : "app-note";
      const cls = dialCls(block.emph, block.face);
      return (
        <p className={cls ? `${base} ${cls}` : base}>
          {ctx.resolve(block.content)}
        </p>
      );
    }
    case "row": {
      const cls = layoutCls(block.align, block.justify);
      return (
        <div className={`app-row${cls ? ` ${cls}` : ""}`}>
          {block.blocks.map((child, i) => (
            <Block key={i} block={child} ctx={ctx} depth={depth + 1} />
          ))}
        </div>
      );
    }
    case "card":
      return (
        <div className="app-card">
          {block.blocks.map((child, i) => (
            <Block key={i} block={child} ctx={ctx} depth={depth + 1} />
          ))}
        </div>
      );
    case "spacer":
      return <div className={`app-spacer app-spacer-${block.size ?? "md"}`} />;
    case "divider":
      return <hr className="app-divider" />;
    case "grid": {
      let cells: string[];
      if (Array.isArray(block.cells)) cells = block.cells;
      else {
        const v = bindValue(ctx, block.cells);
        cells = Array.isArray(v) ? v.map((c) => String(c)) : [];
      }
      return (
        <div className="app-grid" style={{ gridTemplateColumns: `repeat(${block.cols}, minmax(0, 1fr))` }}>
          {cells.map((c, i) => (
            <button
              key={i}
              type="button"
              className="app-cell"
              aria-label={`Cell ${i}`}
              onClick={() => ctx.do(block, "cell", i)}
            >
              {ctx.resolve(c)}
            </button>
          ))}
        </div>
      );
    }
    case "choice": {
      const current = block.value ? ctx.resolve(block.value) : "";
      const tone = toneCls(block.tone);
      return (
        <div className={`app-choice${tone ? ` ${tone}` : ""}`} role="radiogroup">
          {block.options.map((o) => {
            const label = typeof o === "string" ? o : o.label;
            const value = typeof o === "string" ? o : o.value;
            return (
              <button
                key={value}
                type="button"
                role="radio"
                aria-checked={ctx.resolve(value) === current}
                className={`choice-pill${ctx.resolve(value) === current ? " sel" : ""}`}
                onClick={() => ctx.do(block, "choose", value)}
              >
                {label}
              </button>
            );
          })}
        </div>
      );
    }
    case "select":
      return (
        <SelectField
          label={block.label ? ctx.resolve(block.label) : undefined}
          options={block.options}
          value={block.value ? ctx.resolve(block.value) : ""}
          chip={block.chip}
          ariaLabel={block.ariaLabel}
          onChange={(v) => ctx.do(block, "choose", v)}
        />
      );
    case "input":
      return <InputBlock block={block} ctx={ctx} />;
    case "field":
      return (
        <Field
          label={block.label}
          kind={block.kind ?? "text"}
          value={block.value ? ctx.resolve(block.value) : ""}
          placeholder={block.placeholder ? ctx.resolve(block.placeholder) : undefined}
          className={block.className}
          autoFocus={block.autoFocus}
          ariaLabel={block.ariaLabel}
          onChange={(v) => ctx.do(block, "change", { key: block.key ?? "", value: v })}
        />
      );
    case "stat": {
      const cls = dialCls(block.emph, block.face);
      const tone = toneCls(block.tone);
      const classes = `app-stat${cls ? ` ${cls}` : ""}${tone ? ` ${tone}` : ""}`;
      return (
        <div className={classes}>
          <span className="app-stat-label">{ctx.resolve(block.label)}</span>
          <span className="app-stat-value">{ctx.resolve(block.value) || "—"}</span>
        </div>
      );
    }
    case "table": {
      const colsBound = typeof block.cols === "string" ? bindValue(ctx, block.cols) : block.cols;
      const cols = Array.isArray(colsBound) ? colsBound.map((c) => String(c)) : [];
      const rowsBound = typeof block.rows === "string" ? bindValue(ctx, block.rows) : block.rows;
      const rows = Array.isArray(rowsBound) ? rowsBound.filter((r): r is unknown[] => Array.isArray(r)) : [];
      return (
        <table className="app-table">
          <thead>
            <tr>
              {cols.map((c, i) => (
                <th key={i}>{ctx.resolve(c)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                {row.map((cell, j) => (
                  <td key={j}>{ctx.resolve(String(cell))}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      );
    }
    case "image":
      return (
        <img
          className="app-image"
          src={block.src}
          alt={block.alt ?? ""}
          style={block.width ? { maxWidth: block.width } : undefined}
        />
      );
    case "chat": {
      const chat = ctx.chat;
      if (!chat) return <p className="app-note">Conversation unavailable.</p>;
      const msgs = chat.messages.filter((m) => !m.error).slice(-MAX_CHAT_MESSAGES);
      return (
        <div className="app-chat" id="__chat-frame" data-chat="true">
          <div className="app-chat-history">
            {msgs.map((m, i) => (
              <div key={i} className={`app-chat-msg ${m.role}`}>
                <span className="app-chat-role">{m.role === "user" ? "You" : "Assistant"}</span>
                <WorkBody content={m.content || ""} />
              </div>
            ))}
          </div>
          <Composer
            onSend={(t, imgs) => chat.onSend(t, imgs)}
            onStop={() => {}}
            disabled={chat.streaming}
            sendMode={chat.sendMode}
            sendBlocker={null}
          />
        </div>
      );
    }
    case "raw":
      return <RawBlock block={block} ctx={ctx} />;
    // --- layout: full-window composition ---
    case "stack": {
      const layout = layoutCls(block.align, block.justify);
      return (
        <div className={`app-stack${block.fill ? " fill" : ""}${block.gap ? ` gap-${block.gap}` : ""}${layout ? ` ${layout}` : ""}`}>
          {block.blocks.map((child, i) => (
            <Block key={i} block={child} ctx={ctx} depth={depth + 1} />
          ))}
        </div>
      );
    }
    case "columns": {
      const layout = layoutCls(block.align, block.justify);
      const hasSizes = block.sizes && block.sizes.length >= 2;
      const style = hasSizes ? (Object.fromEntries(block.sizes!.map((s, i) => [`--c${i}`, s])) as React.CSSProperties) : undefined;
      return (
        <div
          className={`app-columns${block.fill ? " fill" : ""}${block.gap ? ` gap-${block.gap}` : ""}${layout ? ` ${layout}` : ""}${hasSizes ? " sized" : ""}`}
          data-sizes={hasSizes ? "" : undefined}
          style={style}
        >
          {block.blocks.map((child, i) => (
            <Block key={i} block={child} ctx={ctx} depth={depth + 1} />
          ))}
        </div>
      );
    }
    case "panel": {
      const tone = toneCls(block.tone);
      return (
        <section
          className={`app-panel${block.fill ? " fill" : ""}${block.gap ? ` gap-${block.gap}` : ""}${tone ? ` ${tone}` : ""}`}
          style={block.width ? { flexGrow: 0, flexShrink: 0, flexBasis: `${block.width}px`, width: block.width } : undefined}
        >
          {block.title && <header className="app-panel-head">{ctx.resolve(block.title)}</header>}
          <div className="app-panel-body">
            {block.blocks.map((child, i) => (
              <Block key={i} block={child} ctx={ctx} depth={depth + 1} />
            ))}
          </div>
        </section>
      );
    }
    case "toolbar": {
      const layout = layoutCls(block.align, block.justify);
      return (
        <div className={`app-toolbar${block.sticky ? " sticky" : ""}${layout ? ` ${layout}` : ""}`}>
          {block.blocks.map((child, i) => (
            <Block key={i} block={child} ctx={ctx} depth={depth + 1} />
          ))}
        </div>
      );
    }
    case "list": {
      // Only a declarative frame can resolve `each`; logic brains compute
      // their own arrays and never emit a list.
      const items = ctx.eval ? ctx.eval(block.each) : undefined;
      const list = Array.isArray(items) ? items : [];
      if (!list.length) return null;
      const as = block.as || "item";
      return (
        <div className={`app-list${block.gap ? ` gap-${block.gap}` : ""}`}>
          {list.map((item, i) => {
            const scopedData = { ...ctx.data, [as]: item };
            const scoped: BlockCtx = {
              ...ctx,
              data: scopedData,
              resolve: (s) => resolveText(s, scopedData),
              eval: ctx.eval ? (e) => evalBinding(e, scopedData) : undefined,
            };
            return (
              <div className="app-list-item" key={i}>
                {block.blocks.map((child, j) => (
                  <Block key={j} block={child} ctx={scoped} depth={depth + 1} />
                ))}
              </div>
            );
          })}
        </div>
      );
    }
    // --- app-only components ---
    case "wallhead":
      return (
        <header className="wall-text">
          <h1 className="wall-title">{block.title || "New chat"}</h1>
          <div className="wall-meta">
            {block.meta.map((m, i) => (
              <span key={i} style={{ display: "contents" }}>
                {i > 0 && <span className="wall-meta-sep">·</span>}
                <span className={m.model ? "wall-meta-model" : undefined}>{m.text}</span>
              </span>
            ))}
          </div>
        </header>
      );
    case "wallabel":
      return <WallabelView data={block} />;
    case "work":
      return (
        <article className={`work${block.streaming ? " streaming" : ""}`}>
          <WallabelView data={block.label} />
          <div className="piece">
            {block.piece.map((p, i) => (
              <Block key={i} block={p} ctx={ctx} depth={depth + 1} />
            ))}
          </div>
        </article>
      );
    case "study":
      return <Study thinking={block.content} active={block.active} thinkingMs={block.ms} />;
    case "preparing":
      return (
        <div className="studio-preparing">
          <span className="dot" aria-hidden="true" />
          {block.text ?? "Preparing your workspace…"}
        </div>
      );
    case "status":
      return (
        <div className="work-status" role="status">
          <span className="dot" aria-hidden="true" />
          {block.text}
        </div>
      );
    case "plan":
      return <PlanCard block={block} ctx={ctx} />;
    case "question":
      return (
        <div className="app-question">
          <p className="app-question-text">{block.question}</p>
          {block.assumptions && block.assumptions.length > 0 && (
            <p className="app-question-assumptions">{block.assumptions.join(" · ")}</p>
          )}
          {block.options && block.options.length > 0 && (
            <div className="app-question-actions">
              {block.options.map((o) => (
                <button
                  key={o}
                  type="button"
                  className="choice-pill"
                  onClick={() => ctx.do(block, "click", o)}
                >
                  {o}
                </button>
              ))}
              <button type="button" className="app-question-skip" onClick={() => ctx.do(block, "click", "skip")}>
                Just build it
              </button>
            </div>
          )}
        </div>
      );
    case "interrupted":
      return (
        <div className="studio-interrupted">
          <span className="dot static" aria-hidden="true" />
            <span className="studio-interrupted-text">
            {block.kind === "truncated"
              ? "The model's stream was cut off before the workspace finished."
              : block.kind === "planning"
                ? "Planning failed — the model didn't produce a usable build plan."
                : "The model wrote a workspace I couldn't read."}
          </span>
          {block.canRetry && (
            <button type="button" className="studio-retry" onClick={() => ctx.do(block, "click", "retry")}>
              Try again
            </button>
          )}
          {(block.kind === "truncated" || block.kind === "planning") && block.raw && (
            <details className="studio-raw">
              <summary>Show raw output</summary>
              <pre>{block.raw}</pre>
            </details>
          )}
        </div>
      );
    case "door":
      return (
        <button type="button" className="studio-card" onClick={() => ctx.do(block, "click", block.appId)}>
          <span className="studio-card-title">{block.title}</span>
          <span className="studio-card-arrow" aria-hidden="true">
            →
          </span>
        </button>
      );
    case "critic":
      return (
        <div className={`studio-critic ${block.status}`}>
          <span className="dot static" aria-hidden="true" />
          <span className="studio-critic-text">
            {block.status === "reviewing"
              ? "Reviewing the design…"
              : block.status === "refining"
                ? "Polishing the design…"
                : "Design review: polished"}
          </span>
        </div>
      );
    case "thread":
      return (
        <button type="button" className="studio-card thread-card" onClick={() => ctx.do(block, "click", block.appId)}>
          <span className="thread-card-main">
            <span className="studio-card-title">{block.title}</span>
            <span className="thread-card-status">
              {block.active && <span className="dot thread-card-dot" aria-hidden="true" />}
              {block.status}
            </span>
          </span>
          <span className="studio-card-arrow" aria-hidden="true">
            →
          </span>
        </button>
      );
    case "activity":
      return <ActivityCard block={block} />;
    case "none":
      return <div className="piece-none">{block.text ?? "No reply."}</div>;
    case "caret":
      return <span className="caret" aria-hidden="true" />;
    case "composer": {
      const host = ctx.composer;
      if (!host) return null;
      return (
        <Composer
          onSend={host.onSend}
          onStop={host.onStop}
          disabled={host.disabled}
          sendMode={host.sendMode}
          sendBlocker={host.sendBlocker}
        />
      );
    }
    case "chip":
      if (block.thinking) {
        return <span className="suggestion thinking">{block.label}</span>;
      }
      return (
        <button
          type="button"
          className="suggestion"
          style={block.delay ? { animationDelay: `${block.delay}ms` } : undefined}
          onClick={() => ctx.do(block, "click", block.label)}
        >
          {block.label}
        </button>
      );
    case "dot":
      return <span className={`dot${block.variant === "static" ? " static" : ""}`} aria-hidden="true" />;
    case "greetinghead":
      return (
        <div className="greeting-head">
          <span className="dot greeting-dot" aria-hidden="true" />
          <h1 className="greeting-title">
            {block.title}
            <span className="caret" aria-hidden="true" />
          </h1>
          <p className="greeting-sub">{block.sub}</p>
        </div>
      );
    case "group":
      return (
        <div className={block.className}>
          {block.blocks.map((child, i) => (
            <Block key={i} block={child} ctx={ctx} depth={depth + 1} />
          ))}
        </div>
      );
  }
}

function InputBlock({ block, ctx }: { block: Extract<StudioBlock, { type: "input" }>; ctx: BlockCtx }) {
  const [value, setValue] = useState(block.value ? ctx.resolve(block.value) : "");
  const submit = () => {
    ctx.do(block, "submit", block.kind === "number" ? Number(value) || 0 : value);
  };
  return (
    <div className="app-input-row">
      <Field
        label={block.label ? ctx.resolve(block.label) : undefined}
        kind={block.kind ?? "text"}
        value={value}
        placeholder={block.placeholder ? ctx.resolve(block.placeholder) : undefined}
        onChange={setValue}
      />
      <button type="button" className="btn" onClick={submit}>
        OK
      </button>
    </div>
  );
}

/** Inject the runtime reporter into a raw frame's document (doctype-safe). */
function rawBridgeSrc(src: string): string {
  const bridge =
    "<script>(function(){" +
    'function r(l,m,s){try{parent.postMessage({type:"raw-log",level:l,message:String(m==null?"":m).slice(0,2000),stack:s?String(s).slice(0,4000):undefined},"*")}catch(e){}}' +
    'window.onerror=function(m,_s,_l,_c,e){r("error",m,e&&e.stack);return false};' +
    'addEventListener("unhandledrejection",function(e){var x=e&&e.reason;r("error","Unhandled rejection: "+((x&&x.message)||x),x&&x.stack)});' +
    "var t=[];function ok(){var n=Date.now();t=t.filter(function(x){return n-x<1000});if(t.length>=10)return false;t.push(n);return true}" +
    "var c=console.log,w=console.warn,er=console.error;" +
    'console.log=function(){var a=[].slice.call(arguments);c.apply(console,a);if(ok())r("log",a.join(" "))};' +
    'console.warn=function(){var a=[].slice.call(arguments);w.apply(console,a);if(ok())r("warn",a.join(" "))};' +
    'console.error=function(){var a=[].slice.call(arguments);er.apply(console,a);if(ok())r("error",a.join(" "))}' +
    "})()</scr" + "ipt>";
  const after = (re: RegExp): number => {
    const m = re.exec(src);
    return m ? m.index + m[0].length : -1;
  };
  const at = [after(/<head[^>]*>/i), after(/<html[^>]*>/i), after(/<!doctype[^>]*>/i)].find((i) => i >= 0);
  return at == null ? bridge + src : src.slice(0, at) + bridge + src.slice(at);
}

function RawBlock({ block, ctx }: { block: Extract<StudioBlock, { type: "raw" }>; ctx: BlockCtx }) {
  const iframeRef = useRef<HTMLIFrameElement>(null);

  useEffect(() => {
    const onMsg = (e: MessageEvent) => {
      const d = e.data;
      if (!d || typeof d !== "object") return;
      if (e.source !== iframeRef.current?.contentWindow && !(e.source === window && d.id === block.id)) return;
      if (d.type === "raw-log" && typeof d.message === "string") {
        ctx.log?.({
          level: d.level === "error" || d.level === "warn" ? d.level : "log",
          message: (block.id ? `[${block.id}] ` : "") + d.message,
          stack: typeof d.stack === "string" ? d.stack : undefined,
        });
        return;
      }
      if (d.type === "fx-event" || d.type === "event") {
        const evName = typeof d.event === "string" ? d.event : "event";
        if (!block.events || block.events.includes(evName)) {
          const payload = d.arg !== undefined ? d.arg : d.data !== undefined ? d.data : d;
          ctx.do(block, "event", payload);
        }
      }
    };
    window.addEventListener("message", onMsg);
    return () => window.removeEventListener("message", onMsg);
  }, [block, ctx]);

  const cls = `app-raw${block.fill ? " fill" : ""}`;
  const style: React.CSSProperties = {};
  if (block.width) {
    style.width = `${block.width}px`;
    style.maxWidth = "100%";
  }

  return (
    <iframe
      ref={iframeRef}
      className={cls}
      style={style}
      sandbox="allow-scripts"
      srcDoc={rawBridgeSrc(block.src)}
      title={block.id ?? "Custom Region"}
      data-raw-id={block.id}
    />
  );
}

export function BlocksView({ blocks, ctx }: { blocks: StudioBlock[]; ctx: BlockCtx }) {
  return (
    <>
      {blocks.map((b, i) => (
        <Block key={i} block={b} ctx={ctx} />
      ))}
    </>
  );
}
