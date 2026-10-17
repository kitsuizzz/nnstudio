/**
 * The conversation wall — the studio's primary default app. A pure brain: it
 * reads the live conversation and emits registry blocks (a wallhead, one work
 * per exchange, and a docked composer). Exhibition renders those blocks
 * through the shared component library; no hand-rolled markup of the content.
 */

import { pairTurns, type Turn } from "./turns";
import { formatCompact, formatTime, formatDate } from "./format";
import type { Conversation, CurrentMeta, SendMode, Studio, StudioBlock, Wallabel } from "../types";

export interface WallCtx {
  conversation: Conversation;
  /** Title as currently typed out by the typewriter. */
  title: string;
  current: CurrentMeta;
  streaming: boolean;
  firstTokenReceived: boolean;
  /** Speed counter (tok/s) for the streaming exchange; decode stays null until the stream settles. */
  speeds?: { prefill: number; decode: number | null } | null;
  /** AI-built workspaces the doors can open. */
  studios: Studio[];
  sendMode: SendMode;
  sendBlocker: string | null;
}

export interface WallContent {
  head: StudioBlock;
  works: StudioBlock[];
  dock: StudioBlock[];
}

const LONG_TITLE_CHARS = 600;
const LONG_TITLE_LINES = 12;

function isLongTitle(text: string): boolean {
  if (text.length > LONG_TITLE_CHARS) return true;
  let lines = 1;
  for (let i = 0; i < text.length; i++) if (text.charCodeAt(i) === 10) lines++;
  return lines > LONG_TITLE_LINES;
}

/** "· 12.4k in × 46 out tok/s" — the measured speed of an exchange. */
function speedSuffix(speeds?: { prefill: number; decode: number | null } | null): string {
  if (!speeds || !speeds.prefill) return "";
  return ` · ${formatCompact(speeds.prefill)} in × ${speeds.decode != null ? formatCompact(speeds.decode) : "—"} out tok/s`;
}

interface WorkOpts {
  streaming: boolean;
  thinkingActive: boolean;
  current: CurrentMeta;
  studio?: Studio;
  canRetry: boolean;
  speeds?: { prefill: number; decode: number | null } | null;
}

function workBlock(turn: Turn, opts: WorkOpts): StudioBlock {
  const { prompt, response } = turn;
  const meta = response?.meta;
  const agentName = meta?.agentName ?? opts.current.agentName;
  const model = meta?.model ?? opts.current.model;
  const effort = meta?.effort ?? opts.current.effort;
  const usage = meta?.usage;
  const speeds = meta?.speeds;
  const images = prompt.images ?? [];
  const title = prompt.content.trim();
  const long = isLongTitle(title);
  const showStudy = !!response && (!!response.thinking || opts.thinkingActive);

  const piece: StudioBlock[] = [];
  if (response) {
    if (response.content) piece.push({ type: "text", content: response.content, error: response.error, streaming: opts.streaming });
    else if (opts.streaming) piece.push({ type: "caret" });
    if (response.studioPending) {
      piece.push({ type: "preparing" });
    } else {
      if (response.studioFailed) {
        piece.push({
          type: "interrupted",
          kind: response.studioFailed,
          raw: response.studioFailed === "truncated" ? response.studioRaw : undefined,
          canRetry: opts.canRetry,
        });
      }
      if (opts.studio) piece.push({ type: "door", title: opts.studio.title, appId: opts.studio.id });
    }
  } else {
    piece.push({ type: "none" });
  }

  const label: Wallabel = {
    no: turn.no,
    live: opts.streaming,
    kicker: "You",
    title,
    long,
    time: prompt.at != null ? formatTime(prompt.at) : undefined,
    images: images.length ? images : undefined,
    artist: agentName,
    medium: `· ${model} · effort ${effort}`,
    dims: usage
      ? `${usage.prompt.toLocaleString()} in × ${usage.completion.toLocaleString()} out tokens${speedSuffix(speeds)}`
      : opts.streaming
        ? `counting tokens…${speedSuffix(opts.speeds)}`
        : "— × — tokens",
    study: showStudy
      ? { content: response?.thinking ?? "", active: opts.thinkingActive, ms: response?.thinkingMs }
      : undefined,
  };

  return { type: "work", label, piece, streaming: opts.streaming };
}

export function wallBrain(ctx: WallCtx): WallContent {
  const { conversation, title, current, streaming, firstTokenReceived, studios } = ctx;
  const turns = pairTurns(conversation.messages);
  const lastTurn = turns[turns.length - 1];
  const lastResponse = lastTurn?.response;
  const thinkingActive = streaming && firstTokenReceived && !!lastResponse && lastResponse.content === "";
  const count = turns.length;
  const date = formatDate(conversation.createdAt ?? Date.now());

  const head: StudioBlock = {
    type: "wallhead",
    title: title || "New chat",
    meta: [
      { text: current.agentName },
      { text: current.model, model: true },
      { text: date },
      { text: `${count} ${count === 1 ? "exchange" : "exchanges"}` },
    ],
  };

  const works = turns.map((turn, i) =>
    workBlock(turn, {
      streaming: streaming && i === turns.length - 1,
      thinkingActive: i === turns.length - 1 && thinkingActive,
      current,
      studio: turn.response?.studioId
        ? (studios.find((s) => s.id === turn.response?.studioId) ?? undefined)
        : undefined,
      canRetry: i === turns.length - 1,
      speeds: streaming && i === turns.length - 1 ? ctx.speeds : undefined,
    })
  );

  return { head, works, dock: [{ type: "composer" }] };
}
