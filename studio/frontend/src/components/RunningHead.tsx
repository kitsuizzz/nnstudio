import type { ReactNode } from "react";
import { CodeIcon, MenuIcon } from "./Icon";
import { formatCompact } from "../lib/format";

interface RunningHeadProps {
  /** Conversation title; shown once the wall text has scrolled away. */
  title: string;
  showTitle: boolean;
  /** Estimated size of the next turn's prompt — what the next send would ship. */
  nextTokens: number;
  /** The active model's context window, from the provider's /models endpoint. */
  maxTokens: number | null;
  onMenu: () => void;
  /** The model switcher (the chip that opens the Server / BYOK menu). */
  modelPicker?: ReactNode;
  /** Open the Code pane for this app. */
  onOpenCode?: () => void;
  /** Measured exchange speed (tok/s): live while streaming, final when idle. */
  speeds?: { prefill: number; decode: number | null } | null;
  /** True while the conversation is still streaming. */
  live?: boolean;
  /** Leave a build thread: renders a back button instead of the menu. */
  onBack?: () => void;
}

export default function RunningHead({
  title,
  showTitle,
  nextTokens,
  maxTokens,
  onMenu,
  onBack,
  onOpenCode,
  modelPicker,
  speeds,
  live,
}: RunningHeadProps) {
  return (
    <header className="head">
      <div className="head-left">
        {onBack ? (
          <button className="head-btn head-back" type="button" aria-label="Back to conversation" title="Back to conversation" onClick={onBack}>
            ←
          </button>
        ) : (
          <button
            className="head-btn head-menu"
            type="button"
            aria-label="Open navigation"
            onClick={onMenu}
          >
            <MenuIcon size={17} />
          </button>
        )}
        <div className={`head-title${showTitle ? " visible" : ""}`} aria-hidden={!showTitle}>
          {title}
        </div>
      </div>
      <div className="head-right">
        {modelPicker}
        {onOpenCode && (
          <button
            className="head-btn head-code"
            type="button"
            aria-label="Edit this app's code"
            title="Edit this app's code"
            onClick={onOpenCode}
          >
            <CodeIcon size={17} />
          </button>
        )}
        {nextTokens > 0 && (
          <div className="head-sum" tabIndex={0}>
            {formatCompact(nextTokens)} tokens
            <span className="tooltip" role="tooltip">
              {maxTokens != null
                ? `${nextTokens.toLocaleString()} / ${maxTokens.toLocaleString()} tok used`
                : `${nextTokens.toLocaleString()} tok — the next turn's prompt; the model's max is not published by this provider`}
            </span>
          </div>
        )}
        {speeds && (
          <div className={`head-speeds${live ? " live" : ""}`} tabIndex={0}>
            {formatCompact(speeds.prefill)} prefill ·{" "}
            {speeds.decode != null ? formatCompact(speeds.decode) : "—"} decode tok/s
            <span className="tooltip" role="tooltip">
              Prefill {speeds.prefill.toLocaleString()} tok/s · Decode{" "}
              {speeds.decode != null
                ? `${speeds.decode.toLocaleString()} tok/s (average over the streaming time)`
                : live
                  ? "measured at the end of the stream"
                  : "instant"}
            </span>
          </div>
        )}
      </div>
    </header>
  );
}
