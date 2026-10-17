import { useEffect, useRef, useState } from "react";
import type { AgentOption } from "../lib/agents";

interface AgentSwitcherProps {
  /** The agents the user can ride, Chat Kitsui first. */
  agents: AgentOption[];
  /** The agent this conversation rides. */
  activeId: string;
  onPick: (agentId: string) => void;
  /** Open Settings → Agent Settings. */
  onManage: () => void;
}

const categoryLabel = (category: AgentOption["category"]) => (category === "server" ? "Server" : "BYOK");

/**
 * The running head's agent chip: the agent the chat talks to, and a menu of
 * every agent the user has. The model is shown under it, so the provider stays
 * visible without being the thing the user chooses.
 */
export default function AgentSwitcher({ agents, activeId, onPick, onManage }: AgentSwitcherProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const active = agents.find((a) => a.id === activeId) ?? agents[0];

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    const onDown = (e: PointerEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onDown);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onDown);
    };
  }, [open]);

  return (
    <div className="models-switcher" ref={rootRef}>
      <button
        className="models-chip"
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="Choose an agent"
        title="Choose an agent"
        onClick={() => setOpen((o) => !o)}
      >
        <span className="models-chip-label">{active?.name ?? "Agent"}</span>
        {active && <span className="models-chip-model">{active.model || "no model"}</span>}
        {active && <span className={`models-tag ${active.category}`}>{categoryLabel(active.category)}</span>}
      </button>
      {open && (
        <div className="models-menu" role="menu" aria-label="Agents">
          {agents.map((agent) => {
            const selected = agent.id === activeId;
            return (
              <button
                key={agent.id}
                className={`models-row${selected ? " selected" : ""}`}
                type="button"
                role="menuitemradio"
                aria-checked={selected}
                onClick={() => {
                  onPick(agent.id);
                  setOpen(false);
                }}
              >
                <span className="models-row-id">{agent.name}</span>
                <span className="models-row-sub">
                  {agent.model || "no model"} · {agent.providerName}
                </span>
                <span className={`models-tag ${agent.category}`}>{categoryLabel(agent.category)}</span>
              </button>
            );
          })}
          <button
            className="models-manage"
            type="button"
            role="menuitem"
            onClick={() => {
              setOpen(false);
              onManage();
            }}
          >
            Manage agents…
          </button>
        </div>
      )}
    </div>
  );
}
