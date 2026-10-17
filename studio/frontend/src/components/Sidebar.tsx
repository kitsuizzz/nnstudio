import { useState } from "react";
import {
  ChevronLeftIcon,
  ChevronRightIcon,
  MoonIcon,
  PlusIcon,
  SearchIcon,
  SunIcon,
  TrashIcon,
} from "./Icon";
import type { Conversation } from "../types";
import type { Theme } from "../App";

interface SidebarProps {
  conversations: Conversation[];
  activeId: number | null;
  activeTypedTitle: string;
  streamingId: number | null;
  collapsed: boolean;
  /** Mobile drawer state. */
  open: boolean;
  theme: Theme;
  onToggleTheme: () => void;
  onSelect: (id: number) => void;
  onNewChat: () => void;
  onDelete: (id: number) => void;
  onOpenSearch: () => void;
  onToggleCollapsed: () => void;
  onClose: () => void;
}

const isMac =
  typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform ?? "");

export default function Sidebar({
  conversations,
  activeId,
  activeTypedTitle,
  streamingId,
  collapsed,
  open,
  theme,
  onToggleTheme,
  onSelect,
  onNewChat,
  onDelete,
  onOpenSearch,
  onToggleCollapsed,
  onClose,
}: SidebarProps) {
  const [confirmId, setConfirmId] = useState<number | null>(null);
  const sorted = [...conversations].sort(
    (a, b) => (b.createdAt ?? b.id) - (a.createdAt ?? a.id) || b.id - a.id
  );

  return (
    <>
      {open && <div className="sidebar-backdrop" onClick={onClose} />}
      <aside className={`sidebar${collapsed ? " collapsed" : ""}${open ? " open" : ""}`}>
        <div className="sidebar-head">
          <button
            className="head-btn sidebar-toggle"
            type="button"
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            onClick={onToggleCollapsed}
          >
            {collapsed ? <ChevronRightIcon size={14} /> : <ChevronLeftIcon size={14} />}
          </button>
        </div>

        <button className="sidebar-new" type="button" onClick={onNewChat} title="New chat">
          <PlusIcon size={15} />
          <span className="sidebar-text">New chat</span>
        </button>

        <button
          className="sidebar-search"
          type="button"
          onClick={onOpenSearch}
          title={`Search conversations (${isMac ? "⌘" : "Ctrl+"}K)`}
        >
          <SearchIcon size={13} />
          <span className="sidebar-text">Search</span>
          <kbd className="sidebar-text">{isMac ? "⌘K" : "Ctrl K"}</kbd>
        </button>

        <nav className="conv-list" aria-label="Conversations">
          {sorted.length === 0 ? (
            <div className="conv-empty sidebar-text">No conversations yet</div>
          ) : (
            sorted.map((c) => {
              const title = (c.id === activeId ? activeTypedTitle : c.title) || "New chat";
              const live = c.id === streamingId;
              const confirming = confirmId === c.id;
              return (
                <div
                  key={c.id}
                  className={`conv-item${c.id === activeId ? " active" : ""}`}
                  role="button"
                  tabIndex={0}
                  title={title}
                  onClick={() => onSelect(c.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onSelect(c.id);
                    }
                  }}
                >
                  <span className="conv-letter" aria-hidden="true">
                    {live ? <span className="dot" /> : title.trim().charAt(0).toUpperCase() || "•"}
                  </span>
                  <span className="conv-title sidebar-text">{title}</span>
                  {live && <span className="dot conv-live sidebar-text" role="img" aria-label="In progress" />}
                  {confirming ? (
                    <button
                      className="text-btn danger conv-confirm sidebar-text"
                      type="button"
                      autoFocus
                      onBlur={() => setConfirmId(null)}
                      onClick={(e) => {
                        e.stopPropagation();
                        setConfirmId(null);
                        onDelete(c.id);
                      }}
                      onKeyDown={(e) => e.stopPropagation()}
                    >
                      remove?
                    </button>
                  ) : (
                    <button
                      className="icon-text-btn conv-remove sidebar-text"
                      type="button"
                      aria-label="Delete conversation"
                      title="Delete"
                      onClick={(e) => {
                        e.stopPropagation();
                        setConfirmId(c.id);
                      }}
                      onKeyDown={(e) => e.stopPropagation()}
                    >
                      <TrashIcon size={13} />
                    </button>
                  )}
                </div>
              );
            })
          )}
        </nav>

        <div className="sidebar-foot">
          <button
            className="sidebar-link"
            type="button"
            onClick={onToggleTheme}
            title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
          >
            {theme === "dark" ? <SunIcon size={15} /> : <MoonIcon size={15} />}
            <span className="sidebar-text">{theme === "dark" ? "Light theme" : "Dark theme"}</span>
          </button>
        </div>
      </aside>
    </>
  );
}
