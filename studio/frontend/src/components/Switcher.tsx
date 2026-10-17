import { useEffect, useMemo, useRef, useState } from "react";
import { SearchIcon } from "./Icon";
import type { Conversation } from "../types";

interface SwitcherProps {
  conversations: Conversation[];
  activeId: number | null;
  onSelect: (id: number) => void;
  onNew: () => void;
  onClose: () => void;
}

interface Item {
  key: string;
  label: string;
  hint?: string;
  run: () => void;
}

const MAX_RESULTS = 12;

/** ⌘K: jump between conversations. */
export default function Switcher({
  conversations,
  activeId,
  onSelect,
  onNew,
  onClose,
}: SwitcherProps) {
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const items = useMemo<Item[]>(() => {
    const q = query.trim().toLowerCase();
    const list: Item[] = [];
    if (!q) {
      list.push({ key: "new", label: "New chat", hint: "create", run: onNew });
    }
    const sorted = [...conversations].sort(
      (a, b) => (b.createdAt ?? b.id) - (a.createdAt ?? a.id) || b.id - a.id
    );
    for (const c of sorted) {
      const title = c.title || "New chat";
      if (q && !title.toLowerCase().includes(q)) continue;
      list.push({
        key: String(c.id),
        label: title,
        hint: c.id === activeId ? "current" : undefined,
        run: () => onSelect(c.id),
      });
      if (list.length >= MAX_RESULTS) break;
    }
    return list;
  }, [conversations, activeId, query, onNew, onSelect]);

  useEffect(() => {
    setCursor(0);
  }, [query]);

  const pick = (item: Item | undefined) => {
    if (!item) return;
    item.run();
    onClose();
  };

  return (
    <div
      className="switcher-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="switcher" role="dialog" aria-label="Switch conversation">
        <label className="switcher-input">
          <SearchIcon size={14} />
          <input
            ref={inputRef}
            type="text"
            value={query}
            placeholder="Search conversations…"
            aria-label="Search conversations"
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                e.preventDefault();
                onClose();
              } else if (e.key === "ArrowDown") {
                e.preventDefault();
                setCursor((c) => Math.min(items.length - 1, c + 1));
              } else if (e.key === "ArrowUp") {
                e.preventDefault();
                setCursor((c) => Math.max(0, c - 1));
              } else if (e.key === "Enter") {
                e.preventDefault();
                pick(items[cursor]);
              }
            }}
          />
        </label>
        <ul className="switcher-list" role="listbox">
          {items.length === 0 ? (
            <li className="switcher-none">No match.</li>
          ) : (
            items.map((item, i) => (
              <li
                key={item.key}
                role="option"
                aria-selected={i === cursor}
                className={i === cursor ? "selected" : undefined}
                onMouseEnter={() => setCursor(i)}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => pick(item)}
              >
                <span className="switcher-label">{item.label}</span>
                {item.hint && <span className="switcher-hint">{item.hint}</span>}
              </li>
            ))
          )}
        </ul>
      </div>
    </div>
  );
}
