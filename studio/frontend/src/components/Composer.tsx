import { useRef, useState } from "react";
import { ArrowUpIcon, PaperclipIcon, StopIcon, XIcon } from "./Icon";
import { processImageFile } from "../lib/image";
import type { SendMode } from "../types";

interface ComposerProps {
  onSend: (text: string, images: string[]) => void;
  onStop: () => void;
  disabled: boolean;
  sendMode: SendMode;
  /** Why the composer can't send right now, if anything. */
  sendBlocker: string | null;
}

const MAX_IMAGES = 4;

const isMac =
  typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform ?? "");

/** Docked message box at the bottom of the wall. */
export default function Composer({
  onSend,
  onStop,
  disabled,
  sendMode,
  sendBlocker,
}: ComposerProps) {
  const [value, setValue] = useState("");
  const [images, setImages] = useState<string[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dragDepth = useRef(0);

  const addFiles = async (files: FileList | File[]) => {
    if (disabled) return;
    const list = Array.from(files).filter((f) => f.type.startsWith("image/"));
    const room = MAX_IMAGES - images.length;
    if (!list.length || room <= 0) return;
    const results: string[] = [];
    for (const file of list.slice(0, room)) {
      try {
        results.push(await processImageFile(file));
      } catch {
        // unreadable file, skip it
      }
    }
    if (results.length) setImages((prev) => [...prev, ...results].slice(0, MAX_IMAGES));
  };

  const submit = () => {
    if (disabled || sendBlocker) return;
    if (!value.trim() && images.length === 0) return;
    onSend(value, images);
    setValue("");
    setImages([]);
  };

  const sendHint =
    sendMode === "ctrlEnter" ? `${isMac ? "⌘" : "Ctrl"} ⏎ to send` : "⏎ to send · ⇧⏎ for a new line";

  return (
    <footer className="composer-wrap">
      <div
        className={`composer${dragOver ? " dragging" : ""}`}
        onDragEnter={(e) => {
          e.preventDefault();
          if (disabled) return;
          dragDepth.current += 1;
          setDragOver(true);
        }}
        onDragOver={(e) => {
          e.preventDefault();
        }}
        onDragLeave={() => {
          dragDepth.current = Math.max(0, dragDepth.current - 1);
          if (dragDepth.current === 0) setDragOver(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          dragDepth.current = 0;
          setDragOver(false);
          void addFiles(e.dataTransfer.files);
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp,image/gif"
          multiple
          hidden
          onChange={(e) => {
            void addFiles(e.target.files ?? []);
            e.target.value = "";
          }}
        />

        {images.length > 0 && (
          <div className="composer-attachments">
            {images.map((src, i) => (
              <div className="label-ref" key={i}>
                <img src={src} alt="" />
                <button
                  type="button"
                  className="label-ref-remove"
                  aria-label="Remove image"
                  onClick={() => setImages((prev) => prev.filter((_, j) => j !== i))}
                >
                  <XIcon size={9} />
                </button>
              </div>
            ))}
          </div>
        )}

        <textarea
          className="composer-input"
          rows={1}
          value={value}
          placeholder="Write a message…"
          aria-label="Message input"
          onChange={(e) => setValue(e.target.value)}
          onPaste={(e) => {
            const files = Array.from(e.clipboardData.files);
            if (files.some((f) => f.type.startsWith("image/"))) {
              e.preventDefault();
              void addFiles(files);
            }
          }}
          onKeyDown={(e) => {
            if (e.key !== "Enter") return;
            if (e.nativeEvent.isComposing || e.keyCode === 229) return;
            const shouldSend =
              sendMode === "ctrlEnter" ? e.ctrlKey || e.metaKey : !e.shiftKey;
            if (shouldSend) {
              e.preventDefault();
              submit();
            }
          }}
        />

        {sendBlocker && <div className="composer-blocker">{sendBlocker}</div>}

        <div className="composer-row">
          <button
            className="composer-tool"
            type="button"
            aria-label="Attach image"
            title="Attach image (or drop / paste)"
            disabled={disabled || images.length >= MAX_IMAGES}
            onClick={() => fileInputRef.current?.click()}
          >
            <PaperclipIcon size={14} />
            <span>Attach</span>
          </button>

          <span className="composer-hint">{sendHint}</span>

          {disabled ? (
            <button className="send-btn stop" type="button" aria-label="Stop" title="Stop" onClick={onStop}>
              <StopIcon size={11} />
            </button>
          ) : (
            <button
              className="send-btn"
              type="button"
              aria-label="Send"
              title={sendBlocker ?? "Send"}
              disabled={sendBlocker != null}
              onClick={submit}
            >
              <ArrowUpIcon size={16} />
            </button>
          )}
        </div>
      </div>
    </footer>
  );
}
