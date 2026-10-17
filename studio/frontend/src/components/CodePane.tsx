import { useEffect, useRef, useState } from "react";
import { CodeIcon, XIcon } from "./Icon";

interface CodePaneProps {
  /** The app being edited (a default app that runs on a code brain). */
  appId: string;
  /** Display name for the pane header. */
  title: string;
  /** The app's effective source: the user's override if any, else the shipped one. */
  source: string;
  /** Whether an override is currently stored for this app. */
  edited: boolean;
  /** Persist the buffer as the app's source override. */
  onSave: (source: string) => void;
  /** Drop the stored override; the app reverts to the shipped source. */
  onDiscard: () => void;
  onClose: () => void;
  /** Send the current buffer plus an instruction to the model; resolves the new source. */
  onAssist: (buffer: string, instruction: string) => Promise<string>;
}

/**
 * The Code pane: the direct-editing flow for a built-in app. The pane takes
 * over the stage, shows the app's effective brain source in a plain
 * textarea, and offers both edit flows — save the buffer by hand, or ask the
 * model to apply a change (its reply lands in the buffer for review, never
 * straight into the app).
 */
export default function CodePane({
  appId,
  title,
  source,
  edited,
  onSave,
  onDiscard,
  onClose,
  onAssist,
}: CodePaneProps) {
  const [text, setText] = useState(source);
  const [dirty, setDirty] = useState(false);
  const [assistMsg, setAssistMsg] = useState("");
  const [assisting, setAssisting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const prevSource = useRef(source);

  useEffect(() => {
    if (source !== prevSource.current) {
      prevSource.current = source;
      setText(source);
      setDirty(false);
    }
  }, [source]);

  const handleSave = () => {
    if (!dirty) return;
    onSave(text);
  };

  const runAssist = async () => {
    const instruction = assistMsg.trim();
    if (!instruction || assisting) return;
    setAssisting(true);
    setError(null);
    try {
      const next = await onAssist(text, instruction);
      setText(next);
      setDirty(next !== source);
      setAssistMsg("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setAssisting(false);
    }
  };

  const status = dirty ? "unsaved changes" : edited ? "edited" : "shipped";

  return (
    <main className="code-pane" aria-label={`${title} code`}>
      <header className="code-appbar">
        <h2 className="code-appbar-title">
          <CodeIcon size={15} />
          {title}
        </h2>
        <span className={`code-status${edited ? " edited" : ""}`} data-app={appId}>
          {status}
        </span>
        <div className="code-appbar-actions">
          {edited && (
            <button className="btn ghost small" type="button" onClick={onDiscard}>
              Restore default
            </button>
          )}
          <button className="btn small" type="button" disabled={!dirty} onClick={handleSave}>
            Save
          </button>
          <button
            className="head-btn"
            type="button"
            aria-label="Close code pane"
            title="Close"
            onClick={onClose}
          >
            <XIcon size={16} />
          </button>
        </div>
      </header>
      <div className="code-body">
        <textarea
          className="code-src"
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            setDirty(true);
          }}
          spellCheck={false}
          aria-label={`${title} source code`}
        />
      </div>
      <footer className="code-dock">
        <input
          className="code-assist"
          type="text"
          placeholder="Ask the model to change this code…"
          value={assistMsg}
          onChange={(e) => {
            setAssistMsg(e.target.value);
            setError(null);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") void runAssist();
          }}
          aria-label="AI code assist"
        />
        <button
          className="btn small"
          type="button"
          disabled={assisting || !assistMsg.trim()}
          onClick={() => void runAssist()}
        >
          {assisting ? "Thinking…" : "Ask"}
        </button>
        {error && (
          <span className="code-assist-error" role="alert">
            {error}
          </span>
        )}
      </footer>
    </main>
  );
}
