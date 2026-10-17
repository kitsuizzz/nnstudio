import { useEffect, useRef, useState, type FormEvent } from "react";
import { XIcon } from "./Icon";
import { providersFrom, signin, signup } from "../lib/auth";
import { CAPTCHA_SITE_KEY, loadTurnstile } from "../lib/captcha";
import type { AccountUser, SavedProvider } from "../types";

interface AuthDialogProps {
  mode: "signin" | "signup";
  onClose: () => void;
  /** The server confirmed; adopt the account + its saved providers (chip updates behind the dialog). */
  onDone: (
    user: AccountUser,
    token: string,
    providers: SavedProvider[],
    accessKey: string,
    adminKey: string | null,
    agents: string | null
  ) => void;
  /** Flip between the two forms, keeping what has been typed. */
  onSwitchMode: (mode: "signin" | "signup") => void;
  /** Extra context under the title (e.g. why the key is needed again). */
  hint?: string;
}

/**
 * The sign-in / sign-up dialog, opened from the top bar's account menu.
 *
 * Sign-up is one click: the "I'm not a robot" checkbox is the entire form,
 * and the moment it is solved the dialog submits and flips to a one-time
 * reveal of the account's access key (the only credential, shown exactly
 * once). Sign-in is the reverse: paste that key.
 */
export default function AuthDialog({ mode, onClose, onDone, onSwitchMode, hint }: AuthDialogProps) {
  const [keyValue, setKeyValue] = useState("");
  // The key the just-created account was issued — React state only, never
  // stored: after the dialog closes it exists just server-side and in
  // wherever the user wrote it down.
  const [issuedKey, setIssuedKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // The Turnstile token is single-use, so a second submit while the first is
  // in flight would send a consumed token; a ref guards the race that the
  // (async) busy state cannot.
  const busyRef = useRef(false);

  // The Turnstile "I'm not a robot" widget: sign-up only, and only when a
  // site key was baked in at build time. "loading" until the script and
  // widget are up, "failed" when the script couldn't load (e.g. offline).
  const captchaRef = useRef<HTMLDivElement>(null);
  const widgetIdRef = useRef("");
  const pollRef = useRef(0);
  const [captcha, setCaptcha] = useState<"none" | "loading" | "ready" | "failed">(
    CAPTCHA_SITE_KEY ? "loading" : "none",
  );

  /** The one click: the solved token is the sign-up intent, sent on the spot. */
  const submitSignup = async (captchaToken?: string) => {
    if (busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    setError(null);
    try {
      const res = await signup(captchaToken);
      if (!res.user || !res.token || !res.key) throw new Error("The server returned no session.");
      // Adopt the account first (the chip updates behind the dialog), then
      // flip to the reveal: the key is shown exactly once, in this dialog.
      // The browser minted it, so it is also what unlocks the encrypted store.
      onDone(res.user, res.token, providersFrom(res), res.key, res.adminKey ?? null, res.agents ?? null);
      setIssuedKey(res.key);
    } catch (err) {
      // A token is single-use and the failed request already consumed it —
      // hand the widget a fresh one so the next tick produces a new token.
      if (captchaToken) window.turnstile?.reset(widgetIdRef.current || undefined);
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  };

  useEffect(() => {
    if (mode !== "signup" || !CAPTCHA_SITE_KEY || issuedKey) return;
    let disposed = false;
    setCaptcha("loading");
    void loadTurnstile().then((ts) => {
      if (disposed) return;
      if (!ts) {
        setCaptcha("failed");
        return;
      }
      const el = captchaRef.current;
      if (!el) return;
      const dark = document.documentElement.dataset.theme === "dark";
      widgetIdRef.current = ts.render(el, {
        sitekey: CAPTCHA_SITE_KEY,
        theme: dark ? "dark" : "light",
        onSuccess: (token) => {
          void submitSignup(token);
        },
      });
      setCaptcha("ready");
      // Headless/automated browsers may be handed a token without the widget
      // ever firing onSuccess, so poll for an unsent token and submit it —
      // a token existing means the check was just solved, which *is* the
      // one-click intent. The busy guard makes onSuccess + poll a harmless
      // double trigger.
      pollRef.current = window.setInterval(() => {
        if (disposed || busyRef.current) return;
        const token = window.turnstile?.getResponse(widgetIdRef.current || undefined);
        if (token) void submitSignup(token);
      }, 250);
    });
    return () => {
      disposed = true;
      window.clearInterval(pollRef.current);
      if (widgetIdRef.current) {
        window.turnstile?.remove(widgetIdRef.current);
        widgetIdRef.current = "";
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, issuedKey]);

  const submitSignin = async (e: FormEvent) => {
    e.preventDefault();
    if (busyRef.current || !keyValue.trim()) return;
    busyRef.current = true;
    setBusy(true);
    setError(null);
    try {
      const res = await signin(keyValue);
      if (!res.user || !res.token) throw new Error("The server returned no session.");
      onDone(res.user, res.token, providersFrom(res), keyValue.trim(), res.adminKey ?? null, res.agents ?? null);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  };

  const copyKey = async (key: string) => {
    try {
      await navigator.clipboard.writeText(key);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard unavailable (permissions, non-secure context) — the key
      // stays selectable in the box.
    }
  };

  const reveal = mode === "signup" && issuedKey !== null;

  return (
    <div
      className="auth-overlay"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="auth-dialog"
        role="dialog"
        aria-modal="true"
        aria-label={reveal ? "Your account is ready" : mode === "signup" ? "Create an account" : "Sign in"}
      >
        <button type="button" className="auth-close" aria-label="Close" onClick={onClose}>
          <XIcon size={14} />
        </button>
        {reveal ? (
          <>
            <h2 className="auth-title">Your account is ready</h2>
            <p className="auth-sub">
              This access key is shown <b>only once</b> — it is the only way
              back into this account, so store it somewhere safe.
            </p>
            <div className="auth-key" aria-label="Your access key">
              {issuedKey}
            </div>
            <div className="auth-reveal-actions">
              <button type="button" className="btn" onClick={() => void copyKey(issuedKey)}>
                {copied ? "Copied" : "Copy key"}
              </button>
              <button type="button" className="btn ghost" onClick={onClose}>
                I've stored it
              </button>
            </div>
          </>
        ) : (
          <>
            <h2 className="auth-title">{mode === "signup" ? "Create an account" : "Sign in"}</h2>
            <p className="auth-sub">
              {mode === "signup"
                ? CAPTCHA_SITE_KEY
                  ? "One click is all it takes: tick the box below and your account is created. It issues an access key, shown to you exactly once."
                  : "Click below and your account is created. It issues an access key, shown to you exactly once."
                : hint ?? "Welcome back. Paste the access key you stored when you created your account."}
            </p>
            <form className="auth-form" onSubmit={mode === "signin" ? (e) => void submitSignin(e) : undefined}>
              {mode === "signin" && (
                <label className="field">
                  <span className="field-label">Access key</span>
                  <input
                    value={keyValue}
                    onChange={(e) => setKeyValue(e.target.value)}
                    required
                    autoFocus
                    placeholder="ast_…"
                    className="field-input-mono"
                    spellCheck={false}
                  />
                </label>
              )}
              {mode === "signup" && CAPTCHA_SITE_KEY && (
                <div ref={captchaRef} className="auth-captcha" aria-label="Robot check" />
              )}
              {mode === "signup" && !CAPTCHA_SITE_KEY && (
                <button type="submit" className="btn" disabled={busy} onClick={() => void submitSignup()}>
                  {busy ? "Working…" : "Create account"}
                </button>
              )}
              {mode === "signin" && (
                <button type="submit" className="btn" disabled={busy}>
                  {busy ? "Working…" : "Sign in"}
                </button>
              )}
              {error && (
                <p className="form-error" role="alert">
                  {error}
                </p>
              )}
              {mode === "signup" && CAPTCHA_SITE_KEY && captcha === "failed" && (
                <p className="form-error" role="alert">
                  The robot check couldn't load — check your connection and reload the page.
                </p>
              )}
              <button
                type="button"
                className="auth-switch"
                onClick={() => onSwitchMode(mode === "signup" ? "signin" : "signup")}
              >
                {mode === "signup" ? (
                  <>
                    Already have an account? <b>Sign in</b>
                  </>
                ) : (
                  <>
                    No account yet? <b>Create one</b>
                  </>
                )}
              </button>
            </form>
          </>
        )}
      </div>
    </div>
  );
}
