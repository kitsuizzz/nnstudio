import { useEffect, useRef, useState } from "react";
import { ChevronDownIcon, SettingsIcon, UserIcon } from "./Icon";
import type { AccountUser } from "../types";

export type AccountState = "loading" | "guest" | "user";

interface TopBarProps {
  /** The signed-in account (null until /auth/me resolves or when a guest). */
  account: AccountUser | null;
  /** "loading" = a token cookie exists and /auth/me is in flight. */
  accountState: AccountState;
  onSignIn: () => void;
  onSignUp: () => void;
  onSignOut: () => void;
  /** Opens the back-of-house settings page. */
  onOpenSettings: () => void;
}

/**
 * The one piece of chrome that is always on screen, in every view: the brand
 * on the left, the visitor's account on the right (guest vs. signed in, with
 * the sign-in / sign-up / sign-out menu).
 */
export default function TopBar({ account, accountState, onSignIn, onSignUp, onSignOut, onOpenSettings }: TopBarProps) {
  const [open, setOpen] = useState(false);
  // Two-stage sign-out: the first click arms this (the menu shows the
  // "Are you sure?" confirm + the button re-labels), the second click signs out.
  const [confirming, setConfirming] = useState(false);
  const rootRef = useRef<HTMLElement | null>(null);

  // A confirmation must never survive the menu closing (outside click / Escape)
  // or linger into the next open.
  useEffect(() => {
    if (!open) setConfirming(false);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const label = accountState === "user" && account ? account.name : accountState === "loading" ? "…" : "Guest";
  const initials = account ? account.name.trim().slice(0, 1).toUpperCase() : null;

  return (
    <header className="topbar" ref={rootRef}>
      <span className="topbar-mark">chat.kitsui</span>
      <nav className="topbar-nav" aria-label="the app">
        <span className="topbar-nav-on">Chat</span>
        <a className="topbar-nav-link" href="/">Studio</a>
      </nav>
      <div className="topbar-user-wrap">
        <button type="button" className="topbar-settings" aria-label="Settings" onClick={onOpenSettings}>
          <SettingsIcon size={14} />
        </button>
        <button
          type="button"
          className={open ? "topbar-user open" : "topbar-user"}
          aria-haspopup="menu"
          aria-expanded={open}
          aria-label={
            accountState === "user" && account
              ? `${account.name} — account menu`
              : accountState === "loading"
                ? "Account"
                : "Guest — sign in or create an account"
          }
          disabled={accountState === "loading"}
          onClick={() => setOpen((o) => !o)}
        >
          {accountState === "user" && initials ? (
            <span className="topbar-avatar">{initials}</span>
          ) : (
            <span className="topbar-avatar topbar-avatar-guest">
              <UserIcon size={12} />
            </span>
          )}
          <span className="topbar-user-label">{label}</span>
          <ChevronDownIcon size={12} className={open ? "topbar-chev up" : "topbar-chev"} />
        </button>
        {open && accountState !== "loading" && (
          <div className="topbar-menu" role="menu">
            {accountState === "user" && account ? (
              <>
                <p className="topbar-menu-note">
                  Signed in as <strong>{account.name}</strong>
                  {account.role === "admin" && <span className="topbar-role">Admin</span>}
                </p>
                {confirming && <p className="topbar-menu-confirm">Are you sure?</p>}
                <button
                  type="button"
                  className={confirming ? "btn danger" : "btn ghost"}
                  role="menuitem"
                  onClick={() => {
                    if (!confirming) {
                      setConfirming(true);
                      return;
                    }
                    setOpen(false);
                    setConfirming(false);
                    onSignOut();
                  }}
                >
                  {confirming ? "Yes, I'm sure, let me out." : "Sign out"}
                </button>
              </>
            ) : (
              <>
                <p className="topbar-menu-note">
                  You're browsing as a guest. Create an account to keep signing
                  in as yourself.
                </p>
                <button
                  type="button"
                  className="btn"
                  role="menuitem"
                  onClick={() => {
                    setOpen(false);
                    onSignUp();
                  }}
                >
                  Create an account
                </button>
                <button
                  type="button"
                  className="btn ghost"
                  role="menuitem"
                  onClick={() => {
                    setOpen(false);
                    onSignIn();
                  }}
                >
                  Sign in
                </button>
              </>
            )}
          </div>
        )}
      </div>
    </header>
  );
}
