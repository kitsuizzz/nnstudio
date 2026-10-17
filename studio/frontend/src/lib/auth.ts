import type { AccountUser, GuestSession, SavedProvider } from "../types";
import { APP_BASE } from "./base";
import { credentialFromKey, generateAccessKey, handleFromKey, sealSecret } from "./crypto";

const TOKEN_COOKIE = "aistudio.account";
const COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 365; // one year

/**
 * The signed-in account's opaque bearer token, kept in a cookie (the same
 * pattern as the guest cookie in lib/guest.ts) so a reload restores the sign-in
 * without re-prompting. The token is the only secret stored here — the
 * account record (its handle) is re-fetched from /auth/me on load, and the
 * access key itself is never kept: it is shown once, at creation, and lives
 * only in the user's own password manager / notes.
 */

export function readAuthToken(): string | null {
  try {
    const part = document.cookie.split(/;\s*/).find((c) => c.startsWith(TOKEN_COOKIE + "="));
    if (!part) return null;
    const value = decodeURIComponent(part.slice(TOKEN_COOKIE.length + 1));
    return value || null;
  } catch {
    return null;
  }
}

export function writeAuthToken(token: string): void {
  try {
    document.cookie = `${TOKEN_COOKIE}=${encodeURIComponent(token)}; path=/; max-age=${COOKIE_MAX_AGE_SECONDS}; samesite=lax`;
  } catch {
    // Storage unavailable (private mode, etc.) — the sign-in won't survive a reload.
  }
}

export function clearAuthToken(): void {
  try {
    document.cookie = `${TOKEN_COOKIE}=; path=/; max-age=0; samesite=lax`;
  } catch {
    // Ignore.
  }
}

interface AuthResponse {
  /** The access key — sign-up only, and the only place it is ever returned.
      The browser minted it, so the server never had it to return. */
  key?: string;
  token?: string;
  user?: AccountUser;
  /** The account's saved BYOK providers (if any); /auth/signin returns them. */
  providers?: SavedProvider[];
  /** The account's first saved provider (legacy shape). */
  provider?: SavedProvider;
  /** The account's admin key, sealed — ciphertext only its owner can open. */
  adminKey?: string | null;
  /** The account's sealed agents (ciphertext only its owner can open). */
  agents?: string | null;
}

/** What /auth/me resolves a stored token to, or null when it is unknown. */
export interface MeResponse {
  user: AccountUser;
  /** The account's saved BYOK providers (empty until one has been stored). */
  providers: SavedProvider[];
  /** The account's admin key (sealed client-side), if one was stored. */
  adminKey: string | null;
  /** The account's sealed agents (ciphertext only its owner can open), if any. */
  agents: string | null;
}

/** The /auth endpoints are same-origin (Vite proxy in dev, nginx in docker);
 *  APP_BASE keeps them inside the mount prefix when the app runs under one. */
async function authPost(path: "signup" | "signin", body: Record<string, string>, token?: string): Promise<AuthResponse> {
  const res = await fetch(`${APP_BASE}/auth/${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `Request failed with status ${res.status}`;
    try {
      const data: { detail?: string } = await res.json();
      if (data.detail) detail = data.detail;
    } catch {
      // non-JSON error body
    }
    throw new Error(detail);
  }
  return (await res.json()) as AuthResponse;
}

/** Normalize a profile payload: the provider list, with the legacy
    single-provider shape folded in. */
export function providersFrom(data: { provider?: SavedProvider | null; providers?: SavedProvider[] | null }): SavedProvider[] {
  const list = Array.isArray(data.providers) ? data.providers.filter((p) => p && typeof p.baseUrl === "string") : [];
  if (list.length) return list;
  return data.provider ? [data.provider] : [];
}

/** The identity + saved providers behind the stored token, or null when unknown. */
export async function fetchMe(token: string, signal?: AbortSignal): Promise<MeResponse | null> {
  // Network failures and timeouts PROPAGATE (the caller decides whether the
  // session is dead or merely unreachable); a non-OK response means the
  // server does not recognize this session.
  const res = await fetch(`${APP_BASE}/auth/me`, {
    headers: { Authorization: `Bearer ${token}` },
    signal,
  });
  if (!res.ok) return null;
  let data: {
    user?: AccountUser;
    provider?: SavedProvider | null;
    providers?: SavedProvider[] | null;
    adminKey?: string | null;
    agents?: string | null;
  };
  try {
    data = await res.json();
  } catch {
    return null;
  }
  if (!data.user) return null;
  return { user: data.user, providers: providersFrom(data), adminKey: data.adminKey ?? null, agents: data.agents ?? null };
}

export async function signup(captcha?: string): Promise<AuthResponse> {
  // The browser generates the access key, so the server never has it: the
  // sign-up body carries only SHA-256(key) (the credential) and the display
  // handle. The key is attached to the response for the one-time reveal, and
  // it is the only copy anywhere.
  const key = generateAccessKey();
  const credential = await credentialFromKey(key);
  const body: Record<string, string> = { credential, handle: handleFromKey(key) };
  if (captcha) body.captcha = captcha;
  const res = await authPost("signup", body);
  return { ...res, key };
}

export async function signin(key: string): Promise<AuthResponse> {
  // The access key is the only credential; only its hash ever leaves the
  // browser, so the server can authenticate but never decrypt.
  const credential = await credentialFromKey(key);
  return authPost("signin", { credential });
}

/** Best-effort: invalidate the token server-side even if the request fails. */
export async function signout(token: string): Promise<void> {
  try {
    await fetch(`${APP_BASE}/auth/signout`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
    });
  } catch {
    // The cookie is cleared locally either way.
  }
}

/**
 * Persist (or, when empty, clear) the signed-in account's BYOK providers.
 * Best-effort: a failed save just means the next login re-onboards, so it
 * never blocks the UI.
 *
 * Each apiKey is sealed under the account key before it leaves the browser,
 * so the server stores opaque values; name/baseUrl/model stay readable, which
 * is what lets a device (and the server's own view) know the providers in use.
 */
export async function saveProviders(token: string, providers: SavedProvider[], key: CryptoKey | null): Promise<void> {
  try {
    const stored = await Promise.all(
      providers.map(async (p) => ({ ...p, apiKey: await sealSecret(key, p.apiKey) }))
    );
    await fetch(`${APP_BASE}/auth/profile`, {
      method: "PUT",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({ providers: stored }),
    });
  } catch {
    // Ignore — see above.
  }
}

/**
 * Ask the server for an "are you admin?" challenge (single-use, short-lived).
 * Nothing about the account's user type was readable before this; answering
 * the challenge with an admin key is the only way the server ever learns it.
 */
export async function fetchAdminChallenge(token: string): Promise<string | null> {
  try {
    const res = await fetch(`${APP_BASE}/auth/admin/challenge`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) return null;
    const data: { challenge?: string } = await res.json();
    return data.challenge || null;
  } catch {
    return null;
  }
}

/** Answer a challenge with the admin key; true when the server accepts it. */
export async function proveAdmin(token: string, challenge: string, key: string): Promise<boolean> {
  try {
    const res = await fetch(`${APP_BASE}/auth/admin/prove`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({ challenge, key }),
    });
    return res.ok;
  } catch {
    return false;
  }
}

/** Store (or clear) the account's sealed admin-key copy — server-side it is
    opaque ciphertext, kept so admin can be proven from another device. */
export async function saveAdminKey(token: string, sealed: string | null): Promise<void> {
  try {
    await fetch(`${APP_BASE}/auth/admin/key`, {
      method: "PUT",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({ key: sealed }),
    });
  } catch {
    // Best-effort: proving still works with the in-memory key.
  }
}

/** Store the account's sealed agents in its profile; the providers are left as they are. */
export async function saveAgentsRecord(token: string, sealed: string | null): Promise<void> {
  try {
    await fetch(`${APP_BASE}/auth/profile`, {
      method: "PUT",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({ agents: sealed }),
    });
  } catch {
    // Best-effort, like the provider save.
  }
}

/** A saved provider as the guest-shaped session the greeting brain expects. */
export function providerToGuest(provider: SavedProvider): GuestSession {
  return {
    guest: true,
    providerName: provider.name,
    baseUrl: provider.baseUrl,
    apiKey: provider.apiKey,
    noKey: provider.noKey === true ? true : undefined,
    model: provider.model,
    at: 0,
  };
}

/** A guest-shaped session as the saved-provider shape the profile stores. */
export function guestToProvider(session: GuestSession): SavedProvider {
  return {
    name: session.providerName,
    baseUrl: session.baseUrl,
    apiKey: session.apiKey,
    model: session.model,
    noKey: session.noKey === true ? true : undefined,
  };
}
