/**
 * The encrypted store's HTTP client. Everything here is ciphertext: the
 * gateway sees opaque blobs, never plaintext and never the key.
 */
import { APP_BASE } from "./base";

export interface StoreRecord {
  id: string;
  blob: string;
  updatedAt?: string;
}

export interface StoreSnapshot {
  conversations: StoreRecord[];
  studios: StoreRecord[];
  /** Where the reader left each conversation: one record per conversation. */
  views: StoreRecord[];
  /** The conversation to reopen next visit: one record (`current`). */
  last: StoreRecord[];
}

export interface StoreDelta {
  upserts: { id: string; blob: string }[];
  deletes: string[];
}

export interface StoreSyncBody {
  conversations: StoreDelta;
  studios: StoreDelta;
  views?: StoreDelta;
  last?: StoreDelta;
}

/** Every encrypted record the signed-in account owns, or null on failure. */
export async function fetchStore(token: string): Promise<StoreSnapshot | null> {
  try {
    const res = await fetch(`${APP_BASE}/auth/store`, { headers: { Authorization: `Bearer ${token}` } });
    if (!res.ok) return null;
    const data: { conversations?: StoreRecord[]; studios?: StoreRecord[]; views?: StoreRecord[]; last?: StoreRecord[] } = await res.json();
    return {
      conversations: Array.isArray(data.conversations) ? data.conversations : [],
      studios: Array.isArray(data.studios) ? data.studios : [],
      views: Array.isArray(data.views) ? data.views : [],
      last: Array.isArray(data.last) ? data.last : [],
    };
  } catch {
    return null;
  }
}

/** Apply a diff (upserts + deletes). Best-effort: a failure just retries on the
 *  next debounced save, since the client keeps the plaintext until it lands.
 *  `keepalive` lets the request outlive the page (used as the page is hidden). */
export async function syncStore(token: string, body: StoreSyncBody, keepalive = false): Promise<boolean> {
  try {
    const res = await fetch(`${APP_BASE}/auth/store/sync`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify(body),
      keepalive,
    });
    return res.ok;
  } catch {
    return false;
  }
}
