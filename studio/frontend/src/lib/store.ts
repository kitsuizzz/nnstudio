import type { AppOverrides, Conversation, Studio } from "../types";
import { commit, KV_STORE, openDb, requestToPromise } from "./idb";

const KEY = "conversations";
const STUDIO_KEY = "studios";
const OVERRIDES_KEY = "app-overrides";

function isConversation(x: unknown): x is Conversation {
  if (!x || typeof x !== "object") return false;
  const c = x as Partial<Conversation>;
  return typeof c.id === "number" && typeof c.title === "string" && Array.isArray(c.messages);
}

/**
 * Loads persisted conversations. Resolves to `null` when storage is
 * unavailable (private mode, blocked, corrupted) so the caller can fall back
 * to in-memory state without surfacing an error.
 */
export async function loadConversations(): Promise<Conversation[] | null> {
  try {
    const db = await openDb();
    try {
      const tx = db.transaction(KV_STORE, "readonly");
      const raw = await requestToPromise(tx.objectStore(KV_STORE).get(KEY));
      if (!Array.isArray(raw)) return [];
      return raw.filter(isConversation).map((c) => ({
        ...c,
        inputTokens: typeof c.inputTokens === "number" ? c.inputTokens : 0,
        outputTokens: typeof c.outputTokens === "number" ? c.outputTokens : 0,
      }));
    } finally {
      db.close();
    }
  } catch {
    return null;
  }
}

/** Persists the full conversation list. Failures are swallowed by design. */
export async function saveConversations(conversations: Conversation[]): Promise<void> {
  try {
    const db = await openDb();
    try {
      const tx = db.transaction(KV_STORE, "readwrite");
      tx.objectStore(KV_STORE).put(conversations, KEY);
      await commit(tx);
    } finally {
      db.close();
    }
  } catch {
    // storage unavailable or quota exceeded; state stays in memory
  }
}

function isStudio(x: unknown): x is Studio {
  if (!x || typeof x !== "object") return false;
  const s = x as Partial<Studio>;
  return (
    typeof s.id === "string" &&
    typeof s.title === "string" &&
    !!s.spec &&
    typeof s.spec === "object" &&
    typeof s.conversationId === "number"
  );
}

/** Loads persisted studios (AI-built workspaces). `null` when unavailable. */
export async function loadStudios(): Promise<Studio[] | null> {
  try {
    const db = await openDb();
    try {
      const tx = db.transaction(KV_STORE, "readonly");
      const raw = await requestToPromise(tx.objectStore(KV_STORE).get(STUDIO_KEY));
      if (!Array.isArray(raw)) return [];
      return raw.filter(isStudio);
    } finally {
      db.close();
    }
  } catch {
    return null;
  }
}

/** Persists the full studio list. Failures are swallowed by design. */
export async function saveStudios(studios: Studio[]): Promise<void> {
  try {
    const db = await openDb();
    try {
      const tx = db.transaction(KV_STORE, "readwrite");
      tx.objectStore(KV_STORE).put(studios, STUDIO_KEY);
      await commit(tx);
    } finally {
      db.close();
    }
  } catch {
    // storage unavailable or quota exceeded; state stays in memory
  }
}

function isOverrides(x: unknown): x is AppOverrides {
  if (!x || typeof x !== "object" || Array.isArray(x)) return false;
  for (const [k, v] of Object.entries(x)) {
    if (typeof k !== "string" || typeof v !== "string" || !v.length) return false;
  }
  return true;
}

/** Loads persisted per-app code overrides. `null` when storage is unavailable. */
export async function loadAppOverrides(): Promise<AppOverrides | null> {
  try {
    const db = await openDb();
    try {
      const tx = db.transaction(KV_STORE, "readonly");
      const raw = await requestToPromise(tx.objectStore(KV_STORE).get(OVERRIDES_KEY));
      if (raw == null) return {};
      return isOverrides(raw) ? raw : {};
    } finally {
      db.close();
    }
  } catch {
    return null;
  }
}

/**
 * Persists the full override map. Unlike the studio list, an empty map is
 * written too — dropping the last override must survive a reload.
 */
export async function saveAppOverrides(overrides: AppOverrides): Promise<void> {
  try {
    const db = await openDb();
    try {
      const tx = db.transaction(KV_STORE, "readwrite");
      tx.objectStore(KV_STORE).put(overrides, OVERRIDES_KEY);
      await commit(tx);
    } finally {
      db.close();
    }
  } catch {
    // storage unavailable or quota exceeded; state stays in memory
  }
}

/**
 * Drops the per-browser conversation/studio copies (overrides stay — they are
 * not part of the E2E store). Called once the first sign-in has encrypted and
 * uploaded them, so a later guest session can't resurrect another account's
 * data. Failures are swallowed: the data is already safe server-side.
 */
export async function clearLocalData(): Promise<void> {
  try {
    const db = await openDb();
    try {
      const tx = db.transaction(KV_STORE, "readwrite");
      const store = tx.objectStore(KV_STORE);
      store.delete(KEY);
      store.delete(STUDIO_KEY);
      await commit(tx);
    } finally {
      db.close();
    }
  } catch {
    // ignore
  }
}
