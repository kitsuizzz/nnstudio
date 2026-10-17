import { commit, KEYS_STORE, openDb, requestToPromise } from "./idb";

/**
 * The per-account AES-GCM key cache. A non-extractable CryptoKey survives in
 * IndexedDB across reloads, so a signed-in user decrypts their server-side
 * store without re-entering the access key. Keyed by account id, so switching
 * accounts on one browser never hands over another account's key.
 *
 * The key can't be exported, only used — and it is dropped on sign-out.
 */

export async function cacheKey(userId: number, key: CryptoKey): Promise<void> {
  try {
    const db = await openDb();
    try {
      const tx = db.transaction(KEYS_STORE, "readwrite");
      tx.objectStore(KEYS_STORE).put(key, userId);
      await commit(tx);
    } finally {
      db.close();
    }
  } catch {
    // Storage unavailable (private mode): the unlock prompt covers it.
  }
}

export async function getCachedKey(userId: number): Promise<CryptoKey | null> {
  try {
    const db = await openDb();
    try {
      const tx = db.transaction(KEYS_STORE, "readonly");
      const raw = await requestToPromise(tx.objectStore(KEYS_STORE).get(userId));
      return raw instanceof CryptoKey ? raw : null;
    } finally {
      db.close();
    }
  } catch {
    return null;
  }
}

export async function dropCachedKey(userId: number): Promise<void> {
  try {
    const db = await openDb();
    try {
      const tx = db.transaction(KEYS_STORE, "readwrite");
      tx.objectStore(KEYS_STORE).delete(userId);
      await commit(tx).catch(() => undefined);
    } finally {
      db.close();
    }
  } catch {
    // Ignore.
  }
}
