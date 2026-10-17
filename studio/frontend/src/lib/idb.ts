/**
 * The one IndexedDB the app opens. Two object stores share it:
 *
 * - `kv`    — the guest's conversations / studios / code overrides (and the
 *             signed-in user's, until the first cloud sync claims them).
 * - `keys`  — the account's non-extractable AES-GCM key (see lib/keyvault.ts),
 *             so a reload can decrypt the server-side store without asking
 *             for the access key again.
 *
 * Kept in one module so every opener agrees on the version and the upgrade
 * creates both stores, whoever opens first.
 */

const DB_NAME = "aistudio";
const DB_VERSION = 2;

export const KV_STORE = "kv";
export const KEYS_STORE = "keys";

export function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") {
      reject(new Error("IndexedDB unavailable"));
      return;
    }
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(KV_STORE)) db.createObjectStore(KV_STORE);
      if (!db.objectStoreNames.contains(KEYS_STORE)) db.createObjectStore(KEYS_STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error("IndexedDB open failed"));
    req.onblocked = () => reject(new Error("IndexedDB blocked"));
  });
}

export function requestToPromise<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error("IndexedDB request failed"));
  });
}

/** Run one readwrite transaction to completion (resolves on commit). */
export function commit(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error("IndexedDB write failed"));
    tx.onabort = () => reject(tx.error ?? new Error("IndexedDB write aborted"));
  });
}
