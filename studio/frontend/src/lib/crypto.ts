/**
 * End-to-end encryption for the server-side store.
 *
 * The access key (``ast_`` + 256 random bits) is the only secret: the browser
 * derives an AES-GCM key from it and encrypts every record before upload. The
 * server receives ciphertext and, for authentication, only SHA-256(key) — so
 * it can neither read the data nor derive the key.
 *
 * The derived key is non-extractable and cached in IndexedDB (lib/keyvault.ts),
 * so reloads decrypt silently while the raw bytes never leave the browser.
 */

const KEY_PREFIX = "ast_";
// Fixed KDF domain separation: the same access key always yields the same
// AES key on every device, which is what makes cross-device sync work.
const HKDF_SALT = "aistudio:e2e:v1";
const HKDF_INFO = "conversations";
const IV_BYTES = 12;

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function bytesToBase64Url(bytes: Uint8Array): string {
  return bytesToBase64(bytes).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function base64ToBytes(value: string): Uint8Array<ArrayBuffer> {
  const binary = atob(value);
  const out = new Uint8Array(new ArrayBuffer(binary.length));
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** A fresh access key, exactly as the one-time reveal shows it. */
export function generateAccessKey(): string {
  const raw = new Uint8Array(32);
  crypto.getRandomValues(raw);
  return KEY_PREFIX + bytesToBase64Url(raw);
}

/** The display handle: the key's own opening (non-secret, shown in the top bar). */
export function handleFromKey(key: string): string {
  return key.slice(KEY_PREFIX.length, KEY_PREFIX.length + 6);
}

/**
 * SHA-256(key), hex — the credential the server sees. It authenticates the
 * account without revealing the key, so it can't be turned into the AES key.
 */
export async function credentialFromKey(key: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(key));
  return bytesToHex(new Uint8Array(digest));
}

/**
 * The record-encryption key for an access key. HKDF is deterministic, so every
 * device that holds the key derives the same AES-GCM key; it is non-extractable,
 * so nothing can read it back out of the browser.
 */
export async function deriveEncKey(key: string): Promise<CryptoKey> {
  const material = await crypto.subtle.importKey("raw", new TextEncoder().encode(key), "HKDF", false, [
    "deriveKey",
  ]);
  return crypto.subtle.deriveKey(
    {
      name: "HKDF",
      hash: "SHA-256",
      salt: new TextEncoder().encode(HKDF_SALT),
      info: new TextEncoder().encode(HKDF_INFO),
    },
    material,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"],
  );
}

/** JSON → base64(iv ‖ AES-GCM ciphertext). A fresh random IV per record. */
export async function encryptJson(key: CryptoKey, value: unknown): Promise<string> {
  const iv = new Uint8Array(IV_BYTES);
  crypto.getRandomValues(iv);
  const ciphertext = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv },
    key,
    new TextEncoder().encode(JSON.stringify(value)),
  );
  const out = new Uint8Array(iv.length + ciphertext.byteLength);
  out.set(iv, 0);
  out.set(new Uint8Array(ciphertext), iv.length);
  return bytesToBase64(out);
}

/** The inverse of encryptJson; throws on a wrong key or tampered ciphertext. */
export async function decryptJson<T>(key: CryptoKey, blob: string): Promise<T> {
  const raw = base64ToBytes(blob);
  if (raw.length <= IV_BYTES) throw new Error("malformed record");
  const iv = raw.subarray(0, IV_BYTES);
  const ciphertext = raw.subarray(IV_BYTES);
  const plaintext = await crypto.subtle.decrypt({ name: "AES-GCM", iv }, key, ciphertext);
  return JSON.parse(new TextDecoder().decode(plaintext)) as T;
}

/** Marks a sealed single secret (e.g. the provider apiKey) so the client can
    tell it from a legacy plaintext value and never seals it twice. */
const SECRET_PREFIX = "e2e1:";

/**
 * Seal one secret string under the account key. The server stores the
 * marker-prefixed base64 blob as an opaque value; only a browser holding the
 * access key can open it. An already-sealed value passes through untouched,
 * so a re-save can never double-encrypt.
 */
export async function sealSecret(key: CryptoKey | null, value: string): Promise<string> {
  if (!key || value.startsWith(SECRET_PREFIX)) return value;
  const iv = new Uint8Array(IV_BYTES);
  crypto.getRandomValues(iv);
  const ciphertext = await crypto.subtle.encrypt({ name: "AES-GCM", iv }, key, new TextEncoder().encode(value));
  const out = new Uint8Array(iv.length + ciphertext.byteLength);
  out.set(iv, 0);
  out.set(new Uint8Array(ciphertext), iv.length);
  return SECRET_PREFIX + bytesToBase64(out);
}

/**
 * Open a sealed secret. Unmarked values are legacy plaintext and pass through;
 * so does a sealed value when the key is unavailable or wrong (the caller
 * then simply has an unusable secret rather than a crash).
 */
export async function openSecret(key: CryptoKey | null, value: string): Promise<string> {
  if (!key || !value.startsWith(SECRET_PREFIX)) return value;
  try {
    const raw = base64ToBytes(value.slice(SECRET_PREFIX.length));
    if (raw.length <= IV_BYTES) return value;
    const iv = raw.subarray(0, IV_BYTES);
    const ciphertext = raw.subarray(IV_BYTES);
    const plaintext = await crypto.subtle.decrypt({ name: "AES-GCM", iv }, key, ciphertext);
    return new TextDecoder().decode(plaintext);
  } catch {
    return value;
  }
}
