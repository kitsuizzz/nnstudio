/**
 * Guests keep nothing about providers: a provider set up in Settings lives in
 * the tab's memory for the session only — never in the account database, and
 * never in a cookie — so a reload comes back clean. The only thing here is the
 * purge of the legacy `aistudio.guest` cookie (pre-account-session builds wrote
 * the guest's provider into it; nothing reads it any more).
 */

const GUEST_COOKIE = "aistudio.guest";

export function clearGuestSession(): void {
  try {
    document.cookie = `${GUEST_COOKIE}=; path=/; max-age=0; samesite=lax`;
  } catch {
    // Ignore.
  }
}
