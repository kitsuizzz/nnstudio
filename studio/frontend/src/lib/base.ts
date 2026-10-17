/**
 * The app's mount path under its own origin, without a trailing slash: "" for
 * a root-mounted deployment (docker/nginx), "/dev" when the dev server runs
 * under the /dev/ prefix (Caddy-mounted chat.kitsui.co/dev/*, where every
 * other root path belongs to a different service). Vite bakes the configured
 * `base` into import.meta.env.BASE_URL, so same-origin clients here must
 * address their endpoints through APP_BASE — a bare "/auth/..." would escape
 * the prefix and hit whatever else the domain serves at the root.
 */
export const APP_BASE = import.meta.env.BASE_URL.replace(/\/$/, "");
