import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// Same-origin API: the app talks to its own origin (/v1/... for the
// OpenAI-compatible pass-through, /auth/... for user accounts) and the dev
// server forwards both to the Python gateway. In docker, nginx does the same.
//
// The dev server is also mounted under the /dev/ prefix by the Caddy instance
// in front of chat.kitsui.co (every other root path on that domain belongs to
// a different service). Under the prefix the browser addresses the same
// endpoints as /dev/v1 and /dev/auth. Note the targets stay bare: http-proxy
// *prepends* a target's path to the request path, so a /v1 in the target would
// make the gateway see /v1/dev/v1/... — the rewrite strips /dev instead. The
// root-level entries stay for direct access (a plain root vs the /dev/ mount).
const apiProxy = {
  "/v1": "http://127.0.0.1:8000",
  "/auth": "http://127.0.0.1:8000",
  "/dev/v1": {
    target: "http://127.0.0.1:8000",
    changeOrigin: true,
    rewrite: (path) => path.replace(/^\/dev/, ""),
  },
  "/dev/auth": {
    target: "http://127.0.0.1:8000",
    changeOrigin: true,
    rewrite: (path) => path.replace(/^\/dev/, ""),
  },
};

/** The build's mount path: "/" (the app on its own origin — the docker stack)
    or e.g. "/dev/" when a reverse proxy mounts the app under a prefix. */
function mountPath(raw: string | undefined): string {
  const bare = (raw ?? "").trim().replace(/^\/+|\/+$/g, "");
  return bare ? `/${bare}/` : "/";
}

export default defineConfig(({ command, mode }) => ({
  plugins: [react()],
  // Served under the /dev/ prefix when the dev server (and preview) runs, so
  // every URL the browser requests — page, modules, HMR websocket, and the
  // app's own /dev/v1 + /dev/auth calls — stays inside the prefix Caddy
  // routes to this server. Builds take their mount from VITE_BASE_PATH: "/"
  // by default (the docker image serves the app at / on its own origin), or
  // "/dev/" for the chat.kitsui.co/dev mount. (vite preview inherits the dev prefix.)
  base:
    command === "build"
      ? mountPath(loadEnv(mode, ".", "VITE_").VITE_BASE_PATH)
      : "/dev/",
  server: {
    host: true,
    port: 1234,
    strictPort: true,
    // Vite blocks any Host header other than localhost (DNS-rebinding guard);
    // the dev server is also reachable through the chat app's domain.
    allowedHosts: ["chat.kitsui.co"],
    watch: {
      ignored: ["**/tmp/**"],
    },
    proxy: apiProxy,
  },
  preview: {
    host: true,
    port: 1234,
    strictPort: true,
    allowedHosts: ["chat.kitsui.co"],
    proxy: apiProxy,
  },
}));
