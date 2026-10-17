"""Gateway configuration.

Environment variables (or a .env file) drive the gateway:

- UPSTREAM_BASE_URL  OpenAI-compatible base URL of the real provider,
                      including the /v1 segment (e.g. https://api.openai.com/v1).
- UPSTREAM_API_KEY   The real provider key. Kept server-side; clients never see it.
- GATEWAY_TOKEN      Optional shared secret for /v1. Set = every request must
                     present it (Authorization: Bearer <token>); unset = the
                     gateway is open (the app's platform gateway, where the
                     browser calls /v1 directly and the upstream key stays here).
                      When unset, any non-empty bearer token is accepted (local dev).
- CONTEXT_TOKENS     Optional "model:tokens,..." map used to fill in
                      max_context_tokens when the upstream model endpoint omits it
                      (real OpenAI's does).
- GATEWAY_MODELS     Optional "model-a,model-b" allow-list for /v1/models.
                     Empty = expose everything the upstream lists; "none" =
                     expose no models (the app then hides Server-provided).
- CORS_ORIGINS       Comma-separated origins allowed to call the gateway
                     cross-origin. Defaults to chat.kitsui.co plus the local
                     dev origins (so pointing the app straight at :8000 keeps
                     working); empty = wide open (the old "*" behavior).
- PORT               Port the gateway listens on (informational for uvicorn).
- DB_HOST / DB_PORT / DB_USER / DB_PASSWORD / DB_NAME
                     The accounts database (MariaDB). The defaults suit a
                     local `docker compose up -d mariadb` (loopback :3306,
                     user aistudio); the compose stack points DB_HOST at the
                     mariadb service and passes the MARIADB_* values through.
- TURNSTILE_MODE     off | test | live — the "I'm not a robot" check on
                     /auth/signup. "test" is the official Cloudflare test keys
                     (any domain, every token passes — no account needed);
                     "live" adds a hostname match so tokens can't be replayed
                     on other hosts. Off is the default (fully offline).
- TURNSTILE_SECRET_KEY  The Turnstile secret key for the mode above. No secret
                     key means "off" no matter what the mode says.
- SEARCH_PROVIDER    auto | searxng | duckduckgo — the keyless web-search backend
                     used by POST /v1/websearch. "auto" (default) prefers the
                     SearXNG instance when SEARXNG_URL is set, and falls back to
                     the DuckDuckGo HTML endpoint either way.
- SEARXNG_URL        Base URL of a SearXNG instance with the JSON format enabled
                     (the docker stack ships one). Empty = DuckDuckGo only.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Local dev (CWD=backend): this directory's .env, else the repo root's
    # (docker compose values). pydantic-settings lets the LAST listed file win,
    # so the more specific file is last. In containers no .env ships (it is
    # docker-ignored) and environment variables take precedence over files.
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")

    upstream_base_url: str = "http://127.0.0.1:9400/v1"
    upstream_api_key: str = ""
    gateway_token: str | None = None
    context_tokens: str = ""
    gateway_models: str = ""
    cors_origins: str = (
        "https://chat.kitsui.co,http://localhost:1234,http://127.0.0.1:1234"
    )
    port: int = 8000
    db_host: str = "127.0.0.1"
    db_port: int = 3306
    db_user: str = "aistudio"
    db_password: str = "aistudio"
    db_name: str = "aistudio"
    turnstile_mode: str = "off"
    turnstile_secret_key: str = ""
    search_provider: str = "auto"
    searxng_url: str = ""

    @property
    def captcha_enabled(self) -> bool:
        """The sign-up robot check is on when a real mode is set and has a secret."""
        return self.turnstile_mode in ("test", "live") and bool(
            self.turnstile_secret_key
        )

    @property
    def context_token_map(self) -> dict[str, int]:
        """CONTEXT_TOKENS as {model_id: max_context_tokens}."""
        result: dict[str, int] = {}
        for part in self.context_tokens.split(","):
            part = part.strip()
            if not part or ":" not in part:
                continue
            model, _, raw = part.partition(":")
            try:
                result[model.strip()] = int(raw.strip())
            except ValueError:
                continue
        return result

    @property
    def model_allowlist(self) -> list[str] | None:
        """The /v1/models allow-list: None = expose everything the upstream
        lists; an empty list (GATEWAY_MODELS="none") = expose nothing, so the
        app hides the Server-provided category entirely."""
        raw = self.gateway_models.strip()
        if not raw:
            return None
        if raw.lower() in ("none", "-"):
            return []
        return [m.strip() for m in raw.split(",") if m.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS_ORIGINS as a list; empty = wide open (the old "*" behavior)."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
