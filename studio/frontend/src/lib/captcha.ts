/**
 * The Turnstile "I'm not a robot" site key, baked in at build time
 * (VITE_TURNSTILE_SITE_KEY — see frontend/.env.example). The official test
 * key works on any domain and needs no Cloudflare account; a real deployment
 * builds with its own key. Empty = the robot check is off and the widget is
 * not rendered at all.
 */
export const CAPTCHA_SITE_KEY: string = import.meta.env.VITE_TURNSTILE_SITE_KEY ?? "";

/** The subset of the Turnstile browser API the dialog uses (explicit render). */
type TurnstileApi = {
  render: (
    container: Element,
    options: {
      sitekey: string;
      theme?: string;
      /**
       * Fired the moment a token is solved — in the sign-up dialog this
       * *is* the one-click create action, so the dialog submits on it.
       * (Headless browsers are auto-solved on load; real users tick the box.)
       */
      onSuccess?: (token: string) => void;
    },
  ) => string;
  getResponse: (widgetId?: string) => string;
  reset: (widgetId?: string) => string;
  remove: (widgetId?: string) => void;
};

declare global {
  interface Window {
    turnstile?: TurnstileApi;
  }
}

let loader: Promise<TurnstileApi | null> | null = null;

/**
 * Lazily load the Turnstile script (render=explicit, so nothing renders until
 * we call render()). Resolves to the API, or null when no site key is
 * configured or the script can't load (e.g. offline).
 */
export function loadTurnstile(): Promise<TurnstileApi | null> {
  if (!CAPTCHA_SITE_KEY) return Promise.resolve(null);
  if (window.turnstile) return Promise.resolve(window.turnstile);
  if (!loader) {
    loader = new Promise<TurnstileApi | null>((resolve) => {
      const script = document.createElement("script");
      script.src = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
      script.async = true;
      script.onload = () => resolve(window.turnstile ?? null);
      script.onerror = () => resolve(null);
      document.head.appendChild(script);
    });
  }
  return loader;
}
