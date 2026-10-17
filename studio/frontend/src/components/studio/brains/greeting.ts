/**
 * The welcome screen, re-unified as a sandbox "logic" brain.
 *
 * Unlike the wall brain (which projects live conversation data pushed by the
 * host), the greeting brain is self-driven: it owns its own onboarding state
 * machine (choose → provider), runs the title typewriter, polls the
 * provider's /models endpoint, and gates "continue" on a live provider test.
 * It talks to the host only through the bridge: `ai.render` (blocks), `ai.on`
 * (user events), `ai.fetch` (models/test) and `ai.host` (start / new-session /
 * auth).
 *
 * The greeting only fronts a FIRST visit and the provider fallback: once a
 * provider exists (the host's platform gateway, a saved BYOK session, or a
 * just-finished setup) the host sends the user straight to a new session, so
 * the ready screen is gone. In platform mode "Continue as guest" emits
 * `ai.host("new-session")` — the host already owns the provider; otherwise
 * the provider form is walked and a successful test starts the session and
 * asks for the new-session route too. A 600 ms timer boots the normal flow if
 * the host never answers; a signed-in user with no platform gateway resumes
 * at the provider step.
 *
 * Account creation is the host's job, not the brain's: "Create an account"
 * (and the provider step's "Sign up") call `ai.host("auth", "signup")` to open
 * the host's one-click sign-up dialog. When the host confirms an account now
 * exists it pushes `signedIn` through ctx, and the brain advances the choice
 * step straight to the provider step so onboarding continues.
 *
 * The source is one flat string with no `${` and no backticks, so it can be
 * embedded verbatim in the iframe's <script> and shown in the Code pane.
 */
export const greetingSource = `
var TITLE = "One place for every task";
var SUBTITLE = "Take the most advantage of generative AI";
// The speed sits just above the host's 33ms render throttle, so no frame is
// dropped and the animation runs at the intended pace.
var TYPE_SPEED_MS = 45;

var PROVIDERS = [
  { name: "OpenAI", baseUrl: "https://api.openai.com/v1", models: ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"] },
  { name: "Anthropic", baseUrl: "https://api.anthropic.com/v1", models: ["claude-sonnet-4-5", "claude-opus-4-1", "claude-haiku-4-5"] },
  { name: "Google", baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai", models: ["gemini-2.5-pro", "gemini-2.5-flash"] },
  { name: "OpenRouter", baseUrl: "https://openrouter.ai/api/v1", models: ["anthropic/claude-sonnet-4.5", "openai/gpt-4o"] },
  { name: "Mistral", baseUrl: "https://api.mistral.ai/v1", models: ["mistral-large-latest", "mistral-small-latest"] },
  { name: "Groq", baseUrl: "https://api.groq.com/openai/v1", models: ["llama-3.3-70b-versatile"] },
  { name: "Custom", baseUrl: "", models: [] }
];

// What the AI reaches for first, per provider, when it picks its own model.
var MODEL_PREFERENCE = {
  OpenAI: ["gpt-4o", "o4-mini", "o4", "o3"],
  Anthropic: ["claude-sonnet", "claude-opus"],
  Google: ["gemini-2.5-pro", "gemini"],
  OpenRouter: ["claude-sonnet", "gpt-4o"],
  Mistral: ["mistral-large", "mistral-small"],
  Groq: ["llama-3.3-70b", "llama"]
};

var providerNames = [];
for (var pi = 0; pi < PROVIDERS.length; pi++) providerNames.push(PROVIDERS[pi].name);

// --- onboarding state ---------------------------------------------------------
var step = "choice";
var providerName = PROVIDERS[0].name;
var baseUrl = PROVIDERS[0].baseUrl;
var apiKey = "";
var remoteModels = [];
var modelsLoading = false;
var checking = false;
var providerError = null;
var chosenModel = "";
var typed = "";
var typeTimer = null;
var pollTimer = null;
var booted = false;
var bootTimer = null;
// The host's platform offer (the deployment's server-side model), when any.
var platformCtx = null;
// The host's verdict: a usable provider already exists, so "Continue as guest"
// can start a session without the provider form.
var canStart = false;

var reducedMotion = (function () {
  try {
    return typeof window !== "undefined" &&
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch (e) { return false; }
})();

function findProvider(name) {
  for (var i = 0; i < PROVIDERS.length; i++) if (PROVIDERS[i].name === name) return PROVIDERS[i];
  return PROVIDERS[0];
}
function pickModel(list, provName) {
  var pref = MODEL_PREFERENCE[provName] || [];
  for (var i = 0; i < pref.length; i++) {
    var frag = pref[i];
    for (var j = 0; j < list.length; j++) {
      if (list[j].toLowerCase().indexOf(frag.toLowerCase()) !== -1) return list[j];
    }
  }
  return list[0] || "";
}
function providerReady() { return baseUrl.trim() !== "" && apiKey.trim() !== ""; }
function preset() { return findProvider(providerName); }
function stopTitleTimers() {
  if (typeTimer) { clearInterval(typeTimer); typeTimer = null; }
}

// --- block builders -----------------------------------------------------------
function btn(label, to, ghost, disabled) {
  var b = { type: "button", label: label, to: to };
  if (ghost) b.variant = "ghost";
  if (disabled) b.disabled = true;
  return b;
}
function field(key, label, placeholder, kind, value, autoFocus, className, ariaLabel) {
  var b = { type: "field", key: key, label: label, placeholder: placeholder, value: value };
  if (kind && kind !== "text") b.kind = kind;
  if (autoFocus) b.autoFocus = true;
  if (className) b.className = className;
  b.ariaLabel = ariaLabel || label;
  return b;
}
function sel(label, options, value, ariaLabel, chip) {
  var b = { type: "select", options: options, value: value };
  if (label) b.label = label;
  if (chip) b.chip = chip;
  b.ariaLabel = ariaLabel || label || chip;
  return b;
}
function mkGroup(className, key, blocks) {
  var b = { type: "group", className: className, blocks: blocks };
  if (key) b.key = key;
  return b;
}

// --- render -------------------------------------------------------------------
// The head is emitted separately so the frame can keep it mounted (the
// typewriter keeps running) while the body group remounts on each step change.
function render() {
  var head = { type: "greetinghead", title: typed, sub: SUBTITLE };
  var body;
  if (step === "choice") {
    body = mkGroup("greeting-prompt", "choice", [
      mkGroup("greeting-actions", null, [
        btn("Create an account", "signup", false, false),
        btn("Continue as guest", "provider", true, false)
      ])
    ]);
  } else if (step === "provider") {
    var blocks = [
      sel("LLM Provider", providerNames, providerName, "LLM Provider", null),
      field("baseUrl", "Base URL", "https://api.example.com/v1", "url", baseUrl, false),
      field("apiKey", "API Key", "sk-…", "password", apiKey, true)
    ];
    if (modelsLoading) blocks.push({ type: "note", tone: "hint", content: "Loading models…" });
    if (providerError) blocks.push({ type: "note", tone: "error", content: providerError });
    blocks.push(mkGroup("greeting-actions", null, [
      btn("Sign up", "signup", true, false),
      btn(checking ? "Checking…" : "Continue", "continue-guest", false, !providerReady() || checking)
    ]));
    body = mkGroup("greeting-form", "provider", blocks);
  }
  ai.render({ blocks: [head, body] });
}

// --- state transitions ---------------------------------------------------------
function chooseProvider(name) {
  var next = findProvider(name);
  providerName = next.name;
  baseUrl = next.baseUrl;
  providerError = null;
  if (step === "provider") schedulePoll();
  render();
}
function pickReadyModel() {
  if (remoteModels.length > 0) {
    if (!(chosenModel && remoteModels.indexOf(chosenModel) !== -1)) chosenModel = pickModel(remoteModels, providerName);
  } else if (!chosenModel) {
    chosenModel = preset().models[0] || "";
  }
}
// Gate the entrance on a live provider: /models must accept the key first.
function continueAsGuest() {
  if (!providerReady() || checking) return;
  checking = true;
  providerError = null;
  render();
  ai.fetch("test", [baseUrl.trim(), apiKey.trim()]).then(function (result) {
    checking = false;
    if (!result || !result.ok) {
      providerError = (result && result.message) || "The provider rejected the connection.";
      render();
      return;
    }
    // The provider is live and a model is picked: start the session and hand
    // the user straight to a fresh conversation — no intermediary screen.
    pickReadyModel();
    ai.host("start", { providerName: providerName, baseUrl: baseUrl.trim(), apiKey: apiKey.trim(), model: chosenModel });
    ai.host("new-session");
  }, function () {
    checking = false;
    providerError = "Could not reach the provider.";
    render();
  });
}

// --- model polling (provider step) --------------------------------------------
// Debounced, and a stale response is discarded if the user edited the
// credentials or left the step while the request was in flight.
function schedulePoll() {
  if (step !== "provider") return;
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
  var url = baseUrl.trim(), key = apiKey.trim();
  if (!url || !key) {
    if (remoteModels.length) { remoteModels = []; render(); }
    return;
  }
  pollTimer = setTimeout(function () {
    var reqUrl = url, reqKey = key;
    remoteModels = [];
    modelsLoading = true;
    render();
    ai.fetch("models", [reqUrl, reqKey]).then(function (models) {
      if (step !== "provider" || baseUrl.trim() !== reqUrl || apiKey.trim() !== reqKey) return;
      remoteModels = models || [];
      modelsLoading = false;
      render();
    }, function () {
      if (step !== "provider" || baseUrl.trim() !== reqUrl || apiKey.trim() !== reqKey) return;
      modelsLoading = false;
      render();
    });
  }, 400);
}

// --- title typewriter ----------------------------------------------------------
function typeOut(text, speed, onDone) {
  if (reducedMotion) { typed = text; render(); if (onDone) onDone(); return; }
  stopTitleTimers();
  typed = "";
  render();
  var i = 0;
  typeTimer = setInterval(function () {
    i += 1;
    typed = text.slice(0, i);
    render();
    if (i >= text.length) { clearInterval(typeTimer); typeTimer = null; if (onDone) onDone(); }
  }, speed);
}
// --- event routing -------------------------------------------------------------
ai.on("click", function (arg, blockType) {
  if (blockType === "button") {
    // Both account buttons delegate to the host's one-click sign-up dialog; the
    // brain stays put and the host advances it (choice -> provider) on confirm.
    if (arg === "Create an account" || arg === "Sign up") { ai.host("auth", "signup"); }
    else if (arg === "Continue as guest") {
      // With a usable provider (platform or a saved BYOK setup) the host hands
      // the guest straight to a new session; otherwise the BYOK form is the
      // fallback.
      if (canStart) ai.host("new-session");
      else { step = "provider"; schedulePoll(); render(); }
    }
    else if (arg === "Continue" || arg === "Checking…") { continueAsGuest(); }
  }
});
ai.on("choose", function (arg, blockType) {
  if (blockType !== "select") return;
  if (step === "provider") chooseProvider(arg);
});
ai.on("change", function (arg, blockType) {
  if (blockType !== "field" || !arg || typeof arg !== "object") return;
  var key = arg.key, value = arg.value || "";
  if (key === "baseUrl") { baseUrl = value; if (step === "provider") { providerError = null; schedulePoll(); } render(); }
  else if (key === "apiKey") { apiKey = value; if (step === "provider") { providerError = null; schedulePoll(); } render(); }
});

function boot() {
  if (booted) return;
  booted = true;
  if (bootTimer) { clearTimeout(bootTimer); bootTimer = null; }
  if (reducedMotion) { typed = TITLE; render(); }
  else typeOut(TITLE, TYPE_SPEED_MS);
}
ai.on("ctx", function (data) {
  // The host's verdict on whether a usable provider exists, and a saved
  // guest setup to prefill the fallback form with (the landing page is always
  // shown to guests, so this never auto-advances by itself).
  if (data && typeof data.canStart === "boolean") canStart = data.canStart;
  if (data && data.guest && data.guest.baseUrl && data.guest.apiKey) {
    providerName = data.guest.providerName || PROVIDERS[0].name;
    baseUrl = data.guest.baseUrl;
    apiKey = data.guest.apiKey;
    if (data.guest.model) chosenModel = data.guest.model;
  }
  // Platform mode (the deployment's server-side model): remember the offer so
  // "Continue as guest" knows a provider exists. The host routes signed-in
  // users itself; guests always start here.
  if (data && data.platform && data.platform.baseUrl) {
    platformCtx = data.platform;
    if (!booted) boot();
    return;
  }
  if (booted) return;
  if (data && data.signedIn) {
    // Signed in but not mirrored by the host (its platform probe failed):
    // fall back to the provider form, typewriter skipped.
    booted = true;
    if (bootTimer) { clearTimeout(bootTimer); bootTimer = null; }
    typed = TITLE;
    step = "provider";
    render();
    schedulePoll();
    return;
  }
  boot();
});
// A second, non-boot-gated ctx handler: remember the platform offer whenever
// it arrives, and advance a signed-in user to the provider fallback if they
// are still on the choice step.
ai.on("ctx", function (data) {
  if (data && data.platform && data.platform.baseUrl) platformCtx = data.platform;
  if (data && data.signedIn && step === "choice") {
    step = "provider";
    schedulePoll();
    render();
  }
});
bootTimer = setTimeout(boot, 600);
ai.ready();
`;
