export type ReasoningEffort = "low" | "medium" | "xhigh";
export type SendMode = "enter" | "ctrlEnter";

/** The web tools the model may use. `ask` pauses each call for the user's OK first (off: the call runs at once). */
export interface WebToolSettings {
  search: boolean;
  fetch: boolean;
  ask: { search: boolean; fetch: boolean };
}

export interface Settings {
  sendMode: SendMode;
  webTools: WebToolSettings;
}

export interface LlmConfig {
  baseUrl: string;
  apiKey: string;
  model: string;
  reasoningEffort: ReasoningEffort;
}

export interface LlmProvider {
  id: string;
  name: string;
  baseUrl: string;
  apiKey: string;
  /** The deployment's gateway provider: same-origin /v1, no user key. */
  mode?: "platform";
  /** This provider does not require an API key (local proxies, gateways). */
  noKey?: boolean;
  /**
   * The model ids known for this BYOK provider: the listing its /models
   * endpoint returned when the provider was set up, plus whatever has been
   * picked since. The platform provider carries none — its list is the
   * deployment's static /v1/models listing.
   */
  models?: string[];
}

/**
 * The model a conversation — or the workspace default — rides: a provider in
 * the workspace plus one of its model ids. Which side of the picker the choice
 * belongs to (Server vs BYOK) is derived from the provider itself.
 */
export interface ModelChoice {
  providerId: string;
  model: string;
}

/**
 * A guest's completed onboarding, kept in a cookie (see lib/guest.ts) so a
 * reload resumes straight to "How may I help you?" instead of replaying the
 * onboarding flow.
 */
export interface GuestSession {
  guest: true;
  providerName: string;
  baseUrl: string;
  apiKey: string;
  /** This provider does not require an API key (then `apiKey` is empty). */
  noKey?: boolean;
  model: string;
  at: number;
}

/** A signed-in account's identity, as reported by the gateway's /auth/me.

 * The name is a short handle — the opening of the account's access key — so
 * the chip in the top bar and the stored credential visibly belong together.
 * (The key itself is never part of the identity: /auth/me never returns it.)
 */
export interface AccountUser {
  id: number;
  name: string;
  /**
   * The client-side user type: "admin" once this session has answered the
   * server's challenge with the account's admin key (see Settings → Admin).
   * The server stores no readable role — only a one-way verifier — and learns
   * "admin" solely from a solved challenge; it never returns a role.
   */
  role?: "user" | "admin";
}

/**
 * A signed-in account's saved LLM provider, as returned by /auth/me and
 * /auth/signin. Unlike the guest cookie this lives in the server-side
 * profile — which is what lets a returning user land straight in a
 * conversation on any device. It is the account's own config, so (and only
 * because the user opted in) the server stores and returns the API key too.
 */
export interface SavedProvider {
  /** Stable id, so a model choice survives a move between devices. */
  id?: string;
  name: string;
  baseUrl: string;
  apiKey: string;
  /** The model the account last rode on this provider. */
  model: string;
  /** The model ids kept for this provider (its /models listing). */
  models?: string[];
  /** This provider does not require an API key. */
  noKey?: boolean;
}

/** One tool the chat model may call (native OpenAI tool calling). */
export interface ChatToolDef {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

/** A live record of one web tool call on an assistant message. */
export interface WebToolEvent {
  /** The tool_call id the model sent. */
  id: string;
  name: "web_search" | "web_fetch";
  /** The search query or the fetched URL — the call's subject. */
  arg: string;
  status: "pending" | "allowed" | "denied" | "done" | "error";
  /** web_search: the sources the search returned (for the card's links). */
  sources?: { title: string; url: string }[];
  /** web_fetch: the fetched page's title. */
  title?: string;
  /** A one-line outcome summary shown on the card. */
  detail?: string;
  error?: string;
}

export interface Agent {
  id: string;
  name: string;
  providerId: string;
  model: string;
  /** How hard the model thinks for this agent. */
  reasoningEffort: ReasoningEffort;
  /** The agent's own system prompt (free text). */
  systemPrompt: string;
  /** Include the app's built-in capabilities: the app builder, web tools and app guidance. */
  builtins: boolean;
  /** Chat Kitsui: the built-in agent (present, first, never removed). */
  builtIn?: boolean;
  /** Whether the model accepts image input; the visual review loop needs it. */
  vision?: boolean;
}

export interface Workspace {
  providers: LlmProvider[];
  /** Chat Kitsui (built in, first) and the user's own agents. */
  agents: Agent[];
  /** The default agent for new conversations (Chat Kitsui unless the user picked another). */
  activeAgentId: string | null;
}

/** What produced a given assistant message; stamped when the request is sent. */
export interface TurnMeta {
  agentId: string;
  agentName: string;
  model: string;
  effort: ReasoningEffort;
  usage?: { prompt: number; completion: number };
  /** Measured exchange speed in tokens/second (null decode = not measurable). */
  speeds?: { prefill: number; decode: number | null };
}

/** What the active agent is: shown on the running head and each exchange label. */
export interface CurrentMeta {
  agentName: string;
  model: string;
  effort: ReasoningEffort;
}

/** One item in the builder's TODO list (the two-sided plan). */
export interface PlanTodo {
  title: string;
  /** Chain-of-thought: what the builder intends to do for this step. */
  cot?: string;
}

/** The triage delegates an edit to an existing workspace. */
export interface EditFence {
  /** The title of the workspace to edit (fallback when the id is absent). */
  studio: string;
  /** The id of the workspace to edit — preferred, titles can drift or repeat. */
  studioId?: string;
  /** One-line description of the requested change. */
  change: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  images?: string[];
  thinking?: string;
  thinkingMs?: number;
  error?: boolean;
  /** Epoch ms when the message was created. */
  at?: number;
  /** Assistant messages only. */
  meta?: TurnMeta;
  /** Assistant messages only: the AI built a studio (workspace) in this reply. */
  studioId?: string;
  /** Transient: the studio block is still streaming in. */
  studioPending?: boolean;
  /** The workspace spec arrived but could not be used. */
  studioFailed?: "truncated" | "unparseable" | "planning";
  /** The model's raw output, kept for the collapsed "show raw" view. */
  studioRaw?: string;
  /** The builder's design plan: ordered TODOs + current stage + self-reported %. */
  plan?: { todos: PlanTodo[]; active: number; done: number; pct: number | null };
  /** Phase 1: the AI is asking one clarifying question before building. */
  question?: { question: string; options?: string[]; assumptions?: string[] };
  /** Phase 1 done: the AI delegated to the builder. */
  buildPending?: boolean;
  /** The plan is ready and the build is waiting for the user's go-ahead. */
  awaitingBuild?: boolean;
  /** The builder's one-line topic (display + context in the builder prompt). */
  brief?: string;
  /** Phase 2: the builder is currently streaming. Transient (not persisted). */
  building?: boolean;
  /** The builder's live reasoning (providers that stream reasoning_content). */
  buildThinking?: string;
  /** The raw streamed output for this turn (fences included, capped tail). */
  raw?: string;
  /** The triage delegated an edit to an existing workspace. */
  editPending?: boolean;
  /** The target studio id for an edit. */
  editStudioId?: string;
  /** This reply delegated its build/edit work to a dedicated thread. */
  threadId?: number;
  /** The automatic design critic: screenshot review + at most one polish pass.
      `reviewing`/`refining` are in-flight (cleared on reload). */
  critic?: {
    status: "reviewing" | "refining" | "approved" | "skipped" | "refined";
    /** The defects the critic (or the lint fallback) reported. */
    issues?: string[];
  };
  /** The App Builder's visual review loop, live (transient; cleared on reload):
      the host renders the app, shows the builder the screenshots, and the
      builder fixes the spec — up to a bounded number of passes. */
  review?: { phase: "rendering" | "reviewing" | "fixing"; pass: number; passes: number };
  /** Web tool calls this turn made (search / fetch), in order. */
  tools?: WebToolEvent[];
}

/** A workspace the AI built by composing shared components (no free HTML). */
export interface Studio {
  id: string;
  title: string;
  spec: StudioSpec;
  conversationId: number;
  createdAt: number;
  /** The debug log tail: runtime output captured while the app ran (capped). */
  logs?: StudioLogEntry[];
  /** The user allowed this app to call the AI in-app (asked once per workspace). */
  aiAllowed?: boolean;
}

/** One captured line of app runtime output (console / error / crash). */
export interface StudioLogEntry {
  at: number;
  source: "logic" | "raw" | "render" | "in-app";
  level: "error" | "warn" | "log";
  message: string;
  stack?: string;
  /** Consecutive identical lines collapse into one entry with a count. */
  count?: number;
}

/** A log line as reported by a runtime, before the host stamps it. */
export interface StudioLogInput {
  source: "logic" | "raw" | "render" | "in-app";
  level: "error" | "warn" | "log";
  message: string;
  stack?: string;
}

/** A closed-vocabulary mutation. No code — paths plus expression bindings. */
export type StudioValue =
  | string
  | number
  | boolean
  | null
  | StudioValue[]
  | { [key: string]: StudioValue };
export interface StudioAction {
  /** Set a data path to a value; strings may carry {{expression}} bindings. */
  set?: Record<string, StudioValue>;
  /** Add a number to a numeric data path. */
  inc?: Record<string, number>;
  /** Append expression-resolved values to array data paths. */
  push?: Record<string, StudioValue>;
  /** Remove by index (number) or first matching value (string) from an array path. */
  remove?: Record<string, string | number>;
  /** Flip boolean data paths. */
  toggle?: string[];
  /** Restore data paths to their initial values. */
  reset?: string[];
  /** Navigate to a state after the mutation. */
  to?: string;
  /** Chained actions, run in order after this action's own mutations. */
  actions?: StudioAction[];
}

/** The wall's right-column label: your message, hung like a gallery tag. */
export interface Wallabel {
  /** 1-based exchange number; hidden while `live`. */
  no?: number;
  /** True while this (last) exchange is being produced. */
  live?: boolean;
  /** The kicker line (e.g. "You"). */
  kicker: string;
  /** Your message (the "title" of the exchange). */
  title: string;
  /** True when the title is long enough to clamp. */
  long?: boolean;
  /** Human time, when known. */
  time?: string;
  /** Attached image sources. */
  images?: string[];
  /** The agent that produced the reply. */
  artist: string;
  /** "model · effort x" — the medium line. */
  medium?: string;
  /** The dimensions line (token counts). */
  dims: string;
  /** The model's reasoning, when present. */
  study?: { content: string; active: boolean; ms?: number };
}

/** The type ramp: how prominent a text element is (largest → smallest). */
export type Emph = "display" | "h1" | "h2" | "body" | "small" | "caption";
/** The type family: which of the three Salon faces an element is set in. */
export type Face = "serif" | "sans" | "mono";
/** Cross-axis alignment of a container's children. */
export type Align = "start" | "center" | "end" | "stretch" | "baseline";
/** Main-axis justification of a container's children. */
export type Justify = "start" | "center" | "end" | "between" | "around";
/** The accent a component is tinted with (the Salon's single red + its allies). */
export type Tone = "accent" | "danger" | "ok" | "muted";

/** Fields every block carries (declarative apps). */
export interface StudioBlockBase {
  /** Conditional rendering: the block renders only when this expression is truthy. */
  if?: string;
}

/** One component drawn from the shared registry. AI-exposed blocks are the
 *  closed vocabulary the model may build; app-only blocks (ai: false in the
 *  registry) are the product's own surfaces, composed by hosted apps. */
type StudioBlockVariant =
  | { type: "text"; content: string; error?: boolean; streaming?: boolean; emph?: Emph; face?: Face }
  | { type: "heading"; content: string; emph?: Emph; face?: Face }
  | { type: "code"; content: string; language?: string; highlight?: number[] }
  | { type: "output"; lines?: string[] | `{{${string}}}` }
  | { type: "button"; label: string; to: string; variant?: "primary" | "ghost"; tone?: Tone; disabled?: boolean; action?: StudioAction }
  | { type: "note"; content: string; tone?: "hint" | "error"; emph?: Emph; face?: Face }
  | { type: "row"; blocks: StudioBlock[]; align?: Align; justify?: Justify }
  | { type: "spacer"; size?: "sm" | "md" | "lg" }
  | { type: "divider" }
  | { type: "card"; blocks: StudioBlock[] }
  | { type: "grid"; cols: number; cells: string[] | `{{${string}}}`; selectable?: boolean; onCell?: StudioAction }
  | { type: "choice"; options: (string | { label: string; value: string })[]; value?: string; onChoose?: StudioAction; tone?: Tone }
  | { type: "select"; label?: string; options: (string | { label: string; value: string })[]; value?: string; onChoose?: StudioAction; chip?: string; ariaLabel?: string }
  | { type: "input"; label?: string; placeholder?: string; value?: string; kind?: "text" | "number" | "email" | "url" | "password"; onSubmit?: StudioAction }
  | { type: "stat"; label: string; value: string; emph?: Emph; face?: Face; tone?: Tone }
  | { type: "table"; cols: string[] | `{{${string}}}`; rows: (string | number)[][] | `{{${string}}}` }
  | { type: "image"; src: string; alt?: string; width?: number }
  | { type: "chat" }
  | { type: "raw"; src: string; id?: string; events?: string[]; width?: number; fill?: boolean; onEvent?: StudioAction }
  // --- layout: full-window composition (AI-exposed) ---
  | { type: "stack"; blocks: StudioBlock[]; fill?: boolean; gap?: "sm" | "md" | "lg"; align?: Align; justify?: Justify }
  | { type: "columns"; blocks: StudioBlock[]; fill?: boolean; gap?: "sm" | "md" | "lg"; align?: Align; justify?: Justify; sizes?: number[] }
  | { type: "panel"; title?: string; blocks: StudioBlock[]; fill?: boolean; width?: number; gap?: "sm" | "md" | "lg"; tone?: Tone }
  | { type: "toolbar"; blocks: StudioBlock[]; sticky?: boolean; align?: Align; justify?: Justify }
  | { type: "list"; each: string; as?: string; blocks: StudioBlock[]; gap?: "sm" | "md" | "lg" }
  // --- app-only components (the product's own surfaces) ---
  | { type: "field"; label?: string; placeholder?: string; value?: string; kind?: "text" | "email" | "url" | "password"; key?: string; className?: string; autoFocus?: boolean; ariaLabel?: string }
  | { type: "wallhead"; title: string; meta: { text: string; model?: boolean }[] }
  | ({ type: "wallabel" } & Wallabel)
  | { type: "work"; label: Wallabel; piece: StudioBlock[]; streaming?: boolean }
  | { type: "study"; content: string; active: boolean; ms?: number }
  | { type: "preparing"; text?: string }
  | { type: "question"; question: string; options?: string[]; assumptions?: string[] }
  | { type: "plan"; todos: PlanTodo[]; active: number; done: number; pct: number | null; confirm?: boolean; thinking?: string; thinkingActive?: boolean }
  | { type: "interrupted"; kind: "truncated" | "unparseable" | "planning"; raw?: string; canRetry?: boolean }
  | { type: "door"; title: string; appId: string }
  | { type: "thread"; appId: string; title: string; status: string; active: boolean }
  | { type: "critic"; status: "reviewing" | "refining" | "refined" }
  | { type: "activity"; phase: string; active: boolean; thinking?: string; writing?: string }
  | { type: "none"; text?: string }
  | { type: "caret" }
  | { type: "status"; text: string }
  | { type: "composer" }
  | { type: "chip"; label: string; thinking?: boolean; delay?: number }
  | { type: "greetinghead"; title: string; sub: string }
  | { type: "group"; className: string; key?: string; blocks: StudioBlock[] }
  | { type: "dot"; variant?: "live" | "static" | "replying" };

/** Every block may carry an `if` condition (distributed over the variants). */
export type StudioBlock = StudioBlockVariant & StudioBlockBase;

/** A named screen of the app. Each state is a complete UI snapshot. */
export interface StudioState {
  blocks: StudioBlock[];
  /** Milliseconds to show this state before advancing to `next`. */
  after?: number;
  next?: string;
  /** Blocks pinned in a footer bar, like the composer is docked. */
  dock?: StudioBlock[];
}

/**
 * A workspace: either a declarative state machine, or a "logic" mode where a
 * sandboxed JS brain drives the same component library over a bridge.
 */
export interface StudioSpec {
  title: string;
  /** Declarative mode. */
  initial?: string;
  states?: Record<string, StudioState>;
  /** The app's mutable state (declarative mode). */
  data?: Record<string, unknown>;
  /** Logic mode: JS source running in a sandboxed iframe. */
  logic?: string;
  /**
   * Logic mode: the AI role this app may use in-app (opponent, quizmaster,
   * coach…). Declaring it enables `ai.ask` and drives the user's one-time
   * consent; `system` is the default prompt for calls without their own.
   */
  ai?: { role: string; system: string };
  /**
   * A trusted brain: the product's own apps (welcome, the conversation wall).
   * Unlike a model-authored brain, a trusted brain may emit the app-only
   * components (wallhead, work, composer, …) and render long lists.
   */
  trusted?: boolean;
}

export interface Conversation {
  id: number;
  title: string;
  messages: ChatMessage[];
  inputTokens: number;
  outputTokens: number;
  createdAt?: number;
  /** The agent this conversation rides; the default agent when absent. */
  agentId?: string;
  /** A build thread: a dedicated conversation that carries one build or edit. */
  kind?: "chat" | "build";
  /** The conversation this build thread was spawned from. */
  parentId?: number;
  /** The workspace this thread produced (or edited) once it lands. */
  studioId?: string;
  /** Build threads: the handoff — the tail of the parent conversation, so the
      builder agent sees the ask, the answers and the constraints, not just the
      one-line brief. */
  context?: ChatMessage[];
}

/**
 * An app: a presentation built from the shared component library, driven by a
 * brain. "default" apps are built-in (hosted brain); "studio" apps are
 * AI-built workspaces (declarative/logic brain).
 */
export interface AppRef {
  id: string;
  title: string;
  kind: "default" | "studio";
}

/**
 * Per-app code overrides from direct editing (the Code pane), keyed by app id.
 * An app's effective source is `overrides[appId] ?? defaultSource[appId]`.
 */
export interface AppOverrides {
  [appId: string]: string;
}
