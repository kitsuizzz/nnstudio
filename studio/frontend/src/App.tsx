import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Sidebar from "./components/Sidebar";
import TopBar, { type AccountState } from "./components/TopBar";
import AuthDialog from "./components/AuthDialog";
import RunningHead from "./components/RunningHead";
import AgentSwitcher from "./components/AgentSwitcher";
import WallFrame from "./components/WallFrame";
import Switcher from "./components/Switcher";
import GreetingFrame, { type GuestStart } from "./components/GreetingFrame";
import SettingsPage from "./components/SettingsPage";
import StudioView from "./components/StudioView";
import CodePane from "./components/CodePane";
import { useTypewriter } from "./components/useTypewriter";
import { completeChat, fetchModelInfos, fetchModelMaxTokens, generateTitle, streamChat, type ChatToolHooks, type ModelInfo, type ToolCallRequest, type Usage } from "./lib/api";
import { buildAgentSystemPrompt, codeEditSystem } from "./lib/agent";
import { defaultApp, defaultSource, effectiveSource } from "./lib/apps";
import { runWebTool, webToolDefs } from "./lib/webtools";
import { uid } from "./lib/id";
import { PLATFORM_BASE_URL, PLATFORM_MODEL, PLATFORM_NAME, PLATFORM_PROVIDER_ID, isPlatform, llmFor, providerCategory, providerUsable } from "./lib/platform";
import { defaultChoice, effectiveChoice, providerFor } from "./lib/modelchoice";
import { CHAT_KITSUI_ID, chatKitsui, effectiveAgent, ensureChatKitsui, normalizeAgent, openAgentsRecord, pointChatKitsui, sealAgentsRecord, type AgentOption } from "./lib/agents";
import {
  clearLocalData,
  loadAppOverrides,
  loadConversations,
  loadStudios,
  saveAppOverrides,
} from "./lib/store";
import { clearGuestSession } from "./lib/guest";
import {
  clearAuthToken,
  fetchAdminChallenge,
  fetchMe,
  type MeResponse,
  guestToProvider,
  proveAdmin,
  providerToGuest,
  readAuthToken,
  saveAdminKey,
  saveAgentsRecord,
  saveProviders,
  signout as authSignout,
  writeAuthToken,
} from "./lib/auth";
import { decryptJson, deriveEncKey, encryptJson, openSecret, sealSecret } from "./lib/crypto";
import { cacheKey, dropCachedKey, getCachedKey } from "./lib/keyvault";
import { fetchStore, syncStore, type StoreRecord } from "./lib/cloud";
import { type ReadingSpot } from "./components/readingSpot";
import { parseStudioBlock, splitStudioBlock, builderPrompt, builderEditPrompt, planPrompt, criticPrompt, workspaceReviewPrompt } from "./lib/studio";
import { captureStudioSet } from "./lib/capture";
import { handoffContext } from "./lib/appbuilder";
import { lintSpec } from "./lib/studio-lint";
import { splitPlanBlocks } from "./lib/plan";
import type {
  AccountUser,
  Agent,
  AppOverrides,
  ChatMessage,
  Conversation,
  CurrentMeta,
  GuestSession,
  LlmConfig,
  LlmProvider,
  SavedProvider,
  Settings,
  Studio,
  StudioLogInput,
  StudioSpec,
  ReasoningEffort,
  TurnMeta,
  WebToolEvent,
  WebToolSettings,
  Workspace,
} from "./types";

const SETTINGS_KEY = "aistudio.settings";
const SIDEBAR_KEY = "aistudio.sidebarCollapsed";
const WORKSPACE_KEY = "aistudio.workspace";
const THEME_KEY = "aistudio.theme";
/** How long a theme switch takes to ease (matches the `theme-flip` transitions in index.css). */
const THEME_FLIP_MS = 700;
const SAVE_DEBOUNCE_MS = 300;
/** How long the reader rests before their place is saved. */
const VIEW_SAVE_MS = 1000;
/** The debug-log tail kept per workspace (persisted with the studio record). */
const MAX_STUDIO_LOGS = 50;
/** The raw streamed output kept per message for the inline activity window. */
const RAW_STREAM_MAX = 8_000;
/** How many visual review/fix passes the App Builder gets per build or edit. */
const MAX_REVIEW_PASSES = 2;

/** The built-in apps that run on a code brain (and so have a Code pane). */
const CODE_APP_IDS = ["greeting", "conversation"];

export type Theme = "light" | "dark";

const defaultSettings: Settings = {
  sendMode: "ctrlEnter",
  webTools: { search: true, fetch: true, ask: { search: true, fetch: true } },
};

function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<Settings>;
      const tools = (parsed.webTools ?? {}) as Partial<WebToolSettings>;
      return {
        sendMode: parsed.sendMode === "enter" ? "enter" : defaultSettings.sendMode,
        // Web tools default on, and each call asks first unless the user turns the asking off.
        webTools: {
          search: tools.search !== false,
          fetch: tools.fetch !== false,
          ask: {
            search: tools.ask?.search !== false,
            fetch: tools.ask?.fetch !== false,
          },
        },
      };
    }
  } catch {
    // corrupted settings, fall back to defaults
  }
  return defaultSettings;
}

interface LegacySettings {
  baseUrl?: string;
  apiKey?: string;
  model?: string;
}

/**
 * How long the boot waits on the gateway before giving up on it. A request
 * that hangs (a slow link, a proxy that never answers) must not leave the
 * app stuck half-loaded on a blank screen.
 */
const BOOT_FETCH_MS = 10_000;

/** A timeout signal, with a manual fallback for older browsers. */
function timeoutSignal(ms: number): AbortSignal {
  const withTimeout = AbortSignal as typeof AbortSignal & { timeout?: (ms: number) => AbortSignal };
  if (typeof withTimeout.timeout === "function") return withTimeout.timeout(ms);
  const ctrl = new AbortController();
  window.setTimeout(() => ctrl.abort(), ms);
  return ctrl.signal;
}

/**
 * The deployment's gateway listing (the Server side of the model picker),
 * probed once per session — it is static per deployment, so a second probe
 * answers from memory.
 */
let platformProbe: Promise<{ baseUrl: string; model: string; models: ModelInfo[] } | null> | null = null;
function probePlatform(): Promise<{ baseUrl: string; model: string; models: ModelInfo[] } | null> {
  platformProbe ??= (async () => {
    const models = await fetchModelInfos(PLATFORM_BASE_URL, "", timeoutSignal(BOOT_FETCH_MS));
    if (!models.length) return null;
    const model =
      PLATFORM_MODEL && models.some((m) => m.id === PLATFORM_MODEL) ? PLATFORM_MODEL : models[0].id;
    return { baseUrl: PLATFORM_BASE_URL, model, models };
  })();
  return platformProbe;
}

/**
 * Fold the account's saved BYOK providers into the workspace: the saved copy
 * wins (it is the cross-device source of truth) and keeps its id, so model
 * choices survive a move between devices. Each provider gets an agent to ride.
 */
function mergeSavedProviders(ws: Workspace, opened: SavedProvider[]): Workspace {
  if (opened.length === 0) return ws;
  let providers = [...ws.providers];
  const renames = new Map<string, string>();
  for (const saved of opened) {
    if (!saved.baseUrl.trim()) continue;
    const wantedId = saved.id?.trim() || "";
    const at = providers.findIndex(
      (p) =>
        !isPlatform(p) &&
        ((wantedId && p.id === wantedId) || (p.name === saved.name && p.baseUrl === saved.baseUrl))
    );
    if (at < 0) {
      providers = [
        ...providers,
        { id: wantedId || uid(), name: saved.name, baseUrl: saved.baseUrl, apiKey: saved.apiKey, noKey: saved.noKey, models: saved.models },
      ];
    } else {
      const prev = providers[at];
      const id = wantedId || prev.id;
      if (id !== prev.id) renames.set(prev.id, id);
      providers[at] = {
        ...prev,
        id,
        name: saved.name || prev.name,
        baseUrl: saved.baseUrl || prev.baseUrl,
        // A sealed (or absent) key never replaces an opened one.
        apiKey: saved.apiKey && !saved.apiKey.startsWith("e2e1:") ? saved.apiKey : prev.apiKey,
        noKey: saved.noKey ?? prev.noKey,
        models: saved.models?.length ? saved.models : prev.models,
      };
    }
  }
  const follow = (id: string): string => renames.get(id) ?? id;
  return { ...ws, providers, agents: ws.agents.map((a) => ({ ...a, providerId: follow(a.providerId) })) };
}

/** The saved-provider list without the entry that describes `saved`. */
function withoutProvider(list: SavedProvider[], saved: SavedProvider): SavedProvider[] {
  return list.filter(
    (p) => !((saved.id && p.id === saved.id) || (p.name === saved.name && p.baseUrl === saved.baseUrl))
  );
}

/** The guest's in-session providers as saved entries — what the first sign-in
    claims into the account (the caller seals the keys on upload). */
function claimGuestProviders(ws: Workspace): SavedProvider[] {
  const choice = defaultChoice(ws);
  return ws.providers
    .filter((p) => !isPlatform(p) && (p.apiKey.trim().length > 0 || p.noKey === true))
    .map((p) => ({
      id: p.id,
      name: p.name,
      baseUrl: p.baseUrl,
      apiKey: p.apiKey,
      noKey: p.noKey,
      models: p.models,
      model: choice?.providerId === p.id ? choice.model : p.models?.[0] ?? "",
    }));
}

function seedWorkspace(legacy: LegacySettings | null): Workspace {
  const provider: LlmProvider = {
    id: uid(),
    name: "OpenAI",
    baseUrl: legacy?.baseUrl?.trim() || "https://api.openai.com/v1",
    apiKey: legacy?.apiKey?.trim() ?? "",
    models: legacy?.model?.trim() ? [legacy.model.trim()] : undefined,
  };
  return {
    providers: [provider],
    agents: [chatKitsui(provider.id, legacy?.model?.trim() || "gpt-4o-mini")],
    activeAgentId: CHAT_KITSUI_ID,
  };
}

function loadWorkspace(): Workspace {
  try {
    const raw = localStorage.getItem(WORKSPACE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<Workspace> & { modelChoice?: { providerId?: string; model?: string } | null };
      if (Array.isArray(parsed.providers) && Array.isArray(parsed.agents)) {
        // The platform provider gets its stable id, so the server choice keeps
        // pointing at it across reloads and devices (a legacy random id is remapped).
        const legacyPlatformId = parsed.providers.find((p) => p.mode === "platform" && p.id !== PLATFORM_PROVIDER_ID)?.id ?? null;
        const remap = (id?: string | null): string => (id && id === legacyPlatformId ? PLATFORM_PROVIDER_ID : id ?? "");
        const providers = (parsed.providers as LlmProvider[]).map((p) => (p.id === legacyPlatformId ? { ...p, id: PLATFORM_PROVIDER_ID } : p));
        const rawAgents = parsed.agents as Partial<Agent>[];
        // Earlier builds made one "Assistant" agent per provider; those carry no prompt of their own and fold into Chat Kitsui.
        const agents = rawAgents
          .filter((a) => !(a.name === "Assistant" && !a.systemPrompt && !a.builtIn))
          .map((a) => normalizeAgent({ ...a, providerId: remap(a.providerId) }))
          .filter((a): a is Agent => a !== null);
        const oldActive = rawAgents.find((a) => a.id === parsed.activeAgentId) ?? rawAgents[0];
        const choice = parsed.modelChoice?.providerId
          ? { providerId: remap(parsed.modelChoice.providerId), model: parsed.modelChoice.model ?? "" }
          : oldActive
            ? { providerId: remap(oldActive.providerId), model: oldActive.model ?? "" }
            : { providerId: providers.find((p) => isPlatform(p))?.id ?? "", model: "" };
        const withChat = ensureChatKitsui(agents, choice);
        const activeAgentId = withChat.some((a) => a.id === parsed.activeAgentId) ? (parsed.activeAgentId as string) : CHAT_KITSUI_ID;
        const loaded: Workspace = { providers, agents: withChat, activeAgentId };
        return readAuthToken() ? loaded : guestSafeWorkspace(loaded);
      }
    }
  } catch {
    // corrupted workspace, re-seed below
  }
  let legacy: LegacySettings | null = null;
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) legacy = JSON.parse(raw) as LegacySettings;
  } catch {
    legacy = null;
  }
  return seedWorkspace(legacy);
}

/** A guest's providers and custom agents are session-only: the stored copy keeps just the deployment's server provider and Chat Kitsui, so a reload never resurrects a guest's providers (keys included) or agents. Signed-in accounts re-fetch theirs from the server profile. */
function guestSafeWorkspace(ws: Workspace): Workspace {
  return reconcileChat({
    ...ws,
    providers: ws.providers.filter((p) => isPlatform(p)),
    agents: ws.agents.filter((a) => a.builtIn),
    activeAgentId: CHAT_KITSUI_ID,
  });
}

/** The workspace without the deployment's server provider (used when the gateway answers with no models, so nothing can ride a stale server model). */
function withoutPlatform(ws: Workspace): Workspace {
  if (!ws.providers.some((p) => isPlatform(p))) return ws;
  return reconcileChat({ ...ws, providers: ws.providers.filter((p) => !isPlatform(p)) });
}

/** Chat Kitsui must ride a provider that exists: it keeps its choice when that provider survives; otherwise it moves to the server model, or the next provider left, or to nothing (the picker then asks for a provider). Agents on a removed provider keep their definitions and need a new provider. */
function reconcileChat(ws: Workspace, platformModel = ""): Workspace {
  const chat = ws.agents.find((a) => a.builtIn);
  if (chat && ws.providers.some((p) => p.id === chat.providerId)) return ws;
  const next = ws.providers.find((p) => isPlatform(p)) ?? ws.providers[0];
  const model = next && isPlatform(next) ? platformModel : "";
  return pointChatKitsui(ws, next?.id ?? "", model, false);
}

/** Reload recovery: a build that was in-flight when the app closed is orphaned. */
function recoverConversations(list: Conversation[]): Conversation[] {
  return list.map((c) => ({
    ...c,
    messages: c.messages.map((m) => {
      if (m.role === "assistant" && m.buildPending && !m.studioId && !m.studioFailed && !m.awaitingBuild) {
        return { ...m, building: false, studioFailed: "truncated" as const };
      }
      if (m.role === "assistant" && m.editPending && !m.studioFailed) {
        return { ...m, building: false, studioFailed: "truncated" as const };
      }
      if (m.role === "assistant" && m.building) {
        return { ...m, building: false, review: undefined };
      }
      if (m.role === "assistant" && m.review) {
        return { ...m, review: undefined };
      }
      if (m.role === "assistant" && (m.critic?.status === "reviewing" || m.critic?.status === "refining")) {
        return { ...m, critic: { status: "skipped" as const } };
      }
      return m;
    }),
  }));
}

/** The builder streams progress fences without re-emitting the plan; merge the
    latest active/done/pct into the plan the planner published. */
function mergeProgressPlan(
  plan: ChatMessage["plan"] | null,
  progress: { active: number | null; done: number | null; pct: number | null } | null,
  last: ChatMessage["plan"]
): ChatMessage["plan"] {
  if (plan) return plan;
  if (!progress || !last) return last;
  const active = Math.min(progress.active ?? last.active, last.todos.length);
  const done = Math.min(progress.done ?? last.done, last.todos.length);
  const pct = progress.pct ?? last.pct;
  if (active === last.active && done === last.done && pct === last.pct) return last;
  return { ...last, active, done, pct };
}

/** Decrypt a batch of server records; one bad record is skipped, not fatal. */
async function decryptAll<T>(key: CryptoKey, records: StoreRecord[]): Promise<T[]> {
  const out: T[] = [];
  for (const rec of records) {
    try {
      out.push(await decryptJson<T>(key, rec.blob));
    } catch {
      // Wrong key or tampered ciphertext: leave it out of the loaded set.
    }
  }
  return out;
}

/** The reading spots, for the conversations that are still there; an unreadable record is left out. */
async function decryptViews(
  key: CryptoKey,
  records: StoreRecord[],
  conversations: Conversation[]
): Promise<Map<number, ReadingSpot>> {
  const live = new Set(conversations.map((c) => String(c.id)));
  const out = new Map<number, ReadingSpot>();
  for (const rec of records) {
    if (!live.has(rec.id)) continue;
    try {
      out.set(Number(rec.id), await decryptJson<ReadingSpot>(key, rec.blob));
    } catch {
      // Wrong key or tampered ciphertext: leave it out.
    }
  }
  return out;
}

/** Upload the reading spots that changed since the last sync (only those records; a removed spot is deleted). */
async function syncReadingSpots(
  cloud: { token: string; key: CryptoKey },
  spotsRef: { current: Map<number, ReadingSpot> },
  syncedRef: { current: { views: Map<string, string> } },
  keepalive = false
): Promise<void> {
  const synced = syncedRef.current.views;
  const upserts: { id: string; blob: string }[] = [];
  const signatures = new Map<string, string>();
  for (const [id, spot] of spotsRef.current) {
    const signature = JSON.stringify(spot);
    if (synced.get(String(id)) === signature) continue;
    upserts.push({ id: String(id), blob: await encryptJson(cloud.key, spot) });
    signatures.set(String(id), signature);
  }
  const deletes = [...synced.keys()].filter((id) => !spotsRef.current.has(Number(id)));
  if (!upserts.length && !deletes.length) return;
  const ok = await syncStore(
    cloud.token,
    {
      conversations: { upserts: [], deletes: [] },
      studios: { upserts: [], deletes: [] },
      views: { upserts, deletes },
    },
    keepalive
  );
  if (!ok) return; // retried on the next change
  for (const [id, signature] of signatures) synced.set(id, signature);
  for (const id of deletes) synced.delete(id);
}

/** The conversation the account was last in, or null when unknown. */
async function decryptLast(key: CryptoKey, records: StoreRecord[]): Promise<number | null> {
  const rec = records.find((r) => r.id === "current");
  if (!rec) return null;
  try {
    const value = await decryptJson<{ conversationId?: number }>(key, rec.blob);
    return typeof value.conversationId === "number" ? value.conversationId : null;
  } catch {
    // Wrong key or tampered ciphertext: no pointer.
    return null;
  }
}

/** Upload the last-session pointer when it changed (the account's one record). */
async function syncLastSession(
  cloud: { token: string; key: CryptoKey },
  lastRef: { current: number | null },
  syncedRef: { current: { last: Map<string, string> } },
  keepalive = false
): Promise<void> {
  const synced = syncedRef.current.last;
  const id = lastRef.current;
  if (id == null) {
    if (!synced.has("current")) return;
    const ok = await syncStore(
      cloud.token,
      {
        conversations: { upserts: [], deletes: [] },
        studios: { upserts: [], deletes: [] },
        last: { upserts: [], deletes: ["current"] },
      },
      keepalive
    );
    if (ok) synced.delete("current");
    return;
  }
  const signature = JSON.stringify({ conversationId: id });
  if (synced.get("current") === signature) return;
  const ok = await syncStore(
    cloud.token,
    {
      conversations: { upserts: [], deletes: [] },
      studios: { upserts: [], deletes: [] },
      last: { upserts: [{ id: "current", blob: await encryptJson(cloud.key, { conversationId: id }) }], deletes: [] },
    },
    keepalive
  );
  if (ok) synced.set("current", signature);
}

/** What changed since the last sync: re-encrypted upserts + removed ids. */
async function diffRecords(
  current: { id: string; value: unknown }[],
  synced: Map<string, string>,
  key: CryptoKey,
): Promise<{ upserts: { id: string; blob: string }[]; deletes: string[] }> {
  const upserts: { id: string; blob: string }[] = [];
  const deletes: string[] = [];
  const seen = new Set<string>();
  for (const { id, value } of current) {
    seen.add(id);
    const signature = JSON.stringify(value);
    if (synced.get(id) === signature) continue;
    upserts.push({ id, blob: await encryptJson(key, value) });
  }
  for (const id of synced.keys()) if (!seen.has(id)) deletes.push(id);
  return { upserts, deletes };
}

export default function App() {
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [workspace, setWorkspace] = useState<Workspace>(loadWorkspace);
  // The guest's completed onboarding, for this tab only: a provider set up as
  // a guest is never persisted (no cookie, no stored workspace), so the
  // greeting brain can resume within the session but a reload starts clean.
  const [guest, setGuest] = useState<GuestSession | null>(null);
  // The signed-in account (the top bar's right-hand chip). While a token
  // cookie exists but /auth/me has not answered, the chip shows "…".
  const [account, setAccount] = useState<AccountUser | null>(null);
  /** The platform-provided models: the deployment's gateway (its listing is
      static per deployment). Null when it does not answer — the greeting then
      falls back to the provider form. */
  const [platform, setPlatform] = useState<{ baseUrl: string; model: string; models: ModelInfo[] } | null>(null);
  /** The signed-in account's saved BYOK providers (empty = platform mode). */
  const [profile, setProfile] = useState<SavedProvider[]>([]);
  const [accountState, setAccountState] = useState<AccountState>(() => (readAuthToken() ? "loading" : "guest"));
  const [authMode, setAuthMode] = useState<"signin" | "signup" | null>(null);
  const [showSwitcher, setShowSwitcher] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    try {
      return localStorage.getItem(SIDEBAR_KEY) === "1";
    } catch {
      return false;
    }
  });
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const themeFlipTimer = useRef<number | null>(null);
  const [theme, setTheme] = useState<Theme>(() =>
    document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light"
  );
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [studios, setStudios] = useState<Studio[]>([]);
  const [appOverrides, setAppOverrides] = useState<AppOverrides>({});
  /** Which app's Code pane is open (a code-app id), if any. */
  const [codeApp, setCodeApp] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [activeId, setActiveId] = useState<number | null>(null);
  const [activeStudioId, setActiveStudioId] = useState<string | null>(null);
  const [streamingId, setStreamingId] = useState<number | null>(null);
  const [buildingId, setBuildingId] = useState<number | null>(null);
  const [firstTokenCid, setFirstTokenCid] = useState<number | null>(null);
  /** Live prefill/decode speed (tok/s) of the exchange currently streaming. */
  const [speeds, setSpeeds] = useState<{ prefill: number; decode: number | null } | null>(null);
  /** The active model's context window (max tokens), from the provider's /models endpoint. */
  const [maxTokens, setMaxTokens] = useState<number | null>(null);
  const [titleAnimId, setTitleAnimId] = useState(0);
  const [wallTextVisible, setWallTextVisible] = useState(true);
  /** True once the user completes onboarding this session (prevents the auto-route from yanking them to the wall mid-greeting). */
  const onboardedThisSessionRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const builderAbortRef = useRef<AbortController | null>(null);
  const titleAbortRef = useRef<AbortController | null>(null);
  const idRef = useRef(1);
  /** The live cloud session (token + derived key); null for guests. */
  const cloudRef = useRef<{ token: string; key: CryptoKey } | null>(null);
  /** The last-synced plaintext per record id, so saves upload only changes. */
  const syncedRef = useRef<{
    conversations: Map<string, string>;
    studios: Map<string, string>;
    views: Map<string, string>;
    last: Map<string, string>;
  }>({
    conversations: new Map(),
    studios: new Map(),
    views: new Map(),
    last: new Map(),
  });
  /** Where the reader left each conversation: in memory, and (signed in) synced encrypted. */
  const viewSpotsRef = useRef<Map<number, ReadingSpot>>(new Map());
  const viewSaveRef = useRef<number | null>(null);
  /** The conversation the user was last in: in memory, and (signed in) synced encrypted. */
  const lastRef = useRef<number | null>(null);
  const lastSaveRef = useRef<number | null>(null);
  /** True when a valid session has no cached key (the store is locked). */
  const [needsUnlock, setNeedsUnlock] = useState(false);
  const unlockPromptedRef = useRef(false);
  const activeIdRef = useRef<number | null>(activeId);

  useEffect(() => {
    activeIdRef.current = activeId;
  }, [activeId]);

  useEffect(() => {
    try {
      // A guest's providers are session-only: the stored copy never carries
      // them (or their keys), so a reload comes back clean. A signed-in
      // account's providers come back from the server profile instead.
      const persisted = readAuthToken() ? workspace : guestSafeWorkspace(workspace);
      localStorage.setItem(WORKSPACE_KEY, JSON.stringify(persisted));
    } catch {
      // storage unavailable
    }
  }, [workspace]);

  useEffect(() => {
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    } catch {
      // storage unavailable
    }
  }, [settings]);

  /** Install a loaded record set as the app's state (and its sync baseline). */
  /** Append a runtime line to a studio's debug log (capped; repeats collapse). */
  const appendStudioLog = (studioId: string, input: StudioLogInput) => {
    setStudios((prev) =>
      prev.map((s) => {
        if (s.id !== studioId) return s;
        const logs = s.logs ? [...s.logs] : [];
        const last = logs[logs.length - 1];
        if (
          last &&
          last.source === input.source &&
          last.level === input.level &&
          last.message === input.message &&
          Date.now() - last.at < 3000
        ) {
          logs[logs.length - 1] = { ...last, at: Date.now(), count: (last.count ?? 1) + 1 };
        } else {
          logs.push({ ...input, at: Date.now() });
        }
        return { ...s, logs: logs.slice(-MAX_STUDIO_LOGS) };
      })
    );
  };

  /** Drop a studio's debug log (the Debug panel's Clear). */
  const clearStudioLogs = (studioId: string) => {
    setStudios((prev) => prev.map((s) => (s.id === studioId ? { ...s, logs: undefined } : s)));
  };

  const adopt = (list: Conversation[], studioList: Studio[], views?: Map<number, ReadingSpot>) => {
    setConversations(list);
    setStudios(studioList);
    idRef.current = list.length ? Math.max(...list.map((c) => c.id)) + 1 : 1;
    if (views) viewSpotsRef.current = views;
    // A fresh baseline: nothing is "dirty" until the user changes something.
    syncedRef.current = {
      conversations: new Map(list.map((c) => [String(c.id), JSON.stringify(c)])),
      studios: new Map(studioList.map((s) => [s.id, JSON.stringify(s)])),
      views: new Map([...viewSpotsRef.current].map(([id, spot]) => [String(id), JSON.stringify(spot)])),
      last: new Map(
        lastRef.current == null
          ? []
          : [["current", JSON.stringify({ conversationId: lastRef.current })] as [string, string]],
      ),
    };
    // The incoming set replaces whatever was on screen, so drop the selection
    // (the auto-route then reopens the conversation the account was last in —
    // or a fresh one when there is none).
    setActiveId(null);
    setActiveStudioId(null);
  };

  /** Encrypt and upload a whole record set (the one-shot guest → account claim). */
  const uploadAll = async (token: string, key: CryptoKey, list: Conversation[], studioList: Studio[]) => {
    const convUpserts = await Promise.all(
      list.map(async (c) => ({ id: String(c.id), blob: await encryptJson(key, c) })),
    );
    const studioUpserts = await Promise.all(
      studioList.map(async (s) => ({ id: s.id, blob: await encryptJson(key, s) })),
    );
    const viewUpserts = await Promise.all(
      [...viewSpotsRef.current].map(async ([id, spot]) => ({ id: String(id), blob: await encryptJson(key, spot) })),
    );
    // The guest's open conversation is where the new account should land back.
    const openId = list.some((c) => c.id === lastRef.current) ? lastRef.current : null;
    await syncStore(token, {
      conversations: { upserts: convUpserts, deletes: [] },
      studios: { upserts: studioUpserts, deletes: [] },
      views: { upserts: viewUpserts, deletes: [] },
      last: {
        upserts: openId == null ? [] : [{ id: "current", blob: await encryptJson(key, { conversationId: openId }) }],
        deletes: [],
      },
    });
  };

  /** The reader's spot in a conversation moved: remember it (a signed-in account also syncs it, encrypted). */
  const onReadingSpot = (id: number, spot: ReadingSpot) => {
    viewSpotsRef.current.set(id, spot);
    scheduleViewSync();
  };

  const readingSpotFor = (id: number): ReadingSpot | null => viewSpotsRef.current.get(id) ?? null;

  /** Spots are written once the reader pauses (and at once when the page is hidden or closed). */
  const scheduleViewSync = () => {
    if (viewSaveRef.current !== null) window.clearTimeout(viewSaveRef.current);
    viewSaveRef.current = window.setTimeout(() => {
      viewSaveRef.current = null;
      const cloud = cloudRef.current;
      if (cloud) void syncReadingSpots(cloud, viewSpotsRef, syncedRef);
    }, VIEW_SAVE_MS);
  };

  // The conversation the user is in moved: remember it (a signed-in account
  // syncs it, encrypted). Written once they pause (and at once via the flush below).
  useEffect(() => {
    if (activeId == null) return;
    lastRef.current = activeId;
    if (lastSaveRef.current !== null) window.clearTimeout(lastSaveRef.current);
    lastSaveRef.current = window.setTimeout(() => {
      lastSaveRef.current = null;
      const cloud = cloudRef.current;
      if (cloud) void syncLastSession(cloud, lastRef, syncedRef);
    }, VIEW_SAVE_MS);
  }, [activeId]);

  useEffect(() => {
    const flush = () => {
      const cloud = cloudRef.current;
      if (cloud && document.visibilityState === "hidden") {
        void syncReadingSpots(cloud, viewSpotsRef, syncedRef, true);
        void syncLastSession(cloud, lastRef, syncedRef, true);
      }
    };
    const onPageHide = () => {
      const cloud = cloudRef.current;
      if (cloud) {
        void syncReadingSpots(cloud, viewSpotsRef, syncedRef, true);
        void syncLastSession(cloud, lastRef, syncedRef, true);
      }
    };
    document.addEventListener("visibilitychange", flush);
    window.addEventListener("pagehide", onPageHide);
    return () => {
      document.removeEventListener("visibilitychange", flush);
      window.removeEventListener("pagehide", onPageHide);
    };
  }, []);

  /** Load the account's encrypted store. When the account has none yet, claim
      the browser's local records instead: encrypted, uploaded, then dropped. */
  const adoptCloud = async (
    token: string,
    key: CryptoKey,
    fallback: { conversations: Conversation[]; studios: Studio[] },
  ) => {
    const store = await fetchStore(token);
    if (!store) {
      adopt([], []);
      return;
    }
    if (store.conversations.length === 0 && store.studios.length === 0) {
      if (fallback.conversations.length || fallback.studios.length) {
        await uploadAll(token, key, fallback.conversations, fallback.studios);
        adopt(fallback.conversations, fallback.studios);
        await clearLocalData();
        return;
      }
      adopt([], []);
      return;
    }
    const loadedConversations = await decryptAll<Conversation>(key, store.conversations);
    lastRef.current = await decryptLast(key, store.last);
    adopt(
      loadedConversations,
      await decryptAll<Studio>(key, store.studios),
      await decryptViews(key, store.views, loadedConversations),
    );
  };

  // Persistence: load once, then save (debounced) on every change.
  useEffect(() => {
    let cancelled = false;
    const token = readAuthToken();
    // A guest's providers are session-only; purge the legacy cookie so an old
    // build's write can never resurrect one.
    if (!token) clearGuestSession();
    void (async () => {
      try {
        const overrides = await loadAppOverrides();
        if (cancelled) return;
        if (overrides) setAppOverrides(overrides);
        const loadLocal = async () => {
          const [list, studioList] = await Promise.all([loadConversations(), loadStudios()]);
          return { conversations: recoverConversations(list ?? []), studios: studioList ?? [] };
        };
        if (!token) {
          // Guest: conversations are never persisted — they live in memory for
          // this session only, so a reload starts clean. With no usable BYOK
          // provider on this browser, the server-provided platform model is
          // adopted (no provider form).
          adopt([], []);
          await adoptPlatform();
          return;
        }
        // Signed in: resolve the identity, then the key that unlocks the store.
        // A timeout or network failure must not read as "this session is dead":
        // the token stays for the next load, and this one shows the guest shell
        // instead of waiting forever on a blank screen.
        let me: MeResponse | null = null;
        let reachable = true;
        try {
          me = await fetchMe(token, timeoutSignal(BOOT_FETCH_MS));
        } catch {
          reachable = false;
        }
        if (cancelled) return;
        if (!me) {
          // A token the server rejected is dropped; an unreachable one is kept.
          if (reachable) clearAuthToken();
          setAccount(null);
          setAccountState("guest");
          // Guests keep nothing on disk: start clean.
          adopt([], []);
          await adoptPlatform();
          return;
        }
        setAccount(me.user);
        setAccountState("user");
        const key = await getCachedKey(me.user.id);
        if (cancelled) return;
        if (!key) {
          // The session outlived the browser's key cache (a new device, or site
          // data was cleared): the store stays locked until the key is re-entered.
          // The local records are held in memory as the claim fallback but are
          // left untouched on disk (the save effect is paused while locked). The
          // profile is not adopted either — its apiKey is sealed.
          const local = await loadLocal();
          if (cancelled) return;
          adopt(local.conversations, local.studios);
          setNeedsUnlock(true);
          return;
        }
        cloudRef.current = { token, key };
        // Adopt the profile only once its key exists: the stored apiKeys are
        // sealed, and the derived key is what opens them. With no profile (and
        // no legacy cookie) the account lands in platform mode instead of a form.
        const mode = await adoptProvider(me.providers, token, key);
        await adoptAgents(me.agents, key);
        await adoptPlatform(mode === "none");
        if (await proveAdminFrom(token, me.adminKey, key)) {
          setAccount((a) => (a ? { ...a, role: "admin" } : a));
        }
        await adoptCloud(token, key, await loadLocal());
      } catch {
        // Never leave the app half-loaded (a blank screen): fall back to the
        // guest shell; the next load retries the session.
        if (!cancelled) {
          setAccount(null);
          setAccountState("guest");
          adopt([], []);
          void adoptPlatform();
        }
      } finally {
        if (!cancelled) setLoaded(true);
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- one-shot; reads the initial workspace
  }, []);

  /** True when the default agent (Chat Kitsui) rides a usable BYOK provider. */
  const activeByok = (ws: Workspace): boolean => {
    const provider = providerFor(ws, defaultChoice(ws));
    return !!provider && !isPlatform(provider) && providerUsable(provider);
  };

  /** Adopt the account's saved providers. Chat Kitsui rides the saved default (the model the account last used on it), unless it already rides a usable BYOK provider: BYOK always wins over platform mode. */
  const workspaceForByok = (ws: Workspace, opened: SavedProvider[]): Workspace => {
    const merged = mergeSavedProviders(ws, opened);
    if (opened.length === 0 || activeByok(merged)) return merged;
    const first = opened[0];
    const provider =
      merged.providers.find(
        (p) => !isPlatform(p) && (first.id ? p.id === first.id : p.name === first.name && p.baseUrl === first.baseUrl)
      ) ?? null;
    return provider ? pointChatKitsui(merged, provider.id, first.model) : merged;
  };

  /** Point Chat Kitsui at the deployment's server model, adding the server provider if needed. A usable BYOK default is left alone unless `force`. */
  const workspaceForPlatform = (ws: Workspace, model: string, force = false): Workspace => {
    if (!force && activeByok(ws)) return ws;
    const withPlatform: Workspace = ws.providers.some((p) => isPlatform(p))
      ? ws
      : {
          ...ws,
          providers: [
            ...ws.providers,
            { id: PLATFORM_PROVIDER_ID, name: PLATFORM_NAME, baseUrl: PLATFORM_BASE_URL, apiKey: "", mode: "platform" },
          ],
        };
    const chat = withPlatform.agents.find((a) => a.builtIn);
    const keepModel = chat && chat.providerId === PLATFORM_PROVIDER_ID ? chat.model : "";
    return pointChatKitsui(withPlatform, PLATFORM_PROVIDER_ID, model || keepModel, false);
  };

  /** Probe the deployment's gateway and record its (static) model listing — the Server side of the picker. When it answers, Chat Kitsui is pointed at it too, unless a usable BYOK default already rides. */
  const adoptPlatform = async (force = false): Promise<boolean> => {
    const found = await probePlatform();
    if (!found) {
      // The deployment offers no server models: a platform choice saved when it did must not linger.
      setWorkspace((current) => withoutPlatform(current));
      return false;
    }
    setPlatform(found);
    // The check runs on the latest workspace (a usable BYOK default still wins).
    setWorkspace((current) => workspaceForPlatform(current, found.model, force));
    return true;
  };

  /** Prove the account's stored admin key: the server sends a challenge, the client answers with the key, and only then does the session know it is an admin. The server stores no readable user type — only a one-way verifier — so this is the single moment it learns "admin" at all. */
  const proveAdminFrom = async (token: string, sealed: string | null, key: CryptoKey | null): Promise<boolean> => {
    if (!sealed || !key) return false;
    const adminKey = await openSecret(key, sealed);
    // openSecret hands an unopenable value back unchanged; that is a failure.
    if (!adminKey || adminKey === sealed) return false;
    const challenge = await fetchAdminChallenge(token);
    if (!challenge) return false;
    return proveAdmin(token, challenge, adminKey);
  };

  /** The account's agents, sealed in its profile, restored over the workspace. Chat Kitsui keeps its saved setup when the provider it rides still exists; custom agents and the default come back as saved. */
  const adoptAgents = async (sealed: string | null, key: CryptoKey | null): Promise<void> => {
    const saved = await openAgentsRecord(key, sealed);
    if (!saved) return;
    setWorkspace((ws) => {
      const savedChat = saved.agents.find((a) => a.builtIn);
      // The server provider is re-added by adoptPlatform; until then it still counts as kept.
      const providerKept = (id: string) => id === PLATFORM_PROVIDER_ID || ws.providers.some((p) => p.id === id);
      const current = ws.agents.find((a) => a.builtIn) ?? chatKitsui("", "");
      const chat = savedChat && providerKept(savedChat.providerId) ? savedChat : current;
      const agents = [chat, ...saved.agents.filter((a) => !a.builtIn)];
      const defaultOk = agents.some((a) => a.id === saved.defaultAgentId);
      return { ...ws, agents, activeAgentId: defaultOk ? saved.defaultAgentId : CHAT_KITSUI_ID };
    });
  };

  /** Adopt a signed-in account's saved providers the moment we learn who they are. The server profile is authoritative: resume straight into a conversation from it. With none, the caller falls back to platform mode. */
  const adoptProvider = async (
    saved: SavedProvider[],
    token: string,
    key: CryptoKey | null,
    keepGuest = false
  ): Promise<"profile" | "none"> => {
    if (saved.length) {
      // The stored apiKeys are sealed; open them with the account key before the providers are used.
      const opened: SavedProvider[] = [];
      for (const provider of saved) {
        opened.push({ ...provider, id: provider.id || uid(), apiKey: await openSecret(key, provider.apiKey) });
      }
      setProfile(opened);
      // Profiles saved before provider ids existed get one now (one write; sealing is idempotent, so the stored keys ride along unchanged).
      if (saved.some((p) => !p.id)) void saveProviders(token, opened, key);
      setWorkspace((ws) => workspaceForByok(ws, opened));
      setGuest(opened[0] ? providerToGuest(opened[0]) : null);
      return "profile";
    }
    setProfile([]);
    setGuest(null);
    // A profile-less account must not inherit another account's local rows (a fresh sign-up keeps the guest's own agents).
    setWorkspace((ws) => (keepGuest ? ws : guestSafeWorkspace(ws)));
    return "none";
  };

  // A valid session whose key cache is gone (a new device, or site data was
  // cleared) can't decrypt: ask for the access key once, reusing the sign-in
  // dialog. Dismissing it leaves the app usable, just locked.
  useEffect(() => {
    if (needsUnlock && !unlockPromptedRef.current) {
      unlockPromptedRef.current = true;
      setAuthMode("signin");
    }
  }, [needsUnlock]);

  // Persistence, debounced: guests keep conversations in memory only (nothing
  // is written to disk); a signed-in account writes the encrypted server store
  // — only the records that changed, so a conversation carrying an image isn't
  // re-uploaded on every keystroke.
  useEffect(() => {
    if (!loaded || needsUnlock) return;
    // No account session = a guest: conversations die with the tab.
    if (accountState !== "user") return;
    const t = setTimeout(() => {
      const cloud = cloudRef.current;
      if (!cloud) return; // nothing to sync with
      void (async () => {
        const conv = await diffRecords(
          conversations.map((c) => ({ id: String(c.id), value: c })),
          syncedRef.current.conversations,
          cloud.key,
        );
        const studio = await diffRecords(
          studios.map((s) => ({ id: s.id, value: s })),
          syncedRef.current.studios,
          cloud.key,
        );
        if (!conv.upserts.length && !conv.deletes.length && !studio.upserts.length && !studio.deletes.length) return;
        const ok = await syncStore(cloud.token, { conversations: conv, studios: studio });
        if (!ok) return; // retried on the next change
        for (const c of conversations) syncedRef.current.conversations.set(String(c.id), JSON.stringify(c));
        for (const id of conv.deletes) syncedRef.current.conversations.delete(id);
        for (const s of studios) syncedRef.current.studios.set(s.id, JSON.stringify(s));
        for (const id of studio.deletes) syncedRef.current.studios.delete(id);
      })();
    }, SAVE_DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [conversations, studios, loaded, needsUnlock, accountState]);

  // Overrides are always saved once loaded (even down to an empty map), so a
  // discarded edit survives a reload instead of resurrecting the old override.
  useEffect(() => {
    if (!loaded) return;
    const t = setTimeout(() => void saveAppOverrides(appOverrides), SAVE_DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [appOverrides, loaded]);

  // Leaving a stage closes whatever Code pane was open over it.
  useEffect(() => {
    setCodeApp(null);
  }, [activeId, activeStudioId]);

  // ⌘K / Ctrl+K opens the switcher.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && (e.key === "k" || e.key === "K")) {
        e.preventDefault();
        setShowSwitcher((s) => !s);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const active = conversations.find((c) => c.id === activeId) ?? null;
  const activeStudio = studios.find((s) => s.id === activeStudioId) ?? null;
  /** True while a build thread is open (a `kind: "build"` conversation). */
  const inThread = active?.kind === "build";
  const activeTitle = active?.title ?? "";
  const typedTitle = useTypewriter(activeTitle, titleAnimId);

  /** The agent this conversation rides: its own pick, else the default agent. */
  const activeAgent = effectiveAgent(workspace, active);
  /** The reasoning effort is the agent's own (low until the user picks another). */
  const effectiveEffort: ReasoningEffort = activeAgent?.reasoningEffort ?? "low";

  /** The provider + model the conversation's agent rides. */
  const modelChoice = useMemo(() => effectiveChoice(workspace, active), [workspace, active]);
  const modelProvider = providerFor(workspace, modelChoice);
  const modelId = modelChoice?.model ?? "";
  /** The agent list the picker shows: Chat Kitsui first, then the user's own. */
  const agentOptions: AgentOption[] = useMemo(
    () =>
      workspace.agents.map((a) => {
        const provider = workspace.providers.find((p) => p.id === a.providerId) ?? null;
        return {
          id: a.id,
          name: a.name,
          model: a.model,
          providerName: provider?.name ?? "No provider",
          category: providerCategory(provider),
          usable: providerUsable(provider) && !!a.model,
        };
      }),
    [workspace.providers, workspace.agents]
  );

  /** The provider list Agent Settings manages: the account's saved providers,
      as the database returned them, or a guest's session providers. */
  const agentProviders: SavedProvider[] = useMemo(() => {
    if (accountState === "user") return profile;
    const fallback = defaultChoice(workspace);
    return workspace.providers
      .filter((p) => !isPlatform(p))
      .map((p) => ({
        id: p.id,
        name: p.name,
        baseUrl: p.baseUrl,
        apiKey: p.apiKey,
        noKey: p.noKey,
        models: p.models,
        model: fallback?.providerId === p.id ? fallback.model : p.models?.[0] ?? "",
      }));
  }, [accountState, profile, workspace]);

  /** The build pipeline runs on the agent that owns the conversation: the same
      LLM + message stamp the chat turn used. `vision` gates the visual review
      loop (seeing what it built) and the agent's model is assumed to take
      image input — there is no separate builder agent to configure any more. */
  const builderFor = (llm: LlmConfig, meta: TurnMeta): { llm: LlmConfig; meta: TurnMeta; vision: boolean } => ({
    llm,
    meta,
    vision: true,
  });

  const sendBlocker = !activeAgent
    ? "No agent is configured, so messages can't be sent."
    : !modelProvider || !modelId || !providerUsable(modelProvider)
      ? `Agent “${activeAgent.name}” has no usable provider.`
      : null;

  /** The landing page's "Continue as guest" can start a session right away
      when the platform gateway or a usable BYOK provider already exists. */
  const guestCanStart = platform != null || providerUsable(modelProvider);

  const current: CurrentMeta = useMemo(
    () => ({
      agentName: activeAgent?.name ?? "No agent",
      model: modelId || "—",
      effort: effectiveEffort,
    }),
    [activeAgent, modelId, effectiveEffort]
  );

  // The running head's "xx / max tok" budget needs the active model's context
  // window; ask the provider's single-model endpoint when the conversation's
  // model (or its provider) changes.
  useEffect(() => {
    if (!loaded || !modelProvider || !modelId) {
      setMaxTokens(null);
      return;
    }
    if (!providerUsable(modelProvider)) {
      setMaxTokens(null);
      return;
    }
    const provider = modelProvider;
    let cancelled = false;
    setMaxTokens(null);
    void fetchModelMaxTokens(provider.baseUrl, isPlatform(provider) ? "" : provider.apiKey, modelId).then((n) => {
      if (!cancelled && n != null) setMaxTokens(n);
    });
    return () => {
      cancelled = true;
    };
  }, [loaded, modelProvider, modelId]);

  // The brains of the built-in code apps, injected into every agent's system
  // prompt so the model can see (and reason about) the code of the apps the
  // user may ask it to modify. Tagged so a provider route can key off them.
  const appCode = useMemo(() => {
    const sections = CODE_APP_IDS.map((id) => {
      const info = defaultApp(id);
      const src = effectiveSource(id, appOverrides);
      if (!info || !src) return "";
      return `<app-source id="${id}" title="${info.title}">\n${src}\n</app-source>`;
    }).filter((s) => s.length > 0);
    if (sections.length === 0) return "";
    return (
      "The built-in apps the user may ask about or modify ship with this brain " +
      "source code:\n\n" +
      sections.join("\n\n")
    );
  }, [appOverrides]);

  const toggleTheme = () => {
    const next: Theme = theme === "dark" ? "light" : "dark";
    const root = document.documentElement;
    // The colours ease only while the class is on <html>; the rest of the app keeps its own timing.
    root.classList.add("theme-flip");
    root.setAttribute("data-theme", next);
    if (themeFlipTimer.current !== null) window.clearTimeout(themeFlipTimer.current);
    themeFlipTimer.current = window.setTimeout(() => {
      root.classList.remove("theme-flip");
      themeFlipTimer.current = null;
    }, THEME_FLIP_MS + 50);
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch {
      // storage unavailable
    }
    setTheme(next);
  };

  /**
   * Web tools: the model asks to search or fetch. Each call asks first unless Settings turns that off,
   * then runs against the platform gateway. Events land
   * on the turn's assistant message for the activity window; the answer names its sources.
   */
  const toolConsentRef = useRef<((ok: boolean) => void) | null>(null);
  const [toolAsk, setToolAsk] = useState<ToolCallRequest | null>(null);

  const decideToolConsent = (ok: boolean) => {
    const resolve = toolConsentRef.current;
    toolConsentRef.current = null;
    setToolAsk(null);
    resolve?.(ok);
  };

  const askToolConsent = (call: ToolCallRequest) =>
    new Promise<boolean>((resolve) => {
      toolConsentRef.current?.(false);
      toolConsentRef.current = resolve;
      setToolAsk(call);
    });

  /** Patch one tool event on the (last) assistant message of `cid`. */
  const patchTool = (cid: number, id: string, patch: Partial<WebToolEvent>) => {
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== cid) return c;
        const msgs = [...c.messages];
        const last = msgs[msgs.length - 1];
        if (!last || last.role !== "assistant") return c;
        const tools = [...(last.tools ?? [])];
        const at = tools.findIndex((t) => t.id === id);
        if (at < 0) return c;
        tools[at] = { ...tools[at], ...patch };
        msgs[msgs.length - 1] = { ...last, tools };
        return { ...c, messages: msgs };
      })
    );
  };

  /** The streamChat tool hooks for one turn, when Settings allows the tools. */
  const toolHooksFor = (cid: number, signal: AbortSignal): ChatToolHooks | undefined => {
    const defs = activeAgent?.builtins === false ? [] : webToolDefs(settings.webTools);
    if (defs.length === 0) return undefined;
    return {
      defs,
      onCall: (call) => {
        const name: WebToolEvent["name"] = call.name === "web_fetch" ? "web_fetch" : "web_search";
        const arg =
          call.name === "web_search"
            ? typeof call.args.query === "string"
              ? call.args.query
              : ""
            : typeof call.args.url === "string"
              ? call.args.url
              : "";
        const event: WebToolEvent = { id: call.id, name, arg, status: "pending" };
        setConversations((prev) =>
          prev.map((c) => {
            if (c.id !== cid) return c;
            const msgs = [...c.messages];
            const last = msgs[msgs.length - 1];
            if (!last || last.role !== "assistant") return c;
            msgs[msgs.length - 1] = { ...last, tools: [...(last.tools ?? []), event] };
            return { ...c, messages: msgs };
          })
        );
      },
      approve: (call) =>
        settings.webTools.ask[call.name === "web_fetch" ? "fetch" : "search"] ? askToolConsent(call) : Promise.resolve(true),
      onDenied: (call) => patchTool(cid, call.id, { status: "denied" }),
      execute: async (call) => {
        patchTool(cid, call.id, { status: "allowed" });
        try {
          const run = await runWebTool(call.name, call.args, signal);
          patchTool(cid, call.id, { status: "done", ...run.patch });
          return run.content;
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          patchTool(cid, call.id, { status: "error", error: message });
          throw err;
        }
      },
    };
  };

  const stop = () => {
    decideToolConsent(false);
    abortRef.current?.abort();
    abortRef.current = null;
    builderAbortRef.current?.abort();
    builderAbortRef.current = null;
    titleAbortRef.current?.abort();
    titleAbortRef.current = null;
    setStreamingId(null);
    setBuildingId(null);
    setFirstTokenCid(null);
    setConversations((prev) =>
      prev.map((c) => {
        const last = c.messages[c.messages.length - 1];
        // Only a true placeholder (an empty reply with nothing attached) is
        // dropped. A fence-only message that carries a build, a plan, a
        // workspace, or a design review is real work — keep it.
        if (
          last &&
          last.role === "assistant" &&
          last.content === "" &&
          !last.thinking &&
          !last.studioId &&
          !last.plan &&
          !last.studioFailed &&
          !last.question &&
          !last.buildPending &&
          !last.editPending &&
          !last.critic &&
          !(last.tools && last.tools.length > 0)
        ) {
          return { ...c, messages: c.messages.slice(0, -1) };
        }
        return c;
      })
    );
  };

  const toggleSidebarCollapsed = () => {
    setSidebarCollapsed((prev) => {
      const next = !prev;
      try {
        localStorage.setItem(SIDEBAR_KEY, next ? "1" : "0");
      } catch {
        // storage unavailable
      }
      return next;
    });
  };

  /** Drops abandoned empty conversations, except the one being kept or streamed. */
  const pruneEmpty = useCallback(
    (keepId: number | null) => {
      setConversations((prev) =>
        prev.filter((c) => c.messages.length > 0 || c.id === keepId || c.id === streamingId)
      );
    },
    [streamingId]
  );

  const openConversation = useCallback(
    (id: number) => {
      pruneEmpty(id);
      setWallTextVisible(true);
      setActiveId(id);
      setSidebarOpen(false);
    },
    [pruneEmpty]
  );

  const newChat = useCallback(() => {
    // Reuse an existing empty conversation instead of piling up blanks.
    const blank = conversations.find((c) => c.messages.length === 0 && c.id !== streamingId);
    if (blank) {
      openConversation(blank.id);
      return;
    }
    const id = idRef.current++;
    setConversations((prev) => [
      ...prev,
      {
        id,
        title: "",
        messages: [],
        inputTokens: 0,
        outputTokens: 0,
        createdAt: Date.now(),
      },
    ]);
    setWallTextVisible(true);
    setActiveId(id);
    setSidebarOpen(false);
  }, [conversations, streamingId, openConversation]);

  // A signed-in user lands back where they left: once the session is confirmed
  // and nothing is active, reopen the conversation they were last in (the
  // encrypted `last` pointer), or start a fresh chat when there is none. Guests
  // keep the greeting. Two guards keep a just-signed-in user in the greeting:
  // no guest cookie yet (they haven't set up a provider) or onboarding
  // completed this session (they're on the ready screen about to send).
  useEffect(() => {
    if (!loaded) return;
    // Only signed-in accounts auto-route: they always ride a provider (their
    // own BYOK or the platform model), so they land in a session. Guests
    // ALWAYS start on the landing page (the welcome choice) — "Continue as
    // guest" is what opens a new session for them.
    if (accountState !== "user") return;
    if (guest == null && platform == null) return;
    if (onboardedThisSessionRef.current) return;
    if (activeId != null || activeStudioId != null) return;
    const last = lastRef.current;
    if (last != null && conversations.some((c) => c.id === last)) openConversation(last);
    else newChat();
  }, [loaded, accountState, guest, platform, activeId, activeStudioId, newChat, openConversation, conversations]);

  /** Provider setup (guest onboarding, or a signed-in user re-configuring): Chat Kitsui rides the provider and model chosen here. A signed-in account persists the provider in its profile; a guest's setup lives in this tab only. */
  const startGuest = (cfg: GuestStart) => {
    const signedIn = accountState === "user";
    // Only a genuine guest is held on the greeting's ready screen after this (the flag suppresses the auto-route); a signed-in user goes to the conversation instead.
    if (!signedIn) onboardedThisSessionRef.current = true;
    const provider: LlmProvider = {
      id: uid(),
      name: cfg.providerName,
      baseUrl: cfg.baseUrl,
      apiKey: cfg.apiKey,
      models: cfg.model ? [cfg.model] : [],
    };
    const session: GuestSession = {
      guest: true,
      providerName: cfg.providerName,
      baseUrl: cfg.baseUrl,
      apiKey: cfg.apiKey,
      model: cfg.model,
      at: Date.now(),
    };
    if (signedIn) {
      // The profile is the source of truth (per-account, cross-device).
      const saved: SavedProvider = { ...guestToProvider(session), id: provider.id, models: provider.models };
      persistProviders([...withoutProvider(profile, saved), saved]);
    }
    // A guest's session lives in this tab only: nothing is written to a cookie (or the stored workspace), so a reload returns to the landing page.
    setGuest(session);
    commitWorkspace(pointChatKitsui({ ...workspace, providers: [...workspace.providers, provider] }, provider.id, cfg.model));
  };

  /** The auth dialog confirmed (sign-in, or a new account was created). The account's providers and agents are adopted immediately (mirroring the on-load path). A fresh account keeps what the guest set up in this tab. */
  const completeAuth = (
    user: AccountUser,
    token: string,
    providers: SavedProvider[],
    accessKey: string,
    adminKey: string | null = null,
    agents: string | null = null
  ) => {
    // Derive the record key first: the stored profile is sealed under it, so it can only be adopted once the key exists.
    void (async () => {
      const key = await deriveEncKey(accessKey);
      await cacheKey(user.id, key);
      writeAuthToken(token);
      cloudRef.current = { token, key };
      setNeedsUnlock(false);
      setAccount(user);
      setAccountState("user");
      onboardedThisSessionRef.current = false;
      // A fresh account claims what the guest set up in this session (its providers and agents, sealed and uploaded). An existing profile is authoritative instead.
      const fresh = providers.length === 0 && agents === null;
      const claimed = providers.length === 0 ? claimGuestProviders(workspace) : [];
      const mode = await adoptProvider(providers.length ? providers : claimed, token, key, fresh);
      if (claimed.length) void saveProviders(token, claimed, key);
      if (agents) {
        await adoptAgents(agents, key);
      } else if (workspace.agents.some((a) => !a.builtIn)) {
        persistAgents(workspace.agents, workspace.activeAgentId);
      }
      await adoptPlatform(mode === "none");
      if (await proveAdminFrom(token, adminKey, key)) {
        setAccount((a) => (a ? { ...a, role: "admin" } : a));
      }
      await adoptCloud(token, key, { conversations, studios });
    })();
  };

  const signOutUser = () => {
    const token = readAuthToken();
    if (token) void authSignout(token);
    clearAuthToken();
    // The record key leaves with the session: a shared machine must not keep another account's encrypted store readable after sign-out.
    if (account) void dropCachedKey(account.id);
    cloudRef.current = null;
    setNeedsUnlock(false);
    setAccount(null);
    setAccountState("guest");
    setProfile([]);
    // Sign-out returns to the landing page: the account's keys stop being usable (the rows stay in the profile), its custom agents leave the workspace (they come back with the profile), and Chat Kitsui returns to the server model.
    clearGuestSession();
    setGuest(null);
    setWorkspace((ws) => {
      const unkeyed: Workspace = {
        ...ws,
        providers: ws.providers.map((p) => (isPlatform(p) ? p : { ...p, apiKey: "" })),
        agents: ws.agents.filter((a) => a.builtIn),
        activeAgentId: CHAT_KITSUI_ID,
      };
      return platform ? workspaceForPlatform(unkeyed, platform.model, true) : reconcileChat(unkeyed);
    });
    // The records live server-side (encrypted), so the in-memory copies go too
    // — the last-session pointer included, so the next account never resumes
    // this one's conversation.
    lastRef.current = null;
    adopt([], [], new Map());
    setStreamingId(null);
    setBuildingId(null);
    setFirstTokenCid(null);
    setSpeeds(null);
  };

  /** Persist the account's BYOK providers where a resume can find them: the server profile (their keys sealed under the account key). */
  const persistProviders = (next: SavedProvider[]) => {
    setProfile(next);
    const token = readAuthToken();
    if (token) void saveProviders(token, next, cloudRef.current?.key ?? null);
  };

  /** A signed-in account's agents travel with its profile, sealed. Guests keep theirs for the session only. */
  const persistAgents = (agents: Agent[], defaultAgentId: string | null) => {
    const token = readAuthToken();
    const key = cloudRef.current?.key ?? null;
    if (!token || !key) return;
    void (async () => {
      const sealed = await sealAgentsRecord(key, agents, defaultAgentId);
      if (sealed) await saveAgentsRecord(token, sealed);
    })();
  };

  /** Commit a workspace change; a signed-in account's agents follow it into the profile. */
  const commitWorkspace = (next: Workspace) => {
    setWorkspace(next);
    persistAgents(next.agents, next.activeAgentId);
  };

  /** The greeting brain's model override (a model the onboarding flow picked): Chat Kitsui's model. */
  const changeModel = (model: string) => {
    const chat = workspace.agents.find((a) => a.builtIn);
    if (!chat) return;
    commitWorkspace(pointChatKitsui(workspace, chat.providerId, model));
    setGuest((g) => (g ? { ...g, model, at: Date.now() } : g));
  };

  /** The agent picker: this conversation's agent, or the default agent when no conversation is open. */
  const setConversationAgent = (agentId: string | null) => {
    if (!active) {
      commitWorkspace({ ...workspace, activeAgentId: agentId ?? CHAT_KITSUI_ID });
      return;
    }
    setConversations((prev) => prev.map((c) => (c.id === active.id ? { ...c, agentId: agentId ?? undefined } : c)));
  };

  /** Agent Settings: make an agent the default for new conversations. */
  const setDefaultAgent = (agentId: string) => {
    if (!workspace.agents.some((a) => a.id === agentId)) return;
    commitWorkspace({ ...workspace, activeAgentId: agentId });
  };

  /** Agent Settings: add or update an agent. Chat Kitsui keeps its built-in prompts and capabilities; its provider, model and effort are the user's. */
  const saveAgent = (draft: Agent) => {
    const entry: Agent = draft.builtIn
      ? chatKitsui(draft.providerId, draft.model, draft.reasoningEffort)
      : { ...draft, id: draft.id || uid(), builtins: draft.builtins !== false, builtIn: undefined };
    const exists = workspace.agents.some((a) => a.id === entry.id);
    const list = exists ? workspace.agents.map((a) => (a.id === entry.id ? entry : a)) : [...workspace.agents, entry];
    commitWorkspace({ ...workspace, agents: [...list.filter((a) => a.builtIn), ...list.filter((a) => !a.builtIn)] });
  };

  /** Agent Settings: remove one of the user's agents. Chat Kitsui stays; conversations that rode the agent fall back to the default. */
  const removeAgent = (agentId: string) => {
    const target = workspace.agents.find((a) => a.id === agentId);
    if (!target || target.builtIn) return;
    setConversations((prev) => prev.map((c) => (c.agentId === agentId ? { ...c, agentId: undefined } : c)));
    commitWorkspace({
      ...workspace,
      agents: workspace.agents.filter((a) => a.id !== agentId),
      activeAgentId: workspace.activeAgentId === agentId ? CHAT_KITSUI_ID : workspace.activeAgentId,
    });
  };

  /** Settings → Admin: take the operator's admin key, seal it under the
      account key, store that copy server-side (opaque there), and prove it
      right away. Returns true when the server accepted the proof. */
  const saveAdminKeyForAccount = async (rawKey: string): Promise<boolean> => {
    const accountKey = cloudRef.current?.key ?? null;
    const token = readAuthToken();
    const trimmed = rawKey.trim();
    if (!token || !accountKey || !trimmed) return false;
    const sealed = await sealSecret(accountKey, trimmed);
    await saveAdminKey(token, sealed);
    const ok = await proveAdminFrom(token, sealed, accountKey);
    // A key that does not prove is not kept around.
    if (!ok) await saveAdminKey(token, null);
    setAccount((a) => (a ? { ...a, role: ok ? "admin" : undefined } : a));
    return ok;
  };

  /** Agent Settings → provider box: ride one of the saved providers (its last model). Chat Kitsui is re-pointed and becomes the default. */
  const switchProvider = (provider: SavedProvider) => {
    const id = provider.id;
    const model = provider.model || provider.models?.[0] || "";
    if (!id || !model) return;
    commitWorkspace(pointChatKitsui(workspace, id, model));
  };

  /** Agent Settings: save (add or update) one of the user's providers, and ride it: Chat Kitsui is pointed at it and becomes the default. */
  const saveProvider = (next: SavedProvider, model: string) => {
    const id = next.id || uid();
    const entry: SavedProvider = { ...next, id };
    if (accountState === "user") {
      const at = profile.findIndex((p) => p.id === id);
      persistProviders(at >= 0 ? profile.map((p) => (p.id === id ? entry : p)) : [...profile, entry]);
    }
    setGuest((g) => (g ? { ...providerToGuest(entry), at: Date.now() } : g));
    commitWorkspace(pointChatKitsui(mergeSavedProviders(workspace, [entry]), id, model));
  };

  /** Agent Settings: remove one of the user's providers. Agents on it keep their definitions and need a new provider; Chat Kitsui moves to the server model, or to the next provider left. */
  const removeProvider = (id: string) => {
    if (accountState === "user") persistProviders(profile.filter((p) => p.id !== id));
    const providers = workspace.providers.filter((p) => p.id !== id);
    const agents = workspace.agents.map((a) => (a.providerId === id ? { ...a, providerId: "" } : a));
    commitWorkspace(reconcileChat({ ...workspace, providers, agents }, platform?.model ?? ""));
  };

  /** Agent Settings: ride the deployment's server model again. The user's providers stay saved; Chat Kitsui is pointed at the server and becomes the default. */
  const saveServerDefault = (model: string) => {
    if (!platform) return;
    commitWorkspace(pointChatKitsui(workspaceForPlatform(workspace, model, true), PLATFORM_PROVIDER_ID, model));
  };

  /** Persist a Code-pane edit as the app's source override (empty/same → drop). */
  const saveAppCode = (appId: string, source: string) => {
    if (appId === "greeting") {
      clearGuestSession();
      setGuest(null);
      setWorkspace(seedWorkspace(null));
      try {
        localStorage.removeItem(WORKSPACE_KEY);
      } catch {
        // Ignore: a full clear is a best-effort reset.
      }
    }
    setAppOverrides((prev) => {
      const next = { ...prev };
      const shipped = (defaultSource[appId] ?? "").trim();
      const trimmed = source.trim();
      if (!trimmed || trimmed === shipped) delete next[appId];
      else next[appId] = source;
      return next;
    });
  };

  /** Drop the app's override; it reverts to the shipped source. */
  const discardAppCode = (appId: string) => {
    if (appId === "greeting") {
      clearGuestSession();
      setGuest(null);
      setWorkspace(seedWorkspace(null));
      try {
        localStorage.removeItem(WORKSPACE_KEY);
      } catch {
        // Ignore: a full clear is a best-effort reset.
      }
    }
    setAppOverrides((prev) => {
      if (!(appId in prev)) return prev;
      const next = { ...prev };
      delete next[appId];
      return next;
    });
  };

  /** The Code pane's AI assist: run the instruction against the buffer. */
  const assistAppCode = async (appId: string, buffer: string, instruction: string) => {
    const info = defaultApp(appId);
    if (!info) throw new Error("Unknown app.");
    const provider = modelProvider;
    if (!provider || !modelId || !providerUsable(provider)) {
      throw new Error("No usable provider is available to assist.");
    }
    const llm = llmFor(provider, modelId, activeAgent?.reasoningEffort ?? "low");
    const reply = await completeChat(llm, codeEditSystem(info.title, buffer), instruction);
    const match = reply.match(/```[a-zA-Z0-9]*\n([\s\S]*?)```/);
    if (!match) throw new Error("The model's reply did not contain a code fence.");
    const content = match[1];
    if (!/ai\.(render|on|ready)\(/.test(content)) {
      throw new Error("The model's reply does not look like a brain source.");
    }
    return content;
  };

  const deleteConversation = (id: number) => {
    if (streamingId === id) stop();
    // Deleting a conversation also drops the build threads it spawned.
    const remaining = conversations.filter((c) => c.id !== id && c.parentId !== id);
    for (const c of conversations) if (!remaining.includes(c)) viewSpotsRef.current.delete(c.id);
    scheduleViewSync();
    setConversations(remaining);
    if (activeId === id) {
      const list = [...remaining].sort(
        (a, b) => (b.createdAt ?? b.id) - (a.createdAt ?? a.id)
      );
      if (list.length) {
        openConversation(list[0].id);
      } else {
        // The last conversation is gone: start a fresh one. `newChat` can't
        // be reused here — its closure still sees the pre-delete list and
        // could re-adopt the just-deleted empty conversation.
        const freshId = idRef.current++;
        setConversations((prev) => [
          ...prev,
          {
            id: freshId,
            title: "",
            messages: [],
            inputTokens: 0,
            outputTokens: 0,
            createdAt: Date.now(),
          },
        ]);
        setWallTextVisible(true);
        setActiveId(freshId);
        setSidebarOpen(false);
      }
    }
  };

  /**
   * A message inside a build thread talks to the builder directly — no triage
   * and no nested threads. With a workspace, it edits it; while the plan is
   * waiting, it becomes the new brief and re-plans.
   */
  const sendInThread = async (thread: Conversation, text: string, images: string[]) => {
    const content = text.trim();
    if ((!content && images.length === 0) || streamingId != null || buildingId != null) return;
    const agent = activeAgent;
    const provider = modelProvider;
    if (!agent || !provider || !modelId || !providerUsable(provider)) return;
    const llm = llmFor(provider, modelId, effectiveEffort);
    const meta: TurnMeta = { agentId: agent.id, agentName: agent.name, model: modelId, effort: effectiveEffort };
    const builder = builderFor(llm, meta);
    const userMsg: ChatMessage = {
      role: "user",
      content,
      images: images.length ? images : undefined,
      at: Date.now(),
    };
    const placeholder: ChatMessage = {
      role: "assistant",
      content: "",
      meta: builder.meta,
    };
    setConversations((prev) =>
      prev.map((c) => (c.id === thread.id ? { ...c, messages: [...c.messages, userMsg, placeholder] } : c))
    );

    const studio = thread.studioId ? studios.find((s) => s.id === thread.studioId) ?? null : null;
    if (studio) {
      void launchEdit(thread.id, studio.id, content, builder.llm, thread.context ?? [], studio.spec, images, builder.vision);
      return;
    }
    const lastAssistant = [...thread.messages].reverse().find((m) => m.role === "assistant");
    if (lastAssistant?.awaitingBuild && lastAssistant.plan) {
      // The plan is waiting: the message becomes the new brief and we re-plan.
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== thread.id) return c;
          const msgs = [...c.messages];
          const last = msgs[msgs.length - 1];
          if (last?.role === "assistant") msgs[msgs.length - 1] = { ...last, brief: content, awaitingBuild: false };
          return { ...c, messages: msgs };
        })
      );
    }
    void launchPlan(thread.id, content, builder.llm, images, thread.context ?? []);
  };

  const send = async (text: string, images: string[] = []) => {
    // Inside a build thread the composer talks to the builder directly.
    if (active?.kind === "build") {
      void sendInThread(active, text, images);
      return;
    }
    const content = text.trim();
    if ((!content && images.length === 0) || streamingId != null || buildingId != null) return;
    const agent = activeAgent;
    const provider = modelProvider;
    if (!agent || !provider || !modelId || !providerUsable(provider)) return;
    const llm = llmFor(provider, modelId, effectiveEffort);
    const systemPrompt = buildAgentSystemPrompt(agent, appCode || undefined, studios.map((s) => ({ title: s.title, id: s.id })), settings.webTools);
    const meta: TurnMeta = {
      agentId: agent.id,
      agentName: agent.name,
      model: modelId,
      effort: effectiveEffort,
    };

    const userMsg: ChatMessage = {
      role: "user",
      content,
      images: images.length ? images : undefined,
      at: Date.now(),
    };
    const placeholder: ChatMessage = { role: "assistant", content: "", meta };
    const fallbackTitle = content.slice(0, 40) || "[Image]";
    let cid = activeId;
    let history: ChatMessage[];
    let next: Conversation[];
    let needsTitle = false;

    if (cid == null) {
      cid = idRef.current++;
      needsTitle = true;
      history = [userMsg];
      next = [
        ...conversations,
        {
          id: cid,
          title: fallbackTitle,
          messages: [userMsg, placeholder],
          inputTokens: 0,
          outputTokens: 0,
          createdAt: Date.now(),
        },
      ];
    } else {
      const conv = conversations.find((c) => c.id === cid);
      // A new turn moves on: any plan still waiting for confirmation is
      // abandoned (its card stays, but it can no longer start a build).
      const clean = (conv?.messages ?? [])
        .filter((m) => !m.error)
        .map((m) => (m.awaitingBuild ? { ...m, awaitingBuild: false } : m));
      needsTitle = clean.length === 0;
      history = [...clean, userMsg];
      const title = clean.length === 0 ? fallbackTitle : undefined;
      next = conversations.map((c) =>
        c.id === cid
          ? { ...c, title: title ?? c.title, messages: [...clean, userMsg, placeholder] }
          : c
      );
    }

    const finalCid = cid;
    setConversations(next);
    setActiveId(finalCid);
    setStreamingId(cid);
    setFirstTokenCid(null);

    let reply = "";
    let studioClaimed = false;
    let thinkingStartedAt: number | null = null;
    let thinkingEndAt: number | null = null;
    let firstToken = false;
    const controller = new AbortController();
    abortRef.current = controller;

    // Speed measurement: prefill spans the wait for the first token; decode
    // spans only the time tokens were actually arriving (streamChat times each
    // round's stream), so web-tool rounds can't dilute it with their consent
    // waits, tool runs and follow-up round-trips. One mutable object, shared by
    // the stream callbacks and the settlement below. Prefill is published the
    // moment the first token lands (its window is closed then); decode is
    // deliberately left null while streaming and published only at the end, as
    // an average over the exchange's streaming time.
    const timing = {
      t0: performance.now(),
      tFirst: null as number | null,
      chars: 0,
      usage: null as Usage | null,
      /** Output tokens summed over every round (a tool exchange has several). */
      completion: 0,
    };
    // Rough prompt size (chars/4, flat fee per image) — it stands in for the
    // usage report until the provider sends one.
    const promptEstimate =
      3 +
      Math.ceil(systemPrompt.length / 4) +
      images.length * 100 +
      Math.ceil(history.reduce((n, m) => n + m.content.length, 0) / 4);
    const markFirstToken = () => {
      if (firstToken) return;
      firstToken = true;
      timing.tFirst = performance.now();
      setFirstTokenCid(finalCid);
      setSpeeds({
        prefill: (promptEstimate / Math.max(1, timing.tFirst - timing.t0)) * 1000,
        decode: null,
      });
    };
    setSpeeds(null);

    const streamed = await streamChat(
      llm,
      history,
      systemPrompt,
      (delta) => {
        if (thinkingStartedAt != null && thinkingEndAt == null) {
          thinkingEndAt = performance.now();
        }
        markFirstToken();
        timing.chars += delta.length;
        reply += delta;
        const { visible: planVisible, plan } = splitPlanBlocks(reply);
        const split = splitStudioBlock(planVisible);
        let studioId: string | undefined;
        let showRaw = false;
        if (split.block != null && !studioClaimed) {
          studioClaimed = true;
          const doc = parseStudioBlock(split.block);
          if (doc) {
            const studio: Studio = {
              id: uid(),
              title: doc.title,
              spec: doc.spec,
              conversationId: finalCid,
              createdAt: Date.now(),
            };
            studioId = studio.id;
            setStudios((prev) => [...prev, studio]);
          } else {
            // Malformed block: surface the model's raw output instead of eating it.
            showRaw = true;
          }
        }
        setConversations((prev) =>
          prev.map((c) => {
            if (c.id !== finalCid) return c;
            const msgs = [...c.messages];
            const last = msgs[msgs.length - 1];
            if (last && last.role === "assistant") {
              msgs[msgs.length - 1] = {
                ...last,
                content: showRaw ? reply : split.visible,
                studioId: studioId ?? last.studioId,
                studioPending: split.open,
                studioFailed: split.failed,
                studioRaw: split.failed ? reply : last.studioRaw,
                plan: plan ?? last.plan,
                raw: ((last.raw ?? "") + delta).slice(-RAW_STREAM_MAX),
              };
            }
            return { ...c, messages: msgs };
          })
        );
      },
      (reasoning) => {
        if (thinkingStartedAt == null) {
          thinkingStartedAt = performance.now();
        }
        markFirstToken();
        timing.chars += reasoning.length;
        setConversations((prev) =>
          prev.map((c) => {
            if (c.id !== finalCid) return c;
            const msgs = [...c.messages];
            const last = msgs[msgs.length - 1];
            if (last && last.role === "assistant") {
              msgs[msgs.length - 1] = { ...last, thinking: (last.thinking ?? "") + reasoning };
            }
            return { ...c, messages: msgs };
          })
        );
      },
      (usage) => {
        timing.usage = usage;
        timing.completion += usage.completionTokens;
        // The exact prompt count lands here; refine the prefill speed. Decode
        // stays null — the settlement publishes it as a streaming-time average.
        const first = timing.tFirst;
        if (first != null) {
          setSpeeds((s) => ({
            prefill: (usage.promptTokens / Math.max(1, first - timing.t0)) * 1000,
            decode: s?.decode ?? null,
          }));
        }
        setConversations((prev) =>
          prev.map((c) => {
            if (c.id !== finalCid) return c;
            const msgs = [...c.messages];
            const last = msgs[msgs.length - 1];
            if (last && last.role === "assistant") {
              const prevUsage = last.meta?.usage;
              msgs[msgs.length - 1] = {
                ...last,
                meta: {
                  ...(last.meta ?? meta),
                  usage: {
                    prompt: (prevUsage?.prompt ?? 0) + usage.promptTokens,
                    completion: (prevUsage?.completion ?? 0) + usage.completionTokens,
                  },
                },
              };
            }
            return {
              ...c,
              messages: msgs,
              inputTokens: c.inputTokens + usage.promptTokens,
              outputTokens: c.outputTokens + usage.completionTokens,
            };
          })
        );
      },
      (message) => {
        setConversations((prev) =>
          prev.map((c) => {
            if (c.id !== finalCid) return c;
            const msgs = [...c.messages];
            const last = msgs[msgs.length - 1];
            msgs[msgs.length - 1] = {
              role: "assistant",
              content: last && last.content ? `${last.content}\n\n⚠ ${message}` : message,
              error: true,
              meta: last?.meta ?? meta,
              thinking: last?.thinking,
            };
            return { ...c, messages: msgs };
          })
        );
      },
      controller.signal,
      toolHooksFor(finalCid, controller.signal)
    );

    abortRef.current = null;
    setStreamingId(null);

    // The stream is over: settle the counter from the exact (or best-guess)
    // counts and stamp it on the exchange so it survives a reload.
    const tEnd = performance.now();
    const prefillSpeed =
      timing.tFirst != null
        ? ((timing.usage?.promptTokens ?? promptEstimate) / Math.max(1, timing.tFirst - timing.t0)) * 1000
        : null;
    // Decode divides by the streaming time alone. A web-tool round's consent
    // wait, tool run and follow-up request are waiting — not generation — so
    // they must not drag the rate down. (When no window was measured, e.g. a
    // provider answering in one JSON body, fall back to the span from the
    // first token.) The numerator is the output of every round together, to
    // match the window that produced it.
    const decodeMs = streamed.streamMs > 0 ? streamed.streamMs : timing.tFirst != null ? tEnd - timing.tFirst : 0;
    const decodeSpeed =
      timing.tFirst != null && decodeMs >= 100
        ? ((timing.completion > 0 ? timing.completion : Math.max(1, Math.ceil(timing.chars / 4))) / decodeMs) * 1000
        : null;
    const speedsStamp =
      prefillSpeed != null
        ? { prefill: Math.round(prefillSpeed), decode: decodeSpeed != null ? Math.round(decodeSpeed) : null }
        : null;
    setSpeeds(speedsStamp);

    const thinkingMs =
      thinkingStartedAt != null
        ? Math.round((thinkingEndAt ?? performance.now()) - thinkingStartedAt)
        : null;
    const parsedReply = splitPlanBlocks(reply);
    const hasFence =
      parsedReply.question != null ||
      parsedReply.plan != null ||
      parsedReply.progress != null ||
      parsedReply.buildTopic != null ||
      parsedReply.edit != null;
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== finalCid) return c;
        const msgs = [...c.messages];
        let last = msgs[msgs.length - 1];
        // A fence-only reply (a bare question or plan) is real work — keep it.
        if (last && last.role === "assistant" && last.content === "" && !last.thinking && !hasFence && !(last.tools && last.tools.length > 0)) {
          return { ...c, messages: msgs.slice(0, -1) };
        }
        // An unclosed or malformed fence must never hide the model's words:
        // fall back to the raw reply (the studio path below keeps its own).
        if (
          last &&
          last.role === "assistant" &&
          (parsedReply.truncated || parsedReply.failed) &&
          !last.studioPending &&
          !last.studioFailed
        ) {
          last = { ...last, content: reply };
          msgs[msgs.length - 1] = last;
        }
        // Stamp the measured speed so the running head and wall keep showing it.
        if (last && last.role === "assistant" && last.meta && speedsStamp) {
          last = { ...last, meta: { ...last.meta, speeds: speedsStamp } };
          msgs[msgs.length - 1] = last;
        }
        if (last && last.role === "assistant" && last.studioPending) {
          // Stream stopped mid-spec: keep the intro, card the failure, stash the raw.
          msgs[msgs.length - 1] = {
            ...last,
            content: splitStudioBlock(reply).visible,
            studioPending: false,
            studioFailed: "truncated",
            studioRaw: reply,
          };
          return { ...c, messages: msgs };
        }
        if (last && last.role === "assistant" && last.thinking && thinkingMs != null) {
          msgs[msgs.length - 1] = { ...last, thinkingMs };
          return { ...c, messages: msgs };
        }
        if (msgs[msgs.length - 1] !== c.messages[c.messages.length - 1]) return { ...c, messages: msgs };
        return c;
      })
    );

    if (needsTitle && !controller.signal.aborted) {
      const visibleReply = splitStudioBlock(reply).visible;
      const titleHistory: ChatMessage[] =
        visibleReply || reply
          ? [...history, { role: "assistant", content: visibleReply || reply }]
          : history;
      const titleController = new AbortController();
      titleAbortRef.current = titleController;
      const title = await generateTitle(llm, titleHistory, titleController.signal);
      titleAbortRef.current = null;
      if (title) {
        setConversations((prev) => prev.map((c) => (c.id === finalCid ? { ...c, title } : c)));
        if (activeIdRef.current === finalCid) {
          setTitleAnimId((id) => id + 1);
        }
      }
    }

    // Post-settle dispatch: the AI either asked a question, delegated a build, or delegated an edit.
    if (!controller.signal.aborted) {
      const { question: parsedQ, buildTopic: parsedBuild, edit: parsedEdit } = splitPlanBlocks(reply);
      // The App Builder gets the conversation that produced the request, not
      // just the one-line brief: the ask, the answers, the constraints.
      const settledVisible = splitStudioBlock(splitPlanBlocks(reply).visible).visible;
      const handoff = handoffContext(
        settledVisible ? [...history, { role: "assistant" as const, content: settledVisible }] : history
      );
      const builder = builderFor(llm, meta);
      if (parsedQ) {
        setConversations((prev) =>
          prev.map((c) => {
            if (c.id !== finalCid) return c;
            const msgs = [...c.messages];
            const last = msgs[msgs.length - 1];
            if (last?.role === "assistant") {
              msgs[msgs.length - 1] = { ...last, question: parsedQ, buildPending: undefined };
            }
            return { ...c, messages: msgs };
          })
        );
      } else if (parsedBuild != null) {
        // A build gets its own thread: the parent keeps the reply text and a
        // card; the plan, confirmation, streaming and critic all happen there.
        const threadId = idRef.current++;
        const thread: Conversation = {
          id: threadId,
          kind: "build",
          parentId: finalCid,
          title: `Build: ${parsedBuild}`,
          messages: [
            {
              role: "user",
              content: "Build this workspace: " + parsedBuild,
              images: userMsg.images?.length ? userMsg.images : undefined,
              at: Date.now(),
            },
            { role: "assistant", content: "", meta: builder.meta, brief: parsedBuild, buildPending: true },
          ],
          context: handoff,
          inputTokens: 0,
          outputTokens: 0,
          createdAt: Date.now(),
        };
        setConversations((prev) => [
          ...prev.map((c) => {
            if (c.id !== finalCid) return c;
            const msgs = [...c.messages];
            const last = msgs[msgs.length - 1];
            if (last?.role === "assistant") {
              msgs[msgs.length - 1] = {
                ...last,
                question: undefined,
                buildPending: undefined,
                brief: undefined,
                building: false,
                threadId,
              };
            }
            return { ...c, messages: msgs };
          }),
          thread,
        ]);
        void launchPlan(threadId, parsedBuild, builder.llm, userMsg.images, handoff);
      } else if (parsedEdit != null) {
        const wanted = parsedEdit.studio.trim().toLowerCase();
        const targetStudio =
          studios.find((s) => s.id === parsedEdit.studioId) ??
          studios.find((s) => s.title === parsedEdit.studio) ??
          studios.find((s) => s.title.trim().toLowerCase() === wanted);
        if (targetStudio) {
          // An edit gets its own thread too — same reasoning as a build.
          const threadId = idRef.current++;
          const thread: Conversation = {
            id: threadId,
            kind: "build",
            parentId: finalCid,
            title: `Edit: ${targetStudio.title}`,
            messages: [
              {
                role: "user",
                content: "Edit this workspace: " + parsedEdit.change,
                images: userMsg.images?.length ? userMsg.images : undefined,
                at: Date.now(),
              },
            { role: "assistant", content: "", meta: builder.meta, brief: parsedEdit.change, editPending: true, editStudioId: targetStudio.id },
          ],
          context: handoff,
          inputTokens: 0,
          outputTokens: 0,
          createdAt: Date.now(),
        };
          setConversations((prev) => [
            ...prev.map((c) => {
              if (c.id !== finalCid) return c;
              const msgs = [...c.messages];
              const last = msgs[msgs.length - 1];
              if (last?.role === "assistant") {
                msgs[msgs.length - 1] = {
                  ...last,
                  question: undefined,
                  editPending: undefined,
                  editStudioId: undefined,
                  brief: undefined,
                  building: false,
                  threadId,
                };
              }
              return { ...c, messages: msgs };
            }),
            thread,
          ]);
          void launchEdit(threadId, targetStudio.id, parsedEdit.change, builder.llm, handoff, targetStudio.spec, userMsg.images, builder.vision);
        } else {
          // The fence named a workspace that no longer exists (or was renamed):
          // surface it instead of silently dropping the turn.
          setConversations((prev) =>
            prev.map((c) => {
              if (c.id !== finalCid) return c;
              const msgs = [...c.messages];
              const last = msgs[msgs.length - 1];
              if (last?.role === "assistant") {
                const note = `⚠ I couldn't find a workspace matching "${parsedEdit.studio}".`;
                msgs[msgs.length - 1] = {
                  ...last,
                  editPending: false,
                  error: true,
                  content: last.content ? `${last.content}\n\n${note}` : note,
                };
              }
              return { ...c, messages: msgs };
            })
          );
        }
      }
    }
  };

  /** Append a builder reasoning delta to the message's build-thinking log. */
  const appendBuildThinking = (cid: number, reasoning: string) => {
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== cid) return c;
        const msgs = [...c.messages];
        const last = msgs[msgs.length - 1];
        if (last?.role === "assistant") {
          msgs[msgs.length - 1] = { ...last, buildThinking: (last.buildThinking ?? "") + reasoning };
        }
        return { ...c, messages: msgs };
      })
    );
  };

  /**
   * Phase A: the planner drafts the TODO list and publishes it on the wall.
   * The build does NOT start here — the plan card waits for the user's
   * explicit go-ahead ("Build it"), which starts launchBuild.
   */
  const launchPlan = async (cid: number, brief: string, llm: LlmConfig, images?: string[], context: ChatMessage[] = []) => {
    const planAbort = new AbortController();
    builderAbortRef.current = planAbort;
    setBuildingId(cid);

    // ── Phase A: the planner produces the TODO list (non-streamed). ──
    let planTodos: { title: string; cot?: string }[] | null = null;
    try {
      const planReply = await completeChat(llm, planPrompt(), "Build this workspace: " + brief, planAbort.signal, images, context);
      const { plan: parsedPlan } = splitPlanBlocks(planReply);
      if (parsedPlan && parsedPlan.todos.length > 0) {
        planTodos = parsedPlan.todos;
      }
    } catch {
      // Aborted or network error — handled below.
    }

    if (builderAbortRef.current === planAbort) builderAbortRef.current = null;
    setBuildingId(null);
    if (planAbort.signal.aborted) {
      // The user stopped the planner: unlock the message so the wall settles.
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== cid) return c;
          const msgs = [...c.messages];
          const last = msgs[msgs.length - 1];
          if (last?.role === "assistant" && last.building) {
            msgs[msgs.length - 1] = { ...last, building: false };
          }
          return { ...c, messages: msgs };
        })
      );
      return;
    }

    if (!planTodos) {
      // Planning failed: the model didn't produce a usable plan.
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== cid) return c;
          const msgs = [...c.messages];
          const last = msgs[msgs.length - 1];
          if (!last || last.role !== "assistant") return c;
          msgs[msgs.length - 1] = {
            ...last,
            building: false,
            studioFailed: "planning" as const,
            studioRaw: "The planner did not return a valid TODO list.",
          };
          return { ...c, messages: msgs };
        })
      );
      return;
    }

    // Publish the plan and park the build until the user confirms it.
    const initialPlan = { todos: planTodos, active: 0, done: 0, pct: null as number | null };
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== cid) return c;
        const msgs = [...c.messages];
        const last = msgs[msgs.length - 1];
        if (last?.role !== "assistant") return c;
        msgs[msgs.length - 1] = { ...last, plan: initialPlan, building: false, awaitingBuild: true };
        return { ...c, messages: msgs };
      })
    );
  };

  /**
   * Phase B: the builder streams progress → studio into the message. Runs
   * only after the user confirms the plan (or retries a failed build).
   */
  const launchBuild = async (
    cid: number,
    brief: string,
    llm: LlmConfig,
    context: ChatMessage[],
    images: string[] | undefined,
    planTodos: { title: string; cot?: string }[],
    vision: boolean
  ) => {
    const builderAbort = new AbortController();
    builderAbortRef.current = builderAbort;
    setBuildingId(cid);

    // The user confirmed: close the gate and mark the build live.
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== cid) return c;
        const msgs = [...c.messages];
        const last = msgs[msgs.length - 1];
        if (last?.role !== "assistant") return c;
        msgs[msgs.length - 1] = { ...last, awaitingBuild: false, building: true };
        return { ...c, messages: msgs };
      })
    );

    // ── Phase B: the builder streams progress → studio into the same message. ──
    const builderUserMsg: ChatMessage = {
      role: "user",
      content: "Build this workspace: " + brief,
      at: Date.now(),
      images: images?.length ? images : undefined,
    };
    const builderSystem = builderPrompt(planTodos);

    let builderReply = "";
    let builderStudioClaimed = false;
    let newStudio: Studio | null = null;
    let lastPct = 0;

    await streamChat(
      llm,
      [...context, builderUserMsg],
      builderSystem,
      (delta) => {
        builderReply += delta;
        const { visible, plan, progress, question, buildTopic } = splitPlanBlocks(builderReply);
        void question; void buildTopic;
        const split = splitStudioBlock(visible);
        let newStudioId: string | undefined;
        if (split.block != null && !builderStudioClaimed) {
          builderStudioClaimed = true;
          const doc = parseStudioBlock(split.block);
          if (doc) {
            const studio: Studio = {
              id: uid(),
              title: doc.title,
              spec: doc.spec,
              conversationId: cid,
              createdAt: Date.now(),
            };
            newStudioId = studio.id;
            newStudio = studio;
            setStudios((prev) => [...prev, studio]);
            // The thread knows which workspace it produced (drives its card).
            setConversations((prev) => prev.map((c) => (c.id === cid ? { ...c, studioId: studio.id } : c)));
          }
        }
        // Clamp pct: monotonic, never backward; 100 only once the studio lands.
        if (plan && plan.pct != null) {
          const clamped = Math.max(plan.pct, lastPct);
          lastPct = builderStudioClaimed ? 100 : Math.min(clamped, 99);
          plan.pct = lastPct;
        }
        if (builderStudioClaimed) {
          if (plan) plan.pct = 100;
        }
        setConversations((prev) =>
          prev.map((c) => {
            if (c.id !== cid) return c;
            const msgs = [...c.messages];
            const last = msgs[msgs.length - 1];
            if (last?.role !== "assistant") return c;
            msgs[msgs.length - 1] = {
              ...last,
              plan: mergeProgressPlan(plan, progress, last.plan),
              studioId: newStudioId ?? last.studioId,
              studioPending: split.open,
              studioFailed: split.failed,
              studioRaw: split.failed ? builderReply : last.studioRaw,
              building: true,
              raw: ((last.raw ?? "") + delta).slice(-RAW_STREAM_MAX),
            };
            return { ...c, messages: msgs };
          })
        );
      },
      (reasoning) => {
        appendBuildThinking(cid, reasoning);
      },
      (usage) => {
        setConversations((prev) =>
          prev.map((c) => {
            if (c.id !== cid) return c;
            const msgs = [...c.messages];
            const last = msgs[msgs.length - 1];
            if (last?.role === "assistant" && last.meta) {
              msgs[msgs.length - 1] = {
                ...last,
                meta: {
                  ...last.meta,
                  usage: {
                    prompt: (last.meta.usage?.prompt ?? 0) + usage.promptTokens,
                    completion: (last.meta.usage?.completion ?? 0) + usage.completionTokens,
                  },
                },
              };
            }
            return {
              ...c,
              messages: msgs,
              inputTokens: c.inputTokens + usage.promptTokens,
              outputTokens: c.outputTokens + usage.completionTokens,
            };
          })
        );
      },
      (message) => {
        setConversations((prev) =>
          prev.map((c) => {
            if (c.id !== cid) return c;
            const msgs = [...c.messages];
            const last = msgs[msgs.length - 1];
            if (last?.role === "assistant") {
              msgs[msgs.length - 1] = {
                ...last,
                content: last.content ? `${last.content}\n\n⚠ ${message}` : message,
                error: true,
                building: false,
              };
            }
            return { ...c, messages: msgs };
          })
        );
      },
      builderAbort.signal,
      toolHooksFor(cid, builderAbort.signal)
    );

    builderAbortRef.current = null;
    setBuildingId(null);

    // Builder settled: finalize the message.
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== cid) return c;
        const msgs = [...c.messages];
        const last = msgs[msgs.length - 1];
        if (!last || last.role !== "assistant") return c;
        const updated = { ...last, building: false };
        if (last.studioPending && !last.studioFailed) {
          updated.studioPending = false;
          updated.studioFailed = "truncated" as const;
          updated.studioRaw = builderReply;
        }
        if (updated.studioId && updated.plan) {
          updated.plan = { ...updated.plan, pct: 100, done: updated.plan.todos.length, active: updated.plan.todos.length };
        }
        msgs[msgs.length - 1] = updated;
        return { ...c, messages: msgs };
      })
    );

    // The workspace is on the wall: the App Builder reviews its own render
    // (screenshots + measured layout facts), then the independent critic gates
    // the result. Both are skipped when the build was aborted or never
    // produced a spec.
    if (newStudio && !builderAbort.signal.aborted) {
      const { studio: reviewed, visionOk } = await runVisualReview(cid, newStudio, llm, vision);
      if (!builderAbort.signal.aborted) await runDesignReview(cid, reviewed, llm, context, visionOk);
    }
  };

  /** The user pressed "Build it" on a plan card: start the streamed build. */
  const confirmBuild = () => {
    if (builderAbortRef.current) return;
    const conv = conversations.find((c) => c.id === activeId);
    if (!conv) return;
    const pending = [...conv.messages].reverse().find((m) => m.role === "assistant" && m.awaitingBuild);
    if (!pending?.brief || !pending.plan) return;
    const agent = activeAgent;
    const provider = modelProvider;
    if (!agent || !provider || !modelId || !providerUsable(provider)) return;
    const llm = llmFor(provider, modelId, effectiveEffort);
    const meta: TurnMeta = { agentId: agent.id, agentName: agent.name, model: modelId, effort: effectiveEffort };
    const builder = builderFor(llm, meta);
    const lastUser = [...conv.messages].reverse().find((m) => m.role === "user");
    void launchBuild(conv.id, pending.brief, builder.llm, conv.context ?? [], lastUser?.images, pending.plan.todos, builder.vision);
  };

  /** Phase A (planning) + Phase B (building): two LLM calls for an edit to an existing workspace. */
  const launchEdit = async (cid: number, studioId: string, change: string, llm: LlmConfig, context: ChatMessage[], existingSpec: StudioSpec, images?: string[], vision = true) => {
    const builderAbort = new AbortController();
    builderAbortRef.current = builderAbort;
    setBuildingId(cid);

    // ── Phase A: the planner produces the edit TODO list (non-streamed). ──
    let planTodos: { title: string; cot?: string }[] | null = null;
    try {
      const planReply = await completeChat(llm, planPrompt({ title: existingSpec.title, change }), "Edit this workspace: " + change, builderAbort.signal, images);
      const { plan: parsedPlan } = splitPlanBlocks(planReply);
      if (parsedPlan && parsedPlan.todos.length > 0) {
        planTodos = parsedPlan.todos;
      }
    } catch {
      if (builderAbort.signal.aborted) {
        return;
      }
    }

    if (!planTodos) {
      builderAbortRef.current = null;
      setBuildingId(null);
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== cid) return c;
          const msgs = [...c.messages];
          const last = msgs[msgs.length - 1];
          if (!last || last.role !== "assistant") return c;
          msgs[msgs.length - 1] = {
            ...last,
            building: false,
            studioFailed: "planning" as const,
            studioRaw: "The planner did not return a valid TODO list.",
          };
          return { ...c, messages: msgs };
        })
      );
      return;
    }

    const initialPlan = { todos: planTodos, active: 0, done: 0, pct: null as number | null };
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== cid) return c;
        const msgs = [...c.messages];
        const last = msgs[msgs.length - 1];
        if (last?.role !== "assistant") return c;
        msgs[msgs.length - 1] = { ...last, plan: initialPlan, building: true };
        return { ...c, messages: msgs };
      })
    );

    // ── Phase B: the builder streams progress → updated spec. ──
    const studioLogs = studios.find((s) => s.id === studioId)?.logs;
    const builderSystem = builderEditPrompt(existingSpec, change, planTodos, studioLogs);
    const editUserMsg: ChatMessage = {
      role: "user",
      content: "Apply this edit: " + change,
      at: Date.now(),
      images: images?.length ? images : undefined,
    };

    let builderReply = "";
    let editStudioClaimed = false;
    let editedStudio: Studio | null = null;
    let lastPct = 0;

    await streamChat(
      llm,
      [...context, editUserMsg],
      builderSystem,
      (delta) => {
        builderReply += delta;
        const { visible, plan, progress } = splitPlanBlocks(builderReply);
        const split = splitStudioBlock(visible);
        if (split.block != null && !editStudioClaimed) {
          editStudioClaimed = true;
          const doc = parseStudioBlock(split.block);
          if (doc) {
            editedStudio = { id: studioId, title: doc.title, spec: doc.spec, conversationId: cid, createdAt: Date.now() };
            setStudios((prev) =>
              prev.map((s) => (s.id === studioId ? { ...s, spec: doc.spec, title: doc.title } : s))
            );
            setConversations((prev) => prev.map((c) => (c.id === cid ? { ...c, studioId } : c)));
          }
        }
        if (plan && plan.pct != null) {
          const clamped = Math.max(plan.pct, lastPct);
          lastPct = editStudioClaimed ? 100 : Math.min(clamped, 99);
          plan.pct = lastPct;
        }
        if (editStudioClaimed) {
          if (plan) plan.pct = 100;
        }
        setConversations((prev) =>
          prev.map((c) => {
            if (c.id !== cid) return c;
            const msgs = [...c.messages];
            const last = msgs[msgs.length - 1];
            if (last?.role !== "assistant") return c;
            msgs[msgs.length - 1] = {
              ...last,
              plan: mergeProgressPlan(plan, progress, last.plan),
              studioId,
              studioPending: split.open,
              studioFailed: split.failed,
              studioRaw: split.failed ? builderReply : last.studioRaw,
              building: true,
              raw: ((last.raw ?? "") + delta).slice(-RAW_STREAM_MAX),
            };
            return { ...c, messages: msgs };
          })
        );
      },
      (reasoning) => {
        appendBuildThinking(cid, reasoning);
      },
      (usage) => {
        setConversations((prev) =>
          prev.map((c) => {
            if (c.id !== cid) return c;
            const msgs = [...c.messages];
            const last = msgs[msgs.length - 1];
            if (last?.role === "assistant" && last.meta) {
              msgs[msgs.length - 1] = {
                ...last,
                meta: {
                  ...last.meta,
                  usage: {
                    prompt: (last.meta.usage?.prompt ?? 0) + usage.promptTokens,
                    completion: (last.meta.usage?.completion ?? 0) + usage.completionTokens,
                  },
                },
              };
            }
            return {
              ...c,
              messages: msgs,
              inputTokens: c.inputTokens + usage.promptTokens,
              outputTokens: c.outputTokens + usage.completionTokens,
            };
          })
        );
      },
      (message) => {
        setConversations((prev) =>
          prev.map((c) => {
            if (c.id !== cid) return c;
            const msgs = [...c.messages];
            const last = msgs[msgs.length - 1];
            if (last?.role === "assistant") {
              msgs[msgs.length - 1] = {
                ...last,
                content: last.content ? `${last.content}\n\n⚠ ${message}` : message,
                error: true,
                building: false,
              };
            }
            return { ...c, messages: msgs };
          })
        );
      },
      builderAbort.signal,
      toolHooksFor(cid, builderAbort.signal)
    );

    builderAbortRef.current = null;
    setBuildingId(null);

    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== cid) return c;
        const msgs = [...c.messages];
        const last = msgs[msgs.length - 1];
        if (!last || last.role !== "assistant") return c;
        const updated = { ...last, building: false };
        if (last.studioPending && !last.studioFailed) {
          updated.studioPending = false;
          updated.studioFailed = "truncated" as const;
          updated.studioRaw = builderReply;
        }
        if (updated.plan) {
          updated.plan = { ...updated.plan, pct: 100, done: updated.plan.todos.length, active: updated.plan.todos.length };
        }
        msgs[msgs.length - 1] = updated;
        return { ...c, messages: msgs };
      })
    );

    // The edited workspace is on the wall: the builder reviews its own render,
    // then the independent critic gates the result.
    if (editedStudio && !builderAbort.signal.aborted) {
      const { studio: reviewed, visionOk } = await runVisualReview(cid, editedStudio, llm, vision);
      if (!builderAbort.signal.aborted) await runDesignReview(cid, reviewed, llm, context, visionOk);
    }
  };

  /** The App Builder's live visual-review phase on the thread's message. */
  const setReview = (cid: number, review: ChatMessage["review"]) => {
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== cid) return c;
        const msgs = [...c.messages];
        const last = msgs[msgs.length - 1];
        if (last?.role === "assistant") {
          msgs[msgs.length - 1] = { ...last, review };
        }
        return { ...c, messages: msgs };
      })
    );
  };

  /**
   * The App Builder's visual review loop: render every screen offscreen, show
   * the builder the screenshots + the measured layout facts, and let it rewrite
   * the spec — up to MAX_REVIEW_PASSES times. Fail-open: without vision (or on
   * a capture/network failure) the loop ends and the deterministic lint review
   * still runs.
   */
  const runVisualReview = async (
    cid: number,
    initial: Studio,
    llm: LlmConfig,
    vision: boolean
  ): Promise<{ studio: Studio; visionOk: boolean }> => {
    if (!vision) return { studio: initial, visionOk: false };
    const controller = new AbortController();
    builderAbortRef.current = controller;
    setBuildingId(cid);
    let studio = initial;
    try {
      for (let pass = 1; pass <= MAX_REVIEW_PASSES; pass++) {
        setReview(cid, { phase: "rendering", pass, passes: MAX_REVIEW_PASSES });
        const capture = await captureStudioSet(studio.spec);
        if (controller.signal.aborted) return { studio, visionOk: true };
        if (!capture.images.length) return { studio, visionOk: true };

        setReview(cid, { phase: "reviewing", pass, passes: MAX_REVIEW_PASSES });
        let reply: string;
        try {
          reply = await completeChat(
            llm,
            workspaceReviewPrompt(studio.spec, pass, MAX_REVIEW_PASSES, capture.diagnostics),
            "Review this rendered app.",
            controller.signal,
            capture.images
          );
        } catch {
          // Provider without vision, network failure, or abort: stop reviewing
          // and let the critic fall back to the deterministic lint.
          return { studio, visionOk: false };
        }
        if (/^\s*approved\b/i.test(reply)) return { studio, visionOk: true };

        // The builder rewrote the spec: apply it in place and re-render.
        const split = splitStudioBlock(reply);
        const doc = split.block != null ? parseStudioBlock(split.block) : null;
        if (!doc) return { studio, visionOk: true };
        setReview(cid, { phase: "fixing", pass, passes: MAX_REVIEW_PASSES });
        studio = { ...studio, title: doc.title, spec: doc.spec };
        setStudios((prev) => prev.map((s) => (s.id === studio.id ? { ...s, spec: doc.spec, title: doc.title } : s)));
      }
      return { studio, visionOk: true };
    } finally {
      setReview(cid, undefined);
      if (builderAbortRef.current === controller) builderAbortRef.current = null;
      setBuildingId(null);
    }
  };

  /**
   * The automatic design critic: the independent final gate. The App Builder
   * already reviewed its own render; the critic looks at the same contact sheet
   * as a fresh set of eyes and — when it reports defects (or vision is
   * unavailable and the deterministic lint does) — runs exactly one polish edit
   * on the spec in place. That polish now sees the screenshots it is fixing.
   * Fail-open: a capture or critic failure never touches the build.
   */
  const runDesignReview = async (cid: number, studio: Studio, llm: LlmConfig, context: ChatMessage[], vision: boolean) => {
    const controller = new AbortController();
    builderAbortRef.current = controller;
    setBuildingId(cid);
    // Keep the build's output on screen; just relabel the phase.

    const setCritic = (critic: ChatMessage["critic"]) => {
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== cid) return c;
          const msgs = [...c.messages];
          const last = msgs[msgs.length - 1];
          if (last?.role === "assistant") {
            msgs[msgs.length - 1] = { ...last, critic };
          }
          return { ...c, messages: msgs };
        })
      );
    };

    try {
      setCritic({ status: "reviewing" });

      const lint = lintSpec(studio.spec);
      let findings: string[] = [];
      const bootLogs: StudioLogInput[] = [];
      const capture = await captureStudioSet(studio.spec, (entry) => bootLogs.push(entry));

      if (vision && capture.images.length) {
        try {
          const reply = await completeChat(llm, criticPrompt(), "Review this rendered workspace.", controller.signal, capture.images);
          const trimmed = reply.trim();
          if (/^approved\b/i.test(trimmed)) {
            setCritic({ status: "approved" });
            return;
          }
          findings = trimmed
            .split("\n")
            .map((line) => line.replace(/^[\s\-*]+/, "").trim())
            .filter(Boolean)
            .slice(0, 5);
        } catch {
          // Provider without vision, network failure, or abort: fall through
          // to the deterministic lint review below.
        }
      }
      if (controller.signal.aborted) {
        setCritic({ status: "skipped" });
        return;
      }
      if (!findings.length) findings = lint;
      // Measured layout facts (overflow, clipped text, an empty screen) are
      // real defects too — prepend them so the polish acts on what broke.
      findings = [...capture.diagnostics, ...findings].slice(0, 5);
      // Startup errors from the offscreen boot are real defects too.
      const bootErrors = bootLogs
        .filter((entry) => entry.level === "error")
        .map((entry) => `Runtime error on load: ${entry.message}`);
      if (bootErrors.length) findings = [...bootErrors, ...findings].slice(0, 5);
      if (!findings.length) {
        setCritic({ status: "approved" });
        return;
      }

      setCritic({ status: "refining", issues: findings });
      const todos = findings.map((finding) => ({
        title: finding.replace(/^\d+[.)]\s*/, "").slice(0, 80) || "Fix a design issue",
      }));

      let replyText = "";
      let refined: { title: string; spec: StudioSpec } | null = null;
      const system = builderEditPrompt(studio.spec, "Design review findings:\n" + findings.join("\n"), todos);
      const userMsg: ChatMessage = {
        role: "user",
        content: "Polish this workspace: fix the design review findings.",
        at: Date.now(),
        images: capture.images.length ? capture.images : undefined,
      };

      await streamChat(
        llm,
        [...context, userMsg],
        system,
        (delta) => {
          replyText += delta;
          setConversations((prev) =>
            prev.map((c) => {
              if (c.id !== cid) return c;
              const msgs = [...c.messages];
              const last = msgs[msgs.length - 1];
              if (last?.role === "assistant") {
                msgs[msgs.length - 1] = { ...last, raw: ((last.raw ?? "") + delta).slice(-RAW_STREAM_MAX) };
              }
              return { ...c, messages: msgs };
            })
          );
          const { visible } = splitPlanBlocks(replyText);
          const split = splitStudioBlock(visible);
          if (split.block != null && refined == null) {
            const doc = parseStudioBlock(split.block);
            if (doc) {
              refined = doc;
              setStudios((prev) =>
                prev.map((s) => (s.id === studio.id ? { ...s, spec: doc.spec, title: doc.title } : s))
              );
            }
          }
        },
        (reasoning) => {
          appendBuildThinking(cid, reasoning);
        },
        (usage) => {
          setConversations((prev) =>
            prev.map((c) => {
              if (c.id !== cid) return c;
              const msgs = [...c.messages];
              const last = msgs[msgs.length - 1];
              if (last?.role === "assistant" && last.meta) {
                msgs[msgs.length - 1] = {
                  ...last,
                  meta: {
                    ...last.meta,
                    usage: {
                      prompt: (last.meta.usage?.prompt ?? 0) + usage.promptTokens,
                      completion: (last.meta.usage?.completion ?? 0) + usage.completionTokens,
                    },
                  },
                };
              }
              return {
                ...c,
                messages: msgs,
                inputTokens: c.inputTokens + usage.promptTokens,
                outputTokens: c.outputTokens + usage.completionTokens,
              };
            })
          );
        },
        () => {
          // A failed polish pass leaves the original spec untouched.
        },
        controller.signal
      );

      setCritic(refined != null ? { status: "refined", issues: findings } : { status: "skipped", issues: findings });
    } finally {
      if (builderAbortRef.current === controller) builderAbortRef.current = null;
      setBuildingId(null);
    }
  };

  /** Re-send the active conversation's last prompt (a workspace that failed to build). */
  const retryTurn = () => {
    const conv = conversations.find((c) => c.id === activeId);
    if (!conv) return;
    const lastAssistant = [...conv.messages].reverse().find((m) => m.role === "assistant");
    const lastUser = [...conv.messages].reverse().find((m) => m.role === "user");
    // Build threads retry through the App Builder (the agent that owns them).
    const agent = activeAgent;
    const provider = modelProvider;
    if (!agent || !provider || !modelId || !providerUsable(provider)) return;
    const llm = llmFor(provider, modelId, effectiveEffort);
    const meta: TurnMeta = { agentId: agent.id, agentName: agent.name, model: modelId, effort: effectiveEffort };
    const builder = builderFor(llm, meta);
    // If the failure was in the builder phase, re-launch it from the stored brief.
    if (lastAssistant?.buildPending && lastAssistant.brief) {
      const clean = conv.messages.filter((m) => !m.error);
      if (lastAssistant.plan && (lastAssistant.awaitingBuild || lastAssistant.studioFailed === "truncated")) {
        // The plan exists (or was confirmed): retry starts Phase B directly.
        void launchBuild(conv.id, lastAssistant.brief, builder.llm, [...(conv.context ?? []), ...clean], lastUser?.images, lastAssistant.plan.todos, builder.vision);
      } else {
        // No usable plan yet — re-run the planner.
        void launchPlan(conv.id, lastAssistant.brief, builder.llm, lastUser?.images, conv.context ?? []);
      }
      return;
    }
    // If the failure was in the edit phase, re-launch the edit.
    if (lastAssistant?.editPending && lastAssistant.editStudioId && lastAssistant.brief) {
      const targetStudio = studios.find((s) => s.id === lastAssistant.editStudioId);
      if (targetStudio) {
        const clean = conv.messages.filter((m) => !m.error);
        void launchEdit(conv.id, targetStudio.id, lastAssistant.brief, builder.llm, [...(conv.context ?? []), ...clean, { role: "assistant", content: lastAssistant.content }], targetStudio.spec, lastUser?.images, builder.vision);
        return;
      }
    }
    if (lastUser?.content) void send(lastUser.content, lastUser.images ?? []);
  };

  // What the next send would ship: the same estimate send() uses for its live
  // prefill counter (system prompt + full history at chars/4, a flat fee per
  // image). It grows as the conversation does.
  const nextTurnTokens = useMemo(() => {
    if (!active || !activeAgent) return 0;
    const systemPrompt = buildAgentSystemPrompt(activeAgent, appCode || undefined, studios.map((s) => ({ title: s.title, id: s.id })));
    const clean = active.messages.filter((m) => !m.error);
    const images = clean.reduce((n, m) => n + (m.images?.length ?? 0), 0);
    return (
      3 +
      Math.ceil(systemPrompt.length / 4) +
      images * 100 +
      Math.ceil(clean.reduce((n, m) => n + m.content.length, 0) / 4)
    );
  }, [active, activeAgent, appCode, studios]);

  // The running head shows the live counter while the active conversation is
  // streaming, and the last measured exchange's speed when idle.
  const headSpeeds = useMemo(() => {
    if (!active) return null;
    if (streamingId === active.id) return speeds;
    for (let i = active.messages.length - 1; i >= 0; i--) {
      const s = active.messages[i].meta?.speeds;
      if (s) return s;
    }
    return null;
  }, [active, streamingId, speeds]);

  // Build threads, projected for the parent conversation's cards: title, live
  // status and whether work is happening right now.
  const threadCards = useMemo(() => {
    return conversations
      .filter((c) => c.kind === "build")
      .map((c) => {
        const last = [...c.messages].reverse().find((m) => m.role === "assistant");
        const studio = c.studioId ? studios.find((s) => s.id === c.studioId) : undefined;
        let status = "Planning the build";
        let live = false;
        if (last?.studioFailed) {
          status = "Failed";
        } else if (last?.review?.phase === "rendering") {
          status = "Rendering the app";
          live = true;
        } else if (last?.review?.phase === "reviewing") {
          status = "Inspecting the screenshots";
          live = true;
        } else if (last?.review?.phase === "fixing") {
          status = "Polishing";
          live = true;
        } else if (last?.critic?.status === "reviewing") {
          status = "Reviewing the design";
          live = true;
        } else if (last?.critic?.status === "refining") {
          status = "Polishing";
          live = true;
        } else if (studio) {
          status = "Done";
        } else if (last?.studioPending) {
          status = "Writing the spec";
          live = true;
        } else if (last?.building) {
          status = last.plan?.pct != null ? `Building · ${last.plan.pct}%` : "Building";
          live = true;
        } else if (last?.awaitingBuild) {
          status = "Plan ready — open to confirm";
        }
        return { id: c.id, title: c.title || "Build", status, active: live };
      });
  }, [conversations, studios]);

  return (
    <div className="app">
      <TopBar
        account={account}
        accountState={accountState}
        onSignIn={() => setAuthMode("signin")}
        onSignUp={() => setAuthMode("signup")}
        onSignOut={signOutUser}
        onOpenSettings={() => setShowSettings(true)}
      />
      <div className="app-frame">
        {active != null && !activeStudio && !inThread && (
          <Sidebar
            conversations={conversations.filter((c) => c.kind !== "build")}
            activeId={activeId}
            activeTypedTitle={typedTitle}
            streamingId={streamingId}
            collapsed={sidebarCollapsed}
            open={sidebarOpen}
            theme={theme}
            onToggleTheme={toggleTheme}
            onSelect={openConversation}
            onNewChat={newChat}
            onDelete={deleteConversation}
            onOpenSearch={() => setShowSwitcher(true)}
            onToggleCollapsed={toggleSidebarCollapsed}
            onClose={() => setSidebarOpen(false)}
          />
        )}
        <div className="main">
          {active != null && !activeStudio && (
            <RunningHead
              title={typedTitle || "New chat"}
              showTitle={!wallTextVisible}
              nextTokens={nextTurnTokens}
              maxTokens={maxTokens}
              speeds={headSpeeds}
              live={streamingId != null && streamingId === active.id}
              onMenu={() => setSidebarOpen((o) => !o)}
              onBack={inThread ? () => openConversation(active.parentId ?? active.id) : undefined}
              onOpenCode={inThread ? undefined : () => setCodeApp("conversation")}
              modelPicker={
                !inThread && activeAgent ? (
                  <AgentSwitcher
                    agents={agentOptions}
                    activeId={activeAgent.id}
                    onPick={setConversationAgent}
                    onManage={() => setShowSettings(true)}
                  />
                ) : undefined
              }
            />
          )}
          {!loaded ? (
            // The boot hasn't settled yet (a slow gateway, a session still
            // resolving). Never render nothing — a blank frame reads as "the
            // UI is gone"; a quiet note holds the spot until the app loads.
            <main className="greeting">
              <div className="greeting-prompt">
                <p className="greeting-sub">Loading…</p>
              </div>
            </main>
          ) : activeStudio ? (
            <StudioView
              key={activeStudio.id}
              studio={activeStudio}
              theme={theme}
              onBack={() => setActiveStudioId(null)}
              logs={activeStudio.logs}
              onLog={(entry) => appendStudioLog(activeStudio.id, entry)}
              onClearLogs={() => clearStudioLogs(activeStudio.id)}
              onFixWithAI={(report) => {
                setActiveStudioId(null);
                void send(`Fix these runtime errors in "${activeStudio.title}":\n\n${report}`, []);
              }}
              aiAllowed={activeStudio.aiAllowed}
              onAllowAI={() =>
                setStudios((prev) => prev.map((s) => (s.id === activeStudio.id ? { ...s, aiAllowed: true } : s)))
              }
              chat={
                active
                  ? {
                      messages: active.messages,
                      streaming: streamingId != null,
                      onSend: (t, imgs) => void send(t, imgs ?? []),
                      sendMode: settings.sendMode,
                    }
                  : undefined
              }
              llm={
                activeAgent
                  ? (() => {
                      return modelProvider && providerUsable(modelProvider) && modelId
                        ? llmFor(modelProvider, modelId, effectiveEffort)
                        : undefined;
                    })()
                  : undefined
              }
            />
          ) : codeApp === "conversation" ? (
            <CodePane
              appId="conversation"
              title={defaultApp("conversation")?.title ?? "Conversation"}
              source={effectiveSource("conversation", appOverrides)}
              edited={appOverrides["conversation"] != null}
              onSave={(s) => saveAppCode("conversation", s)}
              onDiscard={() => discardAppCode("conversation")}
              onClose={() => setCodeApp(null)}
              onAssist={(buffer, instruction) => assistAppCode("conversation", buffer, instruction)}
            />
          ) : active ? (
            <WallFrame
              key={active.id}
              conversation={active}
              title={typedTitle}
              current={current}
              streaming={streamingId != null && streamingId === active.id}
              building={buildingId != null && buildingId === active.id}
              firstTokenReceived={
                streamingId != null && firstTokenCid != null && firstTokenCid === streamingId
              }
              speeds={streamingId != null && streamingId === active.id ? speeds : null}
              sendMode={settings.sendMode}
              sendBlocker={sendBlocker}
              source={effectiveSource("conversation", appOverrides)}
              centered={sidebarCollapsed}
              studios={studios}
              threads={threadCards}
              kind={active.kind ?? "chat"}
              readingSpotFor={readingSpotFor}
              onReadingSpot={onReadingSpot}
              onOpenStudio={setActiveStudioId}
              onOpenThread={openConversation}
              onRetry={retryTurn}
              onBuild={confirmBuild}
              onSend={send}
              onStop={stop}
              onWallTextVisible={setWallTextVisible}
            />
          ) : needsUnlock ? (
            <main className="greeting">
              <div className="greeting-prompt">
                <h1 className="greeting-title">Conversations locked</h1>
                <p className="greeting-sub">
                  This browser no longer holds the key that decrypts your conversations. Enter your access key to unlock them.
                </p>
                <div className="greeting-actions">
                  <button type="button" className="btn" onClick={() => setAuthMode("signin")}>
                    Unlock with your access key
                  </button>
                </div>
              </div>
            </main>
          ) : accountState === "guest" || (accountState === "user" && (guest == null || onboardedThisSessionRef.current)) ? (
            <GreetingFrame
              onStart={startGuest}
              onModelChange={changeModel}
              guest={guest}
              platform={platform}
              canStart={guestCanStart}
              onNewSession={() => newChat()}
              onFirstMessage={(t) => void send(t, [])}
              source={effectiveSource("greeting", appOverrides)}
              onOpenAuth={(mode) => setAuthMode(mode)}
              signedIn={accountState === "user"}
              composer={{
                onSend: send,
                onStop: stop,
                disabled: false,
                sendMode: settings.sendMode,
                sendBlocker,
              }}
            />
          ) : (
            // The session is still resolving (a slow /auth/me, a pending
            // unlock). Never render nothing — a blank frame reads as "the UI
            // is gone"; a quiet note holds the spot until the app settles.
            <main className="greeting">
              <div className="greeting-prompt">
                <p className="greeting-sub">Loading…</p>
              </div>
            </main>
          )}
        </div>
      </div>
      {showSwitcher && (
        <Switcher
          conversations={conversations.filter((c) => c.kind !== "build")}
          activeId={activeId}
          onSelect={openConversation}
          onNew={newChat}
          onClose={() => setShowSwitcher(false)}
        />
      )}
      {showSettings && (
        <SettingsPage
          onClose={() => setShowSettings(false)}
          workspace={workspace}
          settings={settings}
          onSettingsChange={setSettings}
          signedIn={accountState === "user"}
          serverAvailable={platform != null}
          adminActive={account?.role === "admin"}
          onSaveAdmin={saveAdminKeyForAccount}
          providers={agentProviders}
          onSwitch={switchProvider}
          onSaveProvider={saveProvider}
          onRemoveProvider={removeProvider}
          onSaveServer={saveServerDefault}
          serverModels={platform?.models.map((m) => m.id) ?? []}
          agents={workspace.agents}
          defaultAgentId={workspace.activeAgentId ?? CHAT_KITSUI_ID}
          onSaveAgent={saveAgent}
          onRemoveAgent={removeAgent}
          onSetDefaultAgent={setDefaultAgent}
        />
      )}
      {authMode && (
        <AuthDialog
          mode={authMode}
          onClose={() => setAuthMode(null)}
          onDone={completeAuth}
          onSwitchMode={setAuthMode}
          hint={
            needsUnlock
              ? "This browser no longer holds the key that decrypts your conversations. Paste your access key to unlock them."
              : undefined
          }
        />
      )}
      {toolAsk && (
        <div className="ai-consent-backdrop">
          <div className="ai-consent" role="dialog" aria-label="Web tool request">
            <p className="ai-consent-text">
              {toolAsk.name === "web_fetch" ? "Fetch this page?" : "Search the web?"}
            </p>
            <p className="ai-consent-sub">
              {toolAsk.name === "web_fetch"
                ? String(toolAsk.args.url ?? "")
                : String(toolAsk.args.query ?? "")}
            </p>
            <p className="ai-consent-sub">The model asked to use the web — only this call is approved.</p>
            <div className="ai-consent-actions">
              <button type="button" className="btn small ai-consent-allow" onClick={() => decideToolConsent(true)}>
                Allow
              </button>
              <button type="button" className="btn ghost small" onClick={() => decideToolConsent(false)}>
                Not now
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
