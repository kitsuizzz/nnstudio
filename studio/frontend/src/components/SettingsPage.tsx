import { useEffect, useMemo, useRef, useState } from "react";
import { XIcon } from "./Icon";
import { listModels, type ModelListing } from "../lib/api";
import { uid } from "../lib/id";
import { PROVIDER_PRESETS } from "../lib/providers";
import { PLATFORM_BASE_URL, PLATFORM_NAME, isPlatform } from "../lib/platform";
import { CHAT_KITSUI_NAME, REASONING_EFFORTS } from "../lib/agents";
import { defaultChoice, providerFor } from "../lib/modelchoice";
import type { Agent, ReasoningEffort, SavedProvider, Settings, Workspace } from "../types";

interface SettingsPageProps {
  onClose: () => void;
  /** The workspace, so the box can show the provider currently in use. */
  workspace: Workspace;
  settings: Settings;
  onSettingsChange: (settings: Settings) => void;
  /** The account is signed in (its providers sync to the server profile). */
  signedIn: boolean;
  /** The deployment offers server models (its gateway lists some). */
  serverAvailable: boolean;
  /** The providers in use: the account's saved list, or a guest's session. */
  providers: SavedProvider[];
  /** Ride one of the saved providers (its last model). */
  onSwitch: (provider: SavedProvider) => void;
  /** Add or update one provider, and ride it (it becomes the default). */
  onSaveProvider: (provider: SavedProvider, model: string) => void;
  /** Drop one provider (and the agents riding it). */
  onRemoveProvider: (id: string) => void;
  /** Ride the deployment's server model again (the providers stay saved). */
  onSaveServer: (model: string) => void;
  /** This session has proven the account's admin key (Settings → Admin). */
  adminActive: boolean;
  /** Take the operator's admin key: seal it, store it, prove it. True when
      the server accepted the challenge. */
  onSaveAdmin: (key: string) => Promise<boolean>;
  /** The deployment's server model ids (the Server side of the model lists). */
  serverModels: string[];
  /** Chat Kitsui first, then the user's own agents. */
  agents: Agent[];
  /** The default agent for new conversations. */
  defaultAgentId: string;
  /** Add or update an agent (Chat Kitsui keeps its built-in prompts and capabilities). */
  onSaveAgent: (agent: Agent) => void;
  /** Remove one of the user's agents (Chat Kitsui stays). */
  onRemoveAgent: (agentId: string) => void;
  /** Make an agent the default for new conversations. */
  onSetDefaultAgent: (agentId: string) => void;
}

const EFFORT_LABEL: Record<ReasoningEffort, string> = { low: "Low", medium: "Medium", xhigh: "Extra high" };

/** The Add/Edit provider form's draft. `server: true` = the deployment's own model. */
interface ProviderDraft {
  server: boolean;
  /** The provider preset pick ("Custom" once renamed). */
  pick: string;
  /** The provider's stable id (kept across edits of the same provider). */
  providerId: string;
  name: string;
  baseUrl: string;
  apiKey: string;
  /** This provider does not require an API key. */
  noKey: boolean;
  model: string;
}

/** The agent editor's draft; `id` is empty for a new agent. */
interface AgentEditDraft {
  id: string;
  builtIn: boolean;
  name: string;
  providerId: string;
  model: string;
  reasoningEffort: ReasoningEffort;
  builtins: boolean;
  systemPrompt: string;
}

export default function SettingsPage({
  onClose,
  workspace,
  settings,
  onSettingsChange,
  signedIn,
  serverAvailable,
  providers,
  onSwitch,
  onSaveProvider,
  onRemoveProvider,
  onSaveServer,
  adminActive,
  onSaveAdmin,
  serverModels,
  agents,
  defaultAgentId,
  onSaveAgent,
  onRemoveAgent,
  onSetDefaultAgent,
}: SettingsPageProps) {
  /** The provider in use right now (workspace default → agent). */
  const currentDefault = defaultChoice(workspace);
  const defaultProvider = providerFor(workspace, currentDefault);
  const currentName =
    defaultProvider && isPlatform(defaultProvider)
      ? `${PLATFORM_NAME} (Server provided)`
      : defaultProvider?.name ?? "No provider";

  // --- The provider box: one control, its menu, the Add/Edit form ----------
  const [menuOpen, setMenuOpen] = useState(false);
  const boxRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!menuOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMenuOpen(false);
    };
    const onDown = (e: PointerEvent) => {
      if (!boxRef.current?.contains(e.target as Node)) setMenuOpen(false);
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onDown);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onDown);
    };
  }, [menuOpen]);

  const [draft, setDraft] = useState<ProviderDraft | null>(null);
  const patchDraft = (patch: Partial<ProviderDraft>) => setDraft((d) => (d ? { ...d, ...patch } : d));

  // --- Agents: the list, and the editor for one agent ----------------------
  const [agentDraft, setAgentDraft] = useState<AgentEditDraft | null>(null);
  const patchAgent = (patch: Partial<AgentEditDraft>) => setAgentDraft((d) => (d ? { ...d, ...patch } : d));
  /** The model ids a provider offers: the server's listing, or the provider's own list. */
  const modelsFor = (providerId: string): string[] => {
    const provider = workspace.providers.find((p) => p.id === providerId);
    if (!provider) return [];
    return isPlatform(provider) ? serverModels : provider.models ?? [];
  };

  // --- Admin: the key that answers the server's challenge -----------------
  const [adminDraft, setAdminDraft] = useState("");
  const [adminBusy, setAdminBusy] = useState(false);
  const [adminMsg, setAdminMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const saveAdmin = async () => {
    const key = adminDraft.trim();
    if (!key || adminBusy) return;
    setAdminBusy(true);
    const ok = await onSaveAdmin(key);
    setAdminBusy(false);
    setAdminDraft(ok ? "" : adminDraft);
    setAdminMsg(
      ok
        ? { ok: true, text: "Verified — admin access is active for this account." }
        : { ok: false, text: "That key doesn't match this account." }
    );
  };

  /** The listing the model picker shows: live from the provider's /models. */
  const [listing, setListing] = useState<ModelListing | null>(null);
  /** Bumped by Retry to re-run the listing request. */
  const [listingRun, setListingRun] = useState(0);
  const [query, setQuery] = useState("");

  const listUrl = draft ? (draft.server ? PLATFORM_BASE_URL : draft.baseUrl.trim()) : "";
  const listKey = draft && !draft.server && !draft.noKey ? draft.apiKey.trim() : "";

  useEffect(() => {
    if (!draft || !listUrl) {
      setListing(null);
      return;
    }
    let cancelled = false;
    setListing(null);
    const t = setTimeout(() => {
      void listModels(listUrl, listKey).then((result) => {
        if (!cancelled) setListing(result);
      });
    }, 400);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [draft, listUrl, listKey, listingRun]);

  const filtered = useMemo(() => {
    const all = listing?.ok ? listing.models : [];
    const q = query.trim().toLowerCase();
    return q ? all.filter((m) => m.id.toLowerCase().includes(q)) : all;
  }, [listing, query]);

  const openAdd = () => {
    setAgentDraft(null);
    const preset = PROVIDER_PRESETS[0];
    setQuery("");
    setListingRun(0);
    setMenuOpen(false);
    setDraft({
      server: false,
      pick: preset.name,
      providerId: uid(),
      name: preset.name,
      baseUrl: preset.baseUrl,
      apiKey: "",
      noKey: false,
      model: "",
    });
  };

  const openEdit = (provider: SavedProvider) => {
    setAgentDraft(null);
    setQuery("");
    setListingRun(0);
    setMenuOpen(false);
    setDraft({
      server: false,
      pick: PROVIDER_PRESETS.find((p) => p.name === provider.name)?.name ?? "Custom",
      providerId: provider.id ?? uid(),
      name: provider.name,
      baseUrl: provider.baseUrl,
      apiKey: provider.apiKey,
      noKey: provider.noKey === true,
      model: provider.model,
    });
  };

  const chooseProvider = (value: string) => {
    setQuery("");
    if (value === "platform") {
      setDraft((d) =>
        d
          ? {
              ...d,
              server: true,
              pick: "platform",
              name: PLATFORM_NAME,
              baseUrl: PLATFORM_BASE_URL,
              apiKey: "",
              noKey: true,
              model: "",
            }
          : d
      );
      return;
    }
    const preset = PROVIDER_PRESETS.find((p) => p.name === value) ?? PROVIDER_PRESETS[PROVIDER_PRESETS.length - 1];
    setDraft((d) =>
      d
        ? {
            ...d,
            server: false,
            pick: preset.name,
            name: preset.name,
            // A custom provider starts from a blank URL (the server's is not
            // its own); a preset knows its own.
            baseUrl: preset.baseUrl || (d.server ? "" : d.baseUrl),
            // Most providers want a key; the checkbox says so explicitly.
            noKey: false,
            model: "",
          }
        : d
    );
  };

  const canSave =
    !!draft &&
    !!draft.model &&
    (draft.server || (!!draft.baseUrl.trim() && (draft.noKey || !!draft.apiKey.trim())));

  const save = () => {
    if (!draft || !canSave) return;
    if (draft.server) {
      onSaveServer(draft.model);
    } else {
      const listed = (listing?.ok ? listing.models : []).map((m) => m.id);
      onSaveProvider(
        {
          id: draft.providerId,
          name: draft.name.trim() || "Custom",
          baseUrl: draft.baseUrl.trim(),
          apiKey: draft.noKey ? "" : draft.apiKey.trim(),
          model: draft.model,
          models: listed.includes(draft.model) ? listed : [draft.model, ...listed],
          noKey: draft.noKey,
        },
        draft.model
      );
    }
    setDraft(null);
  };

  const openAgentAdd = () => {
    setDraft(null);
    const providerId = defaultChoice(workspace)?.providerId ?? workspace.providers[0]?.id ?? "";
    setAgentDraft({
      id: "",
      builtIn: false,
      name: "",
      providerId,
      model: modelsFor(providerId)[0] ?? "",
      reasoningEffort: "low",
      builtins: true,
      systemPrompt: "",
    });
  };

  const openAgentEdit = (agent: Agent) => {
    setDraft(null);
    setAgentDraft({
      id: agent.id,
      builtIn: agent.builtIn === true,
      name: agent.name,
      providerId: agent.providerId,
      model: agent.model,
      reasoningEffort: agent.reasoningEffort,
      builtins: agent.builtins,
      systemPrompt: agent.systemPrompt,
    });
  };

  /** Switching provider keeps the model when the new provider offers it, else takes the provider's first. */
  const chooseAgentProvider = (providerId: string) => {
    const models = modelsFor(providerId);
    setAgentDraft((d) => (d ? { ...d, providerId, model: models.includes(d.model) ? d.model : models[0] ?? "" } : d));
  };

  const canSaveAgent =
    !!agentDraft && agentDraft.name.trim() !== "" && !!agentDraft.providerId && !!agentDraft.model;

  const saveAgentDraft = () => {
    if (!agentDraft || !canSaveAgent) return;
    onSaveAgent({
      id: agentDraft.id,
      name: agentDraft.builtIn ? CHAT_KITSUI_NAME : agentDraft.name.trim(),
      providerId: agentDraft.providerId,
      model: agentDraft.model,
      reasoningEffort: agentDraft.reasoningEffort,
      systemPrompt: agentDraft.builtIn ? "" : agentDraft.systemPrompt,
      builtins: agentDraft.builtIn || agentDraft.builtins,
      builtIn: agentDraft.builtIn || undefined,
    });
    setAgentDraft(null);
  };

  /** The editor's model list: the provider's models, plus the agent's own model if the provider no longer lists it. */
  const listedModels = agentDraft ? modelsFor(agentDraft.providerId) : [];
  const agentModelOptions =
    agentDraft?.model && !listedModels.includes(agentDraft.model) ? [agentDraft.model, ...listedModels] : listedModels;

  return (
    <div className="settings-page">
      <header className="settings-header">
        <button className="settings-close" type="button" aria-label="Close settings" onClick={onClose}>
          <XIcon size={15} />
        </button>
        <h2>Settings</h2>
      </header>
      <div className="settings-body">
        <div className="settings-detail">
          <div className="settings-form">
            <h3>Agent Settings</h3>
            {!signedIn && (
              <p className="field-warn" role="note">
                You're not signed in — your changes aren't saved.
              </p>
            )}
            <h4 className="settings-subhead">Provider</h4>
            <div className="provider-box" ref={boxRef}>
              <button
                id="provider-box"
                className="provider-toggle"
                type="button"
                aria-haspopup="listbox"
                aria-expanded={menuOpen}
                onClick={() => setMenuOpen((o) => !o)}
              >
                <span className="provider-toggle-label">{currentName}</span>
                <span className="provider-toggle-caret" aria-hidden="true">
                  ▾
                </span>
              </button>
              {menuOpen && (
                <div className="provider-menu" role="menu" aria-label="Providers">
                  {providers.map((provider) => {
                    const active = !!provider.id && provider.id === defaultProvider?.id && !isPlatform(defaultProvider);
                    return (
                      <div className={`provider-row${active ? " active" : ""}`} key={provider.id ?? provider.name}>
                        <button
                          type="button"
                          className="provider-row-pick"
                          role="menuitemradio"
                          aria-checked={active}
                          onClick={() => {
                            onSwitch(provider);
                            setMenuOpen(false);
                          }}
                        >
                          <span className="provider-row-name">{provider.name}</span>
                          {active && (
                            <span className="provider-check" aria-hidden="true">
                              ✓
                            </span>
                          )}
                        </button>
                        <div className="provider-row-actions">
                          <button type="button" className="btn ghost small" onClick={() => openEdit(provider)}>
                            Edit
                          </button>
                          <button
                            type="button"
                            className="btn ghost small danger"
                            onClick={() => provider.id && onRemoveProvider(provider.id)}
                          >
                            Remove
                          </button>
                        </div>
                      </div>
                    );
                  })}
                  <button type="button" className="provider-add" role="menuitem" onClick={openAdd}>
                    + Add new provider
                  </button>
                </div>
              )}
            </div>
            {draft && (
              <div className="provider-form">
                <div className="field">
                  <span className="field-label">Provider</span>
                  <select id="agent-provider" value={draft.pick} onChange={(e) => chooseProvider(e.target.value)}>
                    {serverAvailable && <option value="platform">{PLATFORM_NAME} (Server provided)</option>}
                    {PROVIDER_PRESETS.map((p) => (
                      <option key={p.name} value={p.name}>
                        {p.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field-row">
                  <div className="field">
                    <span className="field-label">Name</span>
                    <input
                      id="agent-name"
                      type="text"
                      value={draft.name}
                      readOnly={draft.server}
                      placeholder="My provider"
                      onChange={(e) => patchDraft({ name: e.target.value })}
                    />
                  </div>
                  <div className="field">
                    <span className="field-label">Base URL</span>
                    <input
                      id="agent-url"
                      type="url"
                      value={draft.baseUrl}
                      readOnly={draft.server}
                      placeholder="https://api.example.com/v1"
                      onChange={(e) => patchDraft({ baseUrl: e.target.value })}
                    />
                  </div>
                </div>
                {!draft.server && (
                  <div className="field">
                    <span className="field-label">API key</span>
                    <label className="field-check">
                      <input
                        id="agent-nokey"
                        type="checkbox"
                        checked={draft.noKey}
                        onChange={(e) =>
                          patchDraft({ noKey: e.target.checked, apiKey: e.target.checked ? "" : draft.apiKey })
                        }
                      />
                      This provider does not require an API key
                    </label>
                    {!draft.noKey && (
                      <input
                        id="agent-key"
                        type="password"
                        value={draft.apiKey}
                        placeholder="sk-…"
                        spellCheck={false}
                        onChange={(e) => patchDraft({ apiKey: e.target.value })}
                      />
                    )}
                  </div>
                )}
                <div className="field">
                  <span className="field-label">AI model</span>
                  {!listUrl ? null : listing == null ? (
                    <p className="field-hint">Fetching models…</p>
                  ) : listing.ok ? (
                    <>
                      <input
                        id="agent-model-search"
                        className="models-search"
                        type="search"
                        placeholder="Search models…"
                        aria-label="Search models"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                      />
                      <div className="models-list models-picker">
                        {filtered.map((m) => (
                          <button
                            key={m.id}
                            type="button"
                            className={`models-row${m.id === draft.model ? " selected" : ""}`}
                            aria-pressed={m.id === draft.model}
                            onClick={() => patchDraft({ model: m.id })}
                          >
                            <span className="models-row-id">{m.id}</span>
                            {m.vision === true ? <span className="models-tag">image input</span> : null}
                          </button>
                        ))}
                        {filtered.length === 0 && (
                          <div className="models-empty">No model matches “{query}”.</div>
                        )}
                      </div>
                    </>
                  ) : (
                    <div className="models-error" role="alert">
                      <p className="models-error-text">
                        Couldn't fetch the model list from {listUrl}. {listing.error}
                      </p>
                      <button
                        type="button"
                        className="btn ghost small"
                        onClick={() => setListingRun((n) => n + 1)}
                      >
                        Retry
                      </button>
                    </div>
                  )}
                </div>
                <div className="provider-form-actions">
                  <button type="button" className="btn small" disabled={!canSave} onClick={save}>
                    Save provider
                  </button>
                  <button type="button" className="btn ghost small" onClick={() => setDraft(null)}>
                    Cancel
                  </button>
                </div>
              </div>
            )}
            <h4 className="settings-subhead">Agents</h4>
            <p className="field-hint">
              Chat Kitsui runs on the provider above. Add your own agents for other prompts, models or effort, and click a
              row to make it the default.
            </p>
            <div className="models-list">
              {agents.map((agent) => {
                const isDefault = agent.id === defaultAgentId;
                const provider = workspace.providers.find((p) => p.id === agent.providerId) ?? null;
                const providerLabel = provider ? (isPlatform(provider) ? PLATFORM_NAME : provider.name) : "No provider";
                return (
                  <div className={`agent-row${isDefault ? " active" : ""}`} key={agent.id}>
                    <button
                      type="button"
                      className="agent-row-pick"
                      aria-pressed={isDefault}
                      onClick={() => onSetDefaultAgent(agent.id)}
                    >
                      <span className="agent-row-name">
                        {agent.name}
                        {isDefault && <span className="models-tag default">Default</span>}
                      </span>
                      <span className="agent-row-sub">
                        {agent.model || "no model"} · {providerLabel} · {EFFORT_LABEL[agent.reasoningEffort]} effort
                      </span>
                    </button>
                    <div className="provider-row-actions">
                      <button type="button" className="btn ghost small" onClick={() => openAgentEdit(agent)}>
                        Edit
                      </button>
                      {!agent.builtIn && (
                        <button type="button" className="btn ghost small danger" onClick={() => onRemoveAgent(agent.id)}>
                          Remove
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
              <button type="button" className="provider-add" onClick={openAgentAdd}>
                + Add new agent
              </button>
            </div>
            {agentDraft && (
              <div className="provider-form">
                <div className="field-row">
                  <div className="field">
                    <span className="field-label">Name</span>
                    <input
                      id="agent-edit-name"
                      type="text"
                      value={agentDraft.name}
                      readOnly={agentDraft.builtIn}
                      placeholder="My agent"
                      onChange={(e) => patchAgent({ name: e.target.value })}
                    />
                  </div>
                  <div className="field">
                    <span className="field-label">Reasoning effort</span>
                    <select
                      id="agent-edit-effort"
                      value={agentDraft.reasoningEffort}
                      onChange={(e) => patchAgent({ reasoningEffort: e.target.value as ReasoningEffort })}
                    >
                      {REASONING_EFFORTS.map((effort) => (
                        <option key={effort} value={effort}>
                          {EFFORT_LABEL[effort]}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                <div className="field-row">
                  <div className="field">
                    <span className="field-label">Provider</span>
                    <select
                      id="agent-edit-provider"
                      value={agentDraft.providerId}
                      onChange={(e) => chooseAgentProvider(e.target.value)}
                    >
                      {!agentDraft.providerId && <option value="">No provider</option>}
                      {workspace.providers.map((p) => (
                        <option key={p.id} value={p.id}>
                          {isPlatform(p) ? `${PLATFORM_NAME} (Server provided)` : p.name}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="field">
                    <span className="field-label">AI model</span>
                    <select
                      id="agent-edit-model"
                      value={agentDraft.model}
                      disabled={agentModelOptions.length === 0}
                      onChange={(e) => patchAgent({ model: e.target.value })}
                    >
                      {agentModelOptions.length === 0 && <option value="">No models</option>}
                      {agentModelOptions.map((id) => (
                        <option key={id} value={id}>
                          {id}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                {agentDraft.builtIn ? (
                  <p className="field-hint">
                    Chat Kitsui's prompt and built-in capabilities are fixed. Its provider, model and effort are yours to set.
                  </p>
                ) : (
                  <>
                    <label className="field-check">
                      <input
                        id="agent-edit-builtins"
                        type="checkbox"
                        checked={agentDraft.builtins}
                        onChange={(e) => patchAgent({ builtins: e.target.checked })}
                      />
                      Keep the built-in capabilities (app builder, web tools, app guidance)
                    </label>
                    <div className="field">
                      <span className="field-label">System prompt</span>
                      <textarea
                        id="agent-edit-prompt"
                        rows={6}
                        value={agentDraft.systemPrompt}
                        placeholder="How this agent should behave…"
                        spellCheck={false}
                        onChange={(e) => patchAgent({ systemPrompt: e.target.value })}
                      />
                    </div>
                  </>
                )}
                <div className="provider-form-actions">
                  <button
                    id="agent-edit-save"
                    type="button"
                    className="btn small"
                    disabled={!canSaveAgent}
                    onClick={saveAgentDraft}
                  >
                    Save agent
                  </button>
                  <button type="button" className="btn ghost small" onClick={() => setAgentDraft(null)}>
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </div>
          <div className="settings-form">
            <h3>Web tools</h3>
            <p className="field-hint">
              The model can search the web and read pages, and its answers name the sources they used. With asking on,
              each call waits for your approval.
            </p>
            <label className="field-check">
              <input
                type="checkbox"
                checked={settings.webTools.search}
                onChange={(e) =>
                  onSettingsChange({
                    ...settings,
                    webTools: { ...settings.webTools, search: e.target.checked },
                  })
                }
              />
              Web search
            </label>
            <label className="field-check sub">
              <input
                type="checkbox"
                checked={settings.webTools.ask.search}
                disabled={!settings.webTools.search}
                onChange={(e) =>
                  onSettingsChange({
                    ...settings,
                    webTools: { ...settings.webTools, ask: { ...settings.webTools.ask, search: e.target.checked } },
                  })
                }
              />
              Ask before each search
            </label>
            <label className="field-check">
              <input
                type="checkbox"
                checked={settings.webTools.fetch}
                onChange={(e) =>
                  onSettingsChange({
                    ...settings,
                    webTools: { ...settings.webTools, fetch: e.target.checked },
                  })
                }
              />
              Fetch pages
            </label>
            <label className="field-check sub">
              <input
                type="checkbox"
                checked={settings.webTools.ask.fetch}
                disabled={!settings.webTools.fetch}
                onChange={(e) =>
                  onSettingsChange({
                    ...settings,
                    webTools: { ...settings.webTools, ask: { ...settings.webTools.ask, fetch: e.target.checked } },
                  })
                }
              />
              Ask before each page fetch
            </label>
          </div>
          {signedIn && (
            <div className="settings-form">
              <h3>Admin</h3>
              {adminActive && (
                <p className="field-hint">Admin access is active for this account.</p>
              )}
              <div className="field">
                <span className="field-label">Admin key</span>
                <input
                  id="admin-key"
                  type="password"
                  value={adminDraft}
                  placeholder="adm_…"
                  spellCheck={false}
                  onChange={(e) => {
                    setAdminDraft(e.target.value);
                    setAdminMsg(null);
                  }}
                />
              </div>
              {adminMsg && (
                <p className={adminMsg.ok ? "field-hint" : "field-hint admin-error"}>{adminMsg.text}</p>
              )}
              <div className="provider-form-actions">
                <button type="button" className="btn small" disabled={!adminDraft.trim() || adminBusy} onClick={saveAdmin}>
                  Save & verify
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
