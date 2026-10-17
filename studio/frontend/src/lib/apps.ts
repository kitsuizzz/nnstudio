/**
 * The app model: an app is a presentation built from the shared component
 * library, driven by a brain. "default" apps are built-in (a host TypeScript
 * brain); "studio" apps are AI-built workspaces (declarative/logic brain).
 *
 * DEFAULT_APPS is the manifest of built-in apps.
 */

import { greetingSource } from "../components/studio/brains/greeting";
import { wallSource } from "../components/studio/brains/wall";
import type { AppOverrides, AppRef } from "../types";

export interface DefaultAppInfo extends AppRef {
  /** Whether this app needs an active conversation to be meaningful. */
  needsConversation?: boolean;
}

export const DEFAULT_APPS: DefaultAppInfo[] = [
  { id: "greeting", title: "Start", kind: "default" },
  {
    id: "conversation",
    title: "Conversation",
    kind: "default",
    needsConversation: true,
  },
];

/** Look up a built-in app by id. */
export function defaultApp(id: string): DefaultAppInfo | undefined {
  return DEFAULT_APPS.find((a) => a.id === id);
}

/**
 * The shipped source of each code-brain app, keyed by app id. Only the apps
 * that run on a brain (the greeting and the wall) have one.
 */
export const defaultSource: Record<string, string> = {
  greeting: greetingSource,
  conversation: wallSource,
};

/** The app's effective source: the user's Code-pane override if any, else the shipped one. */
export function effectiveSource(appId: string, overrides: AppOverrides): string {
  return overrides[appId] ?? defaultSource[appId] ?? "";
}
