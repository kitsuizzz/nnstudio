/**
 * The product's chrome — the persistent window components that frame the
 * active app.
 *
 * The component library has two tiers:
 *   1. CONTENT components — registry blocks (lib/studio-blocks.ts), driven by
 *      a brain over the closed do(block, event, arg) vocabulary. Usable by the
 *      AI and by the product's default apps (wall, greeting).
 *   2. CHROME components — the window itself. Host-driven with rich callbacks
 *      (navigation, theme, delete, search), so they are registered here as
 *      shared components the Frame composes, rather than content blocks.
 *
 * Everything the user sees is built from one of these two tiers.
 */

import type { ComponentType } from "react";
import Sidebar from "../components/Sidebar";
import RunningHead from "../components/RunningHead";
import Switcher from "../components/Switcher";
import Composer from "../components/Composer";

export interface ChromeComponent {
  id: string;
  name: string;
  blurb: string;
  component: ComponentType<never>;
}

export const CHROME: ChromeComponent[] = [
  {
    id: "sidebar",
    name: "Sidebar",
    blurb: "Conversation list, new chat, search, and the theme toggle.",
    component: Sidebar,
  },
  {
    id: "running-head",
    name: "RunningHead",
    blurb: "The top bar — the title (once it scrolls away) and the token sum.",
    component: RunningHead,
  },
  {
    id: "switcher",
    name: "Switcher",
    blurb: "The ⌘K command palette for jumping between conversations.",
    component: Switcher,
  },
  {
    id: "composer",
    name: "Composer",
    blurb: "The docked message box — in the wall, the greeting, and the chat block.",
    component: Composer,
  },
];
