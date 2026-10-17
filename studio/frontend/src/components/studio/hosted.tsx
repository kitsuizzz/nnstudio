/**
 * Glue for hosted (default) apps: a default app owns its window/layout and
 * renders its content as registry blocks through BlocksView/Block. It builds
 * a BlockCtx whose resolve is identity (the brain emits fully-resolved blocks)
 * and whose do() dispatches block events to the host.
 */

import type { BlockCtx, ComposerHost } from "./blocks";
import type { StudioBlock } from "../../types";
import type { BlockEvent } from "./blocks";

export type AppEventHandler = (block: StudioBlock, event: BlockEvent, arg: unknown) => void;

/**
 * Build the BlockCtx a hosted app renders with. Pass the composer host when a
 * `composer` block is docked; pass onEvent to receive block interactions.
 */
export function appCtx(onEvent?: AppEventHandler, composer?: ComposerHost): BlockCtx {
  return {
    data: {},
    resolve: (s) => s,
    do: (block, event, arg) => onEvent?.(block, event, arg),
    composer,
  };
}
