/**
 * Where the reader left a conversation: at the bottom (following the stream), or
 * on an exchange with an offset into it. The spot is anchored on the exchange,
 * not on a raw pixel position, so it survives the window or the content changing
 * size between visits.
 */

/** Within this many px of the bottom the reader counts as following the stream. */
export const RESUME_PX = 32;

export interface ReadingSpot {
  /** At the bottom: the stream is followed. */
  following: boolean;
  /** The exchange at the top edge of the view (its index among the wall's exchanges). */
  turn?: number;
  /** How far that exchange's top sits above the view's top edge, in px. */
  offset?: number;
  /** The raw scroll position in px: the fallback when the exchange is not there. */
  top?: number;
}

/** Where the scroller is now, as a spot. */
export function spotAt(el: HTMLElement): ReadingSpot {
  const top = el.scrollTop;
  if (el.scrollHeight - top - el.clientHeight <= RESUME_PX) return { following: true };
  const edge = el.getBoundingClientRect().top;
  const works = el.querySelectorAll<HTMLElement>(".work");
  for (let turn = 0; turn < works.length; turn++) {
    const box = works[turn].getBoundingClientRect();
    if (box.bottom > edge + 1) return { following: false, turn, offset: edge - box.top, top };
  }
  return { following: false, top };
}

/** The scroll position that puts a spot back where it was. */
export function targetFor(el: HTMLElement, spot: ReadingSpot): number {
  if (spot.following) return el.scrollHeight;
  const work = spot.turn === undefined ? undefined : el.querySelectorAll<HTMLElement>(".work")[spot.turn];
  if (work) {
    const box = work.getBoundingClientRect();
    // The exchange's place in the scrolled content, plus the reader's offset into it.
    const absTop = box.top - el.getBoundingClientRect().top + el.scrollTop;
    return absTop + (spot.offset ?? 0);
  }
  return spot.top ?? el.scrollTop;
}
