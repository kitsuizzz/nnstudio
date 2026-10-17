# Studio Kitsui — the logo

Studio Kitsui is one app with two halves: **Studio** (the canvas) and **Chat**
(AI Studio). This folder is the logo kit.

![Studio Kitsui on the light wall](previews/lockup-light.png)

## The mark

A **K drawn as a network**, in the canvas's own vocabulary:

- The **stem** is a column of ports, drawn as rings: the canvas's inputs.
- The **junction**, solid, is where the two arms leave the stem.
- The **arms** are wires, curved the way the canvas draws them.
- The **red dot** is the one live node. It is the Salon's single accent, and it
  means state, never decoration.

The K is the initial of Kitsui; the network is what the app builds.

## The wordmark

**Studio** is set in Instrument Sans Medium, the tool's voice. **Kitsui** is set
in Newsreader Medium Italic, the work's voice: the serif the agent's words are
set in. Both are SIL Open Font License faces. The files carry the letters as
outlines, so the logo never depends on a font being installed.

## Colour

| theme | wall      | ink       | dot       |
|-------|-----------|-----------|-----------|
| light | `#fafaf7` | `#161616` | `#e5321b` |
| dark  | `#1a1a19` | `#ebebe7` | `#ff5a45` |

The app opens dark: the dark theme is the canvas's home.

## Files

| file | what it is |
|------|------------|
| `studio-kitsui-lockup-light.svg`, `-dark.svg` | the mark and the wordmark, for each wall |
| `studio-kitsui-mark-light.svg`, `-dark.svg` | the mark alone |
| `studio-kitsui-app-icon.svg` | the mark on the dark tile, for app icons |
| `studio-kitsui-favicon.svg` | the favicon: solid ports and heavier wires, so it reads at 16 px |
| `lockup-inline.svg` | the lockup as the pages carry it: classes, so each page's own tokens set the colours |
| `previews/*.png` | renders of the above |
| `tools/make_brand.py` | the source of all of it, and of the copies inside the pages |

## Use

- **Clear space:** half the mark's height on every side.
- **Size:** the lockup at 16 px tall or more. Below that, use the mark alone, or
  the favicon.
- **Do not** recolour the dot except to the theme's red; add shadows, outlines
  or gradients; rotate or stretch the logo.

## Where it appears

- The NN Studio page's header and the handbook (`ui/index.html`, `ui/docs.html`):
  the lockup, the favicon, the page titles.
- The Chat half's top bar (`studio/frontend/src/components/TopBar.tsx`, through
  `BrandLockup.tsx`), its page title and favicon.

## Regenerate

    .venv/bin/python brand/tools/make_brand.py --previews

The first run fetches the two fonts into `brand/tools/cache/` (git-ignored; one
network fetch). The same run rewrites the marked regions of `ui/index.html`,
`ui/docs.html`, `studio/frontend/index.html`, `studio/frontend/src/index.css`
and `studio/frontend/src/components/BrandLockup.tsx`, so every copy of the mark
comes from one definition. `tests/brand_test.py`, part of `tests/run.sh`, fails
when a copy drifts. After a regeneration, rebuild the Chat half
(`VITE_BASE_PATH=chat npm run build` in `studio/frontend`).
