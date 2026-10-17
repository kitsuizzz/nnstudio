#!/usr/bin/env python3
"""Studio Kitsui's logo, from one definition.

    .venv/bin/python brand/tools/make_brand.py             # the kit, and both UIs
    .venv/bin/python brand/tools/make_brand.py --previews  # plus the PNG previews

The mark is a K drawn as a small network, in the canvas's own vocabulary: a
stem of ports (rings), a solid junction where the arms leave it, and the one
red dot — the live node, the Salon's single accent. The wordmark is outlined,
so the logo never depends on a font being installed: "Studio" in Instrument
Sans Medium, "Kitsui" in Newsreader Medium Italic (both SIL OFL), fetched once
from Google Fonts into brand/tools/cache/ (git-ignored).

A run rewrites the marked regions of the NN Studio page, the Chat half's page
and stylesheet, and its React component, so no copy of the mark can drift from
this file. tests/brand_test.py checks it.
"""

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from urllib.parse import quote

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BRAND = os.path.join(ROOT, "brand")
CACHE = os.path.join(BRAND, "tools", "cache")
NNSTUDIO_UI = os.path.join(ROOT, "ui", "index.html")
DOCS = os.path.join(ROOT, "ui", "docs.html")
CHAT = os.path.join(ROOT, "studio", "frontend")
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)

# the Salon palette, both themes
LIGHT = {"wall": "#fafaf7", "ink": "#161616", "dot": "#e5321b"}
DARK = {"wall": "#1a1a19", "ink": "#ebebe7", "dot": "#ff5a45"}

# the mark, in its own units: 45 × 49
MW, MH = 45.0, 49.0
NODE = 4.5  # a port's outer radius
RING_SW, WIRE_SW = 2.4, 3.2  # the ring's stroke, the wires'
RING_R = NODE - RING_SW / 2  # so the ring's outer edge stays at NODE
L, R = 4.5, 40.5  # the stem's column, the arms' column
T, M, B = 4.5, 24.5, 44.5  # the top, middle and bottom rows
STEM = "M4.5 9V40"
UPPER = "M4.5 24.5C20 24.5 25 4.5 40.5 4.5"
LOWER = "M4.5 24.5C20 24.5 25 44.5 36 44.5"  # ends at the ring's edge, not its centre

# the lockup and the app icon
CAP = 27.0  # both words share this cap height
GAP = 12.0  # mark → wordmark
WORD_GAP = 0.22  # "Studio" → "Kitsui", in em of the sans
LOGO_H = 20  # the lockup's height on screen, in px
ICON_SCALE = 0.8  # the mark inside the app icon's tile
FAVICON_SCALE = 0.9  # the favicon's mark is bigger: it must read at 16 px
WIRE_BOLD = 4.6  # and its wires heavier, its ports solid

SANS = {
    "query": "Instrument+Sans:wght@500",
    "italic": False,
    "name": "instrument-sans-500",
    "loc": {"wght": 500},
}
SERIF = {
    "query": "Newsreader:ital,opsz,wght@1,6..72,500",
    "italic": True,
    "name": "newsreader-italic-500",
    "loc": {"wght": 500, "opsz": 36},
}


def fmt(v):
    s = ("%.2f" % v).rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def curl(url, ua=None):
    cmd = ["curl", "-fsSL", "--max-time", "90"]
    if ua:
        cmd += ["-A", ua]
    return subprocess.run(cmd + [url], check=True, capture_output=True).stdout


def fetch_face(face):
    dest = os.path.join(CACHE, face["name"] + ".woff2")
    if not os.path.exists(dest):
        os.makedirs(CACHE, exist_ok=True)
        css = curl(
            "https://fonts.googleapis.com/css2?family=%s&display=block" % face["query"],
            UA,
        ).decode("utf-8")
        for m in re.finditer(r"/\* latin \*/\s*@font-face\s*\{([^}]*)\}", css):
            body = m.group(1)
            if ("font-style: italic" in body) == face["italic"]:
                src = re.search(r"url\((https://[^)]+)\)", body).group(1)
                with open(dest, "wb") as f:
                    f.write(curl(src))
                break
        else:
            raise SystemExit("make_brand: no latin face for %s" % face["query"])
    return dest


def load_face(face):
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer

    font = TTFont(fetch_face(face))
    if "fvar" in font:  # pin the variable font to the weight
        axes = {a.axisTag: a.defaultValue for a in font["fvar"].axes}
        axes.update({k: v for k, v in face["loc"].items() if k in axes})
        font = instancer.instantiateVariableFont(font, axes)
    return font


def shape(font, text):
    """(glyph, x offset, advance) per glyph, in font units: kerned by HarfBuzz
    when uharfbuzz is installed, plain advances when it is not."""
    try:
        import uharfbuzz as hb
    except ImportError:
        cmap, hmtx = font.getBestCmap(), font["hmtx"]
        return [(cmap[ord(c)], 0, hmtx[cmap[ord(c)]][0]) for c in text]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "shape.ttf")
        font.flavor = None  # saved as plain TrueType, which HarfBuzz reads
        font.save(path)
        hbf = hb.Font(hb.Face(hb.Blob.from_file_path(path)))
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(hbf, buf, {"kern": True})
        return [
            (font.getGlyphName(i.codepoint), p.x_offset, p.x_advance)
            for i, p in zip(buf.glyph_infos, buf.glyph_positions)
        ]


def outline(font, text, size, x0, baseline):
    """The text as one SVG path, and the x where it ends."""
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen

    s = size / font["head"].unitsPerEm
    glyphs = font.getGlyphSet()
    pen = SVGPathPen(glyphs, ntos=fmt)
    x = x0
    for name, dx, adv in shape(font, text):
        glyphs[name].draw(TransformPen(pen, (s, 0, 0, -s, x + dx * s, baseline)))
        x += adv * s
    return pen.getCommands(), x


def wordmark(sans, serif):
    base = (MH + CAP) / 2
    s_size = CAP * sans["head"].unitsPerEm / sans["OS/2"].sCapHeight
    k_size = CAP * serif["head"].unitsPerEm / serif["OS/2"].sCapHeight
    d1, x = outline(sans, "Studio", s_size, MW + GAP, base)
    d2, x = outline(serif, "Kitsui", k_size, x + WORD_GAP * s_size, base)
    return d1, d2, x


def svg(view_box, body):
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="%s">%s</svg>\n' % (
        view_box,
        body,
    )


def mark_attr(ink, dot):
    """The mark for the files: colours and strokes as attributes."""
    n, rr, w, rsw = fmt(NODE), fmt(RING_R), fmt(WIRE_SW), fmt(RING_SW)
    l, r, t, m, b = fmt(L), fmt(R), fmt(T), fmt(M), fmt(B)
    return (
        f'<g fill="none" stroke="{ink}" stroke-width="{w}" stroke-linecap="round">'
        f'<path d="{STEM}"/><path d="{UPPER}"/><path d="{LOWER}"/></g>'
        f'<circle cx="{l}" cy="{m}" r="{n}" fill="{ink}"/>'
        f'<g fill="none" stroke="{ink}" stroke-width="{rsw}">'
        f'<circle cx="{l}" cy="{t}" r="{rr}"/><circle cx="{l}" cy="{b}" r="{rr}"/>'
        f'<circle cx="{r}" cy="{b}" r="{rr}"/></g>'
        f'<circle cx="{r}" cy="{t}" r="{n}" fill="{dot}"/>'
    )


def mark_class():
    """The mark for the UIs: classes only, so each page's CSS sets the colours."""
    n, rr = fmt(NODE), fmt(RING_R)
    l, r, t, m, b = fmt(L), fmt(R), fmt(T), fmt(M), fmt(B)
    return (
        f'<g class="brand-mark">'
        f'<path class="brand-wire" d="{STEM}"/><path class="brand-wire" d="{UPPER}"/>'
        f'<path class="brand-wire" d="{LOWER}"/>'
        f'<circle class="brand-node" cx="{l}" cy="{m}" r="{n}"/>'
        f'<circle class="brand-ring" cx="{l}" cy="{t}" r="{rr}"/>'
        f'<circle class="brand-ring" cx="{l}" cy="{b}" r="{rr}"/>'
        f'<circle class="brand-ring" cx="{r}" cy="{b}" r="{rr}"/>'
        f'<circle class="brand-dot" cx="{r}" cy="{t}" r="{n}"/></g>'
    )


def lockup_inline(d1, d2, width):
    return (
        f'<svg class="brand-lockup" viewBox="0 0 {fmt(width)} {fmt(MH)}" role="img" '
        f'aria-label="Studio Kitsui">{mark_class()}'
        f'<path class="brand-word" d="{d1}"/><path class="brand-word" d="{d2}"/></svg>'
    )


def lockup_body(pal, d1, d2):
    return (
        mark_attr(pal["ink"], pal["dot"])
        + f'<path fill="{pal["ink"]}" d="{d1}"/><path fill="{pal["ink"]}" d="{d2}"/>'
    )


def app_icon():
    s = ICON_SCALE
    tx, ty = (64 - MW * s) / 2, (64 - MH * s) / 2
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        f'<rect width="64" height="64" rx="14" fill="{DARK["wall"]}"/>'
        f'<g transform="translate({fmt(tx)} {fmt(ty)}) scale({fmt(s)})">'
        f"{mark_attr(DARK['ink'], DARK['dot'])}</g></svg>\n"
    )


def mark_bold(ink, dot):
    """The favicon's mark: solid ports and heavier wires, so it reads at 16 px."""
    n, w = fmt(NODE), fmt(WIRE_BOLD)
    l, r, t, m, b = fmt(L), fmt(R), fmt(T), fmt(M), fmt(B)
    return (
        f'<g fill="none" stroke="{ink}" stroke-width="{w}" stroke-linecap="round">'
        f'<path d="{STEM}"/><path d="{UPPER}"/><path d="{LOWER}"/></g>'
        f'<circle cx="{l}" cy="{t}" r="{n}" fill="{ink}"/>'
        f'<circle cx="{l}" cy="{m}" r="{n}" fill="{ink}"/>'
        f'<circle cx="{l}" cy="{b}" r="{n}" fill="{ink}"/>'
        f'<circle cx="{r}" cy="{b}" r="{n}" fill="{ink}"/>'
        f'<circle cx="{r}" cy="{t}" r="{n}" fill="{dot}"/>'
    )


def favicon():
    s = FAVICON_SCALE
    tx, ty = (64 - MW * s) / 2, (64 - MH * s) / 2
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        f'<rect width="64" height="64" rx="14" fill="{DARK["wall"]}"/>'
        f'<g transform="translate({fmt(tx)} {fmt(ty)}) scale({fmt(s)})">'
        f'{mark_bold(DARK["ink"], DARK["dot"])}</g></svg>\n'
    )


def favicon_link():
    return '<link rel="icon" href="data:image/svg+xml,%s">' % quote(
        favicon().strip(), safe="/:;,=()'"
    )


def brand_css(ink, dot, width):
    return "\n".join(
        [
            ".brand-lockup { display: block; height: %dpx; width: %spx; overflow: visible; color: %s; }"
            % (LOGO_H, fmt(round(LOGO_H * width / MH, 1)), ink),
            ".brand-wire { fill: none; stroke: currentColor; stroke-width: %s; stroke-linecap: round; }"
            % fmt(WIRE_SW),
            ".brand-ring { fill: none; stroke: currentColor; stroke-width: %s; }"
            % fmt(RING_SW),
            ".brand-node, .brand-word { fill: currentColor; }",
            ".brand-dot { fill: %s; }" % dot,
        ]
    )


def component(d1, d2, width):
    children = (
        mark_class() + f'<path className="brand-word" d="{d1}"/>'
        f'<path className="brand-word" d="{d2}"/>'
    ).replace('class="', 'className="')
    return (
        "// Generated by brand/tools/make_brand.py — change the mark there, then run it.\n"
        "// Studio Kitsui's lockup: the mark (a K drawn as a network) and the wordmark.\n\n"
        "export default function BrandLockup({ className }: { className?: string }) {\n"
        "  return (\n"
        "    <svg\n"
        '      className={className ? "brand-lockup " + className : "brand-lockup"}\n'
        f'      viewBox="0 0 {fmt(width)} {fmt(MH)}"\n'
        '      role="img"\n'
        '      aria-label="Studio Kitsui"\n'
        "    >\n"
        f"      {children}\n"
        "    </svg>\n"
        "  );\n"
        "}\n"
    )


def sync(path, start, end, inner):
    """Rewrite what sits between two marker lines, indented like the start one."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    i = text.find(start)
    j = text.find(end, i + len(start)) if i >= 0 else -1
    if i < 0 or j < 0:
        raise SystemExit("make_brand: %s has no %s … %s markers" % (path, start, end))
    indent = re.match(r"[ \t]*", text[text.rfind("\n", 0, i) + 1 : i]).group(0)
    body = "".join(indent + line + "\n" for line in inner.split("\n") if line.strip())
    head = text[: text.index("\n", i) + 1]
    tail = text[text.rfind("\n", 0, j) + 1 :]
    write(path, head + body + tail)


def previews(d1, d2, width):
    out = os.path.join(BRAND, "previews")
    os.makedirs(out, exist_ok=True)
    pad = 16

    def page(bg, body, w, h):
        return svg(
            "%d %d %s %s" % (-pad, -pad, fmt(w + 2 * pad), fmt(h + 2 * pad)),
            f'<rect x="{-pad}" y="{-pad}" width="{fmt(w + 2 * pad)}" '
            f'height="{fmt(h + 2 * pad)}" fill="{bg}"/>' + body,
        )

    shots = {
        "lockup-light": page(LIGHT["wall"], lockup_body(LIGHT, d1, d2), width, MH),
        "lockup-dark": page(DARK["wall"], lockup_body(DARK, d1, d2), width, MH),
        "app-icon": app_icon(),
    }
    for name, text in shots.items():  # Quick Look draws the SVG
        tmp = tempfile.mkdtemp()
        src = os.path.join(tmp, name + ".svg")
        write(src, text)
        subprocess.run(
            ["qlmanage", "-t", "-s", "1024", "-o", tmp, src],
            check=True,
            capture_output=True,
        )
        shutil.move(
            os.path.join(tmp, name + ".svg.png"), os.path.join(out, name + ".png")
        )
        shutil.rmtree(tmp)


def main():
    ap = argparse.ArgumentParser(
        description="Studio Kitsui's logo kit, and the UIs' copies"
    )
    ap.add_argument(
        "--previews",
        action="store_true",
        help="also render the PNG previews (macOS Quick Look)",
    )
    args = ap.parse_args()
    d1, d2, width = wordmark(load_face(SANS), load_face(SERIF))
    view = "0 0 %s %s" % (fmt(MW), fmt(MH))
    write(
        os.path.join(BRAND, "studio-kitsui-mark-light.svg"),
        svg(view, mark_attr(LIGHT["ink"], LIGHT["dot"])),
    )
    write(
        os.path.join(BRAND, "studio-kitsui-mark-dark.svg"),
        svg(view, mark_attr(DARK["ink"], DARK["dot"])),
    )
    lock_view = "0 0 %s %s" % (fmt(width), fmt(MH))
    write(
        os.path.join(BRAND, "studio-kitsui-lockup-light.svg"),
        svg(lock_view, lockup_body(LIGHT, d1, d2)),
    )
    write(
        os.path.join(BRAND, "studio-kitsui-lockup-dark.svg"),
        svg(lock_view, lockup_body(DARK, d1, d2)),
    )
    write(os.path.join(BRAND, "studio-kitsui-app-icon.svg"), app_icon())
    write(os.path.join(BRAND, "studio-kitsui-favicon.svg"), favicon())
    inline = lockup_inline(d1, d2, width)
    write(os.path.join(BRAND, "lockup-inline.svg"), inline + "\n")
    sync(NNSTUDIO_UI, "<!-- brand:logo -->", "<!-- /brand:logo -->", inline)
    sync(DOCS, "<!-- brand:favicon -->", "<!-- /brand:favicon -->", favicon_link())
    sync(DOCS, "<!-- brand:logo -->", "<!-- /brand:logo -->", inline)
    sync(DOCS, "/* brand:css */", "/* /brand:css */", brand_css(DARK["ink"], DARK["dot"], width))
    sync(
        NNSTUDIO_UI, "<!-- brand:favicon -->", "<!-- /brand:favicon -->", favicon_link()
    )
    sync(
        NNSTUDIO_UI,
        "/* brand:css */",
        "/* /brand:css */",
        brand_css("var(--s-text)", "var(--s-dot)", width),
    )
    sync(
        os.path.join(CHAT, "index.html"),
        "<!-- brand:favicon -->",
        "<!-- /brand:favicon -->",
        favicon_link(),
    )
    sync(
        os.path.join(CHAT, "src", "index.css"),
        "/* brand:css */",
        "/* /brand:css */",
        brand_css("var(--text)", "var(--dot)", width),
    )
    write(
        os.path.join(CHAT, "src", "components", "BrandLockup.tsx"),
        component(d1, d2, width),
    )
    if args.previews:
        previews(d1, d2, width)
    print("make_brand: the logo kit is written, and both UIs carry it")


if __name__ == "__main__":
    main()
