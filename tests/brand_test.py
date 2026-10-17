#!/usr/bin/env python3
"""The logo: the kit is written, and every copy of the mark matches its source.

brand/tools/make_brand.py draws the mark and writes brand/lockup-inline.svg.
The NN Studio page, the Chat half's React component and its page all carry
that markup. A copy edited by hand, or a mark changed without a regeneration,
fails here.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILED = []


def check(name, fn):
    try:
        fn()
        print("  ok   " + name)
    except Exception as e:  # noqa: BLE001
        FAILED.append(name)
        print("  FAIL " + name + " — " + str(e))


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def need(cond, msg):
    if not cond:
        raise AssertionError(msg)


INLINE = read("brand/lockup-inline.svg")
PATHS = re.findall(r' d="([^"]+)"', INLINE)
CIRCLES = re.findall(r'cx="([^"]+)" cy="([^"]+)" r="([^"]+)"', INLINE)
KIT = [
    "studio-kitsui-mark-light.svg", "studio-kitsui-mark-dark.svg",
    "studio-kitsui-lockup-light.svg", "studio-kitsui-lockup-dark.svg",
    "studio-kitsui-app-icon.svg", "studio-kitsui-favicon.svg", "lockup-inline.svg",
]
PREVIEWS = ["lockup-light.png", "lockup-dark.png", "app-icon.png"]


def kit_is_written():
    for name in KIT:
        need(os.path.getsize(os.path.join(ROOT, "brand", name)) > 200, name + " is empty")
    for name in PREVIEWS:
        need(os.path.getsize(os.path.join(ROOT, "brand", "previews", name)) > 1000,
             name + " is missing")


def carries_the_mark(rel):
    text = read(rel)
    for d in PATHS:
        need(d in text, "%s lacks a path of the mark" % rel)
    for cx, cy, r in CIRCLES:
        need(re.search(r'cx="%s" cy="%s" r="%s"' % (re.escape(cx), re.escape(cy),
                                                    re.escape(r)), text),
             "%s lacks a port of the mark (cx=%s cy=%s r=%s)" % (rel, cx, cy, r))


def nnstudio_header():
    ui = read("ui/index.html")
    need('<a class="logo" href="/" aria-label="Studio Kitsui">' in ui, "the header's logo link")
    need("<title>Studio Kitsui — Studio</title>" in ui, "the page title")
    need('rel="icon" href="data:image/svg+xml,' in ui, "the favicon")
    need(".brand-lockup {" in ui and ".brand-dot {" in ui, "the brand's stylesheet")


def chat_half():
    html = read("studio/frontend/index.html")
    need("<title>Studio Kitsui — Chat</title>" in html, "the Chat page title")
    need('rel="icon" href="data:image/svg+xml,' in html, "the Chat favicon")
    tsx = read("studio/frontend/src/components/BrandLockup.tsx")
    need("export default function BrandLockup" in tsx, "the React component")
    top = read("studio/frontend/src/components/TopBar.tsx")
    need('import BrandLockup from "./BrandLockup";' in top and "<BrandLockup />" in top,
         "the top bar uses the lockup")
    need("chat.kitsui</span>" not in top, "the old text mark is gone")
    need(".brand-lockup {" in read("studio/frontend/src/index.css"),
         "the Chat stylesheet carries the brand")


check("the kit is written: the masters, the favicon, the previews", kit_is_written)
check("the NN Studio page carries the mark", lambda: carries_the_mark("ui/index.html"))
check("the Chat component carries the mark",
      lambda: carries_the_mark("studio/frontend/src/components/BrandLockup.tsx"))
check("the NN Studio header, title and favicon", nnstudio_header)
check("the handbook carries the mark", lambda: carries_the_mark("ui/docs.html"))
check("the Chat half: title, favicon, and the top bar's lockup", chat_half)

if FAILED:
    print("brand_test: %d failure(s): %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("brand_test: all assertions passed")
