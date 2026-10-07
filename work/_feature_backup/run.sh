#!/usr/bin/env bash
# nnstudio's offline suite: JS syntax, the UI harness, the pure GGUF->canvas
# builder, HTML balance and py_compile. Pass --live to also smoke the app.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
fail=0
step() { printf '%-34s' "$1"; }
ok()   { echo "ok"; }
bad()  { echo "FAIL"; fail=1; }

step "extract the UI script"
sed -n '/^<script>$/,/^<\/script>$/p' ui/index.html | sed '1d;$d' > /tmp/nnstudio_ui_script.js && ok || bad

step "node --check"
node --check /tmp/nnstudio_ui_script.js >/dev/null && ok || bad

step "ui harness (DOM stub)"
node tests/ui_boot.js >/tmp/nnstudio_ui_harness.log 2>&1 && ok || { bad; tail -25 /tmp/nnstudio_ui_harness.log; }

step "GGUF -> canvas builder"
$PY tests/gguf_canvas_test.py >/tmp/nnstudio_gguf_canvas.log 2>&1 && ok || { bad; tail -15 /tmp/nnstudio_gguf_canvas.log; }

step "torch runner"
$PY tests/nn_torch_test.py >/tmp/nnstudio_nn_torch.log 2>&1 && ok || { bad; tail -20 /tmp/nnstudio_nn_torch.log; }

step "html balance"
$PY - <<'PY'
from html.parser import HTMLParser
VOID = {'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}
class P(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True); self.stack=[]; self.errors=[]
    def handle_starttag(self, tag, attrs):
        if tag not in VOID: self.stack.append((tag, self.getpos()))
    def handle_endtag(self, tag):
        if tag in VOID: return
        if not self.stack: self.errors.append("stray </%s>" % tag); return
        top, pos = self.stack.pop()
        if top != tag: self.errors.append("</%s> closes <%s> at %s" % (tag, top, pos))
p = P(); p.feed(open('ui/index.html', encoding='utf-8').read())
if p.stack: p.errors.append("unclosed: " + ", ".join("<%s>" % t for t, _ in p.stack))
raise SystemExit(1 if p.errors else 0)
PY
[ $? -eq 0 ] && ok || bad

step "py_compile"
$PY -m py_compile nn_engine.py nn_server.py nn_torch.py nnstudio_modal.py && ok || bad

if [ "${1:-}" = "--live" ]; then
  step "live app"
  G=$(curl -s -m 10 localhost:8124/api/gpu | head -c 80)
  M=$(curl -s -m 10 localhost:8124/api/models | head -c 80)
  case "$G$M" in *'"cuda"'*'"models"'*) ok;; *) bad;; esac

  step "live runner"
  $PY - <<'PY' >/tmp/nnstudio_live_runner.log 2>&1 && ok || { bad; tail -10 /tmp/nnstudio_live_runner.log; }
import json, urllib.request
B = "http://127.0.0.1:8124"
def post(p, o):
    r = urllib.request.Request(B + p, data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=30).read())
def get(p):
    return json.loads(urllib.request.urlopen(B + p, timeout=30).read())
w = get("/api/where")
assert w["local"]["torch"], w
c = {"type": "canvas", "version": 1, "nodes": [
    {"id": "i", "kind": "input", "name": "tokens", "x": 0, "y": 0, "dout": 1},
    {"id": "l", "kind": "layer", "name": "fc", "x": 200, "y": 0, "din": 1, "dout": 4},
    {"id": "o", "kind": "output", "name": "logits", "x": 400, "y": 0, "din": 4}],
    "edges": [{"from": "i", "to": "l"}, {"from": "l", "to": "o"}]}
r = post("/api/run", {"canvas": c, "feeds": {"i": [1]}, "output": "o"})
assert r["ok"] and set(r["values"]) == {"i", "l", "o"}, r
print("where + run ok")
PY
fi
exit $fail
