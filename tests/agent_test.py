#!/usr/bin/env python3
"""End-to-end: the app, and a real agent over the stateful control channel.

Starts `agent/nnstudio_agent.py` against the running app (control port 8125),
waits until it shows up under Backend -> Local, runs a small design through
POST /api/run, and checks what the agent sent back. The app must be live on
localhost:8124 — this is part of the --live suite.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = "http://localhost:8124"
AGENT = os.path.join(ROOT, "agent", "nnstudio_agent.py")
PY = os.path.join(ROOT, ".venv", "bin", "python")
LOG = "/tmp/nnstudio_agent_test.log"

CANVAS = {
    "type": "canvas", "version": 2,
    "nodes": [
        {"id": "i1", "kind": "input", "name": "x", "x": 0, "y": 0, "dout": 4},
        {"id": "l1", "kind": "layer", "name": "fc", "x": 160, "y": 0, "din": 4, "dout": 3},
        {"id": "o1", "kind": "output", "name": "y", "x": 320, "y": 0, "din": 3},
    ],
    "edges": [{"from": "i1", "to": "l1"}, {"from": "l1", "to": "o1"}],
}
LOAD_CANVAS = json.loads(json.dumps(CANVAS))
LOAD_CANVAS["nodes"][1]["place"] = "system"      # the fc layer offloads to the cpu


def _gb(b):
    b = float(b or 0)
    return "%.2f GB" % (b / 2 ** 30) if b >= 2 ** 30 else "%.1f MB" % (b / 2 ** 20)


def rss_mb(pid):
    """The agent process's resident memory, straight from the kernel —
    /proc on linux, ps on macOS."""
    try:
        with open("/proc/%d/status" % pid) as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return float(out) / 1024.0 if out else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def agent_gpu_mb(pid):
    """What the agent's own pid holds on the gpu, or None without one."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            try:
                return float(parts[1])
            except ValueError:
                return None
    return 0.0                       # not listed: the agent holds no gpu memory


def call(path, body=None, timeout=180):
    req = urllib.request.Request(
        APP + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"} if body is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def wait_for(fn, what, seconds=120):
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            if fn():
                return
        except Exception:  # noqa: BLE001 — the app may not be up for a moment
            pass
        time.sleep(0.5)
    raise SystemExit("timed out waiting for " + what)


def main():
    log = open(LOG, "w")
    t0 = time.time()
    proc = subprocess.Popen([PY, AGENT, "--app", "127.0.0.1:8125", "--once"],
                            stdout=log, stderr=subprocess.STDOUT)
    try:
        # 1 · the agent dials the app and stays — Backend -> Local can see it.
        # The systemd service may already be connected, and `connected` would
        # answer yes for it; wait for the agent *we* started (the newest
        # connection) to be the one run/load will go to.
        wait_for(lambda: (lambda b: bool(b["local"]["connected"]) and
                          ((b["local"]["agent"] or {}).get("since") or 0) >= t0 - 1)
                 (call("/api/backend")[1]),
                 "this run's agent to connect")
        _, info = call("/api/backend")
        a = info["local"]["agent"]
        mine = a["remote"]                       # this run's connection
        s = a["specs"]
        assert a["name"], "the agent reports its name"
        assert s.get("cores") and s.get("ram_gb"), "cores and ram are reported"
        assert s.get("device") in ("cpu", "cuda", "mps"), "a device is reported"
        assert s.get("torch"), "torch is installed (agent needs it): %r" % s.get("torch_error")
        print("  agent: %s — %s · %s · %s" %
              (a["name"], s.get("cpu"), s.get("gpu") or "no gpu", s["device"]))

        # 2 · the whole design runs on the agent, over the channel
        status, res = call("/api/run", {"canvas": CANVAS, "feeds": {"i1": [1, 2, 3, 4]}})
        assert status == 200 and res.get("ok"), res
        vals = res.get("values") or {}
        assert len(vals) == 3, "three boxes ran: %r" % list(vals)
        assert res.get("backend") == a["name"], "the result names the backend"
        assert isinstance(res.get("ms"), int), "it reports the time"
        out = res["outputs"][0]["stats"]
        assert out and out["shape"] and out["shape"][-1] == 3, out
        print("  ran 3 boxes in %d ms on %s (%s) — output %s" %
              (res["ms"], a["name"], res.get("device"), out["shape"]))
        # the data-flow order comes back with the result, and while it runs
        assert res.get("flow") == ["i1", "l1", "o1"], res.get("flow")
        assert res["values"]["i1"]["peek"] == [1.0, 2.0, 3.0, 4.0], res["values"]["i1"]
        status, prog = call("/api/run/progress")
        assert status == 200 and prog.get("ok"), prog
        print("  the data-flow order came back: i1 \u2192 l1 \u2192 o1")

        # 3 · the whole weight set loads and stays warm — the next run reuses it
        status, res = call("/api/load", {"canvas": LOAD_CANVAS})
        assert status == 200 and res.get("ok"), res
        assert res.get("params") and res.get("bytes"), res
        assert (res.get("on_gpu_bytes") or 0) + (res.get("on_system_bytes") or 0) == res["bytes"], res
        assert res.get("on_system_bytes", 0) > 0, \
            "the system-placed layer stays on the cpu: %r" % res
        print("  loaded %d params — %s vram, %s system (%.0f ms)" %
              (res["params"], _gb(res["on_gpu_bytes"]), _gb(res["on_system_bytes"]),
               res.get("ms") or 0))
        status, res = call("/api/run", {"canvas": LOAD_CANVAS, "feeds": {"i1": [1, 2, 3, 4]}})
        assert status == 200 and res.get("ok") and res.get("loaded") is True, res
        print("  re-run reused the loaded weights (%d ms)" % res["ms"])

        # 3b · offload: every model and cache on the backend lets go — and the
        # machine's memory really comes back, not just the reference
        big = json.loads(json.dumps(CANVAS))
        big["nodes"][0]["dout"] = 12000
        big["nodes"][1].update({"din": 12000, "dout": 12000, "dtype": "f16",
                                "place": "system"})
        big["nodes"][2]["din"] = 12000
        status, res = call("/api/load", {"canvas": big})
        assert status == 200 and res.get("ok"), res
        held, held_bytes = rss_mb(proc.pid), res["bytes"]
        status, res = call("/api/offload", {})
        assert status == 200 and res.get("ok"), res
        assert res.get("before") and res.get("after") and res.get("backend"), res
        freed = rss_mb(proc.pid)
        if sys.platform == "darwin":
            # macOS keeps freed pages resident until the kernel is under
            # pressure — libmalloc's regions and the metal driver's alike — so
            # ps rss is a linux signal, not the proof here. The metal ledger
            # is: a gpu-placed load lights it up, and offload empties it.
            gpu_big = json.loads(json.dumps(big))
            gpu_big["nodes"][1].pop("place")
            status, res = call("/api/load", {"canvas": gpu_big})
            assert status == 200 and res.get("ok"), res

            def vram():
                a = (call("/api/backend")[1]["local"].get("agent") or {})
                return ((a.get("usage") or {}).get("vram_used_gb") or 0)

            wait_for(lambda: vram() > 0.1, "the metal working set to fill", seconds=30)
            print("  the metal working set is lit: %.1f GB" % vram())
            status, res = call("/api/offload", {})
            assert status == 200 and res.get("ok"), res
            wait_for(lambda: vram() <= 0.1, "the metal working set to empty", seconds=30)
            print("  offloaded %s \u2014 the metal working set is empty again "
                  "(macOS keeps the pages until pressure; rss %.0f MB)"
                  % (_gb(held_bytes), freed or 0))
        else:
            assert held is not None and freed is not None and held - freed > 150, \
                "offload hands the weights' pages back: %.0f \u2192 %.0f MB" % (held or 0, freed or 0)
            print("  offloaded %s \u2014 rss %.0f \u2192 %.0f MB" % (_gb(held_bytes), held, freed))
        status, res = call("/api/run", {"canvas": LOAD_CANVAS, "feeds": {"i1": [1, 2, 3, 4]}})
        assert status == 200 and res.get("ok") and res.get("loaded") is False, res
        print("  the next run had to rebuild it (loaded=%s)" % res.get("loaded"))

        # 3c · a failed run keeps nothing: its graph is not left breathing
        gpu_canvas = json.loads(json.dumps(CANVAS))
        gpu_canvas["nodes"][0]["dout"] = 12000
        gpu_canvas["nodes"][1].update({"din": 12000, "dout": 12000, "dtype": "f16"})
        gpu_canvas["nodes"][2]["din"] = 12000
        measured = []
        for _ in range(2):
            status, res = call("/api/run", {"canvas": gpu_canvas, "feeds": {}})
            assert status == 400 and "value" in (res.get("error") or ""), res
            time.sleep(1)
            measured.append(agent_gpu_mb(proc.pid))
        if None not in measured:
            assert measured[1] - measured[0] < 64, \
                "a failed run leaves nothing on the gpu: %s \u2192 %s MiB" % tuple(measured)
            print("  a failed run left no graph behind (gpu %d \u2192 %d MiB)"
                  % (measured[0], measured[1]))
        else:
            print("  skip failed-run gpu check (no nvidia-smi)")

        # 4 · current usage is reported (the agent pushes it every 5 s)
        wait_for(lambda: (((call("/api/backend")[1]["local"].get("agent") or {})
                           .get("usage") or {}).get("ram_used_gb")) is not None,
                 "the agent to report usage", seconds=30)
        _, info = call("/api/backend")
        u = info["local"]["agent"]["usage"]
        assert u.get("ram_total_gb") or u.get("vram_total_gb"), u
        print("  usage: ram %.1f/%.1f GB, vram %s" %
              (u["ram_used_gb"], u["ram_total_gb"],
               ("%.1f/%.1f GB" % (u["vram_used_gb"], u["vram_total_gb"]))
               if u.get("vram_used_gb") else "n/a"))

        # 5 · external weights: export the design, read it back, run from a file
        status, res = call("/api/weights/export",
                           {"canvas": CANVAS, "format": "safetensors", "name": "live-test"})
        assert status == 200 and res.get("ok") and res.get("tensors"), res
        assert str(res.get("path", "")).endswith("weights/live-test.safetensors"), res
        print("  exported %d tensors · %s → %s" %
              (res["keys"], _gb(res["bytes"]), res["path"]))
        status, meta = call("/api/weights/meta?file=" + urllib.parse.quote(res["path"]))
        assert status == 200 and meta.get("ok") and meta.get("count") == res["keys"], meta
        print("  metadata: %s · %s" % (meta["kind"],
              ", ".join(t["name"] for t in meta["tensors"])))
        # a file written by hand with known weights proves the run uses it
        import torch
        sys.path.insert(0, os.path.join(ROOT, "agent"))
        import nn_torch
        ones = os.path.join(ROOT, "weights", "_live_ones.safetensors")
        nn_torch.write_safetensors(ones, {"fc.weight": torch.ones(3, 4)})
        rebound = json.loads(json.dumps(CANVAS))
        for node in rebound["nodes"]:
            if node["kind"] == "layer":
                node["file"] = "weights/_live_ones.safetensors"
                node["t"] = "fc.weight"
        status, res2 = call("/api/run", {"canvas": rebound, "feeds": {"i1": [1, 2, 3, 4]}})
        assert status == 200 and res2.get("ok"), res2
        mean = res2["outputs"][0]["stats"]["mean"]
        assert abs(mean - 10.0) < 1e-4, "ones(3,4) @ [1,2,3,4] is all tens: %r" % mean
        print("  the run used the file's weights (ones → \u03bc %.4f)" % mean)
        for p in (ones, os.path.join(ROOT, res["path"])):
            try:
                os.remove(p)
            except OSError:
                pass

        # 6 · a matrix input: the numbers typed on the box ride with the design
        mat = json.loads(json.dumps(CANVAS))
        mat["nodes"][0]["values"] = "1.0 2.0 3.0 4.0"
        status, res = call("/api/run", {"canvas": mat, "feeds": {}})
        assert status == 200 and res.get("ok"), res
        st = res["values"]["i1"]
        assert st["shape"] == [4] and abs(st["mean"] - 2.5) < 1e-6, st
        print("  a matrix input carries its own numbers (mean %.2f)" % st["mean"])
        two = json.loads(json.dumps(CANVAS))
        two["nodes"][0]["values"] = "1 2 3 4\n5 6 7 8"      # two rows: 2 x 4
        status, res = call("/api/run", {"canvas": two, "feeds": {}})
        assert status == 200 and res.get("ok"), res
        assert res["values"]["i1"]["shape"] == [2, 4], res["values"]["i1"]
        print("  a second row makes it 2-D (\u03bc %.4f)" % res["values"]["i1"]["mean"])

        # 7 · a tokenizer / de-tokenizer round trip over the wire
        tok = {"type": "canvas", "version": 2,
               "nodes": [
                   {"id": "tk", "kind": "tokenize", "name": "tok", "x": 0, "y": 0,
                    "vocab": "ab c", "text": "cab"},
                   {"id": "dk", "kind": "detokenize", "name": "detok", "x": 160, "y": 0,
                    "vocab": "ab c"}],
               "edges": [{"from": "tk", "to": "dk"}]}
        status, res = call("/api/run", {"canvas": tok, "feeds": {"tk": "bca"}})
        assert status == 200 and res.get("ok"), res
        assert res.get("texts", {}).get("dk") == "bca", res
        print("  a tokenizer/de-tokenizer round trip: \u201cbca\u201d \u2192 ids \u2192 \u201cbca\u201d")

        # 7b · a huggingface model box: text in, the next token's text out
        no_repo = {"type": "canvas", "version": 2,
                   "nodes": [{"id": "h0", "kind": "hf", "name": "hf 0", "x": 0, "y": 0}],
                   "edges": []}
        status, res = call("/api/run", {"canvas": no_repo, "feeds": {}})
        assert status == 400 and "repo" in (res.get("error") or ""), res
        print("  an hf box without a repo: %s" % res.get("error"))
        hf_box = {"type": "canvas", "version": 2,
                  "nodes": [{"id": "h1", "kind": "hf", "name": "hf 1", "x": 0, "y": 0,
                             "repo": "hf-internal-testing/tiny-random-gpt2"}],
                  "edges": []}
        status, res = call("/api/run", {"canvas": hf_box, "feeds": {"h1": "Hello"}})
        if status == 200 and (res.get("texts") or {}).get("h1"):
            print("  a tiny hf model answered: %r" % res["texts"]["h1"])
            assert "values" in res and "ms" in res, res
        else:
            print("  skip the tiny hf model (no network?): %s"
                  % str(res.get("error"))[:70])

        # 7c · an HF repo turned into native blocks (config + shard headers),
        # then loaded and run — the safetensors shard fetches itself
        try:
            status, res = call("/api/gguf/canvas",
                               {"repo": "hf-internal-testing/tiny-random-LlamaForCausalLM"})
            assert status == 200 and res.get("canvas"), res
            summary = res.get("summary") or {}
            canvas = res["canvas"]
            model_meta = canvas.get("model") or {}
            assert model_meta.get("fidelity") == "exact", model_meta
            bound = 0
            def count_binding(nodes):
                nonlocal bound
                for n in nodes:
                    if n.get("t") and n.get("file"):
                        bound += 1
                    if n.get("kind") == "custom":
                        count_binding(n["inner"]["nodes"])
            count_binding(canvas["nodes"])
            print("  a repo became a canvas: %s \u00b7 %d blocks \u00b7 %d boxes \u00b7 "
                  "%d bound shards \u00b7 %s spec (%s)" %
                  (summary.get("arch"), summary.get("blocks") or 0,
                   summary.get("boxes") or 0, bound,
                   model_meta.get("spec"), model_meta.get("fidelity")))
            assert bound > 10 and summary.get("blocks"), res
            tin = next(n["id"] for n in canvas["nodes"] if n["kind"] == "input")
            status, res = call("/api/run", {"canvas": canvas, "feeds": {tin: [1, 5, 9]}})
            if status == 200 and res.get("ok"):
                logits = next((v["shape"] for v in (res.get("values") or {}).values()
                               if (v.get("shape") or [])[-1:] == [32000]), None)
                print("  \u2026and ran: %d boxes, logits %s" %
                      (len(res.get("values") or {}), logits))
                assert logits == [3, 32000], logits
            else:
                print("  skip running the repo canvas: %s" % str(res.get("error"))[:70])
        except Exception as e:  # noqa: BLE001 — network-dependent
            print("  skip the repo canvas (no network?): %s" % str(e)[:70])

        # 7d · the reverse-model endpoints: an hf_arc row answers done instantly
        sys.path.insert(0, ROOT)
        import nn_arc
        arc_repo = "nnstudio-test/tiny-llama"
        nn_arc.put(arc_repo, {"type": "canvas", "version": 1,
                              "model": {"arch": "llama", "fidelity": "exact"},
                              "nodes": [{"id": "t1", "kind": "input", "name": "tokens",
                                         "x": 0, "y": 0, "dout": 1}],
                              "edges": []},
                   transformers="test", note="the live suite put this here")
        try:
            status, res = call("/api/reverse", {"repo": arc_repo})
            assert status == 200 and res.get("ok"), res
            task = res.get("task") or {}          # already in hf_arc: done at once
            assert task.get("status") == "done" and task.get("outcome") == "ok", res
            status, res = call("/api/reverse/status?repo=" + urllib.parse.quote(arc_repo))
            assert status == 200 and res.get("ok") and res["status"].get("arc"), res
            print("  reverse-model: an hf_arc row answers done at once")
        finally:
            nn_arc.delete(arc_repo)

        # 7e · cancel: a repo with no live job is a no-op; an empty request is
        #      a bad request (the box sends the repo)
        status, res = call("/api/reverse/cancel", {"repo": "nnstudio-test/nothing"})
        assert status == 200 and res.get("ok") and res.get("cancelled") is False, res
        status, res = call("/api/reverse/cancel", {})
        assert status == 400 and not res.get("ok"), res
        print("  \u2026and cancel is a no-op for a repo with no live job")

        # 8 · a huggingface tokenizer: paste the repo, the model's own tokenizer loads
        status, meta = call("/api/tokenizer?repo=gpt2")
        assert status == 200 and meta.get("ok") and meta.get("tokens"), meta
        print("  huggingface tokenizer: gpt2 \u00b7 %s \u00b7 %d tokens"
              % (meta["model"], meta["tokens"]))
        hf = {"type": "canvas", "version": 2,
              "nodes": [
                  {"id": "hk", "kind": "tokenize", "name": "tok", "x": 0, "y": 0,
                   "repo": "gpt2", "text": "Hello world"},
                  {"id": "hd", "kind": "detokenize", "name": "detok", "x": 160, "y": 0,
                   "repo": "gpt2"}],
              "edges": [{"from": "hk", "to": "hd"}]}
        status, res = call("/api/run", {"canvas": hf, "feeds": {"hk": "Hello world"}})
        assert status == 200 and res.get("ok"), res
        assert res["texts"]["hd"] == "Hello world", res["texts"]
        print("  \u2026and \u201cHello world\u201d round-tripped through it")

        # 9 · the text comes from an input box now, through the wire
        hf2 = {"type": "canvas", "version": 2,
               "nodes": [
                   {"id": "hi", "kind": "input", "name": "text in", "x": 0, "y": 0,
                    "input_mode": "text", "text": "cab"},
                   {"id": "hk2", "kind": "tokenize", "name": "tok", "x": 160, "y": 0,
                    "repo": "gpt2"},
                   {"id": "hd2", "kind": "detokenize", "name": "detok", "x": 320, "y": 0,
                    "repo": "gpt2"}],
               "edges": [{"from": "hi", "to": "hk2"}, {"from": "hk2", "to": "hd2"}]}
        status, res = call("/api/run", {"canvas": hf2, "feeds": {}})
        assert status == 200 and res.get("ok"), res
        assert res["texts"]["hd2"] == "cab", res["texts"]
        print("  an input in text mode fed the tokenizer over the wire: \u201ccab\u201d")

        # 10 · logits -> sample (picks one id) -> de-tokenizer -> text
        sample = {"type": "canvas", "version": 2,
                  "nodes": [
                      {"id": "si", "kind": "input", "name": "logits", "x": 0, "y": 0, "dout": 3},
                      {"id": "ss", "kind": "sample", "name": "pick", "x": 160, "y": 0,
                       "temperature": 0},
                      {"id": "sd", "kind": "detokenize", "name": "detok", "x": 320, "y": 0,
                       "vocab": "abc"}],
                  "edges": [{"from": "si", "to": "ss"}, {"from": "ss", "to": "sd"}]}
        status, res = call("/api/run", {"canvas": sample, "feeds": {"si": [0.1, 5.0, 0.2]}})
        assert status == 200 and res.get("ok") and res["texts"]["sd"] == "b", res
        print("  logits \u2192 sample \u2192 id \u2192 de-tokenizer: \u201cb\u201d")
        direct = {"type": "canvas", "version": 2,
                  "nodes": [
                      {"id": "di", "kind": "input", "name": "logits", "x": 0, "y": 0, "dout": 3},
                      {"id": "dd", "kind": "detokenize", "name": "detok", "x": 160, "y": 0,
                       "vocab": "abc"}],
                  "edges": [{"from": "di", "to": "dd"}]}
        status, res = call("/api/run", {"canvas": direct, "feeds": {"di": [0.1, 5.0, 0.2]}})
        assert status == 400 and "sample" in (res.get("error") or ""), res
        assert res.get("nid") == "dd" and res.get("box") == "detok", res
        print("  logits straight into it are refused: %s (block %s)"
              % (res.get("error"), res.get("nid")))

        # 11 · an input with no value: the agent's reason comes back, not a crash
        status, res = call("/api/run", {"canvas": CANVAS, "feeds": {}})
        assert status == 400 and "value" in (res.get("error") or ""), res
        assert res.get("nid") == "i1" and res.get("box") == "x", res
        print("  missing input refused: %s (block %s)" % (res.get("error"), res.get("nid")))

        # 12 · the app notices when the agent hangs up (the always-on service,
        # if any, may still be connected under the same name — look at *this*
        # connection, not at "connected")
        proc.terminate()
        wait_for(lambda: ((call("/api/backend")[1]["local"].get("agent") or {})
                          .get("remote")) != mine,
                 "the app to notice the agent left", seconds=20)
        print("  disconnect noticed")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
