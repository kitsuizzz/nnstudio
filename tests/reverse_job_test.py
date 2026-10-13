#!/usr/bin/env python3
"""The reverse-model background job: the repo-keyed status file, the agent's
checklist (static per-step progress), `.env`-selected model, the job's own
final gates, the hf_arc row, and cancellation when the box is gone. The agent
itself is stubbed (a script that prints OpenCode-shaped events and calls the
report tool), so this tests the pipeline the app polls — not an LLM."""
import json
import os
import signal
import struct
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import nn_arc  # noqa: E402

PY = sys.executable
JOB = os.path.join(ROOT, "tools", "reverse_job.py")

# An OpenCode `run --format json` event shape: a shell tool call, a step, and
# the model's own words. The wrapper turns these into the box's progress line.
# With STUB_REPO set it also walks the checklist, like the real agent does.
STUB_AGENT = r'''
import json, os, subprocess, sys, time
def ev(**kw):
    print(json.dumps(kw), flush=True)
repo = os.environ.get("STUB_REPO")
if repo:
    def rep(*args):
        subprocess.run([sys.executable,
                        os.path.join(os.environ["STUB_ROOT"], "tools", "reverse_model.py"),
                        "report", repo] + list(args), check=False,
                       stdout=subprocess.DEVNULL)
    rep("models", "stubmodel")
    rep("finish", "understand:stubmodel")
    rep("start", "reconstruct:stubmodel")
    rep("finish", "reconstruct:stubmodel")
    rep("finish", "pipeline")
ev(type="step_start", sessionID="ses_stub", part={"type": "step"})
ev(type="tool_use", sessionID="ses_stub",
   part={"type": "tool", "tool": "shell",
         "state": {"status": "completed",
                   "input": {"command": "cd %s && python tools/reverse_model.py check llama"}}})
ev(type="text", sessionID="ses_stub",
   part={"type": "text", "text": "the gate reads clean, writing the spec"})
time.sleep(0.1)
''' % ROOT

STUB_SLEEPER = r'''
import time
time.sleep(60)          # cancelled long before this returns
'''

# stands in for the real `opencode` binary: records its argv and the config /
# credential environment the job passes to it
FAKE_OPENCODE = r'''#!/usr/bin/env python3
import json, os, sys
json.dump({"argv": sys.argv[1:],
           "config": os.environ.get("OPENCODE_CONFIG_CONTENT"),
           "key": os.environ.get("NNSTUDIO_REVERSE_API_KEY")},
          open(os.environ["STUB_OUT"], "w"))
'''


def f16(n):
    return struct.pack("<%de" % n, *([0.0] * n))


def f32(n):
    return struct.pack("<%df" % n, *([0.0] * n))


def write_safetensors(path, tensors):
    header, offset = {}, 0
    for name, (dims, dtype, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(dims),
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for dims, dtype, raw in tensors.values():
            f.write(raw)


def llama_fixture(d):
    """A tiny repo on disk whose family (llama) has a verified spec: the job
    must build it exactly and land the row in hf_arc."""
    cfg = {"model_type": "llama", "hidden_size": 16, "num_hidden_layers": 1,
           "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
           "intermediate_size": 32, "vocab_size": 64, "rms_norm_eps": 1e-5,
           "rope_theta": 10000.0, "max_position_embeddings": 128}
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    tensors = {
        "model.embed_tokens.weight": ([64, 16], "F16", f16(64 * 16)),
        "model.layers.0.input_layernorm.weight": ([16], "F32", f32(16)),
        "model.layers.0.self_attn.q_proj.weight": ([16, 16], "F16", f16(16 * 16)),
        "model.layers.0.self_attn.q_proj.bias": ([16], "F32", f32(16)),
        "model.layers.0.self_attn.k_proj.weight": ([8, 16], "F16", f16(8 * 16)),
        "model.layers.0.self_attn.v_proj.weight": ([8, 16], "F16", f16(8 * 16)),
        "model.layers.0.self_attn.o_proj.weight": ([16, 16], "F16", f16(16 * 16)),
        "model.layers.0.post_attention_layernorm.weight": ([16], "F32", f32(16)),
        "model.layers.0.mlp.gate_proj.weight": ([32, 16], "F16", f16(32 * 16)),
        "model.layers.0.mlp.up_proj.weight": ([32, 16], "F16", f16(32 * 16)),
        "model.layers.0.mlp.down_proj.weight": ([16, 32], "F16", f16(16 * 32)),
        "model.norm.weight": ([16], "F32", f32(16)),
        "lm_head.weight": ([64, 16], "F16", f16(64 * 16)),
    }
    write_safetensors(os.path.join(d, "model.safetensors"), tensors)


def write_stub(d, name, body):
    path = os.path.join(d, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    return path


def run_job(source, state_dir, family=None, env=None):
    e = dict(os.environ)
    e["NNSTUDIO_REVERSE_AGENT"] = "/bin/true"      # a stub agent: exits at once
    e["NNSTUDIO_ARC_DB"] = os.path.join(state_dir, "hf_arc.db")
    e["NNSTUDIO_TASKS_DB"] = os.path.join(state_dir, "tasks.db")
    if env:
        e.update({k: v for k, v in env.items() if v is not None})
        for k, v in env.items():
            if v is None:
                e.pop(k, None)
    cmd = [PY, JOB, source, "--state-dir", state_dir]
    if family:
        cmd += ["--family", family]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=e,
                       timeout=900)
    with open(nn_arc.state_path(source, ".json", state_dir), encoding="utf-8") as f:
        return p.returncode, json.load(f)


def read_state(state_dir, source):
    with open(nn_arc.state_path(source, ".json", state_dir), encoding="utf-8") as f:
        return json.load(f)


with tempfile.TemporaryDirectory() as d:
    # 1 · a family whose spec already passes: the job reads, lets the (stub)
    #     agent run, and the final gate says done + ok
    rc, st = run_job("llama", d, family="llama")
    assert rc == 0, rc
    assert st["repo"] == "llama" and st["family"] == "llama", st
    assert st["phase"] == "done" and st["done"] is True and st["ok"] is True, st
    assert st["check"]["stages"] >= 5 and st["check"]["ok"], st["check"]
    assert st["check"]["matched"] == st["check"]["stages"], \
        "the existing llama spec is bit-exact: %r" % st["check"]
    assert st["seconds"] >= 0 and st["updated"] >= st["started"], st
    log = nn_arc.state_path("llama", ".log", d)
    assert os.path.isfile(log) and "$ " in open(log).read(), "the agent command is logged"
    assert os.path.isfile(nn_arc.state_path("llama", ".sources.txt", d)), \
        "the digest is written"
    assert st.get("todos_total") == 3 and st.get("todo_active") is None, st.get("todos")

    # 2 · a family with no spec at all: the job finishes, but not ok — and says
    #     the gate did not pass
    rc, st = run_job("nosuchfamily", d)
    assert rc == 0 and st["done"] is True and st["ok"] is False, st
    assert "gate" in (st.get("detail") or ""), st
    assert st["check"]["ok"] is False, st["check"]

    # 3 · the agent's stream becomes live progress: the tool it is on, its
    #     latest words, its session, and the base model from .env/env
    stub = write_stub(d, "stub_agent.py", STUB_AGENT)
    rc, st = run_job("llama", d, family="llama", env={
        "NNSTUDIO_REVERSE_AGENT": PY + " " + stub,
        "NNSTUDIO_REVERSE_MODEL": "test/model#high",
        # a bare model id: no endpoint from .env either
        "NNSTUDIO_REVERSE_BASE_URL": "", "NNSTUDIO_REVERSE_API_KEY": ""})
    assert rc == 0 and st["done"] is True and st["ok"] is True, st
    assert st.get("model") == "test/model#high", st
    assert st.get("doing") == "running the conformance gate", st
    assert st.get("tool") == "shell" and st.get("tools", 0) >= 1, st
    assert st.get("session") == "ses_stub", st
    assert st.get("steps") == 1, st
    assert "gate reads clean" in (st.get("last_text") or ""), st
    assert st.get("provider") is None, "a bare model id has no endpoint: %r" % st

    # 3b · an OpenAI-compatible endpoint from .env: the job runs opencode
    #      against it on its own private server — config inline, key by name
    bindir = os.path.join(d, "bin")
    os.makedirs(bindir, exist_ok=True)
    fake = os.path.join(bindir, "opencode")
    with open(fake, "w", encoding="utf-8") as f:
        f.write(FAKE_OPENCODE)
    os.chmod(fake, 0o755)
    call_log = os.path.join(d, "opencode_call.json")
    rc, st = run_job("llama", d, family="llama", env={
        "NNSTUDIO_REVERSE_AGENT": "",              # no stub: use the fake binary
        "PATH": bindir + os.pathsep + os.environ.get("PATH", ""),
        "STUB_OUT": call_log,
        "NNSTUDIO_REVERSE_BASE_URL": "https://openrouter.ai/api/v1",
        "NNSTUDIO_REVERSE_API_KEY": "sk-or-test-secret",
        "NNSTUDIO_REVERSE_MODEL": "anthropic/claude-sonnet-4.5",
    })
    assert rc == 0 and st["done"] is True and st["ok"] is True, st
    assert st.get("model") == "anthropic/claude-sonnet-4.5", st
    assert st.get("provider") == "openrouter.ai", st
    with open(call_log, encoding="utf-8") as f:
        called = json.load(f)
    argv = called["argv"]
    assert argv[:3] == ["run", "--format", "json"] and "--auto" in argv, argv
    assert "--standalone" in argv, "the endpoint job gets its own server: %r" % argv
    assert argv[argv.index("--model") + 1] == "nnstudio/main", argv
    assert "sk-or-test-secret" not in called["config"], \
        "the key never lands in the config: %r" % called["config"]
    assert called["key"] == "sk-or-test-secret", "the key is in the child's environment"
    cfg = json.loads(called["config"])
    prov = cfg["providers"]["nnstudio"]
    assert prov["settings"]["baseURL"] == "https://openrouter.ai/api/v1", prov
    assert prov["models"]["main"]["modelID"] == "anthropic/claude-sonnet-4.5", prov
    assert prov["env"] == ["NNSTUDIO_REVERSE_API_KEY"], prov
    assert prov["package"] == "@opencode/ai/providers/openai-compatible", prov
    assert "sk-or-test-secret" not in open(nn_arc.state_path("llama", ".log", d)).read(), \
        "the job log never prints the key"

    # 4 · the box is deleted while the agent works: the wrapper is told to stop,
    #     kills its child and lands a cancelled status — no final gate run
    sleeper = write_stub(d, "sleeper.py", STUB_SLEEPER)
    e = dict(os.environ)
    e["NNSTUDIO_REVERSE_AGENT"] = PY + " " + sleeper
    e["NNSTUDIO_ARC_DB"] = os.path.join(d, "hf_arc.db")
    e["NNSTUDIO_TASKS_DB"] = os.path.join(d, "tasks.db")
    p = subprocess.Popen([PY, JOB, "llama", "--family", "llama", "--state-dir", d],
                         cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         env=e, start_new_session=True)
    deadline = time.time() + 30
    st = {}
    while time.time() < deadline:
        try:
            st = read_state(d, "llama")
        except (OSError, ValueError):
            st = {}
        if st.get("phase") == "agent":
            break
        time.sleep(0.2)
    assert st.get("phase") == "agent", "the job reached the agent phase: %r" % st
    assert st.get("todo_active") == "Reading Repository", \
        "the checklist starts with the first step: %r" % st.get("todo_active")
    os.kill(p.pid, signal.SIGTERM)                 # what /api/reverse/cancel does
    assert p.wait(timeout=20) == 0, "the wrapper exits cleanly on cancel"
    time.sleep(0.3)                                # let the status write settle
    st = read_state(d, "llama")
    assert st.get("done") is True and st.get("ok") is False, st
    assert st.get("cancelled") is True, st
    assert "halted" in (st.get("detail") or ""), st
    assert "gate_tail" not in st, "cancelling skips the final gate run: %r" % st

    # 5 · a real local repo: the agent fills the checklist, the job's own gate
    #     passes, the exact canvas lands in hf_arc — and a second look is a hit
    repo = os.path.join(d, "tiny-llama")
    os.makedirs(repo)
    llama_fixture(repo)
    db = os.path.join(d, "hf_arc.db")
    stub2 = write_stub(d, "stub_todo.py", STUB_AGENT)
    rc, st = run_job(repo, d, env={
        "NNSTUDIO_REVERSE_AGENT": PY + " " + stub2,
        "STUB_REPO": repo, "STUB_ROOT": ROOT,
        "NNSTUDIO_REVERSE_MODEL": "", "NNSTUDIO_REVERSE_BASE_URL": "",
        "NNSTUDIO_REVERSE_API_KEY": ""})
    assert rc == 0 and st["done"] is True and st["ok"] is True, st
    assert st.get("repo") == repo and st.get("family") == "llama", st
    items = st.get("todos") or []
    titles = [x.get("title") for x in items]
    assert titles == ["Reading Repository",
                      "Understanding Architecture \u2014 stubmodel",
                      "Re-constructing Boxes \u2014 stubmodel",
                      "Connecting the models \u2014 pipeline box", "Done"], titles
    assert all(x.get("state") == "done" for x in items), items
    assert st.get("todos_done") == 5 and st.get("todos_total") == 5, st
    row = nn_arc.get(repo, path=db)
    assert row and row["model"]["fidelity"] == "exact", row
    assert row["model"]["spec"] == "llama", row["model"]
    assert row["nodes"] and row["edges"], "the canvas is in the row"
    rows = nn_arc.rows(path=db)
    assert len(rows) == 1 and rows[0]["repo_id"] == repo, rows
    assert "tiny-llama" in (rows[0]["note"] or ""), rows[0]
    check = nn_arc.get(repo, path=db)
    assert check["model"]["tensors"] == 13, check["model"]
    assert (check["model"].get("audit") or {}).get("ok") is True, \
        "the row carries the clean audit: %r" % check["model"].get("audit")

    # 6 · the connection audit: clean canvas, report written, agent tool works
    assert st.get("audit") and st["audit"]["ok"] is True, st.get("audit")
    audit_file = nn_arc.state_path(repo, ".audit.json", d)
    assert os.path.isfile(audit_file), "the audit report is written next to the job"
    with open(audit_file, encoding="utf-8") as f:
        report = json.load(f)
    assert report["ok"] and report["issues"] == [] and report["repo"] == repo, report
    p = subprocess.run([PY, os.path.join(ROOT, "tools", "reverse_model.py"),
                        "audit", repo], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0 and "every box is wired" in p.stdout, \
        (p.returncode, p.stdout, p.stderr)

# 7 · the agent's stream: a task row's own log is where the wrapper writes, so
#     the app's AI box can tail it live. (A missing nn_tasks import used to send
#     the stream to the state-dir path, silently — the box sat at "waiting".)
with tempfile.TemporaryDirectory() as d2:
    db = os.path.join(d2, "hf_arc.db")
    tdb = os.path.join(d2, "tasks.db")
    sys.path.insert(0, ROOT)
    import nn_tasks  # noqa: E402
    os.environ["NNSTUDIO_ARC_DB"] = db
    os.environ["NNSTUDIO_TASKS_DB"] = tdb
    task = nn_tasks.create("reverse", {"repo": "llama"})
    stream = os.path.join(d2, "stream.log")
    nn_tasks.update(task["id"], log=stream)
    e = dict(os.environ)
    e["NNSTUDIO_REVERSE_AGENT"] = "/bin/true"     # a stub agent: exits at once
    e["NNSTUDIO_ARC_DB"] = db
    e["NNSTUDIO_TASKS_DB"] = tdb
    p = subprocess.run([PY, JOB, "llama", "--family", "llama", "--state-dir", d2,
                        "--task", task["id"]],
                       cwd=ROOT, capture_output=True, text=True, env=e, timeout=900)
    assert p.returncode == 0, (p.returncode, p.stdout[-400:], p.stderr[-400:])
    assert os.path.isfile(stream) and os.path.getsize(stream) > 0, \
        ("the wrapper streams into the task's own log: %r" % p.stderr[-300:])
    assert "reverse-model job for llama" in open(stream, encoding="utf-8").read(), \
        "the wrapper's own milestone is the first thing in it"
    landed = nn_tasks.get(task["id"])
    assert landed["status"] == "done" and landed["outcome"] == "exact", landed
    del os.environ["NNSTUDIO_ARC_DB"]
    del os.environ["NNSTUDIO_TASKS_DB"]

print("reverse_job: all assertions passed")
