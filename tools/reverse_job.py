#!/usr/bin/env python3
"""Run the reverse-model agent for one HF repo, in the background.

The app's `build canvas` starts this when a repo has no `hf_arc` row yet. The
job keeps a status file the app polls and renders **on the box** while the
agent works:

    <slug>.json          phase, elapsed, todos, live gate, done
    <slug>.log           everything the agent printed
    <slug>.todos.json    the checklist the agent fills as it works
    <slug>.sources.txt   the library source digest the agent reads first

Phases: `queued` -> `reading` (source digest) -> `agent` (drafting/iterating)
-> `verifying` (the job's own gates) -> `done` (`ok` tells whether it passed).

The agent reports its progress through `tools/reverse_model.py report …` (the
static TODO list: reading -> understanding/re-constructing per model ->
pipeline). The wrapper merges that file into the status every tick, so the
box names the current step, not just the elapsed time.

The base model comes from the repo's `.env`:

    NNSTUDIO_REVERSE_BASE_URL=...   any OpenAI-compatible endpoint
    NNSTUDIO_REVERSE_API_KEY=...
    NNSTUDIO_REVERSE_MODEL=...      the model id at that provider

With all three set the job runs `opencode run --standalone` against that
endpoint — its own private server, the config passed inline
(`OPENCODE_CONFIG_CONTENT`), the key named in the environment and never
written to a file. With only `NNSTUDIO_REVERSE_MODEL`, it is an ordinary
OpenCode model reference. With none, OpenCode's own default model is used.

When the agent exits, the **job itself** proves the work: the family
conformance gate must run green (tests/spec_conformance_test.py — the library's
own eager run, gate 1e-4) and the repo must build the exact canvas
(`nn_engine.arc_canvas`, bound to the repo's real tensor names and shards,
wired per the agent's `<slug>.pipeline.json` when the repo has several models).
Only then does the canvas land in `hf_arc` — the agent's say-so never does.

When the box that asked for the job is deleted, the app posts
`/api/reverse/cancel`; the wrapper is signalled, kills its agent and lands
`done, ok=false, cancelled=true` — without running the final gates.

    python tools/reverse_job.py <repo-or-family> [--family f] [--timeout 1800]

Set NNSTUDIO_REVERSE_AGENT to swap the agent command (tests use a stub that
exits at once).
"""
import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from urllib.parse import urlsplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import nn_arc      # noqa: E402 — the hf_arc database + job file state
import nn_engine   # noqa: E402 — arc_canvas, the exact-canvas builder
import nn_tasks    # noqa: E402 — the agent_tasks row this job serves

PY = sys.executable


# `.env` at the repo root: NNSTUDIO_REVERSE_MODEL picks the base model the
# agent runs on (provider/model[#variant]); everything else in it is available
# too (NNSTUDIO_REVERSE_AGENT swaps the agent command, e.g. in tests).
def load_dotenv(path):
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                val = val.strip().strip('"').strip("'")
                if key.strip():
                    os.environ.setdefault(key.strip(), val)
    except OSError:
        pass


load_dotenv(os.path.join(ROOT, ".env"))

CANCELLED = {"flag": False}

PROVIDER_ID = "nnstudio"          # the provider id the .env endpoint gets


def provider_config(base, model):
    """An OpenCode config for any OpenAI-compatible endpoint, passed inline.

    Only the credential's *environment variable name* is in the config; the
    value stays in the environment the job was started with."""
    return {
        "$schema": "https://opencode.ai/config.json",
        "providers": {
            PROVIDER_ID: {
                "name": "nnstudio (.env)",
                "env": ["NNSTUDIO_REVERSE_API_KEY"],
                "package": "@opencode/ai/providers/openai-compatible",
                "settings": {"baseURL": base},
                "models": {"main": {"modelID": model, "name": model}},
            }
        },
    }


def _term(_signum, _frame):
    CANCELLED["flag"] = True


def describe_tool(tool, inp):
    """One line the app shows for what the agent is doing right now."""
    inp = inp or {}
    if tool == "shell":
        cmd = str(inp.get("command") or "")
        low = cmd.lower()
        if "spec_conformance_test" in low or "reverse_model.py check" in low:
            return "running the conformance gate"
        if "reverse_model.py inspect" in low:
            return "inspecting the repo's tensors"
        if "reverse_model.py sources" in low:
            return "reading the library source"
        if "tests/run.sh" in low:
            return "running the test suite"
        if " pip install" in low or low.startswith("pip "):
            return "installing a dependency"
        first = re.sub(r"^\s*cd\s+\S+\s*&&\s*", "", cmd).strip()
        first = re.sub(r"\s+", " ", first.splitlines()[0]) if first else ""
        return "running " + ((first[:70] + "\u2026") if len(first) > 70 else
                             (first or "a command"))
    if tool in ("read", "read_file"):
        return "reading " + os.path.basename(
            str(inp.get("filePath") or inp.get("path") or "a file"))
    if tool in ("edit", "write", "patch", "edit_file", "write_file"):
        return "editing " + os.path.basename(
            str(inp.get("filePath") or inp.get("path") or "a file"))
    if tool in ("grep", "glob", "search"):
        return "searching the tree"
    if tool in ("todowrite", "todoread"):
        return "planning"
    if tool == "task":
        return "delegating a subtask"
    return ("running " + str(tool)) if tool else None


class Job:
    def __init__(self, repo, family, state_dir=None, quiet=False, task=None):
        self.repo = nn_arc.canonical(repo)
        self.family = family or ""
        self.dir = state_dir or nn_arc.state_dir()
        self.slug = nn_arc.repo_slug(repo)
        self.path = nn_arc.state_path(repo, ".json", self.dir)
        self.task = task or None           # the agent_tasks row this job serves
        self.log_path = nn_arc.state_path(repo, ".log", self.dir)
        if self.task:
            try:                         # the durable stream, next to the database
                row = nn_tasks.get(self.task)
                if row and row.get("log"):
                    self.log_path = row["log"]
                    os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
                else:
                    print("[reverse-job] warning: task %s has no log path \u2014 "
                          "streaming to %s" % (self.task, self.log_path), flush=True)
            except Exception as e:  # noqa: BLE001 — the state-dir log still works,
                print("[reverse-job] warning: cannot read task %s (%s) \u2014 "
                      "streaming to %s" % (self.task, e, self.log_path), flush=True)
        self.state = {"repo": self.repo, "family": family or None,
                      "slug": self.slug, "phase": "queued",
                      "started": time.time(), "updated": time.time(),
                      "seconds": 0, "done": False, "ok": False}
        os.makedirs(self.dir, exist_ok=True)
        self.flush()

    def flush(self, **fields):
        self.state.update(fields)
        self.state["seconds"] = round(time.time() - self.state["started"], 1)
        self.state["updated"] = time.time()
        nn_arc.atomic_json(self.path, self.state)
        self.sync_task()

    def sync_task(self):
        """Mirror the live progress into the task row the app's list shows —
        only the columns this wrapper owns (the app owns `status`)."""
        if not self.task:
            return
        cols = {"phase": self.state.get("phase")}
        if self.state.get("todos"):
            cols["todo"] = self.state["todos"]
        for k in ("pid", "agent_pid"):
            if self.state.get(k) is not None:
                cols[k] = self.state[k]
        if self.state.get("done"):
            cols.update(task_landing(self.state))
        try:
            import nn_tasks
            nn_tasks.update(self.task, **cols)
        except Exception:  # noqa: BLE001 — the job must never die over the list
            pass

    def log(self, line):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line if line.endswith("\n") else line + "\n")


def task_landing(state):
    """What the job's own gates concluded, as the task row's final state."""
    import time as _t
    if state.get("cancelled"):
        return {"status": "cancelled", "finished": _t.time()}
    if state.get("ok"):
        return {"status": "done", "outcome": "exact", "finished": _t.time()}
    err = str(state.get("error") or state.get("detail") or "")
    broken = any(w in err for w in ("timed out", "crashed", "could not", "failed"))
    return {"status": "done", "finished": _t.time(),
            "outcome": "failed" if broken else "approximate"}


def family_of(source):
    """A repo id / local dir / model_type -> the family name to build a spec for."""
    source = nn_arc.canonical(source)
    if os.path.isdir(source):
        with open(os.path.join(source, "config.json"), encoding="utf-8") as f:
            return json.load(f)["model_type"]
    if "/" in source and not source.endswith(".json"):
        head, rest = source.split("/", 1)          # a Hub subfolder's own config
        repo, _, sub = rest.partition("/")
        cfg = json.loads(nn_engine._hf_fetch(head + "/" + repo,
                                             (sub + "/" if sub else "") + "config.json"))
        return cfg["model_type"]
    return source


def run_gate(family, timeout=900):
    """One conformance run: (ok, matched, stages, fails, tail)."""
    cmd = [PY, os.path.join(ROOT, "tests", "spec_conformance_test.py"), family]
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.SubprocessError as e:  # noqa: BLE001
        return False, 0, 0, [str(e)[:120]], ""
    out = (p.stdout or "") + (p.stderr or "")
    stages = len(re.findall(r"max\|diff\|", out))
    matched = len(re.findall(r"max\|diff\|\s+0\.00e\+00", out))
    fails = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("FAIL")]
    ok = p.returncode == 0 and not fails
    tail = "\n".join(out.strip().splitlines()[-8:])
    return ok, matched, stages, fails, tail


def read_lines(proc, out_list, log_path):
    """Drain the agent's stdout into the log while it runs."""
    with open(log_path, "a", encoding="utf-8") as log:
        for raw in iter(proc.stdout.readline, ""):
            out_list.append(raw)
            log.write(raw)
            log.flush()


def pump_events(lines, seen, state):
    """Fold new agent events into the job's live progress; return the cursor."""
    while seen < len(lines):
        raw, seen = lines[seen], seen + 1
        try:
            ev = json.loads(raw)
        except ValueError:
            continue
        t = ev.get("type")
        part = ev.get("part") or {}
        if ev.get("sessionID"):
            state["session"] = ev["sessionID"]
        if t in ("tool", "tool_use"):
            doing = describe_tool(part.get("tool"),
                                  (part.get("state") or {}).get("input"))
            if doing:
                state["doing"] = doing
                state["tool"] = part.get("tool")
                state["tools"] = (state.get("tools") or 0) + 1
        elif t == "text":
            txt = str(part.get("text") or "").strip()
            if txt:
                state["last_text"] = txt.splitlines()[-1][:160]
        elif t == "step_start":
            state["steps"] = (state.get("steps") or 0) + 1
    return seen


def merge_todos(job):
    """Fold the agent's checklist file into the status the app polls."""
    doc = nn_arc.read_todos(job.repo, job.dir)
    if not doc or doc.get("updated") == job.state.get("todos_updated"):
        return
    items = doc.get("items") or []
    job.state["todos_updated"] = doc.get("updated")
    job.state["todos"] = items
    active = next((x for x in items if x.get("state") == "active"), None)
    job.state["todo_active"] = active.get("title") if active else None
    job.state["todos_done"] = sum(1 for x in items if x.get("state") == "done")
    job.state["todos_total"] = len(items)


def mark_todos(job, key, state, note=None):
    """Move one checklist item (or the active one, with key=None)."""
    doc = nn_arc.read_todos(job.repo, job.dir) or nn_arc.initial_todos(job.repo, job.dir)
    items = doc.get("items") or []
    item = nn_arc.todo_find(items, key) if key else next(
        (x for x in items if x.get("state") == "active"), None)
    if item is None:
        return
    item["state"] = state
    if note:
        item["note"] = note
    nn_arc.write_todos(job.repo, {"repo": job.repo, "items": items}, job.dir)
    merge_todos(job)


def finish_todos(job, ok, note=None):
    """The job's own verdict lands on the checklist: all done, or the step
    that failed plus why."""
    doc = nn_arc.read_todos(job.repo, job.dir) or nn_arc.initial_todos(job.repo, job.dir)
    items = doc.get("items") or []
    if ok:
        for item in items:
            item["state"] = "done"
    else:
        item = nn_arc.todo_find(items, "done") or next(
            (x for x in items if x.get("state") == "active"), None)
        if item is not None:
            item["state"] = "failed"
            if note:
                item["note"] = note
    nn_arc.write_todos(job.repo, {"repo": job.repo, "items": items}, job.dir)
    merge_todos(job)


def ensure_model_todos(job):
    """The agent is drafting a spec but never reported its models: seed the
    checklist with the family as the single model, so the box shows the real
    steps instead of a frozen 'Reading Repository'."""
    doc = nn_arc.read_todos(job.repo, job.dir)
    items = (doc or {}).get("items") or []
    if any(str(x.get("key") or "").startswith("understand:") for x in items):
        return
    items = nn_arc.todo_items([job.family])
    first = nn_arc.todo_find(items, "reading")
    if first:
        first["state"] = "done"
    nxt = nn_arc.todo_find(items, "understand:%s" % job.family)
    if nxt:
        nxt["state"] = "active"
        nxt["note"] = "the agent did not report its models"
    nn_arc.write_todos(job.repo, {"repo": job.repo, "items": items}, job.dir)
    merge_todos(job)
    job.log("# the agent never reported its models: the checklist assumes one (%s)"
            % job.family)


def acquire_lock(job, timeout):
    """One family, one spec: hold the family's lock while the agent edits it.

    Two repos of the same family may be researched at the same time; both
    would rewrite `specs/<family>.json`. The lock serializes them. It is
    released automatically if this process dies."""
    import fcntl
    path = os.path.join(job.dir, (job.family or "family") + ".lock")
    f = None
    try:
        f = open(path, "w")
    except OSError:
        return None
    waited = False
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return f
        except OSError:
            pass
        if CANCELLED["flag"] or time.time() - job.state["started"] > timeout:
            f.close()
            return None
        if not waited:
            waited = True
            job.log("# waiting for another job on family %s" % job.family)
        job.flush(phase="queued",
                  detail="waiting for another job on %s\u2026" % job.family)
        time.sleep(2)


def read_pipeline(job):
    try:
        with open(nn_arc.pipeline_path(job.repo, job.dir), encoding="utf-8") as f:
            doc = json.load(f)
        return doc if isinstance(doc, dict) else None
    except (OSError, ValueError):
        return None


def verify_and_store(job, timeout=900):
    """The job's own proof: the family gates run green and the repo builds the
    exact canvas — only then does the row land in hf_arc."""
    if not (os.path.isdir(job.repo) or "/" in job.repo):
        # a bare family, run by hand: there is no repo to bind or store — the
        # family gate is the whole proof
        job.flush(phase="verifying",
                  detail="running the conformance gate \u2014 %s" % job.family)
        ok, matched, stages, fails, tail = run_gate(job.family, timeout=timeout)
        check = {"ok": ok, "matched": matched, "stages": stages, "fails": fails[:4]}
        return (ok, "the gate passed" if ok else "the gate did not pass",
                {"check": check, "gate_tail": None if ok else tail})
    pipeline = read_pipeline(job)
    sources = []
    if pipeline and pipeline.get("models"):
        for m in pipeline["models"]:
            if isinstance(m, dict) and m.get("source"):
                sources.append(nn_arc.canonical(str(m["source"])))
    sources = sources or [job.repo]
    families = []
    for src in sources:
        try:
            fam = family_of(src)
        except Exception as e:  # noqa: BLE001 — the repo is the thing being read
            return False, "could not read %s: %s" % (src, e), {}
        if fam and fam not in families:
            families.append(fam)
    gate = {}
    for fam in families:
        job.flush(phase="verifying",
                  detail="running the conformance gate \u2014 %s" % fam)
        ok, matched, stages, fails, tail = run_gate(fam, timeout=timeout)
        gate[fam] = {"ok": ok, "matched": matched, "stages": stages, "fails": fails[:4]}
        if not ok:
            return (False, "the %s gate did not pass" % fam,
                    {"check": gate[fam], "gate_tail": tail})
    job.flush(phase="verifying", detail="binding the repo's tensors to the canvas")
    out, last = None, None
    for attempt in range(2):                       # a flaky header read may retry
        try:
            out = nn_engine.arc_canvas(job.repo, pipeline=pipeline, use_arc=False)
            break
        except nn_engine.ArcError as e:
            return False, str(e), {}
        except Exception as e:  # noqa: BLE001 — network/parse hiccup: one more try
            last = e
            job.log("# binding attempt %d failed: %r" % (attempt + 1, e))
            time.sleep(5)
    if out is None:
        return False, "could not build the repo canvas: %s" % last, {}
    canvas = out["canvas"]

    # the connection audit: the gate cannot see a box the wiring never reads
    audit = nn_engine.audit_canvas(canvas)
    try:
        nn_arc.atomic_json(nn_arc.state_path(job.repo, ".audit.json", job.dir),
                           dict(audit, repo=job.repo, updated=time.time()))
    except OSError:
        pass
    job.state["audit"] = {"ok": audit["ok"], "count": audit["count"],
                          "boxes": audit["boxes"], "blocks": audit["blocks"],
                          "issues": audit["issues"][:12]}
    job.flush(audit=job.state["audit"])
    if not audit["ok"]:
        job.log("# audit: %d connection issue(s) in %d block(s)"
                % (audit["count"], len(audit["blocks"])))
        if not job.state.get("repaired"):
            return (False, "the canvas has %d connection issue%s"
                    % (audit["count"], "" if audit["count"] == 1 else "s"),
                    {"repair": True, "audit": job.state["audit"]})
    try:
        import transformers
        version = transformers.__version__
    except Exception:  # noqa: BLE001 — the version is provenance, not proof
        version = None
    note = "pipeline: " + " + ".join(out.get("models") or [])
    if not audit["ok"]:
        note += " \u00b7 %d connection warning(s)" % audit["count"]
    canvas.setdefault("model", {})["audit"] = job.state["audit"]
    try:
        nn_arc.put(job.repo, canvas, transformers=version, note=note)
    except Exception as e:  # noqa: BLE001 — the row is the point; fail loudly
        return False, "could not store the row: %s" % e, {}
    return (True, "the gate passed",
            {"check": gate.get(families[0], {}) if families else {},
             "canvas": out, "gate_tail": None, "audit": job.state["audit"]})


def agent_prompt(job, args, repair=False):
    """What the agent is told to do — the full flow, or one repair pass."""
    digest = nn_arc.state_path(job.repo, ".sources.txt", args.state_dir)
    pipeline = nn_arc.pipeline_path(job.repo, args.state_dir)
    if repair:
        audit_path = nn_arc.state_path(job.repo, ".audit.json", args.state_dir)
        try:
            with open(audit_path, encoding="utf-8") as f:
                audit = json.load(f)
        except (OSError, ValueError):
            audit = {}
        lines = ["  \u2022 %s: %s" % (i.get("box") or "?", i.get("what"))
                 for i in (audit.get("issues") or [])[:8]]
        return (
            "The nnstudio job built the exact canvas from `specs/%s.json`, but "
            "its connection audit found problems \u2014 every box must be wired:\n\n"
            "%s\n\n"
            "The full list is at %s. Fix the spec, then run "
            "`python tools/reverse_model.py check %s` until green and "
            "`python tools/reverse_model.py audit %s` until it reports `ok`. "
            "Do not remove boxes that carry data; add the missing wires or "
            "conditions. Do not ask questions. Stop when both are clean.\n"
            % (job.family, "\n".join(lines) or "  (see the audit file)",
               audit_path, job.family, job.repo))
    return (
        "You are the nnstudio reverse-engineering agent for the HF repo `%s` "
        "(family `%s`). Follow `.opencode/commands/reverse-model.md` exactly — "
        "read it first.\n\n"
        "Your job: research how this repo's model(s) really run on raw PyTorch "
        "(the transformers source, the model card, or the repo's own code) and "
        "draw them with nnstudio's own native boxes in `specs/%s.json`. The job "
        "will then prove the result against the library's eager run and bind it "
        "to this repo's real tensors.\n\n"
        "Report your progress in the checklist as you go — one call at a time, "
        "never batched at the end:\n"
        "  python tools/reverse_model.py report %s models NAME[,NAME...]  "
        "# right after reading the repo: how many models it holds\n"
        "  python tools/reverse_model.py report %s start understand:NAME\n"
        "  python tools/reverse_model.py report %s finish understand:NAME\n"
        "  python tools/reverse_model.py report %s start reconstruct:NAME\n"
        "  python tools/reverse_model.py report %s finish reconstruct:NAME\n"
        "  python tools/reverse_model.py report %s finish pipeline  "
        "# after connecting the models\n"
        "For a multi-model repo, write the wiring to `%s`:\n"
        '  {"models": [{"name": "vae", "source": "%s/vae"}, ...],\n'
        '   "edges":  [{"from": "text", "to": "dit", "fromPort": 0, "toPort": 0}]}\n'
        "The source digest is already written to %s — read it before drafting. "
        "For this repo's tensor names use `python tools/reverse_model.py inspect %s`.\n"
        "Before you finish, run `python tools/reverse_model.py audit %s` and fix "
        "every issue it reports — every box must be wired, or the job will ask "
        "you to do it again.\n"
        "If `specs/%s.json` already exists and is verified, adapt it to this "
        "repo's tensor names instead of re-deriving it. If a primitive is "
        "missing, add it to `agent/nn_torch.py` with a test in "
        "`tests/nn_torch_test.py` and the UI op lists, then continue. Do not ask "
        "questions. Stop only when the gate is green or provably impossible.\n"
        % (job.repo, job.family, job.family,
           job.repo, job.repo, job.repo, job.repo, job.repo, job.repo,
           pipeline, job.repo, digest, job.repo, job.repo, job.family))


def run_agent(job, args, model, base, key, provider_host, prompt,
              detail="drafting the spec"):
    """Spawn the agent and follow it to its exit.

    Returns (state, message): `done` (it exited), `cancelled`, `timed_out`,
    or `failed` (it could not start). The caller owns the final status."""
    try:
        import transformers
        tver = transformers.__version__
    except Exception:  # noqa: BLE001
        tver = None
    override = os.environ.get("NNSTUDIO_REVERSE_AGENT")
    child_env = dict(os.environ)
    child_env["NNSTUDIO_ARC_STATE_DIR"] = args.state_dir
    if override:
        cmd = shlex.split(override)
    else:
        exe = shutil.which("opencode") or os.path.expanduser("~/.opencode/bin/opencode")
        cmd = [exe, "run", "--format", "json", "--auto"]
        if base and key and model:
            child_env["OPENCODE_CONFIG_CONTENT"] = json.dumps(provider_config(base, model))
            cmd += ["--standalone", "--model", "%s/main" % PROVIDER_ID]
        elif model:
            cmd += ["--model", model]           # an OpenCode model reference
        cmd += ["--title", "reverse-model %s" % job.slug, prompt]
    job.flush(phase="agent", detail=detail, transformers=tver,
              command=" ".join(cmd[:4]), model=model or None,
              provider=provider_host, pid=os.getpid())
    job.log("$ " + " ".join(shlex.quote(c) for c in cmd))
    if base and key and model:
        job.log("# provider %s · model %s · key read from NNSTUDIO_REVERSE_API_KEY"
                % (base, model))
    try:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                start_new_session=True, env=child_env)
    except OSError as e:
        return "failed", "the agent could not start: %s" % e
    job.flush(agent_pid=proc.pid)

    lines = []
    import threading
    reader = threading.Thread(target=read_lines, args=(proc, lines, job.log_path),
                              daemon=True)
    reader.start()

    spec = os.path.join(ROOT, "specs", job.family + ".json")
    spec_at_start = os.path.isfile(spec)
    last_gate = 0.0
    seen = 0
    timed_out = False
    while proc.poll() is None:
        if CANCELLED["flag"]:
            break
        if time.time() - job.state["started"] > args.timeout:
            timed_out = True
            job.flush(detail="timed out")
            break
        # every event the agent streams: what tool it is on, its latest words
        seen = pump_events(lines, seen, job.state)
        merge_todos(job)
        if os.path.isfile(spec):
            if not job.state.get("spec_seen") and not spec_at_start:
                ensure_model_todos(job)            # a draft with no reported models
            job.state["spec_seen"] = True
        live = job.state.get("doing") or job.state.get("last_text")
        todo = job.state.get("todo_active")
        stale = time.time() - (job.state.get("todos_updated") or 0) > 150
        detail = ((todo if (todo and not stale) else (live or todo))
                  or ("the spec is drafted" if job.state.get("spec_seen")
                      else "drafting the spec"))
        # a live gate reading, at most every 20 s and only on a settled spec
        field = {}
        if os.path.isfile(spec) and time.time() - last_gate > 20:
            last_gate = time.time()
            ok, matched, stages, fails, _tail = run_gate(job.family, timeout=300)
            field["check"] = {"ok": ok, "matched": matched, "stages": stages,
                              "fails": fails[:4]}
        job.flush(detail=detail, **field)
        time.sleep(2)

    cancelled = CANCELLED["flag"]                 # the signal may have landed
    if cancelled or timed_out:                    # just as the agent was dying
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError:
            proc.terminate()
        return ("cancelled" if cancelled else "timed_out"), None

    reader.join(timeout=10)
    pump_events(lines, seen, job.state)            # events that arrived at exit
    return "done", None


def run_repo(job, args, model, base, key, provider_host):
    """Everything for one repo: digest, family lock, the agent, the proof."""

    # 1 · the source digest the agent should read first
    job.flush(phase="reading", detail="reading the transformers source")
    digest = nn_arc.state_path(job.repo, ".sources.txt", args.state_dir)
    try:
        with open(digest, "w", encoding="utf-8") as f:
            subprocess.run([PY, os.path.join(ROOT, "tools", "reverse_model.py"),
                            "sources", job.family], cwd=ROOT, stdout=f,
                           stderr=subprocess.STDOUT, timeout=300)
        job.log("# source digest: %s" % digest)
    except subprocess.SubprocessError as e:  # noqa: BLE001
        job.log("# source digest failed: %s" % e)

    # 2 · one family, one spec: wait if another job holds the pen
    lock = acquire_lock(job, args.timeout)
    if lock is None:
        if CANCELLED["flag"]:
            return cancel_out(job)
        return timeout_out(job, args.timeout)

    # 3 · the job's own proof, then the row — with one repair pass if the
    #     connection audit finds boxes the wiring never reads
    state, msg = run_agent(job, args, model, base, key, provider_host,
                           agent_prompt(job, args))
    if state == "failed":
        return False, msg, {}
    if state == "cancelled":
        return cancel_out(job)
    if state == "timed_out":
        return timeout_out(job, args.timeout)

    merge_todos(job)
    mark_todos(job, "done", "active")
    ok, detail, extra = verify_and_store(job, timeout=max(300, args.timeout))
    if not ok and extra.get("repair") and not job.state.get("repaired"):
        job.state["repaired"] = True
        mark_todos(job, "done", "wait")
        job.log("# the audit found connection problems: asking the agent to fix them")
        state, msg = run_agent(job, args, model, base, key, provider_host,
                               repair_prompt(job, args),
                               detail="fixing the connection issues the audit found")
        if state == "failed":
            finish_todos(job, False, note=msg)
            return False, msg, {}
        if state == "cancelled":
            finish_todos(job, False, note="cancelled")
            return cancel_out(job)
        if state == "timed_out":
            finish_todos(job, False, note="timed out")
            return timeout_out(job, args.timeout)
        mark_todos(job, "done", "active")
        ok, detail, extra = verify_and_store(job, timeout=max(300, args.timeout))
    finish_todos(job, ok, note=detail if not ok else None)
    return ok, detail, extra


def cancel_out(job):
    job.flush(phase="done", done=True, ok=False, cancelled=True,
              detail="halted \u2014 the task was cancelled", error="cancelled")
    job.log("# cancelled \u2014 the task was stopped")
    return False, "halted \u2014 the task was cancelled", {}


def timeout_out(job, timeout):
    job.flush(phase="done", done=True, ok=False,
              detail="timed out", error="timed out after %ds" % timeout)
    job.log("# timed out")
    return False, "timed out", {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="repo id, url, local folder, or model_type")
    ap.add_argument("--family", default=None)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--state-dir", default=None)
    ap.add_argument("--task", default=None,
                    help="the agent_tasks row this job serves (progress mirrors into it)")
    args = ap.parse_args()
    args.state_dir = args.state_dir or nn_arc.state_dir()

    family = (args.family or "").strip()
    job = Job(args.source, family, args.state_dir, task=args.task)
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    if not family:
        try:
            family = family_of(args.source)
            job.family = family
            job.state["family"] = family
        except Exception as e:  # noqa: BLE001 — the repo is the thing being read
            why = "could not read the repo's config: %s" % e
            if "401" in str(e) or "403" in str(e):
                why += (" \u2014 the repo is gated: set HF_TOKEN (a token that can "
                        "read it) for the app's environment, then restart the task")
            job.log("# " + why)
            job.flush(phase="done", done=True, ok=False,
                      detail="the repo could not be read", error=why)
            return 1
    nn_arc.initial_todos(job.repo, args.state_dir)
    merge_todos(job)

    model = (os.environ.get("NNSTUDIO_REVERSE_MODEL") or
             os.environ.get("OPENCODE_MODEL") or "").strip()
    base = (os.environ.get("NNSTUDIO_REVERSE_BASE_URL") or "").strip()
    key = (os.environ.get("NNSTUDIO_REVERSE_API_KEY") or "").strip()
    direct = bool(base and key and model)          # an endpoint straight from .env
    provider_host = urlsplit(base).netloc if direct else None
    job.state["direct"] = direct
    job.log("# reverse-model job for %s (family %s) at %s"
            % (job.repo, family, time.strftime("%H:%M:%S")))
    if (base or key) and not direct:
        job.log("# note: .env has a base url/key but not both plus a model id"
                " — falling back to OpenCode's own setup")

    try:
        ok, detail, extra = run_repo(job, args, model, base, key, provider_host)
    except Exception as e:  # noqa: BLE001 — never die without a status
        import traceback
        job.log("# the job crashed:\n" + traceback.format_exc())
        ok, detail, extra = False, "the job crashed: %s" % e, {}
        try:
            finish_todos(job, False, note=detail)
        except Exception:  # noqa: BLE001
            pass
    fields = dict(extra or {})
    fields.pop("canvas", None)                     # the status stays small
    job.flush(phase="done", done=True, ok=bool(ok), detail=detail,
              error=None if ok else detail, **fields)
    job.log("# final: ok=%s — %s" % (ok, detail))
    return 0


if __name__ == "__main__":
    sys.exit(main())
