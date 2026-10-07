#!/usr/bin/env python3
"""The reverse-model toolbox — what the AI agent uses while authoring a spec.

    python tools/reverse_model.py list              # specs and their gates
    python tools/reverse_model.py sources llama     # the transformers code to read
    python tools/reverse_model.py inspect <repo>    # a repo's config + tensors
    python tools/reverse_model.py check [family…]   # run the conformance gate
    python tools/reverse_model.py audit <repo>      # what is wrong with the wiring
    python tools/reverse_model.py report <repo> …   # fill the job's TODO list

The `report` subcommand is the static progress the app shows on the box while
the agent works — one step at a time:

    report <repo> models NAME[,NAME…]   # after reading the repo: its models
    report <repo> start <key>           # reading | understand:NAME |
    report <repo> finish <key>          # reconstruct:NAME | pipeline
    report <repo> fail [--note TEXT]    # the step it is on cannot be done
    report <repo> complete              # everything done (the job's own call)

`specs/README.md` documents the spec format; `.opencode/commands/reverse-model.md`
is the agent workflow that ties these together.
"""
import importlib
import inspect
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "agent"))

import nn_engine  # noqa: E402

WANTED = ("ForCausalLM", "Model", "DecoderLayer", "Block", "Attention", "MLP",
          "RMSNorm", "LayerNorm", "RotaryEmbedding", "Conv1D", "Config")
WANTED_FUNCS = ("eager_attention_forward", "repeat_kv", "rotate_half",
                "apply_rotary_pos_emb", "create_causal_mask")


def _emit_from(path, seen, only=None):
    """Print the top-level classes/functions this family is made of — read
    from the file with ast so decorated v5 classes still come out whole."""
    import ast
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        tree = ast.parse(text)
    except Exception:  # noqa: BLE001
        return
    wanted = (lambda n: n in WANTED_FUNCS or
              any(n == s or n.endswith(s) for s in WANTED))
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        name = node.name
        if name in seen or (only is not None and not any(name == x for x in only)):
            continue
        if only is None and not wanted(name):
            continue
        seen.add(name)
        seg = ast.get_source_segment(text, node)
        if not seg:
            continue
        kind = "class" if isinstance(node, ast.ClassDef) else "def"
        print("\n==== %s %s ====" % (kind, name))
        print(seg.rstrip())


def sources(which):
    """Print the transformers code a spec author must read for a family."""
    from transformers import AutoConfig
    try:
        cfg = AutoConfig.for_model(which)
    except Exception as e:  # noqa: BLE001
        print("no config for %r: %s" % (which, e))
        return 1
    cfg_cls = type(cfg)
    mod = sys.modules[cfg_cls.__module__]
    print("# family %s — config %s from %s" % (which, cfg_cls.__name__, mod.__file__))
    print("# read the eager path, end to end — that is the math a spec mirrors.")
    seen = set()
    pkg = os.path.dirname(mod.__file__)
    for name in sorted(os.listdir(pkg)):
        if not name.endswith(".py"):
            continue
        if not (name.startswith("modeling") or name.startswith("configuration") or
                "attention" in name or "masking" in name):
            continue
        _emit_from(os.path.join(pkg, name), seen)
    # the shared eager attention + mask helpers, wherever they live
    for where in ("transformers.integrations", "transformers.modeling_utils",
                  "transformers.masking_utils",
                  "transformers.models.%s.modeling_%s" % (which, which)):
        try:
            shared = importlib.import_module(where)
        except Exception:  # noqa: BLE001
            continue
        _emit_from(shared.__file__, seen, only=WANTED_FUNCS)
    print("\n# check the result with:  python tools/reverse_model.py check %s" % which)
    return 0


def inspect_source(source):
    """A repo's config and tensors, grouped per layer — spec-drafting material."""
    meta, tensors, extra = nn_engine.hf_meta_and_tensors(source)
    print("# %s" % source)
    print(json.dumps(extra["cfg"], indent=2, sort_keys=True)[:4000])
    print("\n# blocks=%s d_model=%s heads=%s kv_heads=%s head_dim=%s"
          % (meta.get("blocks"), meta.get("d_model"), meta.get("heads"),
             meta.get("kv_heads"), meta.get("head_dim")))
    by_orig = extra["by_orig"]
    groups = {}
    for name, info in by_orig.items():
        g = re.sub(r"\.\d+\.", ".{i}.", name)
        groups.setdefault(g, []).append((name, info))
    print("\n# %d tensors, %d name shapes" % (len(by_orig), len(groups)))
    for g in sorted(groups):
        rows = groups[g]
        sample = rows[0][1]
        print("  %-58s %-6s %-14s x%d %s" % (g, sample["dtype"],
              str(sample["dims"]).replace(" ", ""), len(rows),
              "" if len(rows) == 1 else "(one per layer)"))
    mapped = sum(1 for name in by_orig if nn_engine._hf_translate(name))
    print("\n# %d/%d tensor names have a role in the approximate table; the spec "
          "may bind any of them by name" % (mapped, len(by_orig)))
    return 0


def check(args):
    cmd = [sys.executable, os.path.join(ROOT, "tests", "spec_conformance_test.py")] + list(args)
    return subprocess.call(cmd)


def list_specs():
    specs = nn_engine.load_specs()
    if not specs:
        print("no specs yet — specs/*.json")
        return 0
    for name in sorted(specs):
        spec = specs[name]
        ver = spec.get("verified") or {}
        types = ", ".join((spec.get("match") or {}).get("model_type") or [])
        print("%-10s %-28s transformers %-8s %s" % (
            name, types, ver.get("transformers", "?"), ver.get("date", "unverified")))
        if ver.get("note"):
            print("           %s" % ver["note"])
    return 0


TODO_MARKS = {"done": "\u2713", "active": "\u25cf", "failed": "\u2717"}


def print_todos(doc):
    print("# arc todos \u2014 %s" % doc.get("repo"))
    for i, it in enumerate(doc.get("items") or [], 1):
        line = "  %d  %s  %s" % (i, TODO_MARKS.get(it.get("state"), "\u00b7"),
                                 it.get("title") or it.get("key"))
        if it.get("note"):
            line += "  (%s)" % it["note"]
        print(line)


def report(args):
    """The agent's own checklist updates — one repo, one step at a time."""
    import nn_arc
    if len(args) < 2:
        print("usage: report <repo> models A,B | start <key> | finish <key> | "
              "fail [--note TEXT] | complete")
        return 2
    repo, action, rest = args[0], args[1], list(args[2:])
    note = None
    if "--note" in rest:
        i = rest.index("--note")
        note = " ".join(rest[i + 1:]).strip() or None
        rest = rest[:i]
    doc = nn_arc.read_todos(repo) or {"repo": repo, "items": []}
    items = doc.get("items") or []

    def mark(key, state):
        it = nn_arc.todo_find(items, key)
        if it is None:
            return False
        it["state"] = state
        return True

    if action == "models":
        names = [x.strip() for x in ",".join(rest).split(",") if x.strip()]
        if not names:
            print("usage: report <repo> models NAME[,NAME...]")
            return 2
        items = nn_arc.todo_items(names)
        mark("reading", "done")
        mark("understand:%s" % names[0], "active")
    elif action in ("start", "finish"):
        if not rest:
            print("usage: report <repo> %s <key>" % action)
            return 2
        if not mark(rest[0], "active" if action == "start" else "done"):
            print("no such step: %s" % rest[0])
            return 2
    elif action == "fail":
        act = next((x for x in items if x.get("state") == "active"), None)
        if act is None:
            print("nothing is active to fail")
            return 2
        act["state"] = "failed"
        if note:
            act["note"] = note
    elif action == "complete":
        for it in items:
            if it.get("state") != "failed":
                it["state"] = "done"
    else:
        print("unknown report action: %s" % action)
        return 2
    doc = nn_arc.write_todos(repo, {"repo": repo, "items": items})
    print_todos(doc)
    return 0


def audit_source(source):
    """Per-box connection problems in the repo's exact canvas — or in a saved
    canvas JSON. The numeric gate cannot see a drawing problem; this can."""
    if os.path.isfile(source) and source.endswith(".json"):
        with open(source, encoding="utf-8") as f:
            canvas = json.load(f)
    else:
        try:
            canvas = nn_engine.arc_canvas(source, use_arc=False)["canvas"]
        except nn_engine.ArcError as e:
            print("no exact canvas for %s: %s" % (source, e))
            return 2
    audit = nn_engine.audit_canvas(canvas)
    print("# canvas audit \u2014 %s (%d boxes)" % (source, audit["boxes"]))
    if audit["ok"]:
        print("  ok: every box is wired")
        return 0
    print("  %d connection issue%s in %d block%s:"
          % (audit["count"], "" if audit["count"] == 1 else "s",
             len(audit["blocks"]), "" if len(audit["blocks"]) == 1 else "s"))
    order = []
    for is_ in audit["issues"]:
        if is_["where"] not in order:
            order.append(is_["where"])
    for where in order:
        print("  %s" % where)
        for is_ in audit["issues"]:
            if is_["where"] == where:
                print("    %s %s: %s" % ("[!]" if is_["severity"] == "error" else "[~]",
                                         is_.get("box") or "?", is_["what"]))
    return 1


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("list", "sources", "inspect",
                                                "check", "audit", "report"):
        print(__doc__)
        return 2
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "list":
        return list_specs()
    if cmd == "sources":
        if not args:
            print("usage: sources <model_type>")
            return 2
        return sources(args[0])
    if cmd == "inspect":
        if not args:
            print("usage: inspect <repo-or-dir>")
            return 2
        return inspect_source(args[0])
    if cmd == "audit":
        if not args:
            print("usage: audit <repo-or-dir|canvas.json>")
            return 2
        return audit_source(args[0])
    if cmd == "report":
        return report(args)
    return check(args)


if __name__ == "__main__":
    sys.exit(main())
