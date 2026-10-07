#!/usr/bin/env python3
"""Spec conformance: the canvas a family spec builds vs the transformers library.

For every spec (or the families named on the command line), a tiny random
model of that family is built with transformers (eager attention), its state
dict exported to safetensors, the canvas built from the spec, and every stage
compared against the library's own run: the embedding, per-layer q/k/v
projections, every block's output and the logits. A spec is "ground truth"
only when this passes — that is the gate the reverse-model agent iterates on.

    python tests/spec_conformance_test.py            # every spec
    python tests/spec_conformance_test.py llama      # one family
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "agent"))

import nn_engine  # noqa: E402

TOL = 1e-4


def find_box(graph, name, kind=None):
    for box in graph.boxes.values():
        if box.name == name and (kind is None or box.kind == kind):
            return box
        if box.kind == "custom" and box.inner is not None:
            hit = find_box(box.inner, name, kind)
            if hit is not None:
                return hit
    return None


def block_of(graph, label):
    for box in graph.boxes.values():
        if box.kind == "custom" and box.name == label:
            return box
        if box.kind == "custom" and box.inner is not None:
            hit = block_of(box.inner, label)
            if hit is not None:
                return hit
    return None


def value_of(box):
    if box is None:
        return None
    v = box.last
    if isinstance(v, dict):
        v = next((x for x in v.values() if x is not None), None)
    return v


def compare(tag, got, want, fails):
    if got is None:
        fails.append("%s: the canvas reported nothing" % tag)
        return
    got = got.detach().float()
    want = want.detach().float()
    if tuple(got.shape) != tuple(want.shape):
        fails.append("%s: canvas %r vs library %r" % (tag, tuple(got.shape), tuple(want.shape)))
        return
    err = float((got - want).abs().max())
    flag = "" if err <= TOL else "   <-- over tolerance"
    print("    %-24s %-12s max|diff| %.2e%s"
          % (tag, str(tuple(want.shape)).replace(" ", ""), err, flag))
    if err > TOL:
        fails.append("%s: max|diff| %.3e > %.1e" % (tag, err, TOL))


def check_spec(spec):
    family = spec["family"]
    test = spec.get("test") or {}
    types = (spec.get("match") or {}).get("model_type") or []
    model_type = test.get("model_type") or (types[0] if types else None)
    print("== %s (%s)" % (family, model_type))
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except Exception as e:  # noqa: BLE001
        print("  skip: transformers is not installed here (%s)" % str(e)[:70])
        return "skip"
    try:
        cfg = AutoConfig.for_model(model_type, **test.get("config", {}))
        try:
            model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")
        except TypeError:                        # older/newer signature
            cfg._attn_implementation = "eager"
            model = AutoModelForCausalLM.from_config(cfg)
    except Exception as e:  # noqa: BLE001
        print("  skip: could not build %s here (%s)" % (model_type, str(e)[:70]))
        return "skip"
    model.eval()

    import nn_torch as T
    sd = {k: v.detach().float() for k, v in model.state_dict().items()}
    tmp = tempfile.TemporaryDirectory()
    try:
        with open(os.path.join(tmp.name, "config.json"), "w", encoding="utf-8") as f:
            f.write(cfg.to_json_string())
        T.write_safetensors(os.path.join(tmp.name, "model.safetensors"), sd)
        out = nn_engine.hf_canvas(tmp.name)
        canvas = out["canvas"]
        meta = canvas.get("model") or {}
        if meta.get("fidelity") != "exact" or meta.get("spec") != family:
            print("  FAIL: the spec did not build the canvas: %r" % meta)
            return "fail"

        graph = T.compile_canvas(canvas, device="cpu")
        graph.eval()
        tokens = next(n for n in canvas["nodes"] if n["kind"] == "input")
        ids = torch.tensor([1, 5, 9])
        with torch.no_grad():
            graph(feeds={tokens["id"]: ids})

        caps = {}
        hooks = []

        def hook(name):
            def fn(mod, args, out):
                caps[name] = out
            return fn

        hooks.append(model.model.embed_tokens.register_forward_hook(hook("embed")))
        n_layers = len(model.model.layers)
        n_full = 0
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, "self_attn"):       # full-attention layers only
                n_full += 1
                for proj in ("q_proj", "k_proj", "v_proj"):
                    hooks.append(getattr(layer.self_attn, proj)
                                 .register_forward_hook(hook("l%d.%s" % (i, proj))))

            def layer_hook(mod, args, out, i=i):
                # v5 ties hidden_states[-1] to the post-final-norm state; the
                # layer module's own output is the true pre-norm state
                caps["layer%d" % i] = out[0] if isinstance(out, tuple) else out
            hooks.append(layer.register_forward_hook(layer_hook))
        with torch.no_grad():
            hf = model(ids.unsqueeze(0))
        for h in hooks:
            h.remove()

        fails = []
        compare("embedding", value_of(find_box(graph, "token embedding")),
                caps["embed"][0], fails)
        for i in (0, n_layers - 1):
            if not hasattr(model.model.layers[i], "self_attn"):
                continue
            blk = block_of(graph, "blk %d \u00b7 %s" % (i, family))
            for proj in ("q", "k", "v"):
                got = value_of(find_box(blk.inner, "%s proj" % proj)) if blk else None
                compare("l%d.%s proj" % (i, proj), got,
                        caps["l%d.%s_proj" % (i, proj)][0], fails)
        for i in range(n_layers):
            blk = block_of(graph, "blk %d \u00b7 %s" % (i, family))
            got = value_of(find_box(blk.inner, "out", "output")) if blk else None
            compare("block %d out" % i, got, caps["layer%d" % i][0], fails)
        compare("logits", value_of(find_box(graph, "logits", "output")), hf.logits[0], fails)

        if fails:
            for f in fails:
                print("  FAIL: %s" % f)
            return "fail"
        print("  ok: %d stages matched (tol %.0e)" % (3 + n_layers + 3 * n_full, TOL))
        return "ok"
    finally:
        tmp.cleanup()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    specs = nn_engine.load_specs()
    names = args or sorted(specs)
    if not specs:
        print("spec conformance: no specs/ files — nothing to check")
        return 0
    failed, skipped, ok = 0, 0, 0
    for name in names:
        spec = specs.get(name)
        if spec is None:
            print("== %s: no such spec" % name)
            failed += 1
            continue
        res = check_spec(spec)
        ok += res == "ok"
        skipped += res == "skip"
        failed += res == "fail"
    print("spec conformance: %d ok, %d failed, %d skipped" % (ok, failed, skipped))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
