#!/usr/bin/env python3
"""The vision templates, built by the UI, run by the engine: node dumps the
docs bxAddTemplate lands (tests/vision_templates_dump.js), and here they are
compiled and swept — the image out for the VAE, UNet and DiT, class scores for
the ViT, and gradients through the whole VAE for its unsupervised train.

    .venv/bin/python tests/vision_templates_test.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "agent"))
import torch                                                    # noqa: E402

import nn_torch as T                                            # noqa: E402

SIDE = 32


def dump():
    r = subprocess.run(["node", os.path.join(HERE, "vision_templates_dump.js")],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not r.stdout.strip():
        raise SystemExit("the template dump failed:\n" + (r.stdout + r.stderr)[-3000:])
    return json.loads(r.stdout)


def design(tpl, d):
    """One template inside a top-level design: an input box per port, the model
    box, and an output — the wiring the canvas would carry."""
    nodes = [{"id": "in0", "kind": "input", "name": d["in"][0]["name"],
              "x": 0, "y": 0, "dout": d["in"][0]["dout"]}]
    edges = [{"from": "in0", "to": "model", "fromPort": 0, "toPort": 0}]
    for i, p in enumerate(d["in"][1:], start=1):
        nodes.append({"id": "in%d" % i, "kind": "input", "name": p["name"],
                      "x": 0, "y": 200 * i})
        edges.append({"from": "in%d" % i, "to": "model", "fromPort": 0, "toPort": i})
    nodes.append({"id": "model", "kind": "custom", "name": tpl, "x": 400, "y": 0,
                  "model": True, "inner": d["inner"]})
    nodes.append({"id": "out", "kind": "output", "name": "out", "x": 800, "y": 0,
                  "din": d["out"][0]["din"]})
    edges.append({"from": "model", "to": "out", "fromPort": 0, "toPort": 0})
    return {"type": "canvas", "version": 1, "nodes": nodes, "edges": edges}


def main():
    dumps = dump()
    assert sorted(dumps) == ["dit", "unet", "vae", "vit"], sorted(dumps)
    px = torch.rand(SIDE, SIDE, 3)
    checked = 0
    for tpl, want in (("vit", [10]), ("vae", [SIDE, SIDE, 3]),
                      ("unet", [SIDE, SIDE, 3]), ("dit", [SIDE, SIDE, 3])):
        d = dumps[tpl]
        assert d["model"] is True, tpl + " lands as a model box"
        assert len(d["inner"]["nodes"]) > 12, (tpl, len(d["inner"]["nodes"]))
        assert len(d["in"]) >= 1 and len(d["out"]) == 1, (tpl, d["in"], d["out"])
        g = T.compile_canvas(design(tpl, d), train=True)
        feeds = {"in0": px.clone()}
        for i, p in enumerate(d["in"][1:], start=1):
            feeds["in%d" % i] = torch.tensor([5.0])            # the step number
        out = g(feeds=feeds)["out"]
        assert list(out.shape) == want, (tpl, list(out.shape))
        assert torch.isfinite(out).all(), tpl + " went non-finite"
        if tpl == "vae":                       # reconstruction is the MSE run
            loss = torch.nn.functional.mse_loss(out, px)
            loss.backward()
            allp = list(g.parameters())
            grads = [p for p in allp if p.grad is not None]
            assert len(grads) == len(allp) and len(grads) >= 10, \
                "the gradient reaches every VAE weight: %d of %d" % (len(grads), len(allp))
        checked += 1
        print("  ok   %s: %d dot(s) in, %s out"
              % (tpl, len(d["in"]), "x".join(str(x) for x in want)))
    assert checked == 4
    print("vision_templates: all assertions passed")


if __name__ == "__main__":
    main()
