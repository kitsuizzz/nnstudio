#!/usr/bin/env python3
"""The diffusers gate: a tiny random UNet2DConditionModel, run through
diffusers' own eager path and through the canvas the builder makes of it —
conv_in, every down/mid/up block, the final conv: all of it to 1e-4.

No network: the model is constructed here, seeded, and saved to a temp folder.
Every tensor the model has must be bound by exactly one box.

    .venv/bin/python tests/diffusers_unet_test.py
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "agent"))
os.environ.setdefault("NNSTUDIO_ARC_DB",
                      os.path.join(tempfile.mkdtemp(prefix="nnstudio-arc-"), "hf_arc.db"))

import torch                                                    # noqa: E402

import nn_engine                                                # noqa: E402
import nn_torch as T                                            # noqa: E402

SIDE = 8
CTX_LEN = 5


def tiny_unet(seed=0):
    from diffusers import UNet2DConditionModel
    torch.manual_seed(seed)
    cfg = dict(
        sample_size=SIDE, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(16, 32),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=12, attention_head_dim=(4, 4),
        norm_num_groups=4, norm_eps=1e-6, downsample_padding=1,
        use_linear_projection=True, transformer_layers_per_block=1,
    )
    m = UNet2DConditionModel(**cfg).eval()
    d = tempfile.mkdtemp(prefix="nnstudio-tiny-unet-")
    m.save_pretrained(d)
    return m, d


def canvas_tensor(vals, nid):
    v = vals.get(nid)
    if v is None:
        raise SystemExit("the canvas never made %r" % nid)
    return v[None] if v.dim() == 3 else v              # [H, W, C] -> one sample


def main():
    m, folder = tiny_unet()
    built = nn_engine.diffusers_unet_canvas(folder)
    canvas = built["canvas"]

    # every tensor the model has is bound, by exactly one box
    bound = set()
    def walk(inner):
        for n in inner.get("nodes", []):
            for tn in [x.strip() for x in str(n.get("t") or "").split(",") if x.strip()]:
                bound.add(tn)
            if n.get("inner"):
                walk(n["inner"])
    walk(canvas)
    model_names = set(m.state_dict().keys())
    missing = model_names - bound
    extra = bound - model_names
    assert not missing, "unbound tensors: %s" % sorted(missing)[:8]
    assert not extra, "bounds that are not the model's: %s" % sorted(extra)[:8]

    # what the builder cannot mirror, it names instead of approximating
    import shutil
    bad = os.path.join(tempfile.mkdtemp(prefix="nnstudio-sdxlish-"), "unet")
    shutil.copytree(folder, bad)
    with open(os.path.join(bad, "config.json"), encoding="utf-8") as f:
        bcfg = json.load(f)
    bcfg["addition_embed_type"] = "text_time"
    with open(os.path.join(bad, "config.json"), "w", encoding="utf-8") as f:
        json.dump(bcfg, f)
    try:
        nn_engine.diffusers_unet_canvas(bad)
        raise SystemExit("an SDXL-style config was not refused")
    except ValueError as e:
        assert "addition_embed_type" in str(e), str(e)

    # the reference eager pass, with every stage caught for the comparison
    torch.manual_seed(1)
    sample = torch.randn(1, 4, SIDE, SIDE)
    t = torch.tensor([500.0])
    ctx = torch.randn(1, CTX_LEN, 12)
    caught = {}
    watched = ["conv_in", "conv_norm_out", "conv_out"]
    for d, block in enumerate(m.down_blocks):
        watched += ["down_blocks.%d.resnets.%d" % (d, i) for i in range(len(block.resnets))]
        for i, _ in enumerate(getattr(block, "attentions", [])):
            watched.append("down_blocks.%d.attentions.%d" % (d, i))
        if block.downsamplers is not None:
            watched.append("down_blocks.%d.downsamplers.0" % d)
    watched += ["mid_block.resnets.0", "mid_block.attentions.0", "mid_block.resnets.1"]
    for u, block in enumerate(m.up_blocks):
        watched += ["up_blocks.%d.resnets.%d" % (u, i) for i in range(len(block.resnets))]
        for i, _ in enumerate(getattr(block, "attentions", [])):
            watched.append("up_blocks.%d.attentions.%d" % (u, i))
        if block.upsamplers is not None:
            watched.append("up_blocks.%d.upsamplers.0" % u)
    mods = dict(m.named_modules())
    for path in watched:
        mod = mods.get(path)
        if mod is None:
            raise SystemExit("no module %r to watch" % path)
        mod.register_forward_hook(
            (lambda p: lambda mod, inp, out: caught.__setitem__(
                p, out[0] if isinstance(out, tuple) else out))(path))
    with torch.no_grad():
        ref = m(sample, t, encoder_hidden_states=ctx, return_dict=False)[0]

    # the canvas, on the same inputs (channels last, one sample): the outer
    # compile proves the model box's ports; the inner graph carries the stages
    px = sample[0].permute(1, 2, 0)
    g_out = T.compile_canvas(canvas)
    out_vals = g_out(feeds={"latents": px, "timestep": t.clone(), "context": ctx[0]})
    inner = [n for n in canvas["nodes"] if n.get("kind") == "custom"][0]["inner"]
    g = T.compile_canvas(inner)
    vals = g(feeds={"latents": px, "t": t.clone(), "context": ctx[0]})

    def compare(label, node, ref_v):
        got = canvas_tensor(vals, node)
        ref_b = ref_v.permute(0, 2, 3, 1) if ref_v.dim() == 4 else ref_v
        assert tuple(got.shape) == tuple(ref_b.shape), \
            "%s: canvas %s vs reference %s" % (label, list(got.shape), list(ref_b.shape))
        dmax = (got - ref_b).abs().max().item()
        scale = ref_b.abs().max().item() or 1.0
        assert dmax <= 1e-4, ("%s: canvas is off by %.3g (values to %.3g)"
                              % (label, dmax, scale))
        return dmax

    worst = 0.0
    worst_at = ""
    def note(dmax, label):
        nonlocal worst, worst_at
        if dmax > worst:
            worst, worst_at = dmax, label

    # stage by stage: the residual add ends a resnet, the final add a block
    def walk_nodes(inner):
        out = []
        def go(inner):
            for n in inner.get("nodes", []):
                out.append(n)
                if n.get("inner"):
                    go(n["inner"])
        go(inner)
        return out

    ids = {n.get("id") for n in walk_nodes(canvas)}
    checks = []
    checks.append(("conv_in", "conv_in", caught["conv_in"]))
    for path, ref_v in caught.items():
        if path == "conv_in":
            continue
        node = None
        if ".resnets." in path:
            node = nn_engine._did(path + "#res")
        elif ".attentions." in path:
            node = nn_engine._did(path + "#out")
        elif "downsamplers" in path or "upsamplers" in path:
            node = nn_engine._did(path + ".conv")
        elif path in ("conv_norm_out", "conv_out"):
            node = path
        if node is None or node not in ids:
            raise SystemExit("the canvas has no node for %r" % path)
        checks.append((path, node, ref_v))

    for path, node, ref_v in checks:
        if isinstance(ref_v, tuple):
            continue
        dmax = compare(path, node, ref_v)
        note(dmax, path)

    dmax = compare("the output", "sample", ref)
    note(dmax, "the output")
    got = out_vals["out"].permute(2, 0, 1)[None]
    dmax = (got - ref).abs().max().item()
    assert dmax <= 1e-4, "the model box's output is off by %.3g" % dmax
    note(dmax, "the model box")
    print("  ok   the tiny UNet matches stage for stage (worst %.2e at %s)"
          % (worst, worst_at))
    print("diffusers_unet: all assertions passed (%d stages, %d tensors bound)"
          % (len(checks) + 1, len(bound)))


if __name__ == "__main__":
    main()
