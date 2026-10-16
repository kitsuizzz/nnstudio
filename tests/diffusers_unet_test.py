#!/usr/bin/env python3
"""The diffusers gate: tiny random UNet2DConditionModels, run through
diffusers' own eager path and through the canvas the builder makes of them —
conv_in, every down/mid/up block, the final conv: all of it to 1e-4. Two
scenarios: a plain model, and one with SDXL's `text_time` conditioning (time
ids + pooled text embeds riding beside the cross-attention context).

No network: the models are constructed here, seeded, and saved to temp folders.
Every tensor a model has must be bound by exactly one box.

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
TEXT_DIM = 5
TIME_IDS = 3


def tiny_unet(seed=0, kind="plain"):
    """kind: "plain" (both block flavours), "text_time" (SDXL's conditioning),
    or "sdxl_shape" (SDXL's whole topology at tiny channels: a plain first down
    stage, three up stages, [1, 2, 3] transformer layers, two layers a block)."""
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
    if kind == "text_time":
        cfg.update(addition_embed_type="text_time", addition_time_embed_dim=6,
                   projection_class_embeddings_input_dim=TEXT_DIM + TIME_IDS * 6)
    elif kind == "sdxl_shape":
        cfg.update(layers_per_block=2, transformer_layers_per_block=(1, 2, 3),
                   block_out_channels=(16, 32, 48),
                   attention_head_dim=(4, 4, 8),
                   down_block_types=("DownBlock2D", "CrossAttnDownBlock2D",
                                     "CrossAttnDownBlock2D"),
                   up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
                                   "UpBlock2D"),
                   addition_embed_type="text_time", addition_time_embed_dim=6,
                   projection_class_embeddings_input_dim=TEXT_DIM + TIME_IDS * 6,
                   norm_num_groups=4, norm_eps=1e-5)
    m = UNet2DConditionModel(**cfg).eval()
    d = tempfile.mkdtemp(prefix="nnstudio-tiny-unet-")
    m.save_pretrained(d)
    return m, d


def walk_nodes(doc):
    out = []

    def go(inner):
        for n in inner.get("nodes", []):
            out.append(n)
            if n.get("inner"):
                go(n["inner"])
    go(doc)
    return out


def canvas_tensor(vals, nid):
    v = vals.get(nid)
    if v is None:
        raise SystemExit("the canvas never made %r" % nid)
    return v[None] if v.dim() == 3 else v              # [H, W, C] -> one sample


def gate_scenario(label, m, folder, sdxl):
    built = nn_engine.diffusers_unet_canvas(folder)
    canvas = built["canvas"]

    # every tensor the model has is bound, by exactly one box
    bound = set()
    for n in walk_nodes(canvas):
        for tn in [x.strip() for x in str(n.get("t") or "").split(",") if x.strip()]:
            bound.add(tn)
    model_names = set(m.state_dict().keys())
    missing = model_names - bound
    extra = bound - model_names
    assert not missing, "%s: unbound tensors: %s" % (label, sorted(missing)[:8])
    assert not extra, "%s: bounds that are not the model's: %s" % (label, sorted(extra)[:8])
    ports = [n for n in canvas["nodes"] if n.get("kind") == "input"]
    want_ports = 5 if sdxl else 3
    assert len(ports) == want_ports, (label, [p["name"] for p in ports])

    # the reference eager pass, with every stage caught for the comparison
    torch.manual_seed(1)
    sample = torch.randn(1, 4, SIDE, SIDE)
    t = torch.tensor([500.0])
    ctx = torch.randn(1, CTX_LEN, 12)
    ref_feeds = {}
    canvas_feeds = {}
    if sdxl:
        te = torch.randn(1, TEXT_DIM)
        ti = torch.randn(1, TIME_IDS)
        ref_feeds["added_cond_kwargs"] = {"text_embeds": te, "time_ids": ti}
        canvas_feeds["text_embeds"] = te[0]
        canvas_feeds["time_ids"] = ti[0]

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
        ref = m(sample, t, encoder_hidden_states=ctx, return_dict=False, **ref_feeds)[0]

    # the canvas, on the same inputs (channels last, one sample): the outer
    # compile proves the model box's ports; the inner graph carries the stages
    px = sample[0].permute(1, 2, 0)
    g_out = T.compile_canvas(canvas)
    feeds = {"latents": px, "timestep": t.clone(), "context": ctx[0]}
    feeds.update(canvas_feeds)
    out_vals = g_out(feeds=feeds)
    inner_doc = [n for n in canvas["nodes"] if n.get("kind") == "custom"][0]["inner"]
    g = T.compile_canvas(inner_doc)
    inner_feeds = {"latents": px, "t": t.clone(), "context": ctx[0]}
    if sdxl:
        inner_feeds["time_ids"] = canvas_feeds["time_ids"]
        inner_feeds["text_embeds"] = canvas_feeds["text_embeds"]
    vals = g(feeds=inner_feeds)

    def compare(what, node, ref_v):
        got = canvas_tensor(vals, node)
        ref_b = ref_v.permute(0, 2, 3, 1) if ref_v.dim() == 4 else ref_v
        assert tuple(got.shape) == tuple(ref_b.shape), \
            "%s: canvas %s vs reference %s" % (what, list(got.shape), list(ref_b.shape))
        dmax = (got - ref_b).abs().max().item()
        tol = 1e-4 * (1.0 + ref_b.abs().max().item())   # float32, stage by stage
        assert dmax <= tol, ("%s: canvas is off by %.3g (values to %.3g)"
                             % (what, dmax, ref_b.abs().max().item()))
        return dmax

    worst, worst_at = 0.0, ""

    def note(dmax, what):
        nonlocal worst, worst_at
        if dmax > worst:
            worst, worst_at = dmax, what

    checks = [("conv_in", "conv_in", caught["conv_in"])]
    ids = {n.get("id") for n in walk_nodes(canvas)}
    for path, ref_v in caught.items():
        if path == "conv_in":
            continue
        if ".resnets." in path:
            node = nn_engine._did(path + "#res")
        elif ".attentions." in path:
            node = nn_engine._did(path + "#out")
        elif path in ("conv_norm_out", "conv_out"):
            node = path
        else:
            node = nn_engine._did(path + ".conv")
        if node not in ids:
            raise SystemExit("%s: the canvas has no node for %r" % (label, path))
        checks.append((path, node, ref_v))

    for path, node, ref_v in checks:
        note(compare(path, node, ref_v), path)
    note(compare("the output", "sample", ref), "the output")
    got = out_vals["out"].permute(2, 0, 1)[None]
    dmax = (got - ref).abs().max().item()
    assert dmax <= 1e-4, "%s: the model box's output is off by %.3g" % (label, dmax)
    note(dmax, "the model box")
    print("  ok   %s: %d stages, %d tensors bound, worst %.2e at %s"
          % (label, len(checks) + 1, len(bound), worst, worst_at))
    return len(checks) + 1, len(bound)


def main():
    # what the builder cannot mirror, it names instead of approximating
    import shutil
    m0, folder0 = tiny_unet()
    bad = os.path.join(tempfile.mkdtemp(prefix="nnstudio-unmirrored-"), "unet")
    shutil.copytree(folder0, bad)
    with open(os.path.join(bad, "config.json"), encoding="utf-8") as f:
        bcfg = json.load(f)
    bcfg["center_input_sample"] = True
    with open(os.path.join(bad, "config.json"), "w", encoding="utf-8") as f:
        json.dump(bcfg, f)
    try:
        nn_engine.diffusers_unet_canvas(bad)
        raise SystemExit("an unmirrored config was not refused")
    except ValueError as e:
        assert "center_input_sample" in str(e), str(e)

    stages = tensors = 0
    for label, kind, sdxl in (
            ("a plain tiny unet", "plain", False),
            ("with SDXL's text_time", "text_time", True),
            ("SDXL's whole block shape", "sdxl_shape", True)):
        m, folder = tiny_unet(kind=kind)
        n, b = gate_scenario(label, m, folder, sdxl=sdxl)
        stages, tensors = stages + n, tensors + b
    print("diffusers_unet: all assertions passed (%d stages, %d tensors bound)"
          % (stages, tensors))


if __name__ == "__main__":
    main()
