#!/usr/bin/env python3
"""The checkpoint through **transformers alone** — a baseline for the canvas
engine. Loads the repo with `AutoModelForCausalLM.from_pretrained`, whatever
quantization backend the checkpoint declares (this pilot is compressed-tensors
mixed precision: fp8 + nvfp4), splits it with `device_map`, then continues a
prompt greedily and reports prefill and per-token times.

    python text.py                                     # the pilot, 8 tokens
    python text.py --prompt "Once upon a time" --max-new-tokens 16
    python text.py --device-map cuda:0                 # whole gpu (22 GB: no)
    python text.py --device-map auto --load-only       # just the load + split

Notes: a 22 GB checkpoint cannot fit 15.5 GB of VRAM, so `cuda:0` is expected
to fail and `auto` splits it. The tokenizer comes from nnstudio's cache when
it is there (the hub snapshot has none), so no download is needed.

Known wall: transformers' compressed-tensors path **decompresses** the packed
nvfp4 weights to bf16 on the first forward — ~40.7 GB of dense weights for
20.35B params. A 15.5 GB card + 29 GB of ram cannot hold that, so generation
dies with OOM however the load is split. The packed-at-rest canvas engine
(nnstudio) exists for exactly that reason; a fused runtime (vLLM/SGLang with
nvfp4 kernels) is transformers-free. The load itself works and is reported.
"""
import argparse
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

REPO = "unsloth/Qwen3.8-27B-NVFP4"
PROMPT = "The capital of France is"
TOKENIZER_CACHE = os.path.expanduser("~/.cache/nnstudio/tokenizers")


def load_tokenizer(repo):
    """The repo's tokenizer.json — nnstudio's cache first, the hub second."""
    local = os.path.join(TOKENIZER_CACHE, repo.replace("/", "__"), "tokenizer.json")
    if os.path.isfile(local):
        from transformers import PreTrainedTokenizerFast
        return PreTrainedTokenizerFast(tokenizer_file=local), local
    return AutoTokenizer.from_pretrained(repo), repo


def report_split(model):
    """Where the weights sit, in bytes, and how many modules per device.
    Returns the parameter count (iterating parameters again later would poke
    compressed-tensors' offload cache and need the gpu)."""
    by_dev = {}
    total = 0
    for p in model.parameters():
        dev = str(p.device)
        by_dev[dev] = by_dev.get(dev, 0) + p.numel() * p.element_size()
        total += p.numel()
    for dev, n in sorted(by_dev.items(), key=lambda kv: -kv[1]):
        print("  weights on %-9s %6.2f GB" % (dev, n / 2 ** 30))
    dm = getattr(model, "hf_device_map", None)
    if dm:
        counts = {}
        for name, dev in dm.items():
            counts[str(dev)] = counts.get(str(dev), 0) + 1
        print("  modules per device:", dict(sorted(counts.items(),
                                                   key=lambda kv: -kv[1])))
    return total


class StepTimer(StoppingCriteria):
    """One timestamp per forward — generate calls it after every step."""

    def __init__(self):
        self.times = []

    def __call__(self, input_ids, scores, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.times.append(time.time())
        return False


def main():
    ap = argparse.ArgumentParser(description="transformers-only baseline")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--device-map", default="auto",
                    help="auto (gpu+system), a device like cuda:0, or cpu")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--load-only", action="store_true", help="load and split, no generation")
    args = ap.parse_args()

    tok, from_where = load_tokenizer(args.repo)
    print("tokenizer:", from_where)
    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    print("loading %s (device_map=%s, dtype=%s)\u2026"
          % (args.repo, args.device_map, args.dtype), flush=True)
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.repo, device_map=args.device_map, dtype=dtype, low_cpu_mem_usage=True)
    except (torch.OutOfMemoryError, RuntimeError) as e:
        msg = str(e)
        if "out of memory" in msg.lower() or "allocate" in msg.lower():
            print("LOAD FAILED \u2014 out of memory with device_map=%s." % args.device_map)
            print("  the checkpoint is ~22 GB; use --device-map auto to split it.")
            return 2
        raise
    model.eval()
    print("loaded in %.1fs" % (time.time() - t0), flush=True)
    n_params = report_split(model)
    print("  model dtype:", next(model.parameters()).dtype)

    if tok.pad_token_id is None and model.config.eos_token_id is not None:
        tok.pad_token_id = model.config.eos_token_id
    for name in ("eos_token_id", "bos_token_id"):
        if getattr(tok, name, None) is None and getattr(model.config, name, None) is not None:
            setattr(tok, name, getattr(model.config, name))
    if getattr(tok, "padding_side", None) is None:
        tok.padding_side = "left"

    if args.load_only:
        return 0

    enc = tok(args.prompt, return_tensors="pt")
    dev = getattr(model, "device", None) or "cpu"
    input_ids = enc["input_ids"].to(dev)
    print("generating %d tokens (%s) on %s\u2026"
          % (args.max_new_tokens, "greedy", dev), flush=True)
    timer = StepTimer()
    t0 = time.time()
    try:
        with torch.no_grad():
            out = model.generate(input_ids, attention_mask=enc.get("attention_mask",
                                                                   torch.ones_like(input_ids)).to(dev),
                                 max_new_tokens=args.max_new_tokens, do_sample=False,
                                 use_cache=True, stopping_criteria=StoppingCriteriaList([timer]))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except (torch.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower():
            raise
        print("GENERATE FAILED \u2014 out of memory on the first forward.")
        print("  compressed-tensors decompresses the packed weights to bf16 before")
        print("  computing: %.1fB params \u2192 ~%.0f GB dense, but this box has"
              % (n_params / 1e9, n_params * 2 / 2 ** 30))
        print("  15.5 GB vram + ~26 GB free ram. No device_map split changes that;")
        print("  this checkpoint needs a fused nvfp4 runtime (vLLM/SGLang), or the")
        print("  packed-at-rest engine (nnstudio), to run here.")
        return 2
    wall = time.time() - t0
    text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    print("prompt:", repr(args.prompt))
    print("text:  ", repr(text))
    steps = [t - t0 for t in timer.times]
    if len(steps) >= 2:
        gaps = [steps[i] - steps[i - 1] for i in range(1, len(steps))]
        avg = sum(gaps) / len(gaps)
        print("timing: prefill+1 %.2fs | %.2fs/token (%.2f tok/s) | total %.2fs"
              % (steps[0], avg, 1.0 / avg if avg else 0.0, wall))
    else:
        print("timing: total %.2fs" % wall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
