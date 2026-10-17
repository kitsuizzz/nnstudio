# nnstudio's agent — your own backend

The app only draws. This is the program that **runs the design**, on the
machine you choose — ideally the one with the GPU.

## Start it

On the computer that should run the design:

    pip install torch numpy        # the CUDA build of torch to use an NVIDIA gpu;
                                   # on Apple Silicon the plain wheel uses the metal gpu
    pip install gguf               # optional — only for designs bound to real GGUF files
    pip install tokenizers         # optional — only for huggingface tokenizer repos
    pip install transformers       # optional — only for huggingface model boxes
    python nnstudio_agent.py --app <the-app>:8125

On Apple Silicon the agent picks the **metal gpu** (`mps`) automatically:
`place: gpu` (and the cpu-offload bar) land layers on it, `place: system`
keeps them in plain RAM — the memory is unified, but the placement and its
budgets are the same. A GGUF layer cannot run natively there —
metal has no Triton — so it dequantizes to f16 at load, exactly as it does on
a CPU-only machine; the memory numbers the app shows are the metal working
set torch recommends. An op with no metal port falls back to the cpu for that
one call (`PYTORCH_ENABLE_MPS_FALLBACK`, set by the agent), and a run the
metal working set cannot hold is retried on the cpu once, so a design always
runs — with the reason carried to the app's Spec.

The app listens for agents on port **8125** (change it with
`nn_server.py --control-port`). The agent **dials the app, sends this
machine's specs, and the connection stays open** — the app heartbeats it and
dispatches `run` commands over it. You can start it before or after the app;
it retries every few seconds if the app is not reachable yet.

    --app 127.0.0.1:8125     # host:port of the app (default 127.0.0.1:8125)
    --name gpu-box           # the name shown under Backend → Local
    --once                   # connect once; do not retry (handy in tests)

For a backend that **stays up**, install it as a user service — an agent
started by hand dies with the terminal's session, and the service comes back
on its own if it ever crashes:

    cp agent/nnstudio-agent.service ~/.config/systemd/user/   # edit the paths inside
    systemctl --user daemon-reload
    systemctl --user enable --now nnstudio-agent
    journalctl --user -u nnstudio-agent -f                    # its log

On macOS the same thing is a LaunchAgent:

    cp agent/nnstudio-agent.plist ~/Library/LaunchAgents/     # edit the paths inside
    launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/nnstudio-agent.plist
    tail -f /tmp/nnstudio-agent.log                           # its log

Once it is connected, the app's **Backend → Local** says connected, and the
app's **Run** menu shows this machine: its specs (CPU, cores, RAM, GPU, VRAM,
torch, OS), its **live RAM/VRAM usage** (the agent pushes `usage` every few
seconds), the two **load** buttons, and the input fields for a run. Running
the design sends the canvas and the input values here; the engine runs one
forward pass (CUDA when available) and sends every box's value back.

## What runs

`nn_torch.py` next to this file is the engine — the same one the app's tests
exercise offline (`tests/nn_torch_test.py`). It executes the canvas's boxes:
inputs, fully connected layers, embeddings, norms, activations, the tensor
operations, nested boxes and GGUF-bound weights. A Script box travels in the
design but is not executed yet. Nothing in the app itself imports torch; this
is where it lives.

**Placement.** Every weight box may carry `place: "gpu"` or `place: "system"`
(the app's right-click menu writes it, and the Run menu's *load all on …*
buttons write it for the whole design). The engine moves each box to its
device before its forward, so a network can split across VRAM and system RAM
mid-forward.

**External weights.** A weight box may carry `file` (a gguf, safetensors or
pt path — or an `hf:<repo>/<shard>` reference, fetched once into
`~/.cache/nnstudio/hf` and then read like any safetensors file) and `t` (the
tensor name(s) inside it — comma-separated for a weight plus a `.bias`). The
engine loads those tensors instead of the random init; a
file on an outer box reads **through nesting to any depth**, so a GGUF model's
blocks bind their tensors wherever they sit. `file_tensors(path)` is the
metadata the app shows when binding, and `export_weights(canvas, path, fmt)`
writes a whole design's tensors out as one pt / safetensors file. Keep bound
files reachable from the agent (relative paths resolve against the repo,
wherever the agent was started).

**Huggingface repos as canvases.** `hf_canvas(repo)` builds the same native
blocks from a repo's `config.json` and the safetensors **headers** (a range
request per shard — no weights): q/k/v/o, gate/up/down, the norms, embedding
and head, every box bound to the repo's own tensor name and shard, biases
riding with their weight, unknown tensors kept loose. The design stores
`file: "hf:<repo>/<shard>"`; the agent downloads a shard once (huggingface_hub
when present, a plain HTTPS fallback otherwise) the first time a box bound to
it runs, and `offload` never touches the cache on disk.

**Verified specs.** When a family has a spec (`specs/<family>.json`), the
canvas is built by `spec_canvas` instead of the role sketch — the spec's nodes
are native boxes only (the engine's ops now include `permute`, `repeat`,
`mask`, and a scalar `scale`), every weight is bound to its repo name and
shard, optional tensors (q/k norms, biases) simply elide, and the design says
`fidelity: "exact"`. A spec is authored by the reverse-model agent
(`/reverse-model`, `tools/reverse_model.py`) and is only accepted when
`tests/spec_conformance_test.py` matches every stage against the library's
eager run — the llama spec does so at 0.00e+00. Without a spec the canvas is
the older role-based skeleton, marked `approximate`, and the app says which.

**Background reverse jobs.** `POST /api/reverse {repo?, family}` starts (or
joins) `tools/reverse_job.py` for a family without a verified spec; the job
writes `/tmp/opencode/reverse/<family>.json` — phase (`reading` → `agent` →
`done`), elapsed seconds, the agent's latest words, and a live conformance
reading (`{ok, matched, stages, fails}`) — which `GET /api/reverse/status`
serves and the app paints on the box. The job runs detached
(`start_new_session`), pre-writes the source digest for the agent, and runs
the gate one last time itself, so `done`/`ok` never depends on the agent's
self-report. A verified family answers `done` instantly without spawning.

**The data flow.** A run reports every box as its value lands (`on_node`,
nested boxes included) — the agent streams that order to the app as
`progress` messages and returns it as `flow` with the result, and the canvas
animates the lights along the wires from it. A failure keeps the block it is
about: `CanvasError` carries the failing box's node id (attached at compile,
deep in nested graphs too), the agent's error reply includes it as `box` /
`nid`, and the app marks that block in red.

**Input values.** An input box may carry `input_mode`: `python` (its
`input_code` builds the matrix here — one expression, or statements leaving it
in `x`; `torch`, `np` and `math` are available) or `random` (a seeded
`randn(input_rows, size)`). `build_feeds` turns those into the run's feeds;
anything the app sends for the same input wins. A `values` input (no mode)
still takes what the Run menu types in.

**Text boxes.** A `tokenize` box takes **text from its wire** — an `input`
box with `input_mode: "text"` and a `text` value — or, unwired, from its own
`text`. It carries `vocab` (the characters; the index of a char is its id) or
a `repo` — a huggingface repository (`Qwen/Qwen3-8B`, `gpt2`, or a URL). With
a repo, its `tokenizer.json` is downloaded once (cached under
`~/.cache/nnstudio/tokenizers`, `HF_TOKEN` for gated repos) and loaded with
the `tokenizers` package — `pip install tokenizers`, only needed for this.
The ids travel the wire as a tensor. A `detokenize` box takes **ids only** —
a model's logits go through a **`sample`** box first — and its **output is the
text**: the app paints it on the card and returns it in the run's result under
`texts`. A `sample` box takes logits (any size) and picks **one id**: the last
position's distribution, softmax → temperature → top-k/top-p → multinomial,
with `temperature` / `top_k` / `top_p` / `sample_seed` on the box (temperature
0 is greedy).

An **`hf`** box is a whole model by repo: paste `Qwen/Qwen2.5-0.5B-Instruct`
(or any `AutoModelForCausalLM` repo) and the box takes **text** — from a wire,
or its own prompt — and answers with the **next token's text** (greedy, or
sampled with its temperature). The backend loads it with `transformers`
(`pip install transformers`, only needed for this box), on its box's device,
keeps it in memory while the agent lives, and `offload` lets it go. The first
run downloads the repo (the Run menu says so while it does); later runs reuse
it. The weights are not part of the canvas — the box is opaque to the engine's
compile estimate, so a huge repo is not refused up front: the download and its
memory land on the backend, not in the design.

**Warm loads.** A `load` command compiles the design, materialises every
weight on its chosen device and keeps the compiled graph here; a following
`run` of the same design reuses it (the result says `"loaded": true`) instead
of compiling again — and so does a **repeated `load`** (the same canvas and
device come back instantly with `"cached": true`). Change the design and the
next run recompiles — the cache is keyed by the canvas's content.

**Before a load spends minutes.** The agent estimates the design's weight
bytes straight from the JSON (`estimate_canvas`) and refuses a load that
cannot fit — the gpu side against `torch.cuda.mem_get_info`, the system side
against `/proc/meminfo` — with an actionable message, instead of dequantising
tensor after tensor and then dying. While a load that *can* fit runs, it
streams `{"type": "progress", …, "load": {bytes, total}}` messages (at most
every 1.5 s), which the app's Run menu shows as `loading… 55% · 3.0 / 5.4 GB`.
Weights also move straight to their device as their box is built: a
system-placed model never takes a detour through the GPU (that detour used to
stage the whole model in VRAM first, and made anything bigger than the card
impossible to place on the system).

An **`offload`** command lets all of it go —
the warm graph, the engine's GGUF readers and tokenizers, the CUDA pools — and
hands the freed pages back to the OS (glibc included), answering with the
usage before and after. A **failed** `load` or `run` also lets its half-built
graph go: the failing traceback's frames are cleared the moment it is caught
(their locals can hold the whole graph), the CUDA cache is emptied and the
pages go back — a run that dies does not sit on the machine.

**Weight dtypes.** Every weight box is held in the dtype it declares: `f32`,
`f16`, `bf16`, and anything quantised (`q4_k`, `iq4_xs`, …) as `f16` once
dequantised. The forward pass computes in f32 regardless, so mixed designs
just work — the storage follows the design, not the maths.

## The protocol

One TCP connection, newline-delimited JSON both ways:

    -> {"type": "hello", "name": ..., "specs": {...}}
    <- {"type": "welcome", "app": "nnstudio"}
    -> {"type": "usage", "usage": {...}}    (RAM/VRAM right now, every 5 s)
    <- {"type": "ping", "t": ...}          -> {"type": "pong", "t": ...}
    <- {"type": "run", "id": ..., "canvas": ..., "feeds": {...}}
    -> {"type": "progress", "id": ..., "nodes": [...], "done": n}   (during a run)
    -> {"type": "progress", "id": ..., "load": {bytes, total}}     (during a load)
    -> {"type": "result", "id": ..., "ok": true, "result": {..., "flow": [...]}}
    <- {"type": "load", "id": ..., "canvas": ...}
    -> {"type": "result", "id": ..., "ok": true, "result": {params, bytes,
       on_gpu_bytes, on_system_bytes, ...}}
    <- {"type": "offload", "id": ...}
    -> {"type": "result", "id": ..., "ok": true, "result": {before, after}}

## A caveat

The agent runs whatever design the app dispatches — it trusts the app, and the
app trusts whoever can reach its control port. Keep both on a network you
control.
