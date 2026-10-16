# nnstudio

A dedicated app for designing a neural network — **one canvas** with twelve
kinds of boxes, built straight from a GGUF file (or by hand), wired live,
priced for VRAM, and saved as JSON. It only draws: the whole design **runs on
a backend of your own** — a small agent you start on the machine that holds
the GPU, which dials the app and stays connected (see **Backend** and
**Run** below).

## Running the app

    ./run.sh                 # http://localhost:8124
    ./run.sh --port 9000     # or anywhere else

A user service is installed as `nnstudio.service` (port 8124):

    systemctl --user status nnstudio
    journalctl --user -u nnstudio -f

The app has its own venv (`.venv` — `gguf` + `numpy`) and a `models/` link to
the next-token GGUFs. The server is stdlib + the builder: `/` (the UI),
`/api/models` (the local GGUF list the GGUF node's form shows),
`/api/gguf/canvas` (header-only model → canvas), `/api/gpu` (GPU numbers,
straight from `nvidia-smi`), `/api/backend` (who is connected) and
`/api/run` (dispatch the design to the agent). It listens for agents on
`--control-port` (8125 by default).

## Accounts — guests, and what an account is for

Everyone starts a **guest**: drawing, wiring, running and training all work
with no account, and nothing of the session is kept on the server. An account
is only what the server saves a canvas *for* — it carries a username and an
optional email, sealed like everything else. Creating one always goes through
Cloudflare's bot check (Turnstile): the browser solves the challenge, generates
its own token — 32 random bytes, never a password, shown once — and the
register endpoint verifies the challenge with Cloudflare before the row
exists. The account lives in the header's top-right block (icon and name)
throughout.

**The server never sees the token, and cannot read what it saves.** Three
values are derived from the token, in the browser, each under its own label:

| derived | goes | the server keeps |
|---|---|---|
| `auth` = sha256("nnstudio-auth-v1" + token) | every request (cookie / bearer) | its sha256 — a hash of a hash |
| `name tag` = sha256("nnstudio-name-v1" + token + lower(name)) | with each save | the tag itself — keyed, so it cannot be matched to a guessed name |
| data key = sha256("nnstudio-data-v1" + token) → AES-GCM | nothing | — |

Designs are saved from the **project tabs** above the toolbar: `Ctrl/Cmd+S`
saves the active one (a new project asks for its name first), the `+` at the
strip's right opens **New Canvas** or your **Saved Canvas** list — each saved
canvas opens as its own tab.

Names and designs are sealed with AES-GCM *before* they are sent; `users.db`
holds the tag, the sealed name, the sealed design, and the auth hash — no
plaintext name, design, or credential. What stays visible is only metadata:
that an account exists, how many designs it has, their sizes and timestamps.
Losing the token loses the designs for good — there is no reset, server
included. The browser remembers the token (`localStorage`) so a reload stays
unlocked; *log out* forgets it, and a browser without it shows the account
locked ("account 🔒") until the token is pasted again.

Endpoints, sealed end to end: `/api/account/register` (`name_tag`, `name_ct`,
`auth`, `challenge`, `email_ct` when given), `/api/account/login` (`auth`), `/api/account/save`
(`name_tag`, `name_ct`, `blob`), `/api/account/canvases`,
`/api/account/canvas?tag=…`, `/api/account/delete` (`name_tag`). Sealing needs
WebCrypto, which browsers only expose on secure origins — `http://localhost`
counts, https counts, a plain-http LAN address does not (the dialog says so).

A `users.db` from before the sealing is moved aside on first use
(`users.db.pre-e2e`) and a fresh store takes its place; old accounts simply
sign up again.

### From another device — https with a local CA

Sealing needs a secure origin, so a plain-http address such as
`http://10.8.0.9:8124` cannot sign up or save. The server serves https on the
same port (the first byte of a connection says which), with a certificate from
a tiny local authority:

    bash tools/tls_local.sh          # makes .certs/ (git-ignored; keys 0600)
    systemctl --user restart nnstudio.service

It answers to `localhost`, the host's name, and every address the machine
holds. Copy `.certs/ca.crt` to the device once and trust it — over the VPN,
from the device: `scp <user>@10.8.0.9:/path/to/nnstudio/.certs/ca.crt .` —
then per platform:

| platform | trust the CA |
|---|---|
| Android | Settings → Security → Encryption & credentials → Install a certificate → **CA certificate** |
| iOS / iPadOS | install the profile, then Settings → General → About → **Certificate Trust Settings** → enable full trust |
| Linux | `sudo cp ca.crt /usr/local/share/ca-certificates/nnstudio.crt && sudo update-ca-certificates` (Firefox has its own store) |
| macOS | double-click → add to the System keychain → **Always Trust** |
| Windows | double-click → Install → Local Machine → **Trusted Root Certification Authorities** |

Then `https://10.8.0.9:8124` is a fully valid origin: WebCrypto, the bot check
and saving behave; `http://localhost:8124` on the machine itself is unchanged.
If the machine's address changes, `bash tools/tls_local.sh --force` re-makes
the server certificate — the CA, and so every device's trust, is kept
(`--new-ca` starts a fresh authority, requiring a re-trust).

Cloudflare's *test* keys are used until the environment names a real widget,
and the dialog says so. To put a real one in place: create a widget at
[dash.cloudflare.com](https://dash.cloudflare.com/) → **Turnstile** → *Add
widget* (give it the hostname you reach the app on — `localhost` is fine),
copy the **Site Key** and the **Secret Key**, and put them in the repo's
`.env` — the server reads it at startup, and a real environment variable wins
over it. Restart (`systemctl --user restart nnstudio.service`), and the dialog
stops saying "test keys". The site key is public by design (the page hands it
to every browser); the secret one stays on the server.

    NNSTUDIO_TURNSTILE_SITE_KEY=…      # the widget's site key
    NNSTUDIO_TURNSTILE_SECRET_KEY=…    # its secret key
    NNSTUDIO_COOKIE_SECURE=1           # adds Secure to the session cookie (https)

`NNSTUDIO_USERS_DB` moves the accounts/designs database; `users.db` next to the
code (git-ignored) is the default.

## Tests

    ./tests/run.sh           # offline: JS syntax, the DOM harness, the builder, the engine, HTML
    ./tests/run.sh --live    # + the running app and a real agent run
    # the diffusers gate needs the venv's diffusers (the reference): it builds a
    # tiny random UNet2DConditionModel, saves it, and holds the canvas to it

## Files

| file | role |
|---|---|
| `ui/index.html` | the whole app — the canvas, the project tabs above it (Ctrl/Cmd+S saves; the + opens New Canvas / Saved Canvas), the toolbar (Palette · Import · Backend — the server config's door, up top of that menu — · VRAM), the right-click menus; one self-contained page |
| `nn_engine.py` | the GGUF/HF → canvas builder (`build_canvas` is pure; `gguf_canvas`/`hf_canvas`/`diffusers_unet_canvas` read real headers — the diffusers one mirrors UNet2DConditionModel module for module, SDXL's text_time conditioning included) — extracted from next-token's engine |
| `nn_server.py` | the stdlib server: the UI, the GGUF list, the builder endpoint, the GPU badge, the agents' control listener and the run dispatch |
| `nn_users.py` | accounts for saving designs: sealed names/designs, keyed tags, the auth hash, the Turnstile check (`users.db`, git-ignored — ciphertext only) |
| `agent/nnstudio_agent.py` | **the user's backend** — run it on the machine that holds the GPU; it dials the app, reports the machine and executes designs |
| `agent/nn_torch.py` | the canvas→torch engine the agent runs (fed by the app, never by this page) |
| `agent/nn_tokenizers.py` | huggingface `tokenizer.json` fetching + metadata (stdlib until a tokenizer is loaded) |
| `tests/ui_boot.js` | the DOM-stub harness (boxes, wires, sizes, estimate, undo, copy/paste, GGUF node, flags, the menus) |
| `tests/gguf_canvas_test.py` | the pure builder test, on synthetic tensors |
| `tests/nn_torch_test.py` | the engine's offline tests (forward, ops, training, GGUF weights) |
| `tests/diffusers_unet_test.py` | the diffusers gate: tiny random UNet2DConditionModels — plain, SDXL's `text_time`, SDXL's whole block shape — through diffusers' eager path and through `nn_engine.diffusers_unet_canvas`: stage for stage, every tensor bound |
| `tests/nn_users_test.py` | the account store, offline: sealed rows (not one plaintext name), the bot check's answers, the .env loader, design CRUD |
| `tools/quant_bench.py` | what a design's weights cost per sweep: the native kernels timed on real blocks of a GGUF, summed per type |
| `tools/tls_local.sh` | the tiny local CA + server certificate (`.certs/`, git-ignored) for reaching the app over https from your own devices |
| `tests/agent_test.py` | the live end-to-end: a real agent connects and a design runs (`--live`) |
| `weights/` | where `export weights …` writes, and a place to drop pt / safetensors / gguf files for the pickers |
| `models/` | → `~/jevtest/models` (the GGUF files live there) |

## Backend — where the design runs

The heavy work never runs in this page. The toolbar's **Backend** menu holds
the two places a design can run; each section drops down into its own
instructions and settings, and a `use …` chip marks the one in use (remembered
in the browser):

- **Local** — a computer of your own. Start the agent there and it **dials
  this app and stays connected** (one stateful TCP channel, newline-JSON both
  ways; the app heartbeats it). The app listens on port **8125**
  (`--control-port` on `nn_server.py`):

      python nnstudio_agent.py --app <this-app>:8125

  On this machine it runs as the `nnstudio-agent` user service
  (`agent/nnstudio-agent.service`, `Restart=always`) so the backend stays up
  across crashes and logins; see `agent/README.md` to install it elsewhere.

  The agent needs `python3`, `torch` (the CUDA build to use the GPU) and
  `numpy`; `gguf` is optional (only designs bound to real GGUF files need it,
  and the file must sit at the same path on that machine). As soon as it
  connects, the section says **connected** and the **server config's door** —
  the menu's first row — turns live. The box itself is where the machine
  shows up: its specs, its live RAM/VRAM, and the load controls.

  Frozen GGUF layers can run **natively**: their quantized blocks stay packed
  and a Triton kernel (bundled with torch, no new dependency) fuses the
  dequant into the matmul — `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q2_K`, `Q3_K`,
  `IQ4_XS`, `IQ4_NL`, `IQ3_S`, `IQ3_XXS`, `IQ2_S` and `IQ2_XS` today; `F32`
  and anything else dequantizes at load as before. The K-family kernels run
  near half the card's bandwidth; the IQ-family ones are several times
  slower — `tools/quant_bench.py <file>` times a file block by block and
  prices one forward pass, which is usually where a slow decode lives.
  [docs/backend.md](docs/backend.md)
  has the per-platform matrix, and `tools/backend_check.py` reports what your
  machine can do.
- **Remote — Modal (BYOK)** — paste a Modal token (it stays in this browser,
  never in the design) and jobs will run on your own Modal account; the
  section points at `modal.com → Settings → Tokens`. The transport is not
  wired yet: the server config's door enables with a saved key, and says so.

The section in use wears an orange dot, and the choice survives a reload.
A caveat: the control port trusts whoever connects — keep it on a network you
control.

## Run — the whole design, on your backend

**the server config** opens from the **Backend** menu — its door is that
menu's first row — and is the machine's window; with no backend connected it
says so. Open it and it is the one place that shows:

- **the machine** — the agent's specs (CPU and cores, RAM, GPU and VRAM,
  torch/CUDA, OS) and **what it is using right now**: `ram now 14.2 / 30 GB ·
  47%` and `vram now 6.1 / 15.5 GB · 39%`, pushed by the agent every few
  seconds so the numbers stay live;
- **the weights** — a **cpu offload** bar sets the ratio (the first layers
  stay on the gpu, the rest on the system's ram), then **load weights** pins
  every weight box in the design (nested boxes included) and **pre-loads the
  whole design on the agent** — compiled there, every weight materialised where
  it was told, and kept warm for the next run. The bytes come from the JSON first: a design that **cannot fit**
  is refused *instantly* (`its weights need 50.8 GB on the gpu, but only
  15.2 GB is free — put layers on the system, or use load all on system`)
  instead of dequantising for minutes and then failing; while a load *is*
  running the button shows how far it is (`loading… 55% · 3.0 / 5.4 GB`), and
  **re-loading the same design is instant** — the warm graph is handed back
  (`already warm`). Next to the buttons, **offload** lets the backend let go
  of every model and cache it is holding (the warm graph, GGUF readers,
  tokenizers, CUDA pools — the pages go back to the OS), so a machine that
  filled up is clean again; the next run simply rebuilds. Under them sits the
  one number that matters: the design's **weight bytes and where the layers
  sit** — `13.3 GB · all on the gpu`, `… · all on the cpu`, or `… · mixed
  (13.0 GB gpu · 300 MB cpu)` — so the GPU / CPU / split state a run would use
  is a glance
  away; it follows every right-click placement change. After a load, a second
  line says what the backend **actually holds** — `loaded: 25.4 GB · mixed (…)
  · 4012 ms` — which is what *offload* clears. A run right after says *using
  the loaded weights*; change the design and the next run transparently
  recompiles;
- **the run's own numbers** — the run action lives on the `input` box
  (**▶ run the design**) and its window keeps the clocks: **run → result** (the
  browser's wall clock) and **input → output** (the engine's own sweeps), plus
  every box's value after the run — shape and mean, by name.

A single layer's placement is its own business: right-click any weight box →
**load this layer on** → `the gpu — video memory` or `the system — this
machine's ram`. The card wears a small `gpu` / `sys` tag, the choice travels
in the design (`place` in the JSON), and the engine moves each box to its
device before its forward — so a network can split across VRAM and system RAM
mid-forward. Unset boxes take the backend's own device.

A missing or bad input is answered with the agent's own reason, not a crash;
the app itself never imports torch. A run that fails — a missing input, an op
the engine refuses, an out-of-memory — **hands its memory straight back**: the
compiled graph, the frames of its traceback and the CUDA cache all let go, so
a machine that filled up on attempt one is clean for attempt two (the
**offload** button does the same, on demand). One honest limit: the backend
holds every weight box as a real tensor — `f32`, `f16` or `bf16` as the box
declares, and anything quantised (`q4_k`, `iq4_xs`, …) once dequantised at
`f16` — while the maths runs in f32, a slice of the weight at a time (a
half-precision weight is never doubled to compute). The design's estimate
line still counts a quantised box as its file size, so the `loaded:` line
under it is the truth for what a run costs: the Qwen-27B rides at ~51 GB, the
12B at ~22 GB (their files are 13 and 7 GB). A model that still does not fit
its RAM/VRAM fails with *not enough memory to load this design*; the offload
button, a smaller model, or fewer layers per device is the fix.

The agent runs the canvas's boxes as the engine knows them — forward only. A
Script box is carried in the design but not executed by this engine yet. A box
bound to a file (`file` + `t`) loads those tensors instead of the random init
— GGUF, safetensors or pt — so keep the file at the same path on the agent's
machine (GGUF-bound boxes need `gguf` installed there; safetensors is read
built-in, and pt is read with `weights_only`).

## The canvas — thirteen kinds of boxes

nnstudio is **one canvas** and nothing else: a **fixed-size canvas**, full
width, with its tools in a **toolbar strip** above it — **Files** (`export` ·
`import`, the design as JSON; `export weights …`, the whole design into one
pt / safetensors, the same the right-click menu opens for JSON),
**Palette** (categories that expand into the box cards, click or drag),
**Backend** (where the heavy work will run — see below),
**Run** (dispatch the design to it — disabled until a backend is established;
see below) and
**VRAM** (the estimate panel). A blank canvas opens with the **skeleton** —
one `input` wired straight into one `output` — so the network is never a void.
On the canvas
there are **thirteen kinds of boxes** —
everything that is left after an atom-by-atom audit of the canvas — all **the
same size and drawn by the same card renderer** as every other box in the app
(rounded body, type top-left, shape chip top-right, rails, port dots):

- **atomic (static) boxes** — they cannot be divided into anything simpler:
  `input`, `output` (the graph's boundaries), **`Heatmap`** (a vector drawn as
  one picture: centred on its median, spread by its robust deviation onto the
  most square grid its length allows, through a colour ramp — it sits in the
  Palette's **analysis** category and its picture reads as an image
  everywhere), `Activation`, `Operation` (one
  tensor op each), `Norm` (a fused reduction: with no mean/root atoms to
  build it from, it stays whole), **`Script`** — a Python function you
  write (`def run(x): …`) that the backend runs **on the CPU** to prepare an
  input tensor or post-process an output; it holds no weights, and the wire's
  size looks straight through it — and the **text pair**: **`Tokenizer`**
  (text → token ids, mapped by the character vocabulary the box carries) and
  **`De-tokenizer`** (ids → text; its output dot **is** the text) and
  **`Sample`** (logits → one picked id: the last position's distribution is
  sampled with a temperature and seed) — and the **`HF model`** box: paste a
  huggingface repo and it is a whole small language model on the canvas —
  **text in, the next token's text out**;
- **weight-bearing boxes** — they carry a learnable table, so they stay whole:
  `Fully Connected` (a real matmul — it is what a GGUF `*.weight` binds to) and
  `Embedding` (a **table read by index**, not a matmul: one id (`1`) goes in,
  the row it names — the vector — comes out of the box's `vocab × out` table;
  `token_embd.weight` reshaped to `[vocab, d_model]` is exactly this, and the
  LM head, which really is a matmul over that table, stays a fully connected
  box);
- **the container** — `Nested`: it *is* how a non-atomic box is written down
  (up to a whole model: the Palette menu's **GGUF node** is one, with the
  model loaded into it);
- **three block templates** — `Attention`, `State Space` and `MoE` are no
  longer kinds: adding one drops an ordinary **nested box** whose inside is a
  network of atomic boxes that performs the operation (attention: q/k/v/out
  projections, rope, `q·kᵀ`, softmax, `scores·v`; state space: conv, silu,
  scan, out projection; MoE: router, softmax, top-k, expert up/down, weights).
  The operations they are built from are in the operation box's list:
  `transpose` · `dot product` · `matmul` · `concat` · `add` · `subtract` ·
  `multiply` · `divide` · `rope` · `scale` · `topk` · `conv1d` · `scan` ·
  `reshape` (a `scale` is an element-wise multiply by a bound scalar — Gemma's
  per-layer output scale is one; a `concat` joins its inputs' widths).

Every box reads the same three lines: **kind**, **name**, and its own value —
the size (a fully-connected box shows its whole `in ? → out ?`, an embedding
box its table `vocab × out`), or the
settings of a block box (norm · attention · state space · MoE), or the
**activation function** for an activation box, the **operation** for an
operation box, the **function** for a script box (`run(x)`), or the
**vocabulary** for the text pair (`vocab 27`).

| box | kind line | dots |
|---|---|---|
| **input** | `input` | one **output** dot, right edge |
| **Tokenizer** | `Tokenizer` | one **input** dot left — text, from an input box set to `text` (or its own default) — and one **output** dot right: one id per token; right-click → `set the tokenizer …` holds a huggingface repo and a character vocabulary |
| **Embedding** | `Embedding` | input dot left (**one id**, size `1`), output dot right (the vector) |
| **Fully Connected** | `Fully Connected` | input dot left, output dot right |
| **Activation** | `Activation` | input dot left, output dot right |
| **Operation** | `Operation` | as the operation needs: one dot each side (transpose, rope, topk, conv1d, scan, reshape, permute, repeat, mask, scale) or **two input dots** (dot product, matmul, arithmetic) |
| **Norm** | `Norm` | input dot left, output dot right — the size passes through |
| **Script** | `Script` | input dot left, output dot right — the size passes through; right-click → `edit the script …` holds the Python (run on the backend CPU) |
| **Nested** | `Nested` | **as many as you build** (see below) — this is what an attention / state space / MoE template makes |
| **De-tokenizer** | `De-tokenizer` | one **input** dot left — **token ids only** (logits are refused: put a `Sample` in between) — and one **output** dot right: the text; the card shows it after a run |
| **Sample** | `Sample` | one **input** dot left — logits of any size — and one **output** dot right: one picked id (size 1); right-click → `set the sample …` holds the temperature and seed |
| **HF model** | `HF model` | one **input** dot left — text (or token ids) from a wire, or its own prompt — and one **output** dot right: the **next token's text**; right-click → `set the model …` holds the huggingface repo, the prompt and a temperature (0 = greedy); the card shows the text after a run |
| **output** | `output` | one **input** dot, left edge |

Names default to `input 1`, `embedding 1`, `fully connected 1`, `output 1`, …
and the size line shows `?` (dimmed) until you configure it.

A `?` is *unconfigured*, not a guess: fresh boxes start that way, the form's
size fields start empty (placeholder `?`), and it is drawn in a dimmer colour
than a real size. Clearing a size in the configure form makes it `?` again.

- **add** — two ways, same boxes: **the Palette menu in the top bar** — the
  graph's edges lead as **two cards of their own** (`input`, `output`), and
  below them only the categories are listed (`weights` · `text` ·
  `tensor ops` · `blocks` · `containers`); clicking one expands it into its **visual boxes**
  (the same card renderer the canvas draws). **Click** a card to drop its box
  in the middle of the view, or **drag it out** of the menu and release over
  the canvas to land it exactly there — either way it arrives selected. The
  `containers` category leads with the GGUF node at the top level only (inside
  a nested box there is no whole-model card), and the GGUF node asks which
  model to load as soon as it lands; or
  **right-click empty canvas** → a menu of **categories that drop down** —
  `boundaries` (`input`, `output`) · `weights` (`embedding`, `fully
  connected`, `norm`) · `text` (`tokenizer`, `de-tokenizer`, `hf model`) ·
  `tensor ops` (`activation`, `operation`, `script`, `sample`) · `blocks` (the
  three templates) · `containers` (`GGUF`, `nested` — the GGUF card at the top
  level only) · `the design` (paste / JSON); one category opens at a time, and
  a fresh menu always starts collapsed → the box it asks for lands where you
  right-clicked. A **box's own menu is categorized the same way**: `this box`
  (rename, copy, and whatever sets that kind) · `weights` (the dtype, where it
  loads, external files) · `inside` (a nested box's own network) · `flow`
  (deactivate, freeze), with `remove this box` one click away. It asks for
  the size (a fully-connected box asks for its name,
  in and out too; an activation/operation box asks which function/operation;
  leave a size empty and it stays `?`) and drops the box where
  you clicked. Either way the new box arrives
  **selected**, ready to move, configure or delete;
- **sizes check themselves, silently, for every kind of box** (input: its
  `out`; output: its `in`; fully connected: both; embedding: one id in, its
  table's vector out). The check runs on **every new
  connection** and on **every size change**, and the wire's colour is the whole
  story:
  - **grey** — both ends are still `?` (nothing is known yet);
  - **plain** — both sizes known and equal;
  - **red & dashed** — known but different: the connection is refused (while you
    drag, the live wire turns red over a target it cannot join), or an existing
    wire became wrong when you resized an end;

  a `?` end **takes the other end's size** — and remembers which connection
  filled it, so when that size moves again every size it filled **follows
  along** (across pass-throughs and through nested ports), while a size you
  typed keeps its ground; and the check **sweeps the whole
  network** rather than just the wire in hand: it runs on every connection,
  every size change, and every box added or removed, and follows every wire it
  can — any number of hops, forwards or backwards, straight through activation
  boxes and **into (and out of) nested boxes** — until nothing new is known.
  So configuring one end of a long chain fills every box along it, wherever
  they sit. A box that is not a pass-through (a fully-connected box has two
  independent sizes, an embedding box one id in and its vector out) stops the
  sweep at whichever side the wire meets. A typed
  size is never overwritten, so a real mismatch stays visible as a red wire —
  only the sizes the check itself filled keep following. Text is
  never shown anywhere; rewiring an input dot replaces its wire, so a mistake
  is one drag from fixed;
- **every box is checked for validity, and error boxes say so** — a box has to
  take part in the flow: **each of its input dots wired**, **each output dot
  sending somewhere**, and nothing touching a mismatched wire (the size rules
  above). A box that fails is drawn as an **error box**: red dashed border and
  a red `!` in its corner — hover it for the one-line reason (*an input dot is
  not wired* · *its output goes nowhere* · *this box is isolated* · *size
  mismatch on this box's wire* · *something inside this box is in error*).
  **Errors propagate outwards**: if anything nested inside a box is in error —
  however deep — that box is in error too, and so is every box above it.
  Unmapped GGUF tensors (the builder had no role for them) are isolated by
  design, so they — and the blocks that hold them — show up as error boxes;
  that is how you spot them;
- **dtype and a VRAM estimate** — every weight-carrying box carries a **dtype**
  (right-click → `set the dtype`: `f32` · `f16` · `bf16` and the k-quants, each
  with its real bytes per weight; `layer` and `embed` boxes start on f16, norms
  on f32). The **VRAM** menu's estimate panel then sums the whole graph: FCs
  by `in × out`, embedding tables by `vocab × out`, norms by the size flowing
  through them, scales as one weight, and every
  box that came from a GGUF by its **exact** element count and byte size — so a
  loaded model's figure equals the file (27.32 B params · 13.3 GiB for the Qwen
  GGUF, verified against `gguf`'s own totals). `settings` opens the knobs:
  context · batch · layers · kv-heads · head-dim for the inference **KV cache**
  line — and only the blocks that really hold attention cache K/V (a hybrid
  model's state-space layers don't), at the context you run at: it starts at a
  sane 8k, the knob's label shows the model's own maximum, and the panel says
  what that ceiling would cost; totals read **to run** and **to train**, and
  every dtype in the graph gets its own row (its params · its bytes). The
  training bill is itemised the way it is paid — fp32 gradients, fp32
  optimizer moments, and an fp32 master copy only for weights that are not
  already f32 — with optimizer (`none`/`sgd`/`adamw`) + fp32 master weights
  as the knobs; an **estimate all as** switch re-prices the whole graph at
  one dtype (all-f16 = 50.9 GiB for that same model). It stays honest about its
  limits: activations and workspace can't be derived from the structure, and
  quantized weights don't train as-is — the panel says so;
- **rename** — right-click a box → `rename this box`: one name field, for
  every kind of box;
- **go inside a nested box** — **double-click it** (or right-click →
  `go inside`): the canvas becomes that box's own network. Put **input** and
  **output** boxes inside and they *are* the nested box's dots — two input
  boxes inside means **two input dots** outside, in top-to-bottom order, and
  the wire size checks read straight through them (a wire into an unknown
  custom dot fills the input box *inside* the box). Everything you build in
  there is that box's saved definition — it travels with the box, wires
  included, and is exactly as you left it when you go back in; `← back` in the
  top-left corner climbs out, and a nested box's menu is
  `rename · go inside · export this nested box as JSON · import into this
  nested box … · remove`;
- **the design is JSON** — right-click empty canvas → `export the canvas as
  JSON` (the whole canvas, nested boxes and their insides included) or
  `import JSON …` to load one back; a nested box's menu offers the same for
  just that box. Two shapes, one rule:
  ```json
  { "type": "canvas", "version": 1, "nodes": [...], "edges": [...] }
  { "type": "nested", "version": 1, "name": "…", "nodes": [...], "edges": [...] }
  ```
  a box is `{ id, kind, name, x, y }` plus its size(s) — `dout` for an input,
  `din` for an output, `din`/`dout` for a fully-connected box, `vocab` too for
  an embedding box, `act` for an
  activation, `op` (plus `dims` for a shaped reshape) for an operation,
  `norm`/`eps` for a norm, `script` (the Python source) for a script box,
  `inner` for a nested box — every box may also carry
  `t`, the GGUF tensor name(s) it is bound to — and a wire is
  `{ from, to, fromPort, toPort, via? }` (ids name boxes inside the same
  design). **The canvas is saved and loaded at the top level only**: inside a
  nested box those items are gone (there is no canvas there) and only
  `import a nested box …` remains. A **nested design loads anywhere** — at the
  top level, inside a nested box, or into one specific nested box — and always
  arrives as a nested box holding that network. A canvas pasted into a nested
  box is refused, and a canvas is never hidden inside a nested design at any
  depth. **Loading a canvas over a used canvas asks first**: the form warns
  that it replaces everything already on the canvas and the button becomes
  `overwrite` (`cancel` keeps the work). A refused import reddens the field and
  stays open (nothing else is said); a loaded design lands fitted, sizes
  checked, ready to edit;
- **weights from a file** — any weight-bearing box can take its weights from a
  real file instead of the random init: right-click → `load external weights …`,
  pick a **GGUF, safetensors or pt** (the list comes from `models/` and
  `weights/`, and any path can be typed), and the app reads the file's
  **metadata** — every tensor with its shape and dtype — with the tensor that
  matches the block's name **flagged** (`blk.0.attn_q.weight` for a box named
  `q proj`; the `.weight` of a weight/bias pair wins). Binding stores
  `{file, t}` in the design, **takes the tensor's shape as the box's sizes** (a
  `[out, in]` weight sizes a fully-connected box, `[vocab, out]` an embedding,
  a vector a norm) and follows its dtype where one maps (f16/f32/bf16). The
  card's tooltip names the file and tensor, `unbind` clears them, and the add
  form for a weight box has a `load weights …` chip, so a new block can be
  pointed at a file in the same breath. At run time the backend loads those
  tensors instead of initializing randomly (a `.pt`/`.pth`/`.bin` is read with
  `weights_only` — no pickle code; the safetensors reader is built in; a GGUF's
  quantized blocks still dequantize on the way in);
- **export weights** — the Files menu's `export weights …` writes the whole
  design's weights (bound tensors included, unbound ones as they initialize)
  into one **safetensors or pt** file under `weights/`, and **points every
  weight box at its key** in that file — one undoable step. The design now runs
  on exactly those weights, and the file is what another machine needs to
  reproduce the run;
- **the GGUF node** — the Palette menu leads with a **GGUF node** (the empty
  canvas's right-click menu has the same `GGUF box` entry; inside a nested
  box it is gone — a whole model is a top-level thing). Drop it and the
  loader opens right there: pick one of the local GGUFs and press `load`, **or
  paste a huggingface repo** (e.g. `Qwen/Qwen2.5-0.5B`) and press `build
  canvas` — the app reads `config.json` and the safetensors **headers** (no
  weights move) and breaks that repo's transformer into the same native
  blocks: q/k/v/o, gate/up/down and the norms, the embedding and the head,
  every box bound to the repo's own tensor name **and the shard it lives in**
  (`file: hf:<repo>/<file>`, biases riding with their weight), so the backend
  downloads a shard the first time a box bound to it runs (cached under
  `~/.cache/nnstudio/hf`). How the boxes are wired comes from the family's
  **verified spec** (`specs/`): read the reverse-engineering section below.
  The Run menu says which you got: `designed from the llama spec · exact
  (conformance-verified)` or `approximate skeleton — no verified <arch> spec
  yet`.
  **One click** reads the file's header (only the header — no weights, no
  VRAM) and **loads the model into that box** — the box takes the model's
  name (`Qwen3.8-27B`, `Gemma-4-12B-It`, from the header's `general.name`),
  wears the model's own dots (its inside `tokens`/`logits` boundary boxes:
  one id in, the vocab out, `mtp logits` too when the model has a next-token
  block) and is made of the app's own boxes (`input`, `Embedding`, `Fully
  Connected`, `Activation`, `Operation`, `Norm`, `Nested`, `output`, nothing
  bespoke). A box that stands on its own — or is wired only to its own
  `tokens`/`logits` boundaries — gets the model's boundaries with it:
  `tokens` → the box → `logits` (`mtp logits` too), wired, sized and whole,
  and loading another model into the same box **re-crops** them (a `mtp logits`
  it no longer has goes, a new one arrives). Dropped into a bigger design, only
  the box fills and its dots are yours to wire. Cancel and it stays an ordinary
  empty nested box: right-click → `Load GGUF …` reopens the loader any time,
  `go inside` lets you build by hand. Loading into a box that already holds a
  model asks first (`overwrite`), and the whole load (crop included) is one
  undo step. The box **remembers the model's file** (`file`), so the backend
  binds its tensors wherever they sit inside it — any nesting depth. Loading
  into the blank canvas replaces the skeleton and wires the model's own
  boundaries; a design from before the file was remembered (or one whose file
  was cleared) **binds back by name** on the next Run — exactly one matching
  model file wins, anything ambiguous asks you to load it again. **Double-click the box to go inside** (the header says which box
  you are in; `← back` climbs out): there is the model's own network, one
  **nested box per block** (`blk 3 · attn mlp`). Its inside is **folded** —
  the chain is laid out in rows that each read left to right, with the wire to
  the next row sweeping back through the gap below it — so a 48-block gemma
  is eight rows and a 64-block Qwen ten (the next-token head gets its own row,
  its embedding wire riding the free lane down the left). Each block holds its
  real parts:
  `attn_norm` → the attention / state-space / MoE **nested box** (its own
  network of atomic boxes, each bound to the tensor that feeds it: `q proj` =
  `blk.3.attn_q.weight`, `q norm`, `rope`, `q·kᵀ scores`, `softmax`,
  `scores·v`, `out proj`, …) → `post_attention_norm` → residual `add` →
  `ffn_norm` → SwiGLU (`ffn_gate` and `ffn_up` → `silu` → `multiply` →
  `ffn_down`) → residual `add`; the chain runs `tokens` (the id — shape `1`)
  → the `token embedding` **Embedding** box (its `[vocab, d_model]` table read
  by that id) → the blocks → `output_norm` → `lm head` (a real matmul, so an
  ordinary fully connected box) → `logits`, and the model box's own dots are
  exactly those boundary boxes — so the size sweep and **the VRAM estimate read
  straight through it** (the figure still equals the file exactly). Every box is
  **bound to the
  GGUF tensor(s) it came from** (hover it, or `bind a tensor …` to read or edit
  them), and tensors the builder has no role for are never dropped — they land
  as bound boxes inside their block. A **next-token (MTP) block** — Qwen's
  `blk 64 · mtp` — is drawn as the branch it is, not as loose boxes: `h` (the
  last main block) and `e` (the token embedding) → `hnorm`/`enorm` → `concat` →
  `eh proj` → the block's own transformer → `shared head norm` → `mtp logits`,
  sharing the main LM head instead of duplicating it (so the VRAM estimate
  still counts the head once). A design exported before the audit (with old
  `attn`/`ssm`/`moe` boxes in it) still loads: each becomes the nested box it
  should have been, its tensors re-bound to the atoms.;
- **undo / redo** — `Ctrl+Z` (⌘Z) steps back through your actions — a box
  added, moved, renamed, resized or removed; a wire drawn, bent or cut; a box
  moved into a nested box; a canvas or GGUF loaded — **one step per action**,
  and it lands you back wherever you were when it happened (inside a nested box
  included) — with the **camera exactly where you left it**: undo and redo never
  pan or zoom, so the view you were studying stays put. `Ctrl+Shift+Z` or
  `Ctrl+Y` steps forward again, a new action drops the redo queue, and the last
  60 steps are kept. Panning, zooming and going
  in/out of boxes are not actions, and typing in a form field never triggers it
  (the field's own undo stays yours);
- **set the embedding** — right-click an **embedding box** → `set the
  embedding`: the table's `vocab` and the vector's `out`. The card then reads
  `248320 × 5120`;
- **set the activation** — right-click an **activation box** → `set the
  activation`: a list (`none` · `relu` · `leaky relu` · `gelu` · `silu` ·
  `tanh` · `sigmoid` · `softmax`) shown on the box's bottom line. It carries no
  size of its own — it passes the wire's size straight through, so the size
  check looks through it (a chain end that gets broken downstream still turns
  red);
- **static operations** — right-click an **operation box** → `set the
  operation`: `transpose` · `dot product` · `add` · `subtract` · `multiply` ·
  `divide` · `reshape`. An operation holds **no weights**: it only rearranges
  or combines what the wires bring. `transpose` passes the size straight
  through (one dot each side); `reshape` can be given the **output dim** it
  makes (say `128x40` — any separator, or just one number), and then the
  element count is the product, **both** sides must carry it, and the card
  shows `5120 → 128×40`. Leave the field empty and reshape is a plain
  pass-through like transpose. The arithmetic ones are
  **element-wise** — two in-dots of one size, the out-dot that same size;
  `dot product` takes those same two dots but sends a **single number**
  (size 1), so its output cannot feed anything sized. Both input dots of a
  two-wire operation share one size, and the wire check keeps them equal.
  Switching to a smaller operation drops the wire that lost its dot;
- **the text boxes** — a **Tokenizer** takes **text in** (one input dot, left)
  and sends **token ids out** (one id per token). The text comes over the
  wire from an **input box set to `text`** — that is where you type it — or,
  if nothing is wired in, from the tokenizer's own default text (the Run menu
  asks for it then; a wired tokenizer is fed upstream and asks for nothing).
  A **De-tokenizer** takes **token ids only** and puts the **text on its
  output** — the card shows it after a run, and the Run result quotes it too;
  a model's logits go through a **`Sample`** box first (logits in, the last
  position's distribution sampled — temperature · seed, `0` is greedy — one
  id out). Either box can be
  pointed at a **huggingface repo**: right-click → `set the tokenizer …`,
  paste `Qwen/Qwen3-8B` (or the whole URL, or `gpt2`) and press `load the
  tokenizer` — the app fetches the repo's `tokenizer.json` (cached under
  `~/.cache/nnstudio/tokenizers`, `HF_TOKEN` honours gated repos) and shows
  `BPE · 151936 tokens`; the backend loads the same file with the
  `tokenizers` package, so a run tokenizes *and* de-tokenizes exactly like
  the model does. Without a repo, the box falls back to a **character
  vocabulary** (the index of a char is the id, chars outside it are skipped —
  no dependency at all). Both boxes' in-dots accept any size — the wire never
  goes red, because only the run knows what is coming;
- **the input's value** — an **input box** says how its value is made:
  right-click → `set input` (adding one asks in the same form).
  Four modes: `values` (the default — type them in the Run menu); **text** —
  a string the Run menu asks for, which a tokenizer reads off the wire
  (the card says `text`); **python**
  — one expression, or statements leaving the matrix in `x`
  (`x = [[1.0, 2.0], [3.0, 4.0]]`; `torch`, `np` and `math` are in reach, and
  it runs on the backend when the run needs it), or **random** — a seeded
  `randn` of `rows × the box's size`, the same matrix on every run of the
  design. The card's bottom line says `py` or `rand`, the Run menu shows the
  mode instead of a value field, and for a `values` input whatever you type
  there is still what the run gets;
- **the data state** — after a run, the boxes that carry data show what came
  through: an **input** its fed values (`1, 2, 3, 4`; a longer tensor shows
  its first values and trails off with `…`; text is quoted), an **output** its
  result, a **Sample** the id it picked, a **De-tokenizer** the text. The
  tooltip adds the shape and the mean, and the Run result lists every box.
  Any edit clears the states; the next run paints them again;
- **the data flow, live** — while the backend runs a design it reports every
  box the moment its value lands; the canvas lights that box and lets **light
  run along its wires to the next layer**, in data-flow order — inside
  attention / state-space / MoE blocks too, since a nested box reports its
  own steps. A fast run replays the same order once the result is back, so
  the flow is never skipped; any edit clears the lights;
- **when a run fails** — the engine blames a box and the canvas marks it: a
  **red pulsing border** and the reason written on the card (a block nested
  deep marks the box that holds it; the tooltip carries the message). The Run
  menu shows the whole error, and an edit or the next run clears the mark.
  (An error the engine cannot pin to a box — a raw torch shape error, say —
  falls back to matching the name, or just to the Run menu's message);
- **resize** — right-click a box → `resize this box`: just the size(s) — in/out
  for a fully-connected box, the single size for an input/output box
  (`?` placeholder = unconfigured). Both save in place: the box keeps its
  identity, position and wires;
- **remove** — right-click a box → `remove this box`;
- **move** — drag a box: it **snaps to the grid lines** — the 40-unit lattice,
  which the drawn grid includes at every zoom, and a fresh box is born on it
  too. The box itself is **4 × 2 cells** (160 × 80 world units), so its
  corners, middle and ports sit on crossings: that is what keeps columns and
  rows lined up like a breadboard. When the landed box shares an edge or a
  middle with another, that line shows as an amber **guide** for the length of
  the drag;
  **wire** — drag a dot (or a rail) onto another box's input dot: one output
  may feed several inputs, and an input dot keeps exactly one wire (a new wire
  replaces the old);
- **grab several boxes** — `shift`+drag on empty canvas draws a **marquee**; let
  go and every box it touched is selected (a fresh band replaces the selection).
  `shift`+click adds or removes one box from the group, a plain click collapses
  it to that box, and dragging any selected box moves the **whole group** at
  once — offsets intact, the grabbed box doing the alignment snapping.
  `Delete` removes the selected boxes together, in one step that a single
  `Ctrl+Z` takes back;
- **copy & paste** — `Ctrl/Cmd+C` copies the selected boxes **and the wires
  between them** (wires leaving the selection are left behind); `Ctrl/Cmd+V`
  pastes them wherever you are — the same level or **inside a nested box** —
  at the cursor (or the middle of the view; pasting the same spot again
  cascades a little), with **fresh ids**, selected and size-checked, as **one
  undo step**. Right-click a box → `copy this box` (or `copy these N boxes`
  when it is part of a multi-selection) and right-click empty canvas →
  `paste` (at the point you clicked) do the same. The clipboard carries real
  text, so a copy can be pasted back through the JSON form or into another tab,
  and pasting a nested design (a `"nested"` document) lands it as one nested
  box holding it — a canvas is still refused below the top level;
- **deactivate & freeze** — right-click a box → `deactivate this box` (or,
  with several selected, `deactivate these N boxes`): the box is **off** — no
  data flows into it, its wires dim and go dotted, its dots wear the
  deactivated mark, nothing about its dots or its inside is an error any more,
  and its **weights leave the model** (the estimate says how many). A live box
  that consumes a deactivated **output** is **warned**: amber `!` and border,
  its feed dot turns amber, hover says *fed by a deactivated box* — a
  warning, not an error, and it bubbles up through nested boxes like errors
  do. A nested box's deactivated **output dots** carry nothing either: that
  dot dims, its wires go dead and the blocks that used *that* dot are warned;
  when **every** output dot of a nested box is deactivated the block itself
  counts as inactive (it dims, and everything inside it with it). The menu
  says `activate` whenever the card reads inactive, and activating clears both
  the box's own switch and the dead output dots that were holding it off.
  `freeze this box` keeps the weights but takes them
  out of the training step: the panel lists the frozen params (bytes kept, and
  the training memory skipped) and prices only what is trainable, under a
  little bar of weights · KV · gradients+optimizer with a **total to
  run** and a **total to train** (`nothing trainable` when everything is
  frozen — the load-a-GGUF-then-freeze-it case). Both flags **read through
  every level**: deactivate or freeze a nested box and every block inside it
  is off/frozen too; an off card dims, a frozen one wears a small snowflake,
  and hover says which. Both flags save in the JSON (`off` · `frozen`) and
  undo like everything else;
- **move a box inside another** — drag it and drop it on a **nested box** (the
  nested box lights up while you hover): the box moves in, and **every wire is
  kept**. Wires that were already passing through the nested box's own dots are
  **absorbed**: the inside is wired directly and that dot is **reused** for the
  box's own wire, so moving a chain in one box at a time never grows duplicate
  in/out (a chain of any length still shows `1 in · 1 out`). Only when nothing
  inside corresponds is a port grown — and if a freed dot ends up carrying
  nothing at all it is removed again (the later dots shift down to match). A
  box that feeds several things shares its one output port (the fan stays
  outside and keeps working), a box can never be dropped into itself or into
  something that holds it, and the size check runs on the result like any other
  edit;
- **shape a wire** — drag any part of it (everything but the ×): the wire bends
  so its middle passes through where you drop it, **snapped to the grid** like
  the boxes; drop it near its natural middle and it snaps straight again.
  Right-click a bent wire for `reset the wire's shape`;
- **delete a wire** — hover it: the wire lights up and a small **×** appears in
  the **middle of that wire** (click it to remove that wire, and only that
  wire); right-clicking a wire offers the same as `remove this wire`;
- **canvas** — drag empty space to pan, the wheel to zoom (~5 % per mouse
  notch, scaled by how far you scroll so a trackpad stays gentle), double-click
  empty space to fit; a faint **grid** pans and zooms with the design (its step
  is 20 or 40 units — finer than a box, so boxes always land on it; every fifth
  line is a touch stronger). Boxes, wire bends and loaded GGUF layouts all sit
  on that grid, and the view is **anchored tightly**: the reachable region is
  the origin plus the design plus four cells, zooming out stops at 50 %, and
  the little world grows on its own as boxes are placed further out; no panels
  and no popovers besides the right-click menus —
  the toolbar strip above is the one chrome, and it dims while you are inside
  a nested box (there is no canvas there to export);
- nothing stays read-only: renaming, resizing, dtypes, tensor bindings and
  nesting are all edits — right-click empty canvas for `export the canvas as
  JSON` / `import JSON …`. The canvas lives in memory (a reload starts from the
  skeleton — one `input` wired straight into one `output`),
  with undo 60 steps deep (`Ctrl/Cmd+Z`, `Ctrl/Cmd+Shift+Z`, `Ctrl+Y`).

## Verified family specs — the reverse-model agent

A canvas built from a huggingface repo is only as good as the wiring a builder
guesses. For the families we care about, that wiring is no longer a guess: it
is a **spec**, and a spec is only accepted when it survives a numeric gate.

- **A spec is one JSON file** per architecture family (`specs/<family>.json`,
  format in `specs/README.md`): config slots → sizes, weights → repo tensor
  suffixes, and a block described as **native boxes only** — every op the
  engine really executes (`matmul`, `reshape`, `permute`, `rope`, `transpose`,
  `repeat`, `mask`, `scale`, `add`, …), nested for grouping, elidable optional
  tensors (`q_norm`, biases), and expressions like `heads != kv_heads`.
- **The gate is the library's own run.** `tests/spec_conformance_test.py`
  builds a tiny random model of the family with `transformers` (eager path),
  exports its state dict to safetensors, binds it to the canvas the spec
  builds, and compares **every stage** — embedding, q/k/v projections, each
  block's output, the logits. The llama spec matches to `0.00e+00`; the gate
  is 1e-4. (`hidden_states[-1]` is post-final-norm in transformers v5, so the
  gate compares layer module outputs.)
- **The AI agent drafts specs.** `/reverse-model <model_type|repo>` runs the
  workflow in `.opencode/commands/reverse-model.md`: it reads the family's
  source (`tools/reverse_model.py sources llama` prints the config, model,
  decoder layer, attention, MLP, eager attention, rope and mask helpers),
  inspects the repo's tensors (`tools/reverse_model.py inspect <repo>`), writes
  or updates the spec, and runs `tools/reverse_model.py check <family>` until
  every stage is green — the harness catches what reading alone can't (rope
  layout, QK-norm placement, GQA repeat order, mask dtype, v5's tied hidden
  state).
- **At run time there is no transformers.** The canvas executes on our engine;
  the spec only shaped it. Families without a verified spec build the older
  role-based skeleton and are labelled `approximate` — a design always says
  which it got.
- **`build canvas` runs the agent by itself.** Paste a repo whose family has
  no verified spec and the app keeps the approximate design usable while it
  starts the reverse-model job in the background (`tools/reverse_job.py`,
  `POST /api/reverse`) — its progress lives **on the box**, step by step:
  `agent (claude-sonnet-4.5 @openrouter.ai) · qwen2 · running the conformance
  gate · 5/9 stages · 166s`. The base model comes from the repo's `.env` — any
  OpenAI-compatible endpoint: `NNSTUDIO_REVERSE_BASE_URL`,
  `NNSTUDIO_REVERSE_API_KEY` and `NNSTUDIO_REVERSE_MODEL` (OpenRouter is one
  example). With all three the job runs its own private OpenCode server
  against that endpoint, and the key never touches a file; with only the model
  id it is an ordinary OpenCode model reference. When the gate passes, the
  exact canvas replaces the sketch in the same
  box; when it doesn't, the box says so and stays approximate. Jobs are shared
  per family: a second box joins instead of starting another. **Deleting the
  box halts the job** (`POST /api/reverse/cancel`) unless another box on the
  canvas still asks for that family. Under systemd a job runs as its own
  transient unit, so restarting the app doesn't interrupt it.
