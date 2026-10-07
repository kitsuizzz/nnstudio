# nnstudio

A dedicated app for training a neural network — **one canvas** with eight
kinds of boxes, built straight from a GGUF file (or by hand), wired live,
priced for VRAM, and saved as JSON. Extracted from next-token's Training mode
into an app of its own; the training UI, the GGUF→canvas builder and the
tests came along.

## Run

    ./run.sh                 # http://localhost:8124
    ./run.sh --port 9000     # or anywhere else

A user service is installed as `nnstudio.service` (port 8124):

    systemctl --user status nnstudio
    journalctl --user -u nnstudio -f

The app has its own venv (`.venv` — `gguf`, `numpy`, `torch` + CUDA and
`modal`) and a `models/` link to the next-token GGUFs. The server is stdlib +
the builder for the design side; torch is imported only when something runs.
Endpoints: `/` (the UI), `/api/models`, `/api/gguf/canvas` (header-only model
→ canvas), `/api/gpu`, and the runner — `/api/where`, `/api/run`,
`/api/train`, `/api/jobs`, `/api/gguf/run` (see below).

## Tests

    ./tests/run.sh           # offline: JS syntax, the DOM harness, the builder, the torch runner, HTML
    ./tests/run.sh --live    # + the running app

## Files

| file | role |
|---|---|
| `ui/index.html` | the whole app — the canvas, the palette, the right-click menus, the estimate, and the bottom run ribbon/console; one self-contained page |
| `nn_engine.py` | the GGUF → canvas builder (`build_canvas` is pure; `gguf_canvas` reads a real header) — extracted from next-token's engine |
| `nn_torch.py` | the runner: canvas JSON → a torch module, the forward pass with per-box values, char-level training over every non-frozen weight, GGUF tensor loading (`gguf.quants.dequantize`) |
| `nn_server.py` | the stdlib server: the UI, the GGUF list, the builder, the GPU badge, and the run/train/jobs endpoints |
| `nnstudio_modal.py` | the Modal app (`nnstudio-runner`): the same `nn_torch` on a rented GPU, progress through the `nnstudio-work` volume |
| `tests/ui_boot.js` | the DOM-stub harness (boxes, wires, sizes, estimate, undo, copy/paste, GGUF node, flags) |
| `tests/gguf_canvas_test.py` | the pure builder test, on synthetic tensors |
| `tests/nn_torch_test.py` | the runner test: known-weight forwards, nested ports, op shapes, training, freeze, GGUF loading |
| `models/` | → `~/jevtest/models` (the GGUF files live there) |
| `work/training/<job id>/` | a training run's artifacts: `canvas.json`, `data.txt`, `config.json`, `nnstudio.pt` |

## Run the design — infer and train

The bottom **run ribbon** is one collapsed line under the canvas; the console
expands upward and the canvas flexes around it.

- **▶ run** evaluates the design in torch: every box's output is captured and
  its card wears a small teal **value pill** (`μ 0.42`, or the scalar value) —
  hover it for shape · mean · min · max. The run pane lists the top-level
  **input** boxes; give each numbers (`1`, `0.2 0.4 0.6`), and for a token
  input a `generate` count samples that many ids from the output (shown as
  characters once a training set has defined the vocabulary).
- **train** is char-level next-token over the text you paste (or open). Every
  weight that is not **frozen** trains — the existing freeze flag, read
  through nested boxes, exactly as the estimate panel prices it; the ribbon
  reads `trainable 512 · frozen 0`. The console draws the live loss (train
  gold, eval dashed teal), samples, and the checkpoint path; `stop` saves the
  checkpoint first. A run lands in `work/training/<job id>/` (`canvas.json`,
  `data.txt`, `config.json`, `nnstudio.pt`).
- **a GGUF node runs as-is** through the local llama.cpp (`llama-completion`
  with `-no-cnv`, CUDA): the run pane grows a prompt box + `n` and the
  generated text lands in the output pane. The torch engine can also execute
  a GGUF box with its tensors dequantized — that is the path for models that
  fit in VRAM.
- **where** picks the machine: `this machine` (torch + CUDA) or **Modal**
  (`modal · <GPU>` / `auto`, the smallest card the run fits on). Paid jobs
  carry the caps in the pane (default 30 min / $10, shown live in the ribbon);
  the pre-flight line says what is trainable, what it needs and whether it
  fits locally. Modal setup, once:

      .venv/bin/modal token new     # then reload the page
      # the first remote job deploys the nnstudio-runner app automatically

  The job bundle uploads to the `nnstudio-work` volume; a finished remote
  checkpoint is pulled back into `work/training/<job id>/`.

The runner never lies: quantized weights still do not train as-is, an
operation the engine cannot execute is refused by name, and a dataset whose
characters do not match the output's width is caught before anything starts.

## The canvas — eight kinds of boxes

nnstudio is **one canvas** and nothing else: a **fixed-size canvas**
that never grows with the palette's content, with the **palette on the right
scrolling inside its own column**. On it there are **eight kinds of boxes** —
everything that is left after an atom-by-atom audit of the canvas — all **the
same size and drawn by the same card renderer** as every other box in the app
(rounded body, type top-left, shape chip top-right, rails, port dots):

- **atomic (static) boxes** — they cannot be divided into anything simpler:
  `input`, `output` (the graph's boundaries), `Activation`, `Operation` (one
  tensor op each) and `Norm` (a fused reduction: with no mean/root atoms to
  build it from, it stays whole);
- **weight-bearing boxes** — they carry a learnable table, so they stay whole:
  `Fully Connected` (a real matmul — it is what a GGUF `*.weight` binds to) and
  `Embedding` (a **table read by index**, not a matmul: one id (`1`) goes in,
  the row it names — the vector — comes out of the box's `vocab × out` table;
  `token_embd.weight` reshaped to `[vocab, d_model]` is exactly this, and the
  LM head, which really is a matmul over that table, stays a fully connected
  box);
- **the container** — `Nested`: it *is* how a non-atomic box is written down
  (up to a whole model: the palette's **GGUF node** is one, with the model
  loaded into it);
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
**activation function** for an activation box, or the **operation** for an
operation box.

| box | kind line | dots |
|---|---|---|
| **input** | `input` | one **output** dot, right edge |
| **Embedding** | `Embedding` | input dot left (**one id**, size `1`), output dot right (the vector) |
| **Fully Connected** | `Fully Connected` | input dot left, output dot right |
| **Activation** | `Activation` | input dot left, output dot right |
| **Operation** | `Operation` | as the operation needs: one dot each side (transpose, rope, topk, conv1d, scan, reshape) or **two input dots** (dot product, matmul, arithmetic) |
| **Norm** | `Norm` | input dot left, output dot right — the size passes through |
| **Nested** | `Nested` | **as many as you build** (see below) — this is what an attention / state space / MoE template makes |
| **output** | `output` | one **input** dot, left edge |

Names default to `input 1`, `embedding 1`, `fully connected 1`, `output 1`, …
and the size line shows `?` (dimmed) until you configure it.

A `?` is *unconfigured*, not a guess: fresh boxes start that way, the form's
size fields start empty (placeholder `?`), and it is drawn in a dimmer colour
than a real size. Clearing a size in the configure form makes it `?` again.

- **add** — two ways, same boxes: **drag one from the palette on the right**
  (`GGUF`, `input`, `Embedding`, `Fully Connected`, `Activation`, `Operation`,
  `Nested`, `output`)
  (it follows the cursor as a ghost, lights up over the canvas, and lands where
  you release — a plain click drops it in the middle of the view; the GGUF
  node asks which model to load as soon as it lands), or
  **right-click empty canvas** → a small menu (one entry per kind, plus the
  JSON ones) → it asks for the size (a fully-connected box asks for its name,
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
  on f32). The palette's **VRAM estimate** panel then sums the whole graph: FCs
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
  `norm`/`eps` for a norm, `inner` for a nested box — every box may also carry
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
- **the GGUF node** — the palette leads with a **GGUF node** (the empty
  canvas's right-click menu has the same `GGUF box` entry; inside a nested
  box it is gone — a whole model is a top-level thing). Drop it and the
  loader opens right there: pick one of the local GGUFs and press `load`.
  **One click** reads the file's header (only the header — no weights, no
  VRAM) and **loads the model into that box** — the box takes the model's
  name (`Qwen3.8-27B`, `Gemma-4-12B-It`, from the header's `general.name`),
  wears the model's own dots (its inside `tokens`/`logits` boundary boxes:
  one id in, the vocab out, `mtp logits` too when the model has a next-token
  block) and is made of the palette's own boxes (`input`, `Embedding`, `Fully
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
  undo step. **Double-click the box to go inside** (the header says which box
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
- **resize** — right-click a box → `resize this box`: just the size(s) — in/out
  for a fully-connected box, the single size for an input/output box
  (`?` placeholder = unconfigured). Both save in place: the box keeps its
  identity, position and wires;
- **remove** — right-click a box → `remove this box`;
- **move** — drag a box; while it moves it **snaps into line** with the rest of
  the canvas — left / centre / right and top / middle / bottom, against boxes
  of any height — the moment it comes within a few pixels, and the edge it
  snapped to shows as an amber **guide line** for the length of the drag. The
  reach is a few *screen* pixels, so it feels the same at any zoom;
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
  so its middle passes exactly through where you drop it; drop it near its
  natural middle and it snaps straight again. Right-click a bent wire for
  `reset the wire's shape`;
- **delete a wire** — hover it: the wire lights up and a small **×** appears in
  the **middle of that wire** (click it to remove that wire, and only that
  wire); right-clicking a wire offers the same as `remove this wire`;
- **canvas** — drag empty space to pan, the wheel to zoom (~5 % per mouse
  notch, scaled by how far you scroll so a trackpad stays gentle), double-click
  empty space to fit; no toolbar, no panels, no popovers besides the
  right-click menus;
- nothing stays read-only: renaming, resizing, dtypes, tensor bindings and
  nesting are all edits — right-click empty canvas for `export the canvas as
  JSON` / `import JSON …`. The canvas lives in memory (a reload starts blank),
  with undo 60 steps deep (`Ctrl/Cmd+Z`, `Ctrl/Cmd+Shift+Z`, `Ctrl+Y`).
