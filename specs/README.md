# Family specs + the hf_arc database — verified architecture, per family and per repo

A **spec** is the ground-truth answer to "how are this family's boxes wired?".
nnstudio's HF-repo loader (`nn_engine.hf_canvas`) resolves a repo in this order:

1. **`hf_arc`** (the repo-architecture database, `nn_arc.py`, `hf_arc.db`) — a row
   keyed by the repo id holding the serialized nested-box canvas the reverse job
   researched and proved for *this repo*. Instant and exact (`canvas.model.arc`).
2. a **family spec** for its `model_type` — rendered as **native boxes only**,
   nested per block, every leaf one of the app's own, every weight bound to its
   repo tensor and shard.
3. without either, the role-based sketch, marked `approximate` — and `build
   canvas` starts the reverse job, whose checklist fills in on the box until the
   exact canvas (and the `hf_arc` row) are ready.

A spec is trusted only when `tests/spec_conformance_test.py` passes it: a tiny
random model of the family is run through the library's **eager** path and
through the canvas, and every stage (embedding, q/k/v projections, every block,
logits) must match to 1e-4 — in practice they match exactly. The job re-runs
that gate and then binds the spec to the repo's real tensor names
(`nn_engine.arc_canvas`); only on success does the `hf_arc` row land.

## Files

- `specs/<family>.json` — one architecture family (a `model_type` like `llama`).
- `hf_arc.db` — `hf_arc(repo_id TEXT PRIMARY KEY, canvas TEXT, created_at,
  transformers, note)`: each researched repo's canvas as JSON. The table is the
  cache and the deliverable — no agent say-so lands a row, only the job's gates.
- `nn_arc.py` — the database plus the job's file state (`<slug>.json`,
  `<slug>.todos.json`, `<slug>.pipeline.json`).
- `tests/spec_conformance_test.py` — the gate (`python tests/spec_conformance_test.py llama`).
- `tests/reverse_job_test.py` — the job pipeline with a stubbed agent.
- `tools/reverse_job.py` — the background job: agent, checklist, gates, row.
- `tools/reverse_model.py` — the authoring toolbox: `sources`, `inspect`, `check`,
  `list`, and `report` (the checklist the box shows).
- `.opencode/commands/reverse-model.md` — the AI workflow (`/reverse-model llama`).

## Schema

```jsonc
{
  "family": "llama",
  "match": {"model_type": ["llama"]},          // config.json model_type values
  "verified": {"transformers": "5.18.0", "date": "…", "note": "…"},  // set by the gate

  "config": {                                   // spec slot <- config.json key(s)
    "d_model":    {"keys": ["hidden_size"]},
    "blocks":     {"keys": ["num_hidden_layers"]},
    "heads":      {"keys": ["num_attention_heads"]},
    "kv_heads":   {"keys": ["num_key_value_heads"], "from": "heads"},   // fallback slot
    "head_dim":   {"keys": ["head_dim"], "expr": "d_model / heads"},    // computed
    "intermediate": {"keys": ["intermediate_size"]},
    "rope":       {"keys": ["rope_theta"], "default": 10000.0},
    "eps":        {"keys": ["rms_norm_eps", "layer_norm_eps"], "default": 1e-5},
    "act":        {"keys": ["hidden_act"], "default": "silu"}
  },

  "weights": {                                  // top-level tensors (nested blocks bind by `param`)
    "embed": "model.embed_tokens.weight",
    "final_norm": "model.norm.weight",
    "head": "lm_head.weight"                    // may be absent — falls back to `embed`
  },

  "block": {
    "label": "blk {i} · llama",
    "nodes": [
      {"id": "in", "kind": "input", "name": "in", "pos": [0, 0], "dout": "d_model"},
      {"id": "q", "kind": "layer", "name": "q proj", "pos": [480, -160],
       "din": "d_model", "dout": "heads*head_dim", "param": "self_attn.q_proj.weight"},
      {"id": "mask", "kind": "op", "name": "causal mask", "pos": [2460, -80],
       "op": "mask", "mode": "causal"},
      {"id": "out", "kind": "output", "name": "out", "pos": [5320, -160], "din": "d_model"}
    ],
    "edges": [["in", "attn_norm"], ["rope_q", "scores", 0, 0], ["kT", "scores", 0, 1]]
  },

  "test": {                                     // the tiny model the gate builds
    "model_type": "llama",
    "config": {"hidden_size": 32, "num_hidden_layers": 2, …}
  }
}
```

### Node fields
| field | meaning |
|---|---|
| `id` | local name; edges reference it |
| `kind` | `input` · `embed` · `layer` · `norm` · `act` · `op` · `output` (native boxes only) |
| `name`, `pos` | what the box is called and where it sits inside the block |
| `param` | repo tensor suffix; the runner prefixes `model.layers.{i}.` inside a block, binds `…weight` plus a `…bias` when the repo has one |
| `optional` | drop the box when its tensor is absent (and splice its wire) |
| `if` | expression; drop the box when false (e.g. `"heads != kv_heads"`) |
| `dout`/`din`/`vocab`/`size` | numbers or arithmetic over config slots |
| op attrs | `dims` (reshape), `axes` (permute), `repeats`+`dim` (repeat), `factor` (scale), `theta` (rope), `mode`/`window` (mask), `act` (act) |
| `"act": "act"` | an attr named after a config slot resolves to it (`silu` here) |

A bound node keeps the file's own numbers: `dtype` is the storage dtype from the
safetensors header (`bf16`, `f8_e4m3`, `u8`, …), `wbytes` is what the file
actually stores, and `params` is what the block holds when it runs — a packed
4-bit tensor (`weight_packed`, U8) keeps **two weights per byte**, so its logical
count is twice its element count. The estimate and the load use these, not a
default dtype.

`op` names the engine executes: `matmul`, `reshape`, `permute`, `rope`,
`transpose`, `repeat`, `mask`, `scale`, `add`, `subtract`, `multiply`, `divide`,
`concat`, `topk`.

## Adding a family (or a repo)
1. `/reverse-model <model_type|repo>` (or do it by hand with the toolbox) — note
   the app starts the very same job by itself when `build canvas` meets a repo
   without an `hf_arc` row (`tools/reverse_job.py`, `POST /api/reverse`; the
   checklist on the box, the exact canvas swapped in when the gates pass).
2. `python tools/reverse_model.py check <family>` until every stage is green.
3. Multi-model repos: the agent writes `<slug>.pipeline.json` next to the job
   state; the job assembles the models into one top-level pipeline box.
4. `./tests/run.sh` — the suite runs the conformance gate with the rest.

Two library notes that cost us a day:
- pin `attn_implementation="eager"` when comparing — fused kernels hide the math;
- transformers v5 replaces `hidden_states[-1]` with the post-final-norm state
  (`tie_last_hidden_states`), so the gate compares **layer module outputs**.
