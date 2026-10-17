# Backend: quantized layers, and what runs natively

This page is about what the local backend (the agent) can execute and what it
needs from the machine. nnstudio never installs anything — it detects, uses
what is there, and reports what is missing.

## Native GGUF quantized layers

A **frozen** layer box bound to a GGUF file can keep its weights at rest in
ggml's block form and run a fused dequant–GEMM on the GPU: the matrix is never
materialized as f16, so a Q4_K model costs ~4.5 bits per weight of memory
instead of 16. The kernel lives in `agent/nn_quant.py` and is verified
tensor-by-tensor against `gguf.quants.dequantize` (`tests/nn_quant_test.py`).

| how | where | needs |
| --- | --- | --- |
| fused Triton kernel — `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q2_K`, `Q3_K`, `IQ4_XS`, `IQ4_NL` | any layer box on a GPU | PyTorch with Triton (bundled) + CUDA or ROCm |
| dequantize at load → f16 | everywhere else | nothing (this is the original path) |

Rules that never change:

- **Training** always dequantizes at load; masters are f16/f32. No native
  kernels are used for trainable weights.
- Any quant type without a kernel (the IQ3/IQ2 family, `IQ1_*`, …)
  dequantizes at load exactly as before.
- Offloading a native box to system RAM is safe: the forward falls back to the
  reference host-side dequantization.
- The native path adds **no dependencies**: Triton ships inside the PyTorch
  wheel.

The Backend menu shows `quant kernels: Q4_K, Q8_0 · triton/cuda` when the
connected agent has them, or `f16 dequant at load · docs/backend.md` when it
does not. The dtype picker marks the native types with `· native`.

## Vision towers (mmproj)

An mmproj file is a multimodal model's **vision encoder**: patch embedding
(two summed convs), a stack of vision blocks, and the projector that merges
2×2 patch groups into embeddings the text model can read. nnstudio runs the
whole tower in torch as one `vision` box (`agent/nn_torch.py:VisionTower`),
ported from llama.cpp's `clip_graph_qwen3vl` and verified stage-by-stage
against `llama-mtmd-debug`'s tensor dump on synthetic images (gray /
checkerboard / rainbow at 64 and 128 px):

| piece | math |
| --- | --- |
| patch embeds | `conv2d(16×16, stride 16)` twice, summed, + bias |
| positions | the learned table, bilinear-resized (`align_corners`) to the patch grid |
| blocks | LayerNorm (bias), fused qkv, **vision M-RoPE** (pairs `(e, e+36)`; the first section by y, the second by x, base 10000), full attention `1/√d`, GELU (tanh) ffn |
| merger | 2×2 patch groups concatenated, `mm.0` → GELU → `mm.2` |

Supported projector types: `qwen3vl_merger`. Anything else is **refused by
name** when the GGUF is loaded, so a design never lands half-working.
Weights stay f16 at rest (the box's estimate is the file's own bytes); the
forward computes in f32. The tower honors `place` (gpu/system) like any other
weight box.

An image input box emits the **picture itself** as a matrix of RGB channels
(`[h, w, 3]`, 0..1) — it travels the wires like any value, so an output box it
reaches draws it — and the tower does its own preprocessing: resized to `image_size`
(the browser already sends at that side; the engine re-does it for API
clients), normalized with the file's `image_mean`/`image_std`, and cut into
`patch × patch` patches (rows `k = x + grid_w·y`). Pre-cut patch rows are
still accepted on the tower's input for API clients. A run sends the raw
pixels (RGB8, base64) with the design — the backend never reads the
browser's blob store.

Any box that receives a matrix the backend **determines** to be a picture — an
RGB matrix in 0..1 (`[h, w, 3]`, or a square `[n, 3]` pixel list) — gets it
drawn as-is: the run's stats carry the image, and for matrices with no media
store entry behind them a small base64 thumbnail (`image.thumb`) rides along so
the app can draw it. Images that came from an image input carry the blob
instead — the browser shows the full picture.

## Per platform

| platform | native kernels | notes |
| --- | --- | --- |
| NVIDIA CUDA | yes (`triton/cuda`) | any CUDA PyTorch wheel; the driver is the only system piece |
| AMD ROCm | yes (`triton/rocm`) | PyTorch ROCm wheels (ROCm 6+) include Triton |
| Apple MPS (Metal) | no (Triton is CUDA/ROCm-only) | the agent probes the metal gpu and uses it (`device: mps`); GGUF layers dequantize at load (works; more memory). `place: gpu` = the metal working set, `place: system` = plain RAM; the memory is unified either way. Ops metal lacks fall back to the cpu (`PYTORCH_ENABLE_MPS_FALLBACK`, set by the agent); f64 storage becomes f32 (metal has no float64; bf16 needs macOS 14+), and a design the working set cannot hold is retried on the cpu once, with the reason in Spec |
| CPU (Linux/Windows) | no | same fallback; CPU matmul in f32 |

## Optional providers (user-installed)

nnstudio detects these but never installs them. Install them yourself **before**
starting the backend if you want the path they unlock.

| package | unlocks | install (your choice) |
| --- | --- | --- |
| `llama-cpp-python` | whole-model GGUF execution (dedicated box) | `pip install llama-cpp-python` |
| `torchao` | future lower-bit CUDA kernels | `pip install torchao` |
| `bitsandbytes` | future 8/4-bit paths | `pip install bitsandbytes` |

None of these are required. The core canvas, the native GGUF kernels and the
f16 fallback all run with PyTorch + gguf alone.

## Check your machine

    .venv/bin/python tools/backend_check.py

Read-only: it prints platform, torch/Triton, the native kernel set and which
optional providers are present, with hints for what is missing. It installs
nothing and changes nothing.

## Troubleshooting

- A run fails with `INTERNAL ASSERT FAILED` at `c10/cuda/PeerToPeerAccess.cpp`
  (or `NVML_SUCCESS == ... nvmlInit_v2`): PyTorch's CUDA allocator asked the
  driver's NVML for peer/fabric access and it could not initialize there (a
  driver/NVML mismatch, or a container without the driver's NVML). The agent
  keeps the expandable-segments IPC branch off and, when NVML cannot
  initialize, takes the async allocator (`backend:cudaMallocAsync`), which
  never asks NVML; it also probes the GPU once with a real allocation, so a
  machine whose CUDA path is broken runs on the CPU and says why in **Spec**
  (`cuda: ...`). If it still appears, update the NVIDIA driver.
- `quant kernels: f16 dequant at load` when you expected native:
  - no GPU, or PyTorch built for CPU: check `torch.cuda.is_available()`;
  - Triton missing: use a PyTorch wheel that bundles it (all official CUDA and
    ROCm wheels since 2.x), or ignore it — the fallback is correct, just
    heavier.
- A frozen box on the GPU still loads f16: the tensor's type has no kernel
  yet (see the table), or the box is trainable (not `frozen`), or its
  `place` is `system`.
- On Apple Silicon: `mps` appears in Spec when the metal gpu is in use; the
  Spec's `gpu` line shows the chip and its recommended working set, and the
  live `vram now` is what the backend itself holds of it — other apps' gpu
  use is not visible through torch, and the memory is unified with RAM, so
  the `ram now` line matters too. A run the working set cannot hold fails
  once and is retried on the cpu automatically; the next run goes straight
  there (`mps: ...` says why). Raise the cpu offload bar or use *load all on
  system* to keep a big design on plain RAM from the start.
