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

## Per platform

| platform | native kernels | notes |
| --- | --- | --- |
| NVIDIA CUDA | yes (`triton/cuda`) | any CUDA PyTorch wheel; the driver is the only system piece |
| AMD ROCm | yes (`triton/rocm`) | PyTorch ROCm wheels (ROCm 6+) include Triton |
| Apple MPS | no | GGUF loads by dequantizing at load (works; more memory) |
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
  keeps the expandable-segments IPC branch off
  (`TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC=0`) and only asks for expandable
  segments when NVML initializes cleanly, so this should not surface; if it
  still does, update the NVIDIA driver (or start the agent with that variable
  set to `0`).
- `quant kernels: f16 dequant at load` when you expected native:
  - no GPU, or PyTorch built for CPU: check `torch.cuda.is_available()`;
  - Triton missing: use a PyTorch wheel that bundles it (all official CUDA and
    ROCm wheels since 2.x), or ignore it — the fallback is correct, just
    heavier.
- A frozen box on the GPU still loads f16: the tensor's type has no kernel
  yet (see the table), or the box is trainable (not `frozen`), or its
  `place` is `system`.
