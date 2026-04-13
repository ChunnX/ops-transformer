# Blitz Sparse Attention – Codebase Guide

## Known limitations / TODOs

**TODO: variable sabi granularity**
Currently sabi indices are always in units of **128×128** (Q-tile × KV-tile) blocks.
The original design used 128×512 granularity.  A future task is to support both
granularities dynamically — i.e. let the caller choose whether each sabi index
addresses a 128×128 or a 128×512 KV block — without requiring a kernel rebuild.
This affects sabi shape (`ceil(S/128)` KV axis today vs `ceil(S/512)` before),
tiling, and all Python tooling that generates or interprets sabi tensors.

**TODO: batch size > 1 is broken**
Only `B=1` (batch size 1) is known to produce correct results.
For `B>1` the output is incorrect (wrong memory access due to stride mismatches in
the host-side binding or kernel tiling).  All tests and benchmarks should be run
with `B=1` until this is diagnosed and fixed.

---

## Before running anything

Always set the NPU device to unit 5 before running any Python script, test, or benchmark:

```bash
export ASCEND_RT_VISIBLE_DEVICES=5
```

Without this, commands may land on the wrong NPU or fail to find the device.

## Kernel Landscape (the three kernels you need to know)

### PFA – `attention/prompt_flash_attention`
Dense prompt flash-attention. The original source from which BSA was adapted;
most of its file names, tiling infrastructure, and kernel templates are identical.
- Single output: `attention_out` (same dtype as query, BNSD/BSH/BSND)
- No LSE output

### BSA – `experimental/attention/blitz_sparse_attention`  ← this repo
Block-sparse flash-attention built on top of PFA. Adds the `sabi` input (Sparse
Attention Block Indices, `uint16`, shape `[B, N, q_tiles, kv_tiles_kept]`, padded
with `0xFFFF`) to skip entire 128×128 KV blocks.

Key directories:
| Path | Role |
|---|---|
| `op_host/blitz_sparse_attention_def.cpp` | Op schema: inputs, outputs, attrs |
| `op_host/blitz_sparse_attention_infershape.cpp` | Output shape / dtype inference |
| `op_host/blitz_sparse_attention_tiling.cpp` | Tiling + `ConvertContextToPFAParams` |
| `op_host/blitz_sparse_attention_tiling_context.h` | `ContextParamsForPFATiling` struct |
| `op_host/op_api/blitz_sparse_attention_base_aclnn.cpp` | Inner L0 op builder |
| `op_host/op_api/aclnn_blitz_sparse_attention_inner.h/.cpp` | Inner aclnn API |
| `op_host/op_api/aclnn_blitz_sparse_attention.h/.cpp` | Public aclnn API |
| `op_kernel/blitz_sparse_attention_s1s2_bns1_x910_base.h` | Main AICore kernel template |
| `torch_interface/torch_interface.cpp` | PyTorch / pybind11 entry point |
| `benchmark/test_attn.py` | pytest correctness suite that tests only the 1st output - attention output|
| `benchmark/test_lse.py` | pytest correctness suite that tests only the 2nd output - lse |
| `benchmark/test_joint.py` | pytest correctness suite that kointly tests attention and lse outputs |
| `benchmark/benchmark.py` | Latency benchmark + data generators |

Declared outputs: `attention_out` (index 0), `softmax_lse` (index 1, float32).  
`softmax_lse_flag` bool attr (index 8) gates LSE output; when false a `{0}`-shaped
placeholder is passed so the op always receives a valid second output pointer.

Supported layouts: `BNSD`, `BSH`, `BSND`, `NSD`, `SH` (no `TND`).

### FIAS – `attention/fused_infer_attention_score`
Inference-focused fused attention (paged / GQA). Used here as the **LSE golden
standard** because it already emits `softmax_lse` via the `softmaxLseFlag` bool
attr + second required output.

LSE output spec in FIAS:
- Output index 1: `softmax_lse`, dtype `DT_FLOAT`, `FORMAT_ND`
- Shape (non-TND layouts): `{B, N, q_len, 1}` when flag=true; `{0}` when false
- Attr: `softmax_lse_flag` (bool, optional, default false)

PyTorch exposure via `torch_npu.npu_fused_infer_attention_score_v5(...)`.

---

## Important invariants

- BSA tile sizes are fixed at **128×128** (Q-tile × KV-tile).
- `sabi` indices are in units of KV tiles; value `0xFFFF` = skip this tile.
- `isSoftMaxLseEnable` gates UB allocation (+256 B), output init, and write-out
  paths in the kernel; it must be set in tiling *before* workspace sizing.
- `softmax_lse` output shape is `[B, N, q_len]` (float32, 3-D). TND layout is
  not supported for LSE and will return an empty `{0}` tensor.
- Only BNSD layout is exercised in tests/benchmarks on 910B2.

---

## Host-side layer stack

Understanding the 4-layer stack is essential for debugging output-binding bugs.
Each layer has a different job:

```
torch_interface.cpp          ← Python boundary: allocates torch Tensors, calls aclnn public API
  └─ aclnnBlitzSparseAttentionGetWorkspaceSize   (aclnn_blitz_sparse_attention.cpp)
       └─ InnerBlitzSparseAttentionGetWorkspaceSize  (aclnn_blitz_sparse_attention_inner.cpp)
            ├─ l0op::BlitzSparseAttention            (blitz_sparse_attention_base_aclnn.cpp)
            │    └─ executor->AllocTensor()  ← internal scratch tensor, NOT user memory
            │    └─ INFER_SHAPE / ADD_TO_LAUNCHER_LIST_AICORE
            └─ l0op::ViewCopy(l0Out, userTensor)  ← binds scratch → user memory
```

`AllocTensor()` at the L0 layer creates executor-owned scratch. The kernel writes
into that scratch. `ViewCopy` is the only thing that routes kernel output to the
tensor the user actually allocated. If a `ViewCopy` is missing for an output, the
kernel result is silently discarded.

---

## LSE implementation – complete wiring map

The LSE feature was added after the initial BSA implementation. The kernel already
computed LSE internally; what was missing was the full host-side + interface wiring.

### What the kernel already had (no changes needed there)
- `GlobalTensor<float> softmaxLseGm` in `op_kernel/blitz_sparse_attention_s1s2_bns1_x910_base.h`
- `softmaxLseGm.SetGlobalBuffer((__gm__ float*)softmaxLse)` in `Init()`
- `DataCopy` to `softmaxLseGm[headParams->SoftMaxOffset]` in FlashUpdate, gated on `isSoftMaxLseEnable`
- `isSoftMaxLseEnable` and `lseoutputShape` fields in tiling data / tiling context

### Files changed and what each does

| File | Change |
|---|---|
| `op_host/blitz_sparse_attention_def.cpp` | Added `softmax_lse` as REQUIRED output (index 1, DT_FLOAT, ND) and `softmax_lse_flag` as OPTIONAL bool attr (index 8, after `inner_precise`). Same in `config_310p`. |
| `op_host/blitz_sparse_attention_infershape.cpp` | Added `InferBsaLseShape` helper (BNSD/BSND/BSH → `{B,N,S}`, unsupported → `{0}`). Sets LSE output shape and always sets LSE dtype = DT_FLOAT. |
| `op_host/blitz_sparse_attention_tiling.cpp` | In `ConvertContextToPFAParams`: reads `GetOutputShape(1)` → `lseoutputShape`, reads attr index 8 → `isSoftMaxLseEnable`. |
| `op_host/op_api/blitz_sparse_attention.h` | Added `bool softmaxLseFlag` + `const aclTensor *softmaxLse` before `executor` in L0 op declaration. |
| `op_host/op_api/blitz_sparse_attention_base_aclnn.cpp` | Allocated `softmaxLseOut` tensor; added both to `OP_OUTPUT` and `softmaxLseFlag` to `OP_ATTR` in INFER_SHAPE + ADD_TO_LAUNCHER_LIST_AICORE. Exposes `softmaxLseOut` via optional `const aclTensor **softmaxLseOutRet` out-param so callers can bind it to user memory with `ViewCopy`. |
| `op_host/op_api/aclnn_blitz_sparse_attention_inner.h/.cpp` | Added `softmaxLseFlag` + `softmaxLse` params; FIAS-style `{0}` placeholder created when flag=false so the op always gets a valid output pointer. When flag=true, retrieves `l0LseOut` from the L0 builder and calls `l0op::ViewCopy(l0LseOut, softmaxLse, l0Executor)` to bind the kernel's LSE scratch buffer to the user's output tensor. **Critical:** without this ViewCopy the kernel wrote correct LSE values into an internal scratch buffer that was never mapped to user memory, producing garbage output. |
| `op_host/op_api/aclnn_blitz_sparse_attention.h/.cpp` | Public API: added same two params, passes through to inner. |
| `torch_interface/torch_interface.cpp` | Return type `tuple<Tensor,Tensor>`; allocates `lse_out` as `{B,N,S}` float32 when flag=true, `{0}` otherwise; pybind11 updated. |
| `op_kernel/blitz_sparse_attention.cpp` | Added `__gm__ uint8_t* softmaxLse` between `attentionOut` and `workspace` in the kernel entry function `blitz_sparse_attention`, forwarded to `blitz_sparse_attention_FIAS` (was `nullptr`). |
| `benchmark/test_attn.py` | Unpack tuple: `out, _ = torch_bsa.blitz_sparse_attention(...)` |
| `benchmark/benchmark.py` | Same tuple unpack in `_make_our_fn`. |
| `benchmark/test_lse.py` | New file: pytest suite comparing BSA LSE vs FIAS LSE (`rtol=1e-3, atol=1e-3`), dense (sparsity=0) only. |

### Attr index table (def.cpp order → infershape/tiling index)

| Index | Name |
|---|---|
| 0 | `num_heads` |
| 1 | `scale_value` |
| 2 | `pre_tokens` |
| 3 | `next_tokens` |
| 4 | `input_layout` |
| 5 | `num_key_value_heads` |
| 6 | `sparse_mode` |
| 7 | `inner_precise` |
| **8** | **`softmax_lse_flag`** ← new |

### The build-error lesson (critical for future debugging)

When adding a new REQUIRED output to `blitz_sparse_attention_def.cpp`, the CANN
build system regenerates a kernel stub that calls the kernel entry function
`blitz_sparse_attention<TEMPLATE_PARAMS>(...)` with the new output as an extra
positional argument **between the last declared output and `workspace`**.

The kernel entry function `blitz_sparse_attention` (in `op_kernel/blitz_sparse_attention.cpp`,
lines ~320–333) is a thin wrapper that accepts raw `__gm__ uint8_t*` pointers for all
inputs/outputs and forwards them to the actual template `blitz_sparse_attention_FIAS`.
Its parameter list must **exactly match** the output order in `def.cpp`.

After adding `softmax_lse` as output index 1, the generated stub passes:
```
..., attentionOut, softmaxLse, workspace, tiling
```
so the kernel entry function signature had to be updated to match (adding
`__gm__ uint8_t* softmaxLse` between `attentionOut` and `workspace`), and the
forwarding call to `blitz_sparse_attention_FIAS` changed from `nullptr` to `softmaxLse`.

**Rule:** any new output in `def.cpp` = new `__gm__ uint8_t*` param in the same
position in `blitz_sparse_attention.cpp`'s entry function, before `workspace`.

### FIAS placeholder pattern

When `softmaxLseFlag=false` the op must still receive a valid tensor pointer for
the second output. Both FIAS and BSA use the same placeholder: a `{0}`-shaped
ACL_FLOAT tensor whose backing address is a dummy local (`int64_t addr = 0xff`).
See `aclnn_blitz_sparse_attention_inner.cpp` for the implementation.

### ViewCopy pattern for multiple outputs (critical)

In the CANN aclnn layer, `executor->AllocTensor()` creates **internal** scratch
tensors owned by the executor; they are **not** automatically bound to the user's
output memory. The binding is done by `l0op::ViewCopy(internalTensor, userTensor, executor)`.

For a single-output op the pattern is straightforward. For multi-output ops (like
BSA after adding `softmax_lse`) each output needs its own `ViewCopy`. The L0
builder (`blitz_sparse_attention_base_aclnn.cpp`) only returns the primary output
(`attentionOutOut`); secondary outputs are internal and invisible to the caller.

**The fix**: the L0 builder now accepts an optional `const aclTensor **softmaxLseOutRet`
out-param. The inner layer (`aclnn_blitz_sparse_attention_inner.cpp`) passes
`&l0LseOut`, then calls `l0op::ViewCopy(l0LseOut, softmaxLse, l0Executor)`.

**Symptom of a missing ViewCopy**: kernel writes correct values into a scratch
buffer that is never mapped to user memory. `torch::empty` output stays
uninitialized — for zero input this happened to read as 0.0 (not `log(S) = 6.93`),
and for random input values were unrelated to attention statistics.

**Rule**: whenever the L0 builder allocates an output tensor with `AllocTensor()`,
check that a corresponding `ViewCopy` exists in the caller for every output the
user cares about.

### `isSoftMaxLseEnable` side effects in tiling

Setting `isSoftMaxLseEnable=true` in tiling has one non-obvious side effect:
it also sets `enableRingAttention=true`, which **disables the CVDIFF tiling path**.
This is intentional — ring attention and CVDIFF are mutually exclusive — but means
LSE-enabled runs use a different tiling branch than LSE-disabled runs. If you see
tiling or performance differences between `softmax_lse_flag=true/false`, this is why.

### Quick LSE sanity check

For zero input (Q=K=V=0), the attention scores are all 0, so softmax is uniform
over S tokens. LSE must equal `log(S)` regardless of B, N, or D:

```python
q0 = torch.zeros(B, H, S, D, dtype=dtype, device=DEVICE)
_, lse = torch_bsa.blitz_sparse_attention(q0, q0, q0, ..., softmax_lse_flag=True)
assert torch.allclose(lse, torch.full_like(lse, math.log(S)), atol=1e-3)
```

This check is cheap, layout-independent, and catches missing ViewCopy, wrong
`SoftMaxOffset`, incorrect GatherMask, and uninitialized output — all at once.
