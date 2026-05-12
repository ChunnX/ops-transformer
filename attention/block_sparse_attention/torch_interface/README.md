# `bsa_npu` — Python interface for `aclnnBlockSparseAttention`

This directory contains a pybind11 + aclnn bridge that exposes the
`attention/block_sparse_attention` kernel to PyTorch (via `torch_npu`). The
sibling `benchmark/` and `test/` directories call into this extension; the
whole `attention/block_sparse_attention/` subtree is self-contained — nothing
imports from anywhere else in the repo.

## Contents

| File | Purpose |
|---|---|
| `torch_interface.cpp` | pybind11 module. Exposes `block_sparse_attention(q, k, v, block_sparse_mask, block_shape, …) → (attention_out, softmax_lse)`. Translates torch tensors ↔ `aclTensor`, builds the `aclIntArray` host buffers, runs the two-stage aclnn invocation (`GetWorkspaceSize` → `aclrtMalloc` → `aclnnBlockSparseAttention` → `aclrtSynchronizeStream`), and frees workspace. |
| `setup.py` | setuptools config. Uses the `ascendc_extension` helper to set include / lib / RPATH paths for CANN + `torch_npu`. Install name is `bsa_npu`. |
| `build.sh` | Thin wrapper: uninstalls any prior install, sets `PFA_LIB=$1` (= `custom` or `ops_transformer`), runs `pip install . --no-build-isolation`. |
| `ascendc_extension.py` | Helper that builds a `torch_npu.utils.cpp_extension.TorchExtension`. Wires CANN + torch_npu include/library paths and the standard `c10/torch/torch_cpu/torch_npu/torch_python/ascendcl` libs. |

## Python API

```python
import torch
import torch_npu          # registers the NPU backend
import bsa_npu

out, lse = bsa_npu.block_sparse_attention(
    query,                    # [B, N, S, D] (BNSD) or [T, N, D] (TND), fp16/bf16
    key,                      # same layout as query (kv_input_layout)
    value,                    # same layout as key
    block_sparse_mask,        # int8 [B, N, q_blocks, kv_blocks] — 1 keep, 0 drop. REQUIRED.
    block_shape,              # list [block_x, block_y] e.g. [128, 128] or [128, 512]. REQUIRED.
    actual_seq_lengths=[S_q] * B,
    actual_seq_lengths_kv=[S_kv] * B,
    num_heads=N,
    num_key_value_heads=N,
    q_input_layout="BNSD",    # or "TND"
    kv_input_layout="BNSD",   # or "TND"
    mask_type=0,
    scale_value=1.0 / math.sqrt(D),
    inner_precise=0,          # 0 = fp32 softmax (high precision), 1 = fp16
    block_size=128,
    pre_tokens=2147483647,    # int32-max = no causal window
    next_tokens=2147483647,   # int32-max = no causal window
    softmax_lse_flag=False,   # set True to actually compute the LSE second output
    atten_mask=None,          # ACCEPTED for API parity, NEVER FORWARDED to the kernel
)
```

Returns:
- `out`: same dtype + shape as `query`.
- `lse`: `[B, N, S, 1]` float32 if `softmax_lse_flag=True`, else `torch.empty({0})`.

### `atten_mask` is intentionally not forwarded

`block_sparse_attention` does not support fine-grained intra-block masking on
CANN 8.5.0-beta.1 (per the kernel's own README and the aclnn behavior we
observed). The Python signature still accepts `atten_mask` as an optional
arg for API parity, but the C++ side passes `nullptr` to
`aclnnBlockSparseAttentionGetWorkspaceSize`. If you need fine-grained
intra-block masking, use a different sparse-attention kernel (e.g. BSA's
`torch_bsa.blitz_sparse_attention` in `experimental/`).

### LSE output is 4-D

`softmax_lse` has shape `[B, N, S, 1]` (the kernel's infershape sets
`LSE_DIM_D=1`). To compare against FIAS LSE which is also `[B, N, S, 1]` and
typically squeezed, do `lse.squeeze(-1)`.

## Build instructions

### Step 1 — build + install the kernel

```bash
bash build.sh --make_clean -j96 --pkg --soc=ascend910b --ops=block_sparse_attention
./build/cann-ops-transformer-custom_linux-aarch64.run
```

**Do NOT use `--experimental`** — `block_sparse_attention` lives in
`attention/` (regular op), not `experimental/`. Passing `--experimental`
filters it out and the kernel + aclnn header won't be packaged.

This installs:
- Kernel binaries: `$ASCEND_OPP_PATH/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910b/block_sparse_attention/`
- aclnn library: `$ASCEND_OPP_PATH/vendors/custom_transformer/op_api/lib/libcust_opapi.so` (contains `aclnnBlockSparseAttention` + `aclnnBlockSparseAttentionGetWorkspaceSize` symbols)
- aclnn header: `$ASCEND_OPP_PATH/vendors/custom_transformer/op_api/include/aclnnop/aclnn_block_sparse_attention.h`

Verify with:
```bash
ls $ASCEND_OPP_PATH/vendors/custom_transformer/op_api/include/aclnnop/ | grep block
# Expected: aclnn_block_sparse_attention.h
nm -D $ASCEND_OPP_PATH/vendors/custom_transformer/op_api/lib/libcust_opapi.so | grep BlockSparse | head -3
# Expected: a handful of T (text) symbols including aclnnBlockSparseAttention
```

### Step 2 — build + install the torch extension

```bash
cd attention/block_sparse_attention/torch_interface
bash build.sh custom
```

The `custom` argument selects the install path that step 1 produced. (The
alternative `ops_transformer` links against `$ASCEND_TOOLKIT_HOME/ops_transformer/`
which doesn't currently ship `block_sparse_attention` — only the `custom`
vendor build does today.)

`build.sh` uninstalls any prior `bsa_npu` install and
runs `pip install . --no-build-isolation`. On success you'll see:
```
Successfully built bsa_npu
Successfully installed bsa_npu-0.0.1
```

### Smoke test

```bash
ASCEND_RT_VISIBLE_DEVICES=5 python3 - <<'EOF'
import torch, torch_npu, math
import bsa_npu

B, N, S, D = 1, 1, 128, 128
q = torch.randn(B, N, S, D, dtype=torch.bfloat16, device='npu')
k = torch.randn(B, N, S, D, dtype=torch.bfloat16, device='npu')
v = torch.randn(B, N, S, D, dtype=torch.bfloat16, device='npu')
mask = torch.ones(B, N, 1, 1, dtype=torch.int8, device='npu')
scale = 1.0 / math.sqrt(D)

out, lse = m.block_sparse_attention(
    q, k, v, mask, [128, 128],
    actual_seq_lengths=[S], actual_seq_lengths_kv=[S],
    num_heads=N, num_key_value_heads=N,
    q_input_layout='BNSD', kv_input_layout='BNSD',
    scale_value=scale,
)
assert out.shape == (B, N, S, D)
assert torch.isfinite(out).all()
print('OK')
EOF
```

Expected output: `OK`.

## Running the benchmarks + tests

Once the torch extension is installed, run the head-to-head benchmark:

```bash
cd attention/block_sparse_attention/benchmark
ASCEND_RT_VISIBLE_DEVICES=1 python benchmark.py
# Or with coarser KV block granularity:
ASCEND_RT_VISIBLE_DEVICES=1 BSA_BLOCK_SIZE_KV=512 python benchmark.py
```

See `../benchmark/README.md` for what the table shows.

The correctness pytest suite (attention output vs PFA, LSE vs FIAS / Python
float32 reference):

```bash
cd attention/block_sparse_attention/test
ASCEND_RT_VISIBLE_DEVICES=1 pytest test_attn.py test_lse.py test_joint.py -v
```

- `test_attn.py` — attention output vs `npu_fusion_attention` over a
  shape/frame/sparsity grid (H ∈ {1,2,3,4}, S=24000, 2 frames, 10 sparsities).
- `test_lse.py` — softmax_lse: dense vs FIAS, sparse H=1 vs FIAS (definitional
  band), sparse multi-head vs Python float32 reference, zero-input log(S)
  invariant, shape/dtype, disabled-LSE.
- `test_joint.py` — single kernel call with `softmax_lse_flag=True`,
  attn_out + LSE both verified.

Attention output passes strict tolerance (`rtol=0.01, atol=0.001`) at every
sparsity. LSE matches FIAS strictly in dense; under sparse, LSE diverges from
FIAS by ~0.06–0.16 (a definitional difference — the kernel's LSE matches the
Python float32 reference within `~5e-2`, so it's correct per its own
definition).

## Rebuild loop

If you edit the kernel:
```bash
bash build.sh --make_clean -j96 --pkg --soc=ascend910b --ops=block_sparse_attention \
  && ./build/cann-ops-transformer-custom_linux-aarch64.run \
  && (cd attention/block_sparse_attention/torch_interface && bash build.sh custom)
```

If you edit only `torch_interface.cpp`:
```bash
cd attention/block_sparse_attention/torch_interface
bash build.sh custom
```

## Implementation pitfalls (already addressed in `torch_interface.cpp`)

1. **`block_sparse_mask` dtype is `int8` / `torch.int8`** (not bool, not
   uint8). Passing the wrong dtype causes an aclTensor-create error or a
   downstream "param invalid" from the kernel.
2. **LSE shape is 4-D `[B, N, S, 1]`** when `softmax_lse_flag=True`. The
   kernel's infershape hardcodes the trailing 1. Passing a 3-D `[B, N, S]`
   tensor makes `aclnnBlockSparseAttentionGetWorkspaceSize` fail with error
   `561103`. The C++ wrapper allocates 4-D.
3. **`pre_tokens` / `next_tokens` defaults to `2147483647`** ("max int32" =
   unrestricted window). Passing `0` likely rejects the call. Per the
   kernel README the window restriction "当前不支持" — pass infinity, the
   block_sparse_mask carries any sparsity you need.
4. **Layout strings need null-terminated 16-byte buffers** ("BNSD"/"TND"); the
   wrapper uses `char[16]` + `strncpy` (mirrors the kernel's own C++ test
   example).
5. **`aclIntArray` host data must outlive the aclnn call** — the wrapper
   keeps the `std::vector<int64_t>` source alive for the duration and
   destroys the arrays after `aclrtSynchronizeStream`.
6. **No `--experimental` build flag** for this kernel — see Step 1 above.
