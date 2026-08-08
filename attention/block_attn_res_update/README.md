# BlockAttnResUpdate

`BlockAttnResUpdate` is an Ascend 950-only, AIV/RegBase implementation of the BlockAttnRes phase-2 update.
It has six tensor inputs, two tensor outputs, one framework workspace slot, and one tiling argument. The input and
output named `partial_block` form an ACLNNGraph reference tensor and share one logical allocation. The kernel ABI
still carries both `partial_block_ref` and `workspace` address slots. Host tiling requests zero workspace bytes, and
the kernel does not dereference either address. The implementation relies on the reference alias and writes the
updated partial result back through the `partial_block` input address. Direct kernel launchers must pass the same GM
address for `partial_block` and `partial_block_ref`.

```text
block_attn_res_update(
    partial_block,       // FP32 [T, D], input
    delta,               // BF16 [T, D]
    inter_numerator,     // FP32 [T, D]
    inter_max,           // FP32 [T]
    inter_exp_sum,       // FP32 [T]
    effective_query,     // FP32 [D]
    partial_block_ref,   // FP32 [T, D], aliases partial_block
    h,                   // BF16 [T, D]
    workspace,           // ABI slot only; zero bytes requested
    tiling)
```

`eps` is an optional finite positive float attribute. When it is omitted, host tiling uses `1e-6`; the selected value
and the `uint8_t phase2Mode` are serialized into `BlockAttnResUpdateTilingData`. The kernel computes `1 / D` from
`dSize` and never reads an `eps` tensor. The RMS path intentionally uses FP32 `sqrt` followed by direct division
and does not include Newton refinement or special handling for zero/extreme values.

Host tiling requires concrete positive `T` and `D`, with `D <= 8192`. It rejects zero/unknown/negative dimensions,
incorrect ranks, inconsistent related shapes, unsupported input/output dtypes, shapes whose `T * D` element count
cannot be represented by the kernel signed 64-bit GM element-offset type, and shapes whose per-core T size cannot
be represented by `uint32_t`.

## Tiling

The T dimension is split across vector cores; one complete D vector at a given T index always remains on one core
because RMS and dot-product reductions are performed inside that core. The kernel always uses two UB buffers. `tileT`
is capped at `ceil(maxTPerCore / 2)` so the busiest core uses both ping-pong buffers whenever its T size is at least
two; tiling fails if one complete D vector cannot fit with double buffering.

Every T slice in UB starts at a 32-byte boundary. Since the element sizes differ, tiling carries both aligned D
strides:

```text
dAlignFp32 = AlignUp(D, 8)    // 8 FP32 = 32 B
dAlignBf16 = AlignUp(D, 16)   // 16 BF16 = 32 B
phase2Mode = D <= 128 && tileT >= 8 ? BATCH : TOKEN
statsTStride = phase2Mode == BATCH ? AlignUp(tileT, 64) : AlignUp(tileT, 8)
```

The effective reduction length is always `D`; aligned elements are padding only.

## UB layout

```text
0
├─ effective_query : dAlignFp32 * 4
├─ buffer 0
│  ├─ partial / p  : tileT * dAlignFp32 * 4
│  ├─ delta / h    : tileT * dAlignBf16 * 2  (reused after phase 1)
│  ├─ numerator    : tileT * dAlignFp32 * 4
│  └─ stats        : 3 * statsTStride * 4    (interMax/interExpSum/score)
└─ buffer 1 (same layout)
```

UB regions use fixed byte offsets. GM/UB views are materialized with Tensor API `MakeMemPtr`/`MakeTensor`, and
staging uses `CopyGM2UB`/`CopyUB2GM` with `NDExtLayoutPtn` T-axis pitches. A full-register VF load may fetch aligned
tail data beyond the effective D size; invalid lanes are cleared or masked before use, and host tiling
leaves 8 KB of UB unassigned as physical headroom. Synchronization uses
`SetFlag`/`WaitFlag` for `MTE2_V`, `V_MTE3`, and `MTE3_MTE2`; no queue, pipe, or dynamic UB allocator is used.
The VF body uses the public `Reg::LoadAlign`/`Reg::StoreAlign` interfaces for
UB/register movement and a tail `MaskReg` for the effective D elements.
Phase 1 handles full registers with an explicit two-way unroll and independent FP32 square/dot accumulators. The
one- and two-register paths keep the complete query resident across T. The generic path also preloads its first
query pair plus optional odd/tail registers, peels the first pair to initialize the accumulators without per-T
zeroing, and streams only middle query pairs. `Reg::MulAddDst` shortens the accumulation chains without adding UB
scratch or a local-memory barrier. For the only partial register, masked `Add` clears invalid `partial` lanes and a
preload-time zero-shift `Reg::ShiftLefts` clears invalid query lanes before full-mask accumulation. Runtime 0/1
tail/odd routing values are computed by the scalar caller and passed into the VF instead of being derived from
`dSize`/`tSize` inside `__simd_vf__`. Optional tail masks are materialized once through parameter-bounded 0/1 setup
loops and reused by the later optional loops. Mandatory 1VL/second-VREG compute masks remain direct initializations
from a valid 1-to-64 element count.

Compute is split into two ordered VF calls. Phase 1 consumes `partial`, `delta`, and `effective_query`, updates
`partial`, and stores one score per T index. While phase 1 runs, MTE2 loads `inter_numerator`, `inter_max`, and
`inter_exp_sum` into disjoint UB regions. Once phase 1 finishes, MTE3 can write updated `partial` back while phase 2
uses it read-only to update the online-softmax state and form `h`. Separate phase event IDs allow these transfers to
remain in flight without serializing the two handoffs for a buffer.

For `D <= 128` and `tileT >= 8`, phase 2 first vectorizes the online-softmax coefficient update across T, with one
FP32 VREG lane per T index. The normalized alpha and beta overwrite the consumed
`inter_max` and `inter_exp_sum` regions in `stats`; after one local vector-store-to-vector-load barrier, they are
broadcast at each T index while forming `h`. Smaller tiles and larger `D` retain the `TOKEN` phase-2 path.

## Kernel test

The TTK TestSpec and kernel cases are operator-owned assets under `tests/assets/` and `tests/st/arch35/`.
Output 0 aliases input 0 through `output_inplace_indexes=(0,)`, matching the direct kernel ABI.
TTK already supports reference outputs and maps this output slot back to input 0, so no operator-specific framework
patch is required. After building and installing this operator into the active CANN OPP environment, run TTK with
absolute paths to the operator-owned assets:

```shell
TTK_DIR=/path/to/ops-test-kit
OPS_TRANSFORMER_DIR=/path/to/ops-transformer
cd "${TTK_DIR}"
python3 -m ttk kernel \
  -i "${OPS_TRANSFORMER_DIR}/attention/block_attn_res_update/tests/st/arch35/ttk_kernel_block_attn_res_update_st.csv" \
  --plugin "${OPS_TRANSFORMER_DIR}/attention/block_attn_res_update/tests/assets" \
  --compare close \
  --plat Ascend950
```

The plugin selects `close` for both outputs. The CSV supplies `rtol=0.001`, `ptol=0.001`, and
`atol=0.1` for both the FP32 in-place output and BF16 `h`.
