# BlockAttnResUpdate

`BlockAttnResUpdate` is an Ascend 950-only, AIV/RegBase implementation of the BlockAttnRes phase-2 update.
It has six tensor inputs, two tensor outputs, one framework workspace slot, and one tiling argument. The input and
output named `partial_block` form an ACLNNGraph reference tensor and share one logical allocation. The kernel ABI
still carries both `partial_block_ref` and `workspace` address slots. Host tiling requests zero workspace bytes, and
the kernel does not dereference either address. The implementation relies on the reference alias and writes the
updated partial result back through the `partial_block` input address.

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

`epsilon` is an optional float attribute. When it is omitted, host tiling uses `1e-6`; the selected value is
serialized into `BlockAttnResUpdateTilingData` together with `invCols = 1 / D` and the `uint8_t phase2Mode` selected by
host tiling. The kernel never reads an epsilon tensor. The RMS path intentionally uses FP32 `sqrt` followed by direct
division and does not include Newton refinement or special handling for zero/extreme values.

Host tiling requires concrete positive `T` and `D`, with `D <= 8192`. It rejects zero/unknown/negative dimensions,
incorrect ranks, inconsistent related shapes, unsupported input/output dtypes, shapes whose `T * D` element count
cannot be represented by the kernel GM index type, and shapes whose per-core row count cannot be represented by
`uint32_t`.

## Tiling

Rows are split across vector cores; one complete D row always remains on one core because RMS and dot-product
reductions are performed inside that core. The kernel always uses two UB buffers. `tileRows` is capped at
`ceil(maxRowsPerCore / 2)` so the busiest core uses both ping-pong buffers whenever it owns at least two rows; tiling
fails if one complete row cannot fit with double buffering.

Every UB row starts at a 32-byte boundary. Since the element sizes differ, tiling carries both aligned strides:

```text
colsAlignFp32 = AlignUp(D, 8)    // 8 FP32 = 32 B
colsAlignBf16 = AlignUp(D, 16)   // 16 BF16 = 32 B
phase2Mode = D <= 128 && tileRows >= 8 ? ROW_BATCH : ROW_LOCAL
scalarAlign = phase2Mode == ROW_BATCH ? AlignUp(tileRows, 64) : AlignUp(tileRows, 8)
```

The effective reduction length is always `D`; aligned elements are padding only.

## UB layout

```text
0
├─ effective_query : colsAlignFp32 * 4
├─ buffer 0
│  ├─ partial / p  : tileRows * colsAlignFp32 * 4
│  ├─ delta / h    : tileRows * colsAlignBf16 * 2  (reused after phase 1)
│  ├─ numerator    : tileRows * colsAlignFp32 * 4
│  └─ stats        : 3 * scalarAlign * 4         (interMax/interExpSum/score)
└─ buffer 1 (same layout)
```

UB regions use fixed byte offsets. GM/UB views are materialized with Tensor API `MakeMemPtr`/`MakeTensor`, and
staging uses `CopyGM2UB`/`CopyUB2GM` with `NDExtLayoutPtn` row pitches. A full-register VF load may fetch aligned
tail data beyond the effective column count; invalid lanes are cleared or masked before use, and host tiling leaves
8 KB of UB unassigned as physical headroom. Synchronization uses
`SetFlag`/`WaitFlag` for `MTE2_V`, `V_MTE3`, and `MTE3_MTE2`; no queue, pipe, or dynamic UB allocator is used.
The VF body uses the public `Reg::LoadAlign`/`Reg::StoreAlign` interfaces for
UB/register movement and a tail `MaskReg` for the effective `cols` elements.
Phase 1 handles full registers with an explicit two-way unroll and independent FP32 square/dot accumulators. The
one- and two-register paths keep the complete query resident across rows. The generic path also preloads its first
query pair plus optional odd/tail registers, peels the first pair to initialize the accumulators without per-row
zeroing, and streams only middle query pairs. `Reg::MulAddDst` shortens the accumulation chains without adding UB
scratch or a local-memory barrier. For the only partial register, masked `Add` clears invalid `partial` lanes and a
one-time zero-shift `Reg::ShiftLefts` clears invalid query lanes before all-row full-mask accumulation.

Compute is split into two ordered VF calls. Phase 1 consumes `partial`, `delta`, and `effective_query`, updates
`partial`, and stores one score per row. While phase 1 runs, MTE2 loads `inter_numerator`, `inter_max`, and
`inter_exp_sum` into disjoint UB regions. Once phase 1 finishes, MTE3 can write updated `partial` back while phase 2
uses it read-only to update the online-softmax state and form `h`. Separate phase event IDs allow these transfers to
remain in flight without serializing the two handoffs for a buffer.

For `D <= 128` and at least eight rows in the current tile, phase 2 first vectorizes the online-softmax coefficient
update across tokens, with one FP32 VREG lane per token. The normalized alpha and beta overwrite the consumed
`inter_max` and `inter_exp_sum` regions in `stats`; after one local vector-store-to-vector-load barrier, they are
broadcast per row while forming `h`. Smaller tiles and larger `D` retain the row-local phase-2 path.
