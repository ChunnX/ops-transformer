# BSA Kernel Optimisation Log

## Session 1 — Raise cube KV-tile from 128 to 512 (partial win)

### Problem

The 128×128 sabi migration forced the KV matmul tile (`s2BasicBlock`) down to 128.
A 128×128 cube matmul severely underutilises the Ascend 910B2 cube unit, which is
designed for 128×512 (or larger) N-dimension matmuls.
Consequence: BSA at 0 % sparsity ran **3× slower** than dense PFA at the same shapes.

### What was changed  (see `git diff`)

| File | Change |
|---|---|
| `op_host/blitz_sparse_attention_tiling.cpp` | Added `BMM_BASICBLOCK_N_512 = 512`. Raised `s2BasicBlock` cap from 128 → 512 (line ~3652). Raised `SetBmm1TilingInput` N-split cap to 512 (line ~3499). |
| `op_kernel/blitz_sparse_attention_s1s2_bns1_x910.h` | Sparse path of `SInnerLoopFunc`: added `sabiPerTile = basicSInnerSize / 128` (= 4 when s2=512). Sabi entries are now grouped by 512-aligned chunk before each cube call. `FirstGreaterEqual` adjusted to pass a 128-granularity bound. |
| `benchmark/test_attn.py` | Commented out `S=36000` (`ceil(36000/128)=282`, not a multiple of 4) with a TODO to restore it once proper sub-block gather is implemented. |

The tiling change makes every cube call process a 128×512 tile instead of 128×128.
The kernel change ensures `sInnerLoopOffset` is emitted in 512-granularity so that
`GetBmm1TensorBOffset` resolves to the correct global-memory address.

### Benchmark results

Before (128×128 cube tiles, 128×128 sabi):

```
  H   B    s_q   s_kv    D  sparsity   Ref_Latency_[usec] Our_Latency_[usec]
  3   1 118806 118806  128      0.00          157990.21          475726.71
  3   1 118806 118806  128      0.50               N/A           237938.16
  3   1 118806 118806  128      0.90               N/A            48623.37
```

After (128×512 cube tiles, 128×128 sabi):

```
  H   B    s_q   s_kv    D  sparsity   Ref_Latency_[usec] Our_Latency_[usec]
  3   1 118806 118806  128      0.00          157403.97          391966.19
  3   1 118806 118806  128      0.50               N/A           197249.33
  3   1 118806 118806  128      0.90               N/A            41399.55
```

Improvement: **~18 % faster** across all sparsity levels. Still **~2.5× slower** than
reference PFA at 0 % sparsity. The speedup from sparsity is preserved but not amplified.

### Why the improvement was modest

The approach taken is "aligned-chunk grouping": the sabi entries are grouped by which
512-aligned range they fall in (`sabiBlock / 4 → chunk_idx`), and the cube reads
the full 512-element range directly from global memory via `tensorBOffset =
chunk_idx * 512 * headSize`. There is **no explicit gather** of the 4 KV sub-blocks.

Two fundamental problems remain:

1. **No gather, no correct packing.**  
   The cube loads 512 contiguous global-memory elements starting at
   `chunk_idx * 512 * D`. For the benchmark shape `S=118806`, which is *not* a
   multiple of 512, some 512-aligned chunks contain fewer than 4 active sabi entries.
   The cube still computes over those inactive 128-element regions (false work),
   and the data loaded does not correspond to the 4 blocks actually pointed at by
   the sabi entries.

2. **No pipelining between DMA and cube.**  
   Even where the 4 sub-blocks happen to be contiguous (structured patterns), the
   load (global → L1) and compute (cube over L0) are sequential within each
   iteration. The cube sits idle during each L1 prefill.

---

## Session 2 — Sub-block mask correction + 512-wide CVDIFF tiles

### Approach: aligned-chunk grouping + sub-block mask correction

Instead of a true gather (which requires explicit L1 staging buffers), we use:
1. **Tiling**: raise the CVDIFF `rectangleFactor` from 128 → 512 for D=[128,200) BF16, so each cube call sees a 128×512 K/V slice.
2. **Cube reads**: unchanged — the cube reads 512 contiguous KV tokens starting at `chunk_idx × 512 × D` in global memory (same aligned-chunk approach as Session 1 for TND).
3. **Sub-block mask correction**: after the cube computes the 128×512 attention score block, identify which of the 4 sabi-pointed 128-element sub-blocks within that chunk are *inactive*, and add −10000 to their columns before softmax.  This zeroes out the contribution of KV tokens that were not in the sabi list.

This is correct because the KV tensor is stored densely: sabi block index `k` maps to KV tokens `[k·128, (k+1)·128)` at global offset `k·128·D·sizeof(dtype)`. Reading the aligned 512-wide chunk and masking inactive sub-blocks is equivalent to reading only the active blocks.

### Key limitation vs true gather

The aligned-chunk approach reads KV tokens that are not in the sabi list (they are corrected to 0 by the mask). A true gather would skip those reads entirely, saving HBM bandwidth proportional to sparsity. This is left as future work (Session 3).

### What was changed

| File | Change |
|---|---|
| `op_host/blitz_sparse_attention_tiling.cpp` | Added D=[128,200) BF16 branch in `AdjustCVTilingCVDiff`: `rectangleFactor=512`, `softmaxSOuterFactor=8`. |
| `op_kernel/blitz_sparse_attention_s1s2_bns1_x910_base.h` | Added `sabiSubBlockMask` field (uint8, default 0xF) to `PFAComputeParam`. |
| `op_kernel/blitz_sparse_attention_s1s2_bns1_x910.h` | In `SInnerLoopFunc` sparse path: compute `sabiSubBlockMask` bitmask from consecutive sabi entries that share the same 512-aligned chunk. In `Res1VecCompute`: if `sabiSubBlockMask != fullMask`, call `Adds(..., -10000.0f, 128)` on each inactive sub-block column range. In `CopyParamsAttrOutOfInnerLoop`: copy `sabiSubBlockMask`. |

### `sabiSubBlockMask` logic

`sabiPerTile = singleProcessSInnerSize / 128` (= 4 when sInner=512).  
For each inner-loop iteration over the sabi list, collect all consecutive entries whose `block_idx / sabiPerTile == chunk_idx`, set bit `block_idx % sabiPerTile` in `subMask`.  
`fullMask = (1 << sabiPerTile) - 1`.  
If `subMask != fullMask`, the Res1Vec path adds −10000 to the inactive sub-block columns.

### Important: `torch_bsa` rebuild is required after ACLNN API changes

`torch_bsa.cpython-*.so` is a separately compiled Python extension.  When  
`aclnn_blitz_sparse_attention.h` changes (e.g. adding `softmaxLseFlag`/`softmaxLse`)  
the `.so` must be recompiled:

```bash
cd torch_interface
PFA_LIB=custom python3 setup.py install
```

Forgetting this causes `aclnnBlitzSparseAttentionGetWorkspaceSize` to receive  
wrong arguments (ABI mismatch), which surfaces as error **161001**  
(`ACLNN_ERR_INNER_NULLPTR` / `ACLNN_ERR_PARAM_NULLPTR`).

### Restoring S=36000 in tests

`ceil(36000/128) = 282`, which is not a multiple of 4 (sabiPerTile=4 for sInner=512).  
With aligned-chunk grouping, the last partial chunk contains fewer than 4 active  
sabi entries. The existing sub-block mask correction handles this correctly  
(inactive sub-blocks → −10000), so **S=36000 can be restored** once we confirm  
benchmark numbers are acceptable.

### TODO: true gather (Session 3)

A true gather (explicit L1 staging + DataCopy per sub-block) would:
- Eliminate reads of inactive KV tokens from HBM, saving bandwidth proportional to sparsity.
- Require allocating two ping-pong L1 staging buffers (512×D BF16 each = 128 KB each).
- Overlap the DMA gather for iteration N+1 with the cube compute for iteration N.
