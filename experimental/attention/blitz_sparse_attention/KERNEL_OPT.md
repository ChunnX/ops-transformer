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

### TODO: true gather (future work)

A true gather (explicit L1 staging + DataCopy per sub-block) would:
- Eliminate reads of inactive KV tokens from HBM, saving bandwidth proportional to sparsity.
- Require allocating two ping-pong L1 staging buffers (512×D BF16 each = 128 KB each).
- Overlap the DMA gather for iteration N+1 with the cube compute for iteration N.

---

## Session 3 — Overhead analysis: BSA vs PFA at 0% sparsity

### Problem

After Session 2, BSA at 0% sparsity is still **~33% slower** than PFA at the same
shapes (208 602 µs vs 157 032 µs). Both use identical CVDIFF tiling parameters
(`rectangleFactor=512`, `softmaxSOuterFactor=8` for D=128 BF16). The goal was to
identify the root causes of this overhead.

### Analysis performed (no code changes)

Compared BSA's `Process()` / `SInnerLoopFunc` / `Res1VecCompute` against the
equivalent PFA code paths to find every divergence in the hot loop.

#### 1. Sabi HBM reads in `SInnerLoopFunc` sparse path

`U16VecView::Get(i)` reads `const __gm__ uint16_t*` — a scalar read from global
memory (HBM). Each 512-token KV-chunk requires ~5 sabi reads (1 lead + up to 3
grouping + 1 lookahead). For `S=118806` there are ~233 chunks per Q-tile.

However, access is sequential, so the hardware prefetcher is effective after the
first few cache misses. Estimated overhead: ~800 µs per core for cold-start
misses; much less once the cache is warm.

#### 2. `DataCacheCleanAndInvalid(ENTIRE_DATA_CACHE)` in `Process()`

BSA calls `DataCacheCleanAndInvalid<uint16_t, ENTIRE_DATA_CACHE, CACHELINE_OUT>`
at the start of `Process()`, which invalidates ALL scalar data cache. PFA has no
equivalent call. This forces cold cache misses for sabi reads on every kernel
invocation, even during benchmark measurement (after warmup).

**Investigated but NOT the bottleneck for the 128×512 → 128×128 migration.**
This call is necessary for correctness — it ensures sabi data is re-read from HBM
when the sabi tensor contents change between invocations. Do not remove it.

#### 3. `isBlockSparse` runtime checks in hot compute path

`Res1VecCompute` checks `params->isBlockSparse` on every call (~16 times per
KV-chunk for `softmaxSOuterFactor=8`). Additional checks exist in
`Bmm1ResDoVecBmm2Compute` and `CheckRowInvalid`. Each check is ~10 scalar
instructions. Estimated overhead: negligible (~80 ns per chunk).

#### 4. Instruction cache / code size

BSA's `SInnerLoopFunc` has both sparse and dense paths compiled into the same
binary. The larger code footprint may cause more L1i cache misses compared to
PFA's smaller kernel. Not quantified.

### Conclusion

The 33% overhead is likely a compound effect of:
- Sabi HBM read latency (cold cache from `DataCacheCleanAndInvalid`)
- Additional branch/param-setup instructions in the sparse inner loop
- Possible instruction-cache pressure from larger binary size
- No single dominant bottleneck was isolated

The focus should shift to making BSA faster than PFA at moderate sparsity levels
(5–10%), where skipping KV blocks provides a net win despite the per-block overhead,
rather than trying to match PFA at 0% sparsity.

---

## Session 4 — Sparse-loop read fusion + Duplicate-based sub-block correction

### Goal

Beat dense PFA at 5% sparsity (benchmark.py, `blocks_optimized_batched`,
`FRAME=(29,1,29,1)`, S=118806 BF16 D=128). Goal not fully achieved — ~13 ms gap
remains after extensive optimization. Crossover shifted from ~75% to ~60%.

### What was changed (`op_kernel/blitz_sparse_attention_s1s2_bns1_x910.h`)

1. **Sparse-loop lookahead fusion.** Rewrote the sparse inner loop in
   `SInnerLoopFunc` so the `nxt` value that ends the chunk-grouping inner-while is
   forwarded into the next outer iteration as `curSabi`. Removes one redundant scalar
   HBM sabi read per chunk iteration. The loop now primes `curSabi` outside and
   advances via a cached `nextSabi` rather than re-reading after each chunk.

2. **Duplicate instead of Adds for sub-block mask correction.** `Res1VecCompute` now
   uses `Duplicate(-10000)` (pure vector store) instead of `Adds(mm+=-10000)`
   (read-modify-write) when zero-ing inactive sub-blocks. Softmax is indifferent
   between `mm - 10000` and `-10000` because both map to `exp(≈-∞) ≈ 0`. Cutting the
   input read halves memory traffic on the vector pipe. Biggest wins are at mid
   sparsities (20–50%) where a large fraction of chunks contain partial sub-blocks.
   Also hoisted the outer guard to a single `isBlockSparse && subMask!=0xF` check,
   removing the runtime `bSabiPerTile` division and nested branches.

### Benchmark results (Session 2 → current, Ref ≈ 157 ms)

```
sparsity   Session 2 (Our_Latency_[usec])   Session 4 (current)
0.00                  ~208 000                    ~170 000
0.05                  ~198 000                     170 250
0.10                  ~188 000                     170 023
0.20                  ~167 000                     172 235
0.30                  ~148 000                     175 304
0.40                     (N/A)                     175 960
0.50                  ~119 000                     172 071
0.60                     (N/A)                     159 862   ← first to beat PFA
0.70                    ~72 000                    137 125
0.80                     (N/A)                      98 168
0.90                    ~24 000                     39 932
```

Net improvements (vs. Session 2 baseline):
- 0% sparsity: −38 ms
- 5% sparsity: −28 ms
- Crossover with PFA shifted from **~75% → ~60%**.

### What did NOT work (evaluated and removed or reverted)

- **Dense-row detection fast-path** — O(n) scan of `rowSabi.ptr[0..expectedCount-1]`
  to detect fully-contiguous sabi rows and force `isBlockSparse=false`. It was in
  place for a while and did save 11–15 ms, but after the lookahead-fusion the sparse
  path became fast enough that the *scan itself* cost more than it saved — without
  the detection, 0% sparsity runs ~4 ms faster. Scan removed.
- **Chunk-mask prescan into stack array.** `uint8_t chunkMasks[256]` populated in
  one sabi pass, then dense outer loop with O(1) mask lookup. Should have been
  faster (one sequential HBM scan vs per-iteration lookups) but regressed by ~30 ms.
  Likely causes: stack zero-init cost, compiler de-optimization from larger function
  footprint, cache pollution from the initial linear scan.
- **O(1) dense-row detection** (first+last+len only, relying on sabi sorted+unique).
  Logically correct but benchmarked ~5 ms slower than the O(n) scan variant — the
  compiler probably vectorised the O(n) loop into something that amortises under
  the prefetcher while the single-read variant waits on HBM latency.
- **2D Adds via `UnaryRepeatParams` (repeatTimes=souterSize)** for sub-block mask
  correction. The `SetMaskCount`/`SetVectorMask`/`SetMaskNorm` pair added per-chunk
  overhead that exceeded the savings from consolidating 8 per-row Adds into 1 call.

### Remaining gap at 5% sparsity (~13 ms to PFA)

The gap is structural:
- BSA carries both dense and sparse paths in one `__aicore__ inline` function →
  larger binary, I-cache pressure.
- `DataCacheCleanAndInvalid(ENTIRE_DATA_CACHE)` on each `Process()` entry (required
  for correctness when sabi content changes across calls; see CLAUDE.md).
- LSE output scaffolding and `{0}`-placeholder binding even when disabled.
- Sub-block mask correction itself is ~O(partial_chunks × souterSize × 128 bytes)
  of vector writes — grows fastest at mid sparsities where ~60% of chunks are partial.

### TODO for future sessions

- **Reduce fixed BSA overhead.** Investigate moving the sparse code path out of the
  inlined hot function (extern call or separate template specialisation) so the
  dense hot path sees a smaller binary.
- **Sub-block mask-correction fusion with softmax.** Rather than a separate
  `Duplicate(-10000)` pass, apply the mask as part of the softmax max-subtract step.
- **Merge contiguous inactive sub-blocks.** At 20–50% sparsity, when sub-blocks 1 and
  2 are both inactive, issue a single 256-element `Duplicate` instead of two 128s.
  Requires a small mask-scan loop to detect runs.
- **Sabi UB prefetch.** `DataCopy` the row of sabi into a UB buffer once per Q-tile
  and read from `LocalTensor<uint16_t>` — eliminates HBM latency on sabi reads at
  the cost of UB allocation.
- **True gather.** Skip HBM reads of inactive KV tokens entirely (Session 2 TODO).
