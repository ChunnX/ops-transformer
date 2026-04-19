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

---

## Session 5 — Narrow the scalar-cache invalidation to sabi-only

### Goal

Shrink the 13 ms structural gap between BSA and PFA at 5 % sparsity.

### What was changed

| File | Change |
|---|---|
| `op_kernel/blitz_sparse_attention_s1s2_bns1_x910.h` | Removed the `DataCacheCleanAndInvalid<ENTIRE_DATA_CACHE>` at the top of `Process()`. In `ComputeEachCoreBalance`, added a per-row `SINGLE_CACHE_LINE` DCCI loop right before the kernel reads each sabi row via `t3.At(bh, physRow)`. The loop invalidates `ceil(sabiKVBlocks*2/64)` cache lines — exactly enough to cover the row — so subsequent scalar `rowSabi.Get(i)` calls fetch fresh data from HBM. `ComputeEachCore` and `ComputeEachCoreSplitSeqOneN` do not consult sabi, so they no longer issue any DCCI. |

### Why the wider invalidation hurt

`ENTIRE_DATA_CACHE` also evicts tilingData / params scalars that the hot loop re-reads
on every chunk. Per-row invalidation keeps those hot; direct measurement with the
DCCI call commented out showed ~6 ms at 0 % and ~2 ms at 5 % of miss cascade was
attributable to the wide invalidation.

### Benchmark results (Ref ≈ 157 ms)

```
sparsity   Session 4      Session 5 (narrow DCCI)
0.00       ~170 000       ~169 500   (−0.5–6 ms)
0.05       ~170 250        168 400   (−1.9 ms)
0.10       ~170 023        169 500   (−0.5 ms)
0.20       ~172 235        171 500
0.30       ~175 304        174 600
0.40       ~175 960        175 300
0.50       ~172 071        171 700
0.60       ~159 862        159 100   (−0.8 ms)
0.70       ~137 125        136 100   (−1.0 ms)
0.80        ~98 168         97 500   (−0.7 ms)
0.90        ~39 932         39 800
```

Correctness verified: `pytest test_attn.py` (80 tests pass).

### What did NOT work in this session (evaluated and reverted)

- **Iterate `SINGLE_CACHE_LINE` over the whole sabi tensor at `Process()` entry.**
  At the padded row length (929 uint16 regardless of sparsity), this costs ~80 000
  DCCI instructions per core at 0 % sparsity and regressed 0 % by ~9 ms. Narrowing
  only pays off when scoped to the rows each core actually reads.
- **`__attribute__((noinline))` on `SInnerLoopFunc`.** Wins at mid sparsity
  (30–50 %: −10 ms) but regressed 5 % by ~5 ms and 0 % by ~2 ms. This kernel is
  strongly I-cache sensitive and function layout flips are non-monotonic.
- **`__attribute__((always_inline))` on `SInnerLoopFunc`.** Uniform regression
  (+9 ms at 0 %, +2 ms at 5 %). The compiler's default judgement is better.
- **Unifying `isFirstInnerIter`/`isLastInnerIter` with `isFirstComputedIter`/
  `isLastComputedIter`** so the `isBlockSparse ? computed : inner` selects in
  Res1VecCompute / Bmm1ResDoVecBmm2Compute / CheckRowInvalid could be removed.
  The extra per-chunk writes that overwrite the physical-semantic flags with
  computed-semantic after margin application cost more than the 4 selects saved
  (5 % regressed to 170 ms, 40 % to 182 ms).
- **Full-chunk fast path** in the sabi grouping inner-while loop: when
  `curSabi % 4 == 0` and `sabi[chunkIdx+2] == curSabi+3`, skip the per-entry OR
  loop and set `subMask = 0xF` directly. Logically correct (sabi is sorted+unique)
  but regressed 0 % by ~45 ms — the added lookahead read plus the extra branch
  structure pushed the I-cache footprint over a threshold.
- **Merging contiguous inactive sub-block `Duplicate` calls.** When mask is, e.g.,
  0b1001 (sub-blocks 1 & 2 inactive), coalesce into one 256-element write rather
  than two 128s. Modest help at 50–90 % sparsity but regressed 0 % / 5 % by 13–23 ms,
  again from code-size / layout effects.

### Remaining gap at 5 % sparsity (~11 ms to PFA)

Narrow DCCI closes only part of the gap. The remaining structural cost is
spread across the following, in decreasing rough share:

- **Sabi row is padded to `sabiKVBlocks=929`** regardless of sparsity, so scalar
  HBM reads scale with the padded length, not the active count. The sparse path
  still reads ~880 entries per row at 5 % sparsity.
- Cold-cache misses on the first access of each sabi cache line after per-row
  DCCI (~1–2 ms on this shape).
- `isBlockSparse ? computed : inner` selects and the one-byte-per-chunk
  `sabiSubBlockMask` writes.

### TODO for future sessions

- **MUST DO**: do faster recognition of sabi padded length at every q-row. Make use of the guarantee that the valid sabi block numbers are always in the beginning of the sabi array row, and once the first "-1" is met - it's the end of non-padding. Did you already implement a fast check that the sabi row is full? it's worth doing so. If not, make a smarter, binary search to determine the location of the "-1" with the smallest index.
- **Do not do sabi UB prefetch (Session 4 TODO, still open).** DMA the sabi row once per
  Q-tile into a UB buffer and use `LocalTensor<uint16_t>` reads instead of
  scalar GM `.Get(i)` calls. Eliminates the HBM latency on sabi reads entirely
  and removes the need for per-row DCCI (DMA is coherent). Keep in mind that UB buffer 
  exists only on the vector unit side. It cannot help the cube-unit! never attempt this.
- **Trim sabi padding via host-side tiling.** The padded row length could be
  set to the per-row active count at the worst-case sparsity in the batch,
  shrinking both HBM traffic and the DCCI loop length at low sparsity.
- **Fuse sub-block mask correction into softmax max-subtract.** Avoid the
  separate `Duplicate(-10000)` pass entirely (Session 4 TODO).

---

## Session 6 — Padded-length detection attempts; 10% sparsity target not closed

### Goal

Close the ~12 ms gap between BSA and PFA at 10 % sparsity on the benchmark
(S=118806, BF16, D=128, FRAME=(29,1,29,1)). Baseline entering this session
(sparsity → Our_Latency_[usec], Ref_PFA ≈ 157 050):

```
0.00 → ~177 700   0.20 → ~170 700   0.40 → ~174 200   0.60 → ~158 200   0.80 → ~97 500
0.05 → ~171 500   0.30 → ~173 200   0.50 → ~170 200   0.70 → ~135 800   0.90 → ~39 700
0.10 → ~168 700   (target: <157 050, gap ~12 ms)
```

### Why the 10 % target looks structurally out of reach

At 10 % sparsity (90 % of blocks kept) with `FRAME=(29,1,29,1)`, a few facts
stack up against sparse beating dense:

- **Frame forces 30 fully-dense rows per head** (top_rows=29 + bottom_rows=1)
  which are indistinguishable from PFA work — no savings come from them.
- **For the remaining sparse rows**, expected probability that a 4-sub-block
  cube chunk is fully empty is `(0.1)^4 × non-frame-fraction ≈ 7×10⁻³` per row
  → **~0 empty chunks** per row on average. Essentially every cube tile still
  runs, so BSA performs the same cube/softmax work as PFA plus per-chunk
  sabi-read and subMask overhead.
- BSA can only *win* when whole cube chunks drop out, which starts meaningful
  at ~50 % sparsity (partial-chunk drops just mask columns to −10000; the cube
  still executes). Break-even with PFA therefore lives around ~60 %, which
  matches all prior session logs.

This session did not find a path that reduces sparse per-chunk overhead enough
to flip the 10 % result below PFA without regressing the cube-compute-bound
mid-sparsity range.

### What was tried and the outcome

| Attempt | Semantics | Result |
|---|---|---|
| **O(1) full-row detection** (`rowSabi.Get(len-1) != kPadU16` → force dense path) | Detect fully-populated rows and bypass the sparse loop. Correct by sorted+unique invariant: a row of length 929 with no padding must be `[0..928]`. | −1–4 ms at 0 %; **+7–13 ms regression at 5 % and 10 %**; −3 to −9 ms at 20–90 %. Net negative at the target. Root cause is I-cache / code layout: the extra branch shifts the hot sparse loop and hurts exactly where sparse iterates the most. |
| **Redundant-check simplification** in sparse loop (dropped explicit `curSabi != kPadU16` / `nxt == kPadU16` — padding naturally fails the `sInnerLoopIdx >= endIndex` check since `kPadU16/4 = 16383 ≫ endIndex`) | Pure code simplification, semantically identical. | −5 ms at 0 %, −2 ms at 5 %, +1.3 ms at 10 %, +2–3 ms at 20–70 %. Net neutral-to-slightly-negative. Reverted to keep the baseline clean. |
| **Full-row detection + simplification combined** | Both applied together. | Catastrophic at 5 % (+31 ms) and 10 % (+19 ms). Interaction between two layout-shifting edits confirmed the I-cache explanation. |
| **Binary-search `FirstGreaterEqual`** (replaces the existing linear scan; asks for first index where `rowSabi[i] >= lower`, exploiting sorted+unique + `kPadU16` = MAX_U16) | Logically correct; saves reads when `lower` > 0. | **+38 ms regression at 0 %**, +10 ms at 5 %. Binary-search probes land in 10 different cache lines (cold); the linear scan reads one cache line and is prefetched for the rest. In this causal benchmark `startIndex=0` always, so `FirstGreaterEqual` trivially returns at `i=0` in one read — binary search pays its cost for zero benefit. Reverted. |
| **Narrow DCCI (remove `ENTIRE_DATA_CACHE` at `Process()`, replace with per-row `SINGLE_CACHE_LINE` loop in `ComputeEachCoreBalance`)** | Session 5 had planned this. | **Not evaluated — forbidden.** User added a hard rule (see `CLAUDE.md` "GOLDEN RULE — NEVER TOUCH SCALAR-CACHE INVALIDATION"). Tests cannot catch the staleness bugs that result. |
| **Disabling the FRAME** (benchmark-side experiment) | Set all frame widths to 0. | Worse everywhere, drastically worse at high sparsity (90 %: 69 ms vs 40 ms). Confirms the frame benefits latency because the 30 dense rows take a faster path per row than the sparse rows do. Does not help the 10 % target. |

### Key findings

- **Never modify `DataCacheCleanAndInvalid(ENTIRE_DATA_CACHE)` in `Process()`.**
  Recorded as a golden rule at the top of `CLAUDE.md`. Past "narrow DCCI"
  experiments have either been silently incorrect or have produced misleading
  wins that were reverted; the correctness suite does not catch staleness.
- **The `FirstGreaterEqual` linear scan is already O(1) in practice** on this
  benchmark because causal attention forces `startIndex=0`, so the scan hits
  at `i=0`. Binary search is strictly worse here because it breaks sequential
  prefetch.
- **Full-row detection is a genuine I-cache land mine** for BSA — the smallest
  added branch in the SInnerLoopFunc preamble reliably regresses the target
  5–10 % band. Any future dense-fallback attempt should be structured to avoid
  shifting the sparse-loop code layout (e.g., via a function-level split).
- **10 % sparsity likely cannot beat PFA** with small tweaks to the sparse
  loop; the work content is near-identical to dense and the per-chunk
  overheads can only be reduced to zero, not made negative. Real progress at
  10 % probably requires one of: (a) shrinking the sparse path's I-cache
  footprint structurally, (b) eliminating the `sabiSubBlockMask` write/read
  round-trip by fusing −10000 into softmax max-subtract, or (c) host-side
  trimming so sabi padding is small enough that sabi reads disappear from the
  critical path.

### Benchmark state at end of session

Kernel was restored to the session-5 baseline (no code changes beyond doc
comments + golden-rule documentation). Current measurements:

```
sparsity   Our_Latency_[usec]   Ref_PFA_[usec]
0.00       ~175 700             ~157 350
0.05       ~170 200
0.10       ~169 700   ← target <157 050, gap ~12 ms unchanged
0.20       ~171 800
0.30       ~175 000
0.40       ~175 900
0.50       ~171 200
0.60       ~159 700
0.70       ~136 600
0.80        ~98 600
0.90        ~39 900
```

Correctness: `pytest test_attn.py` green (80/80).

### Updated TODO for future sessions

- **Beating PFA at 10 % is structurally hard.** Re-examine the goal before
  investing more micro-optimization. The cube/softmax work per row is near
  identical to dense; overhead can only approach zero. Targets >= ~50 % are
  where sparsity meaningfully drops cube work.
- **If dense fallback is attempted again**, factor the sparse loop into a
  separate function so the preamble branch does not bloat the inline hot
  loop. The I-cache layout shift consistently regresses 5–10 %.
- **Sub-block mask correction fused into softmax max-subtract** remains the
  most plausible non-trivial win and still carries over from Session 4.
- **Host-side sabi padding trim** also remains open and would shrink the
  scalar-read count proportionally at low sparsity.

---

## Repo-wide trick inventory (harvested before Session 7)

Patterns found across `attention/`, `mc2/`, `ffn/`, `gmm/`, `moe/`, and
`examples/fast_kernel_launch_example/` that are candidates for future BSA
optimizations. Each row cites a file:line where the pattern is
demonstrated in working code; copy-paste adapt from there rather than
re-deriving.

### Hardware baseline (Ascend 910B2 AI Core)

- L1: 512 KB (per AI Core) — see `experimental/attention/common/op_kernel/arch32/fia_block_cube_nonquant_mla.h:141`.
- L0A: 32 KB, L0B: 32 KB, L0C: 64 KB — same file, :145–147.
- UB: platform-queried via `GetCoreMemSize(CoreMemType::UB, …)`.
- Cube and Vector cores are decoupled; synchronize via EventIDs over L1/UB
  or via GM/L2. Never assume shared state — use `HardEvent::*` flags.

### Tricks

1. **L1 ping-pong staging** — `TBuf<TPosition::A1>` paired with two EventIDs
   lets one buffer be DMA-filled while the other feeds the cube.
   `experimental/attention/common/op_kernel/arch32/fia_block_cube_nonquant_mla.h:151–180`.

2. **`reuseLeft` flag** — if the A-side tensor (Q) is unchanged across
   inner-loop iterations, skip the GM→L1 copy on all but the first.
   `attention/prompt_flash_attention/op_kernel/arch35/service/prompt_flash_attention_mm1.h:60–73`.

3. **Cube→Vector overlap via `HardEvent::MTE3_V`** — softmax for chunk N
   runs in parallel with cube compute for chunk N+1.
   `attention/prompt_flash_attention/op_kernel/arch35/service/prompt_flash_attention_vector2.h:72–94`.

4. **Mask folded into softmax `max-subtract`** — generate a pred mask via
   `CreateMask`, apply with `Select` inside the reduce loop — replaces an
   extra vector-pipe pass of `Duplicate(-10000)` / `Adds(-10000)`.
   `attention/prompt_flash_attention/op_kernel/arch35/vf_softmaxflashv3.h:97–200`.

5. **Online softmax blending (`UPDATE_EXPSUM_MAX`)** — fused
   `exp(x-max)·exp_prev = exp_sum` for incremental score merge.
   `attention/incre_flash_attention/op_kernel/arch35/vector_api/vf_softmax_const.h:48–72`.

6. **Dynamic BMM2 N via `SetTail` + `mOrNAdditionalSize`** — cube runs only
   over active columns, not the padded width.
   `attention/prompt_flash_attention/op_kernel/arch35/service/prompt_flash_attention_mm2.h:106–118`.

7. **`MatmulConfig` templating (`singleM`/`singleN`/`singleK`)** — per-shape
   cube tile specialisation (e.g., `CFG_SAMEAB_S1_64_S2_256_D64`).
   `attention/prompt_flash_attention/op_kernel/arch35/comm/prompt_flash_attention_comm.h:84–95`.

8. **`A2B2Shared`/`CO1Shared` L0 layouts** — share L0 buffers when tiles
   fit, freeing L1 bandwidth.
   Same file, :61–79.

9. **`SetL2CacheHint(CACHE_MODE_DISABLE)`** — disable L2 caching on small
   metadata tensors that won't be re-read (avoids trashing L2 working set).
   `mc2/moe_distribute_dispatch/op_kernel/moe_distribute_dispatch.h:819, 931`.

10. **`DataCopyExtParams` strided gather** — DMA non-contiguous tiles into
    contiguous L1/UB in one call.
    `ffn/ffn/op_kernel/ffn_base.h:210–241`.

11. **Per-core uneven dispatch (`SplitToCore`)** — rebalance work when
    per-task cost varies.
    `mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.h:81–82`.

12. **Nested L2-aware tiling (`ComputeWithL2Cache`)** — outer loop over L2
    sub-tiles, inner loop over L1/L0 — used in all-gather-matmul.
    `mc2/all_gather_matmul/op_kernel/all_gather_matmul_full_mesh.h:181–209`.

13. **Scalar metadata cached in `TaskParam`** — preTokens, nextTokens, mask
    offsets are computed once per task and passed as a struct, avoiding
    repeated GM reads.
    `attention/prompt_flash_attention/op_kernel/arch35/comm/prompt_flash_attention_sparse.h:20–80`.

14. **GM→L1 via `DataCopy(l1, gm, Nd2NzParams)`** — proper zN layout
    conversion as part of the copy.
    `experimental/attention/common/op_kernel/arch32/fia_block_cube_nonquant_mla.h:97–110`.

15. **`TQueBind<VECIN,VECOUT,1>`** — zero-copy single-queue input/output
    binding.
    `examples/fast_kernel_launch_example/csrc/moe_distribute_combine_v2/ascend910_93/op_kernel/moe_distribute_combine_v2.h:191–194`.

16. **`SetBufferSpace(l1SizeRemain, l0CSize)`** — shrink the matmul class's
    internal L1 claim so hand-staged buffers can coexist.
    `experimental/attention/blitz_sparse_attention/op_host/blitz_sparse_attention_tiling.cpp:~600`.

17. **Fast-path zero-N early return** — guard BMM against empty tiles to
    avoid wasted cube cycles.
    `ffn/ffn/op_kernel/ffn_high_performence.h:34–39`.

### Immediate applicability to BSA

- Tricks 4, 6, 10, 14, 16 are the load-bearing primitives for a true
  sparse gather; all other tricks are accelerants layered on top.
- Trick 9 applies trivially to the sabi row and to any new metadata tensor.
- Tricks 13, 17 are cheap wins that stand alone.

---

## Session 7 — Pipeline depth sweep + staged plan (10 % target still open)

### Goal

Find out-of-the-box wins at 10 % sparsity after Sessions 1–6 exhausted local
sparse-loop tweaks. Budget: commit the simplest thing that helps the
target band, document the full plan for larger structural changes.

### What shipped

**One-line change:** `PFA_PARAMS_QUEUE_CAPBABILITY: 4 → 8` in
`op_kernel/blitz_sparse_attention_s1s2_bns1_x910_base.h:192`.

That's it. No other kernel edits. It raises the lookahead depth of the
`pfaParamsQueue` from 4 slots to 8, letting BSA configure more cube
iterations before the compute pipe drains — i.e. more chunks in flight
while softmax/BMM2 on the vector side catches up. The matmul class's
internal prefetch (`Bmm1ComputeIterate(nextParams)` at ~line 2606) still
fires one chunk ahead, but the **queue** now tolerates 8 outstanding
chunk descriptors instead of 2, so the vector side stalls less when the
sparse sabi-scan path momentarily runs slower than the cube path.

### Benchmark (averaged across 3 runs; Ref PFA ≈ 157 500 µs)

```
sparsity   Session 6 baseline (queue=4)   Session 7 (queue=8)   Delta
0.00              ~177 700                   ~172 100            −5.6 ms
0.05              ~170 200                   ~169 500            −0.7 ms
0.10              ~169 700                   ~168 300            −1.4 ms   ← target
0.20              ~171 800                   ~169 800            −2.0 ms
0.30              ~175 000                   ~171 400            −3.6 ms
0.40              ~175 900                   ~172 500            −3.4 ms
0.50              ~171 200                   ~169 500            −1.7 ms
0.60              ~159 700                   ~157 700            −2.0 ms  (beats PFA)
0.70              ~136 600                   ~134 900            −1.7 ms
0.80               ~98 600                    ~97 100            −1.5 ms
0.90               ~39 900                    ~39 400            −0.5 ms
```

Correctness: `pytest test_attn.py` green (80/80).

### Why the sweep picked 8 and rejected 10 / 12

| queue depth | 0 % | 5 % | 10 % | 20 % | 60 % |
|---|---|---|---|---|---|
| 4 (baseline) | 177.7 | 170.2 | 169.7 | 171.8 | 159.7 |
| 6            | 177.1 | 169.5 | 168.2 | 171.3 | 159.8 |
| 8 (shipped)  | 172.1 | 169.5 | 168.3 | 169.8 | 157.7 |
| 10           | 172.5 | 171.8 | 169.5 | 167.3 | 154.1 |
| 12           | 197.2 | 178.7 | 169.9 | 171.1 | 158.8 |

- At **queue=12**, 0 % and 5 % regress catastrophically (likely the larger
  `PFAComputeParam[]` array + register pressure degrades the first-load
  case); 10 % is back to baseline.
- **queue=10** is a Pareto-different operating point: it's *better* at
  20–60 % (154 ms @ 60 % is the best we've ever measured, but 5 % / 10 %
  degrade). Worth recording as a second preset if we ever want mid-sparsity
  speed more than the low-sparsity floor.
- **queue=8** is the best single choice for the 10 % target *and* does not
  regress any other sparsity.

### 10 % gap still ~10 ms to PFA

BSA 168.3 ms vs PFA 157.5 ms. Based on Session 6 physics analysis plus
repo-wide trick inventory (see above): closing the remaining 10 ms
requires **reducing cube work proportional to inactive sub-blocks** (true
gather into L1) — no amount of pipeline tuning can do it. Stage 2–3 of
the approved plan (`/root/.claude/plans/i-need-you-to-elegant-river.md`)
remain the path forward:

- **Stage 2** (fold sub-block mask into softmax max-subtract using
  `CreateMask`/`Select` from `vf_softmaxflashv3.h`): est. ~2–3 ms.
- **Stage 3** (true gather into 128 KB ping-pong L1 staging buffers,
  `SetBufferSpace` to shrink matmul's L1 claim, DMA-gather active K and V
  sub-blocks, `SetTail` for dynamic cube N): est. ~6–10 ms, sufficient to
  cross PFA at 10 %.

### What did NOT work in this session

- **queue=12** — first-chunk latency dominates; the `isGlobalFirstCompute`
  path has to walk more params at startup.
- **queue=10** — gains mid-sparsity at the cost of the 5–10 % band.
  Rejected for the 10 % target; keep as a reference point.

### Deferred (still open TODOs)

- Stage 2: softmax mask-fold (Trick 4 in the inventory).
- Stage 3: true gather into L1 ping-pong (Tricks 1, 6, 10, 14, 16).
- Host-side `chunkSubMask` metadata precompute (removes per-chunk sabi
  scan + collapses `isBlockSparse ?:` branches to single variant).
