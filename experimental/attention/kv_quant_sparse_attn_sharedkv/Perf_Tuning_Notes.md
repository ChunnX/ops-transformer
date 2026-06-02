# KvQuantSparseAttnSharedkv (SCFA) Performance Tuning Notes

Scope: only `experimental/attention/kv_quant_sparse_attn_sharedkv` is modified. Kernel used
comes from this folder. `mixed_quant_sparse_flash_mla` is not touched.

Benchmark case (`my_test`, the only ENABLED param set):
`B=48, S1=2, S2=8192, N1=128, N2=1, D=512, K=1024, cmp_ratio=4, ori_win_left=127,
tile_size=64, rope_head_dim=64`, SCFA template, fp8_e4m3.

Measurement: 5 msprof runs, report the average of `Task Duration(us)` for
`Op Name == KvQuantSparseAttnSharedkv` (harness: `/home/w00927905/scfa_perf/tune_scfa.sh <tag>`).

## Result

| Build | avg (us) | note |
|-------|----------|------|
| baseline            | 136.151 | pre-tuning |
| **a05 load-last reorder** | **131.463** | **-4.7us / -3.4%, accuracy PASS (kept)** |
| b06 phy-addr UB staging | 131.296 | neutral (within noise), reverted |

Accuracy stays PASS (99.99% vs 99.50% threshold) throughout. **Best kept: a05, 131.463us.**

## The winning change: AIC load-last reorder (a05)

File: `op_kernel/arch35/kv_quant_sparse_attn_sharedkv_scfa_kernel.h`, `ProcessMainLoop`
inner `s2LoopCount` loop.

### Why it works — read the metric correctly

The misleading signal is the `cube_utilization(%)` CSV field (~98%): it counts stall
cycles as "busy". The real cube utilization is `aic_mac_time / aicore_time`, which at
baseline was ~50%. So the cube is **feed/stall-bound, not compute-bound** — it spends
half its time stalled, not computing.

The stall source: this case runs the `IS_SPLIT_G` path (gSize=128 splits the M dimension
across 2 physical AIC cores that share dequantized KV only through GM). Each iteration the
AIC's `IterateLoadQK` blocks on `v0ResGm.WaitCrossCore()` — it waits for the AIV core to
finish dequantizing the next KV tile and publishing it to GM. With LoadQK issued first,
the cube sat idle on that wait before doing any matmul.

### The reorder

Split the per-iteration body by core and, on the AIC side, issue the matmuls **before**
the blocking load:

- AIV: run `ProcessVec0` (the dequant producer) first, then Vec1/Vec2.
- AIC: run `IterateBmm1` and `IterateBmm2` first, then `IterateLoadQK` **last**.

Because Bmm1/Bmm2 of the current task operate on tiles dequantized in earlier iterations,
they no longer depend on the tile LoadQK is about to wait for. The cube now overlaps its
matmul work with the AIV dequant of the next tile, instead of stalling first then computing.

### Effect (per-pipe, a05)

- `aic_mac_ratio` 0.50 -> 0.53, `aicore_time` ~142 -> 135.4us.
- The two physical AIC cores rebalanced; both land at ~135us.

## Where the floor is now (balanced regime)

After a05, the op is **balanced across the two core types**:

| | time (us) | top pipes |
|---|---|---|
| AIC (`aicore_time`) | 135.28 | mac 71.7 (.53), mte2 60.3 (.45), mte1 55.1 (.41), fixpipe 28.3 |
| AIV (`aiv_time`)    | 135.31 | scalar 78.9 (.58), mte2 73.8 (.55), vec 60.8 (.45), mte3 34.6 |

The two cores march in lockstep: a full cross-core barrier (`CrossCoreSetFlag/WaitFlag(15)`)
fires every s2 loop, so wall-clock per iteration is `max(AIC path, AIV path)`. Both are
~135us, so neither side has slack — reducing one alone tends to shift the bottleneck to the
other rather than reduce wall-clock.

Structural limits (analyzed, not pursued):
- The GM round-trip for dequant KV is required: gSize=128 forces M-split across 2 AIC that
  share KV only via GM. The non-split path is invalid here (M too large for one L0C).
- L0C buffering is already effectively double-buffered between Bmm1/Bmm2; the in-call
  `SetFlag/WaitFlag<M_FIX>` is a true mac->fixpipe data dependency, so extra L0C buffers
  give little.
- `aic_mac` 71.7us is the irreducible compute floor for this tiling.

## Candidate next lever (not yet landed)

AIV's largest pipe is `aiv_scalar` 78.9us, driven by `GetRealS2Addr` issuing up to 8
per-token **GM `GetValue`** scalar loads per inner step. Those indices are **contiguous in
GM**, and inside `ProcessSparseKv` the scalar gather serializes with the following
`CopyInKvSparse` MTE2 copy. Batching the gather into one up-front `DataCopy` (GM->UB),
double-buffered to overlap the previous tile, would trade slow GM scalar latency for one
MTE2 copy plus fast UB reads, shortening the AIV iteration and potentially moving the
barrier. Note: an earlier staging attempt (a04) was neutral, but that was measured while
AIC (142us) was the sole bottleneck; the balanced regime after a05 changes the calculus.
This is the recommended direction if pushing past 131us.

## Why a05 is the practical floor (verified by b06)

b06 implemented exactly the phy-addr UB staging proposed above (single contiguous
`DataCopyPad` of the precomputed phy-addr slice into a UB scratch + `MTE2->S` sync, with
`GetRealS2Addr` reading via fast UB `GetValue`). Accuracy PASS, but **perf was neutral
(131.296us, within run-to-run noise)** — so it was reverted.

The neutral result is informative: the scalar gather is **not** the gating serial path; it
already overlaps with the real long poles. Re-reading the per-pipe data, the loop is
**producer-bound on the AIV dequant**:

- Per iteration AIV runs Vec0 (the dequant producer) first, then the cube's `IterateLoadQK`
  blocks on `v0ResGm.WaitCrossCore()`. Vec0's cost is dominated by `aiv_mte2` 73.8us
  (loading quantized KV from GM) + `aiv_vec` 60.8us (fp8->bf16 dequant), **not** the scalar
  gather. The cube's matmuls (Bmm1+Bmm2 ~ `aic_mac` 71.7us total) are ~2x faster than the
  AIV producer, so the cube **idles waiting for dequant** — that is the 63us AIC bubble.
- Speeding a producer-bound loop requires speeding the producer (KV load + dequant), which
  is structural work, or running the consumer further ahead.

Running the consumer ahead is **blocked by L1 capacity**: `l1RightBuffers` is a 3-deep ring
of `s2BaseSize*512*sizeof(Q_T)` = 128KB each = **384KB**, plus `l1Q` 96KB = **480KB of the
512KB L1 already in use**. A 4th buffer (needed to deepen the pipeline past the per-s2-loop
barrier) does not fit. So the barrier-depth restructure is both infeasible (L1-bound) and
unhelpful (producer-bound).

**Conclusion:** for this case/tiling, a05 (131.463us) is at the practical floor. Further
gains would require algorithm-level reduction of AIV dequant work (less KV loaded/dequanted
per tile) or a tiling change that frees L1 — both outside incremental kernel tuning.
