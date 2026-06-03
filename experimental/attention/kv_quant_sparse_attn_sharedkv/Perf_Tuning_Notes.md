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
| a05 load-last reorder | 131.463 | -4.7us / -3.4%, accuracy PASS |
| b06 phy-addr UB staging | 131.296 | neutral (within noise), reverted |
| g01 s2BaseSize=64 retile | 182.138 | +50.7us / +38% REGRESSION, accuracy PASS, reverted |
| **h01 v0 lead-by-1 (deeper pipe)** | **109.067** | **-22.4us / -17.0% vs a05, accuracy PASS (kept, new floor)** |
| h02 v0 lead-by-2 | 110.170 | +1.1us vs h01 (neutral/slightly worse), accuracy PASS, reverted |
| g02 s2BaseSize=64 + h01 lead-by-1 | 142.951 | +33.9us / +31% vs h01 REGRESSION, accuracy PASS, reverted |
| g03 s2BaseSize=256 retile | — | builds, but **NPU ERROR at runtime** (L0C/tile capacity), reverted |

Accuracy stays PASS (99.99% vs 99.50% threshold) for all *runnable* builds. **Best kept: h01, 109.067us.**
**Task-3 verdict (block-size retiling, multiple attempts g01/g02/g03): s2BaseSize=128 is optimal — both smaller (64) and larger (256) regress or fail.**

> **Important:** h01 (below) overturns the earlier "a05 is the producer-bound floor"
> conclusion that the middle of this document argues at length. Those sections
> (`Where the floor is now`, `Why a05 is the practical floor`, the `b06`/`c07`/`g01`
> analyses, and the `Empirical split` section) are **kept for the record but
> superseded**: the cube is NOT hard producer-bound, because the per-iteration
> flag-15 barrier is **AIV-group only** — the cube does not participate in it and
> can therefore run a full round ahead once the producer leads. See "## h01" at the
> end for the corrected model.

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

## Refined floor model

Re-reading the main loop and `ProcessVec0`: the hard `CrossCoreSetFlag/WaitFlag(15)`
rendezvous fires **every steady s2 iteration**, inside `ProcessVec0` (block_vector.h:674-675),
right after the dequant store and before `v0ResGm.SetCrossCore()` publishes to the cube. It
is a **group barrier across all AIV cores** — idle cores (`hasLoad==0`) spin it exactly
`maxS2LoopCnt` times (kernel.h:336-341) so the count matches. This couples the whole AIV
group per iteration and is required for SPLIT_G KV-sharing correctness; the fixed handshake
count means **it cannot be split or relaxed without desyncing the idle cores and
deadlocking.** (The drain/epilogue flag-15 sites at kernel.h:414/461 are the tail-flush of
the same barrier.)

Steady-state coupling is therefore the per-iteration AIV-group barrier plus the pipelined
producer/consumer buffer syncs (`v0ResGm`, `bmm1/bmm2`, `l1P`, depth-3/DB). The **cube idles
on `v0ResGm.WaitCrossCore()`** waiting for the AIV dequant producer. Consequences:

- **Cube utilization is bounded by the AIV per-iter time.** Any cube-only change (deeper L0C,
  bmm2 DB) cannot raise `aic_mac_ratio` while AIV stays ~132us — the cube just idles on the
  producer. Both perf goals require **reducing the AIV path**, nothing else.
- AIV per-iter pipes: `scalar` 78.9 / `mte2` 73.8 / `vec` 60.8 / `mte3` 34.6, core 132us.
  Busiest pipe is only 78.9us, so ~53us is intra-AIV serialization + the per-iter group
  barrier. Recovering it via deeper buffering is blocked (UB full, see c07); via scalar
  staging is neutral (b06); via splitting the barrier is unsafe (deadlock).

## c07 (FAILED): bmm2Buffers single -> double-buffer

Tried promoting `bmm2Buffers` (UB) from `BuffersPolicySingleBuffer` to `BuffersPolicyDB`
(mirroring `bmm1Buffers`), to decouple the cube's bmm2 fixpipe-write from the AIV
`ProcessVec2` consumer (cube stalls on `outputBuf.WaitCrossCore()`, block_cube.h:341).
Required growing the UB arena from `mm1ResultSize*2 + mm2ResultSize` (48KB) to
`mm1ResultSize*2 + mm2ResultSize*2` (80KB, +32KB for the 2nd 32KB mm2 buffer).

**Result: NPU execution error, accuracy FAIL — reverted.** Root cause is **UB capacity**,
not addressing (`ProcessVec2` reads via the buffer's own `GetTensor`; DB call sites are
symmetric with bmm1 so ping/pong parity stays locked). No static UB check, so it compiled
but at runtime the +32KB pushed total UB past physical capacity and the bmm2 fixpipe write
aliased a live vector-core buffer. **Takeaway: UB is full** — any UB-growing change (deeper
bmm2, deeper V0 ping-pong, larger staging scratch) is dead on arrival here.

## Redesign exploration: what's blocked vs what could break the floor

Implementable rearrangements under the UB+L1 limits are exhausted:
| Lever | Status |
|---|---|
| Larger `s2BaseSize` (fewer barriers) | L1-capped: `l1Right` 3×128KB + `l1P` 32KB + `l1Q` 96KB = 512KB full. 256-tile needs 2×256KB ring, breaks depth-3. |
| Deeper pipeline (preload>2) | L1-capped: needs 4th `l1Right` (128KB), no room. |
| Deeper UB buffering (bmm2 DB / V0 ping-pong) | UB-capped: c07 proved +32KB overflows. |
| Scalar phy-addr staging | Neutral (b06). |
| Cube-only (L0C depth, bmm2 DB) | Cannot help: cube is producer-bound, idles on AIV. |
| Split/relax flag-15 barrier | Unsafe: fixed group-handshake count, deadlocks idle cores. |

**Genuine improvement requires an algorithm-level redesign (changes numerics/scope, needs
explicit authorization):**
1. **Reduce KV dequanted per tile.** `aiv_mte2` 73.8us + `aiv_vec` 60.8us scale with KV
   volume. Skipping fully-masked tokens, or a tighter sparse-block selection, cuts both
   producer poles directly. Medium win; touches selection numerics.
2. **Tiling that frees L1** (e.g. stage D in halves so `l1Right` is 64KB, enabling a 4-deep
   ring) to deepen the pipeline past the producer barrier. Medium win; large retiling +
   accuracy re-validation.

> **Ruled out — dequant-on-cube / fuse dequant into the matmul K-load.** Moving the
> fp8->bf16 dequant onto the cube (so it consumes fp8 KV in-line and the `v0ResGm` GM
> round-trip + flag-15 barrier disappear) is **not possible on this hardware**: the cube
> matmul K-load path does not support in-line dequantization. The dequant must stay on the
> AIV, so the GM round-trip and the producer-bound coupling are structural and cannot be
> removed by relocating the dequant.

**Standing conclusion:** a05 (131.5us) is the floor reachable by tiling/pipeline/sync
rearrangement under the UB+L1 capacity limits — both UB and L1 are full, the cube is
producer-bound on AIV dequant, the per-iter AIV-group barrier is fixed, and the AIV is at
its pipe-serialization floor with no room to deepen buffers. Breaking past requires one of
the algorithm-level redesigns above.

## Empirical split: pure-V vs pure-C (validates the producer-bound model)

To measure each core type's standalone path, we stubbed one side down to **sync-only**
(all compute removed, but every cross-core handshake + flag-15 group barrier kept) and ran
the other side for real, via `/home/w00927905/scfa_perf/perf_only.sh` (5 msprof runs, no
accuracy gate — the stub builds are numerically wrong by design).

- **pure-V** (cube stubbed, AIV real): `block_cube.h` IterateLoadQK/Bmm1/Bmm2 reduced to
  their `WaitCrossCore/SetCrossCore` pairs.
- **pure-C** (vector stubbed, AIC real): `block_vector.h` ProcessVec0/Vec1/Vec2 reduced to
  their `v0ResGm`/`bmm1ResBuf`/`outputBuf`/`bmm2ResBuf` handshakes + the flag-15 barrier.

| build | avg (us) | busiest pipes |
|---|---|---|
| pure-V (AIV path alone, a05)   | **88.481** | aiv scalar .88, mte2 .74, vec .71 (aic all 0) |
| pure-C (AIC path alone, a05)   | **93.815** | aic mac .78, mte1 .65, mte2 .62, fix .28 (aiv ~0) |
| **a05 combined (current)** | **131.463** | balanced AIC≈AIV≈135us |
| pure-V (AIV path alone, OLD)   | **89.835** | aiv scalar .88, mte2 .73, vec .69 (aic all 0) |
| pure-C (AIC path alone, OLD)   | **92.060** | aic mac .81, mte1 .66, mte2 .59, fix .29 (aiv ~0) |
| baseline (OLD combined)        | 136.151 | — |

**OLD vs a05 standalone paths are identical within noise** (V ~89-90, C ~92-94). This is the
clean proof that load-last (a05) changes *only issue order*, not either side's work. The win
is entirely in overlap:
- OLD: overlap hidden `89.8+92.1-136.2 = 45.7us`; serial coupling `136.2-92.1 = 44.1us`.
- a05: overlap hidden `88.5+93.8-131.5 = 50.8us`; serial coupling `131.5-93.8 = 37.7us`.
- a05 recovered `44.1-37.7 = 6.4us` of the coupling bubble (= the -4.7us combined win, within
  noise). Both impls are producer-bound: cube's no-wait floor (~92us) is hit only when the
  vector relays instantly; the real AIV producer (~89us) imposes the rest.

### What it proves

1. **No single dominant core.** Standalone the two paths are nearly equal (88.5 vs 93.8us);
   pure-C is the slightly longer one. Neither alone explains the 131.5us combined time.
2. **The bottleneck is CV coupling, not either path's length.** Combined (131.5) far exceeds
   `max(88.5, 93.8)=93.8`. If the pipeline overlapped perfectly the op would run ~93.8us.
   - Overlap actually achieved: `88.5+93.8 - 131.5 = 50.8us` hidden → only **~57%** of the
     shorter path overlaps; **~43% stays serial.**
   - The non-overlapped **~37.7us** (`131.5 - 93.8`) is exactly the cube idling on
     `v0ResGm.WaitCrossCore()` waiting for the AIV dequant producer, plus the per-s2
     flag-15 group barrier and bmm1/bmm2 buffer handshakes that re-align the two cores
     every iteration. This is the 63us AIC bubble seen in the combined per-pipe data,
     net of what a05's load-last reorder already recovered.
3. **Quantitatively confirms producer-bound.** pure-C = 93.8us is the cube's *no-wait* floor
   (the AIV stub relays instantly). The combined run pushes the cube to 131.5 — the extra
   37.7us is precisely the wait on the real AIV producer (which itself needs 88.5us). The
   handshake granularity (one barrier per s2 iteration) plus shallow pipelining (L1/UB full,
   can't deepen — see c07 and the floor model) is what prevents that wait from overlapping.

### Implication for next steps

Lowering combined time toward `max`≈93.8us (or below) requires shrinking the AIV producer
path. The GM round-trip that forces the barrier **cannot** be removed by moving dequant onto
the cube (hardware does not support in-line dequant on the matmul K-load — see "Ruled out"
above), so the only remaining lever is reducing the dequant work itself: skip fully-masked
tokens / tighter sparse-block selection to cut `aiv_mte2`/`aiv_vec`. Pure pipeline/sync
rearrangement cannot close the 37.7us coupling gap because the buffers needed to deepen
overlap don't fit.

## g01 (FAILED): s2BaseSize=64 retile — empirically tested, big regression

Per explicit request, retiled the s2 base block 128 -> 64 to (a) halve `l1Right` (128KB->64KB),
freeing L1 for a deeper preload ring, and (b) finer-grained CV overlap. This required changing
`s2BaseSize` in **four** places that must agree or the flag-15 barrier count desyncs:
`block_cube.h`/`block_vector.h` constexpr, `kernel.h` `constInfo.s2BaseSize`, host
`S2_BASE_SIZE` (workspace), **and the sibling metadata AICPU op**
(`kv_quant_sparse_attn_sharedkv_metadata/.../*_aicpu.cpp:161` `s2BaseSize_=128U->64U`) — that op
computes per-core `FA_S2_MAX_NUM`, which drives the idle-core flag-15 spin count; if it stayed
128-based while the kernel iterated 64-based the group barrier would deadlock. The build harness
rebuilds both ops, so this was feasible.

**Result: accuracy PASS (99.665%), but perf 182.138us — a +50.7us / +38% REGRESSION. Reverted.**

Why it regresses (per-pipe, all ratios collapsed):
- s2=128 a05: `aic_mac` 0.78, `aiv_scalar` 0.88 (cores busy).
- s2=64 g01:  `aic_mac` 0.48, `aiv_scalar` 0.56, every pipe ~0.4-0.56 (cores ~50% IDLE).

Halving the tile **doubled the s2 iteration count**, which doubled every per-iteration cost:
the flag-15 AIV-group barrier, all cross-core handshakes (`v0ResGm`/bmm1/bmm2), and the scalar
phy-addr gather — while **halving matmul efficiency** (N=64 is a poor MAC tile). The producer
work (total KV dequanted) is unchanged, so finer tiling adds pure coupling/overhead to a loop
that is already producer/coupling-bound. This is the decisive empirical confirmation of the
producer-bound model: **the cost is per-iteration coupling × #iterations, so fewer/larger tiles
help and smaller tiles hurt.** A deeper preload ring cannot recover this — the +51us is doubled
fixed barrier overhead + structural N=64 MAC inefficiency, neither of which is buffer-depth-bound.

Corollary (block-size sweep is pinned): larger s2 (256) would cut iterations/barriers but is
**L1-infeasible** (`l1Right` 2×256KB + `l1Q` 96KB > 512KB even at a 2-deep ring). So s2=128 is
the optimum: smaller regresses (measured), larger doesn't fit (capacity). a05 stays the floor.


## h01 (WIN, kept): v0 dequant leads LoadQK by one round (deeper pipeline)

**Result: 109.067us avg (median 108.947, stdev 0.59%), accuracy PASS 99.9948%.
-22.4us / -17.0% vs a05. This is the new floor and is the kept HEAD change.**

This is Task 2 ("preload the v0 dequant one more round so the current loop's LoadQK
depends on v0 dequant results from a *previous* round"). It is the single biggest win
so far and **disproves the producer-bound floor model** argued above.

### The change (kernel.h `ProcessMainLoop`)

The pipeline was 3-deep with `Vec0` (AIV dequant producer) and `IterateLoadQK` (AIC
KV-load consumer) in the **same** iteration: LoadQK(tile T) waited on
`v0ResGm.WaitCrossCore()` for the dequant that Vec0(tile T) had just produced *this*
iteration. h01 shifts the **entire cube consumer chain plus the AIV Vec1/Vec2 back by
exactly one global iteration**, leaving Vec0 in place:

| stage | a05 tile | h01 tile |
|---|---|---|
| Vec0 (AIV, producer) | `taskId`   | `taskId`   (unchanged) |
| LoadQK (AIC)         | `taskId`   | `taskId-1` |
| Bmm1 / Vec1          | `taskId-1` | `taskId-2` |
| Bmm2 / Vec2          | `taskId-2` | `taskId-3` |

So LoadQK(T) now reads the v0ResGm tile Vec0 produced in the **previous** iteration —
the dequant is already done, the cube does not stall on it.

Mechanics (all in `op_kernel/arch35/kv_quant_sparse_attn_sharedkv_scfa_kernel.h`):
- `runInfo[3]` -> `runInfo[4]` (4 tiles now live: T, T-1, T-2, T-3), indexed `taskId%4`.
  The per-tile buffer-slot index `taskIdMod3` is **unchanged** (still `%3`) — l1Right and
  v0ResGm rings stay 3-deep and still exactly fit their live sets (l1Right {T-1,T-2,T-3}
  = 3 distinct mod-3; v0ResGm {T, T-1} = 2 tiles in a 3-deep ring).
- Consumer gates shifted +1 taskId: Bmm1/Vec1 `taskId>0`->`taskId>1`, Bmm2/Vec2
  `taskId>1`->`taskId>2`, LoadQK `notLastTwoLoop`->`taskId>0 && !loadDrainOff`.
- `PRELOAD_NUM` 2 -> 3 and a new third drain step (`loadDrainOff`) so the deepest tile
  flushes Vec0->LoadQK->Bmm1->Bmm2. The `lastBN` switch now stages three shut-offs:
  case 0 stops Vec0, case 1 stops LoadQK, case 2 stops Bmm1/Vec1.
- **No L1/UB growth, no metadata/tiling change.** v0ResGm is in GM; the only extra
  storage is one more `RunInfo` struct (host stack). This is why h01 fits where the
  earlier "deepen the pipeline" levers did not — it deepens *scheduling* depth, not
  buffer depth.

### Why the flag-15 accounting stays safe (no deadlock)

The flag-15 group barrier fires once per `ProcessVec0`. The total fire count on an active
core is `#real-Vec0 + (else-branch + epilogue fires while maxS2LoopCnt>0)`, which always
sums to the initial `maxS2LoopCnt` (= idle-core spin count from metadata) **regardless of
PRELOAD_NUM**, because the drain fires are guarded by `maxS2LoopCnt>0` and just shift one
fire from the epilogue to the in-loop else-branch. `#real-Vec0` and the metadata are both
unchanged, so the count is identical to a05. Accuracy PASS confirms it.

### Why it works — the corrected model (flag-15 is AIV-only)

The earlier model claimed the cube was hard producer-bound and that the per-iteration
flag-15 barrier capped any gain. The first half is right (the cube idles on the dequant),
**but the cap reasoning was wrong: flag-15 is a barrier across the AIV *group only* — the
AIC/cube does not participate in it.** So once the dequant is produced a round ahead, the
cube is free to run a full iteration ahead of the producer and overlap its matmuls with a
much larger fraction of the AIV producer path. It is not pinned to the producer per
iteration.

Per-pipe effect (h01 vs a05):
- `aic_mac_ratio` 0.53 -> **0.69** (cube far less idle), `aic_mte1` 0.41 -> 0.54,
  `aic_mte2` 0.45 -> 0.60.
- AIV pipes essentially unchanged (`aiv_scalar` ~0.73, `aiv_mte2` ~0.69, `aiv_vec` ~0.56)
  — h01 changes only *issue/overlap*, not the producer's work, just like a05.
- Combined 131.5 -> 109.1us, now much closer to the standalone pure-C floor (~93.8us, see
  Empirical split). The serial coupling gap `combined - pure-C` shrank from ~37.7us to
  ~15us: most of the cube's old wait on the producer is now hidden.

### Updated floor

h01 (109.067us) is the new floor for this case/tiling. The remaining ~15us gap to the
~93.8us pure-C standalone is the residual producer coupling: the cube still cannot start a
tile's Bmm1 until that tile's dequant lands, and leading by exactly one round hides one
round of it. Leading by **two** rounds would need v0ResGm and the runInfo ring deeper
(GM + host stack, both free) and another drain step — a plausible next attempt (h02) if
the residual gap is worth chasing; risk is the same flag-15 drain accounting, which h01
shows is self-balancing. The AIV producer path (~88-93us) remains the ultimate floor
unless dequant work itself is reduced (algorithm-level, see redesign section).

## h02 (tested, neutral/slightly worse, reverted): v0 dequant leads by TWO rounds

**Result: 110.170us avg (median 110.275, stdev 0.45%), accuracy PASS 99.9948%.
+1.1us vs h01 — the lead-by-1 gain does NOT compound. Reverted to h01.**

Extended h01 to lead Vec0 by two rounds: cube consumer chain shifted two global stages
(LoadQK tile T-2, Bmm1 T-3, Bmm2 T-4), `runInfo[4]->[5]`, `PRELOAD_NUM` 3->4, a 4-step
drain (two `loadDrainOff`/`notLast` stages). v0ResGm stayed 3-deep (live set {T,T-1,T-2}
= exactly 3 tiles; the cross-core flags keep it correct, verified by the PASS).

Per-pipe: `aic_mac_ratio` 0.69 (h01) -> **0.67** (h02) — the cube did *not* get busier
from the deeper lead; it got marginally less so. Why lead-by-2 doesn't help:

- h01 already hides the one round of producer wait that scheduling *can* hide. The
  residual ~15us gap to the pure-C floor is **not** another round of stall — it is the
  irreducible per-tile dependency: a tile's Bmm1 cannot start until that tile's dequant
  (AIV) + GM->L1 load (cube) have completed, and the AIV producer path (~88-93us) is the
  real floor. Leading the producer further ahead cannot make the producer faster.
- Lead-by-2 adds one more drain iteration of fixed overhead and pushes v0ResGm to a
  zero-slack 3-deep ring (live set exactly fills it), slightly *reducing* overlap.

So **lead-by-1 (h01) is the optimum** for this depth lever. Pushing past 109us needs
reducing the AIV dequant work itself (algorithm-level), not deeper scheduling. A 4-deep
v0ResGm at lead-2 (GM + host-workspace change) was judged low-EV and not pursued: the
bottleneck is the producer path length, not v0ResGm ring slack.

## g02 (tested, regression, reverted): s2BaseSize=64 + h01 lead-by-1 pipeline

**Result: 142.951us avg (median 142.835, stdev 0.13%), accuracy PASS 99.665%.
+33.9us / +31% vs h01 — still a clear regression, reverted to s2=128.**

Re-test of the s2BaseSize=64 retile (g01) but now on top of h01's deeper pipeline, to
answer whether the lead-by-1 overlap rescues the smaller block. It does *partially*
(g01 182us -> g02 143us, the overlap recovers ~39us of idle), but s2=64 stays far above
h01's 109us floor. Same 5 coordinated edit sites as g01 (block_cube.h, block_vector.h,
kernel.h, tiling.cpp, metadata_aicpu.cpp), no deadlock — the flag-15 self-balancing
accounting held with the doubled `maxS2LoopCnt`.

Per-pipe (g02 vs h01): `aic_mac` 0.637 vs 0.69, every pipe ~0.45-0.64. The regression is
**structural and unaffected by scheduling**:
- s2=64 halves the matmul N-tile, so MAC efficiency drops (`aic_mac` 0.64 vs 0.69) — the
  cube does less useful work per L0C pass.
- Iteration count (and the AIV-group flag-15 barrier count) **doubles**: twice the
  per-tile fixed overhead (sync, address calc, softmax setup) for the same total work.
- h01's overlap can hide producer *latency*, but cannot remove the doubled barrier/setup
  *count* or restore MAC width.

**Conclusion: s2=64 is worse regardless of pipeline depth. The win direction is not a
smaller block.**

## g03 (tested, infeasible at runtime, reverted): s2BaseSize=256 retile

**Result: BUILDS cleanly (104s), but accuracy run fails with NPU ERROR (fulfill=0).
Reverted to s2=128.**

The high-risk/high-reward opposite bet to g01/g02: a *larger* S2 block (256) to halve the
iteration/barrier count and widen the matmul N-tile for better MAC efficiency. Same 5
coordinated sites set to 256. It compiles (the vf softmax templates and MatmulK1 accept
the constant), but the kernel faults at runtime — the 256-wide tile exceeds an on-chip
capacity/tile constraint (L0C accumulator and/or the `EQ_128_SCFA` vf basic-block
specializations are sized for N<=128; v0ResGm / l1Right L1 budget also doubles). Making
s2=256 actually run would require re-sizing L0C/L1 buffers and adding N=256 vf
specializations — a vector-core rewrite, out of scope for a tiling-constant sweep.

**Conclusion: s2=256 is not reachable by the block-size constant alone.** Combined with
g01/g02, this brackets the optimum: s2BaseSize=128 (h01) is the best runnable block size
for this case — smaller regresses, larger does not run. **Task 3 (retiling / basic block
size) is complete with a well-supported negative result; h01 remains the kept floor.**
