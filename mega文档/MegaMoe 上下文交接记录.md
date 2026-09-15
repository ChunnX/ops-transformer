# MegaMoe Context Handoff

## Purpose and Scope

This document preserves the working context for future agents working on MegaMoe. It is intentionally focused on the current `arch92` A8W8 Wave implementation, the earlier `arch35` refactor context that it builds on, and the synchronization and scheduling conclusions established while reading the code.

Primary repository:

```text
D:/gitcode/ops-transformer_910b2
```

Primary files for the current discussion:

```text
mc2/mega_moe/op_kernel/arch92/mega_moe_a8w8_wave.h
mc2/mega_moe/op_kernel/arch92/mega_moe.h
mc2/mega_moe/op_kernel/arch92/stage/mega_moe_token_dispatch.h
mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm1_activation.h
mc2/mega_moe/op_kernel/arch92/stage/mega_moe_gmm2_combine.h
mc2/mega_moe/op_kernel/arch35/blaze/gemm/block/block_scheduler_swizzle.h
```

This document describes the code as inspected on 2026-09-08. Do not infer that every historical migration experiment below is enabled in the current build without checking the current branch.

## Current Worktree

The branch and source-change observations below are the historical 2026-09-08 snapshot. The UB documentation reference was updated on 2026-09-12 after consolidation:

```text
branch: master...origin/master
source changes: none observed during the last inspection
current UB reference (consolidated 2026-09-12):
  mega_moe_arch92_a8w8_384k_ub_plan.md
```

The two earlier UB audit documents have been merged into [the final UB plan](/D:/gitcode/ops-transformer_910b2/mega_moe_arch92_a8w8_384k_ub_plan.md). Use that file as the single maintained reference for layout formulas, optimization candidates, and manual evaluation constraints. Consolidation changed documentation only, not kernel or tiling code.

## Terminology

### Physical block, logical job, and C/V pair

`BlockWorkspaceContext` identifies a physical block's workspace slot. It is used where physical isolation matters, such as per-block workspace and flag addresses.

`BlockJobContext` identifies a logical job participating in a stage:

```cpp
struct BlockJobContext {
    uint32_t jobIndex;
    uint32_t totalJobs;
};
```

Task partitioning should use `jobIndex`, not assume it is always equal to physical `blockIdx`. A mapping layer may remap physical blocks to a subset of logical jobs.

In arch92's 1C1V design, think of one physical block `j` as a cooperating Cube/Vector pair `Cj/Vj`. They use the same logical task mapping when they must process the same GMM tile, while workspace addressing can remain physical-block based.

### M group and GMM tile

`GMM1_TILE_M` is normally 256. An M group is a contiguous row group of at most that many expert tokens.

A logical GMM tile has both M and N extents. For a 256x256 GMM tile:

```text
one M group = up to 256 token rows
one N tile  = up to 256 output columns
```

For GMM1, one logical scheduler tile maps to a Gate/Up pair: it performs two MMAD calls, one for each half of the output. For GMM2, one logical scheduler tile maps to one MMAD call.

### Wave

A Wave is an upper-level grouping of one or more expert slices. It is chosen to target a fixed amount of GMM1 work across all C cores. It is not a hardware wavefront and it is not a Dispatch chunk.

`ExpertTokenPosition` records:

```text
expertIdx
tokenIndexInExpert
```

It describes where the next Wave begins in the concatenation of local experts. If a long expert crosses a Wave boundary, the next Wave starts at the recorded token index inside the same expert.

## Architecture Context

### A5 / arch35

The original A5 code supports A8W8, A8W4, A4W4, layered variants, shared experts, multiple combine modes, and several communication/synchronization paths. Earlier refactoring work extracted many small helper functions and arranged them by role, including AIV compute, AIV communication, block MMAD, prologue, and epilogue pieces.

Important design intent from that refactor:

```text
Atomic reusable helpers should not own cross-stage pipeline ordering.
The template-level kernel should own Dispatch/GMM/SwiGLU/GMM2/Combine ordering.
```

Dispatch remains an independent, expert-grained upstream operation. It can be scheduled before GMM but does not inherently require AIC execution.

### A6 / arch92

Arch92 is the A6 migration direction. Its key execution assumption is 1C1V, unlike A5's A8W4 arrangement with two AIV roles. The first migration target is A8W8 only.

The current `MegaMoeArch92A8W8Wave` template is explicitly a 1C1V A8W8 Wave design:

```text
one AIC: GMM1 and GMM2
one AIV: Dispatch, SwiGLU/activation quant, and Combine
```

There is no arch92 equivalent of A5 AIV0 continuously executing Blaze prologue while AIV1 performs Dispatch/SwiGLU/Combine. Do not carry A5 `GetSubBlockIdx()==0/1` reasoning into arch92 Generic paths.

Historical migration concerns that should be revalidated against the target CANN package:

```text
* A6 CANN originally lacked a directly usable MX FP8 BlockMmad specialization equivalent
  to the A5 `Blaze::Gemm::Block::BlockMmad<MatmulWithScaleMx, ...>` path.
* Arch92 therefore contains dedicated MegaMoe block-MMAD adaptation work.
* FP8 types such as fp8_e4m3fn_t/fp8_e5m2_t/fp8_e8m0_t are global C API types in the
  affected SDK, not `AscendC::` namespaced types.
* A CANN 9.1.0 internal/interface mismatch around `CopyL0C2UB` was seen in the shared
  expert GMM1 path. A temporary shared-expert bypass may have been used in a migration
  branch; never retain such a bypass in a feature-complete branch without an explicit decision.
```

The top-level `op_kernel/arch92/mega_moe_apt.cpp` placement and host-side SOC definitions were also part of the migration. The host registration path may need explicit `ascend910_96 -> arch92` wiring outside the MegaMoe folder; this should be checked in the actual compile repository.

## Top-Level Arch92 Process

`MegaMoeArch92A8W8Wave::Process()` is in `arch92/mega_moe_a8w8_wave.h`.

High-level order:

```text
1. AIV pre-processing
   QuantizeLocalTokens
   GatherAndSendExpertMasks
   ResetSyncStatus
   optional shared-expert input preparation

2. SyncAll

3. optional shared-expert GMM1

4. CrossRankSyncInWorldSize

5. ProcessMoeExpertStages
   This is the normal MoE Wave path described below.

6. AIV exports expert token counts, then SyncAll

7. optional shared-expert GMM2

8. AIV CrossRankSyncInWorldSize and UnpermuteTokens
```

The current discussion deliberately ignores shared experts unless explicitly stated otherwise.

## Wave Capacity and Initial Dispatch Prefetch

The Wave target is declared as:

```cpp
constexpr uint32_t ARCH92_GMM1_TILES_PER_JOB_TARGET = 2U;
```

It means the scheduler aims for approximately two **logical GMM1 tiles per C job per Wave**. It does not mean every C core executes exactly two calls, and it does not mean two Gate/Up MMAD operations.

The actual capacity computation in `ProcessMoeExpertStages()` is:

```cpp
schedulerN = gmm1OutputDim / ACTIVATION_N_HALF;
gmm1TilesPerMGroup = ceil(schedulerN / L1_TILE_N);
waveMGroupTarget = ceil(
    ARCH92_GMM1_TILES_PER_JOB_TARGET * totalJobs /
    gmm1TilesPerMGroup);
```

For a typical example:

```text
Gate width = Up width = 4096
L1_TILE_N = 256
total C jobs = 64

gmm1TilesPerMGroup = 4096 / 256 = 16
target logical GMM1 tiles = 2 * 64 = 128
waveMGroupTarget = 128 / 16 = 8 M groups
wave capacity = 8 * 256 = up to 2048 tokens
```

The correct interpretation is:

> One M group of at most 256 tokens needs 16 logical GMM1 tiles to cover its N direction. It is not that one token needs 16 cores.

### Initial Dispatch prefetch

Before the formal Wave loop, only AIV runs:

```cpp
while (IsPositionWithinWave(dispatchPosition, dispatchedWaveMGroupCount)) {
    if (dispatchPosition.tokenIndexInExpert == 0) {
        expertTokenCount = DispatchMoeExpert(dispatchPosition.expertIdx);
    } else {
        expertTokenCount = GetExpertTokenCountFromWorkspace(...);
    }
    AdvanceArch92ExpertTokenPositionInWave(...);
}
```

Consequences:

```text
* The first Wave's required experts are Dispatch-prefetched before entering the main loop.
* Dispatch itself remains whole-expert granularity. A large expert is dispatched once.
* If that expert crosses a Wave boundary, only `tokenIndexInExpert` advances. The next
  Wave consumes a later slice of the already-dispatched expert; it does not dispatch it again.
* AIC does not wait for the whole initial Wave to finish dispatching. It enters the main
  path and waits only for the M group needed by its own first GMM1 tile.
```

## Actual Wave Main Loop

The central condition is:

```cpp
while (currentWaveNeedsGmm1 || nextWaveNeedsDispatch) {
    // AIV: dispatch one next-Wave expert position when needed
    // AIC/AIV: process one current-Wave GMM1 expert slice
}
```

This is the most important pipeline relation:

```text
Dispatch(Wave w + 1)
        overlaps with
GMM1 + SwiGLU(Wave w)
```

It is incorrect to describe the inner loop as Dispatch/GMM1/SwiGLU all belonging to the same Wave.

For every iteration:

```text
AIV, if nextWaveNeedsDispatch:
  Dispatch one full expert if first encountered.
  Otherwise only load its recorded token count.
  Advance dispatchPosition by as much of that expert as fits in Wave w+1.

All participating cores, if currentWaveNeedsGmm1:
  Determine one expert slice of Wave w.
  Advance gmm1Position by the capacity remaining in Wave w.
  Run GMM1 plus its AIV activation epilogue for that slice.
```

When neither current GMM1 work nor next-Wave Dispatch remains, the code processes GMM2/Combine for every expert slice covered by the just-finished Wave, then begins the next Wave.

There is no global barrier saying all blocks must finish GMM1 before any block can start GMM2. Each block is governed by its own input-ready flags and local Tile ownership.

## Dispatch: Work Partition and GMM1 Readiness

The Wave calls `RunMoeExpertDispatchJob(..., UseRankSharding=true, ...)`.

For a current expert, dispatch first obtains and publishes its received token count. If nonzero, it partitions work as follows:

```text
rankShardCount = countWorkspace.blockNum / worldSize
dispatchShardIdx = jobIndex; dispatchShardIdx < worldSize * rankShardCount;
                   dispatchShardIdx += totalJobs

remoteRankIdx = dispatchShardIdx / rankShardCount
rankShardIdx  = dispatchShardIdx % rankShardCount
```

For each owned shard, the AIV copies the selected token range from that remote rank. Thus the practical partition is:

```text
remote rank
  -> a shard inside that rank's token segment
  -> assigned to logical Dispatch jobs
```

It is not simply one flat expert-token range divided over all cores.

After a shard's data is written, `PublishGmm1TileReady()` determines which 256-row GMM1 M groups the shard overlaps, then atomically adds the number of rows contributed to each group:

```cpp
AtomicAdd(dispatchToGmm1Flag[mGroup], overlapRowCount);
```

Before the first tile in a given M group, GMM1 waits for exact completion:

```cpp
targetValue = min(tileM, config.m - mLoc);
while (targetValue != ReadGmByPassDCache(dispatchToGmm1Flag[waveIdx])) {
}
```

This means a C core can start GMM1 for expert/M-group 0 as soon as all its rows are present. It does not wait for Dispatch of the full expert or Wave.

## Scheduler and startBlockIdx

### Scheduler

The common scheduler is:

```cpp
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerSwizzle<3, 1>;
```

Its job is only to decompose the current GMM problem into deterministic logical tiles:

```text
problem shape: [M, N, K]
tile shape: [tileM, tileN]
tileIdx -> (mLoc, nLoc, kLoc)
tileIdx -> actual tail shape
```

It does not know experts, Waves, physical blocks, flags, or execution progress.

With `BlockSchedulerSwizzle<3, 1>`:

```text
loopFirst  = ceil(N / tileN)
loopSecond = ceil(M / tileM)
tileNum    = loopFirst * loopSecond
```

It groups up to three adjacent N tiles and snakes the M direction between adjacent groups. Do not assume a linear `tileIdx` is row-major `(M,N)` order.

### startBlockIdx

`startBlockIdx` is a rotating logical-core offset. It specifies which logical C job receives scheduler `tile 0` for the next GMM call.

For logical C job `j` out of `B` participating jobs:

```cpp
startLoopIdx = (j - startBlockIdx + B) % B;

for (loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += B) {
    // this job owns loopIdx
}
```

Equivalent ownership rule:

```text
tile q belongs to job (startBlockIdx + q) % B
```

After every GMM call, including on jobs that owned no Tile:

```cpp
startBlockIdx = (startBlockIdx + tileNum) % blockNum;
```

The purpose is to avoid repeatedly assigning small-expert work to low-numbered cores.

Important correction: the current arch92 Wave code creates **one** local `startBlockIdx` at the start of `ProcessMoeExpertStages()` and passes it to both GMM1 and GMM2 calls:

```text
GMM1 expert slices
  -> GMM2 expert slices of the same Wave
  -> GMM1/GMM2 of later Waves
```

It is therefore a single continuous logical Tile-ownership rotation across GMM stages, experts, and Waves in this implementation. It is not two independent GMM1/GMM2 offsets.

This remains static deterministic assignment, not dynamic work stealing. A C core idle in GMM1 does not acquire GMM2 work merely because it is idle; its GMM2 ownership still follows the scheduler plus the current `startBlockIdx`.

`gmTileSequence` is separate. It is a per C/V producer-consumer sequence used for GM handoff flags. Do not confuse it with `loopIdx` or `startBlockIdx`.

## GMM1 Execution and Handoff

For one GMM1 logical Tile:

```text
1. AIC obtains (mLoc, nLoc, actualShape) from the scheduler.
2. It waits for the relevant Dispatch M group to be complete.
3. It executes two MMAD operations:
   * Gate: B offset nLoc
   * Up:   B offset nLoc + outputN
4. Both outputs are BF16 and are written to the GMM1 output area.
5. AIC publishes the GMM1-to-AIV completion state.
6. AIV waits for the corresponding completion sequence, reads the paired Gate/Up result,
   runs SwiGLU plus activation MX quantization, and publishes GMM2 input readiness.
```

For the arch92 Wave path, `UseGmHandoff=true` is passed into GMM1. The AIV Generic handoff epilogue polls:

```cpp
expectedReadySequence = gmTileSequence + 1;
while (ReadGmByPassDCache(gmmToEpilogueFlag) < expectedReadySequence) {
}
```

Then it consumes the corresponding GMM1 Tile from GM, performs SwiGLU/quantization, and increments the GMM2-input-ready counter for the relevant M group.

The AIV does not receive a GMM1 completion flag and then "start GMM1". It receives that flag and starts **SwiGLU**.

## GMM2 Execution and Combine

For GMM2, each C core's Tile loop first waits for SwiGLU/activation data.

In the Wave-grained mode, the required counter is:

```text
targetLoops =
  ceil(actual M group rows / activationTileM) *
  ceil(GMM2 K / 256)
```

The AIC waits until the activation-ready counter for that M group equals this value, then runs its GMM2 Tile(s).

### Non-quantized Combine

For `COMBINE_NO_QUANT` with GM handoff:

```text
AIC GMM2 Tile writes BF16 output to GM
  -> FIX_S event establishes completion of the preceding FIX write
  -> Scalar writes ++gmTileSequence to gmmToEpilogueFlag through WriteGmByPassDCache
  -> AIV polls that flag using ReadGmByPassDCache
  -> AIV copies the Tile GM -> UB and calls CombineTokens
```

The relevant producer order is:

```cpp
blockMmad(... gmBlockC ...);
SetFlag<HardEvent::FIX_S>(0);
WaitFlag<HardEvent::FIX_S>(0);
WriteGmByPassDCache(gmmToEpilogueFlag, ++gmTileSequence);
```

`FIX_S` is the hardware ordering edge from the FIX pipeline to Scalar. The AIV does not directly observe data-write completion. It observes the later GM sequence write, which is ordered after the FIX write.

Non-quantized Combine is scheduler-Tile aligned in this implementation: the AIV walks the same `startLoopIdx, loopIdx += blockNum` sequence as its paired AIC and consumes each resulting GMM2 Tile as it becomes available.

### Quantized Combine

Current quantized Combine uses a separate token-group scheduler. It is not paired one-for-one with the GMM2 scheduler Tile loop.

For each completed GMM2 Tile, AIC calls `NotifyCombineConsumersOfTileCompletion()`:

```text
1. Wait FIX_S to make the GMM2 GM write visible.
2. Map the tile's M offset to a Combine token group.
3. Atomically increment every Combine sync counter consuming that group.
```

The AIV-side `ScheduleQuantizedMoeExpertCombine()`:

```text
* assigns token groups to Combine logical jobs via ComputeCombineGroupsForCore;
* optionally splits one token group among multiple AIV jobs;
* waits until its group counter equals
  ceil(tokenHiddenDim / L1_TILE_N), i.e. all GMM2 N tiles for that group;
* runs CombineQuantizedTokenRange for its assigned tokens.
```

#### Large-expert AIV under-utilization

The mapping is calculated from the full expert group count before filtering to the current Wave slice:

```text
groupCount = ceil(expertTokenCount / 256)
```

When `groupCount >= logicalCoreCount`, each token group is assigned to exactly one V job. Jobs process
additional groups with `groupStride = logicalCoreCount`. For example, with 64 V jobs:

```text
V0: group 0, 64, 128, ...
V1: group 1, 65, 129, ...
```

If a large expert's current Wave slice contains eight token groups, only the eight V jobs owning those groups
execute quantized Combine in that call. The other 56 V jobs find no group in
`[sliceGroupBegin, sliceGroupEnd)` and return from the function without processing tokens. A tail Wave with
fewer groups activates even fewer V jobs. Each active V job waits for all GMM2 N tiles of its group and then
processes up to 256 complete token rows by itself.

The other V jobs are not necessarily stalled because there is no all-core barrier here; they can advance into
later outer-loop work or wait on later dependency flags. Nevertheless, AIV utilization for this Wave slice's
quantized Combine is limited to the number of token groups in the slice.

This is a likely arch92 large-expert performance issue. Redistributing each Wave's groups over all V jobs requires
coordinated changes to the AIV ownership calculation, AIC counter publication, and workspace counter
allocation/reset. Changing only `ScheduleQuantizedMoeExpertCombine()` would make the producer and consumers
disagree on counter ownership.

MX quantization itself is 32-element granular, and a 256-wide N Tile contains eight whole MX groups. Therefore there is no mathematical requirement that MX quantization must wait for an entire hidden vector. The current all-N-tiles-per-token-group waiting behavior is a scheduling and data-ownership choice of this implementation, not an MX-format law.

## Matrix Scaling and Data Types

The current A8W8 GMMs use an MX scaled FP8 MMAD path with BF16 output:

```text
FP8 A + ScaleA
FP8 B + ScaleB
  -> MatmulWithScaleMx / BlockMmad
  -> BF16 C
```

`ScaleA` is internally produced activation scale:

```text
GMM1 ScaleA: input-token quantization before Dispatch, carried through Dispatch
GMM2 ScaleA: SwiGLU output quantization
```

`ScaleB` is the external weight scale input:

```text
GMM1 ScaleB: weightScales1
GMM2 ScaleB: weightScales2
```

Scale is bound to the quantized data, FP8 format, grouping, and quantization rule. It is not bound to a particular dequantized output type. The same FP8 data plus matching scale can represent BF16, FP16, or FP32 values, subject to each destination type's range and rounding.

For MX groups along K, the mathematical operation is:

```text
C[m,n] = sum over K groups g of
  ScaleA[m,g] * ScaleB[g,n] * Dot(QA[m,g], QB[g,n])
```

This is mathematically equivalent to explicit dequantization followed by matrix multiplication. Hardware need not materialize full BF16 A/B matrices: it can form FP8 grouped dot products, apply the group scale product to partial sums, accumulate, then convert to BF16. This transformation does not introduce mathematical error; the dominant losses arise from FP8 quantization and finite-precision accumulation/output conversion.

## Important Corrections and Non-obvious Facts

```text
* Dispatch is expert-grained. `dispatchPosition` tracks consumption of a previously
  dispatched expert across Waves; it does not make Dispatch itself token-slice-grained.

* Initial Dispatch prefetch is AIV-only. AIC begins its GMM path and waits on the
  relevant Dispatch M-group counter, not a full-Wave Dispatch barrier.

* Main-loop Dispatch prepares Wave w+1, while GMM1/SwiGLU process Wave w.

* `ARCH92_GMM1_TILES_PER_JOB_TARGET == 2` is a target aggregate per Wave. It is
  not a guarantee that each core executes exactly two GMM1 logical Tiles or two MMAD calls.

* One GMM1 logical Tile contains Gate and Up MMADs. GMM1 completion requires both.

* Current arch92 Wave has one `startBlockIdx` shared sequentially by GMM1 and GMM2;
  older statements that it has separate GMM1/GMM2 offsets are incorrect for this code.

* `startBlockIdx` balances static logical tile ownership. It does not implement a
  dynamic queue or let a GMM1-idle C core steal an available GMM2 task.

* AIV receiving a GMM1 completion flag means it can begin SwiGLU. It does not mean
  AIV runs GMM1.

* GMM2 non-quant Combine is Tile-aligned. Current quantized Combine is token-group
  scheduled and waits for the GMM2 N tiles needed by its group.
```

## Recommended Reading Order for a Future Agent

1. `arch92/mega_moe_a8w8_wave.h`
   Read `ProcessMoeExpertStages()` first. It owns Wave positions, prefetch, the main
   Dispatch/GMM1 loop, and the later GMM2/Combine loop.

2. `arch92/stage/mega_moe_token_dispatch.h`
   Read `RunMoeExpertDispatchJob()`, `DispatchExpertTokensByRankShard()`, and
   `PublishGmm1TileReady()`.

3. `arch92/stage/mega_moe_gmm1_activation.h`
   Read `WaitForGmm1InputReady()`, the AIC GMM1 Tile executor, and the AIV GM handoff
   epilogue.

4. `arch92/stage/mega_moe_gmm2_combine.h`
   Read `WaitForGmm2InputReady()`, `Gmm2AicMmadGeneric()`,
   `Gmm2AivGmHandoffEpilogue()`, `NotifyCombineConsumersOfTileCompletion()`, and
   `ScheduleQuantizedMoeExpertCombine()`.

5. `arch35/blaze/gemm/block/block_scheduler_swizzle.h`
   Read this only after understanding the Wave controller. It answers how a chosen
   GMM problem is converted from logical tile indices to `(mLoc, nLoc)` coordinates.

## Suggested Verification Questions Before Making Changes

```text
1. Is the target path arch35, arch92, Wave, non-Wave, layered, or shared-expert?
2. Is the code running with TopkWeightsPrefetch enabled? This changes some GMM1
   completion/status behavior.
3. Is Combine non-quantized or quantized? The AIV consumption model differs.
4. Does the new logic preserve the precise producer/consumer ordering around
   FIX_S, GM flags, and bypass-DCache reads?
5. Does every logical job execute identical `startBlockIdx` updates, including when
   it owns zero Tiles?
6. If changing Wave capacity, are both `gmm1TilesPerMGroup` and the Dispatch lead
   relationship still valid?
7. If moving helper functions, has pipeline ownership stayed in the template-level
   controller rather than becoming hidden inside a supposedly reusable atomic helper?
8. If building arch92, does the target CANN package expose the required MX FP8 Cube
   APIs, including all headers and types used by the custom block-MMAD implementation?
```
