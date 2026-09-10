# Arch22 shared experts after the master chunk-pipeline migration

## Scheduling and ownership

The Ascend910_93 implementation keeps shared experts dense and batch-local.
Shared GMM1, activation and GMM2 execute once per invocation, not once per
received routing round. Master's routed chunk/receive-round loops, rank mask,
round tables and route-cache refreshes remain intact.

- AIC runs shared GMM1 before routed GMM1 preparation is consumed.
- In the first chunk's first receive round, AIV publishes the shared activation
  after dispatch; AIC consumes it after routed GMM1 and runs shared GMM2.
- All AIVs consume shared GMM2 completion once before the first Combine.
  Quantized shared epilogues can overlap routed GMM2. No all-AIV barrier is
  introduced inside the half-AIV Unpermute branch.
- Shared output is `[shared_expert, full_batch_token, hidden]`. Each chunk's
  Unpermute uses the full-batch expert stride and its own token base. Routed
  output, probabilities and permutation indices retain master's chunk layout.

Flags 11/12/13 belong to the shared pipeline. Input-ready and activation-ready
reuse V2C flag 12 in ordered phases. AIV-only SyncAll owns flag 14. Introducing
mixed-core SyncAll while shared flags are live would conflict with 11–13.

## Precision and memory

Shared GMM1/GMM2 intermediates reuse a separate workspace region appended after
master's complete routing allocation, including per-chunk receive-round tables.
The shared activation must be published before its GMM1 buffer can be reused.
INT8/INT4 GMM2 epilogues publish BF16 to a separate result buffer; this preserves
the rounding point and avoids W4A8 high/low-row in-place aliasing. Unpermute adds
routed weighted outputs in FP32 and shared outputs in a separate FP32 sum.
For BF16/FP16 output, each branch is rounded to the output dtype before their
final FP32 addition and output cast, matching the unfused reference boundaries.

Token-active masks apply to routed assignments; shared experts still consume
the original dense X. Rank masks keep master's early-return and communication
semantics. Zero shared experts bypass the extra computation and handshakes.
Ascend910B shared experts remain explicitly rejected by tiling.

## Trade-off

Keeping the dense batch lifetime is a conservative migration, not a claim of
optimal large-batch performance. Its workspace scales with shared-expert count
and full batch size, and the first routed chunk can wait for full-batch shared
work. Chunk-local shared execution is a separate optimization requiring matched
input offsets, scale strides, buffer lifetimes and per-chunk synchronization.
Moving shared GMM2 behind routed GMM2 to overlap Combine also requires a separate
schedule and profiling; Unpermute's final shared addition cannot consume an
unfinished shared result.

## Validation

The migration adds tiling cases for M=1025, a small receive budget and a rank mask.
Validation must use freshly generated kernel scripts and matching master Python
bindings: old bindings lack the mask-buffer argument and cause an ABI mismatch
before kernel execution. Kernel compilation and host/API smoke tests do not
replace device precision tests, especially for multi-chunk and INT4 paths.

### Recorded migration checks (2026-09-09)

The migration is based on master `e14caf335`. The original feature history is
retained at `backup/arch22-shared-expert-v1-pre-master-20260908`.

- Fresh Ascend910_93 build: all 18 kernel targets compiled.
- Host tiling tests: 12/12 passed. API tests: 11/11 passed; shared-input API
  cases are smoke tests using runtime stubs, not device precision checks.
- INT8 and INT4 device checks: one EP=2, shared=4 case per dtype passed the
  harness cosine threshold of 0.999 (M=4, H=2048, topK=8, experts/rank=32).
  This is limited coverage, not a precision guarantee for other shapes.
- BF16 device checks: all five tested cases failed comparison (EP=2, H=1024,
  topK=2, experts/rank=4): M=8 with shared=0/2/4, and M=1025 with shared=0/2
  and receive budget 128. Expert-count outputs also disagreed with the
  reference. Failures with shared=0 mean shared arithmetic alone is not a
  sufficient explanation; they do not prove the failures originate in master.
- Exact-master device comparison is still pending. A supplemental BF16
  experts/rank=8 run could not start because HCCL port 16666 was already bound.
  All 16 NPUs were occupied when checked; no other jobs were stopped.

Device checks use seed 20260908 and Python bindings generated from this tree,
with an isolated custom OPP install. Local evidence is retained in
`/tmp/mega-moe-rebase-{host-test,api-test,quant-run,bf16-v2-run}.log` and
`/tmp/mega-moe-rebase-{quant,bf16-v2}-results`. Input cases are in
`/tmp/mega-moe-rebase-{quant,bf16,bf16-epr8}-cases.json`.
An unmodified master worktree for the control is at
`/tmp/mega-moe-master-baseline-20260908`.

Before accepting precision, run master and the migrated branch with matching
bindings, seed, inputs and isolated OPP packages on two idle NPUs. Compare the
zero-shared cases first, then resolve the BF16 failures and rerun shared and
multi-chunk cases. Masked execution and broader quantized shapes still need
device coverage. No performance improvement is claimed without profiling.

### BF16 routing fixes and verification (2026-09-09)

The exact master control subsequently reproduced both zero-shared failures
(M=8 and M=1025). Inspection identified three related routing defects:

- A 2D GM-to-UB DataCopyPad pads each expert-count row to 32 bytes, whereas
  consumers index a compact `[source_rank, local_expert]` cache. For unaligned
  expert counts, copy into temporary padded rows and compact after MTE2 completes.
  Aligned expert counts retain the direct-copy path.
- ExpertTokenNums initialization and accumulation used aligned DataCopy for
  potentially shorter rows. Use exact-byte DataCopyPad with producer/consumer
  events around vector operations and GM writes.
- Receive-round intersections are calculated in receiver row order, but remote
  A/D buffers use sender row order. The transfer offset must be the original
  sender prefix plus the intersection's offset within the block. Reload the
  original chunk prefixes, not the preceding round's shifted prefixes. Empty
  intersections retain a valid base offset; temporary tables are 32-byte aligned.

After these fixes, all five original BF16 cases passed the unchanged cosine
threshold of 0.999 (minimum observed 0.999918). All ten rank/case expert-count
outputs matched the CPU reference exactly. The final source also compiled all
18 Ascend910_93 kernel configurations. Evidence:
`/tmp/mega-moe-bf16-actual-fix.log`,
`/tmp/mega-moe-bf16-actual-fix-results`, and
`/tmp/mega-moe-full-fix-build.log`.

Build caveat: generated scripts prefer their copied `op_kernel` tree over
BUILD_KERNEL_SRC. Early attempted fix runs still used the old snapshot and are
not fix-validation evidence. The successful run used the updated snapshot;
the final full build's snapshot was checked against the working source.

The standalone `tests/st/arch22_shared_precision.py` regression adds repeated
calls, active-prefix masks, exact counts and a relative-L2 check. Its first run
exposed missing test communicator initialization, which has been corrected;
rerunning it was blocked by execution approval limits. Its full matrix and
INT8/INT4 device regression **after these routing fixes** remain pending. The
earlier quantized passes predate these fixes and are not substitutes for reruns.

### Shared reduction rounding fix (2026-09-10)

Restore the branch rounding previously introduced by `4cf085ad7`, removed by
the FP32 shared-GMM2 experiment `6ea278b14`, and not restored when `6de7312ba`
returned shared GMM2 to BF16. Non-FP32 outputs now compute
`T1(float(T1(routed_sum)) + float(T1(shared_sum)))`. Shared experts are summed
independently in FP32 before their branch cast. The initial implementation in
`9bfb56a52` used existing queue slots to hold the rounded branches.
Zero-shared and FP32-output paths retain their existing behavior. This also
applies to quantized paths producing BF16; their device regression is required.

The BF16 regression script now uses ND weights and INT8 active masks, as required
by the interface. Compilation and tests for this rounding fix were intentionally
not run at the user's request. The earlier results above predate this fix and
must not be taken as validation of it. Recheck the supplied EP16 cases with the
full dual-benchmark criteria, not only cosine similarity.

### Shared reduction queue isolation (2026-09-10)

The user reported a severe regression after `9bfb56a52`: DeepSeek INT8/SwiGLU
shared=1 went from 13/13 to 0/13 passing, GLM BF16/SwiGLU shared=2 from 10/13
to 0/13, while Kimi INT8/SiTU shared=4 remained at 13/13. These errors are
not explained by ordinary rounding. Unpermute does not select this reduction
by activation type, so Kimi passing does not establish that it bypasses it.

Replace the queue-backed branch temporaries with dedicated VECCALC buffers:
keep the rounded routed result in its own FP32 accumulator, reduce shared
outputs into a separate FP32 accumulator, and use one dedicated T1 scratch
for both branch casts. Explicit vector barriers separate scratch consumers
from reuse. VECIN and VECOUT queues now serve only their normal GM transfer
roles; no saved branch is written into VECIN or read back from VECOUT.
This removes the suspect queue-lifetime/synchronization pattern; the exact
instruction responsible for the reported regression is not yet demonstrated.

Non-FP32 shared paths cap the hidden tile at 1024 elements to bound UB use.
At that tile size, the new FP32 accumulator and BF16/FP16 cast scratch require
6 KiB; larger hidden dimensions use additional tiles with the original row
stride and a final partial tile when necessary. GM workspace, cross-core
scheduling, zero-shared and FP32-output paths are unchanged. Extra tiling and
barriers may affect performance; no speedup is claimed.

Compilation and device tests for this replacement were not run, per the user's
request. Precision is not yet validated. Rerun all three reported suites with
the unchanged full comparison criteria, including Kimi, and cover hidden-dimension
tails, active masks and receive-round splitting before accepting this fix.
