# A8W4 call chain and PR 10302 static suspects

## Actual device chain

The A8W4 specialization is selected when `Weight1Type` is `fp4x2_e2m1_t` and dispatch quant output is
`fp8_e4m3fn_t`. `mega_moe_apt.cpp` instantiates `MegaMoeA8W4Wave`, whose `Process()` enters the common
`MegaMoe::ProcessWave` and delegates only the MoE expert section to `ProcessMoeExpertStages`.

The outer order is input quant/mask/reset/shared-input, the existing all-role input barrier, optional shared GMM1,
cross-rank input sync, A8W4 dynamic waves, expert count export, optional shared GMM2, cross-rank output sync, and
unpermute. This is why `OP_PRE_MOE`, count export, output sync, unpermute and `OP_POST_MOE` are traced outside the
wave scheduler.

Within A8W4 waves, AIV1 dispatches the first full wave. In steady state it alternates next-wave expert Dispatch with
current-wave Activation. AIC executes GMM1 MMAD, AIV0 executes W4-to-W8 prologue, and AIV1 consumes GMM1 output for
activation/quantization. Each completed wave immediately enters GMM2. On the current mainline quantized-Combine
path, GMM2 AIC publishes a counter for every completed 256-row group/N-tile contribution; the paired AIV1 waits for
its group's counter to reach the N-tile count and then sends only its owned token range. There is no separate
`NotifyWaveGmm2Ready`/row-ring drain in this active path. The trace therefore records per-group submit intervals and
does not synthesize a per-schedule completion event.

The synchronization chain is:

- Dispatch `MTE3_S` -> AtomicAdd `flagDispatchToGmm1Ptr[expert, 256-row-group]` -> GMM1 AIC equality polling.
- GMM1 AIC `FIX_S` -> per-AIC monotonic `flagGmmToEpiloguePtr` sequence (or prefetch tile-status slot) -> AIV1
  polling.
- Activation output `MTE3_S` -> AtomicAdd `flagActivationToGmm2Ptr[expert, 256-row-group]` -> GMM2 AIC equality
  polling.
- GMM2 AIC `FIX_S` -> AtomicAdd on `gmm2CombineSyncCounterPtr[expert, group/job-slot]` -> paired AIV1 equality
  polling -> group-local Combine sends.
- Quantized Combine reuses its UB through `MTE3_MTE2`; this does not prove Scalar-visible MTE3 completion. The trace
  records group work as submit-only. Event 13 remains reserved for protocol/report compatibility but is not emitted
  on this mainline path, avoiding a trace-only serialization point between schedules.

## Latest-mainline comparison base

This branch is based on upstream `master` at `a65a6e3018425dba19372e2def01a8db2f758aa5`. Relative to the collected
PR-10302 trace base `58219154d497e51b1c4840b9d085ed7290cc83b1`, the current A8W4 device path must be compared by semantic event
IDs rather than identical source locations: the active GMM2/Combine handoff has changed from the older wave-ready
scan/drain protocol to the per-group shared-counter protocol described above. Protocol version, event IDs, record
layout and report tool remain identical so good/bad raw files are directly comparable.

## Requested commits

- `47e8d7517` changes `GmmKernel::BlockScheduler` from `BlockSchedulerSwizzle<3, 1>` to `<3, 0>` in
  `mega_moe_gmm_common.h`. Both A8W4 GMM1 and GMM2 instantiate this scheduler. It is a high-priority suspect because
  it changes tile order, AIV0/AIC rendezvous order, weight reuse and the identity of the tail core.
- `26dcbd35b` removes vector barriers in input `SendAndQuantBuffInit`. A8W4 executes this path, but the affected
  buffer reuse still has a hard event; it is a lower-priority main-stage suspect.
- `c8ebc494b` introduces the AIV1 wave-ahead Dispatch schedule used directly by `MegaMoeA8W4Wave`. It is
  high-priority for lost Dispatch/Activation overlap or new AIV1 serialization.
- `54fc175dd` persists Dispatch prefix state in `cumsumInfoPtr` and reloads it with an MTE2-to-Scalar completion
  before each expert Dispatch. This is on the A8W4 steady-state path and may add AIV1 MTE2 work or contention.
- `66ea39f46` changes continuation-slice token-count reads to the compact block-major table. It is active for A8W4,
  but is a small Scalar read-path change compared with the scheduler changes.
- `d417dd855` changes polling backoff for shared-expert GMM2/unpermute readiness. It affects A8W4 only when shared
  experts are configured and is outside normal MoE GMM2-ready polling.
- `c2c10c560` inserts the final AIV0 `MTE3_MTE2` dependency before all-AIV final combine. It is active on the A8W4
  tail and can add final-wave serialization, although it protects real UB reuse.
- `51f9a6a67` resets the drained combine row-ring sequence. It is active at every A8W4 wave boundary and changes
  which reuse events the following wave consumes.
- `143cfeb62` exposes Dispatch configuration/scratch to W4 waves and is instantiated by A8W4, but its direct change
  is mainly plumbing that enables the later wave-ahead schedule.

The first bisection candidates are therefore `47e8d7517`, `c8ebc494b`, and `54fc175dd`, followed by the final-combine
and row-ring pair `c2c10c560`/`51f9a6a67`. `d417dd855` should only be prioritized when `sharedExpertNum > 0` and the
regression lies after the main MoE waves.
