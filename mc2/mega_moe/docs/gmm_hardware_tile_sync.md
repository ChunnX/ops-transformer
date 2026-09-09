# GMM GM-output hardware notifications

The Ascend950 wave kernels use bounded mode-4 cross-core notifications for paired GM-output tiles. AIC queues Ready on `PIPE_FIX` after the complete logical output tile, avoiding the previous per-tile `FIX_S` wait and Scalar GM-ready store. The consumer waits on `PIPE_MTE2` before reading the result and returns ACK on `PIPE_MTE3` after its existing epilogue stores.

## Covered paths

| Precision | MoE GMM1 / activation | MoE GMM2 / non-quantized Combine |
| --- | --- | --- |
| A8W8 | GM-output prefetch path: AIV0, bounded hardware notifications | AIV1, bounded hardware notifications |
| A8W4 | Both topk-weight modes: AIV1, bounded hardware notifications | AIV1, bounded hardware notifications |
| A4W4 | GM-output prefetch path: AIV0, bounded hardware notifications | AIV1, bounded hardware notifications |

A8W8/A4W4 non-prefetch GMM1 writes directly to UB and retains its existing two-buffer hardware handshake. A limit of 15 notifications is not permission to overwrite an occupied UB buffer. GMM1 always produces an interleaved logical tile from one BlockMmad call. The consumer releases one logical tile only after processing all of its M subtiles (including both 128-row subtiles of a 256-row prefetch tile and a possible tail). Ready/ACK counts do not increase for those subtiles.

Shared A8W4 GMM1 also writes GM and uses bounded notifications to AIV1 in both the Wave and layered shared-expert entry points. Shared GMM1 always disables TopK weight prefetch; shared A8W8/A4W4 therefore write UB and retain their existing two-buffer handshake. Shared GMM2 has no paired Combine consumer and keeps its existing completion protocol.

The sync pointers are assigned by the wave MoE and shared A8W4 schedules. Layered MoE completion counters, quantized Combine, input-ready checks, L1/L0 synchronization, UnitFlag, and GMM object lifetimes retain their existing behavior. Those paths can still contain `FIX_S`; this change replaces the paired GM-output notification protocol, not every synchronization in the operator.

## Independent limits and event IDs

`Gmm1SwigluSync` and `Gmm2CombineSync` own separate counters and event IDs, retained across all expert slices and waves, including lagged GMM2. Both are drained after the complete expert stream. Each AIC counter records acquired logical tiles minus ACKs consumed by that AIC. Starting with the 16th cumulative tile on a channel, AIC consumes one ACK on `PIPE_S` before every acquisition. An already-arrived ACK satisfies that wait immediately. The counter is not a live measurement of consumer lag: completed ACKs remain counted until AIC consumes them.

Shared GMM1 uses a separate `Gmm1SwigluSync` instance with `EventPair::SHARED`. Its counter persists across all shared experts and remains alive in the outer `ProcessWave` / layered `Process` until the complete MoE stage finishes. The shared entry does not drain its remaining ACKs, so AIC can submit MoE GMM work while shared notifications are still outstanding. The producer still consumes ACKs at the fifteen-tile limit inside shared GMM1. Layered retains its pre-existing `SyncAll<false>()` after shared GMM1; independent event IDs do not remove that barrier or other data dependencies.

Each channel independently bounds unread Ready notifications and unconsumed ACKs to at most 15, including events queued but not yet completed. ACKs from GMM2 cannot replenish GMM1 credits. There are no initial dummy notifications or modulo resets, and an empty channel drains without waiting.

For one channel, let `T` be reserved tiles and `A` the ACKs consumed by AIC. The producer maintains `T - A <= 15`. A consumer can return an ACK only after consuming a Ready, so both unread Ready and unconsumed ACK counts are bounded by `T - A`. Reserving before BlockMmad also accounts for FIX notifications that have been submitted but have not executed yet. Draining inside a wave would add an unnecessary barrier; resetting the count there without draining would break this bound.

| Channel / consumer | Ready: AIC set → AIV wait | ACK: AIV set → AIC wait |
| --- | --- | --- |
| GMM1 / AIV0 | 8 → 8 | 9 → 9 |
| GMM1 / AIV1 | 24 → 8 | 9 → 25 |
| GMM2 / AIV1 | 26 → 10 | 11 → 27 |
| Shared GMM1 / AIV1 | 28 → 12 | 13 → 29 |

These IDs are separate from the W4 prologue flags 0/1 and generic UB epilogue flags 4..7. AIV0 and AIV1 have separate mode-4 namespaces.

The covered paths always use hardware notifications with a fixed limit of 15 per channel. Shared GMM1, MoE GMM1 and MoE GMM2 can each retain fifteen notifications independently. `EventPair` selects the internal event namespace; it is not an external switch or configurable window.

`mega_moe_gmm_epilogue_sync.h` names both endpoints explicitly, following the existing `NotifyVector` / `WaitForCube` / `EndSync` style:

| Pair | AIC: before / after submitting a tile | AIV: before reading / after consuming a tile |
| --- | --- | --- |
| `Gmm1SwigluSync` | `WaitForSwiglu()` / `NotifySwiglu()` | `WaitForGmm1()` / `NotifyGmm1()` |
| `Gmm2CombineSync` | `WaitForCombine()` / `NotifyCombine()` | `WaitForGmm2()` / `NotifyGmm2()` |

The AIC wait reserves one tile and only waits for an acknowledgement when 15 tiles are pending. Each pair's `EndSync()` consumes its remaining acknowledgements after the complete MoE stream. The optional GMM1 pointer selects bounded pairing where configured; every covered GM-output specialization assigns it, including shared A8W4. Layered MoE retains its original completion protocol.

Reference: [Ascend C CrossCoreWaitFlag documentation](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreWaitFlag_ISASI.html).

## Integration and validation

This change adapts [PR #11652](https://gitcode.com/cann/ops-transformer/pull/11652) to the always-interleaved GMM1 paths and extends it to shared A8W4 GMM1. The existing W4 wave loops and A8W8/A8W4 lagged GMM2 schedule are preserved; the sync objects live outside those loops, and final drains follow the last GMM2/Combine call. Paired non-quantized wave GMM2 callers always initialize their sync pointer before dispatch; shared GMM2 does not use that channel.

The integrated header passed 98 CPU protocol schedules with simulated hardware events: both SwiGLU peers, empty channels, 15/16 boundaries, slow producers/consumers, independent 15+15 Ready notifications and complete ACK drains. Thirty Ascend950 MTE kernel specializations compiled successfully, covering both prefetch modes, all three precisions, mixed shared experts, A8W8 NZ/E5M2 and quantized Combine. Validation sources remain in a local temporary directory.

After adding the independent shared channel, the 98 schedules were rerun and 32 event-isolation cases verified exact IDs/pipes, empty work, the 15/16 boundary, and simultaneous 15+15+15 pending notifications. The cases hold either shared Ready or shared ACK notifications while MoE runs, then verify separate final drains. Fourteen MTE specializations were rebuilt: all three pure precisions, shared A8W4 paired with each MoE precision, and shared A8W8/A4W4 paired with MoE A8W4, each with MoE prefetch enabled/disabled. The layered shared entry received the same pointer/lifetime wiring but is outside the successful device-compilation coverage described below.

These checks do not execute the kernel on an NPU. Device compilation isolated the MTE entry because the installed Hcomm lacks the `BatchHandle` / `BatchCommit` APIs needed for the full URMA entry. Numerical correctness, runtime synchronization and performance still require validation on compatible hardware and software.

Performance must be measured separately for the final kernel. Earlier A8W4 experiments with different window sizes or communication stubs are not performance measurements of this change.
