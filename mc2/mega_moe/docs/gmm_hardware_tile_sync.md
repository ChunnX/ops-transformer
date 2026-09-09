# GMM GM-output hardware notifications

The Ascend950 wave kernels use bounded mode-4 cross-core notifications for paired GM-output tiles. AIC queues Ready on `PIPE_FIX` after the complete logical output tile, avoiding the previous per-tile `FIX_S` wait and Scalar GM-ready store. The consumer waits on `PIPE_MTE2` before reading the result and returns ACK on `PIPE_MTE3` after its existing epilogue stores.

## Covered paths

| Precision | GMM1 / activation | GMM2 / non-quantized Combine |
| --- | --- | --- |
| A8W8 | GM-output prefetch path: AIV0, bounded hardware notifications | AIV1, bounded hardware notifications |
| A8W4 | Both topk-weight modes: AIV1, bounded hardware notifications | AIV1, bounded hardware notifications |
| A4W4 | GM-output prefetch path: AIV0, bounded hardware notifications | AIV1, bounded hardware notifications |

A8W8/A4W4 non-prefetch GMM1 writes directly to UB and retains its existing one/two-buffer hardware handshake. A limit of 15 notifications is not permission to overwrite an occupied UB buffer. The GMM1 prefetch consumer releases one logical tile only after processing all of its subtiles; non-interleaved gate/up MMAD results share one Ready.

The new pointers are assigned by the wave MoE schedules. Shared-expert and layered completion counters, quantized Combine, input-ready checks, L1/L0 synchronization, UnitFlag, and GMM object lifetimes retain their existing behavior. Those paths can still contain `FIX_S`; this change replaces the paired wave GM-output notification protocol, not every synchronization in the operator.

## Independent limits and event IDs

GMM1 and GMM2 each own a `GmmTileSync`, retained across all expert slices and waves, including lagged GMM2. Both are drained after the complete expert stream. Each AIC counter records acquired logical tiles minus ACKs consumed by that AIC. Before acquiring tile 16 on a channel, AIC consumes one ACK on `PIPE_S`. An already-arrived ACK satisfies that wait immediately.

Each channel independently bounds unread Ready notifications and unconsumed ACKs to at most 15, including events queued but not yet completed. ACKs from GMM2 cannot replenish GMM1 credits. There are no initial dummy notifications or modulo resets, and an empty channel drains without waiting.

| Channel / consumer | Ready: AIC set → AIV wait | ACK: AIV set → AIC wait |
| --- | --- | --- |
| GMM1 / AIV0 | 8 → 8 | 9 → 9 |
| GMM1 / AIV1 | 24 → 8 | 9 → 25 |
| GMM2 / AIV1 | 26 → 10 | 11 → 27 |

These IDs are separate from the W4 prologue flags 0/1 and generic UB epilogue flags 4..7. AIV0 and AIV1 have separate mode-4 namespaces.

`MEGA_MOE_GMM_HARD_TILE_SYNC` defaults to 1. Compile with 0 to use the previous notification protocol. `MEGA_MOE_GMM_MAX_INFLIGHT_TILES` defaults to 15 and is statically restricted to [1,15]. Both are build-time controls.

Reference: [Ascend C CrossCoreWaitFlag documentation](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreWaitFlag_ISASI.html).

## Validation

The standalone host protocol test exercises the actual helper with pipeline/event-ID stubs. It covers both consumer mappings, empty and unequal channels, 15/16-tile boundaries, producer/consumer delays, independent 15+15 outstanding Ready events, and final drains (98 schedules). It checks accounting and routing, not NPU pipeline semantics. See [test instructions](../tests/ut/op_kernel/gmm_tile_sync/README.md).

Performance must be measured separately for the final kernel. Earlier A8W4 experiments with different window sizes or communication stubs are not performance measurements of this change.
