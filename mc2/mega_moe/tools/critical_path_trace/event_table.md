# A8W4 trace event table

All timestamps are Scalar observations from `GetSystemCycle()`. `END_COMPLETE` means the event ends after a pipeline
completion visible to Scalar; `END_SUBMIT` means asynchronous work may still be in flight. `TRACE_ONLY_COMPLETION`
identifies a completion event added only in the diagnostic build.

| ID | Event | Role | Begin | End | Completion / wait semantics |
|---:|---|---|---|---|---|
| 1 | `OP_PRE_MOE` | all | `ProcessWave` entry | existing input-preparation `SyncAll<false>` returns | Existing all-role boundary |
| 2 | `DISPATCH_WORK` | AIV1 | before one expert `DispatchMoeExpert` | after its existing `MTE3_S` and row-ready publish | Complete |
| 3 | `WAIT_FOR_DISPATCH` | AIC; initial count wait also AIV0 | before token-count or dispatch-row polling | polling returns | Explicit wait; GMM1 child where contained |
| 4 | `GMM1_PROLOGUE` | AIV0 | before one scheduler tile's two W4 prologues | trace-only `MTE3_S` | Complete; diagnostic synchronization |
| 5 | `GMM1_COMPUTE` | AIC | before one scheduler tile's dispatch wait | existing tile `FIX_S`, then ready publish | Complete; contains ID 3 |
| 6 | `WAIT_FOR_GMM1` | AIV1 | before per-tile sequence/status polling | polling returns | Explicit child of ID 7 |
| 7 | `ACTIVATION_WORK` | AIV1 | before one scheduler tile's GMM1-ready wait | activation epilogue's existing `MTE3_S` and GMM2 publish | Complete; contains ID 6 |
| 8 | `WAIT_FOR_ACTIVATION` | AIC | before per-group activation flag polling | polling returns | Explicit child of ID 10 |
| 9 | `GMM2_PROLOGUE` | AIV0 | before one scheduler tile's W4 prologue | trace-only `MTE3_S` | Complete; diagnostic synchronization |
| 10 | `GMM2_COMPUTE` | AIC | before one scheduler tile's activation wait | trace-only tile `FIX_S` | Complete; contains ID 8; diagnostic synchronization |
| 11 | `WAIT_FOR_GMM2` | combine AIV workers | before scanning all per-AIC ready cache lines | MTE2/reduce/backoff polling returns | Explicit child of ID 12/13 |
| 12 | `COMBINE_WORK` | AIV1 workers | before non-final expert ready wait | combine send loop returns | Submit only; contains ID 11 |
| 13 | `FINAL_COMBINE_OR_DRAIN` | AIV0/AIV1 | final all-AIV combine submit or natural wave drain | submit for final combine; trace-only `MTE3_S` for drain | Drain instance is the true send completion |
| 14 | `COUNT_EXPORT` | AIV | before compact count export | existing export `MTE3_S` | Complete |
| 15 | `UNPERMUTE` | AIV0/AIV1 | before unpermute | trace-only tail `MTE3_S` | Complete; diagnostic synchronization |
| 16 | `OP_POST_MOE` | all | after MoE waves | before raw publication | Role-local tail interval |
| 17 | `CROSS_RANK_INPUT_SYNC` | all | before input GM-signal synchronization | formal synchronization returns | Explicit wait |
| 18 | `CROSS_RANK_OUTPUT_SYNC` | AIV0/AIV1 | before output GM-signal synchronization | formal synchronization returns | Explicit child of ID 16 |
| 19 | `SHARED_GMM1_ACTIVATION` | all | before optional shared GMM1 | role-specific trace-only `FIX_S`/`MTE3_S` | Complete coarse optional stage |
| 20 | `SHARED_GMM2` | all | before optional shared GMM2 | role-specific trace-only `FIX_S`/`MTE3_S` | Complete coarse optional stage |

Each event stores `wave`, `expert`, `tile`, token range and work count. GMM tile IDs are the actual scheduler
`loopIdx`, not a derived top-level call index. Record headers store rank, role, logical core, local process interval,
`written_count` and `dropped_count`.
