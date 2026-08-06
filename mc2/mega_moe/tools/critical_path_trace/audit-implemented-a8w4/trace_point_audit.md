# Trace point audit

- Status: `pass`
- Phase: `implemented`
- Trace: `mega-moe-a8w4-critical-path`
- Repository: `/home/oiseau/codes/ops-transformer`
- Revision: `c6568d155d5af1cc6f540c5bac4206b93a3ed144`
- Branch: `codex/mega-moe-a8w4-critical-path-trace`
- Dirty: `True`
- Spec SHA-256: `e9f81b37a4a09ac1ca8640143c708b121b93c9d4e2d1d5f898cf554f19ff34d6`

## Placement summary

| Point | Roles | Resolved location | Target line | Placement | Boundary | Bridge | Why |
|---|---|---|---:|---|---|---|---|
| a8w4_process_begin_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe.h:2195 | 2195 | before | causal | none | establish the local process epoch |
| a8w4_work_end_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe.h:2445 | 2446 | before | actual | AIV uses V_S EVENT_ID0; AIC reuses existing FIX_S publication | separate useful work from trace emission |
| a8w4_dispatch_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe.h:297 | 298 | before | causal | none | measure dispatch prefetch placement relative to A8W4 compute |
| a8w4_dispatch_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe.h:308 | 309 | before | actual | MTE3_S EVENT_ID0 | prove dispatch GM writes complete |
| a8w4_gmm1_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1133 | 1134 | before | causal | none | locate AIC GMM1 after dispatch readiness |
| a8w4_gmm1_end_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe.h:2301 | 2302 | before | actual | existing per-tile FIX_S publication | measure actual AIC GMM1 completion |
| a8w4_gmm1_prologue_begin_S | AIV0 | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1206 | 1207 | before | causal | none | expose W4-to-W8 prologue occupancy |
| a8w4_gmm1_prologue_end_S | AIV0 | mc2/mega_moe/op_kernel/arch35/mega_moe.h:2310 | 2311 | before | actual | MTE3_S EVENT_ID0 | measure completed W4-to-W8 prologue work |
| a8w4_swiglu_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_trace.h:551 | 552 | before | causal | none | measure AIV1 epilogue after tile readiness |
| a8w4_swiglu_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe.h:2317 | 2318 | before | actual | existing per-call MTE3_S | measure completed SwiGLU and MX quant output |
| a8w4_gmm2_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1138 | 1139 | before | causal | none | locate GMM2 after SwiGLU readiness |
| a8w4_gmm2_end_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe.h:2391 | 2392 | before | actual | existing per-tile FIX_S publication | measure actual AIC GMM2 completion/publication |
| a8w4_gmm2_prologue_begin_S | AIV0 | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1211 | 1212 | before | causal | none | expose GMM2 W4-to-W8 prologue occupancy |
| a8w4_gmm2_prologue_end_S | AIV0 | mc2/mega_moe/op_kernel/arch35/mega_moe.h:2400 | 2401 | before | actual | MTE3_S EVENT_ID0 | measure completed W4-to-W8 prologue work |
| a8w4_combine_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_trace.h:559 | 560 | before | causal | none | measure Combine only after GMM2 output is visible |
| a8w4_combine_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe.h:2409 | 2410 | before | actual | MTE2_S EVENT_ID0 | measure completed remote combine writes |
| a8w4_combine_tile_wait_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1388 | 1389 | before | causal | none | measure the exact per-tile GMM2-to-Combine wait |
| a8w4_combine_tile_wait_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1401 | 1403 | before | actual | none | close the exact per-tile readiness wait |
| a8w4_combine_tile_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1402 | 1403 | before | causal | none | separate GMM2 wait from Combine work |
| a8w4_combine_tile_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1434 | 1435 | before | actual | MTE3_MTE2 followed by MTE2_S EVENT_ID0 | measure per-tile Combine work through actual remote write completion |

## Point evidence

### `a8w4_process_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:2195` in `MegaMoe::Process`; `before` target line `2195`.
- Cardinality: once per participating physical role
- Boundary claim: `causal` — entry to the MTE A8W4 process before useful work
- Why: establish the local process epoch
- Dependency: kernel entry
- Pipeline proof: `Scalar` → `Scalar` using `none` — the clock is the first trace action in Process
- Perturbation: one clock read
- Source SHA-256: `c13170e8e9c5eaecf9a5ab03606d522bd39febdea7da7eba5f54756f9e2d0e11`

```text
  2193  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  2194      if (CriticalPathTracePtr() != nullptr) {
  2195          criticalPathTrace_.processBegin = CriticalPathTrace::Now(); // TRACE_POINT: a8w4_process_begin_S
  2196      }
  2197  #endif
```

### `a8w4_work_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:2445` in `MegaMoe::Process`; `before` target line `2446`.
- Cardinality: once per participating physical role
- Boundary claim: `actual` — all role-local useful work has completed
- Why: separate useful work from trace emission
- Dependency: AIV Unpermute drain or AIC final compute return
- Pipeline proof: `mixed` → `Scalar` using `AIV uses V_S EVENT_ID0; AIC reuses existing FIX_S publication` — Unpermute transfers its final MTE3 completion to Vector through MTE3_V, then the trace-only V_S transfers it to Scalar; AIC compute returns after FIX_S publication
- Perturbation: AIV adds one V_S completion bridge; all roles add one clock read
- Source SHA-256: `c13170e8e9c5eaecf9a5ab03606d522bd39febdea7da7eba5f54756f9e2d0e11`

```text
  2443              AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
  2444          }
  2445          // TRACE_POINT: a8w4_work_end_S
  2446          criticalPathTrace_.workEnd = CriticalPathTrace::Now();
  2447          SyncAll<false>();
```

### `a8w4_dispatch_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:297` in `MegaMoe::BeginDispatchTrace`; `before` target line `298`.
- Cardinality: once per routed expert on an active AIV1
- Boundary claim: `causal` — before SendCntCal and dispatch metadata/token movement for the expert
- Why: measure dispatch prefetch placement relative to A8W4 compute
- Dependency: the caller selected the next expert
- Pipeline proof: `Scalar` → `Scalar` using `none` — both dispatch call sites invoke this helper immediately before SendCntCal
- Perturbation: one clock read
- Source SHA-256: `c13170e8e9c5eaecf9a5ab03606d522bd39febdea7da7eba5f54756f9e2d0e11`

```text
   295      if (CriticalPathTracePtr() != nullptr) {
   296          criticalPathTrace_.PrepareExpertStage0(expertIdx);
   297          // TRACE_POINT: a8w4_dispatch_begin_S
   298          criticalPathTrace_.BeginExpertStage0(CriticalPathTrace::Now());
   299      }
```

### `a8w4_dispatch_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:308` in `MegaMoe::EndDispatchTrace`; `before` target line `309`.
- Cardinality: once per routed expert on an active AIV1
- Boundary claim: `actual` — the expert dispatch MTE3 writes have completed
- Why: prove dispatch GM writes complete
- Dependency: MetaInfoCalAndDispatch returned
- Pipeline proof: `MTE3` → `Scalar` using `MTE3_S EVENT_ID0` — the clock follows the trace-only MTE3_S set/wait
- Perturbation: forces the expert dispatch tail to complete
- Source SHA-256: `c13170e8e9c5eaecf9a5ab03606d522bd39febdea7da7eba5f54756f9e2d0e11`

```text
   306          AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
   307          AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
   308          // TRACE_POINT: a8w4_dispatch_end_S
   309          uint32_t dispatchEndCycle = CriticalPathTrace::Now();
   310          criticalPathTrace_.EndExpertStage0(dispatchEndCycle);
```

### `a8w4_gmm1_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1133` in `AicComputeA8W4`; `before` target line `1134`.
- Cardinality: first owned GMM1 tile per routed expert and AIC
- Boundary claim: `causal` — after WaitForUpstreamReady and before the first blockMmad
- Why: locate AIC GMM1 after dispatch readiness
- Dependency: dispatch wave ready
- Pipeline proof: `Scalar polling` → `Scalar` using `none` — the clock is after WaitForUpstreamReady in the tile loop
- Perturbation: one clock read
- Source SHA-256: `d8be4e6f4d0b93322e4438f3b8d86c9e00c1ef1f7e8924c5df455c685830e53c`

```text
  1131          if constexpr (Policy::IS_GMM1) {
  1132              if (criticalPathTrace != nullptr && criticalPathTrace->NeedsExpertStage0Begin()) {
  1133                  // TRACE_POINT: a8w4_gmm1_begin_S
  1134                  uint32_t a8w4Gmm1BeginCycle = CriticalPathTrace::Now();
  1135                  criticalPathTrace->BeginExpertStage0(a8w4Gmm1BeginCycle);
```

### `a8w4_gmm1_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:2301` in `MegaMoe::Process`; `before` target line `2302`.
- Cardinality: last owned GMM1 tile per routed expert and AIC
- Boundary claim: `actual` — all owned GMM1 tiles have been published after FIX_S
- Why: measure actual AIC GMM1 completion
- Dependency: AicComputeA8W4 returned
- Pipeline proof: `FIX` → `Scalar` using `existing per-tile FIX_S publication` — every specialized tile publishes only after FIX_S before the function returns
- Perturbation: one clock read
- Source SHA-256: `c13170e8e9c5eaecf9a5ab03606d522bd39febdea7da7eba5f54756f9e2d0e11`

```text
  2299                  if (criticalPathTrace_.HasExpertStage0Work()) {
  2300                      // Every A8W4 tile publishes only after FIX_S, so return is an actual GMM1 completion boundary.
  2301                      // TRACE_POINT: a8w4_gmm1_end_S
  2302                      uint32_t gmm1EndCycle = CriticalPathTrace::Now();
  2303                      criticalPathTrace_.EndExpertStage0(gmm1EndCycle);
```

### `a8w4_gmm1_prologue_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1206` in `AivPrologueA8W4`; `before` target line `1207`.
- Cardinality: first owned GMM1 weight tile per expert and AIV0
- Boundary claim: `causal` — before the first GMM1 blockPrologue issue
- Why: expose W4-to-W8 prologue occupancy
- Dependency: the expert prologue loop is entered
- Pipeline proof: `Scalar` → `Scalar` using `none` — the sample precedes the first blockPrologue call
- Perturbation: one clock read
- Source SHA-256: `d8be4e6f4d0b93322e4438f3b8d86c9e00c1ef1f7e8924c5df455c685830e53c`

```text
  1204          if constexpr (Policy::IS_GMM1) {
  1205              if (args.criticalPathTrace != nullptr && args.criticalPathTrace->NeedsExpertStage0Begin()) {
  1206                  // TRACE_POINT: a8w4_gmm1_prologue_begin_S
  1207                  uint32_t a8w4Gmm1PrologueBeginCycle = CriticalPathTrace::Now();
  1208                  args.criticalPathTrace->BeginExpertStage0(a8w4Gmm1PrologueBeginCycle);
```

### `a8w4_gmm1_prologue_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:2310` in `MegaMoe::Process`; `before` target line `2311`.
- Cardinality: after the last owned GMM1 weight tile completes
- Boundary claim: `actual` — all owned GMM1 prologue MTE3 writes to L1 have completed
- Why: measure completed W4-to-W8 prologue work
- Dependency: AivPrologueA8W4 and BlockPrologue destruction returned
- Pipeline proof: `MTE3` → `Scalar` using `MTE3_S EVENT_ID0` — the clock follows a trace-only MTE3_S set/wait after the prologue event rings are reclaimed
- Perturbation: one MTE3_S completion bridge and one clock read per active expert
- Source SHA-256: `c13170e8e9c5eaecf9a5ab03606d522bd39febdea7da7eba5f54756f9e2d0e11`

```text
  2308                      AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
  2309                      AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
  2310                      // TRACE_POINT: a8w4_gmm1_prologue_end_S
  2311                      uint32_t gmm1PrologueEndCycle = CriticalPathTrace::Now();
  2312                      criticalPathTrace_.EndExpertStage0(gmm1PrologueEndCycle);
```

### `a8w4_swiglu_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_trace.h:551` in `RecordA8W4SwiGluBeginAfterReady`; `before` target line `552`.
- Cardinality: first ready GMM1 output tile per expert and AIV1
- Boundary claim: `causal` — after status/sequence polling and before the first GMM1-output load
- Why: measure AIV1 epilogue after tile readiness
- Dependency: AIC published the tile-ready state
- Pipeline proof: `Scalar polling` → `Scalar` using `none` — both call sites invoke this helper only after their ready loop
- Perturbation: one clock read
- Source SHA-256: `fb88e0a8f06bda4ab3dfdc5874e6145ced5c12f6cf7ccbd8d9fd68bc82828bc8`

```text
   549  {
   550      if (trace != nullptr && trace->NeedsExpertStage1Begin()) {
   551          // TRACE_POINT: a8w4_swiglu_begin_S
   552          trace->BeginExpertStage1(CriticalPathTrace::Now());
   553      }
```

### `a8w4_swiglu_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:2317` in `MegaMoe::Process`; `before` target line `2318`.
- Cardinality: after the last owned SwiGLU tile completes per expert
- Boundary claim: `actual` — the final SwiGLU data/scale MTE3 writes and ready publication have completed
- Why: measure completed SwiGLU and MX quant output
- Dependency: AivGmm1PostA8W4 returned
- Pipeline proof: `MTE3/Scalar` → `Scalar` using `existing per-call MTE3_S` — BlockEpilogueSwigluMxQuant returns only after MTE3_S and its Scalar ready AtomicAdd; the outer clock follows the final call
- Perturbation: one clock read; no additional completion bridge
- Source SHA-256: `c13170e8e9c5eaecf9a5ab03606d522bd39febdea7da7eba5f54756f9e2d0e11`

```text
  2315              } else {
  2316                  if (criticalPathTrace_.HasExpertStage1Work()) {
  2317                      // TRACE_POINT: a8w4_swiglu_end_S
  2318                      uint32_t swigluEndCycle = CriticalPathTrace::Now();
  2319                      criticalPathTrace_.EndExpertStage1(swigluEndCycle);
```

### `a8w4_gmm2_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1138` in `AicComputeA8W4`; `before` target line `1139`.
- Cardinality: first owned GMM2 tile per routed expert and AIC
- Boundary claim: `causal` — after WaitForUpstreamReady and before first GMM2 blockMmad
- Why: locate GMM2 after SwiGLU readiness
- Dependency: SwiGLU output ready
- Pipeline proof: `Scalar polling` → `Scalar` using `none` — the sample follows the upstream wait
- Perturbation: one clock read
- Source SHA-256: `d8be4e6f4d0b93322e4438f3b8d86c9e00c1ef1f7e8924c5df455c685830e53c`

```text
  1136              }
  1137          } else if (criticalPathTrace != nullptr && criticalPathTrace->NeedsExpertStage1Begin()) {
  1138              // TRACE_POINT: a8w4_gmm2_begin_S
  1139              uint32_t a8w4Gmm2BeginCycle = CriticalPathTrace::Now();
  1140              criticalPathTrace->BeginExpertStage1(a8w4Gmm2BeginCycle);
```

### `a8w4_gmm2_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:2391` in `MegaMoe::Process`; `before` target line `2392`.
- Cardinality: last owned GMM2 tile per routed expert and AIC
- Boundary claim: `actual` — all owned GMM2 tiles have passed FIX_S publication
- Why: measure actual AIC GMM2 completion/publication
- Dependency: AicComputeA8W4 returned
- Pipeline proof: `FIX` → `Scalar` using `existing per-tile FIX_S publication` — tile completion notification or GM epilogue publication follows FIX_S
- Perturbation: one clock read
- Source SHA-256: `c13170e8e9c5eaecf9a5ab03606d522bd39febdea7da7eba5f54756f9e2d0e11`

```text
  2389                  if (criticalPathTrace_.HasExpertStage1Work()) {
  2390                      // Every A8W4 tile publishes only after FIX_S, so return is an actual GMM2 completion boundary.
  2391                      // TRACE_POINT: a8w4_gmm2_end_S
  2392                      uint32_t gmm2EndCycle = CriticalPathTrace::Now();
  2393                      criticalPathTrace_.EndExpertStage1(gmm2EndCycle);
```

### `a8w4_gmm2_prologue_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1211` in `AivPrologueA8W4`; `before` target line `1212`.
- Cardinality: first owned GMM2 weight tile per expert and AIV0
- Boundary claim: `causal` — before first GMM2 blockPrologue issue
- Why: expose GMM2 W4-to-W8 prologue occupancy
- Dependency: the expert prologue loop is entered
- Pipeline proof: `Scalar` → `Scalar` using `none` — the sample precedes blockPrologue
- Perturbation: one clock read
- Source SHA-256: `d8be4e6f4d0b93322e4438f3b8d86c9e00c1ef1f7e8924c5df455c685830e53c`

```text
  1209              }
  1210          } else if (args.criticalPathTrace != nullptr && args.criticalPathTrace->NeedsExpertStage1Begin()) {
  1211              // TRACE_POINT: a8w4_gmm2_prologue_begin_S
  1212              uint32_t a8w4Gmm2PrologueBeginCycle = CriticalPathTrace::Now();
  1213              args.criticalPathTrace->BeginExpertStage1(a8w4Gmm2PrologueBeginCycle);
```

### `a8w4_gmm2_prologue_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:2400` in `MegaMoe::Process`; `before` target line `2401`.
- Cardinality: after the last owned GMM2 weight tile completes
- Boundary claim: `actual` — all owned GMM2 prologue MTE3 writes to L1 have completed
- Why: measure completed W4-to-W8 prologue work
- Dependency: AivPrologueA8W4 and BlockPrologue destruction returned
- Pipeline proof: `MTE3` → `Scalar` using `MTE3_S EVENT_ID0` — the clock follows a trace-only MTE3_S set/wait after the prologue event rings are reclaimed
- Perturbation: one MTE3_S completion bridge and one clock read per active expert
- Source SHA-256: `c13170e8e9c5eaecf9a5ab03606d522bd39febdea7da7eba5f54756f9e2d0e11`

```text
  2398                      AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
  2399                      AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
  2400                      // TRACE_POINT: a8w4_gmm2_prologue_end_S
  2401                      uint32_t gmm2PrologueEndCycle = CriticalPathTrace::Now();
  2402                      criticalPathTrace_.EndExpertStage1(gmm2PrologueEndCycle);
```

### `a8w4_combine_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_trace.h:559` in `RecordA8W4CombineBeginAfterReady`; `before` target line `560`.
- Cardinality: first ready combine tile/group per expert and AIV1
- Boundary claim: `causal` — after GM tile sequence or group counter readiness and before first combine load
- Why: measure Combine only after GMM2 output is visible
- Dependency: GMM2 tile/group ready
- Pipeline proof: `Scalar polling` → `Scalar` using `none` — both inline and quantized call sites invoke the helper after their ready wait
- Perturbation: one clock read
- Source SHA-256: `fb88e0a8f06bda4ab3dfdc5874e6145ced5c12f6cf7ccbd8d9fd68bc82828bc8`

```text
   557  {
   558      if (trace != nullptr && trace->NeedsExpertStage2Begin()) {
   559          // TRACE_POINT: a8w4_combine_begin_S
   560          trace->BeginExpertStage2(CriticalPathTrace::Now());
   561      }
```

### `a8w4_combine_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:2409` in `MegaMoe::Process`; `before` target line `2410`.
- Cardinality: last combine send per routed expert and AIV1
- Boundary claim: `actual` — the last MTE3 completion dependency has reached Scalar
- Why: measure completed remote combine writes
- Dependency: GroupMatmulWithCombine returned
- Pipeline proof: `MTE3 transferred to MTE2` → `Scalar` using `MTE2_S EVENT_ID0` — AivGmm2Post transfers every tile completion to Scalar before returning; the outer bridge closes the aggregate interval
- Perturbation: one aggregate completion bridge at expert end
- Source SHA-256: `c13170e8e9c5eaecf9a5ab03606d522bd39febdea7da7eba5f54756f9e2d0e11`

```text
  2407                      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
  2408                      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
  2409                      // TRACE_POINT: a8w4_combine_end_S
  2410                      uint32_t combineEndCycle = CriticalPathTrace::Now();
  2411                      criticalPathTrace_.EndExpertStage2(combineEndCycle);
```

### `a8w4_combine_tile_wait_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1388` in `AivGmm2PostA8W4`; `before` target line `1389`.
- Cardinality: before each owned Combine tile readiness poll
- Boundary claim: `causal` — immediately before polling the next GMM2 tile sequence
- Why: measure the exact per-tile GMM2-to-Combine wait
- Dependency: the scheduler selected the next owned tile
- Pipeline proof: `Scalar` → `Scalar` using `none` — the sample directly precedes the gmmToEpilogueFlag polling loop
- Perturbation: one clock read per owned Combine tile
- Source SHA-256: `d8be4e6f4d0b93322e4438f3b8d86c9e00c1ef1f7e8924c5df455c685830e53c`

```text
  1386  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1387          if (args.criticalPathTrace != nullptr) {
  1388              // TRACE_POINT: a8w4_combine_tile_wait_begin_S
  1389              args.criticalPathTrace->BeginCombineTileWait(
  1390                  args.expertIdx, tileOrdinal, CriticalPathTrace::Now());
```

### `a8w4_combine_tile_wait_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1401` in `AivGmm2PostA8W4`; `before` target line `1403`.
- Cardinality: after each owned Combine tile becomes ready
- Boundary claim: `actual` — the tile sequence has reached expectedReadySequence
- Why: close the exact per-tile readiness wait
- Dependency: gmmToEpilogueFlag reached expectedReadySequence
- Pipeline proof: `Scalar polling` → `Scalar` using `none` — the shared ready sample directly follows the readiness loop
- Perturbation: shares the tile-begin clock read
- Source SHA-256: `d8be4e6f4d0b93322e4438f3b8d86c9e00c1ef1f7e8924c5df455c685830e53c`

```text
  1399          RecordA8W4CombineBeginAfterReady(args.criticalPathTrace);
  1400          if (args.criticalPathTrace != nullptr) {
  1401              // TRACE_POINT: a8w4_combine_tile_wait_end_S
  1402              // TRACE_POINT: a8w4_combine_tile_begin_S
  1403              args.criticalPathTrace->BeginCombineTile(
```

### `a8w4_combine_tile_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1402` in `AivGmm2PostA8W4`; `before` target line `1403`.
- Cardinality: after each owned Combine tile becomes ready
- Boundary claim: `causal` — the tile sequence is visible and GM-to-UB movement is next
- Why: separate GMM2 wait from Combine work
- Dependency: gmmToEpilogueFlag reached expectedReadySequence
- Pipeline proof: `Scalar polling` → `Scalar` using `none` — the shared ready sample directly precedes Combine work
- Perturbation: one clock read per owned Combine tile shared with wait end
- Source SHA-256: `d8be4e6f4d0b93322e4438f3b8d86c9e00c1ef1f7e8924c5df455c685830e53c`

```text
  1400          if (args.criticalPathTrace != nullptr) {
  1401              // TRACE_POINT: a8w4_combine_tile_wait_end_S
  1402              // TRACE_POINT: a8w4_combine_tile_begin_S
  1403              args.criticalPathTrace->BeginCombineTile(
  1404                  args.expertIdx, tileOrdinal, CriticalPathTrace::Now());
```

### `a8w4_combine_tile_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1434` in `AivGmm2PostA8W4`; `before` target line `1435`.
- Cardinality: after each owned Combine tile send completes
- Boundary claim: `actual` — the tile MTE3 completion dependency has reached Scalar
- Why: measure per-tile Combine work through actual remote write completion
- Dependency: CombineTokens returned
- Pipeline proof: `MTE3 transferred to MTE2` → `Scalar` using `MTE3_MTE2 followed by MTE2_S EVENT_ID0` — the timestamp follows both completion bridges
- Perturbation: one MTE2-to-Scalar bridge and one clock read per owned Combine tile
- Source SHA-256: `d8be4e6f4d0b93322e4438f3b8d86c9e00c1ef1f7e8924c5df455c685830e53c`

```text
  1432              AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
  1433              AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
  1434              // TRACE_POINT: a8w4_combine_tile_end_S
  1435              args.criticalPathTrace->EndCombineTile(
  1436                  args.expertIdx, tileOrdinal, CriticalPathTrace::Now());
```

## Working tree status

```text
M mc2/mega_moe/op_kernel/arch35/mega_moe.h
 M mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h
 M mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h
 M mc2/mega_moe/op_kernel/arch35/mega_moe_trace.h
 M mc2/mega_moe/tools/critical_path_trace/README.md
 M mc2/mega_moe/tools/critical_path_trace/audit-implemented-a8w4/trace_point_audit.json
 M mc2/mega_moe/tools/critical_path_trace/audit-implemented-a8w4/trace_point_audit.md
 M mc2/mega_moe/tools/critical_path_trace/mega_moe_trace_contract.py
 M mc2/mega_moe/tools/critical_path_trace/run_critical_path_trace.py
 M mc2/mega_moe/tools/critical_path_trace/test_critical_path_trace.py
 M mc2/mega_moe/tools/critical_path_trace/trace-spec-a8w4.json
 M mc2/mega_moe/tools/critical_path_trace/trace_tool.py
```
