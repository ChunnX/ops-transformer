# Trace point audit

- Status: `pass`
- Phase: `implemented`
- Trace: `mega-moe-a8w4-critical-path`
- Repository: `/home/oiseau/codes/ops-transformer`
- Revision: `b33ab9697975f9cd441b014dbb847dae3afe4f86`
- Branch: `codex/mega-moe-a8w4-critical-path-trace`
- Dirty: `True`
- Spec SHA-256: `5d71b53535e86085f4106187c1c2584b48b2c5c3f407d1deace761cb73a16649`

## Placement summary

| Point | Roles | Resolved location | Target line | Placement | Boundary | Bridge | Why |
|---|---|---|---:|---|---|---|---|
| a8w4_process_begin_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe.h:2204 | 2204 | before | causal | none | establish the local process epoch |
| a8w4_work_end_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe.h:2454 | 2455 | before | actual | AIV uses V_S EVENT_ID0; AIC reuses existing FIX_S publication | separate useful work from trace emission |
| a8w4_dispatch_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe.h:299 | 300 | before | causal | none | measure dispatch prefetch placement relative to A8W4 compute |
| a8w4_dispatch_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe.h:310 | 311 | before | actual | MTE3_S EVENT_ID0 | prove dispatch GM writes complete |
| a8w4_gmm1_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1154 | 1155 | before | causal | none | locate AIC GMM1 after dispatch readiness |
| a8w4_gmm1_end_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe.h:2310 | 2311 | before | actual | existing per-tile FIX_S publication | measure actual AIC GMM1 completion |
| a8w4_gmm1_prologue_begin_S | AIV0 | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1236 | 1237 | before | causal | none | expose W4-to-W8 prologue occupancy |
| a8w4_gmm1_prologue_end_S | AIV0 | mc2/mega_moe/op_kernel/arch35/mega_moe.h:2319 | 2320 | before | actual | MTE3_S EVENT_ID0 | measure completed W4-to-W8 prologue work |
| a8w4_swiglu_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_trace.h:629 | 630 | before | causal | none | measure AIV1 epilogue after tile readiness |
| a8w4_swiglu_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe.h:2326 | 2327 | before | actual | existing per-call MTE3_S | measure completed SwiGLU and MX quant output |
| a8w4_gmm2_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1159 | 1160 | before | causal | none | locate GMM2 after SwiGLU readiness |
| a8w4_gmm2_end_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe.h:2400 | 2401 | before | actual | existing per-tile FIX_S publication | measure actual AIC GMM2 completion/publication |
| a8w4_gmm2_prologue_begin_S | AIV0 | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1241 | 1242 | before | causal | none | expose GMM2 W4-to-W8 prologue occupancy |
| a8w4_gmm2_prologue_end_S | AIV0 | mc2/mega_moe/op_kernel/arch35/mega_moe.h:2409 | 2410 | before | actual | MTE3_S EVENT_ID0 | measure completed W4-to-W8 prologue work |
| a8w4_combine_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_trace.h:637 | 638 | before | causal | none | measure Combine only after GMM2 output is visible |
| a8w4_combine_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe.h:2418 | 2419 | before | actual | MTE2_S EVENT_ID0 | measure completed remote combine writes |
| a8w4_combine_tile_wait_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1418 | 1419 | before | causal | none | measure the exact per-tile GMM2-to-Combine wait |
| a8w4_combine_tile_wait_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1431 | 1433 | before | actual | none | close the exact per-tile readiness wait |
| a8w4_combine_tile_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1432 | 1433 | before | causal | none | separate GMM2 wait from Combine work |
| a8w4_combine_tile_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1464 | 1465 | before | actual | MTE3_MTE2 followed by MTE2_S EVENT_ID0 | measure per-tile Combine work through actual remote write completion |
| a8w4_gmm1_wave_wait_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1128 | 1129 | before | causal | none | measure Dispatch-to-GMM1 readiness wait per M-wave |
| a8w4_gmm1_wave_wait_end_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1144 | 1146 | before | actual | none | close the retained per-wave Dispatch readiness wait |
| a8w4_gmm1_wave_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1145 | 1146 | before | causal | none | start the GMM1 M-wave active envelope after readiness |
| a8w4_gmm1_wave_end_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1213 | 1214 | before | actual | existing per-tile FIX_S | close the M-wave active envelope at its latest published tile |

## Point evidence

### `a8w4_process_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:2204` in `MegaMoe::Process`; `before` target line `2204`.
- Cardinality: once per participating physical role
- Boundary claim: `causal` — entry to the MTE A8W4 process before useful work
- Why: establish the local process epoch
- Dependency: kernel entry
- Pipeline proof: `Scalar` → `Scalar` using `none` — the clock is the first trace action in Process
- Perturbation: one clock read
- Source SHA-256: `39b1ed544381a31d51e312fc7c53a0f745e67e270a412ff0322a9060fef17407`

```text
  2202  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  2203      if (CriticalPathTracePtr() != nullptr) {
  2204          criticalPathTrace_.processBegin = CriticalPathTrace::Now(); // TRACE_POINT: a8w4_process_begin_S
  2205      }
  2206  #endif
```

### `a8w4_work_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:2454` in `MegaMoe::Process`; `before` target line `2455`.
- Cardinality: once per participating physical role
- Boundary claim: `actual` — all role-local useful work has completed
- Why: separate useful work from trace emission
- Dependency: AIV Unpermute drain or AIC final compute return
- Pipeline proof: `mixed` → `Scalar` using `AIV uses V_S EVENT_ID0; AIC reuses existing FIX_S publication` — Unpermute transfers its final MTE3 completion to Vector through MTE3_V, then the trace-only V_S transfers it to Scalar; AIC compute returns after FIX_S publication
- Perturbation: AIV adds one V_S completion bridge; all roles add one clock read
- Source SHA-256: `39b1ed544381a31d51e312fc7c53a0f745e67e270a412ff0322a9060fef17407`

```text
  2452              AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
  2453          }
  2454          // TRACE_POINT: a8w4_work_end_S
  2455          criticalPathTrace_.workEnd = CriticalPathTrace::Now();
  2456          SyncAll<false>();
```

### `a8w4_dispatch_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:299` in `MegaMoe::BeginDispatchTrace`; `before` target line `300`.
- Cardinality: once per routed expert on an active AIV1
- Boundary claim: `causal` — before SendCntCal and dispatch metadata/token movement for the expert
- Why: measure dispatch prefetch placement relative to A8W4 compute
- Dependency: the caller selected the next expert
- Pipeline proof: `Scalar` → `Scalar` using `none` — both dispatch call sites invoke this helper immediately before SendCntCal
- Perturbation: one clock read
- Source SHA-256: `39b1ed544381a31d51e312fc7c53a0f745e67e270a412ff0322a9060fef17407`

```text
   297      if (CriticalPathTracePtr() != nullptr) {
   298          criticalPathTrace_.PrepareExpertStage0(expertIdx);
   299          // TRACE_POINT: a8w4_dispatch_begin_S
   300          criticalPathTrace_.BeginExpertStage0(CriticalPathTrace::Now());
   301      }
```

### `a8w4_dispatch_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:310` in `MegaMoe::EndDispatchTrace`; `before` target line `311`.
- Cardinality: once per routed expert on an active AIV1
- Boundary claim: `actual` — the expert dispatch MTE3 writes have completed
- Why: prove dispatch GM writes complete
- Dependency: MetaInfoCalAndDispatch returned
- Pipeline proof: `MTE3` → `Scalar` using `MTE3_S EVENT_ID0` — the clock follows the trace-only MTE3_S set/wait
- Perturbation: forces the expert dispatch tail to complete
- Source SHA-256: `39b1ed544381a31d51e312fc7c53a0f745e67e270a412ff0322a9060fef17407`

```text
   308          AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
   309          AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
   310          // TRACE_POINT: a8w4_dispatch_end_S
   311          uint32_t dispatchEndCycle = CriticalPathTrace::Now();
   312          criticalPathTrace_.EndExpertStage0(dispatchEndCycle);
```

### `a8w4_gmm1_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1154` in `AicComputeA8W4`; `before` target line `1155`.
- Cardinality: first owned GMM1 tile per routed expert and AIC
- Boundary claim: `causal` — after WaitForUpstreamReady and before the first blockMmad
- Why: locate AIC GMM1 after dispatch readiness
- Dependency: dispatch wave ready
- Pipeline proof: `Scalar polling` → `Scalar` using `none` — the clock is after WaitForUpstreamReady in the tile loop
- Perturbation: one clock read
- Source SHA-256: `3bab7cecaab907a1229fccdc5d46634784abb951a89608ee63c50ccc634b6698`

```text
  1152          if constexpr (Policy::IS_GMM1) {
  1153              if (criticalPathTrace != nullptr && criticalPathTrace->NeedsExpertStage0Begin()) {
  1154                  // TRACE_POINT: a8w4_gmm1_begin_S
  1155                  uint32_t a8w4Gmm1BeginCycle = CriticalPathTrace::Now();
  1156                  criticalPathTrace->BeginExpertStage0(a8w4Gmm1BeginCycle);
```

### `a8w4_gmm1_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:2310` in `MegaMoe::Process`; `before` target line `2311`.
- Cardinality: last owned GMM1 tile per routed expert and AIC
- Boundary claim: `actual` — all owned GMM1 tiles have been published after FIX_S
- Why: measure actual AIC GMM1 completion
- Dependency: AicComputeA8W4 returned
- Pipeline proof: `FIX` → `Scalar` using `existing per-tile FIX_S publication` — every specialized tile publishes only after FIX_S before the function returns
- Perturbation: one clock read
- Source SHA-256: `39b1ed544381a31d51e312fc7c53a0f745e67e270a412ff0322a9060fef17407`

```text
  2308                  if (criticalPathTrace_.HasExpertStage0Work()) {
  2309                      // Every A8W4 tile publishes only after FIX_S, so return is an actual GMM1 completion boundary.
  2310                      // TRACE_POINT: a8w4_gmm1_end_S
  2311                      uint32_t gmm1EndCycle = CriticalPathTrace::Now();
  2312                      criticalPathTrace_.EndExpertStage0(gmm1EndCycle);
```

### `a8w4_gmm1_prologue_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1236` in `AivPrologueA8W4`; `before` target line `1237`.
- Cardinality: first owned GMM1 weight tile per expert and AIV0
- Boundary claim: `causal` — before the first GMM1 blockPrologue issue
- Why: expose W4-to-W8 prologue occupancy
- Dependency: the expert prologue loop is entered
- Pipeline proof: `Scalar` → `Scalar` using `none` — the sample precedes the first blockPrologue call
- Perturbation: one clock read
- Source SHA-256: `3bab7cecaab907a1229fccdc5d46634784abb951a89608ee63c50ccc634b6698`

```text
  1234          if constexpr (Policy::IS_GMM1) {
  1235              if (args.criticalPathTrace != nullptr && args.criticalPathTrace->NeedsExpertStage0Begin()) {
  1236                  // TRACE_POINT: a8w4_gmm1_prologue_begin_S
  1237                  uint32_t a8w4Gmm1PrologueBeginCycle = CriticalPathTrace::Now();
  1238                  args.criticalPathTrace->BeginExpertStage0(a8w4Gmm1PrologueBeginCycle);
```

### `a8w4_gmm1_prologue_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:2319` in `MegaMoe::Process`; `before` target line `2320`.
- Cardinality: after the last owned GMM1 weight tile completes
- Boundary claim: `actual` — all owned GMM1 prologue MTE3 writes to L1 have completed
- Why: measure completed W4-to-W8 prologue work
- Dependency: AivPrologueA8W4 and BlockPrologue destruction returned
- Pipeline proof: `MTE3` → `Scalar` using `MTE3_S EVENT_ID0` — the clock follows a trace-only MTE3_S set/wait after the prologue event rings are reclaimed
- Perturbation: one MTE3_S completion bridge and one clock read per active expert
- Source SHA-256: `39b1ed544381a31d51e312fc7c53a0f745e67e270a412ff0322a9060fef17407`

```text
  2317                      AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
  2318                      AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
  2319                      // TRACE_POINT: a8w4_gmm1_prologue_end_S
  2320                      uint32_t gmm1PrologueEndCycle = CriticalPathTrace::Now();
  2321                      criticalPathTrace_.EndExpertStage0(gmm1PrologueEndCycle);
```

### `a8w4_swiglu_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_trace.h:629` in `RecordA8W4SwiGluBeginAfterReady`; `before` target line `630`.
- Cardinality: first ready GMM1 output tile per expert and AIV1
- Boundary claim: `causal` — after status/sequence polling and before the first GMM1-output load
- Why: measure AIV1 epilogue after tile readiness
- Dependency: AIC published the tile-ready state
- Pipeline proof: `Scalar polling` → `Scalar` using `none` — both call sites invoke this helper only after their ready loop
- Perturbation: one clock read
- Source SHA-256: `b89e25d85aeab70d037809e190ae8f97cdd697a3c27ec73a640f79c21658cef4`

```text
   627  {
   628      if (trace != nullptr && trace->NeedsExpertStage1Begin()) {
   629          // TRACE_POINT: a8w4_swiglu_begin_S
   630          trace->BeginExpertStage1(CriticalPathTrace::Now());
   631      }
```

### `a8w4_swiglu_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:2326` in `MegaMoe::Process`; `before` target line `2327`.
- Cardinality: after the last owned SwiGLU tile completes per expert
- Boundary claim: `actual` — the final SwiGLU data/scale MTE3 writes and ready publication have completed
- Why: measure completed SwiGLU and MX quant output
- Dependency: AivGmm1PostA8W4 returned
- Pipeline proof: `MTE3/Scalar` → `Scalar` using `existing per-call MTE3_S` — BlockEpilogueSwigluMxQuant returns only after MTE3_S and its Scalar ready AtomicAdd; the outer clock follows the final call
- Perturbation: one clock read; no additional completion bridge
- Source SHA-256: `39b1ed544381a31d51e312fc7c53a0f745e67e270a412ff0322a9060fef17407`

```text
  2324              } else {
  2325                  if (criticalPathTrace_.HasExpertStage1Work()) {
  2326                      // TRACE_POINT: a8w4_swiglu_end_S
  2327                      uint32_t swigluEndCycle = CriticalPathTrace::Now();
  2328                      criticalPathTrace_.EndExpertStage1(swigluEndCycle);
```

### `a8w4_gmm2_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1159` in `AicComputeA8W4`; `before` target line `1160`.
- Cardinality: first owned GMM2 tile per routed expert and AIC
- Boundary claim: `causal` — after WaitForUpstreamReady and before first GMM2 blockMmad
- Why: locate GMM2 after SwiGLU readiness
- Dependency: SwiGLU output ready
- Pipeline proof: `Scalar polling` → `Scalar` using `none` — the sample follows the upstream wait
- Perturbation: one clock read
- Source SHA-256: `3bab7cecaab907a1229fccdc5d46634784abb951a89608ee63c50ccc634b6698`

```text
  1157              }
  1158          } else if (criticalPathTrace != nullptr && criticalPathTrace->NeedsExpertStage1Begin()) {
  1159              // TRACE_POINT: a8w4_gmm2_begin_S
  1160              uint32_t a8w4Gmm2BeginCycle = CriticalPathTrace::Now();
  1161              criticalPathTrace->BeginExpertStage1(a8w4Gmm2BeginCycle);
```

### `a8w4_gmm2_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:2400` in `MegaMoe::Process`; `before` target line `2401`.
- Cardinality: last owned GMM2 tile per routed expert and AIC
- Boundary claim: `actual` — all owned GMM2 tiles have passed FIX_S publication
- Why: measure actual AIC GMM2 completion/publication
- Dependency: AicComputeA8W4 returned
- Pipeline proof: `FIX` → `Scalar` using `existing per-tile FIX_S publication` — tile completion notification or GM epilogue publication follows FIX_S
- Perturbation: one clock read
- Source SHA-256: `39b1ed544381a31d51e312fc7c53a0f745e67e270a412ff0322a9060fef17407`

```text
  2398                  if (criticalPathTrace_.HasExpertStage1Work()) {
  2399                      // Every A8W4 tile publishes only after FIX_S, so return is an actual GMM2 completion boundary.
  2400                      // TRACE_POINT: a8w4_gmm2_end_S
  2401                      uint32_t gmm2EndCycle = CriticalPathTrace::Now();
  2402                      criticalPathTrace_.EndExpertStage1(gmm2EndCycle);
```

### `a8w4_gmm2_prologue_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1241` in `AivPrologueA8W4`; `before` target line `1242`.
- Cardinality: first owned GMM2 weight tile per expert and AIV0
- Boundary claim: `causal` — before first GMM2 blockPrologue issue
- Why: expose GMM2 W4-to-W8 prologue occupancy
- Dependency: the expert prologue loop is entered
- Pipeline proof: `Scalar` → `Scalar` using `none` — the sample precedes blockPrologue
- Perturbation: one clock read
- Source SHA-256: `3bab7cecaab907a1229fccdc5d46634784abb951a89608ee63c50ccc634b6698`

```text
  1239              }
  1240          } else if (args.criticalPathTrace != nullptr && args.criticalPathTrace->NeedsExpertStage1Begin()) {
  1241              // TRACE_POINT: a8w4_gmm2_prologue_begin_S
  1242              uint32_t a8w4Gmm2PrologueBeginCycle = CriticalPathTrace::Now();
  1243              args.criticalPathTrace->BeginExpertStage1(a8w4Gmm2PrologueBeginCycle);
```

### `a8w4_gmm2_prologue_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:2409` in `MegaMoe::Process`; `before` target line `2410`.
- Cardinality: after the last owned GMM2 weight tile completes
- Boundary claim: `actual` — all owned GMM2 prologue MTE3 writes to L1 have completed
- Why: measure completed W4-to-W8 prologue work
- Dependency: AivPrologueA8W4 and BlockPrologue destruction returned
- Pipeline proof: `MTE3` → `Scalar` using `MTE3_S EVENT_ID0` — the clock follows a trace-only MTE3_S set/wait after the prologue event rings are reclaimed
- Perturbation: one MTE3_S completion bridge and one clock read per active expert
- Source SHA-256: `39b1ed544381a31d51e312fc7c53a0f745e67e270a412ff0322a9060fef17407`

```text
  2407                      AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
  2408                      AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
  2409                      // TRACE_POINT: a8w4_gmm2_prologue_end_S
  2410                      uint32_t gmm2PrologueEndCycle = CriticalPathTrace::Now();
  2411                      criticalPathTrace_.EndExpertStage1(gmm2PrologueEndCycle);
```

### `a8w4_combine_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_trace.h:637` in `RecordA8W4CombineBeginAfterReady`; `before` target line `638`.
- Cardinality: first ready combine tile/group per expert and AIV1
- Boundary claim: `causal` — after GM tile sequence or group counter readiness and before first combine load
- Why: measure Combine only after GMM2 output is visible
- Dependency: GMM2 tile/group ready
- Pipeline proof: `Scalar polling` → `Scalar` using `none` — both inline and quantized call sites invoke the helper after their ready wait
- Perturbation: one clock read
- Source SHA-256: `b89e25d85aeab70d037809e190ae8f97cdd697a3c27ec73a640f79c21658cef4`

```text
   635  {
   636      if (trace != nullptr && trace->NeedsExpertStage2Begin()) {
   637          // TRACE_POINT: a8w4_combine_begin_S
   638          trace->BeginExpertStage2(CriticalPathTrace::Now());
   639      }
```

### `a8w4_combine_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:2418` in `MegaMoe::Process`; `before` target line `2419`.
- Cardinality: last combine send per routed expert and AIV1
- Boundary claim: `actual` — the last MTE3 completion dependency has reached Scalar
- Why: measure completed remote combine writes
- Dependency: GroupMatmulWithCombine returned
- Pipeline proof: `MTE3 transferred to MTE2` → `Scalar` using `MTE2_S EVENT_ID0` — AivGmm2Post transfers every tile completion to Scalar before returning; the outer bridge closes the aggregate interval
- Perturbation: one aggregate completion bridge at expert end
- Source SHA-256: `39b1ed544381a31d51e312fc7c53a0f745e67e270a412ff0322a9060fef17407`

```text
  2416                      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
  2417                      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
  2418                      // TRACE_POINT: a8w4_combine_end_S
  2419                      uint32_t combineEndCycle = CriticalPathTrace::Now();
  2420                      criticalPathTrace_.EndExpertStage2(combineEndCycle);
```

### `a8w4_combine_tile_wait_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1418` in `AivGmm2PostA8W4`; `before` target line `1419`.
- Cardinality: before each owned Combine tile readiness poll
- Boundary claim: `causal` — immediately before polling the next GMM2 tile sequence
- Why: measure the exact per-tile GMM2-to-Combine wait
- Dependency: the scheduler selected the next owned tile
- Pipeline proof: `Scalar` → `Scalar` using `none` — the sample directly precedes the gmmToEpilogueFlag polling loop
- Perturbation: one clock read per owned Combine tile
- Source SHA-256: `3bab7cecaab907a1229fccdc5d46634784abb951a89608ee63c50ccc634b6698`

```text
  1416  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1417          if (args.criticalPathTrace != nullptr) {
  1418              // TRACE_POINT: a8w4_combine_tile_wait_begin_S
  1419              args.criticalPathTrace->BeginCombineTileWait(
  1420                  args.expertIdx, tileOrdinal, CriticalPathTrace::Now());
```

### `a8w4_combine_tile_wait_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1431` in `AivGmm2PostA8W4`; `before` target line `1433`.
- Cardinality: after each owned Combine tile becomes ready
- Boundary claim: `actual` — the tile sequence has reached expectedReadySequence
- Why: close the exact per-tile readiness wait
- Dependency: gmmToEpilogueFlag reached expectedReadySequence
- Pipeline proof: `Scalar polling` → `Scalar` using `none` — the shared ready sample directly follows the readiness loop
- Perturbation: shares the tile-begin clock read
- Source SHA-256: `3bab7cecaab907a1229fccdc5d46634784abb951a89608ee63c50ccc634b6698`

```text
  1429          RecordA8W4CombineBeginAfterReady(args.criticalPathTrace);
  1430          if (args.criticalPathTrace != nullptr) {
  1431              // TRACE_POINT: a8w4_combine_tile_wait_end_S
  1432              // TRACE_POINT: a8w4_combine_tile_begin_S
  1433              args.criticalPathTrace->BeginCombineTile(
```

### `a8w4_combine_tile_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1432` in `AivGmm2PostA8W4`; `before` target line `1433`.
- Cardinality: after each owned Combine tile becomes ready
- Boundary claim: `causal` — the tile sequence is visible and GM-to-UB movement is next
- Why: separate GMM2 wait from Combine work
- Dependency: gmmToEpilogueFlag reached expectedReadySequence
- Pipeline proof: `Scalar polling` → `Scalar` using `none` — the shared ready sample directly precedes Combine work
- Perturbation: one clock read per owned Combine tile shared with wait end
- Source SHA-256: `3bab7cecaab907a1229fccdc5d46634784abb951a89608ee63c50ccc634b6698`

```text
  1430          if (args.criticalPathTrace != nullptr) {
  1431              // TRACE_POINT: a8w4_combine_tile_wait_end_S
  1432              // TRACE_POINT: a8w4_combine_tile_begin_S
  1433              args.criticalPathTrace->BeginCombineTile(
  1434                  args.expertIdx, tileOrdinal, CriticalPathTrace::Now());
```

### `a8w4_combine_tile_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1464` in `AivGmm2PostA8W4`; `before` target line `1465`.
- Cardinality: after each owned Combine tile send completes
- Boundary claim: `actual` — the tile MTE3 completion dependency has reached Scalar
- Why: measure per-tile Combine work through actual remote write completion
- Dependency: CombineTokens returned
- Pipeline proof: `MTE3 transferred to MTE2` → `Scalar` using `MTE3_MTE2 followed by MTE2_S EVENT_ID0` — the timestamp follows both completion bridges
- Perturbation: one MTE2-to-Scalar bridge and one clock read per owned Combine tile
- Source SHA-256: `3bab7cecaab907a1229fccdc5d46634784abb951a89608ee63c50ccc634b6698`

```text
  1462              AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
  1463              AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
  1464              // TRACE_POINT: a8w4_combine_tile_end_S
  1465              args.criticalPathTrace->EndCombineTile(
  1466                  args.expertIdx, tileOrdinal, CriticalPathTrace::Now());
```

### `a8w4_gmm1_wave_wait_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1128` in `AicComputeA8W4`; `before` target line `1129`.
- Cardinality: first readiness visit for each expert M-wave on an active AIC
- Boundary claim: `causal` — immediately before the first readiness poll retained for this expert and M-wave
- Why: measure Dispatch-to-GMM1 readiness wait per M-wave
- Dependency: the swizzled scheduler selected a tile in the M-wave
- Pipeline proof: `Scalar` → `Scalar` using `none` — the sample directly precedes WaitForUpstreamReady
- Perturbation: one retained clock read per active expert M-wave
- Source SHA-256: `3bab7cecaab907a1229fccdc5d46634784abb951a89608ee63c50ccc634b6698`

```text
  1126              if (shouldWait && criticalPathTrace != nullptr &&
  1127                  criticalPathTrace->NeedsGmm1WaveWait(expertIdx, waveIdx)) {
  1128                  // TRACE_POINT: a8w4_gmm1_wave_wait_begin_S
  1129                  criticalPathTrace->BeginGmm1WaveWait(
  1130                      expertIdx, waveIdx, CriticalPathTrace::Now());
```

### `a8w4_gmm1_wave_wait_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1144` in `AicComputeA8W4`; `before` target line `1146`.
- Cardinality: after the retained readiness wait for each expert M-wave
- Boundary claim: `actual` — the Dispatch row-count condition for this wave is satisfied
- Why: close the retained per-wave Dispatch readiness wait
- Dependency: WaitForUpstreamReady returned
- Pipeline proof: `Scalar polling` → `Scalar` using `none` — the shared sample directly follows the wait
- Perturbation: shares the wave-envelope begin clock read
- Source SHA-256: `3bab7cecaab907a1229fccdc5d46634784abb951a89608ee63c50ccc634b6698`

```text
  1142              if (shouldWait && criticalPathTrace != nullptr &&
  1143                  criticalPathTrace->NeedsGmm1WaveBegin(expertIdx, waveIdx)) {
  1144                  // TRACE_POINT: a8w4_gmm1_wave_wait_end_S
  1145                  // TRACE_POINT: a8w4_gmm1_wave_begin_S
  1146                  criticalPathTrace->BeginGmm1Wave(
```

### `a8w4_gmm1_wave_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1145` in `AicComputeA8W4`; `before` target line `1146`.
- Cardinality: first owned tile retained for each expert M-wave and AIC
- Boundary claim: `causal` — the first retained tile in this M-wave is about to issue MMAD
- Why: start the GMM1 M-wave active envelope after readiness
- Dependency: Dispatch readiness is satisfied
- Pipeline proof: `Scalar` → `Scalar` using `none` — the shared sample precedes the first retained blockMmad
- Perturbation: shares the wait-end clock read
- Source SHA-256: `3bab7cecaab907a1229fccdc5d46634784abb951a89608ee63c50ccc634b6698`

```text
  1143                  criticalPathTrace->NeedsGmm1WaveBegin(expertIdx, waveIdx)) {
  1144                  // TRACE_POINT: a8w4_gmm1_wave_wait_end_S
  1145                  // TRACE_POINT: a8w4_gmm1_wave_begin_S
  1146                  criticalPathTrace->BeginGmm1Wave(
  1147                      expertIdx, waveIdx, CriticalPathTrace::Now());
```

### `a8w4_gmm1_wave_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1213` in `AicComputeA8W4`; `before` target line `1214`.
- Cardinality: latest completed owned tile for each expert M-wave and AIC
- Boundary claim: `actual` — the latest owned tile observed for this wave has passed FIX_S publication; the envelope may include interleaved swizzle work from other waves
- Why: close the M-wave active envelope at its latest published tile
- Dependency: the current GMM1 tile has been published
- Pipeline proof: `FIX` → `Scalar` using `existing per-tile FIX_S` — the sample follows the existing FIX_S and ready publication
- Perturbation: one clock read per owned GMM1 tile; only the latest value is retained
- Source SHA-256: `3bab7cecaab907a1229fccdc5d46634784abb951a89608ee63c50ccc634b6698`

```text
  1211          if constexpr (Policy::IS_GMM1) {
  1212              if (criticalPathTrace != nullptr) {
  1213                  // TRACE_POINT: a8w4_gmm1_wave_end_S
  1214                  criticalPathTrace->EndGmm1Wave(
  1215                      expertIdx, waveIdx, CriticalPathTrace::Now());
```

## Working tree status

```text
M mc2/mega_moe/op_kernel/arch35/mega_moe.h
 M mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h
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
