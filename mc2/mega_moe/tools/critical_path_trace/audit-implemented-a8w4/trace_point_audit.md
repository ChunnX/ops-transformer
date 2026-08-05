# Trace point audit

- Status: `pass`
- Phase: `implemented`
- Trace: `mega-moe-a8w4-critical-path`
- Repository: `/home/oiseau/codes/ops-transformer`
- Revision: `a2d605e95214d90b538d1e654953e244568e22f9`
- Branch: `codex/mega-moe-a8w4-critical-path-trace`
- Dirty: `True`
- Spec SHA-256: `82497757c2ddd1c37721380221ff782d8a1644e4b899683b21b3c16b97aadd22`

## Placement summary

| Point | Roles | Resolved location | Target line | Placement | Boundary | Bridge | Why |
|---|---|---|---:|---|---|---|---|
| a8w4_process_begin_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe.h:2179 | 2179 | before | causal | none | establish the local process epoch |
| a8w4_work_end_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe.h:2429 | 2430 | before | actual | AIV uses V_S EVENT_ID0; AIC reuses existing FIX_S publication | separate useful work from trace emission |
| a8w4_dispatch_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe.h:295 | 296 | before | causal | none | measure dispatch prefetch placement relative to A8W4 compute |
| a8w4_dispatch_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe.h:306 | 307 | before | actual | MTE3_S EVENT_ID0 | prove dispatch GM writes complete |
| a8w4_gmm1_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1132 | 1133 | before | causal | none | locate AIC GMM1 after dispatch readiness |
| a8w4_gmm1_end_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe.h:2285 | 2286 | before | actual | existing per-tile FIX_S publication | measure actual AIC GMM1 completion |
| a8w4_gmm1_prologue_begin_S | AIV0 | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1205 | 1206 | before | causal | none | expose W4-to-W8 prologue occupancy |
| a8w4_gmm1_prologue_end_S | AIV0 | mc2/mega_moe/op_kernel/arch35/mega_moe.h:2294 | 2295 | before | actual | MTE3_S EVENT_ID0 | measure completed W4-to-W8 prologue work |
| a8w4_swiglu_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_trace.h:484 | 485 | before | causal | none | measure AIV1 epilogue after tile readiness |
| a8w4_swiglu_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe.h:2301 | 2302 | before | actual | existing per-call MTE3_S | measure completed SwiGLU and MX quant output |
| a8w4_gmm2_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1137 | 1138 | before | causal | none | locate GMM2 after SwiGLU readiness |
| a8w4_gmm2_end_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe.h:2375 | 2376 | before | actual | existing per-tile FIX_S publication | measure actual AIC GMM2 completion/publication |
| a8w4_gmm2_prologue_begin_S | AIV0 | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1210 | 1211 | before | causal | none | expose GMM2 W4-to-W8 prologue occupancy |
| a8w4_gmm2_prologue_end_S | AIV0 | mc2/mega_moe/op_kernel/arch35/mega_moe.h:2384 | 2385 | before | actual | MTE3_S EVENT_ID0 | measure completed W4-to-W8 prologue work |
| a8w4_combine_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_trace.h:492 | 493 | before | causal | none | measure Combine only after GMM2 output is visible |
| a8w4_combine_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe.h:2393 | 2394 | before | actual | MTE2_S EVENT_ID0 | measure completed remote combine writes |

## Point evidence

### `a8w4_process_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:2179` in `MegaMoe::Process`; `before` target line `2179`.
- Cardinality: once per participating physical role
- Boundary claim: `causal` — entry to the MTE A8W4 process before useful work
- Why: establish the local process epoch
- Dependency: kernel entry
- Pipeline proof: `Scalar` → `Scalar` using `none` — the clock is the first trace action in Process
- Perturbation: one clock read
- Source SHA-256: `f13eb7b7b7a2a87387e580b154a3b536ffa11e8cfd3dcb51c1fb8f2392045a8e`

```text
  2177  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  2178      if (CriticalPathTracePtr() != nullptr) {
  2179          criticalPathTrace_.processBegin = CriticalPathTrace::Now(); // TRACE_POINT: a8w4_process_begin_S
  2180      }
  2181  #endif
```

### `a8w4_work_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:2429` in `MegaMoe::Process`; `before` target line `2430`.
- Cardinality: once per participating physical role
- Boundary claim: `actual` — all role-local useful work has completed
- Why: separate useful work from trace emission
- Dependency: AIV Unpermute drain or AIC final compute return
- Pipeline proof: `mixed` → `Scalar` using `AIV uses V_S EVENT_ID0; AIC reuses existing FIX_S publication` — Unpermute transfers its final MTE3 completion to Vector through MTE3_V, then the trace-only V_S transfers it to Scalar; AIC compute returns after FIX_S publication
- Perturbation: AIV adds one V_S completion bridge; all roles add one clock read
- Source SHA-256: `f13eb7b7b7a2a87387e580b154a3b536ffa11e8cfd3dcb51c1fb8f2392045a8e`

```text
  2427              AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
  2428          }
  2429          // TRACE_POINT: a8w4_work_end_S
  2430          criticalPathTrace_.workEnd = CriticalPathTrace::Now();
  2431          SyncAll<false>();
```

### `a8w4_dispatch_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:295` in `MegaMoe::BeginDispatchTrace`; `before` target line `296`.
- Cardinality: once per routed expert on an active AIV1
- Boundary claim: `causal` — before SendCntCal and dispatch metadata/token movement for the expert
- Why: measure dispatch prefetch placement relative to A8W4 compute
- Dependency: the caller selected the next expert
- Pipeline proof: `Scalar` → `Scalar` using `none` — both dispatch call sites invoke this helper immediately before SendCntCal
- Perturbation: one clock read
- Source SHA-256: `f13eb7b7b7a2a87387e580b154a3b536ffa11e8cfd3dcb51c1fb8f2392045a8e`

```text
   293      if (CriticalPathTracePtr() != nullptr) {
   294          criticalPathTrace_.PrepareExpertStage0(expertIdx);
   295          // TRACE_POINT: a8w4_dispatch_begin_S
   296          criticalPathTrace_.BeginExpertStage0(CriticalPathTrace::Now());
   297      }
```

### `a8w4_dispatch_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:306` in `MegaMoe::EndDispatchTrace`; `before` target line `307`.
- Cardinality: once per routed expert on an active AIV1
- Boundary claim: `actual` — the expert dispatch MTE3 writes have completed
- Why: prove dispatch GM writes complete
- Dependency: MetaInfoCalAndDispatch returned
- Pipeline proof: `MTE3` → `Scalar` using `MTE3_S EVENT_ID0` — the clock follows the trace-only MTE3_S set/wait
- Perturbation: forces the expert dispatch tail to complete
- Source SHA-256: `f13eb7b7b7a2a87387e580b154a3b536ffa11e8cfd3dcb51c1fb8f2392045a8e`

```text
   304          AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
   305          AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
   306          // TRACE_POINT: a8w4_dispatch_end_S
   307          uint32_t dispatchEndCycle = CriticalPathTrace::Now();
   308          criticalPathTrace_.EndExpertStage0(dispatchEndCycle);
```

### `a8w4_gmm1_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1132` in `AicComputeA8W4`; `before` target line `1133`.
- Cardinality: first owned GMM1 tile per routed expert and AIC
- Boundary claim: `causal` — after WaitForUpstreamReady and before the first blockMmad
- Why: locate AIC GMM1 after dispatch readiness
- Dependency: dispatch wave ready
- Pipeline proof: `Scalar polling` → `Scalar` using `none` — the clock is after WaitForUpstreamReady in the tile loop
- Perturbation: one clock read
- Source SHA-256: `a2da55d2022a4f250a035781fc62d6b7ef0610b502685cee8e954372f5448bb7`

```text
  1130          if constexpr (Policy::IS_GMM1) {
  1131              if (criticalPathTrace != nullptr && criticalPathTrace->NeedsExpertStage0Begin()) {
  1132                  // TRACE_POINT: a8w4_gmm1_begin_S
  1133                  uint32_t a8w4Gmm1BeginCycle = CriticalPathTrace::Now();
  1134                  criticalPathTrace->BeginExpertStage0(a8w4Gmm1BeginCycle);
```

### `a8w4_gmm1_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:2285` in `MegaMoe::Process`; `before` target line `2286`.
- Cardinality: last owned GMM1 tile per routed expert and AIC
- Boundary claim: `actual` — all owned GMM1 tiles have been published after FIX_S
- Why: measure actual AIC GMM1 completion
- Dependency: AicComputeA8W4 returned
- Pipeline proof: `FIX` → `Scalar` using `existing per-tile FIX_S publication` — every specialized tile publishes only after FIX_S before the function returns
- Perturbation: one clock read
- Source SHA-256: `f13eb7b7b7a2a87387e580b154a3b536ffa11e8cfd3dcb51c1fb8f2392045a8e`

```text
  2283                  if (criticalPathTrace_.HasExpertStage0Work()) {
  2284                      // Every A8W4 tile publishes only after FIX_S, so return is an actual GMM1 completion boundary.
  2285                      // TRACE_POINT: a8w4_gmm1_end_S
  2286                      uint32_t gmm1EndCycle = CriticalPathTrace::Now();
  2287                      criticalPathTrace_.EndExpertStage0(gmm1EndCycle);
```

### `a8w4_gmm1_prologue_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1205` in `AivPrologueA8W4`; `before` target line `1206`.
- Cardinality: first owned GMM1 weight tile per expert and AIV0
- Boundary claim: `causal` — before the first GMM1 blockPrologue issue
- Why: expose W4-to-W8 prologue occupancy
- Dependency: the expert prologue loop is entered
- Pipeline proof: `Scalar` → `Scalar` using `none` — the sample precedes the first blockPrologue call
- Perturbation: one clock read
- Source SHA-256: `a2da55d2022a4f250a035781fc62d6b7ef0610b502685cee8e954372f5448bb7`

```text
  1203          if constexpr (Policy::IS_GMM1) {
  1204              if (args.criticalPathTrace != nullptr && args.criticalPathTrace->NeedsExpertStage0Begin()) {
  1205                  // TRACE_POINT: a8w4_gmm1_prologue_begin_S
  1206                  uint32_t a8w4Gmm1PrologueBeginCycle = CriticalPathTrace::Now();
  1207                  args.criticalPathTrace->BeginExpertStage0(a8w4Gmm1PrologueBeginCycle);
```

### `a8w4_gmm1_prologue_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:2294` in `MegaMoe::Process`; `before` target line `2295`.
- Cardinality: after the last owned GMM1 weight tile completes
- Boundary claim: `actual` — all owned GMM1 prologue MTE3 writes to L1 have completed
- Why: measure completed W4-to-W8 prologue work
- Dependency: AivPrologueA8W4 and BlockPrologue destruction returned
- Pipeline proof: `MTE3` → `Scalar` using `MTE3_S EVENT_ID0` — the clock follows a trace-only MTE3_S set/wait after the prologue event rings are reclaimed
- Perturbation: one MTE3_S completion bridge and one clock read per active expert
- Source SHA-256: `f13eb7b7b7a2a87387e580b154a3b536ffa11e8cfd3dcb51c1fb8f2392045a8e`

```text
  2292                      AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
  2293                      AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
  2294                      // TRACE_POINT: a8w4_gmm1_prologue_end_S
  2295                      uint32_t gmm1PrologueEndCycle = CriticalPathTrace::Now();
  2296                      criticalPathTrace_.EndExpertStage0(gmm1PrologueEndCycle);
```

### `a8w4_swiglu_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_trace.h:484` in `RecordA8W4SwiGluBeginAfterReady`; `before` target line `485`.
- Cardinality: first ready GMM1 output tile per expert and AIV1
- Boundary claim: `causal` — after status/sequence polling and before the first GMM1-output load
- Why: measure AIV1 epilogue after tile readiness
- Dependency: AIC published the tile-ready state
- Pipeline proof: `Scalar polling` → `Scalar` using `none` — both call sites invoke this helper only after their ready loop
- Perturbation: one clock read
- Source SHA-256: `7bc505d11a304042a5b31f664b960969bd41059c808378a4c529bf7dc3f7ea19`

```text
   482  {
   483      if (trace != nullptr && trace->NeedsExpertStage1Begin()) {
   484          // TRACE_POINT: a8w4_swiglu_begin_S
   485          trace->BeginExpertStage1(CriticalPathTrace::Now());
   486      }
```

### `a8w4_swiglu_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:2301` in `MegaMoe::Process`; `before` target line `2302`.
- Cardinality: after the last owned SwiGLU tile completes per expert
- Boundary claim: `actual` — the final SwiGLU data/scale MTE3 writes and ready publication have completed
- Why: measure completed SwiGLU and MX quant output
- Dependency: AivGmm1PostA8W4 returned
- Pipeline proof: `MTE3/Scalar` → `Scalar` using `existing per-call MTE3_S` — BlockEpilogueSwigluMxQuant returns only after MTE3_S and its Scalar ready AtomicAdd; the outer clock follows the final call
- Perturbation: one clock read; no additional completion bridge
- Source SHA-256: `f13eb7b7b7a2a87387e580b154a3b536ffa11e8cfd3dcb51c1fb8f2392045a8e`

```text
  2299              } else {
  2300                  if (criticalPathTrace_.HasExpertStage1Work()) {
  2301                      // TRACE_POINT: a8w4_swiglu_end_S
  2302                      uint32_t swigluEndCycle = CriticalPathTrace::Now();
  2303                      criticalPathTrace_.EndExpertStage1(swigluEndCycle);
```

### `a8w4_gmm2_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1137` in `AicComputeA8W4`; `before` target line `1138`.
- Cardinality: first owned GMM2 tile per routed expert and AIC
- Boundary claim: `causal` — after WaitForUpstreamReady and before first GMM2 blockMmad
- Why: locate GMM2 after SwiGLU readiness
- Dependency: SwiGLU output ready
- Pipeline proof: `Scalar polling` → `Scalar` using `none` — the sample follows the upstream wait
- Perturbation: one clock read
- Source SHA-256: `a2da55d2022a4f250a035781fc62d6b7ef0610b502685cee8e954372f5448bb7`

```text
  1135              }
  1136          } else if (criticalPathTrace != nullptr && criticalPathTrace->NeedsExpertStage1Begin()) {
  1137              // TRACE_POINT: a8w4_gmm2_begin_S
  1138              uint32_t a8w4Gmm2BeginCycle = CriticalPathTrace::Now();
  1139              criticalPathTrace->BeginExpertStage1(a8w4Gmm2BeginCycle);
```

### `a8w4_gmm2_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:2375` in `MegaMoe::Process`; `before` target line `2376`.
- Cardinality: last owned GMM2 tile per routed expert and AIC
- Boundary claim: `actual` — all owned GMM2 tiles have passed FIX_S publication
- Why: measure actual AIC GMM2 completion/publication
- Dependency: AicComputeA8W4 returned
- Pipeline proof: `FIX` → `Scalar` using `existing per-tile FIX_S publication` — tile completion notification or GM epilogue publication follows FIX_S
- Perturbation: one clock read
- Source SHA-256: `f13eb7b7b7a2a87387e580b154a3b536ffa11e8cfd3dcb51c1fb8f2392045a8e`

```text
  2373                  if (criticalPathTrace_.HasExpertStage1Work()) {
  2374                      // Every A8W4 tile publishes only after FIX_S, so return is an actual GMM2 completion boundary.
  2375                      // TRACE_POINT: a8w4_gmm2_end_S
  2376                      uint32_t gmm2EndCycle = CriticalPathTrace::Now();
  2377                      criticalPathTrace_.EndExpertStage1(gmm2EndCycle);
```

### `a8w4_gmm2_prologue_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:1210` in `AivPrologueA8W4`; `before` target line `1211`.
- Cardinality: first owned GMM2 weight tile per expert and AIV0
- Boundary claim: `causal` — before first GMM2 blockPrologue issue
- Why: expose GMM2 W4-to-W8 prologue occupancy
- Dependency: the expert prologue loop is entered
- Pipeline proof: `Scalar` → `Scalar` using `none` — the sample precedes blockPrologue
- Perturbation: one clock read
- Source SHA-256: `a2da55d2022a4f250a035781fc62d6b7ef0610b502685cee8e954372f5448bb7`

```text
  1208              }
  1209          } else if (args.criticalPathTrace != nullptr && args.criticalPathTrace->NeedsExpertStage1Begin()) {
  1210              // TRACE_POINT: a8w4_gmm2_prologue_begin_S
  1211              uint32_t a8w4Gmm2PrologueBeginCycle = CriticalPathTrace::Now();
  1212              args.criticalPathTrace->BeginExpertStage1(a8w4Gmm2PrologueBeginCycle);
```

### `a8w4_gmm2_prologue_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:2384` in `MegaMoe::Process`; `before` target line `2385`.
- Cardinality: after the last owned GMM2 weight tile completes
- Boundary claim: `actual` — all owned GMM2 prologue MTE3 writes to L1 have completed
- Why: measure completed W4-to-W8 prologue work
- Dependency: AivPrologueA8W4 and BlockPrologue destruction returned
- Pipeline proof: `MTE3` → `Scalar` using `MTE3_S EVENT_ID0` — the clock follows a trace-only MTE3_S set/wait after the prologue event rings are reclaimed
- Perturbation: one MTE3_S completion bridge and one clock read per active expert
- Source SHA-256: `f13eb7b7b7a2a87387e580b154a3b536ffa11e8cfd3dcb51c1fb8f2392045a8e`

```text
  2382                      AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
  2383                      AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
  2384                      // TRACE_POINT: a8w4_gmm2_prologue_end_S
  2385                      uint32_t gmm2PrologueEndCycle = CriticalPathTrace::Now();
  2386                      criticalPathTrace_.EndExpertStage1(gmm2PrologueEndCycle);
```

### `a8w4_combine_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_trace.h:492` in `RecordA8W4CombineBeginAfterReady`; `before` target line `493`.
- Cardinality: first ready combine tile/group per expert and AIV1
- Boundary claim: `causal` — after GM tile sequence or group counter readiness and before first combine load
- Why: measure Combine only after GMM2 output is visible
- Dependency: GMM2 tile/group ready
- Pipeline proof: `Scalar polling` → `Scalar` using `none` — both inline and quantized call sites invoke the helper after their ready wait
- Perturbation: one clock read
- Source SHA-256: `7bc505d11a304042a5b31f664b960969bd41059c808378a4c529bf7dc3f7ea19`

```text
   490  {
   491      if (trace != nullptr && trace->NeedsExpertStage2Begin()) {
   492          // TRACE_POINT: a8w4_combine_begin_S
   493          trace->BeginExpertStage2(CriticalPathTrace::Now());
   494      }
```

### `a8w4_combine_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe.h:2393` in `MegaMoe::Process`; `before` target line `2394`.
- Cardinality: last combine send per routed expert and AIV1
- Boundary claim: `actual` — the last MTE3 completion dependency has reached Scalar
- Why: measure completed remote combine writes
- Dependency: GroupMatmulWithCombine returned
- Pipeline proof: `MTE3 transferred to MTE2` → `Scalar` using `MTE2_S EVENT_ID0` — CombineTokenGroup/AivGmm2Post leave completion on MTE2 and the trace adds MTE2_S
- Perturbation: one narrow completion bridge at expert end
- Source SHA-256: `f13eb7b7b7a2a87387e580b154a3b536ffa11e8cfd3dcb51c1fb8f2392045a8e`

```text
  2391                      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
  2392                      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
  2393                      // TRACE_POINT: a8w4_combine_end_S
  2394                      uint32_t combineEndCycle = CriticalPathTrace::Now();
  2395                      criticalPathTrace_.EndExpertStage2(combineEndCycle);
```

## Working tree status

```text
M mc2/mega_moe/op_kernel/arch35/mega_moe.h
 M mc2/mega_moe/op_kernel/arch35/mega_moe_trace.h
 M mc2/mega_moe/tools/critical_path_trace/README.md
 M mc2/mega_moe/tools/critical_path_trace/test_critical_path_trace.py
 M mc2/mega_moe/tools/critical_path_trace/trace-spec-a8w4.json
 M mc2/mega_moe/tools/critical_path_trace/trace_tool.py
```
