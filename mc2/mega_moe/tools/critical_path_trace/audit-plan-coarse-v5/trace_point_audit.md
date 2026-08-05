# Trace point audit

- Status: `pass`
- Phase: `plan`
- Trace: `mega-moe-wave-critical-path`
- Repository: `/home/mace/workspace/ops-transformer_megaMoeOpt_trace_20260803`
- Revision: `b25081aa4d333037c7fc898b678e98c74eafdaf7`
- Branch: `codex/mega-moe-critical-path-trace-20260803`
- Dirty: `True`
- Spec SHA-256: `6304575de97a3d24cc2d8a54a425c0438aac5292731c5f36ac89c6884b8eb70e`

## Placement summary

| Point | Roles | Resolved location | Target line | Placement | Boundary | Bridge | Why |
|---|---|---|---:|---|---|---|---|
| process_begin_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2719 | 2719 | before | causal | none | Provides one local shared-axis reference for AIC, AIV0, and AIV1 inside the same physical AI Block, without asserting clock alignment across AI Blocks or ranks; also exposes role launch skew before input preparation. |
| work_end_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2772 | 2772 | after | actual | existing PublishGmm2Ready FIX_S and Unpermute final MTE3_V waits; no additional hot-path bridge | Separates the real role-local useful-work tail from the diagnostic rendezvous and y record emission overhead. |
| dispatch_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:1083 | 1083 | before | causal | none | Measures the actual per-wave AIV1 prefetch window, including send-count calculation, rather than the outer wave that may prefetch two waves at once. |
| dispatch_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:1086 | 1086 | after | actual | trace-only MTE3_S using recycled EVENT_ID0 on the last expert of the wave | Proves when each AIV1 has completed the token/scale/meta-info writes that release GMM1 and permits direct comparison with GMM1 readiness. |
| gmm1_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:467 | 467 | after | causal | trace-only FIX_S using recycled EVENT_ID0 before the first sampled tile | Distinguishes Dispatch-ready and prior ping-pong dependency time from actual first GMM1 MMAD issue on each AIC. |
| gmm1_end_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2519 | 2519 | after | actual | trace-only FIX_S using recycled EVENT_ID0 when this core recorded GMM1 work | Makes the GMM1 interval end at actual local FIX completion instead of helper return or AIV notification issue. |
| swiglu_begin_S | AIV0 | mc2/mega_moe/op_kernel/arch35/block_epilogue_swiglu_mx_quant.h:1144 | 1144 | after | causal | trace-only V_S using recycled EVENT_ID0 at the common epilogue entry | Places the common SwiGLU begin after both non-prefetch WaitForCube and prefetch MTE2_V input readiness, regardless of the current epilogue branch. |
| swiglu_end_S | AIV0 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2519 | 2519 | after | actual | existing per-tile MTE3_S in BlockEpilogueSwigluMxQuant::operator(); no added end bridge | Shows the actual AIV0 epilogue completion and its overlap with AIC GMM2 and AIV1 Dispatch/Combine. |
| gmm2_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:467 | 467 | after | causal | none | Measures GMM2 only after the current wave's SwiGLU-ready counter is visible, leaving publication-to-observation latency outside GMM2 compute. |
| gmm2_end_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2537 | 2537 | after | actual | existing FIX_S inside PublishGmm2Ready; no extra bridge | Proves local GMM2 direct-GM completion and ready publication timing while keeping remote AIV1 observation as a separate handoff gap. |
| combine_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2407 | 2407 | after | causal | existing MTE2_V and V_S in WaitGmm2Ready | Exposes the publication-to-ready-observation handoff and starts Combine only after the full AIC fan-in is proven. |
| combine_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2441 | 2441 | after | actual | trace-only MTE2_S using recycled EVENT_ID0 after the drain | Ends Combine at actual remote MTE3 send completion rather than at Scalar issue or ring-drain enqueue time. |

## Point evidence

### `process_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2719` in `MegaMoeWave::Process`; `before` target line `2719`.
- Cardinality: once at MegaMoeWave::Process entry per physical role/core record
- Boundary claim: `causal` — Scalar has entered the current MTE A8W8 MegaMoeWave Process body and no operator work has yet been issued by this role/core.
- Why: Provides one local shared-axis reference for AIC, AIV0, and AIV1 inside the same physical AI Block, without asserting clock alignment across AI Blocks or ranks; also exposes role launch skew before input preparation.
- Dependency: mixed-kernel task entry for this physical role/core
- Pipeline proof: `kernel scheduler and Scalar control` → `Scalar` using `none` — The clock is the first diagnostic instruction immediately before the first Process statement.
- Perturbation: One 32-bit GetSystemCycle and one named Scalar assignment per role/core.
- Source SHA-256: `27bc1f7ee949ab4aa22a73df3014dd0ccf441670309250ea4ab8f6367b9ab00a`

```text
  2717  #endif
  2718      // 保存入口时的溢出模式，并初始化输入准备阶段使用的 UB。
  2719      int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
  2720      SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
  2721      SendMaskBufferConfig sendMaskBufferConfig = SendAndQuantBuffInit();
```

### `work_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2772` in `MegaMoeWave::Process`; `after` target line `2772`.
- Cardinality: once after useful work and overflow-mode restoration per physical role/core record
- Boundary claim: `actual` — All useful work owned by this role is complete; for AIV roles Unpermute has consumed its final MTE3_V completion event, and routed AIC publications already passed FIX_S.
- Why: Separates the real role-local useful-work tail from the diagnostic rendezvous and y record emission overhead.
- Dependency: role-local GMM/Combine completion and, on AIV, final Unpermute y store completion
- Pipeline proof: `FIX and Scalar for AIC; MTE3, Vector, and Scalar for AIV` → `Scalar` using `existing PublishGmm2Ready FIX_S and Unpermute final MTE3_V waits; no additional hot-path bridge` — The sample is after ProcessRoutedExpertWaves, cross-rank completion, Unpermute's final MTE3_V wait, and control-state restoration, but before either trace rendezvous.
- Perturbation: One 32-bit GetSystemCycle and assignment before trace-only terminal synchronization.
- Source SHA-256: `27bc1f7ee949ab4aa22a73df3014dd0ccf441670309250ea4ab8f6367b9ab00a`

```text
  2770      }
  2771      // 恢复入口时保存的溢出模式。
  2772      SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
  2773  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  2774      if (CriticalPathTracePtr() != nullptr) {
```

### `dispatch_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:1083` in `MegaMoeWave::DispatchExpert`; `before` target line `1083`.
- Cardinality: first dispatched expert per physical AIV1 core per captured routed-expert wave
- Boundary claim: `causal` — The first expert in this dispatch wave is about to start SendCntCal on the AIV1 Scalar/control path.
- Why: Measures the actual per-wave AIV1 prefetch window, including send-count calculation, rather than the outer wave that may prefetch two waves at once.
- Dependency: the monotonic dispatch cursor selected this expert and any prior Combine ring reuse was drained by the preceding wave
- Pipeline proof: `AIV1 Scalar control` → `Scalar` using `none` — DispatchExpert is invoked exactly once per expert by DispatchExpertsUntil; the first-expert test maps expertIdx to the current host-derived experts_per_wave.
- Perturbation: One first-wave-slot check plus one clock sample per active AIV1/wave.
- Source SHA-256: `27bc1f7ee949ab4aa22a73df3014dd0ccf441670309250ea4ab8f6367b9ab00a`

```text
  1081  #endif
  1082          uint64_t sendCnt = 0;
  1083          SendCntCal(static_cast<int32_t>(expertIdx), sendCnt);
  1084          if (UpdateGroupParams<AddrUpdateMode::GMM1>(state, expertIdx, sendCnt)) {
  1085              UpdateGlobalBuffer<AddrUpdateMode::GMM1>(gmmAddrInfo, state);
```

### `dispatch_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:1086` in `MegaMoeWave::DispatchExpert`; `after` target line `1086`.
- Cardinality: last dispatched expert per physical AIV1 core per captured routed-expert wave
- Boundary claim: `actual` — The last expert handled by this AIV1 in the wave has completed all Dispatch MTE3 writes and ready publications.
- Why: Proves when each AIV1 has completed the token/scale/meta-info writes that release GMM1 and permits direct comparison with GMM1 readiness.
- Dependency: CopyGMToGMPerToken has recycled every dynamic ring slot and MetaInfoCalAndDispatch has issued its final publication
- Pipeline proof: `MTE3` → `Scalar` using `trace-only MTE3_S using recycled EVENT_ID0 on the last expert of the wave` — All EVENT_ID0..5 Dispatch ring lifetimes are paired before MetaInfoCalAndDispatch returns; the added MTE3_S orders the last output/publication into Scalar before sampling.
- Perturbation: One MTE3-to-Scalar bridge and one clock sample per active AIV1/wave.
- Source SHA-256: `27bc1f7ee949ab4aa22a73df3014dd0ccf441670309250ea4ab8f6367b9ab00a`

```text
  1084          if (UpdateGroupParams<AddrUpdateMode::GMM1>(state, expertIdx, sendCnt)) {
  1085              UpdateGlobalBuffer<AddrUpdateMode::GMM1>(gmmAddrInfo, state);
  1086              MetaInfoCalAndDispatch(gmmAddrInfo, static_cast<int32_t>(expertIdx), bufferConfig);
  1087          }
  1088  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

### `gmm1_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:467` in `MegaMoeImpl::Detail::AicComputeGeneric`; `after` target line `467`.
- Cardinality: first actually assigned GMM1 tile per physical AIC core per captured routed-expert wave
- Boundary claim: `causal` — Dispatch GM readiness polling and any outstanding AIC-to-AIV0 acknowledgement have passed and the first GMM1 tile is about to construct and issue MMAD work.
- Why: Distinguishes Dispatch-ready and prior ping-pong dependency time from actual first GMM1 MMAD issue on each AIC.
- Dependency: WaitForUpstreamReady observed the complete Dispatch wave count and any WaitForVector dependency for this owner completed
- Pipeline proof: `Scalar GM visibility and FIX` → `Scalar` using `trace-only FIX_S using recycled EVENT_ID0 before the first sampled tile` — The hook is after both upstream readiness and ping-pong waits in AicComputeGeneric; FIX_S transfers the FIX wait/completion dependency to Scalar before the clock.
- Perturbation: One conditional slot check, one FIX-to-Scalar bridge, and one clock per active AIC/wave.
- Source SHA-256: `5e8084d1de9143df589dbad8084b4f2626bf6c8732852881f918048c21fd3cab`

```text
   465                  WaitForUpstreamReady<Policy, IsShared>(workSet.gmmAddrInfo, config, mLoc);
   466              }
   467              if constexpr (CombineQuantMode == COMBINE_NO_QUANT && !GMM2_DIRECT_GM) {
   468                  if (args.vecSetSyncCom2 >= 2) {
   469                      WaitForVector(args.pingpongIdx);
```

### `gmm1_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2519` in `MegaMoeWave::ProcessRoutedExpertWaves`; `after` target line `2519`.
- Cardinality: after the last GMM1 tile assigned to each physical AIC core in a captured routed-expert wave
- Boundary claim: `actual` — All GMM1 FIX work assigned to this AIC in the wave is complete.
- Why: Makes the GMM1 interval end at actual local FIX completion instead of helper return or AIV notification issue.
- Dependency: ProcessGmm1Wave returned after issuing every owned GMM1 tile
- Pipeline proof: `FIX` → `Scalar` using `trace-only FIX_S using recycled EVENT_ID0 when this core recorded GMM1 work` — The bridge is after the complete wave loop and executes only for records with a causal begin, so its Scalar clock cannot run ahead of the final FIX operation.
- Perturbation: One FIX-to-Scalar completion bridge and clock per active AIC/wave.
- Source SHA-256: `27bc1f7ee949ab4aa22a73df3014dd0ccf441670309250ea4ab8f6367b9ab00a`

```text
  2517
  2518          ProcessGmm1Wave(
  2519              batchBegin, batchEnd, gmm1State, gmm1AddrInfo, vecSetSyncCom);
  2520  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0 && \
  2521      MEGA_MOE_ENABLE_FINE_GRAIN_CRITICAL_PATH_TRACE == 0
```

### `swiglu_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/block_epilogue_swiglu_mx_quant.h:1144` in `BlockEpilogueSwigluMxQuant::operator()`; `after` target line `1144`.
- Cardinality: first non-empty epilogue tile per physical AIV0 core per captured routed-expert wave
- Boundary claim: `causal` — A non-empty epilogue invocation has all upstream GMM1/input dependencies ordered into Vector and is about to start SwiGLU/quant setup and Vector work.
- Why: Places the common SwiGLU begin after both non-prefetch WaitForCube and prefetch MTE2_V input readiness, regardless of the current epilogue branch.
- Dependency: the caller passed its AIC-to-AIV0 V wait or MTE2_V data-ready wait before entering the epilogue
- Pipeline proof: `Vector` → `Scalar` using `trace-only V_S using recycled EVENT_ID0 at the common epilogue entry` — Every routed wave epilogue branch calls this operator only after its readiness wait; V_S orders those V-pipeline waits before the first recorded clock.
- Perturbation: One conditional slot check, one Vector-to-Scalar bridge, and one clock per active AIV0/wave.
- Source SHA-256: `36c3efd3d9a096b605f12d30e6577a0448a602d8cb6ee30c4ccde9400280dc85`

```text
  1142      blockCoord_ = blockCoord;
  1143
  1144      if (singleM_ == 0) {
  1145          return;
  1146      }
```

### `swiglu_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2519` in `MegaMoeWave::ProcessRoutedExpertWaves`; `after` target line `2519`.
- Cardinality: after the last epilogue tile assigned to each physical AIV0 core in a captured routed-expert wave
- Boundary claim: `actual` — Every SwiGLU/quant GM output owned by this AIV0 in the wave has completed.
- Why: Shows the actual AIV0 epilogue completion and its overlap with AIC GMM2 and AIV1 Dispatch/Combine.
- Dependency: all BlockEpilogueSwigluMxQuant invocations assigned to the core returned
- Pipeline proof: `MTE3` → `Scalar` using `existing per-tile MTE3_S in BlockEpilogueSwigluMxQuant::operator(); no added end bridge` — Each operator return consumes MTE3_V and MTE3_S after data/scale copies and then publishes its ready count; the outer wave return therefore follows the last actual GM completion.
- Perturbation: One clock and named assignment per active AIV0/wave; no additional completion serialization.
- Source SHA-256: `27bc1f7ee949ab4aa22a73df3014dd0ccf441670309250ea4ab8f6367b9ab00a`

```text
  2517
  2518          ProcessGmm1Wave(
  2519              batchBegin, batchEnd, gmm1State, gmm1AddrInfo, vecSetSyncCom);
  2520  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0 && \
  2521      MEGA_MOE_ENABLE_FINE_GRAIN_CRITICAL_PATH_TRACE == 0
```

### `gmm2_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:467` in `MegaMoeImpl::Detail::AicComputeGeneric`; `after` target line `467`.
- Cardinality: first actually assigned GMM2 tile per physical AIC core per captured routed-expert wave
- Boundary claim: `causal` — Scalar GM polling has observed the required SwiGLU-ready count for the first owned GMM2 wave tile and MMAD issue is next.
- Why: Measures GMM2 only after the current wave's SwiGLU-ready counter is visible, leaving publication-to-observation latency outside GMM2 compute.
- Dependency: WaitForUpstreamReady observed the wave-grained swigluToGmm2Flag target
- Pipeline proof: `MTE3 completion published through Scalar-visible GM AtomicAdd` → `Scalar` using `none` — The bypass GM polling loop runs on Scalar and returns only after epilogue MTE3_S and ready AtomicAdd; the immediately following Scalar clock is causally ordered without a cross-pipeline bridge.
- Perturbation: One conditional slot check and one clock per active AIC/wave.
- Source SHA-256: `5e8084d1de9143df589dbad8084b4f2626bf6c8732852881f918048c21fd3cab`

```text
   465                  WaitForUpstreamReady<Policy, IsShared>(workSet.gmmAddrInfo, config, mLoc);
   466              }
   467              if constexpr (CombineQuantMode == COMBINE_NO_QUANT && !GMM2_DIRECT_GM) {
   468                  if (args.vecSetSyncCom2 >= 2) {
   469                      WaitForVector(args.pingpongIdx);
```

### `gmm2_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2537` in `MegaMoeWave::ProcessRoutedExpertWaves`; `after` target line `2537`.
- Cardinality: after the last ready publication for GMM2 work assigned to each physical AIC core in a captured routed-expert wave
- Boundary claim: `actual` — All owned GMM2 FIX writes for the wave are complete and the final per-expert ready publication has been issued by this AIC.
- Why: Proves local GMM2 direct-GM completion and ready publication timing while keeping remote AIV1 observation as a separate handoff gap.
- Dependency: ProcessGmm2Wave completed every expert and PublishGmm2Ready ran after each expert
- Pipeline proof: `FIX then Scalar bypass GM store` → `Scalar` using `existing FIX_S inside PublishGmm2Ready; no extra bridge` — PublishGmm2Ready executes FIX_S before WriteGmByPassDCache; the outer sample follows the last publication in source and Scalar order. It does not claim AIV1 has observed it.
- Perturbation: One clock and named assignment per active AIC/wave; existing production FIX_S publication semantics are reused.
- Source SHA-256: `27bc1f7ee949ab4aa22a73df3014dd0ccf441670309250ea4ab8f6367b9ab00a`

```text
  2535          const uint32_t gmm1EndCursor = startBlockIdx_;
  2536
  2537          ProcessGmm2Wave(batchBegin, batchEnd, gmm2State, gmm2AddrInfo, vecSetSyncCom);
  2538  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0 && \
  2539      MEGA_MOE_ENABLE_FINE_GRAIN_CRITICAL_PATH_TRACE == 0
```

### `combine_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2407` in `MegaMoeWave::ProcessCombineExperts`; `after` target line `2407`.
- Cardinality: first non-empty token range per physical AIV1 core per captured routed-expert wave
- Boundary claim: `causal` — This AIV1 owns at least one token and its first expert's GMM2-ready scan has observed all AIC publications; meta-info load and first row send follow.
- Why: Exposes the publication-to-ready-observation handoff and starts Combine only after the full AIC fan-in is proven.
- Dependency: WaitGmm2Ready reduced all per-AIC ready cache lines to at least blockNum
- Pipeline proof: `MTE2 then Vector reduction` → `Scalar` using `existing MTE2_V and V_S in WaitGmm2Ready` — WaitGmm2Ready returns only after the V_S-ordered sum is read on Scalar and meets the all-AIC target; the marker is immediately after that return.
- Perturbation: One conditional slot check and one clock per active AIV1/wave.
- Source SHA-256: `27bc1f7ee949ab4aa22a73df3014dd0ccf441670309250ea4ab8f6367b9ab00a`

```text
  2405              GetCombineOwnedRange(currentExpertTokenNum);
  2406          if (currentCoreTokenRange.count != 0U) {
  2407              WaitGmm2Ready(expertIdx);
  2408  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  2409              // TRACE_POINT: combine_begin_S
```

### `combine_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2441` in `MegaMoeWave::ProcessCombineExperts`; `after` target line `2441`.
- Cardinality: after the final remote-send ring drain per physical AIV1 core per captured routed-expert wave
- Boundary claim: `actual` — Every token row issued by this AIV1 for the wave has completed its remote MTE3 send.
- Why: Ends Combine at actual remote MTE3 send completion rather than at Scalar issue or ring-drain enqueue time.
- Dependency: DrainCombineRowRing consumed every MTE3_MTE2 event for each actually used row slot
- Pipeline proof: `MTE3 completion dependency held by MTE2` → `Scalar` using `trace-only MTE2_S using recycled EVENT_ID0 after the drain` — The ring drain waits only the slots that were issued; MTE2_S transfers those completed MTE3 dependencies to Scalar before sampling.
- Perturbation: One MTE2-to-Scalar bridge and one clock per active AIV1/wave.
- Source SHA-256: `27bc1f7ee949ab4aa22a73df3014dd0ccf441670309250ea4ab8f6367b9ab00a`

```text
  2439
  2440      // 下一 wave 的 dispatch 会复用同一片 UB 和事件编号，离开前必须回收所有已使用槽位。
  2441      DrainCombineRowRing(rowSequence);
  2442  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  2443      // TRACE_POINT: combine_end_S
```

## Working tree status

```text
M mc2/mega_moe/op_kernel/arch35/mega_moe_trace.h
 M mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h
 M mc2/mega_moe/tools/critical_path_trace/README.md
 M mc2/mega_moe/tools/critical_path_trace/artifacts/synthetic/index.html
 M mc2/mega_moe/tools/critical_path_trace/artifacts/synthetic/rank0.contract-check.json
 M mc2/mega_moe/tools/critical_path_trace/artifacts/synthetic/rank0.html
 M mc2/mega_moe/tools/critical_path_trace/artifacts/synthetic/rank0.normalized.json
 M mc2/mega_moe/tools/critical_path_trace/artifacts/synthetic/rank0.order-check.json
 M mc2/mega_moe/tools/critical_path_trace/artifacts/synthetic/rank0.raw
 M mc2/mega_moe/tools/critical_path_trace/artifacts/synthetic/rank0.runtime.json
 M mc2/mega_moe/tools/critical_path_trace/artifacts/synthetic/run-manifest.json
 M mc2/mega_moe/tools/critical_path_trace/audit-implemented-coarse-v5/trace_point_audit.json
 M mc2/mega_moe/tools/critical_path_trace/audit-implemented-coarse-v5/trace_point_audit.md
 M mc2/mega_moe/tools/critical_path_trace/audit-implemented/trace_point_audit.json
 M mc2/mega_moe/tools/critical_path_trace/audit-implemented/trace_point_audit.md
 M mc2/mega_moe/tools/critical_path_trace/audit-plan-coarse-v5/trace_point_audit.json
 M mc2/mega_moe/tools/critical_path_trace/audit-plan-coarse-v5/trace_point_audit.md
 M mc2/mega_moe/tools/critical_path_trace/audit-plan/trace_point_audit.json
 M mc2/mega_moe/tools/critical_path_trace/audit-plan/trace_point_audit.md
 M mc2/mega_moe/tools/critical_path_trace/mega_moe_trace_contract.py
 M mc2/mega_moe/tools/critical_path_trace/run_critical_path_trace.py
 M mc2/mega_moe/tools/critical_path_trace/test_critical_path_trace.py
 M mc2/mega_moe/tools/critical_path_trace/trace-spec-coarse-v5.json
 M mc2/mega_moe/tools/critical_path_trace/trace-spec.json
 M mc2/mega_moe/tools/critical_path_trace/trace_tool.py
 M mc2/mega_moe/tools/critical_path_trace/validation-report.json
 M mc2/mega_moe/tools/critical_path_trace/verify_trace_off.py
?? mc2/mega_moe/docs/
```
