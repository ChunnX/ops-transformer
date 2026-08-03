# Trace point audit

- Status: `pass`
- Phase: `plan`
- Trace: `mega-moe-wave-critical-path`
- Repository: `/home/mace/workspace/ops-transformer_megaMoeOpt_trace_20260803`
- Revision: `4b3d39658e1d4f5d1f5b60acedfe2df8201e08b2`
- Branch: `codex/mega-moe-critical-path-trace-20260803`
- Dirty: `True`
- Spec SHA-256: `8e450ba62fba5c9499f47bf1bf624bc389d5c2b20ffd102abf7fee4e06cc6fd8`

## Placement summary

| Point | Roles | Resolved location | Target line | Placement | Boundary | Bridge | Why |
|---|---|---|---:|---|---|---|---|
| process_begin_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2622 | 2622 | before | causal | none | Provides the one shared-axis per-rank reference and exposes role launch skew before input preparation. |
| work_end_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2675 | 2675 | after | actual | existing PublishGmm2Ready FIX_S and Unpermute final MTE3_V waits; no additional hot-path bridge | Separates the real role-local useful-work tail from the diagnostic rendezvous and y record emission overhead. |
| dispatch_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:1077 | 1077 | before | causal | none | Measures the actual per-wave AIV1 prefetch window, including send-count calculation, rather than the outer wave that may prefetch two waves at once. |
| dispatch_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:1080 | 1080 | after | actual | trace-only MTE3_S using recycled EVENT_ID0 on the last expert of the wave | Proves when each AIV1 has completed the token/scale/meta-info writes that release GMM1 and permits direct comparison with GMM1 readiness. |
| gmm1_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:487 | 487 | before | causal | trace-only FIX_S using recycled EVENT_ID0 before the first sampled tile | Distinguishes Dispatch-ready and prior ping-pong dependency time from actual first GMM1 MMAD issue on each AIC. |
| gmm1_end_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2435 | 2435 | after | actual | trace-only FIX_S using recycled EVENT_ID0 when this core recorded GMM1 work | Makes the GMM1 interval end at actual local FIX completion instead of helper return or AIV notification issue. |
| swiglu_begin_S | AIV0 | mc2/mega_moe/op_kernel/arch35/block_epilogue_swiglu_mx_quant.h:1156 | 1156 | before | causal | trace-only V_S using recycled EVENT_ID0 at the common epilogue entry | Places the common SwiGLU begin after both non-prefetch WaitForCube and prefetch MTE2_V input readiness, regardless of the current epilogue branch. |
| swiglu_end_S | AIV0 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2435 | 2435 | after | actual | existing per-tile MTE3_S in BlockEpilogueSwigluMxQuant::operator(); no added end bridge | Shows the actual AIV0 epilogue completion and its overlap with AIC GMM2 and AIV1 Dispatch/Combine. |
| gmm2_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:487 | 487 | before | causal | none | Measures GMM2 only after the current wave's SwiGLU-ready counter is visible, leaving publication-to-observation latency outside GMM2 compute. |
| gmm2_end_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2452 | 2452 | after | actual | existing FIX_S inside PublishGmm2Ready; no extra bridge | Proves local GMM2 direct-GM completion and ready publication timing while keeping remote AIV1 observation as a separate handoff gap. |
| combine_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2343 | 2343 | after | causal | existing MTE2_V and V_S in WaitGmm2Ready | Exposes the publication-to-ready-observation handoff and starts Combine only after the full AIC fan-in is proven. |
| combine_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2371 | 2371 | after | actual | trace-only MTE2_S using recycled EVENT_ID0 after the drain | Ends Combine at actual remote MTE3 send completion rather than at Scalar issue or ring-drain enqueue time. |

## Point evidence

### `process_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2622` in `MegaMoeWave::Process`; `before` target line `2622`.
- Cardinality: once at MegaMoeWave::Process entry per physical role/core record
- Boundary claim: `causal` — Scalar has entered the current MTE A8W8 MegaMoeWave Process body and no operator work has yet been issued by this role/core.
- Why: Provides the one shared-axis per-rank reference and exposes role launch skew before input preparation.
- Dependency: mixed-kernel task entry for this physical role/core
- Pipeline proof: `kernel scheduler and Scalar control` → `Scalar` using `none` — The clock is the first diagnostic instruction immediately before the first Process statement.
- Perturbation: One 32-bit GetSystemCycle and one named Scalar assignment per role/core.
- Source SHA-256: `7066bb0b6b7d6cfb2f93c5106e30158f413f914cb9702923e9e8d37b47865709`

```text
  2620  #endif
  2621      // 保存入口时的溢出模式，并初始化输入准备阶段使用的 UB。
  2622      int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
  2623      SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
  2624      SendMaskBufferConfig sendMaskBufferConfig = SendAndQuantBuffInit();
```

### `work_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2675` in `MegaMoeWave::Process`; `after` target line `2675`.
- Cardinality: once after useful work and overflow-mode restoration per physical role/core record
- Boundary claim: `actual` — All useful work owned by this role is complete; for AIV roles Unpermute has consumed its final MTE3_V completion event, and routed AIC publications already passed FIX_S.
- Why: Separates the real role-local useful-work tail from the diagnostic rendezvous and y record emission overhead.
- Dependency: role-local GMM/Combine completion and, on AIV, final Unpermute y store completion
- Pipeline proof: `FIX and Scalar for AIC; MTE3, Vector, and Scalar for AIV` → `Scalar` using `existing PublishGmm2Ready FIX_S and Unpermute final MTE3_V waits; no additional hot-path bridge` — The sample is after ProcessRoutedExpertWaves, cross-rank completion, Unpermute's final MTE3_V wait, and control-state restoration, but before either trace rendezvous.
- Perturbation: One 32-bit GetSystemCycle and assignment before trace-only terminal synchronization.
- Source SHA-256: `7066bb0b6b7d6cfb2f93c5106e30158f413f914cb9702923e9e8d37b47865709`

```text
  2673      }
  2674      // 恢复入口时保存的溢出模式。
  2675      SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
  2676  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  2677      if (CriticalPathTracePtr() != nullptr) {
```

### `dispatch_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:1077` in `MegaMoeWave::DispatchExpert`; `before` target line `1077`.
- Cardinality: first dispatched expert per physical AIV1 core per captured routed-expert wave
- Boundary claim: `causal` — The first expert in this dispatch wave is about to start SendCntCal on the AIV1 Scalar/control path.
- Why: Measures the actual per-wave AIV1 prefetch window, including send-count calculation, rather than the outer wave that may prefetch two waves at once.
- Dependency: the monotonic dispatch cursor selected this expert and any prior Combine ring reuse was drained by the preceding wave
- Pipeline proof: `AIV1 Scalar control` → `Scalar` using `none` — DispatchExpert is invoked exactly once per expert by DispatchExpertsUntil; the first-expert test maps expertIdx to the current host-derived experts_per_wave.
- Perturbation: One first-wave-slot check plus one clock sample per active AIV1/wave.
- Source SHA-256: `7066bb0b6b7d6cfb2f93c5106e30158f413f914cb9702923e9e8d37b47865709`

```text
  1075  #endif
  1076          uint64_t sendCnt = 0;
  1077          SendCntCal(static_cast<int32_t>(expertIdx), sendCnt);
  1078          if (UpdateGroupParams<AddrUpdateMode::GMM1>(state, expertIdx, sendCnt)) {
  1079              UpdateGlobalBuffer<AddrUpdateMode::GMM1>(gmmAddrInfo, state);
```

### `dispatch_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:1080` in `MegaMoeWave::DispatchExpert`; `after` target line `1080`.
- Cardinality: last dispatched expert per physical AIV1 core per captured routed-expert wave
- Boundary claim: `actual` — The last expert handled by this AIV1 in the wave has completed all Dispatch MTE3 writes and ready publications.
- Why: Proves when each AIV1 has completed the token/scale/meta-info writes that release GMM1 and permits direct comparison with GMM1 readiness.
- Dependency: CopyGMToGMPerToken has recycled every dynamic ring slot and MetaInfoCalAndDispatch has issued its final publication
- Pipeline proof: `MTE3` → `Scalar` using `trace-only MTE3_S using recycled EVENT_ID0 on the last expert of the wave` — All EVENT_ID0..5 Dispatch ring lifetimes are paired before MetaInfoCalAndDispatch returns; the added MTE3_S orders the last output/publication into Scalar before sampling.
- Perturbation: One MTE3-to-Scalar bridge and one clock sample per active AIV1/wave.
- Source SHA-256: `7066bb0b6b7d6cfb2f93c5106e30158f413f914cb9702923e9e8d37b47865709`

```text
  1078          if (UpdateGroupParams<AddrUpdateMode::GMM1>(state, expertIdx, sendCnt)) {
  1079              UpdateGlobalBuffer<AddrUpdateMode::GMM1>(gmmAddrInfo, state);
  1080              MetaInfoCalAndDispatch(gmmAddrInfo, static_cast<int32_t>(expertIdx), bufferConfig);
  1081          }
  1082  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

### `gmm1_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:487` in `MegaMoeImpl::Detail::AicComputeGeneric`; `before` target line `487`.
- Cardinality: first actually assigned GMM1 tile per physical AIC core per captured routed-expert wave
- Boundary claim: `causal` — Dispatch GM readiness polling and any outstanding AIC-to-AIV0 acknowledgement have passed and the first GMM1 tile is about to construct and issue MMAD work.
- Why: Distinguishes Dispatch-ready and prior ping-pong dependency time from actual first GMM1 MMAD issue on each AIC.
- Dependency: WaitForUpstreamReady observed the complete Dispatch wave count and any WaitForVector dependency for this owner completed
- Pipeline proof: `Scalar GM visibility and FIX` → `Scalar` using `trace-only FIX_S using recycled EVENT_ID0 before the first sampled tile` — The hook is after both upstream readiness and ping-pong waits in AicComputeGeneric; FIX_S transfers the FIX wait/completion dependency to Scalar before the clock.
- Perturbation: One conditional slot check, one FIX-to-Scalar bridge, and one clock per active AIC/wave.
- Source SHA-256: `c6ab40df3abbd4b814bfb026ee396c702a7b4582892510c319b2a1d81ae3d6c0`

```text
   485          }
   486  #endif
   487          typename BlockMmad::BlockShape singleShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
   488                                                     Get<K_VALUE>(actualShape), 0};
   489
```

### `gmm1_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2435` in `MegaMoeWave::ProcessRoutedExpertWaves`; `after` target line `2435`.
- Cardinality: after the last GMM1 tile assigned to each physical AIC core in a captured routed-expert wave
- Boundary claim: `actual` — All GMM1 FIX work assigned to this AIC in the wave is complete.
- Why: Makes the GMM1 interval end at actual local FIX completion instead of helper return or AIV notification issue.
- Dependency: ProcessGmm1Wave returned after issuing every owned GMM1 tile
- Pipeline proof: `FIX` → `Scalar` using `trace-only FIX_S using recycled EVENT_ID0 when this core recorded GMM1 work` — The bridge is after the complete wave loop and executes only for records with a causal begin, so its Scalar clock cannot run ahead of the final FIX operation.
- Perturbation: One FIX-to-Scalar completion bridge and clock per active AIC/wave.
- Source SHA-256: `7066bb0b6b7d6cfb2f93c5106e30158f413f914cb9702923e9e8d37b47865709`

```text
  2433
  2434          ProcessGmm1Wave(
  2435              batchBegin, batchEnd, gmm1State, gmm1AddrInfo, vecSetSyncCom);
  2436  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  2437          if constexpr (g_coreType == AIC) {
```

### `swiglu_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/block_epilogue_swiglu_mx_quant.h:1156` in `BlockEpilogueSwigluMxQuant::operator()`; `before` target line `1156`.
- Cardinality: first non-empty epilogue tile per physical AIV0 core per captured routed-expert wave
- Boundary claim: `causal` — A non-empty epilogue invocation has all upstream GMM1/input dependencies ordered into Vector and is about to start SwiGLU/quant setup and Vector work.
- Why: Places the common SwiGLU begin after both non-prefetch WaitForCube and prefetch MTE2_V input readiness, regardless of the current epilogue branch.
- Dependency: the caller passed its AIC-to-AIV0 V wait or MTE2_V data-ready wait before entering the epilogue
- Pipeline proof: `Vector` → `Scalar` using `trace-only V_S using recycled EVENT_ID0 at the common epilogue entry` — Every routed wave epilogue branch calls this operator only after its readiness wait; V_S orders those V-pipeline waits before the first recorded clock.
- Perturbation: One conditional slot check, one Vector-to-Scalar bridge, and one clock per active AIV0/wave.
- Source SHA-256: `7ffbc82e32cc4b9e533fd09a427abd5de799855d5c5aef45b61c851bf336fe04`

```text
  1154      }
  1155  #endif
  1156      vlForHalfNumber_ = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t); // 256 / 2 = 128
  1157      UBBlockSize_ = BLOCK_SIZE;                                         // 32
  1158      elementAfterReduce_ = AscendC::VECTOR_REG_WIDTH / UBBlockSize_;    // 256 / 32 = 8
```

### `swiglu_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2435` in `MegaMoeWave::ProcessRoutedExpertWaves`; `after` target line `2435`.
- Cardinality: after the last epilogue tile assigned to each physical AIV0 core in a captured routed-expert wave
- Boundary claim: `actual` — Every SwiGLU/quant GM output owned by this AIV0 in the wave has completed.
- Why: Shows the actual AIV0 epilogue completion and its overlap with AIC GMM2 and AIV1 Dispatch/Combine.
- Dependency: all BlockEpilogueSwigluMxQuant invocations assigned to the core returned
- Pipeline proof: `MTE3` → `Scalar` using `existing per-tile MTE3_S in BlockEpilogueSwigluMxQuant::operator(); no added end bridge` — Each operator return consumes MTE3_V and MTE3_S after data/scale copies and then publishes its ready count; the outer wave return therefore follows the last actual GM completion.
- Perturbation: One clock and named assignment per active AIV0/wave; no additional completion serialization.
- Source SHA-256: `7066bb0b6b7d6cfb2f93c5106e30158f413f914cb9702923e9e8d37b47865709`

```text
  2433
  2434          ProcessGmm1Wave(
  2435              batchBegin, batchEnd, gmm1State, gmm1AddrInfo, vecSetSyncCom);
  2436  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  2437          if constexpr (g_coreType == AIC) {
```

### `gmm2_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:487` in `MegaMoeImpl::Detail::AicComputeGeneric`; `before` target line `487`.
- Cardinality: first actually assigned GMM2 tile per physical AIC core per captured routed-expert wave
- Boundary claim: `causal` — Scalar GM polling has observed the required SwiGLU-ready count for the first owned GMM2 wave tile and MMAD issue is next.
- Why: Measures GMM2 only after the current wave's SwiGLU-ready counter is visible, leaving publication-to-observation latency outside GMM2 compute.
- Dependency: WaitForUpstreamReady observed the wave-grained swigluToGmm2Flag target
- Pipeline proof: `MTE3 completion published through Scalar-visible GM AtomicAdd` → `Scalar` using `none` — The bypass GM polling loop runs on Scalar and returns only after epilogue MTE3_S and ready AtomicAdd; the immediately following Scalar clock is causally ordered without a cross-pipeline bridge.
- Perturbation: One conditional slot check and one clock per active AIC/wave.
- Source SHA-256: `c6ab40df3abbd4b814bfb026ee396c702a7b4582892510c319b2a1d81ae3d6c0`

```text
   485          }
   486  #endif
   487          typename BlockMmad::BlockShape singleShape{Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape),
   488                                                     Get<K_VALUE>(actualShape), 0};
   489
```

### `gmm2_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2452` in `MegaMoeWave::ProcessRoutedExpertWaves`; `after` target line `2452`.
- Cardinality: after the last ready publication for GMM2 work assigned to each physical AIC core in a captured routed-expert wave
- Boundary claim: `actual` — All owned GMM2 FIX writes for the wave are complete and the final per-expert ready publication has been issued by this AIC.
- Why: Proves local GMM2 direct-GM completion and ready publication timing while keeping remote AIV1 observation as a separate handoff gap.
- Dependency: ProcessGmm2Wave completed every expert and PublishGmm2Ready ran after each expert
- Pipeline proof: `FIX then Scalar bypass GM store` → `Scalar` using `existing FIX_S inside PublishGmm2Ready; no extra bridge` — PublishGmm2Ready executes FIX_S before WriteGmByPassDCache; the outer sample follows the last publication in source and Scalar order. It does not claim AIV1 has observed it.
- Perturbation: One clock and named assignment per active AIC/wave; existing production FIX_S publication semantics are reused.
- Source SHA-256: `7066bb0b6b7d6cfb2f93c5106e30158f413f914cb9702923e9e8d37b47865709`

```text
  2450          const uint32_t gmm1EndCursor = startBlockIdx_;
  2451
  2452          ProcessGmm2Wave(batchBegin, batchEnd, gmm2State, gmm2AddrInfo, vecSetSyncCom);
  2453  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  2454          if constexpr (g_coreType == AIC) {
```

### `combine_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2343` in `MegaMoeWave::ProcessCombineExperts`; `after` target line `2343`.
- Cardinality: first non-empty token range per physical AIV1 core per captured routed-expert wave
- Boundary claim: `causal` — This AIV1 owns at least one token and its first expert's GMM2-ready scan has observed all AIC publications; meta-info load and first row send follow.
- Why: Exposes the publication-to-ready-observation handoff and starts Combine only after the full AIC fan-in is proven.
- Dependency: WaitGmm2Ready reduced all per-AIC ready cache lines to at least blockNum
- Pipeline proof: `MTE2 then Vector reduction` → `Scalar` using `existing MTE2_V and V_S in WaitGmm2Ready` — WaitGmm2Ready returns only after the V_S-ordered sum is read on Scalar and meets the all-AIC target; the marker is immediately after that return.
- Perturbation: One conditional slot check and one clock per active AIV1/wave.
- Source SHA-256: `7066bb0b6b7d6cfb2f93c5106e30158f413f914cb9702923e9e8d37b47865709`

```text
  2341              GetCombineOwnedRange(currentExpertTokenNum);
  2342          if (currentCoreTokenRange.count != 0U) {
  2343              WaitGmm2Ready(expertIdx);
  2344  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  2345              if (CriticalPathTracePtr() != nullptr && criticalPathTrace_.NeedsStage1Begin()) {
```

### `combine_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2371` in `MegaMoeWave::ProcessCombineExperts`; `after` target line `2371`.
- Cardinality: after the final remote-send ring drain per physical AIV1 core per captured routed-expert wave
- Boundary claim: `actual` — Every token row issued by this AIV1 for the wave has completed its remote MTE3 send.
- Why: Ends Combine at actual remote MTE3 send completion rather than at Scalar issue or ring-drain enqueue time.
- Dependency: DrainCombineRowRing consumed every MTE3_MTE2 event for each actually used row slot
- Pipeline proof: `MTE3 completion dependency held by MTE2` → `Scalar` using `trace-only MTE2_S using recycled EVENT_ID0 after the drain` — The ring drain waits only the slots that were issued; MTE2_S transfers those completed MTE3 dependencies to Scalar before sampling.
- Perturbation: One MTE2-to-Scalar bridge and one clock per active AIV1/wave.
- Source SHA-256: `7066bb0b6b7d6cfb2f93c5106e30158f413f914cb9702923e9e8d37b47865709`

```text
  2369
  2370      // 下一 wave 的 dispatch 会复用同一片 UB 和事件编号，离开前必须回收所有已使用槽位。
  2371      DrainCombineRowRing(rowSequence);
  2372  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  2373      if (rowSequence != 0U && CriticalPathTracePtr() != nullptr && criticalPathTrace_.HasStage1Work()) {
```

## Working tree status

```text
M  mc2/mega_moe/op_kernel/arch35/block_epilogue_swiglu_mx_quant.h
M  mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h
A  mc2/mega_moe/op_kernel/arch35/mega_moe_trace.h
M  mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h
A  mc2/mega_moe/tools/critical_path_trace/README.md
A  mc2/mega_moe/tools/critical_path_trace/artifacts/synthetic/index.html
A  mc2/mega_moe/tools/critical_path_trace/artifacts/synthetic/rank0.contract-check.json
A  mc2/mega_moe/tools/critical_path_trace/artifacts/synthetic/rank0.html
A  mc2/mega_moe/tools/critical_path_trace/artifacts/synthetic/rank0.normalized.json
A  mc2/mega_moe/tools/critical_path_trace/artifacts/synthetic/rank0.order-check.json
A  mc2/mega_moe/tools/critical_path_trace/artifacts/synthetic/rank0.raw
A  mc2/mega_moe/tools/critical_path_trace/artifacts/synthetic/rank0.runtime.json
A  mc2/mega_moe/tools/critical_path_trace/artifacts/synthetic/run-manifest.json
A  mc2/mega_moe/tools/critical_path_trace/audit-implemented/trace_point_audit.json
A  mc2/mega_moe/tools/critical_path_trace/audit-implemented/trace_point_audit.md
A  mc2/mega_moe/tools/critical_path_trace/audit-plan/trace_point_audit.json
A  mc2/mega_moe/tools/critical_path_trace/audit-plan/trace_point_audit.md
A  mc2/mega_moe/tools/critical_path_trace/generate_synthetic_fixture.py
A  mc2/mega_moe/tools/critical_path_trace/mega_moe_trace_contract.py
A  mc2/mega_moe/tools/critical_path_trace/run_critical_path_trace.py
A  mc2/mega_moe/tools/critical_path_trace/test_critical_path_trace.py
A  mc2/mega_moe/tools/critical_path_trace/trace-spec.json
A  mc2/mega_moe/tools/critical_path_trace/trace_tool.py
A  mc2/mega_moe/tools/critical_path_trace/validation-report.json
A  mc2/mega_moe/tools/critical_path_trace/verify_trace_off.py
```
