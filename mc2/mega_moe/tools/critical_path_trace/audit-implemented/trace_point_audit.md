# Trace point audit

- Status: `pass`
- Phase: `implemented`
- Trace: `mega-moe-wave-critical-path`
- Repository: `/home/mace/workspace/ops-transformer_megaMoeOpt_trace_20260803`
- Revision: `b25081aa4d333037c7fc898b678e98c74eafdaf7`
- Branch: `codex/mega-moe-critical-path-trace-20260803`
- Dirty: `True`
- Spec SHA-256: `45b0e6fd94f863d9be44a89b3bdda7bf042306630688adbc3c11ed8bc31a4a86`

## Placement summary

| Point | Roles | Resolved location | Target line | Placement | Boundary | Bridge | Why |
|---|---|---|---:|---|---|---|---|
| process_begin_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2715 | 2719 | before | causal | none | Provides one local shared-axis reference for AIC, AIV0, and AIV1 inside the same physical AI Block, without asserting clock alignment across AI Blocks or ranks; also exposes role launch skew before input preparation. |
| work_end_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2775 | 2772 | after | actual | existing PublishGmm2Ready FIX_S and Unpermute final MTE3_V waits; no additional hot-path bridge | Separates the real role-local useful-work tail from the diagnostic rendezvous and y record emission overhead. |
| dispatch_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:1064 | 1083 | before | causal | none | Measures the actual per-wave AIV1 prefetch window, including send-count calculation, rather than the outer wave that may prefetch two waves at once. |
| dispatch_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:1089 | 1086 | after | actual | trace-only MTE3_S using recycled EVENT_ID0 on the last expert of the wave | Proves when each AIV1 has completed the token/scale/meta-info writes that release GMM1 and permits direct comparison with GMM1 readiness. |
| gmm1_expert_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:482 | 467 | after | causal | trace-only FIX_S using recycled EVENT_ID0 before the first sampled tile | 把一批专家的长 GMM1 外框拆成逐专家 issue 区间，使同一 wave 内专家间的 FIX 尾部和依赖等待可以单独显示。 |
| gmm1_expert_end_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2319 | 2313 | after | issue | none; intentionally issue-only | 给逐专家 GMM1 一个低扰动结束边界；与下一专家 causal begin 之间显示 FIX 尾部、等待和调度间隙。 |
| swiglu_expert_begin_S | AIV0 | mc2/mega_moe/op_kernel/arch35/block_epilogue_swiglu_mx_quant.h:1156 | 1175 | before | causal | trace-only V_S using recycled EVENT_ID0 at the common epilogue entry | 把一批专家合并的 SwiGLU 长外框拆开，避免把专家间 WaitForCube/输入等待误认为 SwiGLU 计算。 |
| swiglu_expert_end_S | AIV0 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2323 | 2313 | after | actual | existing per-tile MTE3_S; no added end bridge | 标出每个专家真实 SwiGLU/MX 输出完成点，使后续到下一专家 causal begin 的等待/调度间隙可见。 |
| gmm2_expert_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:486 | 467 | after | causal | none | 拆出逐专家 GMM2，使 SwiGLU-ready 等待不再被一批专家的外框吞入；也可验证 GMM2 是否确实近似连续输出 GM。 |
| gmm2_expert_end_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2364 | 2358 | after | actual | existing FIX_S inside PublishGmm2Ready; no extra bridge | 逐专家证明 GMM2 直写 GM 与 ready 发布完成，并把到下一专家 begin 的等待/调度间隙独立显示。 |
| combine_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2409 | 2407 | after | causal | existing MTE2_V and V_S in WaitGmm2Ready | Exposes the publication-to-ready-observation handoff and starts Combine only after the full AIC fan-in is proven. |
| combine_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2443 | 2441 | after | actual | trace-only MTE2_S using recycled EVENT_ID0 after the drain | Ends Combine at actual remote MTE3 send completion rather than at Scalar issue or ring-drain enqueue time. |

## Point evidence

### `process_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2715` in `MegaMoeWave::Process`; `before` target line `2719`.
- Cardinality: once at MegaMoeWave::Process entry per physical role/core record
- Boundary claim: `causal` — Scalar has entered the current MTE A8W8 MegaMoeWave Process body and no operator work has yet been issued by this role/core.
- Why: Provides one local shared-axis reference for AIC, AIV0, and AIV1 inside the same physical AI Block, without asserting clock alignment across AI Blocks or ranks; also exposes role launch skew before input preparation.
- Dependency: mixed-kernel task entry for this physical role/core
- Pipeline proof: `kernel scheduler and Scalar control` → `Scalar` using `none` — The clock is the first diagnostic instruction immediately before the first Process statement.
- Perturbation: One 32-bit GetSystemCycle and one named Scalar assignment per role/core.
- Source SHA-256: `27bc1f7ee949ab4aa22a73df3014dd0ccf441670309250ea4ab8f6367b9ab00a`

```text
  2713  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  2714      if (CriticalPathTracePtr() != nullptr) {
  2715          criticalPathTrace_.processBegin = CriticalPathTrace::Now(); // TRACE_POINT: process_begin_S
  2716      }
  2717  #endif
```

### `work_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2775` in `MegaMoeWave::Process`; `after` target line `2772`.
- Cardinality: once after useful work and overflow-mode restoration per physical role/core record
- Boundary claim: `actual` — All useful work owned by this role is complete; for AIV roles Unpermute has consumed its final MTE3_V completion event, and routed AIC publications already passed FIX_S.
- Why: Separates the real role-local useful-work tail from the diagnostic rendezvous and y record emission overhead.
- Dependency: role-local GMM/Combine completion and, on AIV, final Unpermute y store completion
- Pipeline proof: `FIX and Scalar for AIC; MTE3, Vector, and Scalar for AIV` → `Scalar` using `existing PublishGmm2Ready FIX_S and Unpermute final MTE3_V waits; no additional hot-path bridge` — The sample is after ProcessRoutedExpertWaves, cross-rank completion, Unpermute's final MTE3_V wait, and control-state restoration, but before either trace rendezvous.
- Perturbation: One 32-bit GetSystemCycle and assignment before trace-only terminal synchronization.
- Source SHA-256: `27bc1f7ee949ab4aa22a73df3014dd0ccf441670309250ea4ab8f6367b9ab00a`

```text
  2773  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  2774      if (CriticalPathTracePtr() != nullptr) {
  2775          // TRACE_POINT: work_end_S
  2776          criticalPathTrace_.workEnd = CriticalPathTrace::Now();
  2777          SyncAll<false>();
```

### `dispatch_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:1064` in `MegaMoeWave::DispatchExpert`; `before` target line `1083`.
- Cardinality: 每个物理 AIV1、每个实际 routed wave 一次；写入该 wave 首专家 index 对应的动态槽
- Boundary claim: `causal` — The first expert in this dispatch wave is about to start SendCntCal on the AIV1 Scalar/control path.
- Why: Measures the actual per-wave AIV1 prefetch window, including send-count calculation, rather than the outer wave that may prefetch two waves at once.
- Dependency: the monotonic dispatch cursor selected this expert and any prior Combine ring reuse was drained by the preceding wave
- Pipeline proof: `AIV1 Scalar control` → `Scalar` using `none` — DispatchExpert is invoked exactly once per expert by DispatchExpertsUntil; the boundary cursor advances with GetWaveExpertCount, including the latest small-tail merge rule.
- Perturbation: 每个 AIV1/wave 一次动态边界判断和时钟读取；结束时追加三次互不冲突的 workspace bypass-GM 标量写。
- Source SHA-256: `27bc1f7ee949ab4aa22a73df3014dd0ccf441670309250ea4ab8f6367b9ab00a`

```text
  1062          }
  1063  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1064          // TRACE_POINT: dispatch_begin_S
  1065  #if MEGA_MOE_ENABLE_FINE_GRAIN_CRITICAL_PATH_TRACE != 0
  1066          if (CriticalPathTracePtr() != nullptr && expertIdx == criticalPathDispatchWaveEnd_) {
```

### `dispatch_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:1089` in `MegaMoeWave::DispatchExpert`; `after` target line `1086`.
- Cardinality: 每个物理 AIV1、每个实际 routed wave 的最后一个专家一次；与 begin 共用 wave 首专家 index
- Boundary claim: `actual` — The last expert handled by this AIV1 in the wave has completed all Dispatch MTE3 writes and ready publications.
- Why: Proves when each AIV1 has completed the token/scale/meta-info writes that release GMM1 and permits direct comparison with GMM1 readiness.
- Dependency: CopyGMToGMPerToken has recycled every dynamic ring slot and MetaInfoCalAndDispatch has issued its final publication
- Pipeline proof: `MTE3` → `Scalar` using `trace-only MTE3_S using recycled EVENT_ID0 on the last expert of the wave` — All EVENT_ID0..5 Dispatch ring lifetimes are paired before MetaInfoCalAndDispatch returns; the added MTE3_S orders the last output/publication into Scalar before sampling.
- Perturbation: One MTE3-to-Scalar bridge and one clock sample per active AIV1/wave.
- Source SHA-256: `27bc1f7ee949ab4aa22a73df3014dd0ccf441670309250ea4ab8f6367b9ab00a`

```text
  1087          }
  1088  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1089          // TRACE_POINT: dispatch_end_S
  1090  #if MEGA_MOE_ENABLE_FINE_GRAIN_CRITICAL_PATH_TRACE != 0
  1091          if (expertIdx + 1U == criticalPathDispatchWaveEnd_ && CriticalPathTracePtr() != nullptr &&
```

### `gmm1_expert_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:482` in `MegaMoeImpl::Detail::AicComputeGeneric`; `after` target line `467`.
- Cardinality: 每个物理 AIC 对实际分配到 tile 的 routed expert 各记录一次，数组下标等于本卡 expert index
- Boundary claim: `causal` — 当前专家的 Dispatch GM readiness 与前序 ping-pong 依赖已满足，本核第一个 GMM1 tile 即将发出 MMAD。
- Why: 把一批专家的长 GMM1 外框拆成逐专家 issue 区间，使同一 wave 内专家间的 FIX 尾部和依赖等待可以单独显示。
- Dependency: WaitForUpstreamReady 已观察当前专家 Dispatch wave count，且需要时 WaitForVector 已完成
- Pipeline proof: `Scalar GM visibility and FIX` → `Scalar` using `trace-only FIX_S using recycled EVENT_ID0 before the first sampled tile` — 打点位于 AicComputeGeneric 的 readiness/ping-pong waits 之后；FIX_S 仅把该 causal begin 依赖传到 Scalar。
- Perturbation: 每个活跃专家一次动态状态检查、一次 FIX→Scalar causal bridge、一次时钟读取；结束时追加三次互不冲突的 workspace bypass-GM 标量写。
- Source SHA-256: `5e8084d1de9143df589dbad8084b4f2626bf6c8732852881f918048c21fd3cab`

```text
   480                  AscendC::SetFlag<AscendC::HardEvent::FIX_S>(EVENT_ID0);
   481                  AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(EVENT_ID0);
   482                  // TRACE_POINT: gmm1_expert_begin_S
   483                  args.criticalPathTrace->BeginExpertStage0(CriticalPathTrace::Now());
   484              }
```

### `gmm1_expert_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2319` in `MegaMoeWave::ProcessGmm1Wave`; `after` target line `2313`.
- Cardinality: 每个已捕获且本核实际参与的 routed expert，在 GroupMatmulWithSwigluQuant 返回后记录一次
- Boundary claim: `issue` — 当前专家本核所有 GMM1 MMAD/NotifyVector 已发出，但不额外声称 FIX 已完成。
- Why: 给逐专家 GMM1 一个低扰动结束边界；与下一专家 causal begin 之间显示 FIX 尾部、等待和调度间隙。
- Dependency: GroupMatmulWithSwigluQuant 已返回，当前专家分配给本核的 tile 已全部发出
- Pipeline proof: `Scalar issue state; FIX may still be pending` → `Scalar` using `none; intentionally issue-only` — 采样紧跟当前专家 helper 返回；不增加逐专家 FIX_S，避免把原本可流水的 FIX 尾部强制串行化。
- Perturbation: 每个活跃专家一次时钟读取和三次 workspace 标量写；无新增完成 bridge。该 span 以 issue/半透明显示。
- Source SHA-256: `27bc1f7ee949ab4aa22a73df3014dd0ccf441670309250ea4ab8f6367b9ab00a`

```text
  2317              if constexpr (g_coreType == AIC) {
  2318                  if (criticalPathTrace_.HasExpertStage0Work()) {
  2319                      // TRACE_POINT: gmm1_expert_end_S
  2320                      criticalPathTrace_.EndExpertStage0(CriticalPathTrace::Now());
  2321                  }
```

### `swiglu_expert_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/block_epilogue_swiglu_mx_quant.h:1156` in `BlockEpilogueSwigluMxQuant::operator()`; `before` target line `1175`.
- Cardinality: 每个物理 AIV0 对实际执行 epilogue 的 routed expert 各记录一次；同一专家后续 tile 复用其 expert-index 槽
- Boundary claim: `causal` — 当前专家首个非空 epilogue 已完成上游 GMM1/输入就绪等待，并即将开始 SwiGLU 与 MX 量化。
- Why: 把一批专家合并的 SwiGLU 长外框拆开，避免把专家间 WaitForCube/输入等待误认为 SwiGLU 计算。
- Dependency: the caller passed its AIC-to-AIV0 V wait or MTE2_V data-ready wait before entering the epilogue
- Pipeline proof: `Vector` → `Scalar` using `trace-only V_S using recycled EVENT_ID0 at the common epilogue entry` — Every routed wave epilogue branch calls this operator only after its readiness wait; V_S orders those V-pipeline waits before the first recorded clock.
- Perturbation: 每个被捕获活跃专家一次槽检查、一次 V→Scalar causal bridge 和一次时钟读取；同专家后续 tile 不重复 begin。
- Source SHA-256: `36c3efd3d9a096b605f12d30e6577a0448a602d8cb6ee30c4ccde9400280dc85`

```text
  1154          AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
  1155          AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
  1156          // TRACE_POINT: swiglu_expert_begin_S
  1157          criticalPathTrace_->BeginExpertStage0(MegaMoeImpl::CriticalPathTrace::Now());
  1158      }
```

### `swiglu_expert_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2323` in `MegaMoeWave::ProcessGmm1Wave`; `after` target line `2313`.
- Cardinality: 每个已捕获且本核实际执行 epilogue 的 routed expert，在 GroupMatmulWithSwigluQuant 返回后记录一次
- Boundary claim: `actual` — 当前专家在本 AIV0 上的全部 epilogue tile 已返回；其量化 data/scale MTE3 写回均已完成且 ready 均已发布。
- Why: 标出每个专家真实 SwiGLU/MX 输出完成点，使后续到下一专家 causal begin 的等待/调度间隙可见。
- Dependency: 当前专家的全部 BlockEpilogueSwigluMxQuant::operator() 调用已返回
- Pipeline proof: `MTE3 completion and Scalar ready publication` → `Scalar` using `existing per-tile MTE3_S; no added end bridge` — 每个 operator 返回前已有 MTE3_S 并随后发布 ready；helper 返回后的 Scalar 时钟因此晚于当前专家最后一个 tile 的完成与发布。
- Perturbation: 每个被捕获活跃专家一次条件检查和结束时钟；不新增 MTE3 bridge，不再逐 tile 重复采样。
- Source SHA-256: `27bc1f7ee949ab4aa22a73df3014dd0ccf441670309250ea4ab8f6367b9ab00a`

```text
  2321                  }
  2322              } else if (subBlockIdx_ == 0U && criticalPathTrace_.HasExpertStage0Work()) {
  2323                  // TRACE_POINT: swiglu_expert_end_S
  2324                  criticalPathTrace_.EndExpertStage0(CriticalPathTrace::Now());
  2325              }
```

### `gmm2_expert_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h:486` in `MegaMoeImpl::Detail::AicComputeGeneric`; `after` target line `467`.
- Cardinality: 每个物理 AIC 对实际分配到 GMM2 tile 的 routed expert 各记录一次，数组下标等于本卡 expert index
- Boundary claim: `causal` — 当前专家的 SwiGLU-ready counter 已被 Scalar 观察，本核第一个 GMM2 MMAD 即将发出。
- Why: 拆出逐专家 GMM2，使 SwiGLU-ready 等待不再被一批专家的外框吞入；也可验证 GMM2 是否确实近似连续输出 GM。
- Dependency: WaitForUpstreamReady observed the wave-grained swigluToGmm2Flag target
- Pipeline proof: `MTE3 completion published through Scalar-visible GM AtomicAdd` → `Scalar` using `none` — The bypass GM polling loop runs on Scalar and returns only after epilogue MTE3_S and ready AtomicAdd; the immediately following Scalar clock is causally ordered without a cross-pipeline bridge.
- Perturbation: 每个活跃专家一次动态状态检查和时钟读取；不新增跨流水 bridge。
- Source SHA-256: `5e8084d1de9143df589dbad8084b4f2626bf6c8732852881f918048c21fd3cab`

```text
   484              }
   485          } else if (args.criticalPathTrace != nullptr && args.criticalPathTrace->NeedsExpertStage1Begin()) {
   486              // TRACE_POINT: gmm2_expert_begin_S
   487              args.criticalPathTrace->BeginExpertStage1(CriticalPathTrace::Now());
   488          }
```

### `gmm2_expert_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2364` in `MegaMoeWave::ProcessGmm2Wave`; `after` target line `2358`.
- Cardinality: 每个已捕获且本核实际参与的 routed expert，在 PublishGmm2Ready 后记录实际完成
- Boundary claim: `actual` — 当前专家本核所有 GMM2 FIX 直写已完成，且该 AIC 的 per-expert ready 已发布。
- Why: 逐专家证明 GMM2 直写 GM 与 ready 发布完成，并把到下一专家 begin 的等待/调度间隙独立显示。
- Dependency: GroupMatmulGmm2 已发出当前专家所有 tile，PublishGmm2Ready 随后执行
- Pipeline proof: `FIX then Scalar bypass GM store` → `Scalar` using `existing FIX_S inside PublishGmm2Ready; no extra bridge` — PublishGmm2Ready 内现有 FIX_S 在 WriteGmByPassDCache 前证明本核 FIX 完成；紧随其后的 Scalar 时钟不声称 AIV1 已观察 ready。
- Perturbation: 每个活跃专家一次时钟读取和三次 workspace 标量写；复用生产路径已有 FIX_S，不新增 completion bridge。
- Source SHA-256: `27bc1f7ee949ab4aa22a73df3014dd0ccf441670309250ea4ab8f6367b9ab00a`

```text
  2362              if constexpr (g_coreType == AIC) {
  2363                  if (criticalPathTrace_.HasExpertStage1Work()) {
  2364                      // TRACE_POINT: gmm2_expert_end_S
  2365                      criticalPathTrace_.EndExpertStage1(CriticalPathTrace::Now());
  2366                  }
```

### `combine_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2409` in `MegaMoeWave::ProcessCombineExperts`; `after` target line `2407`.
- Cardinality: 每个物理 AIV1、每个实际 routed wave 的首个非空 token range；写入该 wave 首专家 index
- Boundary claim: `causal` — This AIV1 owns at least one token and its first expert's GMM2-ready scan has observed all AIC publications; meta-info load and first row send follow.
- Why: Exposes the publication-to-ready-observation handoff and starts Combine only after the full AIC fan-in is proven.
- Dependency: WaitGmm2Ready reduced all per-AIC ready cache lines to at least blockNum
- Pipeline proof: `MTE2 then Vector reduction` → `Scalar` using `existing MTE2_V and V_S in WaitGmm2Ready` — WaitGmm2Ready returns only after the V_S-ordered sum is read on Scalar and meets the all-AIC target; the marker is immediately after that return.
- Perturbation: One conditional slot check and one clock per active AIV1/wave.
- Source SHA-256: `27bc1f7ee949ab4aa22a73df3014dd0ccf441670309250ea4ab8f6367b9ab00a`

```text
  2407              WaitGmm2Ready(expertIdx);
  2408  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  2409              // TRACE_POINT: combine_begin_S
  2410  #if MEGA_MOE_ENABLE_FINE_GRAIN_CRITICAL_PATH_TRACE != 0
  2411              if (CriticalPathTracePtr() != nullptr && criticalPathTrace_.NeedsExpertStage1Begin()) {
```

### `combine_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h:2443` in `MegaMoeWave::ProcessCombineExperts`; `after` target line `2441`.
- Cardinality: 每个物理 AIV1、每个实际 routed wave 的最终远端发送 ring drain 后一次
- Boundary claim: `actual` — Every token row issued by this AIV1 for the wave has completed its remote MTE3 send.
- Why: Ends Combine at actual remote MTE3 send completion rather than at Scalar issue or ring-drain enqueue time.
- Dependency: DrainCombineRowRing consumed every MTE3_MTE2 event for each actually used row slot
- Pipeline proof: `MTE3 completion dependency held by MTE2` → `Scalar` using `trace-only MTE2_S using recycled EVENT_ID0 after the drain` — The ring drain waits only the slots that were issued; MTE2_S transfers those completed MTE3 dependencies to Scalar before sampling.
- Perturbation: One MTE2-to-Scalar bridge and one clock per active AIV1/wave.
- Source SHA-256: `27bc1f7ee949ab4aa22a73df3014dd0ccf441670309250ea4ab8f6367b9ab00a`

```text
  2441      DrainCombineRowRing(rowSequence);
  2442  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  2443      // TRACE_POINT: combine_end_S
  2444  #if MEGA_MOE_ENABLE_FINE_GRAIN_CRITICAL_PATH_TRACE != 0
  2445      if (rowSequence != 0U && CriticalPathTracePtr() != nullptr &&
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
