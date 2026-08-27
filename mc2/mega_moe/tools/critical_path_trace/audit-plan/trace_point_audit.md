# Trace point audit

- Status: `pass`
- Phase: `plan`
- Trace: `mega-moe-pr10302-token-group-wave-active-task-critical-path`
- Repository: `/home/mace/workspace/ops-transformer_megaMoeOpt`
- Revision: `d60c90ed7310181b6398d83a3033ae6b2f08c6ed`
- Branch: `codex/pr10302-wave-interleave-sendindex-critical-path-trace`
- Dirty: `True`
- Spec SHA-256: `00ccd2b5a5a0faa7a007b23705856617f398f635fcb8758e2c6340187023f641`

## Placement summary

| Point | Roles | Resolved location | Target line | Placement | Boundary | Bridge | Why |
|---|---|---|---:|---|---|---|---|
| process_begin_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1293 | 1293 | before | causal | program order | 提供同一 AI Block 内三角色共同参考起点。 |
| input_buffer_init_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1303 | 1303 | before | causal | program order | 分离输入准备 UB 布局与后续 token 量化。 |
| input_buffer_init_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1303 | 1303 | after | issue | program order only | 量化前单独显示 UB 地址构造和零张量发出成本。 |
| input_quantize_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1323 | 1323 | before | causal | program order | 量化本卡输入 token 的固定成本独立可见。 |
| input_quantize_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1323 | 1323 | after | actual | production MTE3_MTE2 ring drain | 区分 token 量化与后续 compact route-index 通信。 |
| route_index_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1335 | 1335 | before | causal | program order | 单独量化 compact route index 构造、count 整理与跨 rank 发送。 |
| route_index_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1335 | 1335 | after | actual | production ring drain | 把 compact route index/count 远端发送完成与 workspace 清零分开。 |
| workspace_reset_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1347 | 1347 | before | causal | program order | 量化 flag/workspace 清零成本和同步等待。 |
| workspace_reset_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1347 | 1347 | after | actual | tracing-only MTE3_S | 不把未完成的 MTE3 清零尾部错误归入后续本地 barrier。 |
| local_prepare_sync_begin_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1372 | 1372 | before | causal | program order | 直接量化 AIC 等待 AIV 输入准备和 flag 可见的本地 barrier。 |
| local_prepare_sync_end_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1372 | 1372 | after | actual | production rendezvous | 区分单卡角色同步与随后跨 rank 同步。 |
| cross_rank_sync_begin_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1113 | 1113 | before | causal | program order | 直接量化 Dispatch 前由最慢 rank 决定的全卡等待。 |
| cross_rank_sync_end_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1113 | 1113 | after | actual | production remote barrier | 把全 rank 等待与本地专家 buffer 初始化分开。 |
| expert_buffer_init_begin_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:990 | 990 | before | causal | program order | 量化进入首轮 Dispatch/GMM 前的专家阶段 buffer 和状态初始化。 |
| expert_buffer_init_end_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:994 | 994 | after | actual | program order | 分离本地视图构造与后续 cursor/problem 状态建立及首个 Dispatch。 |
| work_end_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1393 | 1393 | after | actual | production completion plus program order | 把有效工作与 tracing 终结同步/落盘分离。 |
| dispatch_task_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_token_dispatch.h:624 | 624 | before | causal | program order | 只绘制实际拥有 token 行的 Dispatch helper，排除 count/cursor 和无工作专家。 |
| dispatch_task_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_token_dispatch.h:624 | 624 | after | actual | production helper's existing ring drain and MTE3_S | 只覆盖本核实际负责的专家行 Dispatch，不包住下一专家或 cursor 推进。 |
| dispatch_scalar_gap_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_token_dispatch.h:624 | 624 | after | causal | program order after the existing Dispatch completion contract | 直接量化相邻有效 Dispatch 之间由专家 count 查询、rolling owner 规划、Wave cursor 推进以及相邻预取调用衔接产生的 Scalar 空隙。 |
| dispatch_scalar_gap_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_token_dispatch.h:624 | 624 | before | actual | none; same-pipeline program order | 判断 HTML 中 Dispatch 之间的白色区间是否就是 AIV1 的逐专家 Scalar 扫描，而不是隐藏的通信或等待。 |
| gmm1_scalar_setup_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/common/mega_moe_trace.h:289 | 289 | before | causal | program order | 判断 GMM1 空洞是否来自逐专家状态准备、地址更新和 rolling ownership 跳过，而不是通信依赖。 |
| gmm1_scalar_setup_end_S | AIC | mc2/mega_moe/op_kernel/arch35/common/mega_moe_trace.h:461 | 461 | before | actual | program order | 把 Scalar 调度与随后真实 Dispatch-ready 等待分开。 |
| gmm1_dispatch_wait_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:81 | 81 | before | causal | program order | 直接验证下一段 GMM1 是否仍被 Dispatch-ready flag 阻塞。 |
| gmm1_dispatch_wait_end_S | AIC | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:89 | 89 | before | actual | bypass-GM polling then program order | 量化 Dispatch 是否真的落在两段 GMM1 之间。 |
| gmm1_vector_wait_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:97 | 97 | before | causal | program order | 判断前一段 SwiGLU 是否尚未释放 GMM1 输出槽，从而阻塞下一段 GMM1。 |
| gmm1_vector_wait_end_S | AIC | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:105 | 105 | before | actual | production event wait then program order | 把 SwiGLU backpressure 与下一段 GMM1 实际 MMAD 分开。 |
| gmm1_problem_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:70 | 70 | before | causal | SCALAR, FIX_S, or V_S selected by production variant | 排除首 tile 的 Dispatch-ready/ping-pong 等待，并拆开 Wave 内多个 problem。 |
| gmm1_problem_end_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:779 | 779 | after | actual | tracing-only FIX_S | 只把本 problem 真正完成的 FIX 工作染色，使 problem 间白色可解释为调度/依赖空洞。 |
| swiglu_problem_begin_S | AIV0 | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:73 | 73 | before | causal | V_S or production program order | 排除 WaitForCube/ready，并拆开 Wave 内多次 SwiGLU。 |
| swiglu_problem_end_S | AIV0 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:779 | 779 | after | issue | program order only | 不把下一 problem 前的等待算入 SwiGLU。 |
| gmm2_problem_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h:519 | 519 | before | causal | bypass-GM ready polling then program order | 排除 SwiGLU-ready 轮询并拆开 Wave 内多个 problem。 |
| gmm2_problem_end_S | AIC | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:892 | 892 | after | actual | tracing-only FIX_S | 让大量短 GMM2 段覆盖真实 FIX 执行，而不是只显示 Scalar 发出。 |
| combine_group_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h:1178 | 1178 | after | causal | bypass-GM polling then program order | 直接观察本次新增的 256-token group ready 是否让 Combine 提前启动，并排除 ready 轮询。 |
| combine_group_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h:1207 | 1207 | before | actual | tracing-only MTE3_S | 只覆盖当前 256-token group 切片，并在远端发送完成后结束。 |

## Point evidence

### `process_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1293` in `MegaMoeA8W8Wave::Process`; `before` target line `1293`.
- Cardinality: 每个物理 AI Block 的 AIC/AIV0/AIV1 各一次。
- Boundary claim: `causal` — Process 的第一条诊断指令，尚未执行正常算子语句。
- Why: 提供同一 AI Block 内三角色共同参考起点。
- Dependency: kernel role entered Process
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker precedes the first production statement
- Perturbation: 每角色一次时钟读取和寄存器赋值。
- Source SHA-256: `3e12d159f0a60c9c7af4ea6e4fd068e9cc980867caf40eb785b9dd3efcaca0a2`

```text
  1291  #endif
  1292      // 保存入口时的溢出模式，并初始化输入准备阶段使用的 UB。
  1293      int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
  1294      SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
  1295  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

### `input_buffer_init_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1303` in `MegaMoeA8W8Wave::Process`; `before` target line `1303`.
- Cardinality: 每个 AIV0/AIV1 各一次。
- Boundary claim: `causal` — 即将构造 quant/route-index/reset 的 UB 视图并发出 resetTensor Duplicate。
- Why: 分离输入准备 UB 布局与后续 token 量化。
- Dependency: overflow mode configured
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker is immediately before SendAndQuantBuffInit
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `3e12d159f0a60c9c7af4ea6e4fd068e9cc980867caf40eb785b9dd3efcaca0a2`

```text
  1301      }
  1302  #endif
  1303      SendAndQuantBuffInit();
  1304  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1305      if constexpr (g_coreType == AIV) {
```

### `input_buffer_init_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1303` in `MegaMoeA8W8Wave::Process`; `after` target line `1303`.
- Cardinality: 与 input_buffer_init_begin_S 一一对应。
- Boundary claim: `issue` — UB 视图已构造且 Duplicate 已发出；Vector 尾部可继续运行。
- Why: 量化前单独显示 UB 地址构造和零张量发出成本。
- Dependency: SendAndQuantBuffInit returned
- Pipeline proof: `Scalar plus Vector issue` → `Scalar` using `program order only` — no tracing-only Vector drain is inserted
- Perturbation: 每个 AIV 一次时钟读取，不新增 Vector drain。
- Source SHA-256: `3e12d159f0a60c9c7af4ea6e4fd068e9cc980867caf40eb785b9dd3efcaca0a2`

```text
  1301      }
  1302  #endif
  1303      SendAndQuantBuffInit();
  1304  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1305      if constexpr (g_coreType == AIV) {
```

### `input_quantize_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1323` in `MegaMoeA8W8Wave::Process`; `before` target line `1323`.
- Cardinality: 每个 AIV0/AIV1 各一次。
- Boundary claim: `causal` — 即将进入 QuantizeLocalTokens。
- Why: 量化本卡输入 token 的固定成本独立可见。
- Dependency: input UB views are configured
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker precedes QuantizeLocalTokens
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `3e12d159f0a60c9c7af4ea6e4fd068e9cc980867caf40eb785b9dd3efcaca0a2`

```text
  1321      }
  1322  #endif
  1323      QuantizeLocalTokens<QuantMode, QuantOutType, ActivationType, TopkWeightsType, TopkWeightsPrefetch>(
  1324          aivJob_, commonConfig_, params_, quantProcessConfig_, quantProcessScratch_);
  1325  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

### `input_quantize_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1323` in `MegaMoeA8W8Wave::Process`; `after` target line `1323`.
- Cardinality: 与 input_quantize_begin_S 一一对应。
- Boundary claim: `actual` — 本核负责 token 的量化结果已写回 GM。
- Why: 区分 token 量化与后续 compact route-index 通信。
- Dependency: QuantizeLocalTokens returned
- Pipeline proof: `MTE2/Vector/MTE3` → `Scalar` using `production MTE3_MTE2 ring drain` — QuantizeLocalTokens waits both MTE3_MTE2 events before return
- Perturbation: 每个 AIV 一次时钟读取；复用 helper 已有双 buffer MTE3 回收。
- Source SHA-256: `3e12d159f0a60c9c7af4ea6e4fd068e9cc980867caf40eb785b9dd3efcaca0a2`

```text
  1321      }
  1322  #endif
  1323      QuantizeLocalTokens<QuantMode, QuantOutType, ActivationType, TopkWeightsType, TopkWeightsPrefetch>(
  1324          aivJob_, commonConfig_, params_, quantProcessConfig_, quantProcessScratch_);
  1325  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

### `route_index_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1335` in `MegaMoeA8W8Wave::Process`; `before` target line `1335`.
- Cardinality: 每个 AIV0/AIV1 各一次。
- Boundary claim: `causal` — 即将进入 GatherAndSendExpertCompactRoutes。
- Why: 单独量化 compact route index 构造、count 整理与跨 rank 发送。
- Dependency: input quantization helper returned
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker immediately precedes the compact route-index helper
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `3e12d159f0a60c9c7af4ea6e4fd068e9cc980867caf40eb785b9dd3efcaca0a2`

```text
  1333      }
  1334  #endif
  1335      GatherAndSendExpertCompactRoutes(aivJob_, commonConfig_, params_, g_winRankAddr_, sendMaskConfig_,
  1336                                       sendMaskScratch_);
  1337  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

### `route_index_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1335` in `MegaMoeA8W8Wave::Process`; `after` target line `1335`.
- Cardinality: 与 route_index_begin_S 一一对应。
- Boundary claim: `actual` — 本核所有 compact route/count ring 槽的 MTE3 写已完成。
- Why: 把 compact route index/count 远端发送完成与 workspace 清零分开。
- Dependency: GatherAndSendExpertCompactRoutes returned
- Pipeline proof: `Vector/MTE3` → `Scalar` using `production ring drain` — helper returns after every active compact-route MTE3 slot is drained
- Perturbation: 每个 AIV 一次时钟读取；复用 helper 生产事件。
- Source SHA-256: `3e12d159f0a60c9c7af4ea6e4fd068e9cc980867caf40eb785b9dd3efcaca0a2`

```text
  1333      }
  1334  #endif
  1335      GatherAndSendExpertCompactRoutes(aivJob_, commonConfig_, params_, g_winRankAddr_, sendMaskConfig_,
  1336                                       sendMaskScratch_);
  1337  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

### `workspace_reset_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1347` in `MegaMoeA8W8Wave::Process`; `before` target line `1347`.
- Cardinality: 每个 AIV0/AIV1 各一次。
- Boundary claim: `causal` — 即将发出本核负责的 workspace 清零。
- Why: 量化 flag/workspace 清零成本和同步等待。
- Dependency: compact route-index helper returned
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker immediately precedes ResetSyncStatus
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `3e12d159f0a60c9c7af4ea6e4fd068e9cc980867caf40eb785b9dd3efcaca0a2`

```text
  1345      }
  1346  #endif
  1347      ResetSyncStatus<TopkWeightsPrefetch>(aivJob_, params_, resetBatchElementCount_, resetTensor_);
  1348  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1349      if constexpr (g_coreType == AIV) {
```

### `workspace_reset_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1347` in `MegaMoeA8W8Wave::Process`; `after` target line `1347`.
- Cardinality: 与 workspace_reset_begin_S 一一对应。
- Boundary claim: `actual` — 本核此前发出的 workspace 清零 MTE3 已完成。
- Why: 不把未完成的 MTE3 清零尾部错误归入后续本地 barrier。
- Dependency: ResetDispatchWorkspace returned
- Pipeline proof: `MTE3` → `Scalar` using `tracing-only MTE3_S` — MTE3_S SetFlag/WaitFlag completes immediately before sampling
- Perturbation: 诊断态每个 AIV 增加一次 MTE3_S；会把原本由随后 SyncAll 吸收的等待前移。
- Source SHA-256: `3e12d159f0a60c9c7af4ea6e4fd068e9cc980867caf40eb785b9dd3efcaca0a2`

```text
  1345      }
  1346  #endif
  1347      ResetSyncStatus<TopkWeightsPrefetch>(aivJob_, params_, resetBatchElementCount_, resetTensor_);
  1348  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1349      if constexpr (g_coreType == AIV) {
```

### `local_prepare_sync_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1372` in `MegaMoeA8W8Wave::Process`; `before` target line `1372`.
- Cardinality: 每个物理 AI Block 的三角色各一次。
- Boundary claim: `causal` — 即将进入输入准备后的 SyncAll。
- Why: 直接量化 AIC 等待 AIV 输入准备和 flag 可见的本地 barrier。
- Dependency: role-local input preparation issued
- Pipeline proof: `mixed local pipelines` → `Scalar` using `program order` — marker is immediately before the mixed AIC/AIV rendezvous
- Perturbation: 每角色一次时钟读取。
- Source SHA-256: `3e12d159f0a60c9c7af4ea6e4fd068e9cc980867caf40eb785b9dd3efcaca0a2`

```text
  1370      }
  1371  #endif
  1372      SyncAll<false>(); // AIC 等待 AIV 完成输入准备与 flag 清零后再进入计算
  1373  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1374      if (CriticalPathTracePtr() != nullptr) {
```

### `local_prepare_sync_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1372` in `MegaMoeA8W8Wave::Process`; `after` target line `1372`.
- Cardinality: 与 local_prepare_sync_begin_S 一一对应。
- Boundary claim: `actual` — 本 AI Block 所有角色已越过输入准备 rendezvous。
- Why: 区分单卡角色同步与随后跨 rank 同步。
- Dependency: SyncAll returned
- Pipeline proof: `mixed local pipelines` → `Scalar` using `production rendezvous` — sample occurs after SyncAll
- Perturbation: 每角色一次时钟读取。
- Source SHA-256: `3e12d159f0a60c9c7af4ea6e4fd068e9cc980867caf40eb785b9dd3efcaca0a2`

```text
  1370      }
  1371  #endif
  1372      SyncAll<false>(); // AIC 等待 AIV 完成输入准备与 flag 清零后再进入计算
  1373  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1374      if (CriticalPathTracePtr() != nullptr) {
```

### `cross_rank_sync_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1113` in `MegaMoeA8W8Wave::ProcessGmmPipeline`; `before` target line `1113`.
- Cardinality: 每个物理 AI Block 的三角色各一次。
- Boundary claim: `causal` — 即将进入 CrossRankSyncInWorldSize。
- Why: 直接量化 Dispatch 前由最慢 rank 决定的全卡等待。
- Dependency: local input preparation rendezvous completed
- Pipeline proof: `Scalar/remote GM` → `Scalar` using `program order` — marker immediately precedes the first cross-rank barrier
- Perturbation: 每角色一次时钟读取。
- Source SHA-256: `3e12d159f0a60c9c7af4ea6e4fd068e9cc980867caf40eb785b9dd3efcaca0a2`

```text
  1111      }
  1112  #endif
  1113      CrossRankSyncInWorldSize(params_.peermemInfo.rankSyncInWorldPtr, rankId_, worldSize_, aivJob_);
  1114  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1115      if (CriticalPathTracePtr() != nullptr) {
```

### `cross_rank_sync_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1113` in `MegaMoeA8W8Wave::ProcessGmmPipeline`; `after` target line `1113`.
- Cardinality: 与 cross_rank_sync_begin_S 一一对应。
- Boundary claim: `actual` — 本角色已观察到所有 rank 输入准备完成。
- Why: 把全 rank 等待与本地专家 buffer 初始化分开。
- Dependency: CrossRankSyncInWorldSize returned
- Pipeline proof: `Scalar/remote GM` → `Scalar` using `production remote barrier` — sample occurs after the cross-rank wait
- Perturbation: 每角色一次时钟读取。
- Source SHA-256: `3e12d159f0a60c9c7af4ea6e4fd068e9cc980867caf40eb785b9dd3efcaca0a2`

```text
  1111      }
  1112  #endif
  1113      CrossRankSyncInWorldSize(params_.peermemInfo.rankSyncInWorldPtr, rankId_, worldSize_, aivJob_);
  1114  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1115      if (CriticalPathTracePtr() != nullptr) {
```

### `expert_buffer_init_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:990` in `MegaMoeA8W8Wave::ProcessMoeExpertStages`; `before` target line `990`.
- Cardinality: 每个物理 AI Block 的三角色各一次。
- Boundary claim: `causal` — 即将构造 Dispatch、Combine 和 token-count export 的本地视图。
- Why: 量化进入首轮 Dispatch/GMM 前的专家阶段 buffer 和状态初始化。
- Dependency: cross-rank input barrier completed
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker precedes the three buffer initialization helpers
- Perturbation: 每角色一次时钟读取。
- Source SHA-256: `3e12d159f0a60c9c7af4ea6e4fd068e9cc980867caf40eb785b9dd3efcaca0a2`

```text
   988      }
   989  #endif
   990      tokenDispatchScratch_.expertRevNumsGlobalTensor.SetGlobalBuffer(
   991          reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.expertRevTokenNumsPtr));
   992      DispatchBuffInit();
```

### `expert_buffer_init_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:994` in `MegaMoeA8W8Wave::ProcessMoeExpertStages`; `after` target line `994`.
- Cardinality: 与 expert_buffer_init_begin_S 一一对应。
- Boundary claim: `actual` — 三个 buffer 初始化 helper 已完成；这些 helper 仅构造 Scalar/LocalTensor 状态。
- Why: 分离本地视图构造与后续 cursor/problem 状态建立及首个 Dispatch。
- Dependency: all expert buffer helpers returned
- Pipeline proof: `Scalar` → `Scalar` using `program order` — helpers only bind buffer views and scalar fields
- Perturbation: 每角色一次时钟读取。
- Source SHA-256: `3e12d159f0a60c9c7af4ea6e4fd068e9cc980867caf40eb785b9dd3efcaca0a2`

```text
   992      DispatchBuffInit();
   993      CombineBufferConfig combineBufferConfig = InitCombineBuffers();
   994      PrepareMoeExpertTokenCountTable(commonConfig_, countWorkspace_, params_, tokenDispatchScratch_);
   995  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
   996      if (CriticalPathTracePtr() != nullptr) {
```

### `work_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1393` in `MegaMoeA8W8Wave::Process`; `after` target line `1393`.
- Cardinality: 每个物理 AI Block 的 AIC/AIV0/AIV1 各一次。
- Boundary claim: `actual` — 正常 routed/shared 流水、跨 rank 尾部和 Unpermute 已结束，尚未进入 tracing rendezvous。
- Why: 把有效工作与 tracing 终结同步/落盘分离。
- Dependency: all production work for this role returned
- Pipeline proof: `mixed production pipelines` → `Scalar` using `production completion plus program order` — sample is after ProcessGmmPipeline, final AIV cross-rank wait/Unpermute, and control restoration
- Perturbation: 每角色一次时钟读取和寄存器赋值。
- Source SHA-256: `3e12d159f0a60c9c7af4ea6e4fd068e9cc980867caf40eb785b9dd3efcaca0a2`

```text
  1391      // 恢复入口时保存的溢出模式。
  1392      SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
  1393      exceptionDump_.UpdateStage(MegaMoeImpl::Stage::COMPLETE);
  1394  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1395      if (CriticalPathTracePtr() != nullptr) {
```

### `dispatch_task_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_token_dispatch.h:624` in `MegaMoeImpl::RunGlobalWaveDispatchStage`; `before` target line `624`.
- Cardinality: 所有 AIV1，每次 PlanWaveExpertDispatch 为本核分配非空专家行区间时一次。
- Boundary claim: `causal` — 本核即将发送当前专家的连续行区间。
- Why: 只绘制实际拥有 token 行的 Dispatch helper，排除 count/cursor 和无工作专家。
- Dependency: dispatch plan assigned a non-empty row range to this AIV1
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker is inside HasWaveDispatchRows and immediately before the helper call
- Perturbation: 每个本核活跃专家任务一次时钟和状态更新。
- Source SHA-256: `ba690cc5a6e8e82c1aaaba3c7ad552241b8fe9836438aa47040a4861b3778131`

```text
   622              }
   623  #endif
   624              DispatchWaveExpertRows<ActivationType, QuantScaleType, PipelineTileM, TopkWeightsPrefetch,
   625                                     UseCompactRouteIndex, PublishGroupReady>(
   626                  context, common, syncLayout, params, winRankAddr, scratch, expertIdx, expertRange);
```

### `dispatch_task_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_token_dispatch.h:624` in `MegaMoeImpl::RunGlobalWaveDispatchStage`; `after` target line `624`.
- Cardinality: 与 dispatch_task_begin_S 一一对应。
- Boundary claim: `actual` — helper 已排空使用过的 copy ring 并发布本核行区间的 Dispatch-ready。
- Why: 只覆盖本核实际负责的专家行 Dispatch，不包住下一专家或 cursor 推进。
- Dependency: DispatchWaveExpertRows returned
- Pipeline proof: `MTE2/Vector/MTE3 and Scalar ready publication` → `Scalar` using `production helper's existing ring drain and MTE3_S` — CopyTokensAndMetaForDispatch consumes every used MTE3 completion event before return; ready publication follows production MTE3_S
- Perturbation: 一次结束时钟和四个独占 scratch 写；无新增 MTE3 bridge。
- Source SHA-256: `ba690cc5a6e8e82c1aaaba3c7ad552241b8fe9836438aa47040a4861b3778131`

```text
   622              }
   623  #endif
   624              DispatchWaveExpertRows<ActivationType, QuantScaleType, PipelineTileM, TopkWeightsPrefetch,
   625                                     UseCompactRouteIndex, PublishGroupReady>(
   626                  context, common, syncLayout, params, winRankAddr, scratch, expertIdx, expertRange);
```

### `dispatch_scalar_gap_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_token_dispatch.h:624` in `MegaMoeImpl::RunGlobalWaveDispatchStage`; `after` target line `624`.
- Cardinality: 所有 AIV1，每对连续有效 Dispatch task 之间最多一段；最后一个 task 后的未闭合尾段会丢弃，不跨入 GMM2/Combine。
- Boundary claim: `causal` — 前一个有效 Dispatch 已完成并写入其 trace 段；本 AIV1 开始寻找下一个拥有非空专家行的 Dispatch task。
- Why: 直接量化相邻有效 Dispatch 之间由专家 count 查询、rolling owner 规划、Wave cursor 推进以及相邻预取调用衔接产生的 Scalar 空隙。
- Dependency: previous DispatchWaveExpertRows returned and dispatch_task_end_S was sampled
- Pipeline proof: `Scalar after production MTE3 completion/ready publication` → `Scalar` using `program order after the existing Dispatch completion contract` — the sample follows EndStage0, whose timestamp is taken only after the production helper has drained its used copy ring and published ready
- Perturbation: 每个有效 Dispatch 结束后增加一次时钟和一个 Scalar 状态赋值；最后一个 task 的未使用起点也会采样，但不写 scratch。
- Source SHA-256: `ba690cc5a6e8e82c1aaaba3c7ad552241b8fe9836438aa47040a4861b3778131`

```text
   622              }
   623  #endif
   624              DispatchWaveExpertRows<ActivationType, QuantScaleType, PipelineTileM, TopkWeightsPrefetch,
   625                                     UseCompactRouteIndex, PublishGroupReady>(
   626                  context, common, syncLayout, params, winRankAddr, scratch, expertIdx, expertRange);
```

### `dispatch_scalar_gap_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_token_dispatch.h:624` in `MegaMoeImpl::RunGlobalWaveDispatchStage`; `before` target line `624`.
- Cardinality: 与 dispatch_scalar_gap_begin_S 一一对应，仅当后续确实找到另一个非空 Dispatch task 时闭合。
- Boundary claim: `actual` — 下一有效专家的 count/owner 规划已完成且本核已获得非空连续行区间，即将建立下一段 Dispatch task。
- Why: 判断 HTML 中 Dispatch 之间的白色区间是否就是 AIV1 的逐专家 Scalar 扫描，而不是隐藏的通信或等待。
- Dependency: PlanWaveExpertDispatch assigned a non-empty row interval for the next task
- Pipeline proof: `Scalar` → `Scalar` using `none; same-pipeline program order` — the sample is guarded by HasDispatchGap and HasWaveDispatchRows after planning, immediately before PrepareStage0/dispatch_task_begin_S
- Perturbation: 每个真实 inter-Dispatch gap 增加一次时钟和四个独占 scratch word 写；结束时钟在 scratch 写之前采样，因此写入开销不计入该段。
- Source SHA-256: `ba690cc5a6e8e82c1aaaba3c7ad552241b8fe9836438aa47040a4861b3778131`

```text
   622              }
   623  #endif
   624              DispatchWaveExpertRows<ActivationType, QuantScaleType, PipelineTileM, TopkWeightsPrefetch,
   625                                     UseCompactRouteIndex, PublishGroupReady>(
   626                  context, common, syncLayout, params, winRankAddr, scratch, expertIdx, expertRange);
```

### `gmm1_scalar_setup_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/common/mega_moe_trace.h:289` in `CriticalPathTrace::PrepareGmm1Gap`; `before` target line `289`.
- Cardinality: 每个 AIC 从上一个活跃 GMM1 problem 后开始，跨越连续空专家或无 owner problem，直到下一个 owned tile 的 ready 检查。
- Boundary claim: `causal` — 开始为后续 GMM1 owned problem 扫描并准备专家状态；连续无 owner problem 合并为一个低扰动区间。
- Why: 判断 GMM1 空洞是否来自逐专家状态准备、地址更新和 rolling ownership 跳过，而不是通信依赖。
- Dependency: ProcessGmm1Wave entered the next expert/problem attempt
- Pipeline proof: `Scalar` → `Scalar` using `program order` — the first pending setup timestamp is sampled before expert state preparation; later skipped experts retain that timestamp
- Perturbation: 每段一次起点时钟；连续无 owner 专家不逐项写 trace，避免诊断写放大空洞。
- Source SHA-256: `6a7bc25a88585d0b7ef56c99e937e35c4cb3794cc7b57a68fd967cb638cc442f`

```text
   287          // TRACE_POINT: gmm1_scalar_setup_begin_S
   288          if (gmm1SetupBegin_ == 0U) {
   289              gmm1SetupBegin_ = begin;
   290          }
   291          gmm1GapWave_ = waveIdx;
```

### `gmm1_scalar_setup_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/common/mega_moe_trace.h:461` in `CriticalPathTrace::EndGmm1Setup`; `before` target line `461`.
- Cardinality: 与聚合后的 gmm1_scalar_setup_begin_S 一一对应。
- Boundary claim: `actual` — 本核已找到下一个 owned tile 并完成 BlockMmad/problem 构造，或本 Wave 已无 owned tile。
- Why: 把 Scalar 调度与随后真实 Dispatch-ready 等待分开。
- Dependency: owned-tile ready check is about to start, or ProcessGmm1Wave is returning
- Pipeline proof: `Scalar` → `Scalar` using `program order` — active paths close setup immediately before WaitForGmm1InputReady; tail paths close it at Wave exit
- Perturbation: 每聚合段一次结束时钟和四个 scratch 标量写。
- Source SHA-256: `6a7bc25a88585d0b7ef56c99e937e35c4cb3794cc7b57a68fd967cb638cc442f`

```text
   459          }
   460          // TRACE_POINT: gmm1_scalar_setup_end_S
   461          AppendSegment(gmm1GapWave_, gmm1GapExpert_, gmm1SetupBegin_, value, 2U, stage2CapturedCount_);
   462          gmm1SetupBegin_ = 0U;
   463      }
```

### `gmm1_dispatch_wait_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:81` in `TraceGmm1DispatchWaitBegin`; `before` target line `81`.
- Cardinality: 每个 AIC 活跃 GMM1 problem 的首个 owned tile。
- Boundary claim: `causal` — 即将轮询当前 256-row group 的 Dispatch-ready flag。
- Why: 直接验证下一段 GMM1 是否仍被 Dispatch-ready flag 阻塞。
- Dependency: scheduler assigned the current tile to this AIC
- Pipeline proof: `Scalar` → `Scalar` using `program order` — helper is called immediately before WaitForGmm1InputReady
- Perturbation: 每个活跃 problem 一次时钟；setup scratch 写位于该采样之前。
- Source SHA-256: `8ba04f8c99f1e3a236aa47f15458c4c6fe824024d55919ebcc6ba3784f050132`

```text
    79      if (trace != nullptr && trace->NeedsStage0Begin()) {
    80          // TRACE_POINT: gmm1_dispatch_wait_begin_S
    81          trace->BeginGmm1DispatchWait(CriticalPathTrace::Now());
    82      }
    83  }
```

### `gmm1_dispatch_wait_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:89` in `TraceGmm1DispatchWaitEnd`; `before` target line `89`.
- Cardinality: 与 gmm1_dispatch_wait_begin_S 一一对应。
- Boundary claim: `actual` — 目标 Dispatch-ready 值已经被本核观察到。
- Why: 量化 Dispatch 是否真的落在两段 GMM1 之间。
- Dependency: WaitForGmm1InputReady returned
- Pipeline proof: `AIV1 remote MTE3 plus ready publication` → `Scalar` using `bypass-GM polling then program order` — helper is called immediately after WaitForGmm1InputReady
- Perturbation: 每个活跃 problem 一次时钟和四个 scratch 标量写。
- Source SHA-256: `8ba04f8c99f1e3a236aa47f15458c4c6fe824024d55919ebcc6ba3784f050132`

```text
    87      if (trace != nullptr && trace->NeedsStage0Begin()) {
    88          // TRACE_POINT: gmm1_dispatch_wait_end_S
    89          trace->EndGmm1DispatchWait(CriticalPathTrace::Now());
    90      }
    91  }
```

### `gmm1_vector_wait_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:97` in `TraceGmm1VectorWaitBegin`; `before` target line `97`.
- Cardinality: 非 prefetch GMM1 中确实需要回收 UB ping-pong 槽的活跃 problem。
- Boundary claim: `causal` — Dispatch 已 ready，即将等待 AIV0 归还当前 ping-pong 槽。
- Why: 判断前一段 SwiGLU 是否尚未释放 GMM1 输出槽，从而阻塞下一段 GMM1。
- Dependency: production vecSetSyncCom requires a ping-pong recycle
- Pipeline proof: `AIV0 Vector/MTE3` → `Scalar` using `program order` — helper is called only inside the branch that immediately invokes WaitForVector
- Perturbation: 仅在生产路径本来会 WaitForVector 时增加一次时钟。
- Source SHA-256: `8ba04f8c99f1e3a236aa47f15458c4c6fe824024d55919ebcc6ba3784f050132`

```text
    95      if (trace != nullptr && trace->NeedsStage0Begin()) {
    96          // TRACE_POINT: gmm1_vector_wait_begin_S
    97          trace->BeginGmm1VectorWait(CriticalPathTrace::Now());
    98      }
    99  }
```

### `gmm1_vector_wait_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:105` in `TraceGmm1VectorWaitEnd`; `before` target line `105`.
- Cardinality: 与 gmm1_vector_wait_begin_S 一一对应。
- Boundary claim: `actual` — AIV0 已归还所需 ping-pong 槽。
- Why: 把 SwiGLU backpressure 与下一段 GMM1 实际 MMAD 分开。
- Dependency: WaitForVector returned
- Pipeline proof: `AIV0 Vector/MTE3` → `Scalar` using `production event wait then program order` — helper is called immediately after WaitForVector
- Perturbation: 每次真实 WaitForVector 增加一次时钟和四个 scratch 标量写。
- Source SHA-256: `8ba04f8c99f1e3a236aa47f15458c4c6fe824024d55919ebcc6ba3784f050132`

```text
   103      if (trace != nullptr && trace->NeedsStage0Begin()) {
   104          // TRACE_POINT: gmm1_vector_wait_end_S
   105          trace->EndGmm1VectorWait(CriticalPathTrace::Now());
   106      }
   107  }
```

### `gmm1_problem_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:70` in `TraceGmm1ProblemBeginAfterReady`; `before` target line `70`.
- Cardinality: 所有 AIC，每个拥有 tile 的 GMM1 problem 一次。
- Boundary claim: `causal` — 首个 owned GMM1 tile 已 ready，即将发出。
- Why: 排除首 tile 的 Dispatch-ready/ping-pong 等待，并拆开 Wave 内多个 problem。
- Dependency: first owned tile's production readiness wait returned
- Pipeline proof: `Dispatch/FIX or ping-pong producer` → `Scalar` using `SCALAR, FIX_S, or V_S selected by production variant` — all routed generic call sites invoke the helper after their production wait
- Perturbation: 每活跃 problem 一次条件判断、必要的因果 bridge 和时钟。
- Source SHA-256: `8ba04f8c99f1e3a236aa47f15458c4c6fe824024d55919ebcc6ba3784f050132`

```text
    68      if constexpr (g_coreType == AscendC::AIC) {
    69          // TRACE_POINT: gmm1_problem_begin_S
    70          trace->BeginStage0(CriticalPathTrace::Now());
    71      } else {
    72          // TRACE_POINT: swiglu_problem_begin_S
```

### `gmm1_problem_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:779` in `MegaMoeA8W8Wave::ProcessGmm1Wave`; `after` target line `779`.
- Cardinality: 与有 owned tile 的 gmm1_problem_begin_S 一一对应。
- Boundary claim: `actual` — 本 problem 已发出的全部 FIX 工作已完成并桥接到 Scalar。
- Why: 只把本 problem 真正完成的 FIX 工作染色，使 problem 间白色可解释为调度/依赖空洞。
- Dependency: RunGmm1ActivationByMode returned with an owned tile
- Pipeline proof: `FIX` → `Scalar` using `tracing-only FIX_S` — HasStage0Begin gates an immediately preceding FIX_S SetFlag/WaitFlag pair
- Perturbation: 每个活跃 AIC problem 增加一次 FIX_S、一次时钟和四个 scratch 写；诊断态会抑制跨 problem FIX 尾部重叠。
- Source SHA-256: `3e12d159f0a60c9c7af4ea6e4fd068e9cc980867caf40eb785b9dd3efcaca0a2`

```text
   777          uint32_t waveTokenStartIndex =
   778              static_cast<uint32_t>(gmm1ExpertState.globalTokenStartIndex) + gmm1Position.tokenIndexInExpert;
   779          RunGmm1GenericByWeightFormat<QuantOutType, ActivationType, QuantScaleOutType, GMM1_TILE_M, EPILOGUE_TILE_M,
   780                                       TopkWeightsPrefetch, GMM1_INTERLEAVED, true>(
   781              gmmExecutionConfig_, params_, epilogueOp_, gmm1AddrInfo, gmm1WaveProblemShape, waveTokenStartIndex,
```

### `swiglu_problem_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:73` in `TraceGmm1ProblemBeginAfterReady`; `before` target line `73`.
- Cardinality: 所有 AIV0，每个拥有 epilogue tile 的 GMM1 problem 一次。
- Boundary claim: `causal` — 首个 owned epilogue tile 已 ready，即将执行 SwiGLU/MX。
- Why: 排除 WaitForCube/ready，并拆开 Wave 内多次 SwiGLU。
- Dependency: first owned epilogue tile readiness wait returned
- Pipeline proof: `GMM1/FIX` → `Scalar` using `V_S or production program order` — marker is after WaitForCube/WaitGmm1TileStatus and before the first epilogue tile helper
- Perturbation: 每活跃 problem 一次条件判断、因果 bridge 和时钟。
- Source SHA-256: `8ba04f8c99f1e3a236aa47f15458c4c6fe824024d55919ebcc6ba3784f050132`

```text
    71      } else {
    72          // TRACE_POINT: swiglu_problem_begin_S
    73          trace->BeginStage0(CriticalPathTrace::Now());
    74      }
    75  }
```

### `swiglu_problem_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:779` in `MegaMoeA8W8Wave::ProcessGmm1Wave`; `after` target line `779`.
- Cardinality: 与有 owned tile 的 swiglu_problem_begin_S 一一对应。
- Boundary claim: `issue` — 当前 problem helper 已返回并完成生产路径的输出/ready 提交。
- Why: 不把下一 problem 前的等待算入 SwiGLU。
- Dependency: RunGmm1ActivationByMode returned on AIV0
- Pipeline proof: `Vector/MTE3 issue and ready publication` → `Scalar` using `program order only` — no tracing-only completion event is introduced; boundary is conservatively issue
- Perturbation: 一次时钟和四个 scratch 写；不增加额外排空。
- Source SHA-256: `3e12d159f0a60c9c7af4ea6e4fd068e9cc980867caf40eb785b9dd3efcaca0a2`

```text
   777          uint32_t waveTokenStartIndex =
   778              static_cast<uint32_t>(gmm1ExpertState.globalTokenStartIndex) + gmm1Position.tokenIndexInExpert;
   779          RunGmm1GenericByWeightFormat<QuantOutType, ActivationType, QuantScaleOutType, GMM1_TILE_M, EPILOGUE_TILE_M,
   780                                       TopkWeightsPrefetch, GMM1_INTERLEAVED, true>(
   781              gmmExecutionConfig_, params_, epilogueOp_, gmm1AddrInfo, gmm1WaveProblemShape, waveTokenStartIndex,
```

### `gmm2_problem_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h:519` in `TraceGmm2ProblemBeginAfterReady`; `before` target line `519`.
- Cardinality: 所有 AIC，每个拥有 tile 的 GMM2 problem 一次。
- Boundary claim: `causal` — 首个 owned GMM2 tile 已 ready，即将发出。
- Why: 排除 SwiGLU-ready 轮询并拆开 Wave 内多个 problem。
- Dependency: first owned tile observed SwiGLU-ready
- Pipeline proof: `AIV0 MTE3/ready` → `Scalar` using `bypass-GM ready polling then program order` — helper is called only after WaitForGmm2InputReady returns
- Perturbation: 每活跃 problem 一次条件判断和时钟。
- Source SHA-256: `69f4ff431c0de4f77db8db0e68fa3b22da58b4f02728d5563908e1c969a118f6`

```text
   517      }
   518      // TRACE_POINT: gmm2_problem_begin_S
   519      trace->BeginStage1(CriticalPathTrace::Now());
   520  }
   521  #endif
```

### `gmm2_problem_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:892` in `MegaMoeA8W8Wave::ProcessGmm2Wave`; `after` target line `892`.
- Cardinality: 与有 owned tile 的 gmm2_problem_begin_S 一一对应。
- Boundary claim: `actual` — 本 problem 已发出的全部 GMM2 FIX 工作已完成并桥接到 Scalar。
- Why: 让大量短 GMM2 段覆盖真实 FIX 执行，而不是只显示 Scalar 发出。
- Dependency: RunMoeExpertGmm2Stage returned with an owned tile
- Pipeline proof: `FIX` → `Scalar` using `tracing-only FIX_S` — HasStage1Begin gates an immediately preceding FIX_S SetFlag/WaitFlag pair
- Perturbation: 每个活跃 AIC problem 增加一次 FIX_S、一次时钟和四个 scratch 写；诊断态会串行化跨 problem FIX 尾部。
- Source SHA-256: `3e12d159f0a60c9c7af4ea6e4fd068e9cc980867caf40eb785b9dd3efcaca0a2`

```text
   890          }
   891  #endif
   892          RunGmm2ByMode<COMBINE_NO_QUANT, QuantOutType, ActivationType, Weight1Type, QuantScaleOutType, false, false,
   893                        GMM1_TILE_M, TopkWeightsPrefetch, false, GMM1_INTERLEAVED, true, true>(
   894              gmmExecutionConfig_, gmm2AddrInfo, gmm2WaveProblemShape, runtimeState, nullptr, isWholeExpert,
```

### `combine_group_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h:1178` in `MegaMoeImpl::CombineWaveExpertRows`; `after` target line `1178`.
- Cardinality: 所有参与 Combine 的 AIV，每个本核 owned expert-local 256-token group 切片一次。
- Boundary claim: `causal` — 当前 expert-local 256-token group 的 GMM2 N-tile 计数已齐，即将读取并发送本核所负责行。
- Why: 直接观察本次新增的 256-token group ready 是否让 Combine 提前启动，并排除 ready 轮询。
- Dependency: current 256-token group's complete GMM2 N-tile count was observed
- Pipeline proof: `AIC FIX plus bypass-GM counter` → `Scalar` using `bypass-GM polling then program order` — marker is immediately after WaitWaveCombineGroupReady
- Perturbation: 每活跃 group 一次条件判断和时钟。
- Source SHA-256: `69f4ff431c0de4f77db8db0e68fa3b22da58b4f02728d5563908e1c969a118f6`

```text
  1176              rowsInCurrentGroup = remainingTokenCount;
  1177          }
  1178          WaitWaveCombineGroupReady(common, syncLayout, params, expertIdx, tokenGroupIdx);
  1179  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1180          if (criticalPathTrace != nullptr) {
```

### `combine_group_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h:1207` in `MegaMoeImpl::CombineWaveExpertRows`; `before` target line `1207`.
- Cardinality: 与 combine_group_begin_S 一一对应。
- Boundary claim: `actual` — 当前 group 切片此前发出的所有 MTE3 send 已完成并桥接到 Scalar。
- Why: 只覆盖当前 256-token group 切片，并在远端发送完成后结束。
- Dependency: the active group slice has issued all of its Combine rows
- Pipeline proof: `MTE2/Vector/MTE3` → `Scalar` using `tracing-only MTE3_S` — MTE3_S SetFlag/WaitFlag is immediately before sampling, so every earlier remote send is complete
- Perturbation: 每个本核活跃 group 增加一次 MTE3_S 完成桥、一次时钟和四个 scratch 写；诊断态会抑制跨 group MTE3 尾部重叠。
- Source SHA-256: `69f4ff431c0de4f77db8db0e68fa3b22da58b4f02728d5563908e1c969a118f6`

```text
  1205          }
  1206  #endif
  1207          processedTokenCount += rowsInCurrentGroup;
  1208      }
  1209  }
```

## Working tree status

```text
AM mc2/mega_moe/op_kernel/arch35/common/mega_moe_trace.h
MM mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h
MM mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h
MM mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h
M  mc2/mega_moe/op_kernel/arch35/stage/mega_moe_token_dispatch.h
A  mc2/mega_moe/tools/critical_path_trace/README.md
A  mc2/mega_moe/tools/critical_path_trace/audit-implemented/trace_point_audit.json
A  mc2/mega_moe/tools/critical_path_trace/audit-implemented/trace_point_audit.md
A  mc2/mega_moe/tools/critical_path_trace/audit-plan/trace_point_audit.json
A  mc2/mega_moe/tools/critical_path_trace/audit-plan/trace_point_audit.md
AM mc2/mega_moe/tools/critical_path_trace/generate_synthetic_fixture.py
AM mc2/mega_moe/tools/critical_path_trace/mega_moe_trace_contract.py
A  mc2/mega_moe/tools/critical_path_trace/pack_normalized_trace.py
A  mc2/mega_moe/tools/critical_path_trace/run_critical_path_trace.py
A  mc2/mega_moe/tools/critical_path_trace/test_critical_path_trace.py
A  mc2/mega_moe/tools/critical_path_trace/trace-spec.json
A  mc2/mega_moe/tools/critical_path_trace/trace_tool.py
A  mc2/mega_moe/tools/critical_path_trace/verify_trace_off.py
```
