# Trace point audit

- Status: `pass`
- Phase: `plan`
- Trace: `mega-moe-pr10302-token-group-wave-active-task-critical-path`
- Repository: `/home/mace/workspace/ops-transformer_megaMoeOpt`
- Revision: `a2b1c66f36efd5e93756492635f895eb10d90a5f`
- Branch: `codex/pr10302-wave-interleave-sendindex-critical-path-trace`
- Dirty: `True`
- Spec SHA-256: `76a523a7604e09a72b713072b9eb7816740962d02a9719d8644a90d4ee5f13b5`

## Placement summary

| Point | Roles | Resolved location | Target line | Placement | Boundary | Bridge | Why |
|---|---|---|---:|---|---|---|---|
| process_begin_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1356 | 1356 | before | causal | program order | 提供同一 AI Block 内三角色共同参考起点。 |
| input_buffer_init_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1366 | 1366 | before | causal | program order | 分离输入准备 UB 布局与后续 token 量化。 |
| input_buffer_init_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1366 | 1366 | after | issue | program order only | 量化前单独显示 UB 地址构造和零张量发出成本。 |
| input_quantize_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1386 | 1386 | before | causal | program order | 量化本卡输入 token 的固定成本独立可见。 |
| input_quantize_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1386 | 1386 | after | actual | production MTE3_MTE2 ring drain | 区分 token 量化与后续 compact route-index 通信。 |
| route_index_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1398 | 1398 | before | causal | program order | 单独量化 compact route index 构造、count 整理与跨 rank 发送。 |
| route_index_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1398 | 1398 | after | actual | production ring drain | 把 compact route index/count 远端发送完成与 workspace 清零分开。 |
| workspace_reset_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1410 | 1410 | before | causal | program order | 量化 flag/workspace 清零成本和同步等待。 |
| workspace_reset_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1410 | 1410 | after | actual | tracing-only MTE3_S | 不把未完成的 MTE3 清零尾部错误归入后续本地 barrier。 |
| local_prepare_sync_begin_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1435 | 1435 | before | causal | program order | 直接量化 AIC 等待 AIV 输入准备和 flag 可见的本地 barrier。 |
| local_prepare_sync_end_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1435 | 1435 | after | actual | production rendezvous | 区分单卡角色同步与随后跨 rank 同步。 |
| cross_rank_sync_begin_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1148 | 1148 | before | causal | program order | 直接量化 Dispatch 前由最慢 rank 决定的全卡等待。 |
| cross_rank_sync_end_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1148 | 1148 | after | actual | production remote barrier | 把全 rank 等待与本地专家 buffer 初始化分开。 |
| expert_buffer_init_begin_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1009 | 1009 | before | causal | program order | 量化进入首轮 Dispatch/GMM 前的专家阶段 buffer 和状态初始化。 |
| expert_buffer_init_end_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1013 | 1013 | after | actual | program order | 分离本地视图构造与后续 cursor/problem 状态建立及首个 Dispatch。 |
| work_end_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1488 | 1488 | after | actual | production completion plus program order | 把有效工作与 tracing 终结同步/落盘分离。 |
| dispatch_task_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_token_dispatch.h:624 | 624 | before | causal | program order | 只绘制实际拥有 token 行的 Dispatch helper，排除 count/cursor 和无工作专家。 |
| dispatch_task_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_token_dispatch.h:624 | 624 | after | actual | production helper's existing ring drain and MTE3_S | 只覆盖本核实际负责的专家行 Dispatch，不包住下一专家或 cursor 推进。 |
| dispatch_scalar_gap_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_token_dispatch.h:624 | 624 | after | causal | program order after the existing Dispatch completion contract | 直接量化相邻有效 Dispatch 之间由专家 count 查询、rolling owner 规划、Wave cursor 推进以及相邻预取调用衔接产生的 Scalar 空隙。 |
| dispatch_scalar_gap_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_token_dispatch.h:624 | 624 | before | actual | none; same-pipeline program order | 判断 HTML 中 Dispatch 之间的白色区间是否就是 AIV1 的逐专家 Scalar 扫描，而不是隐藏的通信或等待。 |
| gmm1_scalar_setup_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/common/mega_moe_trace.h:359 | 359 | before | causal | program order | 判断 GMM1 空洞是否来自逐专家状态准备、地址更新和 rolling ownership 跳过，而不是通信依赖。 |
| gmm1_scalar_setup_end_S | AIC | mc2/mega_moe/op_kernel/arch35/common/mega_moe_trace.h:545 | 545 | before | actual | program order | 把 Scalar 调度与随后真实 Dispatch-ready 等待分开。 |
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
| final_combine_drain_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:987 | 987 | before | causal | program order | 确认最后一个 Combine group 后是否仍有 row-ring MTE3 尾部需要回收。 |
| final_combine_drain_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:987 | 987 | after | actual | production MTE3_MTE2 event waits | 把 Combine 发送尾部与后续 MoE 收尾逻辑分开。 |
| moe_epilogue_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1114 | 1114 | before | causal | program order | 显示最终 Combine 与本地汇合之间的 EndSync、状态复位和 count 导出成本。 |
| moe_epilogue_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1121 | 1121 | after | actual | production helper completion | 定位 Combine 后非 barrier 的 V 核尾部成本。 |
| moe_tail_sync_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1167 | 1167 | before | causal | program order | 直接量化快 V 核等待慢 V 核完成最终 Combine/尾部工作的时间。 |
| moe_tail_sync_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1167 | 1167 | after | actual | production PIPE_ALL plus SyncAll | 把本卡负载不均衡等待与输出跨卡同步分开。 |
| shared_expert_gmm2_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1186 | 1186 | before | causal | program order | 显示 V 侧在共享专家 GMM2 调用路径上的调度区间。 |
| shared_expert_gmm2_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1186 | 1186 | after | issue | program order only | 区分共享专家调用开销与后续输出同步；共享结果等待可能继续落在 Unpermute 内。 |
| output_cross_rank_sync_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1454 | 1454 | before | causal | program order | 直接量化本 rank 等待其他 rank 完成 Combine 发送的时间。 |
| output_cross_rank_sync_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1454 | 1454 | after | actual | production rank handshake | 把慢 rank 的 Combine 尾部与本地 Unpermute 分开。 |
| unpermute_init_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1468 | 1468 | before | causal | program order | 把 UB view/config 构造与 Unpermute 数据流水分开。 |
| unpermute_init_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1468 | 1468 | after | actual | program order | 识别极端配置下 buffer 规划本身是否异常。 |
| unpermute_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1477 | 1477 | before | causal | program order | 量化 Combine 接收数据、top-k/shared 累加与最终 y 写回的整体成本。 |
| unpermute_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1477 | 1477 | after | actual | production MTE3_V event wait | 确认尾部慢点是否来自本地反量化/累加/写 y，而不是同步。 |

## Point evidence

### `process_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1356` in `MegaMoeA8W8Wave::Process`; `before` target line `1356`.
- Cardinality: 每个物理 AI Block 的 AIC/AIV0/AIV1 各一次。
- Boundary claim: `causal` — Process 的第一条诊断指令，尚未执行正常算子语句。
- Why: 提供同一 AI Block 内三角色共同参考起点。
- Dependency: kernel role entered Process
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker precedes the first production statement
- Perturbation: 每角色一次时钟读取和寄存器赋值。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1354  #endif
  1355      // 保存入口时的溢出模式，并初始化输入准备阶段使用的 UB。
  1356      int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
  1357      SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
  1358  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

### `input_buffer_init_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1366` in `MegaMoeA8W8Wave::Process`; `before` target line `1366`.
- Cardinality: 每个 AIV0/AIV1 各一次。
- Boundary claim: `causal` — 即将构造 quant/route-index/reset 的 UB 视图并发出 resetTensor Duplicate。
- Why: 分离输入准备 UB 布局与后续 token 量化。
- Dependency: overflow mode configured
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker is immediately before SendAndQuantBuffInit
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1364      }
  1365  #endif
  1366      SendAndQuantBuffInit();
  1367  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1368      if constexpr (g_coreType == AIV) {
```

### `input_buffer_init_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1366` in `MegaMoeA8W8Wave::Process`; `after` target line `1366`.
- Cardinality: 与 input_buffer_init_begin_S 一一对应。
- Boundary claim: `issue` — UB 视图已构造且 Duplicate 已发出；Vector 尾部可继续运行。
- Why: 量化前单独显示 UB 地址构造和零张量发出成本。
- Dependency: SendAndQuantBuffInit returned
- Pipeline proof: `Scalar plus Vector issue` → `Scalar` using `program order only` — no tracing-only Vector drain is inserted
- Perturbation: 每个 AIV 一次时钟读取，不新增 Vector drain。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1364      }
  1365  #endif
  1366      SendAndQuantBuffInit();
  1367  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1368      if constexpr (g_coreType == AIV) {
```

### `input_quantize_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1386` in `MegaMoeA8W8Wave::Process`; `before` target line `1386`.
- Cardinality: 每个 AIV0/AIV1 各一次。
- Boundary claim: `causal` — 即将进入 QuantizeLocalTokens。
- Why: 量化本卡输入 token 的固定成本独立可见。
- Dependency: input UB views are configured
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker precedes QuantizeLocalTokens
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1384      }
  1385  #endif
  1386      QuantizeLocalTokens<QuantMode, QuantOutType, ActivationType, TopkWeightsType, TopkWeightsPrefetch>(
  1387          aivJob_, commonConfig_, params_, quantProcessConfig_, quantProcessScratch_);
  1388  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

### `input_quantize_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1386` in `MegaMoeA8W8Wave::Process`; `after` target line `1386`.
- Cardinality: 与 input_quantize_begin_S 一一对应。
- Boundary claim: `actual` — 本核负责 token 的量化结果已写回 GM。
- Why: 区分 token 量化与后续 compact route-index 通信。
- Dependency: QuantizeLocalTokens returned
- Pipeline proof: `MTE2/Vector/MTE3` → `Scalar` using `production MTE3_MTE2 ring drain` — QuantizeLocalTokens waits both MTE3_MTE2 events before return
- Perturbation: 每个 AIV 一次时钟读取；复用 helper 已有双 buffer MTE3 回收。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1384      }
  1385  #endif
  1386      QuantizeLocalTokens<QuantMode, QuantOutType, ActivationType, TopkWeightsType, TopkWeightsPrefetch>(
  1387          aivJob_, commonConfig_, params_, quantProcessConfig_, quantProcessScratch_);
  1388  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

### `route_index_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1398` in `MegaMoeA8W8Wave::Process`; `before` target line `1398`.
- Cardinality: 每个 AIV0/AIV1 各一次。
- Boundary claim: `causal` — 即将进入 GatherAndSendExpertCompactRoutes。
- Why: 单独量化 compact route index 构造、count 整理与跨 rank 发送。
- Dependency: input quantization helper returned
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker immediately precedes the compact route-index helper
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1396      }
  1397  #endif
  1398      GatherAndSendExpertCompactRoutes(aivJob_, commonConfig_, params_, g_winRankAddr_, sendMaskConfig_,
  1399                                       sendMaskScratch_);
  1400  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

### `route_index_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1398` in `MegaMoeA8W8Wave::Process`; `after` target line `1398`.
- Cardinality: 与 route_index_begin_S 一一对应。
- Boundary claim: `actual` — 本核所有 compact route/count ring 槽的 MTE3 写已完成。
- Why: 把 compact route index/count 远端发送完成与 workspace 清零分开。
- Dependency: GatherAndSendExpertCompactRoutes returned
- Pipeline proof: `Vector/MTE3` → `Scalar` using `production ring drain` — helper returns after every active compact-route MTE3 slot is drained
- Perturbation: 每个 AIV 一次时钟读取；复用 helper 生产事件。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1396      }
  1397  #endif
  1398      GatherAndSendExpertCompactRoutes(aivJob_, commonConfig_, params_, g_winRankAddr_, sendMaskConfig_,
  1399                                       sendMaskScratch_);
  1400  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

### `workspace_reset_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1410` in `MegaMoeA8W8Wave::Process`; `before` target line `1410`.
- Cardinality: 每个 AIV0/AIV1 各一次。
- Boundary claim: `causal` — 即将发出本核负责的 workspace 清零。
- Why: 量化 flag/workspace 清零成本和同步等待。
- Dependency: compact route-index helper returned
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker immediately precedes ResetSyncStatus
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1408      }
  1409  #endif
  1410      ResetSyncStatus<TopkWeightsPrefetch>(aivJob_, params_, resetBatchElementCount_, resetTensor_);
  1411  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1412      if constexpr (g_coreType == AIV) {
```

### `workspace_reset_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1410` in `MegaMoeA8W8Wave::Process`; `after` target line `1410`.
- Cardinality: 与 workspace_reset_begin_S 一一对应。
- Boundary claim: `actual` — 本核此前发出的 workspace 清零 MTE3 已完成。
- Why: 不把未完成的 MTE3 清零尾部错误归入后续本地 barrier。
- Dependency: ResetDispatchWorkspace returned
- Pipeline proof: `MTE3` → `Scalar` using `tracing-only MTE3_S` — MTE3_S SetFlag/WaitFlag completes immediately before sampling
- Perturbation: 诊断态每个 AIV 增加一次 MTE3_S；会把原本由随后 SyncAll 吸收的等待前移。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1408      }
  1409  #endif
  1410      ResetSyncStatus<TopkWeightsPrefetch>(aivJob_, params_, resetBatchElementCount_, resetTensor_);
  1411  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1412      if constexpr (g_coreType == AIV) {
```

### `local_prepare_sync_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1435` in `MegaMoeA8W8Wave::Process`; `before` target line `1435`.
- Cardinality: 每个物理 AI Block 的三角色各一次。
- Boundary claim: `causal` — 即将进入输入准备后的 SyncAll。
- Why: 直接量化 AIC 等待 AIV 输入准备和 flag 可见的本地 barrier。
- Dependency: role-local input preparation issued
- Pipeline proof: `mixed local pipelines` → `Scalar` using `program order` — marker is immediately before the mixed AIC/AIV rendezvous
- Perturbation: 每角色一次时钟读取。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1433      }
  1434  #endif
  1435      SyncAll<false>(); // AIC 等待 AIV 完成输入准备与 flag 清零后再进入计算
  1436  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1437      if (CriticalPathTracePtr() != nullptr) {
```

### `local_prepare_sync_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1435` in `MegaMoeA8W8Wave::Process`; `after` target line `1435`.
- Cardinality: 与 local_prepare_sync_begin_S 一一对应。
- Boundary claim: `actual` — 本 AI Block 所有角色已越过输入准备 rendezvous。
- Why: 区分单卡角色同步与随后跨 rank 同步。
- Dependency: SyncAll returned
- Pipeline proof: `mixed local pipelines` → `Scalar` using `production rendezvous` — sample occurs after SyncAll
- Perturbation: 每角色一次时钟读取。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1433      }
  1434  #endif
  1435      SyncAll<false>(); // AIC 等待 AIV 完成输入准备与 flag 清零后再进入计算
  1436  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1437      if (CriticalPathTracePtr() != nullptr) {
```

### `cross_rank_sync_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1148` in `MegaMoeA8W8Wave::ProcessGmmPipeline`; `before` target line `1148`.
- Cardinality: 每个物理 AI Block 的三角色各一次。
- Boundary claim: `causal` — 即将进入 CrossRankSyncInWorldSize。
- Why: 直接量化 Dispatch 前由最慢 rank 决定的全卡等待。
- Dependency: local input preparation rendezvous completed
- Pipeline proof: `Scalar/remote GM` → `Scalar` using `program order` — marker immediately precedes the first cross-rank barrier
- Perturbation: 每角色一次时钟读取。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1146      }
  1147  #endif
  1148      CrossRankSyncInWorldSize(params_.peermemInfo.rankSyncInWorldPtr, rankId_, worldSize_, aivJob_);
  1149  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1150      if (CriticalPathTracePtr() != nullptr) {
```

### `cross_rank_sync_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1148` in `MegaMoeA8W8Wave::ProcessGmmPipeline`; `after` target line `1148`.
- Cardinality: 与 cross_rank_sync_begin_S 一一对应。
- Boundary claim: `actual` — 本角色已观察到所有 rank 输入准备完成。
- Why: 把全 rank 等待与本地专家 buffer 初始化分开。
- Dependency: CrossRankSyncInWorldSize returned
- Pipeline proof: `Scalar/remote GM` → `Scalar` using `production remote barrier` — sample occurs after the cross-rank wait
- Perturbation: 每角色一次时钟读取。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1146      }
  1147  #endif
  1148      CrossRankSyncInWorldSize(params_.peermemInfo.rankSyncInWorldPtr, rankId_, worldSize_, aivJob_);
  1149  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1150      if (CriticalPathTracePtr() != nullptr) {
```

### `expert_buffer_init_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1009` in `MegaMoeA8W8Wave::ProcessMoeExpertStages`; `before` target line `1009`.
- Cardinality: 每个物理 AI Block 的三角色各一次。
- Boundary claim: `causal` — 即将构造 Dispatch、Combine 和 token-count export 的本地视图。
- Why: 量化进入首轮 Dispatch/GMM 前的专家阶段 buffer 和状态初始化。
- Dependency: cross-rank input barrier completed
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker precedes the three buffer initialization helpers
- Perturbation: 每角色一次时钟读取。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1007      }
  1008  #endif
  1009      tokenDispatchScratch_.expertRevNumsGlobalTensor.SetGlobalBuffer(
  1010          reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.expertRevTokenNumsPtr));
  1011      DispatchBuffInit();
```

### `expert_buffer_init_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1013` in `MegaMoeA8W8Wave::ProcessMoeExpertStages`; `after` target line `1013`.
- Cardinality: 与 expert_buffer_init_begin_S 一一对应。
- Boundary claim: `actual` — 三个 buffer 初始化 helper 已完成；这些 helper 仅构造 Scalar/LocalTensor 状态。
- Why: 分离本地视图构造与后续 cursor/problem 状态建立及首个 Dispatch。
- Dependency: all expert buffer helpers returned
- Pipeline proof: `Scalar` → `Scalar` using `program order` — helpers only bind buffer views and scalar fields
- Perturbation: 每角色一次时钟读取。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1011      DispatchBuffInit();
  1012      CombineBufferConfig combineBufferConfig = InitCombineBuffers();
  1013      PrepareMoeExpertTokenCountTable(commonConfig_, countWorkspace_, params_, tokenDispatchScratch_);
  1014  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1015      if (CriticalPathTracePtr() != nullptr) {
```

### `work_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1488` in `MegaMoeA8W8Wave::Process`; `after` target line `1488`.
- Cardinality: 每个物理 AI Block 的 AIC/AIV0/AIV1 各一次。
- Boundary claim: `actual` — 正常 routed/shared 流水、跨 rank 尾部和 Unpermute 已结束，尚未进入 tracing rendezvous。
- Why: 把有效工作与 tracing 终结同步/落盘分离。
- Dependency: all production work for this role returned
- Pipeline proof: `mixed production pipelines` → `Scalar` using `production completion plus program order` — sample is after ProcessGmmPipeline, final AIV cross-rank wait/Unpermute, and control restoration
- Perturbation: 每角色一次时钟读取和寄存器赋值。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1486      // 恢复入口时保存的溢出模式。
  1487      SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
  1488      exceptionDump_.UpdateStage(MegaMoeImpl::Stage::COMPLETE);
  1489  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1490      if (CriticalPathTracePtr() != nullptr) {
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

- Where: `mc2/mega_moe/op_kernel/arch35/common/mega_moe_trace.h:359` in `CriticalPathTrace::PrepareGmm1Gap`; `before` target line `359`.
- Cardinality: 每个 AIC 从上一个活跃 GMM1 problem 后开始，跨越连续空专家或无 owner problem，直到下一个 owned tile 的 ready 检查。
- Boundary claim: `causal` — 开始为后续 GMM1 owned problem 扫描并准备专家状态；连续无 owner problem 合并为一个低扰动区间。
- Why: 判断 GMM1 空洞是否来自逐专家状态准备、地址更新和 rolling ownership 跳过，而不是通信依赖。
- Dependency: ProcessGmm1Wave entered the next expert/problem attempt
- Pipeline proof: `Scalar` → `Scalar` using `program order` — the first pending setup timestamp is sampled before expert state preparation; later skipped experts retain that timestamp
- Perturbation: 每段一次起点时钟；连续无 owner 专家不逐项写 trace，避免诊断写放大空洞。
- Source SHA-256: `05a286e04454e59ea939ee60486c8839b1d1f3c4a67e7d63aabbc4e2bbe465dd`

```text
   357          // TRACE_POINT: gmm1_scalar_setup_begin_S
   358          if (gmm1SetupBegin_ == 0U) {
   359              gmm1SetupBegin_ = begin;
   360          }
   361          gmm1GapWave_ = waveIdx;
```

### `gmm1_scalar_setup_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/common/mega_moe_trace.h:545` in `CriticalPathTrace::EndGmm1Setup`; `before` target line `545`.
- Cardinality: 与聚合后的 gmm1_scalar_setup_begin_S 一一对应。
- Boundary claim: `actual` — 本核已找到下一个 owned tile 并完成 BlockMmad/problem 构造，或本 Wave 已无 owned tile。
- Why: 把 Scalar 调度与随后真实 Dispatch-ready 等待分开。
- Dependency: owned-tile ready check is about to start, or ProcessGmm1Wave is returning
- Pipeline proof: `Scalar` → `Scalar` using `program order` — active paths close setup immediately before WaitForGmm1InputReady; tail paths close it at Wave exit
- Perturbation: 每聚合段一次结束时钟和四个 scratch 标量写。
- Source SHA-256: `05a286e04454e59ea939ee60486c8839b1d1f3c4a67e7d63aabbc4e2bbe465dd`

```text
   543          }
   544          // TRACE_POINT: gmm1_scalar_setup_end_S
   545          AppendSegment(gmm1GapWave_, gmm1GapExpert_, gmm1SetupBegin_, value, 2U, stage2CapturedCount_);
   546          gmm1SetupBegin_ = 0U;
   547      }
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
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

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
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

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
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

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

### `final_combine_drain_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:987` in `MegaMoeA8W8Wave::ProcessCombineWave`; `before` target line `987`.
- Cardinality: 最终 Wave 上所有 AIV 各一次。
- Boundary claim: `causal` — 最终 Wave 的 Combine 行已全部发出，即将消费仍在途的 row-ring 完成事件。
- Why: 确认最后一个 Combine group 后是否仍有 row-ring MTE3 尾部需要回收。
- Dependency: final Wave Combine helpers returned
- Pipeline proof: `MTE3` → `Scalar` using `program order` — marker is immediately before the production row-ring drain
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
   985      }
   986  #endif
   987      DrainCombineRowBuffers(rowSequence, activeBufferConfig.rowBufferCount);
   988  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
   989      if (isFinalWave && CriticalPathTracePtr() != nullptr) {
```

### `final_combine_drain_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:987` in `MegaMoeA8W8Wave::ProcessCombineWave`; `after` target line `987`.
- Cardinality: 与 final_combine_drain_begin_S 一一对应。
- Boundary claim: `actual` — 最终 Wave 的全部已用 row-ring MTE3 完成事件已经消费。
- Why: 把 Combine 发送尾部与后续 MoE 收尾逻辑分开。
- Dependency: DrainCombineRowBuffers returned
- Pipeline proof: `MTE3` → `Scalar` using `production MTE3_MTE2 event waits` — the helper consumes every active row-ring completion event before return
- Perturbation: 每个 AIV 一次时钟读取，复用生产 drain。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
   985      }
   986  #endif
   987      DrainCombineRowBuffers(rowSequence, activeBufferConfig.rowBufferCount);
   988  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
   989      if (isFinalWave && CriticalPathTracePtr() != nullptr) {
```

### `moe_epilogue_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1114` in `MegaMoeA8W8Wave::ProcessMoeExpertStages`; `before` target line `1114`.
- Cardinality: 每个 AIV 在最终 Combine 返回后一次。
- Boundary claim: `causal` — 最终 Combine/drain 已返回，即将执行 MoE 专家阶段尾部逻辑。
- Why: 显示最终 Combine 与本地汇合之间的 EndSync、状态复位和 count 导出成本。
- Dependency: the final ProcessCombineWave returned
- Pipeline proof: `mixed` → `Scalar` using `program order` — marker is gated by !hasNextWave immediately after ProcessCombineWave
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1112          }
  1113  #endif
  1114          waveBeginPosition = waveEndPosition;
  1115      }
  1116
```

### `moe_epilogue_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1121` in `MegaMoeA8W8Wave::ProcessMoeExpertStages`; `after` target line `1121`.
- Cardinality: 与 moe_epilogue_begin_S 一一对应。
- Boundary claim: `actual` — EndSync、ping-pong 复位和 compact expert count 导出均已返回。
- Why: 定位 Combine 后非 barrier 的 V 核尾部成本。
- Dependency: expert-count export returned
- Pipeline proof: `Scalar/MTE2/MTE3` → `Scalar` using `production helper completion` — the only exporting AIV drains its MTE3 before return; other AIVs have no export work
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1119      }
  1120      gmm1PingPongIdx_ = 0;
  1121      ExportCompactExpertTokenCounts(commonConfig_, countWorkspace_, params_, tokenDispatchScratch_);
  1122  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1123      if constexpr (g_coreType == AIV) {
```

### `moe_tail_sync_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1167` in `MegaMoeA8W8Wave::ProcessGmmPipeline`; `before` target line `1167`.
- Cardinality: 所有 AIV 在 MoE 专家阶段返回后一次。
- Boundary claim: `causal` — 即将执行 MoE 尾部 PIPE_ALL 和本卡 AIV SyncAll。
- Why: 直接量化快 V 核等待慢 V 核完成最终 Combine/尾部工作的时间。
- Dependency: ProcessMoeExpertStages returned
- Pipeline proof: `mixed local pipelines` → `Scalar` using `program order` — marker precedes the production PIPE_ALL/SyncAll pair
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1165  #endif
  1166          PipeBarrier<PIPE_ALL>();
  1167          SyncAll<true>();
  1168  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1169          if (CriticalPathTracePtr() != nullptr) {
```

### `moe_tail_sync_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1167` in `MegaMoeA8W8Wave::ProcessGmmPipeline`; `after` target line `1167`.
- Cardinality: 与 moe_tail_sync_begin_S 一一对应。
- Boundary claim: `actual` — 本卡所有 AIV 已完成 MoE 专家流水并通过 SyncAll。
- Why: 把本卡负载不均衡等待与输出跨卡同步分开。
- Dependency: production AIV SyncAll returned
- Pipeline proof: `all local AIV pipelines` → `Scalar` using `production PIPE_ALL plus SyncAll` — marker is immediately after the existing synchronization
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1165  #endif
  1166          PipeBarrier<PIPE_ALL>();
  1167          SyncAll<true>();
  1168  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1169          if (CriticalPathTracePtr() != nullptr) {
```

### `shared_expert_gmm2_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1186` in `MegaMoeA8W8Wave::ProcessGmmPipeline`; `before` target line `1186`.
- Cardinality: 每个 AIV 各一次；sharedExpertNum=0 时仅覆盖分支判断。
- Boundary claim: `causal` — 即将判断并按需进入共享专家 GMM2 helper；实际 Cube 计算仍由 AIC 完成。
- Why: 显示 V 侧在共享专家 GMM2 调用路径上的调度区间。
- Dependency: MoE tail SyncAll returned
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker immediately precedes the sharedExpertNum branch on AIV roles
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1184      if (sharedExpertNum_ > 0) {
  1185          exceptionDump_.UpdateStage(MegaMoeImpl::Stage::SHARED_EXPERT_GMM2);
  1186          ProcessSharedExpertGmm2();
  1187      }
  1188  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

### `shared_expert_gmm2_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1186` in `MegaMoeA8W8Wave::ProcessGmmPipeline`; `after` target line `1186`.
- Cardinality: 与 shared_expert_gmm2_begin_S 一一对应。
- Boundary claim: `issue` — AIV 角色已从共享专家 GMM2 helper 返回。
- Why: 区分共享专家调用开销与后续输出同步；共享结果等待可能继续落在 Unpermute 内。
- Dependency: ProcessSharedExpertGmm2 returned on AIV
- Pipeline proof: `Scalar control` → `Scalar` using `program order only` — no tracing-only completion event is inserted
- Perturbation: 每个 AIV 一次时钟读取，不增加 Cube/FIX drain。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1184      if (sharedExpertNum_ > 0) {
  1185          exceptionDump_.UpdateStage(MegaMoeImpl::Stage::SHARED_EXPERT_GMM2);
  1186          ProcessSharedExpertGmm2();
  1187      }
  1188  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

### `output_cross_rank_sync_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1454` in `MegaMoeA8W8Wave::Process`; `before` target line `1454`.
- Cardinality: 所有 AIV 各一次。
- Boundary claim: `causal` — 即将进入 Unpermute 前的输出跨 rank 同步。
- Why: 直接量化本 rank 等待其他 rank 完成 Combine 发送的时间。
- Dependency: local GMM pipeline returned
- Pipeline proof: `Scalar/network` → `Scalar` using `program order` — marker immediately precedes the output rank handshake
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1452          }
  1453  #endif
  1454          CrossRankSyncInWorldSize(params_.peermemInfo.rankSyncInWorldPtr, rankId_, worldSize_, aivJob_);
  1455  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1456          if (CriticalPathTracePtr() != nullptr) {
```

### `output_cross_rank_sync_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1454` in `MegaMoeA8W8Wave::Process`; `after` target line `1454`.
- Cardinality: 与 output_cross_rank_sync_begin_S 一一对应。
- Boundary claim: `actual` — 所有 rank 的输出握手已完成，本卡 Combine 接收区可读。
- Why: 把慢 rank 的 Combine 尾部与本地 Unpermute 分开。
- Dependency: CrossRankSyncInWorldSize returned
- Pipeline proof: `Scalar/network` → `Scalar` using `production rank handshake` — marker is immediately after the existing cross-rank synchronization
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1452          }
  1453  #endif
  1454          CrossRankSyncInWorldSize(params_.peermemInfo.rankSyncInWorldPtr, rankId_, worldSize_, aivJob_);
  1455  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1456          if (CriticalPathTracePtr() != nullptr) {
```

### `unpermute_init_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1468` in `MegaMoeA8W8Wave::Process`; `before` target line `1468`.
- Cardinality: 所有 AIV 各一次。
- Boundary claim: `causal` — 即将构造当前 AIV 的 Unpermute buffer 视图。
- Why: 把 UB view/config 构造与 Unpermute 数据流水分开。
- Dependency: output cross-rank sync returned
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker immediately precedes InitTokenUnpermuteBuffers
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1466          }
  1467  #endif
  1468          UnpermuteBufferConfig unpermuteBufferConfig = InitTokenUnpermuteBuffers();
  1469  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1470          if (CriticalPathTracePtr() != nullptr) {
```

### `unpermute_init_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1468` in `MegaMoeA8W8Wave::Process`; `after` target line `1468`.
- Cardinality: 与 unpermute_init_begin_S 一一对应。
- Boundary claim: `actual` — Unpermute UB view 和当前 job bufferConfig 已构造完成。
- Why: 识别极端配置下 buffer 规划本身是否异常。
- Dependency: InitTokenUnpermuteBuffers returned
- Pipeline proof: `Scalar` → `Scalar` using `program order` — helper only constructs local tensor views and returns synchronously
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1466          }
  1467  #endif
  1468          UnpermuteBufferConfig unpermuteBufferConfig = InitTokenUnpermuteBuffers();
  1469  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1470          if (CriticalPathTracePtr() != nullptr) {
```

### `unpermute_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1477` in `MegaMoeA8W8Wave::Process`; `before` target line `1477`.
- Cardinality: 所有 AIV 各一次。
- Boundary claim: `causal` — Unpermute buffer 已构造，即将进入本核 token range。
- Why: 量化 Combine 接收数据、top-k/shared 累加与最终 y 写回的整体成本。
- Dependency: Unpermute buffer initialization completed
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker immediately precedes UnpermuteTokens
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1475          }
  1476  #endif
  1477          UnpermuteTokens<CombineQuantMode, TopkWeightsType, TopkWeightsPrefetch, GMM1_TILE_M>(
  1478              tokenUnpermuteConfig_, commonConfig_, params_, tokenUnpermuteScratch_, unpermuteBufferConfig);
  1479  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

### `unpermute_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1477` in `MegaMoeA8W8Wave::Process`; `after` target line `1477`.
- Cardinality: 与 unpermute_begin_S 一一对应。
- Boundary claim: `actual` — 本核负责 token 的最终 y MTE3 已全部完成。
- Why: 确认尾部慢点是否来自本地反量化/累加/写 y，而不是同步。
- Dependency: UnpermuteTokens returned
- Pipeline proof: `MTE2/Vector/MTE3` → `Scalar` using `production MTE3_V event wait` — UnpermuteTokens consumes its outputBufferEvent after the final batch before return
- Perturbation: 每个 AIV 一次时钟读取，复用 UnpermuteTokens 的生产 MTE3_V drain。
- Source SHA-256: `b94c9ea3d079c6ab84fa33464216df632b035833640d93697f7ebabf0dcb0b4d`

```text
  1475          }
  1476  #endif
  1477          UnpermuteTokens<CombineQuantMode, TopkWeightsType, TopkWeightsPrefetch, GMM1_TILE_M>(
  1478              tokenUnpermuteConfig_, commonConfig_, params_, tokenUnpermuteScratch_, unpermuteBufferConfig);
  1479  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

## Working tree status

```text
M mc2/mega_moe/op_kernel/arch35/common/mega_moe_trace.h
 M mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h
 M mc2/mega_moe/tools/critical_path_trace/README.md
 M mc2/mega_moe/tools/critical_path_trace/audit-implemented/trace_point_audit.json
 M mc2/mega_moe/tools/critical_path_trace/audit-implemented/trace_point_audit.md
 M mc2/mega_moe/tools/critical_path_trace/audit-plan/trace_point_audit.json
 M mc2/mega_moe/tools/critical_path_trace/audit-plan/trace_point_audit.md
 M mc2/mega_moe/tools/critical_path_trace/mega_moe_trace_contract.py
 M mc2/mega_moe/tools/critical_path_trace/pack_normalized_trace.py
 M mc2/mega_moe/tools/critical_path_trace/run_critical_path_trace.py
 M mc2/mega_moe/tools/critical_path_trace/test_critical_path_trace.py
 M mc2/mega_moe/tools/critical_path_trace/trace-spec.json
 M mc2/mega_moe/tools/critical_path_trace/trace_tool.py
?? mc2/mega_moe/REMOVE_COMBINE_GLOBAL_SYNC_DESIGN.md
?? mc2/mega_moe/REMOVE_DISPATCH_GLOBAL_SYNC_DESIGN.md
?? mc2/mega_moe/REMOVE_GLOBAL_SYNC_DESIGN.md
```
