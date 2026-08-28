# Trace point audit

- Status: `pass`
- Phase: `plan`
- Trace: `mega-moe-pr10302-token-group-wave-active-task-critical-path`
- Repository: `/home/mace/workspace/ops-transformer_megaMoeOpt_wave_interleave_trace`
- Revision: `99b65637adac9456c313c505a7aa4a2b34e61058`
- Branch: `codex/pr10302-wave-interleave-sendindex-critical-path-trace`
- Dirty: `True`
- Spec SHA-256: `779497cfb4e60978b8731351d09b6cb5f563c74291757d382a7ddc3ec1fc9c35`

## Placement summary

| Point | Roles | Resolved location | Target line | Placement | Boundary | Bridge | Why |
|---|---|---|---:|---|---|---|---|
| process_begin_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1387 | 1387 | before | causal | program order | 提供同一 AI Block 内三角色共同参考起点。 |
| input_buffer_init_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1397 | 1397 | before | causal | program order | 分离输入准备 UB 布局与后续 token 量化。 |
| input_buffer_init_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1397 | 1397 | after | issue | program order only | 量化前单独显示 UB 地址构造和零张量发出成本。 |
| input_quantize_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1417 | 1417 | before | causal | program order | 量化本卡输入 token 的固定成本独立可见。 |
| input_quantize_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1417 | 1417 | after | actual | production MTE3_MTE2 ring drain | 区分 token 量化与后续 compact route-index 通信。 |
| route_index_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1429 | 1429 | before | causal | program order | 单独量化 compact route index 构造、count 整理与跨 rank 发送。 |
| route_index_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1429 | 1429 | after | actual | production ring drain | 把 compact route index/count 远端发送完成与 workspace 清零分开。 |
| workspace_reset_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1441 | 1441 | before | causal | program order | 量化 flag/workspace 清零成本和同步等待。 |
| workspace_reset_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1441 | 1441 | after | actual | tracing-only MTE3_S | 不把未完成的 MTE3 清零尾部错误归入后续本地 barrier。 |
| local_prepare_sync_begin_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1466 | 1466 | before | causal | program order | 直接量化 AIC 等待 AIV 输入准备和 flag 可见的本地 barrier。 |
| local_prepare_sync_end_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1466 | 1466 | after | actual | production rendezvous | 区分单卡角色同步与随后跨 rank 同步。 |
| cross_rank_sync_begin_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1179 | 1179 | before | causal | program order | 直接量化 Dispatch 前由最慢 rank 决定的全卡等待。 |
| cross_rank_sync_end_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1179 | 1179 | after | actual | production remote barrier | 把全 rank 等待与本地专家 buffer 初始化分开。 |
| expert_buffer_init_begin_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1031 | 1031 | before | causal | program order | 量化进入首轮 Dispatch/GMM 前的专家阶段 buffer 和状态初始化。 |
| expert_buffer_init_end_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1038 | 1038 | after | actual | program order | 分离本地视图构造与后续 cursor/problem 状态建立及首个 Dispatch。 |
| work_end_S | AIC, AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1519 | 1519 | after | actual | production completion plus program order | 把有效工作与 tracing 终结同步/落盘分离。 |
| dispatch_task_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_token_dispatch.h:624 | 624 | before | causal | program order | 只绘制实际拥有 token 行的 Dispatch helper，排除 count/cursor 和无工作专家。 |
| dispatch_task_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_token_dispatch.h:624 | 624 | after | actual | production helper's existing ring drain and MTE3_S | 只覆盖本核实际负责的专家行 Dispatch，不包住下一专家或 cursor 推进。 |
| dispatch_scalar_gap_begin_S | AIV1 | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_token_dispatch.h:624 | 624 | after | causal | program order after the existing Dispatch completion contract | 直接量化相邻有效 Dispatch 之间由专家 count 查询、rolling owner 规划、Wave cursor 推进以及相邻预取调用衔接产生的 Scalar 空隙。 |
| dispatch_scalar_gap_end_S | AIV1 | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_token_dispatch.h:624 | 624 | before | actual | none; same-pipeline program order | 判断 HTML 中 Dispatch 之间的白色区间是否就是 AIV1 的逐专家 Scalar 扫描，而不是隐藏的通信或等待。 |
| gmm1_scalar_setup_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/common/mega_moe_trace.h:371 | 371 | before | causal | program order | 判断 GMM1 空洞是否来自逐专家状态准备、地址更新和 rolling ownership 跳过，而不是通信依赖。 |
| gmm1_scalar_setup_end_S | AIC | mc2/mega_moe/op_kernel/arch35/common/mega_moe_trace.h:595 | 595 | before | actual | program order | 把 Scalar 调度与随后真实 Dispatch-ready 等待分开。 |
| gmm1_dispatch_wait_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:106 | 106 | before | causal | program order | 直接验证下一段 GMM1 是否仍被 Dispatch-ready flag 阻塞。 |
| gmm1_dispatch_wait_end_S | AIC | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:114 | 114 | before | actual | bypass-GM polling then program order | 量化 Dispatch 是否真的落在两段 GMM1 之间。 |
| gmm1_vector_wait_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:122 | 122 | before | causal | program order | 判断前一段 SwiGLU 是否尚未释放 GMM1 输出槽，从而阻塞下一段 GMM1。 |
| gmm1_vector_wait_end_S | AIC | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:130 | 130 | before | actual | production event wait then program order | 把 SwiGLU backpressure 与下一段 GMM1 实际 MMAD 分开。 |
| gmm1_problem_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:71 | 71 | before | causal | SCALAR, FIX_S, or V_S selected by production variant | 排除每个 tile 前的 Dispatch-ready/ping-pong 等待，并展开 Wave 内实际计算块。 |
| gmm1_problem_end_S | AIC | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:95 | 95 | before | actual | production or tracing-only FIX_S | 只把当前 tile 真正完成的 FIX 工作染色，使 tile 间依赖等待保持空白。 |
| swiglu_problem_begin_S | AIV0 | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:74 | 74 | before | causal | V_S or production program order | 排除每个 tile 前的 WaitForCube/ready，并展开 Wave 内多次 SwiGLU。 |
| swiglu_problem_end_S | AIV0 | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:98 | 98 | before | actual | production MTE3_S then program order | 不把下一 tile 前的等待算入 SwiGLU。 |
| gmm2_problem_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h:535 | 535 | before | causal | bypass-GM ready polling then program order | 排除每个 tile 前的 SwiGLU-ready 轮询并展开 Wave 内实际 GMM2 块。 |
| gmm2_problem_end_S | AIC | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h:542 | 542 | before | actual | production FIX_S | 让每个 GMM2 tile 覆盖真实 FIX 完成，而不是只显示 Scalar 发出。 |
| gmm2_input_wait_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h:515 | 515 | before | causal | program order | 显式显示同一 Wave 内每个 GMM2 problem 前等待 SwiGLU/Quant 输出的时间。 |
| gmm2_input_wait_end_S | AIC | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h:523 | 523 | before | actual | production bypass-GM polling | 直接量化每个 GMM2 problem 的 activation 输入等待。 |
| gmm2_ready_fix_wait_begin_S | AIC | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h:699 | 699 | before | causal | program order | 区分 GMM2 MMAD/FIX 发出与生产 group-ready 的逐 tile FIX completion 等待。 |
| gmm2_ready_fix_wait_end_S | AIC | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h:707 | 707 | before | actual | production FIX_S | 将逐 tile FIX completion 与随后 AtomicAdd counter 分开；本段不包含 counter 原子写。 |
| combine_group_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h:550 | 550 | before | causal | bypass-GM polling then program order | 观察一对一 sequence ready 后每个 Combine tile 的真实发送，并排除 flag 轮询。 |
| combine_group_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h:562 | 562 | before | actual | tracing-only MTE3_S | 只覆盖当前 Combine 工作单元，并在远端发送完成后结束。 |
| final_combine_drain_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1009 | 1009 | before | causal | program order | 确认最后一个 Combine group 后是否仍有 row-ring MTE3 尾部需要回收。 |
| final_combine_drain_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1009 | 1009 | after | actual | production MTE3_MTE2 event waits | 把 Combine 发送尾部与后续 MoE 收尾逻辑分开。 |
| moe_epilogue_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1145 | 1145 | before | causal | program order | 显示最终 Combine 与本地汇合之间的 EndSync、状态复位和 count 导出成本。 |
| moe_epilogue_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1152 | 1152 | after | actual | production helper completion | 定位 Combine 后非 barrier 的 V 核尾部成本。 |
| moe_tail_sync_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1198 | 1198 | before | causal | program order | 直接量化快 V 核等待慢 V 核完成最终 Combine/尾部工作的时间。 |
| moe_tail_sync_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1198 | 1198 | after | actual | production PIPE_ALL plus SyncAll | 把本卡负载不均衡等待与输出跨卡同步分开。 |
| shared_expert_gmm2_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1217 | 1217 | before | causal | program order | 显示 V 侧在共享专家 GMM2 调用路径上的调度区间。 |
| shared_expert_gmm2_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1217 | 1217 | after | issue | program order only | 区分共享专家调用开销与后续输出同步；共享结果等待可能继续落在 Unpermute 内。 |
| output_cross_rank_sync_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1485 | 1485 | before | causal | program order | 直接量化本 rank 等待其他 rank 完成 Combine 发送的时间。 |
| output_cross_rank_sync_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1485 | 1485 | after | actual | production rank handshake | 把慢 rank 的 Combine 尾部与本地 Unpermute 分开。 |
| unpermute_init_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1499 | 1499 | before | causal | program order | 把 UB view/config 构造与 Unpermute 数据流水分开。 |
| unpermute_init_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1499 | 1499 | after | actual | program order | 识别极端配置下 buffer 规划本身是否异常。 |
| unpermute_begin_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1508 | 1508 | before | causal | program order | 量化 Combine 接收数据、top-k/shared 累加与最终 y 写回的整体成本。 |
| unpermute_end_S | AIV0, AIV1 | mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1508 | 1508 | after | actual | production MTE3_V event wait | 确认尾部慢点是否来自本地反量化/累加/写 y，而不是同步。 |

## Point evidence

### `process_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1387` in `MegaMoeA8W8Wave::Process`; `before` target line `1387`.
- Cardinality: 每个物理 AI Block 的 AIC/AIV0/AIV1 各一次。
- Boundary claim: `causal` — Process 的第一条诊断指令，尚未执行正常算子语句。
- Why: 提供同一 AI Block 内三角色共同参考起点。
- Dependency: kernel role entered Process
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker precedes the first production statement
- Perturbation: 每角色一次时钟读取和寄存器赋值。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1385  #endif
  1386      // 保存入口时的溢出模式，并初始化输入准备阶段使用的 UB。
  1387      int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
  1388      SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
  1389  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

### `input_buffer_init_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1397` in `MegaMoeA8W8Wave::Process`; `before` target line `1397`.
- Cardinality: 每个 AIV0/AIV1 各一次。
- Boundary claim: `causal` — 即将构造 quant/route-index/reset 的 UB 视图并发出 resetTensor Duplicate。
- Why: 分离输入准备 UB 布局与后续 token 量化。
- Dependency: overflow mode configured
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker is immediately before SendAndQuantBuffInit
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1395      }
  1396  #endif
  1397      SendAndQuantBuffInit();
  1398  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1399      if constexpr (g_coreType == AIV) {
```

### `input_buffer_init_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1397` in `MegaMoeA8W8Wave::Process`; `after` target line `1397`.
- Cardinality: 与 input_buffer_init_begin_S 一一对应。
- Boundary claim: `issue` — UB 视图已构造且 Duplicate 已发出；Vector 尾部可继续运行。
- Why: 量化前单独显示 UB 地址构造和零张量发出成本。
- Dependency: SendAndQuantBuffInit returned
- Pipeline proof: `Scalar plus Vector issue` → `Scalar` using `program order only` — no tracing-only Vector drain is inserted
- Perturbation: 每个 AIV 一次时钟读取，不新增 Vector drain。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1395      }
  1396  #endif
  1397      SendAndQuantBuffInit();
  1398  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1399      if constexpr (g_coreType == AIV) {
```

### `input_quantize_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1417` in `MegaMoeA8W8Wave::Process`; `before` target line `1417`.
- Cardinality: 每个 AIV0/AIV1 各一次。
- Boundary claim: `causal` — 即将进入 QuantizeLocalTokens。
- Why: 量化本卡输入 token 的固定成本独立可见。
- Dependency: input UB views are configured
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker precedes QuantizeLocalTokens
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1415      }
  1416  #endif
  1417      QuantizeLocalTokens<QuantMode, QuantOutType, ActivationType, TopkWeightsType, TopkWeightsPrefetch>(
  1418          aivJob_, commonConfig_, params_, quantProcessConfig_, quantProcessScratch_);
  1419  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

### `input_quantize_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1417` in `MegaMoeA8W8Wave::Process`; `after` target line `1417`.
- Cardinality: 与 input_quantize_begin_S 一一对应。
- Boundary claim: `actual` — 本核负责 token 的量化结果已写回 GM。
- Why: 区分 token 量化与后续 compact route-index 通信。
- Dependency: QuantizeLocalTokens returned
- Pipeline proof: `MTE2/Vector/MTE3` → `Scalar` using `production MTE3_MTE2 ring drain` — QuantizeLocalTokens waits both MTE3_MTE2 events before return
- Perturbation: 每个 AIV 一次时钟读取；复用 helper 已有双 buffer MTE3 回收。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1415      }
  1416  #endif
  1417      QuantizeLocalTokens<QuantMode, QuantOutType, ActivationType, TopkWeightsType, TopkWeightsPrefetch>(
  1418          aivJob_, commonConfig_, params_, quantProcessConfig_, quantProcessScratch_);
  1419  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

### `route_index_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1429` in `MegaMoeA8W8Wave::Process`; `before` target line `1429`.
- Cardinality: 每个 AIV0/AIV1 各一次。
- Boundary claim: `causal` — 即将进入 GatherAndSendExpertCompactRoutes。
- Why: 单独量化 compact route index 构造、count 整理与跨 rank 发送。
- Dependency: input quantization helper returned
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker immediately precedes the compact route-index helper
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1427      }
  1428  #endif
  1429      GatherAndSendExpertCompactRoutes(aivJob_, commonConfig_, params_, g_winRankAddr_, sendMaskConfig_,
  1430                                       sendMaskScratch_);
  1431  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

### `route_index_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1429` in `MegaMoeA8W8Wave::Process`; `after` target line `1429`.
- Cardinality: 与 route_index_begin_S 一一对应。
- Boundary claim: `actual` — 本核所有 compact route/count ring 槽的 MTE3 写已完成。
- Why: 把 compact route index/count 远端发送完成与 workspace 清零分开。
- Dependency: GatherAndSendExpertCompactRoutes returned
- Pipeline proof: `Vector/MTE3` → `Scalar` using `production ring drain` — helper returns after every active compact-route MTE3 slot is drained
- Perturbation: 每个 AIV 一次时钟读取；复用 helper 生产事件。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1427      }
  1428  #endif
  1429      GatherAndSendExpertCompactRoutes(aivJob_, commonConfig_, params_, g_winRankAddr_, sendMaskConfig_,
  1430                                       sendMaskScratch_);
  1431  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

### `workspace_reset_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1441` in `MegaMoeA8W8Wave::Process`; `before` target line `1441`.
- Cardinality: 每个 AIV0/AIV1 各一次。
- Boundary claim: `causal` — 即将发出本核负责的 workspace 清零。
- Why: 量化 flag/workspace 清零成本和同步等待。
- Dependency: compact route-index helper returned
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker immediately precedes ResetSyncStatus
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1439      }
  1440  #endif
  1441      ResetSyncStatus<TopkWeightsPrefetch>(aivJob_, params_, resetBatchElementCount_, resetTensor_);
  1442  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1443      if constexpr (g_coreType == AIV) {
```

### `workspace_reset_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1441` in `MegaMoeA8W8Wave::Process`; `after` target line `1441`.
- Cardinality: 与 workspace_reset_begin_S 一一对应。
- Boundary claim: `actual` — 本核此前发出的 workspace 清零 MTE3 已完成。
- Why: 不把未完成的 MTE3 清零尾部错误归入后续本地 barrier。
- Dependency: ResetDispatchWorkspace returned
- Pipeline proof: `MTE3` → `Scalar` using `tracing-only MTE3_S` — MTE3_S SetFlag/WaitFlag completes immediately before sampling
- Perturbation: 诊断态每个 AIV 增加一次 MTE3_S；会把原本由随后 SyncAll 吸收的等待前移。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1439      }
  1440  #endif
  1441      ResetSyncStatus<TopkWeightsPrefetch>(aivJob_, params_, resetBatchElementCount_, resetTensor_);
  1442  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1443      if constexpr (g_coreType == AIV) {
```

### `local_prepare_sync_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1466` in `MegaMoeA8W8Wave::Process`; `before` target line `1466`.
- Cardinality: 每个物理 AI Block 的三角色各一次。
- Boundary claim: `causal` — 即将进入输入准备后的 SyncAll。
- Why: 直接量化 AIC 等待 AIV 输入准备和 flag 可见的本地 barrier。
- Dependency: role-local input preparation issued
- Pipeline proof: `mixed local pipelines` → `Scalar` using `program order` — marker is immediately before the mixed AIC/AIV rendezvous
- Perturbation: 每角色一次时钟读取。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1464      }
  1465  #endif
  1466      SyncAll<false>(); // AIC 等待 AIV 完成输入准备与 flag 清零后再进入计算
  1467  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1468      if (CriticalPathTracePtr() != nullptr) {
```

### `local_prepare_sync_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1466` in `MegaMoeA8W8Wave::Process`; `after` target line `1466`.
- Cardinality: 与 local_prepare_sync_begin_S 一一对应。
- Boundary claim: `actual` — 本 AI Block 所有角色已越过输入准备 rendezvous。
- Why: 区分单卡角色同步与随后跨 rank 同步。
- Dependency: SyncAll returned
- Pipeline proof: `mixed local pipelines` → `Scalar` using `production rendezvous` — sample occurs after SyncAll
- Perturbation: 每角色一次时钟读取。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1464      }
  1465  #endif
  1466      SyncAll<false>(); // AIC 等待 AIV 完成输入准备与 flag 清零后再进入计算
  1467  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1468      if (CriticalPathTracePtr() != nullptr) {
```

### `cross_rank_sync_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1179` in `MegaMoeA8W8Wave::ProcessGmmPipeline`; `before` target line `1179`.
- Cardinality: 每个物理 AI Block 的三角色各一次。
- Boundary claim: `causal` — 即将进入 CrossRankSyncInWorldSize。
- Why: 直接量化 Dispatch 前由最慢 rank 决定的全卡等待。
- Dependency: local input preparation rendezvous completed
- Pipeline proof: `Scalar/remote GM` → `Scalar` using `program order` — marker immediately precedes the first cross-rank barrier
- Perturbation: 每角色一次时钟读取。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1177      }
  1178  #endif
  1179      CrossRankSyncInWorldSize(params_.peermemInfo.rankSyncInWorldPtr, rankId_, worldSize_, aivJob_);
  1180  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1181      if (CriticalPathTracePtr() != nullptr) {
```

### `cross_rank_sync_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1179` in `MegaMoeA8W8Wave::ProcessGmmPipeline`; `after` target line `1179`.
- Cardinality: 与 cross_rank_sync_begin_S 一一对应。
- Boundary claim: `actual` — 本角色已观察到所有 rank 输入准备完成。
- Why: 把全 rank 等待与本地专家 buffer 初始化分开。
- Dependency: CrossRankSyncInWorldSize returned
- Pipeline proof: `Scalar/remote GM` → `Scalar` using `production remote barrier` — sample occurs after the cross-rank wait
- Perturbation: 每角色一次时钟读取。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1177      }
  1178  #endif
  1179      CrossRankSyncInWorldSize(params_.peermemInfo.rankSyncInWorldPtr, rankId_, worldSize_, aivJob_);
  1180  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1181      if (CriticalPathTracePtr() != nullptr) {
```

### `expert_buffer_init_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1031` in `MegaMoeA8W8Wave::ProcessMoeExpertStages`; `before` target line `1031`.
- Cardinality: 每个物理 AI Block 的三角色各一次。
- Boundary claim: `causal` — 即将构造 Dispatch、Combine 和 token-count export 的本地视图。
- Why: 量化进入首轮 Dispatch/GMM 前的专家阶段 buffer 和状态初始化。
- Dependency: cross-rank input barrier completed
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker precedes the three buffer initialization helpers
- Perturbation: 每角色一次时钟读取。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1029      }
  1030  #endif
  1031      tokenDispatchScratch_.expertRevNumsGlobalTensor.SetGlobalBuffer(
  1032          reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.expertRevTokenNumsPtr));
  1033      DispatchBuffInit();
```

### `expert_buffer_init_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1038` in `MegaMoeA8W8Wave::ProcessMoeExpertStages`; `after` target line `1038`.
- Cardinality: 与 expert_buffer_init_begin_S 一一对应。
- Boundary claim: `actual` — 三个 buffer 初始化 helper 已完成；这些 helper 仅构造 Scalar/LocalTensor 状态。
- Why: 分离本地视图构造与后续 cursor/problem 状态建立及首个 Dispatch。
- Dependency: all expert buffer helpers returned
- Pipeline proof: `Scalar` → `Scalar` using `program order` — helpers only bind buffer views and scalar fields
- Perturbation: 每角色一次时钟读取。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1036          combineBufferConfig = InitCombineBuffers();
  1037      }
  1038      PrepareMoeExpertTokenCountTable(commonConfig_, countWorkspace_, params_, tokenDispatchScratch_);
  1039  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1040      if (CriticalPathTracePtr() != nullptr) {
```

### `work_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1519` in `MegaMoeA8W8Wave::Process`; `after` target line `1519`.
- Cardinality: 每个物理 AI Block 的 AIC/AIV0/AIV1 各一次。
- Boundary claim: `actual` — 正常 routed/shared 流水、跨 rank 尾部和 Unpermute 已结束，尚未进入 tracing rendezvous。
- Why: 把有效工作与 tracing 终结同步/落盘分离。
- Dependency: all production work for this role returned
- Pipeline proof: `mixed production pipelines` → `Scalar` using `production completion plus program order` — sample is after ProcessGmmPipeline, final AIV cross-rank wait/Unpermute, and control restoration
- Perturbation: 每角色一次时钟读取和寄存器赋值。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1517      // 恢复入口时保存的溢出模式。
  1518      SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
  1519      exceptionDump_.UpdateStage(MegaMoeImpl::Stage::COMPLETE);
  1520  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1521      if (CriticalPathTracePtr() != nullptr) {
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

- Where: `mc2/mega_moe/op_kernel/arch35/common/mega_moe_trace.h:371` in `CriticalPathTrace::PrepareGmm1Gap`; `before` target line `371`.
- Cardinality: 每个 AIC 从上一个活跃 GMM1 problem 后开始，跨越连续空专家或无 owner problem，直到下一个 owned tile 的 ready 检查。
- Boundary claim: `causal` — 开始为后续 GMM1 owned problem 扫描并准备专家状态；连续无 owner problem 合并为一个低扰动区间。
- Why: 判断 GMM1 空洞是否来自逐专家状态准备、地址更新和 rolling ownership 跳过，而不是通信依赖。
- Dependency: ProcessGmm1Wave entered the next expert/problem attempt
- Pipeline proof: `Scalar` → `Scalar` using `program order` — the first pending setup timestamp is sampled before expert state preparation; later skipped experts retain that timestamp
- Perturbation: 每段一次起点时钟；连续无 owner 专家不逐项写 trace，避免诊断写放大空洞。
- Source SHA-256: `842d9f48bde18c13aaaebb9f035bbf0ece8e5098502890a2ce93eb6448f87340`

```text
   369          // TRACE_POINT: gmm1_scalar_setup_begin_S
   370          if (gmm1SetupBegin_ == 0U) {
   371              gmm1SetupBegin_ = begin;
   372          }
   373          gmm1GapWave_ = waveIdx;
```

### `gmm1_scalar_setup_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/common/mega_moe_trace.h:595` in `CriticalPathTrace::EndGmm1Setup`; `before` target line `595`.
- Cardinality: 与聚合后的 gmm1_scalar_setup_begin_S 一一对应。
- Boundary claim: `actual` — 本核已找到下一个 owned tile 并完成 BlockMmad/problem 构造，或本 Wave 已无 owned tile。
- Why: 把 Scalar 调度与随后真实 Dispatch-ready 等待分开。
- Dependency: owned-tile ready check is about to start, or ProcessGmm1Wave is returning
- Pipeline proof: `Scalar` → `Scalar` using `program order` — active paths close setup immediately before WaitForGmm1InputReady; tail paths close it at Wave exit
- Perturbation: 每聚合段一次结束时钟和四个 scratch 标量写。
- Source SHA-256: `842d9f48bde18c13aaaebb9f035bbf0ece8e5098502890a2ce93eb6448f87340`

```text
   593          }
   594          // TRACE_POINT: gmm1_scalar_setup_end_S
   595          AppendSegment(gmm1GapWave_, gmm1GapExpert_, gmm1SetupBegin_, value, 2U, stage2CapturedCount_);
   596          gmm1SetupBegin_ = 0U;
   597      }
```

### `gmm1_dispatch_wait_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:106` in `TraceGmm1DispatchWaitBegin`; `before` target line `106`.
- Cardinality: 每个 AIC 活跃 GMM1 problem 的首个 owned tile。
- Boundary claim: `causal` — 即将轮询当前 256-row group 的 Dispatch-ready flag。
- Why: 直接验证下一段 GMM1 是否仍被 Dispatch-ready flag 阻塞。
- Dependency: scheduler assigned the current tile to this AIC
- Pipeline proof: `Scalar` → `Scalar` using `program order` — helper is called immediately before WaitForGmm1InputReady
- Perturbation: 每个活跃 problem 一次时钟；setup scratch 写位于该采样之前。
- Source SHA-256: `bde807a62afaedcc1272fc82b1f2226fa0de172a849123304bd0bf644850d0f5`

```text
   104      if (trace != nullptr && trace->NeedsStage0Begin()) {
   105          // TRACE_POINT: gmm1_dispatch_wait_begin_S
   106          trace->BeginGmm1DispatchWait(CriticalPathTrace::Now());
   107      }
   108  }
```

### `gmm1_dispatch_wait_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:114` in `TraceGmm1DispatchWaitEnd`; `before` target line `114`.
- Cardinality: 与 gmm1_dispatch_wait_begin_S 一一对应。
- Boundary claim: `actual` — 目标 Dispatch-ready 值已经被本核观察到。
- Why: 量化 Dispatch 是否真的落在两段 GMM1 之间。
- Dependency: WaitForGmm1InputReady returned
- Pipeline proof: `AIV1 remote MTE3 plus ready publication` → `Scalar` using `bypass-GM polling then program order` — helper is called immediately after WaitForGmm1InputReady
- Perturbation: 每个活跃 problem 一次时钟和四个 scratch 标量写。
- Source SHA-256: `bde807a62afaedcc1272fc82b1f2226fa0de172a849123304bd0bf644850d0f5`

```text
   112      if (trace != nullptr && trace->NeedsStage0Begin()) {
   113          // TRACE_POINT: gmm1_dispatch_wait_end_S
   114          trace->EndGmm1DispatchWait(CriticalPathTrace::Now());
   115      }
   116  }
```

### `gmm1_vector_wait_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:122` in `TraceGmm1VectorWaitBegin`; `before` target line `122`.
- Cardinality: 非 prefetch GMM1 中确实需要回收 UB ping-pong 槽的活跃 problem。
- Boundary claim: `causal` — Dispatch 已 ready，即将等待 AIV0 归还当前 ping-pong 槽。
- Why: 判断前一段 SwiGLU 是否尚未释放 GMM1 输出槽，从而阻塞下一段 GMM1。
- Dependency: production vecSetSyncCom requires a ping-pong recycle
- Pipeline proof: `AIV0 Vector/MTE3` → `Scalar` using `program order` — helper is called only inside the branch that immediately invokes WaitForVector
- Perturbation: 仅在生产路径本来会 WaitForVector 时增加一次时钟。
- Source SHA-256: `bde807a62afaedcc1272fc82b1f2226fa0de172a849123304bd0bf644850d0f5`

```text
   120      if (trace != nullptr && trace->NeedsStage0Begin()) {
   121          // TRACE_POINT: gmm1_vector_wait_begin_S
   122          trace->BeginGmm1VectorWait(CriticalPathTrace::Now());
   123      }
   124  }
```

### `gmm1_vector_wait_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:130` in `TraceGmm1VectorWaitEnd`; `before` target line `130`.
- Cardinality: 与 gmm1_vector_wait_begin_S 一一对应。
- Boundary claim: `actual` — AIV0 已归还所需 ping-pong 槽。
- Why: 把 SwiGLU backpressure 与下一段 GMM1 实际 MMAD 分开。
- Dependency: WaitForVector returned
- Pipeline proof: `AIV0 Vector/MTE3` → `Scalar` using `production event wait then program order` — helper is called immediately after WaitForVector
- Perturbation: 每次真实 WaitForVector 增加一次时钟和四个 scratch 标量写。
- Source SHA-256: `bde807a62afaedcc1272fc82b1f2226fa0de172a849123304bd0bf644850d0f5`

```text
   128      if (trace != nullptr && trace->NeedsStage0Begin()) {
   129          // TRACE_POINT: gmm1_vector_wait_end_S
   130          trace->EndGmm1VectorWait(CriticalPathTrace::Now());
   131      }
   132  }
```

### `gmm1_problem_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:71` in `TraceGmm1ProblemBeginAfterReady`; `before` target line `71`.
- Cardinality: 所有 AIC，每个本核 owned GMM1 scheduler tile 一次。
- Boundary claim: `causal` — 当前 owned GMM1 tile 已 ready，即将发出。
- Why: 排除每个 tile 前的 Dispatch-ready/ping-pong 等待，并展开 Wave 内实际计算块。
- Dependency: first owned tile's production readiness wait returned
- Pipeline proof: `Dispatch/FIX or ping-pong producer` → `Scalar` using `SCALAR, FIX_S, or V_S selected by production variant` — all routed generic call sites invoke the helper after their production wait
- Perturbation: 每活跃 tile 一次条件判断、必要的因果 bridge 和时钟。
- Source SHA-256: `bde807a62afaedcc1272fc82b1f2226fa0de172a849123304bd0bf644850d0f5`

```text
    69      if constexpr (g_coreType == AscendC::AIC) {
    70          // TRACE_POINT: gmm1_problem_begin_S
    71          trace->BeginStage0(CriticalPathTrace::Now());
    72      } else {
    73          // TRACE_POINT: swiglu_problem_begin_S
```

### `gmm1_problem_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:95` in `TraceGmm1TileEnd`; `before` target line `95`.
- Cardinality: 与 gmm1_problem_begin_S 逐 tile 一一对应。
- Boundary claim: `actual` — 当前 GMM1 tile 的 FIX 工作已完成并桥接到 Scalar。
- Why: 只把当前 tile 真正完成的 FIX 工作染色，使 tile 间依赖等待保持空白。
- Dependency: the current owned GMM1 tile helper returned
- Pipeline proof: `FIX` → `Scalar` using `production or tracing-only FIX_S` — prefetch callers invoke after production FIX_S; non-prefetch callers request the FIX bridge in TraceGmm1TileEnd
- Perturbation: 非 prefetch 路径每个 owned tile 增加一次 FIX_S；prefetch 复用生产 FIX_S；每 tile 写一个稀疏段。
- Source SHA-256: `bde807a62afaedcc1272fc82b1f2226fa0de172a849123304bd0bf644850d0f5`

```text
    93      if constexpr (g_coreType == AscendC::AIC) {
    94          // TRACE_POINT: gmm1_problem_end_S
    95          trace->EndStage0(CriticalPathTrace::Now());
    96      } else {
    97          // TRACE_POINT: swiglu_problem_end_S
```

### `swiglu_problem_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:74` in `TraceGmm1ProblemBeginAfterReady`; `before` target line `74`.
- Cardinality: 所有 AIV0，每个本核 owned epilogue scheduler tile 一次。
- Boundary claim: `causal` — 当前 owned epilogue tile 已 ready，即将执行 SwiGLU/MX。
- Why: 排除每个 tile 前的 WaitForCube/ready，并展开 Wave 内多次 SwiGLU。
- Dependency: first owned epilogue tile readiness wait returned
- Pipeline proof: `GMM1/FIX` → `Scalar` using `V_S or production program order` — marker is after WaitForCube/WaitGmm1TileStatus and before the first epilogue tile helper
- Perturbation: 每活跃 tile 一次条件判断、因果 bridge 和时钟。
- Source SHA-256: `bde807a62afaedcc1272fc82b1f2226fa0de172a849123304bd0bf644850d0f5`

```text
    72      } else {
    73          // TRACE_POINT: swiglu_problem_begin_S
    74          trace->BeginStage0(CriticalPathTrace::Now());
    75      }
    76  }
```

### `swiglu_problem_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h:98` in `TraceGmm1TileEnd`; `before` target line `98`.
- Cardinality: 与 swiglu_problem_begin_S 逐 tile 一一对应。
- Boundary claim: `actual` — 当前 SwiGLU/MX tile 的 Vector 与 MTE3 输出已完成。
- Why: 不把下一 tile 前的等待算入 SwiGLU。
- Dependency: the current epilogue tile helper returned on AIV0
- Pipeline proof: `Vector/MTE3` → `Scalar` using `production MTE3_S then program order` — activationQuantOp drains MTE3 before returning; caller immediately invokes TraceGmm1TileEnd
- Perturbation: 每 tile 一次时钟和四个 scratch 写；复用 activationQuantOp 内已有的 MTE3_S，不增加排空。
- Source SHA-256: `bde807a62afaedcc1272fc82b1f2226fa0de172a849123304bd0bf644850d0f5`

```text
    96      } else {
    97          // TRACE_POINT: swiglu_problem_end_S
    98          trace->EndStage0(CriticalPathTrace::Now());
    99      }
   100  }
```

### `gmm2_problem_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h:535` in `TraceGmm2ProblemBeginAfterReady`; `before` target line `535`.
- Cardinality: 所有 AIC，每个本核 owned GMM2 scheduler tile 一次。
- Boundary claim: `causal` — 当前 owned GMM2 tile 已 ready，即将发出。
- Why: 排除每个 tile 前的 SwiGLU-ready 轮询并展开 Wave 内实际 GMM2 块。
- Dependency: first owned tile observed SwiGLU-ready
- Pipeline proof: `AIV0 MTE3/ready` → `Scalar` using `bypass-GM ready polling then program order` — helper is called only after WaitForGmm2InputReady returns
- Perturbation: 每活跃 tile 一次条件判断和时钟。
- Source SHA-256: `3002f302a7d41494d530425ccd3e5ad34d55384941e16f701020d79a45d98344`

```text
   533      }
   534      // TRACE_POINT: gmm2_problem_begin_S
   535      trace->BeginStage1(CriticalPathTrace::Now());
   536  }
   537
```

### `gmm2_problem_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h:542` in `TraceGmm2TileEndAfterFix`; `before` target line `542`.
- Cardinality: 与 gmm2_problem_begin_S 逐 tile 一一对应。
- Boundary claim: `actual` — 当前 GMM2 tile 的 FIX 写回及生产通知已完成。
- Why: 让每个 GMM2 tile 覆盖真实 FIX 完成，而不是只显示 Scalar 发出。
- Dependency: the current GMM2 tile production FIX and ready publication returned
- Pipeline proof: `FIX` → `Scalar` using `production FIX_S` — both pairwise and group-ready callers invoke only after their production FIX_S path
- Perturbation: 复用 pairwise/group-ready 生产 FIX_S；每 tile 一次时钟和四个 scratch 写。
- Source SHA-256: `3002f302a7d41494d530425ccd3e5ad34d55384941e16f701020d79a45d98344`

```text
   540      if (trace != nullptr && trace->HasStage1Begin()) {
   541          // TRACE_POINT: gmm2_problem_end_S
   542          trace->EndStage1(CriticalPathTrace::Now());
   543      }
   544  }
```

### `gmm2_input_wait_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h:515` in `TraceGmm2InputWaitBegin`; `before` target line `515`.
- Cardinality: 所有有 owned tile 的 AIC，每次进入生产 GMM2 activation-ready 轮询一次。
- Boundary claim: `causal` — 当前 owned M group 即将等待 activationToGmm2 ready。
- Why: 显式显示同一 Wave 内每个 GMM2 problem 前等待 SwiGLU/Quant 输出的时间。
- Dependency: the scheduler selected an owned GMM2 M group
- Pipeline proof: `Scalar` → `Scalar` using `program order` — sample is immediately before the production WaitForGmm2InputReady call
- Perturbation: 每次生产 ready 轮询增加一次起点时钟；不新增同步。
- Source SHA-256: `3002f302a7d41494d530425ccd3e5ad34d55384941e16f701020d79a45d98344`

```text
   513      if (trace != nullptr) {
   514          // TRACE_POINT: gmm2_input_wait_begin_S
   515          trace->BeginGmm2InputWait(CriticalPathTrace::Now());
   516      }
   517  }
```

### `gmm2_input_wait_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h:523` in `TraceGmm2InputWaitEnd`; `before` target line `523`.
- Cardinality: 与 gmm2_input_wait_begin_S 一一对应。
- Boundary claim: `actual` — 生产 WaitForGmm2InputReady 已观察到 ready 并返回。
- Why: 直接量化每个 GMM2 problem 的 activation 输入等待。
- Dependency: the production ready wait returned
- Pipeline proof: `AIV0 ready publication` → `Scalar` using `production bypass-GM polling` — sample is immediately after WaitForGmm2InputReady
- Perturbation: 每次 ready 返回增加一次时钟和四个 scratch word 写；不新增同步。
- Source SHA-256: `3002f302a7d41494d530425ccd3e5ad34d55384941e16f701020d79a45d98344`

```text
   521      if (trace != nullptr) {
   522          // TRACE_POINT: gmm2_input_wait_end_S
   523          trace->EndGmm2InputWait(CriticalPathTrace::Now());
   524      }
   525  }
```

### `gmm2_ready_fix_wait_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h:699` in `MegaMoeImpl::NotifyWaveCombineGroupCompletion`; `before` target line `699`.
- Cardinality: 所有 routed A8W8 GMM2 owned N tile，每次发布 256-token group ready 前一次。
- Boundary claim: `causal` — 当前 GMM2 tile 的 blockMmad 已返回，马上等待 FIX 完成后再 AtomicAdd group counter。
- Why: 区分 GMM2 MMAD/FIX 发出与生产 group-ready 的逐 tile FIX completion 等待。
- Dependency: the current blockMmad call returned
- Pipeline proof: `Scalar issue boundary after Cube/FIX submission` → `Scalar` using `program order` — sample immediately precedes the production FIX_S used by group-ready publication
- Perturbation: 每个 owned N tile 增加一次起点时钟；它位于生产 FIX_S 之前，会改变诊断态 Scalar/FIX 调度。
- Source SHA-256: `3002f302a7d41494d530425ccd3e5ad34d55384941e16f701020d79a45d98344`

```text
   697              if (criticalPathTrace != nullptr) {
   698                  // TRACE_POINT: gmm2_ready_fix_wait_begin_S
   699                  criticalPathTrace->BeginGmm2ReadyFixWait(CriticalPathTrace::Now());
   700              }
   701  #endif
```

### `gmm2_ready_fix_wait_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h:707` in `MegaMoeImpl::NotifyWaveCombineGroupCompletion`; `before` target line `707`.
- Cardinality: 与 gmm2_ready_fix_wait_begin_S 一一对应。
- Boundary claim: `actual` — 当前 tile 的 FIX 写回已完成并桥接到 Scalar，但 group counter 尚未 AtomicAdd。
- Why: 将逐 tile FIX completion 与随后 AtomicAdd counter 分开；本段不包含 counter 原子写。
- Dependency: the production FIX_S wait returned
- Pipeline proof: `FIX` → `Scalar` using `production FIX_S` — sample is after WaitFlag<FIX_S> and before AtomicAdd of the group-ready counter
- Perturbation: 每个 owned N tile 一次结束时钟和四个 scratch word 写；时间戳/写 scratch 发生在 AtomicAdd 前。
- Source SHA-256: `3002f302a7d41494d530425ccd3e5ad34d55384941e16f701020d79a45d98344`

```text
   705              if (criticalPathTrace != nullptr) {
   706                  // TRACE_POINT: gmm2_ready_fix_wait_end_S
   707                  criticalPathTrace->EndGmm2ReadyFixWait(CriticalPathTrace::Now());
   708              }
   709  #endif
```

### `combine_group_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h:550` in `TracePairwiseCombineTileBegin`; `before` target line `550`.
- Cardinality: 非量化 A8W8 的配对 AIV1 每个 owned GMM2 tile 一次；量化旧路径仍按 256-token group 一次。
- Boundary claim: `causal` — 当前 pairwise tile sequence 已 ready，即将 MTE2 读取并发送；量化路径表示 group counter 已齐。
- Why: 观察一对一 sequence ready 后每个 Combine tile 的真实发送，并排除 flag 轮询。
- Dependency: the paired AIC tile sequence was observed
- Pipeline proof: `AIC FIX plus bypass-GM sequence` → `Scalar` using `bypass-GM polling then program order` — pairwise helper is called immediately after WaitUntilGmFlagAtLeast
- Perturbation: 每活跃工作单元一次条件判断和时钟。
- Source SHA-256: `3002f302a7d41494d530425ccd3e5ad34d55384941e16f701020d79a45d98344`

```text
   548      if (trace != nullptr && trace->NeedsStage1Begin()) {
   549          // TRACE_POINT: combine_group_begin_S
   550          trace->BeginStage1(CriticalPathTrace::Now());
   551      }
   552  }
```

### `combine_group_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h:562` in `TracePairwiseCombineTileEnd`; `before` target line `562`.
- Cardinality: 与 combine_group_begin_S 的 pairwise tile 或量化 group 工作单元一一对应。
- Boundary claim: `actual` — 当前 tile/group 此前发出的所有 MTE3 send 已完成并桥接到 Scalar。
- Why: 只覆盖当前 Combine 工作单元，并在远端发送完成后结束。
- Dependency: the active pairwise tile has issued all remote sends
- Pipeline proof: `MTE2/MTE3` → `Scalar` using `tracing-only MTE3_S` — MTE3_S SetFlag/WaitFlag is immediately before sampling
- Perturbation: 每个活跃工作单元增加一次 MTE3_S 完成桥、一次时钟和四个 scratch 写。
- Source SHA-256: `3002f302a7d41494d530425ccd3e5ad34d55384941e16f701020d79a45d98344`

```text
   560      AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID5);
   561      // TRACE_POINT: combine_group_end_S
   562      trace->EndStage1(CriticalPathTrace::Now());
   563  }
   564  #endif
```

### `final_combine_drain_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1009` in `MegaMoeA8W8Wave::ProcessCombineWave`; `before` target line `1009`.
- Cardinality: 最终 Wave 上所有 AIV 各一次。
- Boundary claim: `causal` — 最终 Wave 的 Combine 行已全部发出，即将消费仍在途的 row-ring 完成事件。
- Why: 确认最后一个 Combine group 后是否仍有 row-ring MTE3 尾部需要回收。
- Dependency: final Wave Combine helpers returned
- Pipeline proof: `MTE3` → `Scalar` using `program order` — marker is immediately before the production row-ring drain
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1007      }
  1008  #endif
  1009      DrainCombineRowBuffers(rowSequence, activeBufferConfig.rowBufferCount);
  1010  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1011      if (isFinalWave && CriticalPathTracePtr() != nullptr) {
```

### `final_combine_drain_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1009` in `MegaMoeA8W8Wave::ProcessCombineWave`; `after` target line `1009`.
- Cardinality: 与 final_combine_drain_begin_S 一一对应。
- Boundary claim: `actual` — 最终 Wave 的全部已用 row-ring MTE3 完成事件已经消费。
- Why: 把 Combine 发送尾部与后续 MoE 收尾逻辑分开。
- Dependency: DrainCombineRowBuffers returned
- Pipeline proof: `MTE3` → `Scalar` using `production MTE3_MTE2 event waits` — the helper consumes every active row-ring completion event before return
- Perturbation: 每个 AIV 一次时钟读取，复用生产 drain。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1007      }
  1008  #endif
  1009      DrainCombineRowBuffers(rowSequence, activeBufferConfig.rowBufferCount);
  1010  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1011      if (isFinalWave && CriticalPathTracePtr() != nullptr) {
```

### `moe_epilogue_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1145` in `MegaMoeA8W8Wave::ProcessMoeExpertStages`; `before` target line `1145`.
- Cardinality: 每个 AIV 在最终 Combine 返回后一次。
- Boundary claim: `causal` — 最终 Combine/drain 已返回，即将执行 MoE 专家阶段尾部逻辑。
- Why: 显示最终 Combine 与本地汇合之间的 EndSync、状态复位和 count 导出成本。
- Dependency: the final ProcessCombineWave returned
- Pipeline proof: `mixed` → `Scalar` using `program order` — marker is gated by !hasNextWave immediately after ProcessCombineWave
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1143          }
  1144  #endif
  1145          waveBeginPosition = waveEndPosition;
  1146      }
  1147
```

### `moe_epilogue_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1152` in `MegaMoeA8W8Wave::ProcessMoeExpertStages`; `after` target line `1152`.
- Cardinality: 与 moe_epilogue_begin_S 一一对应。
- Boundary claim: `actual` — EndSync、ping-pong 复位和 compact expert count 导出均已返回。
- Why: 定位 Combine 后非 barrier 的 V 核尾部成本。
- Dependency: expert-count export returned
- Pipeline proof: `Scalar/MTE2/MTE3` → `Scalar` using `production helper completion` — the only exporting AIV drains its MTE3 before return; other AIVs have no export work
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1150      }
  1151      gmm1PingPongIdx_ = 0;
  1152      ExportCompactExpertTokenCounts(commonConfig_, countWorkspace_, params_, tokenDispatchScratch_);
  1153  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1154      if constexpr (g_coreType == AIV) {
```

### `moe_tail_sync_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1198` in `MegaMoeA8W8Wave::ProcessGmmPipeline`; `before` target line `1198`.
- Cardinality: 所有 AIV 在 MoE 专家阶段返回后一次。
- Boundary claim: `causal` — 即将执行 MoE 尾部 PIPE_ALL 和本卡 AIV SyncAll。
- Why: 直接量化快 V 核等待慢 V 核完成最终 Combine/尾部工作的时间。
- Dependency: ProcessMoeExpertStages returned
- Pipeline proof: `mixed local pipelines` → `Scalar` using `program order` — marker precedes the production PIPE_ALL/SyncAll pair
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1196  #endif
  1197          PipeBarrier<PIPE_ALL>();
  1198          SyncAll<true>();
  1199  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1200          if (CriticalPathTracePtr() != nullptr) {
```

### `moe_tail_sync_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1198` in `MegaMoeA8W8Wave::ProcessGmmPipeline`; `after` target line `1198`.
- Cardinality: 与 moe_tail_sync_begin_S 一一对应。
- Boundary claim: `actual` — 本卡所有 AIV 已完成 MoE 专家流水并通过 SyncAll。
- Why: 把本卡负载不均衡等待与输出跨卡同步分开。
- Dependency: production AIV SyncAll returned
- Pipeline proof: `all local AIV pipelines` → `Scalar` using `production PIPE_ALL plus SyncAll` — marker is immediately after the existing synchronization
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1196  #endif
  1197          PipeBarrier<PIPE_ALL>();
  1198          SyncAll<true>();
  1199  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1200          if (CriticalPathTracePtr() != nullptr) {
```

### `shared_expert_gmm2_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1217` in `MegaMoeA8W8Wave::ProcessGmmPipeline`; `before` target line `1217`.
- Cardinality: 每个 AIV 各一次；sharedExpertNum=0 时仅覆盖分支判断。
- Boundary claim: `causal` — 即将判断并按需进入共享专家 GMM2 helper；实际 Cube 计算仍由 AIC 完成。
- Why: 显示 V 侧在共享专家 GMM2 调用路径上的调度区间。
- Dependency: MoE tail SyncAll returned
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker immediately precedes the sharedExpertNum branch on AIV roles
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1215      if (sharedExpertNum_ > 0) {
  1216          exceptionDump_.UpdateStage(MegaMoeImpl::Stage::SHARED_EXPERT_GMM2);
  1217          ProcessSharedExpertGmm2();
  1218      }
  1219  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

### `shared_expert_gmm2_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1217` in `MegaMoeA8W8Wave::ProcessGmmPipeline`; `after` target line `1217`.
- Cardinality: 与 shared_expert_gmm2_begin_S 一一对应。
- Boundary claim: `issue` — AIV 角色已从共享专家 GMM2 helper 返回。
- Why: 区分共享专家调用开销与后续输出同步；共享结果等待可能继续落在 Unpermute 内。
- Dependency: ProcessSharedExpertGmm2 returned on AIV
- Pipeline proof: `Scalar control` → `Scalar` using `program order only` — no tracing-only completion event is inserted
- Perturbation: 每个 AIV 一次时钟读取，不增加 Cube/FIX drain。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1215      if (sharedExpertNum_ > 0) {
  1216          exceptionDump_.UpdateStage(MegaMoeImpl::Stage::SHARED_EXPERT_GMM2);
  1217          ProcessSharedExpertGmm2();
  1218      }
  1219  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

### `output_cross_rank_sync_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1485` in `MegaMoeA8W8Wave::Process`; `before` target line `1485`.
- Cardinality: 所有 AIV 各一次。
- Boundary claim: `causal` — 即将进入 Unpermute 前的输出跨 rank 同步。
- Why: 直接量化本 rank 等待其他 rank 完成 Combine 发送的时间。
- Dependency: local GMM pipeline returned
- Pipeline proof: `Scalar/network` → `Scalar` using `program order` — marker immediately precedes the output rank handshake
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1483          }
  1484  #endif
  1485          CrossRankSyncInWorldSize(params_.peermemInfo.rankSyncInWorldPtr, rankId_, worldSize_, aivJob_);
  1486  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1487          if (CriticalPathTracePtr() != nullptr) {
```

### `output_cross_rank_sync_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1485` in `MegaMoeA8W8Wave::Process`; `after` target line `1485`.
- Cardinality: 与 output_cross_rank_sync_begin_S 一一对应。
- Boundary claim: `actual` — 所有 rank 的输出握手已完成，本卡 Combine 接收区可读。
- Why: 把慢 rank 的 Combine 尾部与本地 Unpermute 分开。
- Dependency: CrossRankSyncInWorldSize returned
- Pipeline proof: `Scalar/network` → `Scalar` using `production rank handshake` — marker is immediately after the existing cross-rank synchronization
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1483          }
  1484  #endif
  1485          CrossRankSyncInWorldSize(params_.peermemInfo.rankSyncInWorldPtr, rankId_, worldSize_, aivJob_);
  1486  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1487          if (CriticalPathTracePtr() != nullptr) {
```

### `unpermute_init_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1499` in `MegaMoeA8W8Wave::Process`; `before` target line `1499`.
- Cardinality: 所有 AIV 各一次。
- Boundary claim: `causal` — 即将构造当前 AIV 的 Unpermute buffer 视图。
- Why: 把 UB view/config 构造与 Unpermute 数据流水分开。
- Dependency: output cross-rank sync returned
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker immediately precedes InitTokenUnpermuteBuffers
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1497          }
  1498  #endif
  1499          UnpermuteBufferConfig unpermuteBufferConfig = InitTokenUnpermuteBuffers();
  1500  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1501          if (CriticalPathTracePtr() != nullptr) {
```

### `unpermute_init_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1499` in `MegaMoeA8W8Wave::Process`; `after` target line `1499`.
- Cardinality: 与 unpermute_init_begin_S 一一对应。
- Boundary claim: `actual` — Unpermute UB view 和当前 job bufferConfig 已构造完成。
- Why: 识别极端配置下 buffer 规划本身是否异常。
- Dependency: InitTokenUnpermuteBuffers returned
- Pipeline proof: `Scalar` → `Scalar` using `program order` — helper only constructs local tensor views and returns synchronously
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1497          }
  1498  #endif
  1499          UnpermuteBufferConfig unpermuteBufferConfig = InitTokenUnpermuteBuffers();
  1500  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
  1501          if (CriticalPathTracePtr() != nullptr) {
```

### `unpermute_begin_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1508` in `MegaMoeA8W8Wave::Process`; `before` target line `1508`.
- Cardinality: 所有 AIV 各一次。
- Boundary claim: `causal` — Unpermute buffer 已构造，即将进入本核 token range。
- Why: 量化 Combine 接收数据、top-k/shared 累加与最终 y 写回的整体成本。
- Dependency: Unpermute buffer initialization completed
- Pipeline proof: `Scalar` → `Scalar` using `program order` — marker immediately precedes UnpermuteTokens
- Perturbation: 每个 AIV 一次时钟读取。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1506          }
  1507  #endif
  1508          UnpermuteTokens<CombineQuantMode, TopkWeightsType, TopkWeightsPrefetch, GMM1_TILE_M>(
  1509              tokenUnpermuteConfig_, commonConfig_, params_, tokenUnpermuteScratch_, unpermuteBufferConfig);
  1510  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

### `unpermute_end_S`

- Where: `mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h:1508` in `MegaMoeA8W8Wave::Process`; `after` target line `1508`.
- Cardinality: 与 unpermute_begin_S 一一对应。
- Boundary claim: `actual` — 本核负责 token 的最终 y MTE3 已全部完成。
- Why: 确认尾部慢点是否来自本地反量化/累加/写 y，而不是同步。
- Dependency: UnpermuteTokens returned
- Pipeline proof: `MTE2/Vector/MTE3` → `Scalar` using `production MTE3_V event wait` — UnpermuteTokens consumes its outputBufferEvent after the final batch before return
- Perturbation: 每个 AIV 一次时钟读取，复用 UnpermuteTokens 的生产 MTE3_V drain。
- Source SHA-256: `e87eedf9032f0f54fc9011dffbb516ebcf4f3aa4ac9f0db62595b62be1facf9a`

```text
  1506          }
  1507  #endif
  1508          UnpermuteTokens<CombineQuantMode, TopkWeightsType, TopkWeightsPrefetch, GMM1_TILE_M>(
  1509              tokenUnpermuteConfig_, commonConfig_, params_, tokenUnpermuteScratch_, unpermuteBufferConfig);
  1510  #if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0
```

## Working tree status

```text
M mc2/mega_moe/op_kernel/arch35/common/mega_moe_trace.h
 M mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h
 M mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm1_activation.h
 M mc2/mega_moe/op_kernel/arch35/stage/mega_moe_gmm2_combine.h
 M mc2/mega_moe/tools/critical_path_trace/README.md
 M mc2/mega_moe/tools/critical_path_trace/audit-implemented/trace_point_audit.json
 M mc2/mega_moe/tools/critical_path_trace/audit-implemented/trace_point_audit.md
 M mc2/mega_moe/tools/critical_path_trace/audit-plan/trace_point_audit.json
 M mc2/mega_moe/tools/critical_path_trace/audit-plan/trace_point_audit.md
 M mc2/mega_moe/tools/critical_path_trace/mega_moe_trace_contract.py
 M mc2/mega_moe/tools/critical_path_trace/run_critical_path_trace.py
 M mc2/mega_moe/tools/critical_path_trace/test_critical_path_trace.py
 M mc2/mega_moe/tools/critical_path_trace/trace-spec.json
 M mc2/mega_moe/tools/critical_path_trace/trace_tool.py
```
