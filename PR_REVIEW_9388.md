# PR #9388 代码审查报告

| 项目 | 内容 |
|------|------|
| PR | [add hybrid template for dispatch training](https://gitcode.com/cann/ops-transformer/pull/9388) |
| 作者 | zhong-zixin |
| 分支 | `hybrid-exp` -> `master` |
| Commit 数 | 5（最新 `16a8cc93f`） |
| 变更规模 | 19 文件，+2935 / -253 行 |
| 审查日期 | 2026-08-03 |
| CI 状态 | `ci-pipeline-failed` / `api-check-failed` / `stat/needs-squash` |

> **审查基线**：merge-base `ea55db8bed` -> HEAD `16a8cc93f`。初版误用 `upstream/master` 作为 diff base，将 master 上 PR 之后合入的 12 个无关 commit 变更计入审查范围，本版已修正。

---

## 一、PR 概要

本 PR 为 MoE EP（Expert Parallelism）分发/合并通信链路引入 **hybrid 网络拓扑支持**，在原有 DIRECT（全互联）模式基础上新增 HYBRID 模式——节点内（scaleup）通过 HCCLS window 直传，跨节点（scaleout）通过 proxy rank 中转。变更贯穿 ACLNN API -> Host Tiling -> Kernel -> Torch Extension 全链路。

### Commit 列表

| # | SHA | 类型 | 标题 |
|---|-----|------|------|
| 1 | 506f8b25f | feat | dispatch supports hybrid communication |
| 2 | 53f434311 | fix | synchronize hybrid dispatch count reuse |
| 3 | 4b7db2ad5 | fix | adapt hybrid dispatch channel handle indexing |
| 4 | a70d47421 | perf | pipeline hybrid dispatch per scaleout slot |
| 5 | 16a8cc93f | refactor | streamline hybrid dispatch payload routing |

### 变更文件清单（19 个文件）

| 模块 | 文件 | 变更量 |
|------|------|--------|
| **A. dispatch hybrid** | `mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch_hybrid.h` | +2002（新增） |
| | `mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch_base.h` | +81/-56 |
| | `mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch.cpp` | +26/- |
| | `mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch_tiling.h` | +17 |
| | `mc2/moe_ep_dispatch/op_host/op_tiling/moe_ep_dispatch_tiling.cpp` | +327/- |
| | `mc2/moe_ep_dispatch/op_host/moe_ep_dispatch_def.cpp` | +31/- |
| | `mc2/moe_ep_dispatch/op_api/aclnn_moe_ep_dispatch.cpp` | +57/- |
| **B. dispatch epilogue** | `mc2/moe_ep_dispatch_epilogue/op_api/aclnn_moe_ep_dispatch_epilogue.cpp` | +43/- |
| | `mc2/moe_ep_dispatch_epilogue/op_host/moe_ep_dispatch_epilogue_def.cpp` | +8/- |
| | `mc2/moe_ep_dispatch_epilogue/op_host/op_tiling/moe_ep_dispatch_epilogue_tiling.cpp` | +102/- |
| | `mc2/moe_ep_dispatch_epilogue/op_kernel/moe_ep_dispatch_epilogue_tiling.h` | +3 |
| **C. combine** | `mc2/moe_ep_combine/op_api/aclnn_moe_ep_combine.cpp` | +42/- |
| | `mc2/moe_ep_combine/op_host/moe_ep_combine_def.cpp` | +16/- |
| | `mc2/moe_ep_combine/op_host/op_tiling/moe_ep_combine_tiling.cpp` | +98/- |
| | `mc2/moe_ep_combine/op_kernel/moe_ep_combine.cpp` | +9/- |
| | `mc2/moe_ep_combine/op_kernel/moe_ep_combine.h` | +27/- |
| | `mc2/moe_ep_combine/op_kernel/moe_ep_combine_tiling.h` | +4 |
| **D. elastic_buffer** | `torch_extension/cann_ops_transformer/ops/csrc/elastic_buffer.cpp` | +125/- |
| | `torch_extension/cann_ops_transformer/ops/elastic_buffer.py` | +170/- |

---

## 二、审查结论

### 合入建议：**可合入，建议处理以下问题**

无阻塞级问题。存在 1 个性能取舍项、2 个待改进项、6 个后续优化建议。

### 问题统计

| 严重程度 | 数量 |
|---------|------|
| 阻塞 | 0 |
| 性能取舍 | 1 |
| 待改进 | 2 |
| 后续建议 | 6 |

---

## 三、性能取舍

### 1. [moe_ep_dispatch_hybrid.h:1547] `SendEntries` 远程路径每条 entry 调用 `Drain`

`SendEntries` 中每次 `WriteWithNotifyNbi` 后立即 `hcomm_.Drain(commHandle)`，序列化所有 HCCS 写，无法利用 channel pipeline。

**但这是规避 URMA CQ 超时和偶现卡死的稳定性措施**。在底层免 Drain 问题解决前，不能简单改为循环末尾统一 Drain。代码中 TODO 注释也确认了这是已知限制。

> **状态**: 性能问题成立，现阶段稳定性取舍，暂不修改。

---

## 四、待改进项

### 1. [moe_ep_dispatch_hybrid.h:1986] `TransferDispatchPayloads` 中 `SyncAll` 阻断 scaleout→scaleup 流水

`TransferDispatchPayloads` 的执行流程：

```
SendScaleoutPayloadsToProxy()    // 跨节点发送
BuildFanoutSendEntries()         // 构建节点内 fanout 发送记录
    ↓
SyncAll<true>()                  // ← 等待全部 scaleout 发送 + fanout 记录构建完成
    ↓
SendScaleupPayloads()            // 节点内发送才开始
    ↓
SyncAll<true>()                  // 等待节点内发送完成
ClearReceivedScaleoutStatuses()
```

第一个 `SyncAll` 要求所有 AIV 完成 scaleout 发送和 fanout 记录构建后，节点内 payload 发送才能开始。这意味着 scaleout 通信和 scaleup 通信无法重叠，对于跨节点延迟较高的场景，节点内 AIV 会空闲等待。

> **建议**: 研究是否可以让部分 fanout 记录构建完成后即开始对应的 scaleup 发送，无需等待全部 scaleout 完成。需要确保 fanout 记录的 GM 可见性在 scaleup 读取前已建立。这属于后续性能优化方向，不影响正确性。

### 2. [三个 op definition] `rank_num_per_server` 默认值仍为 `1`，与 ElasticBuffer 内部默认值 `2` 不一致

| 位置 | 默认值 |
|------|--------|
| `moe_ep_dispatch_def.cpp:104` | `Int(1)` |
| `moe_ep_dispatch_epilogue_def.cpp:63` | `Int(1)` |
| `moe_ep_combine_def.cpp:65` | `Int(1)` |
| `elastic_buffer.cpp:748` `rankNumPerServer_` | `2` |
| `elastic_buffer.cpp:838` `moeRankSizePerServer_` | `2` |

ElasticBuffer 的 `EnsureMoeContext` 在 `Build` 后会用 `rankNumPerUbDomain_` 覆盖 `rankNumPerServer_`，实际运行时不会使用默认值，因此不影响功能正确性。但 op definition 侧的 attr 默认值 `1` 与 ElasticBuffer 侧默认值 `2` 不一致，容易误导排查。

> **建议**: 统一三个 op definition 的 `rank_num_per_server` attr 默认值为 `2`，或与 ElasticBuffer 侧对齐。

---

## 五、后续建议项

| # | 文件:行号 | 描述 | 说明 |
|---|----------|------|------|
| 1 | `moe_ep_dispatch_hybrid.h:1691-1710` | busy-wait 轮询退避周期 | `WaitScaleoutSlotCount`/`WaitScaleoutSlotReady` 使用 100 cycle 退避 busy-wait 轮询 GM，可研究动态调整退避周期。 |
| 2 | `moe_ep_dispatch_hybrid.h:1290-1306` | `SendScaleoutPayloadShard` 逐 entry SyncFunc | 每个 entry 执行 S_MTE2 + MTE2_S 两次同步，shard 内 entry 多时开销显著。可批量读取 send entry 到 UB 后统一处理。 |
| 3 | `moe_ep_dispatch_hybrid.h:1926` | `PipeBarrier<PIPE_ALL>()` 可研究缩小 | 保守同步可能损失性能。分片间依赖主要是 MTE3 后的 V/S 复用，可以研究缩小；但末尾还涉及 GM 发布、`SyncAll` 和后续 `TPipe::Reset`，不能把所有位置统一换成两个 `SyncFunc`。非正确性问题。 |
| 4 | `moe_ep_combine_tiling.h:40` | `scaleoutSlotAlignedBytes` 被 tiling 计算但 kernel 未使用 | combine kernel 透明使用最终 offset，不直接引用这些字段。如无后续计划可移除以减少 tiling data 体积。 |
| 5 | `moe_ep_combine_tiling.cpp:43` | `MAX_EP_WORLD_SIZE` 从 128 提升到 1024 | 仅影响验证上限，无数组以该常量为尺寸。1024 rank 场景下 UB 中的 `sendRankSlotTensor_`（按 `epWorldSize_` 分配）需 4KB UB，需确认大规模部署时 UB 容量充足。 |
| 6 | `elastic_buffer.cpp:1232` | DIRECT 模式下 route tensor 仍被分配 | `routeDstScaleout` 和 `routeScaleoutSlot` 各 `numTokens * topK * sizeof(int32_t)` 字节在 DIRECT 模式下全为 -1 且不被消费。不影响正确性，大 token 场景下浪费显存。可考虑在 DIRECT 模式下分配 `{1, 1}` 占位 tensor。 |

---

## 六、已确认无问题的项（初版误报，已排除）

以下项目在初版报告中列为问题，经核对不构成功能或正确性问题：

| 初版编号 | 描述 | 排除原因 |
|---------|------|---------|
| - | Combine FP8 window sizing 遗漏 scales | Combine 和 Python sizing 使用 `hidden * 2`（`max_out_dtype_size = 2`）预留 dispatch slot。FP8 实际为 `hidden * 1 + scales`，合法 scales 规格下不超过这份 2 字节预留。全部支持范围 `hidden=1..8192`、`topk=1..32`、float32/MX scales 均无溢出。 |
| - | Python sizing 未考虑 FP8 scales | 同上，`max_out_dtype_size = 2` 保守覆盖，无需新增 dtype 参数。 |
| - | Host counter 搬出 32B 是未初始化数据错误 | AscendC `DataCopy` 按 32B datablock 搬运。`HostPinnedCounter` 分配 4 个 `int64_t`（32B），只有第一个元素被轮询，尾部不参与功能。DIRECT 原模板也采用相同写法。 |
| - | DIRECT 路径不写 `route_*` 输出 | 统一从 `elastic_buffer.py` 调用，ACLNN 接口不对外开放，Python 中 DIRECT 模式不使用这些参数。不构成实际问题。 |
| - | Python sizing 逻辑重复 | 已提取 `_MoeEpWindowLayout`、`_get_moe_ep_window_layout`、`_get_moe_ep_direct_window_bytes`。**已修复**。 |
| - | 残留 `MOE_EP_DEBUG_FORCE_NETWORK_MODE` | 当前 dispatch、epilogue、combine 和 ElasticBuffer 中均无 `MOE_EP_DEBUG_*`、`getenv`、`PRINTF` 等调试代码。**已移除**。 |
| - | 静态 sizing 遍历所有合法 `rank_num_per_server` | 问题描述成立但不是正确性问题。sizing 时 context 尚未创建，拿不到实际拓扑；接口不能新增拓扑参数，只能保守取最大布局。`world_size <= 1024`，计算开销很小，主要代价是可能多分配 window 内存。**已知设计取舍**。 |
| - | `publishedStatus - 1U` uint32 下溢 | 正常协议下 `ClearReceivedScaleoutStatuses` 前已通过 `WaitScaleoutSlotCount` 等到状态非零；零 payload 也会发布 `1`。远端异常时会先卡在等待阶段，不会执行到减一。如需防御应超时报错或断言，而非 `continue`（会掩盖协议错误）。**不构成功能问题**。 |
| - | 单 server 降级日志 | 降级逻辑正确，`networkMode` 已有 `OP_LOGD` 输出。**可观测性建议，非功能问题**。 |
| - | `moeRankSizePerServer_` 默认值不一致 | 两个默认值都不会参与实际计算，`Build` 后用 `rankNumPerUbDomain_` 覆盖。**不影响功能**（但 op definition 侧 attr 默认值仍建议统一，见待改进 #2）。 |

---

## 七、模块详细审查

### 模块 A: moe_ep_dispatch hybrid

**变更规模**: 新增 `moe_ep_dispatch_hybrid.h`（2002 行模板头文件），`moe_ep_dispatch_tiling.cpp` +327 行，`moe_ep_dispatch_base.h` +81 行，`aclnn_moe_ep_dispatch.cpp` +57 行，其余文件少量适配。总计 +2510/-56 行。

**核心逻辑**: Hybrid 模式实现两级通信拓扑——节点内（scaleup）通过 HCCLS window 直传，跨节点（scaleout）通过 proxy rank 中转。

流程: `CalSendCnt`（多核并行统计 per-rank/per-expert/per-scaleout token count）-> `ExchangeCount`（HCCLS 通知远端 rank）-> `GetRecvCount`（AIV0 轮询接收）-> `PreparePayloads`（构建 payload stash + send entry + route table）-> `TransferDispatchPayloads`（scaleout 发送到 proxy + fanout entry 构建 + scaleup 发送 + status 清理）

**架构亮点**:
- **2D 分片设计**: count shard x token range 的二维切分，AIV0 专责接收 count，其余 AIV 接管 payload 构建，避免接收与构建的流水线冲突
- **O(1) 路由查找**: `sendRouteIndexByScaleoutTensor_` 直接索引 scaleout->route entry 映射
- **UB 复用规划周密**: `InitRouteStateTensors` 将 `numRecvPerRankBuf_` 精确切片为 slot 前缀、rank 去重、route 状态等多个生命周期不重叠的 tensor
- **跨核可见性处理**: counter 使用 `AtomicAdd` + `PipeBarrier<PIPE_MTE3>` 写入，`ReadGmByPassDCache` 读取，配合 `SyncAll<true>` 确保多核数据一致
- **Status notification 协议清晰**: slot 0 携带总 count+1，其余 slot 携带 `SCALEOUT_SLOT_READY`，producer-consumer 模式严谨
- **Tiling 校验全面**: 新增 `cached_route_*` 输入和 `route_*` 输出的 shape/dtype/format 校验完整，`topo_type` 和 `rank_num_per_server` 属性校验（整除关系、正数）到位
- **DIRECT/HYBRID 路径隔离干净**: `if constexpr (NetworkMode == TILINGKEY_HYBRID)` 分支编译期隔离

**整体评价**: 架构设计成熟，流水同步（MTE2/V/S/MTE3 间的 SyncFunc 配对）在 2000 行代码中基本无遗漏，UB buffer 复用生命周期管理清晰。`SendEntries` per-entry Drain 是已知稳定性取舍。`TransferDispatchPayloads` 中 scaleout→scaleup 的 `SyncAll` 是可后续优化的流水瓶颈。`MOE_EP_DEBUG_FORCE_NETWORK_MODE` 已移除。

---

### 模块 B: moe_ep_dispatch_epilogue

**变更规模**: +156/- 行

**核心逻辑**: epilogue tiling 新增 HYBRID 模式窗口布局分支，窗口分为 state -> scaleout recv data -> scaleup final recv data -> scaleout recv status -> combine data -> payload stash 六段。DIRECT 模式保持原有四段布局。

**整体评价**: 窗口 offset 链式推导可追溯，HYBRID/DIRECT 路径分离良好。perSlotBytes 计算含 scales，FP8 场景覆盖正确。

---

### 模块 C: moe_ep_combine

**变更规模**: +196/- 行

**核心逻辑**: combine 新增 `topoType` + `rankNumPerServer` 参数支持 hybrid 网络模式，窗口布局按 DIRECT/HYBRID 分路计算。移除从未使用的 `bias0`/`bias1` 参数和硬编码为 1 的 `expertAlignment`。

**亮点**:
- kernel 新增 `src_token_idx` 和 `src_topK_idx` 边界检查（`moe_ep_combine.h:437-439`），防止越界访问 `topkWeightsGm_` 和 `xGm_`，是重要安全改进
- API 精简：移除 bias 参数降低接口复杂度
- 向后兼容的 attr 默认值：`topo_type=0`(DIRECT)、`rank_num_per_server=1`、`has_topk_weights=true`，确保旧调用方不受影响

**整体评价**: 功能逻辑正确，hybrid 窗口布局设计合理。perSlotBytes 使用 `hidden * 2` 保守覆盖 FP8 scales，无 sizing 遗漏。

---

### 模块 D: elastic_buffer

**变更规模**: +295/- 行

**核心逻辑**: Python/C++ 接口同步新增 route count/dst_scaleout/scaleout_slot 三组 tensor，贯穿 Python->C++->ACLNN 调用链。C++ extension 对 optional output 创建 dummy tensor（`at::empty({1}, ...)`）而非传空，与 REQUIRED 输出变更一致。`cachedRouteCount/DstScaleout/ScaleoutSlot` 的 all-or-none 校验和 hybrid+cached 交叉校验逻辑严谨。

Python sizing 已提取 `_MoeEpWindowLayout`、`_get_moe_ep_window_layout`、`_get_moe_ep_direct_window_bytes` 共享 helper，消除了计算逻辑重复。`max_out_dtype_size = 2` 保守覆盖 FP8 scales。

**整体评价**: Python/C++ buffer 公式与 C++ tiling 的 HYBRID 窗口布局结构一致。静态 sizing 遍历所有合法 `rank_num_per_server` 取 max 是已知设计取舍（sizing 时 context 尚未创建）。

---

## 八、PR 元信息问题

| 项目 | 问题 |
|------|------|
| PR 描述 | body 为空模板，描述/关联Issue/测试/文档更新均未填写 |
| 类型标签 | 未勾选任何类型 |
| CI 状态 | `ci-pipeline-failed` + `api-check-failed` |
| Squash | 标记 `stat/needs-squash`，5 个 commit 需 squash |
| Commit 消息 | commit 2-4 无 body 说明，仅 commit 5 有简要说明 |

> **建议**: 补充 PR 描述（改动原因、方法、测试验证），勾选类型标签（新特性+性能优化+重构），修复 CI 失败，squash commit 后重新提交。

---

## 九、审查方法说明

### 初版修正记录

| 版本 | diff base | 文件数 | 问题数 | 说明 |
|------|-----------|--------|--------|------|
| v1 | `upstream/master`（错误） | 62 | 49（含 9 阻塞） | 误含 master 上 PR 之后合入的 12 个无关 commit |
| v2 | `ea55db8bed`（正确） | 19 | 17（含 8 需修复） | 修正 base，但含多项判断错误 |
| v3（本版） | `ea55db8bed`（正确） | 19 | 9（0 阻塞） | 核对用户反馈，排除误报，补充遗漏 |

### 本版核对变更

- **排除误报 9 项**: FP8 scales 遗漏（2 项，`hidden*2` 覆盖）、Host counter 32B（AscendC datablock 对齐）、DIRECT route_* 输出（ACLNN 不对外开放）、uint32 下溢（协议保证）、单 server 降级日志（非功能）、默认值不一致（不参与计算）、单 server 降级日志
- **标记已修复 2 项**: Python sizing 逻辑重复（已提取 helper）、`MOE_EP_DEBUG_FORCE_NETWORK_MODE`（已移除）
- **标记设计取舍 1 项**: 静态 sizing 遍历所有 `rank_num_per_server`
- **保留性能取舍 1 项**: `SendEntries` per-entry Drain（稳定性措施）
- **新增遗漏 1 项**: `TransferDispatchPayloads` SyncAll 阻断 scaleout→scaleup 流水
- **新增待改进 1 项**: op definition `rank_num_per_server` 默认值仍为 1
- **更新 commit**: `07cbbc6af` -> `16a8cc93f`

### 审查方法

由子 agent 分模块深度审查后汇总，核对基于 `git diff ea55db8bed..16a8cc93f` 逐文件分析。审查重点：内存安全、同步正确性、数据竞争、逻辑正确性、Tiling 一致性、API 参数校验、Python/C++ 接口一致性、性能、代码风格。
