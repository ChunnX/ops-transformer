# PR #9163 检视报告

## PR 概要

| 项目 | 内容 |
|------|------|
| **标题** | [WIP] Add hybrid template for dispatch |
| **分支** | `hybrid-stage1-stable` → `master` |
| **作者** | zhong-zixin |
| **状态** | Draft (WIP), 标签: `api-check-failed`, `ci-pipeline-failed`, `stat/needs-squash` |
| **规模** | 21 文件, +3042/-389 行 |
| **PR Body** | 未填写（模板空） |

### 变更概述

两个提交：
1. **`b6d5024d0` feat(mc2): move moe ep send buffers to HCCL window** — 将 dispatch/combine 的 send buffer 从 workspace 迁移到 HCCL 通信窗口，重构窗口布局为 state + recv + send 三段式。
2. **`1fb668b67` feat: dispatch supports hybrid communication** — 为 moe_ep_dispatch 新增 hybrid 通信模式（scaleout 跨服务器 + scaleup 服务器内两阶段通信），新增 1878 行 `moe_ep_dispatch_hybrid.h`。同时为 combine/epilogue 添加 `topo_type`/`rank_num_per_server` 属性支持。

---

## 阻塞级问题 [阻塞]

无。

> **已澄清的误报（B1/B2）**：初版报告曾将 `routeMaskTableCopyParams_` / `routeSlotTableCopyParams_` 的 `srcStride=0, dstStride=0` 标记为阻塞级 stride 不匹配。经查证 Ascend C `DataCopyPad` 文档（`DataCopyExtParams` 表4），stride 单位取决于操作数逻辑位置：**UB 侧为 32B DataBlock，GM 侧为字节**。`PaddingMode::Normal`（默认）下 UB 侧每个 block 自动 padding 到 32B 对齐，`dstStride=0` 表示相邻 block 无额外间隔即"紧挨 32B 对齐 entry"。因此 `srcStride=0/dstStride=0` 正确表达了"GM 紧凑排列、UB 按 32B 对齐步进"的布局，两个方向（GM→UB cache 路径、UB→GM WriteRouteTableToGM）的 UB 布局一致，`WriteScaleoutPayloadFromRoute` 按 `routeEntryIndex * routeScaleupMaskEntryElements_` 读取正确。B1/B2 为误报，已撤回。

---

## 讨论级问题 [讨论] (4)

### D1: API 接口不向后兼容

三个算子的 API 签名均有破坏性变更：
- **dispatch**: 新增 5 个必选输出 + 5 个可选输入 + 2 个属性
- **combine**: 删除 `biasOptional0/1`（原校验必须为 nullptr，属清理），`topkWeightsOptional` → `topkWeights` 重命名
- **epilogue**: `recv_topk_weights`/`recv_scales` 从 OPTIONAL 改 REQUIRED，`expert_alignment` 属性替换为 `has_topk_weights`+`topo_type`+`rank_num_per_server`

**评估**: PR 同步更新了 torch_extension 绑定层，mc2 模块与调用方协同演进。但 PR body 未说明兼容性影响，建议补充。`combine` 的 `bias` 删除是合理的（原实现强制 nullptr）。

### D2: Direct/Hybrid 类代码大量重复

`MoeEpDispatch`（direct）和 `MoeEpDispatchHybrid` 两个独立类无继承关系，`SplitToCore`、`CalSendCntPerRank`、`CalSendCntPerExpert`、`WriteSlotToLocal` 等函数大量重复。常量定义两份独立。维护时一方修改另一方易遗漏。

### D4: MAX_EP_WORLD_SIZE 从 128 改为 1024

已验证 `HCCL_MAX_RANK_SIZE = 1024`（`mc2_moe_context.h:20`），数组不会越界。但 1024 卡的 window 大小需求极端（约 71GB），实际被 `maxWindowSize` 校验拦截。建议添加 `static_assert` 明确约束。

### D5: Python `get_moe_ep_ccl_buffer_size` 的 max 策略可能预留过多内存

遍历 `world_size` 所有因子作为 `rank_num_per_server` 候选取最大值（`elastic_buffer.py:404-408`）。因静态方法无法获知实际拓扑，保守策略合理，但可能导致预留超出实际需求的 HCCL buffer。建议文档说明此行为。

### D6: 文档未描述 HYBRID 模式公式

`ElasticBuffer.md` 只展示 DIRECT 模式公式，但代码实际返回 `max(direct, hybrid)`。用户按文档手算可能得到偏小结果。

---

## 建议级问题 [建议] (6)

| # | 文件 | 描述 |
|---|------|------|
| S1 | `hybrid.h:58` vs `dispatch.h:60` | `BITS_PER_BYTE` 类型不一致 (int32_t vs uint32_t)，当前表达式中结果一致，仅风格不统一 |
| S2 | `hybrid.h:1593-1598` | `PollBackOff` 纯 busy-wait 无 yield/pause 指令 |
| S4 | `elastic_buffer.py:321,345-362` | `_get_moe_ep_minimum_window_bytes` 内部计算的 `state_buffer_size`/`dispatch_per_slot_bytes` 等中间量在 `get_moe_ep_ccl_buffer_size` 中为 HYBRID 分支重复计算，可提取公共计算段 |
| S5 | `elastic_buffer.py:381-382` | `scaleout_recv_status_size` 含 512 乘数天然对齐，外层再 align 只增加形式一致性，无实际影响 |
| S6 | `elastic_buffer.cpp:757 vs 687` | `moeRankSizePerServer_` 初始值不一致 (1 vs 2)，属不同对象，Build() 会覆盖，正常流程不触发不一致 |
| S8 | PR Body | 描述/测试/文档更新等模板字段均未填写 |
| S9 | `hybrid.h:1839` | `tpipe_->Reset()` 对调用顺序有隐式依赖，已有注释（line 1836）但仍建议补充 UB 生命周期说明 |

---

## 亮点

1. **Hybrid 架构设计清晰** — `if constexpr` 编译期分发，proxy/fanout 两阶段通信，route 表 O(1) 查找
2. **Window 布局重构合理** — send buffer 移入 window 后数据流语义更清晰（本卡→recv window，远端→send window→HCOMM）
3. **Python/C++ 公式精确对齐** — DIRECT/HYBRID 双模式窗口大小计算逐项验证一致
4. **校验覆盖完整** — 新增参数在 shape/dtype/format 三维度均有校验，cached_route 的 all-or-none 逻辑正确
5. **公共函数提取** — `MoeEpDispatchBase` 命名空间消除 Direct 模式的内联分片写循环
6. **slot 前缀预聚合** — `GetSlotStartNum` 通过 `ReduceCounterPrefix` 避免循环内 `AtomicAdd`

---

## 总结

| 维度 | 评分 | 说明 |
|------|------|------|
| 架构设计 | 良好 | Hybrid 模式设计合理，编译期分发无运行时开销 |
| 代码质量 | 一般 | Direct/Hybrid 代码重复高（D2），PR body 未填写（S8） |
| 正确性 | 通过 | 无阻塞级正确性问题 |
| 兼容性 | 需确认 | API 签名破坏性变更（D1），需确保所有调用方同步更新，PR body 应说明影响范围 |
| 文档 | 不足 | HYBRID 公式未文档化（D6） |

**结论**: PR 处于 WIP 状态，CI 未通过。无阻塞级正确性问题。剩余建议主要为文档补充（D1/S8/D6）和代码重复治理（D2/S4），不阻塞合并。
