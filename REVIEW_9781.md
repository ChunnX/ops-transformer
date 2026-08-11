# PR #9781 检视报告

## PR 信息

| 项目 | 内容 |
|------|------|
| 标题 | [WIP] slightly optimize hybrid mode of dispatch |
| 作者 | zhong-zixin |
| 分支 | `hybrid-perf-clean` → `master` |
| 状态 | Draft / WIP |
| 标签 | `api-check-failed`, `ci-pipeline-passed`, `stat/needs-squash` |
| PR 描述 | 空（模板未填写） |

## 变更概述

本 PR 对 `mc2/moe_ep_dispatch` 和 `mc2/moe_ep_combine` 系列算子的 hybrid 模式进行性能优化和重构，涉及 12 个文件。主要变更分为三类：

1. **命名规范化** — `DIRECT_MODE`/`HYBRID_MODE` → `NETWORK_DIRECT`/`NETWORK_HYBRID`，`topoType` → `requestedNetworkMode`，`rankSizePerServer` → `rankNumPerServer`，`numScaleoutRanks` → `serverNum`，`numAivStage1/2` → `scaleoutAivNum/scaleupAivNum`
2. **Tiling 结构体重组** — 将 `MoeEpDispatchInfo` 中的扁平字段拆分为 `hybrid`、`window`、`workspace` 三个子结构体；`scalesBytes`/`perSlotBytes` 从 `MoeEpCommonTilingData` 移至 `MoeEpDispatchInfo`；combine 和 epilogue 同步精简
3. **Fanout 转发算法重写** — 移除两阶段 BuildFanoutSendEntries → SendFanoutEntries 架构，替换为直接轮询转发的 `SendFanoutPayloads`，消除中间 fanout entry/count GM workspace

---

## 问题清单

### P1-02: `perSlotQueue_` 双 buffer 流水收益有限

**严重程度：低（性能优化）**

**位置：** `moe_ep_dispatch_hybrid.h:1667` — `InitPreparePayloadsBuffers`

```cpp
// 旧代码
tpipe_->InitBuffer(perSlotQueue_, 1, perSlotBytes_);
// 新代码
tpipe_->InitBuffer(perSlotQueue_, BUFFER_NUM, perSlotBytes_);  // BUFFER_NUM = 2
```

**评估：**

按合法上限保守计算，Prepare 阶段 UB 峰值约 137.44 KiB，Transfer 阶段约 48.06 KiB，均低于 248 KiB 上限，不存在 UB 越界风险。

实际问题在于：当前 `CopyPayloadToLocal` 是单 slot 申请、搬运、释放的串行流程，双 buffer 的流水重叠收益不明显。若要发挥 double buffering 的优势，需要配合 `CopyPayloadToLocal` 的流水化改造（如提前为下一个 slot 申请 buffer），否则双 buffer 仅增加了 UB 占用而未带来对应的性能收益。

**建议：** 若短期无计划改造 `CopyPayloadToLocal` 流水，可暂保留双 buffer（为后续优化预留空间）；若希望立即获得收益，需同步调整搬运逻辑使其能重叠两个 slot 的处理。

---

## 正面改进

### P2-01: `rsvdCnt == 0` 守卫（Bug 修复）

**位置：** `moe_ep_dispatch_hybrid.h` — `CalSendCntPerRank` 和 `CalSendCntPerScaleout`

```cpp
GatherMask(tempTensorInt16, rankIdsTensor, maskTensorInt16, true, mask, {1, 1, 0, 0}, rsvdCnt);
+        if (rsvdCnt == 0U) {
+            continue;
+        }
```

**评估：** 当 GatherMask 结果为空（`rsvdCnt == 0`）时跳过后续 UB 读写操作，避免不必要的 `SyncFunc` 和 `SetValue`。正确的守卫。

---

### P2-02: 移除 `WriteRemoteWindowByChunks` 和 `hcomm.h` include（死代码清理）

**位置：** `moe_ep_dispatch_base.h`

**评估：**
- `WriteRemoteWindowByChunks` 在整个 codebase 中无调用方，确认为死代码
- `hcomm.h` include 从 `moe_ep_dispatch_base.h` 移除，但 `moe_ep_dispatch.h` 和 `moe_ep_dispatch_hybrid.h` 中仍保留各自的 `hcomm.h` include，`hcomm_` 类型仍可用
- `DISPATCH_CQE_MAX_WRITE_SIZE` 常量随 `WriteRemoteWindowByChunks` 一起移除，无其他引用

---

### P2-03: 非交互式轮询替代阻塞等待（性能改进）

**位置：** `moe_ep_dispatch_hybrid.h` — `SendFanoutPayloads`

**评估：** 旧的 `WaitScaleoutSlotCount`/`WaitScaleoutSlotReady` 是纯阻塞轮询，新代码改为 `TryGetScaleoutSlotCount`/`TryScaleoutSlotReady` 非阻塞尝试 + 多源轮转调度（`sourceStateTensor_` 记录每源进度）。当某源未就绪时可切换到其他源继续工作，减少 idle 等待。

`SOURCE_STATE_UNREAD` 的 int64 编码（高 32 位 nextSlot / 低 32 位 slotsLeft）设计正确，`GetSourceState`/`SetSourceState` 的位操作一致。

---

### P2-04: count 发送按通信目标分组（架构改进）

**位置：** `moe_ep_dispatch_hybrid.h` — count 发送相关逻辑

**评估：** 新代码将 count 发送分为两类：
1. **非通信目标 rank** — 由分片核发送
2. **通信目标 rank**（同 server 内 rank + 跨 server proxy rank）— 由后续负责发 payload 的核发送

当前实现仍保持"count 与对应 payload 使用相同 owner 核"的原则，确保每个通信通道只由一个核使用，避免多核竞争同一 HCCL 通道。

---

## 通用问题

### PR 描述为空

PR 标记为 WIP，描述体仅包含模板内容未填写。考虑到变更涉及核心算法重写（fanout 转发机制）和窗口布局计算变更，建议补充：
- 变更原因和目标
- FP8 场景的测试验证情况
- 对 `scaleoutReservedPerSlotBytes` stride 变更的设计说明
- 性能数据对比

### `api-check-failed` 标签

PR 带有 `api-check-failed` 标签，建议排查 API 检查失败原因并修复后再提交正式检视。

---

## 总结

| 类别 | 数量 | 明细 |
|------|------|------|
| 低风险 | 1 | P1-02: 双 buffer 流水收益有限 |
| 正面改进 | 4 | P2-01~P2-04 |

**建议：** 补充 PR 描述，评估 P1-02 双 buffer 是否需要配合 `CopyPayloadToLocal` 流水化改造，修复 `api-check-failed` 后推进正式合入。
