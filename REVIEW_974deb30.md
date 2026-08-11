# Commit `974deb30` 审查报告

## 概要

| 字段 | 值 |
|------|-----|
| Commit | `974deb309c79827e1754dec03eafe027f3c3b333` |
| 标题 | feat: add hybrid dispatch scaleout payload flow |
| 作者 | zhong-zixin |
| 日期 | 2026-07-20 |
| 规模 | 3 文件, +792/-258 |
| 父提交 | `f62fe6b39` (generate hybrid dispatch route slots) |

## 改动概述

本 commit 在前一 commit（路由表项生成）基础上，实现 hybrid dispatch 的 **scaleout payload 发送/接收/fanout 完整流程**，包括跨服务器 payload 代理发送、proxy rank 转发、接收端 fanout 到本服务器内各 rank。同时将 `Init` 函数拆分为多个职责单一的子函数，将 `DedupAndSend`/`CountRouteSlotsForCurrentCore`/`SendPhase` 等函数的逻辑提取为更细粒度的可复用函数。

## 逐文件改动

### 1. `mc2/moe_ep_dispatch/op_host/op_tiling/moe_ep_dispatch_tiling.cpp` (+5/-1)

**workspace 布局扩展**：新增 `scaleoutNotifyBytes`（每超节点 8 字节 notify）和 `readyStatusBytes`（512 字节就绪标志），追加到 workspace 总大小。

### 2. `mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch_hybrid.h` (+763/-256)

#### 2.1 Init 拆分

将原 `Init` 拆分为 8 个子函数：

| 函数 | 职责 |
|------|------|
| `InitTilingFields` | 从 tilingData 读取基础字段 |
| `InitAlignmentFields` | 计算对齐相关字段（routeScaleupSlotBytes_ 等） |
| `InitGlobalTensors` | SetGlobalBuffer 绑定所有 GM tensor |
| `InitCopyParams` | 构造 DataCopyParams/DataCopyExtParams |
| `InitWorkspaceAddresses` | 计算 workspace 各区域地址 |
| `InitDebugInfo` | 写 debug_out 路径标记 |
| `InitReadyStatus` | 初始化就绪状态窗口 |
| `InitSendCountBuffers` | CalSendCnt 阶段 UB buffer 初始化 |

#### 2.2 CalSendCnt 阶段拆分

| 函数 | 职责 |
|------|------|
| `InitSendCountRange` | 按 rank/expert 分核计算范围 |
| `FlushSendCountToWorkspace` | 将 sendCnt 写回 GM workspace |
| `SendCountToRemoteRank` | 通过 hcomm 发送 count 到远端 rank |
| `CopyLocalCountToWindow` | 将本地 count 写入 win 区 |

#### 2.3 路由表生成阶段拆分

| 函数 | 职责 |
|------|------|
| `InitRouteSlotCounterTensors` | 初始化 slot counter UB tensor |
| `CountRouteSlotsForToken` | 单 token 的路由 slot 计数 |
| `AllocateScaleupSlotForExpert` | 为 expert 分配 scaleup slot（本地递增） |
| `UpdateScaleoutRouteForExpert` | 更新 scaleout 路由表项（mask + slot） |

#### 2.4 scaleout payload 发送流程（新增）

| 函数 | 职责 |
|------|------|
| `GetScaleoutSlotAddr` | 计算 scaleout workspace 中 slot 地址 |
| `GetScaleoutStatusAddr` | 计算 scaleout status 地址 |
| `GetCurrentScaleupIndex` | 本 rank 在服务器内的 index |
| `GetProxyRankId` | 获取目标 scaleout 的 proxy rank |
| `WriteScaleoutPayloadFromRoute` | 根据 route 表写 payload 到 scaleout workspace |
| `InitPostSendHcomm` | 发送后通信初始化 |
| `InitScaleupSendRange` | scaleup 发送范围初始化 |
| `WriteScaleupSlotsToRank` | 写 scaleup slot 到目标 rank |
| `SendScaleoutPayloadsToProxy` | 发送 scaleout payload 到各 proxy rank |
| `WriteScaleoutNotifyToWorkspace` | 写 scaleout notify（含 slot count 汇总） |
| `SendScaleoutPayloadToProxy` | 单个 scaleout payload 发送到 proxy |

#### 2.5 scaleout payload 接收/fanout 流程（新增）

| 函数 | 职责 |
|------|------|
| `FanoutReceivedScaleoutPayloads` | 接收端 fanout 入口（仅核 0） |
| `FanoutScaleoutPayload` | 单个 scaleout slot 的 fanout |
| `FanoutScaleoutPayloadToRank` | fanout 到单个 scaleup rank |
| `WriteFinalReadyStatusForRemoteSource` | 写最终就绪状态 |

#### 2.6 SendPhase 拆分

| 函数 | 职责 |
|------|------|
| `InitSendPhaseBuffers` | SendPhase UB buffer 初始化 |
| `InitRouteStateTensors` | route 表 UB tensor 切片 |
| `InitRouteStateForToken` | 每 token 的 route UB 初始化 |
| `PrepareTokenPayload` | 准备 token 数据（DataCopy x/topk/scales） |
| `ProcessCachedTokenRoutes` | cached 路径处理 |

#### 2.7 ProcessHybridDispatch 新流程

```
CalSendCnt → SyncAll → ExchangeCount → GetRecvCount → SetRecvNumPerExpert → SetRecvNumPerRank
→ PipeBarrier → Reset → SendPhase → SyncAll
→ InitPostSendHcomm → WriteToRemoteWindow
→ SendScaleoutPayloadsToProxy → FanoutReceivedScaleoutPayloads → WaitDispatch
```

#### 2.8 ExchangeCount/GetRecvCount 分核策略变更

`ExchangeCount` 从 `SplitToCore(rankSizePerServer_, ...)` 改为 `SplitToCore(epWorldSize_, ...)`，`GetRecvCount` 和 `SetRecvNumPerExpert` 同步改为 `epWorldSize_`。

### 3. `torch_extension/cann_ops_transformer/ops/elastic_buffer.py` (+34/-6)

- 新增 `rank_num_per_server` / `scaleout_rank_count` / `scaleout_mask_word_count` 计算
- 新增 `scaleout_per_slot_bytes`（含 payload + mask + slotIdx）
- `dispatch_buffer_size` 从单一 `scaleup_recv_data` 扩展为 4 段：`scaleout_recv_data + scaleup_recv_data + scaleout_recv_status + scaleup_recv_status`

---

## 功能问题

### 阻断

| # | 位置 | 问题 |
|---|------|------|
| A1 | hybrid.h:137/1236 `WriteScaleoutPayloadFromRoute` | **声明并定义但从未被调用**。该函数负责根据 route 表写 token payload 到 `scaleoutSendWorkspaceAddr_`，但全文件无调用点。`SendScaleoutPayloadToProxy` 从该 workspace 读数据发送，读到的是未初始化垃圾数据，scaleout 跨服务器 payload 流程完全失效。 |

### 逻辑 bug

| # | 位置 | 问题 |
|---|------|------|
| A2 | hybrid.h:894-922 `SetRecvNumPerRank` | `GetRecvCount` 已改为读 `epWorldSize_` 范围（offset 0 起），但 `SetRecvNumPerRank` 仍用 `serverStartRank_`/`serverEndRank_` 做偏移。多服务器场景（`serverStartRank_ != 0`）下 `numRecvPerRankGMTensor_` 写入值全部错位，下游 combine 读到错误的接收数量。`SetRecvNumPerExpert` 已同步修改，唯独 `SetRecvNumPerRank` 遗漏。 |

### 需确认

| # | 位置 | 问题 |
|---|------|------|
| A3 | hybrid.h:831 `GetRecvCount` | UB 需求从 `serverRankCount * 64` 扩大为 `epWorldSize_ * 64`，当 `epWorldSize_ > ~1600` 时可能溢出 `topkIdsBuf_`（约 100KB）。需确认 tiling 对 `epWorldSize_` 的上界约束。 |
| A4 | hybrid.h:1360-1377 `WriteScaleoutNotifyToWorkspace` | `SetValue`（S 管线）后用 `PipeBarrier<PIPE_MTE3>`（MTE3 管线），不保证 S 管线 SetValue 已完成。需确认 AscendC 中 `GlobalTensor::SetValue` 的管线归属和 `PipeBarrier` 的跨管线语义。 |
| A5 | hybrid.h:1208/1384 `GetScaleoutSlotAddr`/`SendScaleoutPayloadToProxy` | `scaleoutSlotCount` 汇总所有 AIV 的 counter，若超过 `axisMaxBS_` 会越界到下一个 scaleoutIndex 的区域。需确认 tiling 是否保证每 server 的 scaleout slot 总数 ≤ `axisMaxBS_`。 |

---

## 性能问题

| # | 位置 | 问题 | 严重度 |
|---|------|------|--------|
| B1 | hybrid.h:1365-1368 `WriteScaleoutNotifyToWorkspace` | 循环内 `aivNum_` 次 GM scalar 读汇总 counter，可用 `DataCopy` + `ReduceSum` 向量化 | 性能 |
| B2 | hybrid.h:1461-1490 `FanoutScaleoutPayload` | 循环内 `scaleoutMaskWordCount_ + rankSizePerServer_` 次 GM scalar 读 route metadata，可先 `DataCopy` 到 UB | 性能 |
| B3 | hybrid.h:1236-1267 `WriteScaleoutPayloadFromRoute` | 循环内大量 GM scalar 读+写 route 表（且当前为死代码，修复 A1 后需同步优化） | 性能 |
| B4 | hybrid.h:1540-1544 `FanoutReceivedScaleoutPayloads` | 忙等待中每次迭代 `DataCopyPad` + `SyncFunc`，轮询开销高 | 性能 |
| B5 | hybrid.h:1517 `FanoutReceivedScaleoutPayloads` | 仅核 0 执行，单点串行，其他核空闲 | 性能 |
| B6 | hybrid.h:1399/1509 `SendScaleoutPayloadToProxy`/`WriteFinalReadyStatusForRemoteSource` | 循环内每 rank `hcomm_.Drain`，串行化发送，可批量 `WriteNbi` 后统一 `Drain` | 性能 |
| B7 | hybrid.h:300/405 `scaleoutNotifyCnt_` | 声明并赋值但从未使用，死代码 | 非阻断 |
| B8 | hybrid.h:1216 `GetScaleoutStatusAddr` | 每 scaleoutIndex 占 `axisMaxBS_ * 512` 字节，实际只用 8 字节，空间浪费 | 非阻断 |

---

## 与前一 commit 的兼容性

| # | 结论 |
|---|------|
| C1 | `DedupAndSend`/`CountRouteSlotsForCurrentCore`/`GetSlotStartNum` 正确拆分保留，签名与调用点匹配 |
| C2 | `ExchangeCount` 分核策略从 `rankSizePerServer_` 改为 `epWorldSize_`，`SetRecvNumPerExpert` 已同步但 `SetRecvNumPerRank` 遗漏（见 A2） |
| C3 | `WriteToRemoteWindow` 仍用 `rankSizePerServer_` 分核（仅发本服务器 scaleup slot），设计正确 |
| C4 | route 表 UB→GM 写回逻辑（`WriteRouteTableToGM`）保留不变，GM offset 计算一致 |
| C5 | tiling.cpp 新增 workspace 区域与 C++ 端 `InitWorkspaceAddresses` 偏移计算一致 |
| C6 | Python 端 `rank_num_per_server` 默认为 `world_size`（当 debug 环境变量未设置时），需确认与 C++ tiling 端 `rankSizePerServer_` 一致性 |

---

## 优先级

| 优先级 | 编号 | 理由 |
|--------|------|------|
| **P0** | A1 | scaleout payload 流程完全失效（未调用 WriteScaleoutPayloadFromRoute） |
| **P0** | A2 | 多服务器场景 numRecvPerRank 错位 |
| P1 | A3/A4/A5 | 需确认潜在阻断 |
| P2 | B1-B6 | 性能优化 |
| P3 | B7/B8 | 死代码/资源浪费 |
