# Moe EP Dispatch Hybrid 节点内通信模板实施方案

## 1. 目标范围

本阶段新增跨超模板 `moe_ep_dispatch_hybrid.h`，在判断为跨超场景时进入新模板。当前只实现和验证“目标 rank 都在本超节点内”的通信路径，跨超节点 Stage1 / Stage2 新通信逻辑暂不实现。

本阶段同时切换 dispatch 使用的 workspace / win 区布局。旧 direct 模板也需要适配新布局，但 direct 的行为和输出必须保持不变。

## 2. 设计约束

1. 原非跨超模板不能被大幅改写。`moe_ep_dispatch.h` 保留原模板主体和原 direct 流程。
2. `moe_ep_dispatch_base.h` 不是基类文件，只放无 UB 状态的可复用函数，例如地址计算、rank 关系、debug 写入和分片 `WriteNbi`。
3. 不把原模板整体搬到 `base.h`。凡是持有或调度 `TBuf` / `TQue` / `TPipe` 生命周期的逻辑，都保留在 direct 或 hybrid 模板内部。
4. 本阶段 workspace / win 区要切换到兼容 hybrid 的布局，direct 模板同步适配。
5. 变量命名不能使用单字母，统一使用有业务含义的命名，例如 `epWorldSize`、`rankSizePerServer`、`superNodeCount`。
6. `debug_out` 只用于开发期观测和 hook 打印，不进入默认 golden 比较。

## 3. 文件组织

新增文件：

```text
mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch_base.h
mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch_hybrid.h
```

保留文件：

```text
mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch.h
```

职责划分：

```text
moe_ep_dispatch_base.h
  公共常量
  slot 地址计算函数
  window 地址计算函数
  comm handle 获取函数
  rank / supernode 关系计算函数
  debug_out 写入函数
  remote window 分片写入函数

moe_ep_dispatch.h
  原 direct 模板
  保留 direct 主流程
  调用 base.h 中的公共函数适配新 workspace / win 布局

moe_ep_dispatch_hybrid.h
  新 hybrid 模板
  当前只实现本超节点内通信
  本超节点内发送复用 base.h 中的公共地址和拷贝辅助函数
  remote supernode 目标只记录 debug_out，暂不发送
```

## 4. 公共函数抽取方案

`moe_ep_dispatch_base.h` 使用 namespace + inline 函数，不使用继承，也不保存任何 UB 资源：

```cpp
namespace MoeEpDispatchBase {

__aicore__ inline GM_ADDR GetWindowAddrByRankId(...);
__aicore__ inline uint64_t GetScaleupSlotOffset(...);
__aicore__ inline GM_ADDR GetScaleupSendSlotAddr(...);
__aicore__ inline GM_ADDR GetScaleupFinalRecvSlotAddr(...);
__aicore__ inline uint64_t GetCommHandle(...);
__aicore__ inline bool IsRankInSameServer(...);
__aicore__ inline void WriteUint32ToDebug(...);
__aicore__ inline void WriteRemoteWindowByChunks(...);

}
```

优先抽取：

| 内容 | 是否抽取 | 原因 |
|---|---|---|
| align / offset 计算 | 否 | 由 tiling 统一产出 offset，kernel 侧只消费 tiling 字段 |
| window base 地址计算 | 是 | 两个模板完全一致 |
| final scaleup slot 地址计算 | 是 | 节点内路径和 epilogue 都依赖 direct-compatible final slot |
| comm handle 获取 | 是 | direct 和 hybrid 都需要 |
| rank 与 supernode 关系判断 | 是 | hybrid 必需，direct 可复用 |
| `WriteSlotToLocal` 的底层地址计算 | 是 | 复用地址部分，不强行复用完整流程 |
| 分片 `WriteNbi` + `Drain` 写 remote window | 是 | 底层发送动作可复用，rank 范围由调用方决定 |
| slot notify 写入 | 否 | notify 的等待范围与模板流程绑定，保留在各模板内 |
| `CalSendCnt` / `Communication` / `WaitDispatch` 主流程 | 否 | direct 当前按全局 `epWorldSize` 同步，hybrid 节点内路径不能原样复用 |
| `ProcessDirect` | 否 | 保留在原 direct 模板里 |
| route metadata 构造 | 否 | hybrid 专属 |
| scaleout payload 构造 | 否 | hybrid 专属 |

这样可以避免把原模板整体搬到 `base.h`，同时保证新旧模板对布局和地址计算的一致性。

## 5. 超节点内发送复用边界

原 direct 模板的发送逻辑可以复用，但只能复用底层动作，不能直接复用完整 direct 主流程。

### 5.1 可以复用的部分

```text
token slot 构造
topk 内按 targetRankId 去重
targetRankId 分配 scaleup slot
写 scaleup send workspace
写本 rank scaleup final recv data
按 CQE_MAX_WRITE_SIZE 分片 WriteNbi
Drain
写 slot notify
```

这些逻辑的语义在“目标 rank 位于本超节点内”时与 direct 一致，因此适合抽成公共函数。

建议抽取函数：

```cpp
namespace MoeEpDispatchBase {

__aicore__ inline uint64_t GetScaleupSlotOffset(
    uint32_t rankId,
    uint32_t slotIndex,
    uint32_t maxTokenCount,
    uint32_t scaleupSlotAlignedBytes);

__aicore__ inline GM_ADDR GetScaleupSendSlotAddr(
    GM_ADDR scaleupSendWorkspaceAddr,
    uint32_t targetRankId,
    uint32_t slotIndex,
    uint32_t maxTokenCount,
    uint32_t scaleupSlotAlignedBytes);

__aicore__ inline GM_ADDR GetScaleupFinalRecvSlotAddr(
    GM_ADDR scaleupFinalRecvBaseAddr,
    uint32_t sourceRankId,
    uint32_t slotIndex,
    uint32_t maxTokenCount,
    uint32_t scaleupSlotAlignedBytes);

__aicore__ inline void WriteRemoteWindowByChunks(
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> &hcomm,
    uint64_t commHandle,
    GM_ADDR remoteWindowAddr,
    GM_ADDR localWorkspaceAddr,
    uint64_t sendBytes);

}
```

### 5.2 不能原样复用的部分

以下 direct 主流程不能直接给 hybrid 节点内路径使用：

```text
CalSendCnt()
Communication()
GetRecvCount()
SendPhase()
WriteToRemoteWindow()
WaitDispatch()
```

原因是这些函数当前都隐含全局 `epWorldSize` 语义：

```text
Communication()
  遍历全 epWorldSize，给所有 rank 写 count 和 notify

GetRecvCount()
  等待全 epWorldSize 个 count 状态

WriteToRemoteWindow()
  遍历全 epWorldSize，向所有目标 rank 写 window 和 slot notify

WaitDispatch()
  等待全 epWorldSize 个 slot 状态
```

Hybrid 当前阶段只做超节点内通信。如果仍然等待全局 rank，会出现两类问题：

```text
不向 remote supernode rank 发 notify
  -> Wait 可能 hang

继续向 remote supernode rank 发 0 notify
  -> 能跑通但掩盖 hybrid 路径问题，不符合“只完成节点内通信”的阶段目标
```

因此 hybrid 需要独立主流程，只在主流程内部复用公共底层函数。

### 5.3 Hybrid 节点内主流程

Hybrid 初版建议新增：

```cpp
ProcessScaleup()
  ResetIntraServerCounters()
  CalSendCountHybridIntraServer()
  SyncAll()
  ExchangeCount()
  GetRecvCount()
  PipeBarrier()
  ResetPipe()
  SendPhase()
  SyncAll()
  WriteToRemoteWindowHybridIntraServer()
  WaitDispatch()
```

其中：

| hybrid 函数 | 处理方式 |
|---|---|
| `ResetIntraServerCounters` | 可复用 direct counter 清零逻辑，但清零范围要确认 |
| `CalSendCountHybridIntraServer` | 只统计本超节点目标，remote 目标计入 `debug_out.remoteRouteCount` |
| `ExchangeCount` | 只向 `[serverStartRank, serverEndRank)` 写 count notify |
| `GetRecvCount` | 只等待本超节点 source rank；`numRecvPerRank` 仍保持全局长度，非本超节点置 0 |
| `SendPhase` | 独立实现主循环，复用 slot 构造、去重、slot 写入 helper |
| `WriteToRemoteWindowHybridIntraServer` | 只遍历本超节点 rank，复用分片写 window helper |
| `WaitDispatch` | 只等待本超节点 source rank 的 slot notify |

## 6. Hybrid 模板接入

kernel 入口根据 `NetworkMode` 选择模板：

```cpp
if constexpr (NetworkMode == NETWORK_MODE_HYBRID) {
    MoeEpDispatchHybrid<...> dispatchOp;
    dispatchOp.Init(...);
    dispatchOp.Process();
} else {
    MoeEpDispatch<...> dispatchOp;
    dispatchOp.Init(...);
    dispatchOp.Process();
}
```

节点内通信基础量：

```text
epWorldSize
rankSizePerServer
superNodeCount = epWorldSize / rankSizePerServer

currentServerIndex = epRankId / rankSizePerServer
rankIndexInServer = epRankId % rankSizePerServer
serverStartRank = currentServerIndex * rankSizePerServer
serverEndRank = serverStartRank + rankSizePerServer
```

本阶段 hybrid 只支持：

```text
targetRankId >= serverStartRank
targetRankId <  serverEndRank
```

节点内目标走复用路径：

```text
topk expert -> targetRankId
targetRankId in current server
  -> 使用 scaleup final slot
  -> 写 direct-compatible final window
  -> recv_src_metadata 仍写 [src_rank, src_token_idx, src_topk_idx, scaleup_slot]
```

remote supernode 目标：

```text
targetRankId not in current server
  -> debug_out.remoteRouteCount += 1
  -> debug_out.errorFlags 标记 remote route unsupported
  -> 本阶段不发送
```

## 7. Workspace / Win 区前后对比

### 7.1 命名

```text
epWorldSize              : EP 全局 rank 数
rankSizePerServer        : 单超节点内 rank 数
superNodeCount           : 超节点数量，epWorldSize / rankSizePerServer
maxTokenCount            : 最大 token 数
scaleupSlotRawBytes      : 原 direct token slot raw bytes
scaleoutMaskWordCount    : ceil(rankSizePerServer / 32)
scaleoutSlotRawBytes     : scaleupSlotRawBytes
                           + scaleoutMaskWordCount * sizeof(uint32_t)
                           + rankSizePerServer * sizeof(int32_t)
scaleupSlotAlignedBytes  : align512(scaleupSlotRawBytes)
scaleoutSlotAlignedBytes : align512(scaleoutSlotRawBytes)
```

### 7.2 Workspace：旧布局与新布局

```text
旧 direct workspace

+-------------------------------+  workspaceBase
| counter                       |  align512(epWorldSize * sizeof(int32_t))
+-------------------------------+
| send count per rank           |  align512(2 * epWorldSize * sizeof(int32_t))
+-------------------------------+
| send count per expert         |  align512(localExpertCount * epWorldSize * sizeof(int32_t))
+-------------------------------+
| globalA / UB reserved         |  current tiling reserved bytes
+-------------------------------+
| direct slot stash             |  epWorldSize * maxTokenCount * scaleupSlotAlignedBytes
+-------------------------------+

新 dispatch workspace

+-------------------------------+  workspaceBase
| counter                       |  align512(epWorldSize * sizeof(int32_t))
+-------------------------------+
| send count per rank           |  align512(2 * epWorldSize * sizeof(int32_t))
+-------------------------------+
| send count per expert         |  align512(localExpertCount * epWorldSize * sizeof(int32_t))
+-------------------------------+
| globalA / UB reserved         |  current tiling reserved bytes
+-------------------------------+
| scaleup send workspace        |  epWorldSize * maxTokenCount * scaleupSlotAlignedBytes
+-------------------------------+
| scaleout send workspace       |  superNodeCount * maxTokenCount * scaleoutSlotAlignedBytes
+-------------------------------+
| route workspace               |  route entry / cached hybrid private metadata workspace
+-------------------------------+
```

说明：

1. `scaleup send workspace` 保留 `epWorldSize` 维度，direct 模板可以用它替代原 `direct slot stash`。
2. 本阶段 hybrid 节点内发送只使用 `scaleup send workspace`。
3. `scaleout send workspace` 和 `route workspace` 本阶段先完成布局和 offset 预留，remote 逻辑后续填充。
4. 旧 direct 模板需要把原 `slotWorkspaceAddr_` 指到 `scaleupSendWorkspaceOffset`。

### 7.3 Win 区：旧布局与新布局

```text
旧 win layout

+-----------------------------------+  winBase
| dispatch count state              |
+-----------------------------------+
| dispatch slot state               |
+-----------------------------------+
| combine state                     |
+-----------------------------------+
| dispatch final data               |  epWorldSize * maxTokenCount * scaleupSlotAlignedBytes
+-----------------------------------+
| combine data                      |
+-----------------------------------+

新 win layout

+-----------------------------------+  winBase
| dispatch count state              |
+-----------------------------------+
| dispatch slot state               |
+-----------------------------------+
| combine state                     |
+-----------------------------------+
| scaleout recv data                |  superNodeCount * maxTokenCount * scaleoutSlotAlignedBytes
+-----------------------------------+
| scaleup final recv data           |  epWorldSize * maxTokenCount * scaleupSlotAlignedBytes
+-----------------------------------+
| scaleout recv status              |  alignStatus(superNodeCount * maxTokenCount * 512)
+-----------------------------------+
| scaleup recv status               |  alignStatus(epWorldSize * maxTokenCount * 512)
+-----------------------------------+
| combine data                      |
+-----------------------------------+
```

说明：

1. direct 模板原来写 `dispatch final data`，新布局下改为写 `scaleup final recv data`。
2. epilogue 后续应读取 `scaleupFinalRecvDataOffset`，不能继续假设 old `winDataOffset`。
3. 为降低改动风险，可以在 tiling 里保留兼容字段 `winDataOffset = scaleupFinalRecvDataOffset`，让旧 epilogue 代码先少改。
4. `scaleout recv data` 和 `scaleout recv status` 本阶段先预留，节点内路径不消费。

### 7.4 Direct 模板适配点

direct 模板需要适配新布局，但不改变行为：

```text
原 slotWorkspaceAddr_
  -> 新 scaleupSendWorkspaceOffset

原 localWinAddr_
  -> 新 scaleupFinalRecvDataOffset

原 slot state offset
  -> 如果新布局保留 dispatch slot state 位置，则不变

原 combine data offset
  -> 需要跟随新 win layout 后移
```

direct 写入地址仍然按全局 rank 编址：

```text
scaleup slot offset =
    (sourceRankId * maxTokenCount + slotIndex) * scaleupSlotAlignedBytes
```

这样 direct 输出和 epilogue 读取结果保持一致。

## 8. Tiling 修改点

tiling 需要同时给 direct 和 hybrid 提供新布局字段：

```text
scaleupSendWorkspaceOffset
scaleoutSendWorkspaceOffset
routeWorkspaceOffset

scaleoutRecvDataOffset
scaleupFinalRecvDataOffset
scaleoutRecvStatusOffset
scaleupRecvStatusOffset
combineDataOffset

scaleupSlotAlignedBytes
scaleoutSlotAlignedBytes
scaleoutMaskWordCount
```

校验建议：

1. `epWorldSize > 0`。
2. `rankSizePerServer > 0`。
3. `epWorldSize % rankSizePerServer == 0`。
4. `superNodeCount >= 1`。
5. `scaleupSlotAlignedBytes` 和 `scaleoutSlotAlignedBytes` 均 512 对齐。
6. 新 workspace 总大小不能超过实际 workspace。
7. 新 win 总大小不能超过 context 提供的 window size。

## 9. debug_out 与测试 hook

当前 `debug_out` 为 `uint8`，默认 1M。建议使用固定 byte offset 编码，hook 负责打印和解码，不进入默认 golden 比较。

```text
debug_out[0]  = path marker，direct / hybrid
debug_out[1]  = networkMode
debug_out[2]  = superNodeCount low byte
debug_out[3]  = rankSizePerServer low byte
debug_out[4]  = currentServerIndex low byte

debug_out[8..11]   = remoteRouteCount，uint32 little-endian
debug_out[16..19]  = localSendCount，uint32 little-endian
debug_out[24..27]  = errorFlags，uint32 little-endian
```

本阶段关键观测：

```text
path marker == hybrid
remoteRouteCount == 0
errorFlags == 0
```

测试框架 hook：

1. 打印 `debug_out` 前若干字节。
2. 解码固定字段。
3. 打印非零区域。
4. 不把 hybrid route metadata 纳入默认比较。

## 10. 测试用例设计

### 10.1 新布局下 direct 回归

目标：证明 workspace / win 区切换后，旧 direct 模板行为不变。

测试：

1. 原有 dispatch / epilogue / combine 用例全跑。
2. cached / non-cached 都跑。
3. continue 多轮场景跑。
4. 比较最终 `recv_x`、`recv_topk_idx`、`recv_topk_weights`、`combine`。

通过标准：

```text
最终输出与旧 direct golden 一致
无新增同步错误
无新增越界或 window size 错误
```

### 10.2 强制 hybrid + 单超节点

临时代码版本：

```text
NetworkMode 强制走 hybrid
rankSizePerServer = epWorldSize
superNodeCount = 1
```

测试：

1. 跑原有所有非跨超 case。
2. 对比最终 dispatch / combine 输出。
3. hook 打印并解码 `debug_out`。

通过标准：

```text
path marker == hybrid
superNodeCount == 1
remoteRouteCount == 0
最终结果与 direct golden 一致
```

### 10.3 多超节点配置，本超节点内发送

构造测试数据时，所有 expert 映射到本 server rank 范围：

```text
targetRankId >= serverStartRank
targetRankId <  serverEndRank
```

测试重点：

1. hybrid 模板进入节点内复用路径。
2. `scaleup send workspace` 写入正确。
3. `scaleup final recv data` 写入正确。
4. epilogue 从 `scaleupFinalRecvDataOffset` 读取正确。
5. `recv_src_metadata` 保持 direct-compatible。

通过标准：

```text
remoteRouteCount == 0
errorFlags == 0
最终 dispatch / combine 输出正确
```

### 10.4 Remote 目标负向调试用例

构造至少一个 remote supernode 目标。

当前阶段不要求算子正确完成跨超通信，但必须暴露状态：

```text
remoteRouteCount > 0
errorFlags 标记 unsupported remote route
```

该用例作为 debug-only 或 expected-fail，不进入常规回归。

## 11. 分阶段执行计划

### Phase 1：抽公共函数，不抽基类

1. 新增 `moe_ep_dispatch_base.h`。
2. 只抽 layout、地址、rank 关系、comm handle 等可确认复用函数。
3. `moe_ep_dispatch.h` 保留原 direct 主流程。
4. 所有新增变量使用有意义命名。

### Phase 2：切换 workspace / win 新布局

1. tiling 新增 scaleup / scaleout workspace offset。
2. tiling 新增 scaleout recv / scaleup final recv offset。
3. direct 模板改用 `scaleupSendWorkspaceOffset` 和 `scaleupFinalRecvDataOffset`。
4. epilogue 读取 offset 对齐到 `scaleupFinalRecvDataOffset`。
5. 先跑 direct 全量回归。

### Phase 3：新增 hybrid 模板入口

1. 新增 `moe_ep_dispatch_hybrid.h`。
2. launcher 按 `NetworkMode` 选择 direct 或 hybrid。
3. hybrid 初版只实现 `ProcessHybridIntraServerOnly()`。
4. 单超节点强制 hybrid，验证入口和布局。

### Phase 4：实现本超节点内发送复用

1. hybrid 内计算 `serverStartRank` 和 `serverEndRank`。
2. 新增 `SendPhase()`，不直接调用 direct `SendPhase()`。
3. 新增 `WriteToRemoteWindowHybridIntraServer()`，只遍历本超节点 rank。
4. 新增 `WaitDispatch()`，只等待本超节点 source rank。
5. 本超节点内目标写 `scaleup send workspace` 和 `scaleup final recv data`。
6. remote 目标只写 `debug_out`，暂不通信。
7. 多超节点 local-only 数据验证。

### Phase 5：后续完整跨超预留

1. 实现 scaleout payload 构造。
2. 实现 route metadata 写入。
3. 实现 Stage1 / Stage2 跨超通信。
4. 实现 cached hybrid replay。
5. 测试框架补充带 topology 信息的 hybrid route golden 后，再比较 route metadata。
