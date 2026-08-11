# Hybrid Dispatch 多轮死锁分析报告

## 1. 环境

- 代码分支：`hybrid-exp`，提交 `5d5fbdf8e`
- 算子：`moe_ep_dispatch` hybrid 模板（`mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch_hybrid.h`）
- 测试脚本：`/home/zerozill/.openclaw/projects/moe_train_test/moe_train_test.py`
- NPU 环境变量：`MOE_EP_DEBUG_RANK_NUM_PER_SERVER=2`（控制 hybrid 模板内超节点 rank 数）
- 硬件：单机双卡，`rank_per_dev=2`，`world_size=2`

## 2. 触发条件

| 条件 | 结果 |
|------|------|
| `RANK_NUM_PER_SERVER=2` + `is_performance=1` + `is_continue=1` | **卡死** |
| `RANK_NUM_PER_SERVER=1` | 不卡 |
| `FORCE_NETWORK_MODE=0`（走 direct 模板） | 不卡 |
| `is_performance=0`（单轮） | 不卡 |
| 单跑 dispatch（不接 epilogue）`is_performance=1` | 不卡 |
| `topK=1` | 不卡 |
| `topK=2` | 卡 |
| BF16 和 FP8 | 都卡 |
| 卡死轮次 | 不固定（第 16 轮只是偶尔），换随机种子轮次变化 |

## 3. 卡死现象

### 3.1 卡死位置

rank 0 AIV0 卡在 `GetRecvCount` 轮询循环，`sum=1, cmp=2`（只收到本卡 count，rank 1 的 count 永远不到）。

rank 1 每轮正常完成：count 交换、payload 发送、final ready 发布。

### 3.2 关键日志（第 16 轮，loop=16）

**rank 1（正常）**：
```
sendcount: rank=1 aiv=0 dstRank=0 notifyVal=12262131630081
sendcount done: rank=1 aiv=0 dstRank=0
recvcount poll: rank=1 sum=2 cmp=2 poll=0 r0=1 r1=1
sendscaleup: rank=1 aiv=0 dstRank=0
payload done: rank=1 aiv=0 dstRank=0 hasRemote=1
writeready remote: rank=1 aiv=0 src=1 dst=0 addr=19793356786176 srcAddr=... val=12262131630081
finalready drain begin: rank=1 aiv=0 dstRank=0
finalready drain done: rank=1 aiv=0 dstRank=0
```
rank 1 每轮重复上述序列，count 交换和 final ready 都正常完成。

**rank 0（卡死）**：
```
recvcount poll: rank=0 sum=1 cmp=2 poll=0 r0=1 r1=0
recvcount poll: rank=0 sum=1 cmp=2 poll=1 r0=1 r1=0
recvcount poll: rank=0 sum=1 cmp=2 poll=2 r0=1 r1=0
recvcount poll: rank=0 sum=1 cmp=2 poll=1000
recvcount poll: rank=0 sum=1 cmp=2 poll=2000
...（永久卡死）
```
rank 0 的 `r1=0` 表示 rank 1 的 count notify 永远没到 rank 0 的 count 状态区。

### 3.3 地址验证

```
rank 1 AIV0 remote 写到 rank 0: addr=19793356786176
rank 0 epilogue r1Addr: 19793356786176
```
地址匹配，dispatch 发往的目的地址和 epilogue 接收轮询的地址一致。

### 3.4 count 交换机制

**发送**（`SendCountToRemoteRank`）：
```cpp
hcomm_.WriteWithNotifyNbi(commHandle, remoteCountAddr, srcWorkspaceAddr,
                          moeExpertNumPerRank_ * sizeof(int32_t),
                          notifyAddr, notifyValue);
hcomm_.Drain(commHandle);
```
- `notifyAddr` = 目标 rank window 的 `cntWinStateOffset + srcRankId * WIN_ADDR_ALIGN`
- `notifyValue` = 低32位 state=1，高32位 count

**接收**（`GetRecvCount`，AIV0）：
```cpp
while (sumOfFlag != compareFlag) {
    DataCopy(recvCounterTensor, recvCounterGMTensor_, statusCopyParams_);
    ReduceSum(...);
    sumOfFlag = numRecvPerRankTensor_.GetValue(0);
}
// 清理 count 状态区
DataCopy(recvCounterGMTensor_, recvCounterTensor, clearStatusCopyParams_);
```

## 4. 代码路径

### 4.1 Hybrid `PrepareDispatchPayloads`
```
CalSendCnt()                    — 全核计算 count
SyncAll<true>()                 — 等 counter 写完
ExchangeCount()                 — 全核发 count（HCOMM WriteWithNotifyNbi + Drain）
SyncAll<true>()                 — 等本 Rank count 全部发出
if (aivId_ == 0U) {
    GetRecvCount()              — AIV0 轮询等所有 rank count 到达，然后清理
}
// AIV1.. 不等，直接进 PreparePayloads
PipeBarrier<PIPE_ALL>()
tpipe_->Reset()
PreparePayloads()               — AIV1.. 构建 payload stash + 发送描述
SyncAll<true>()                 — 等 AIV0 收完 count + AIV1.. 构建完
```

### 4.2 Hybrid `TransferDispatchPayloads`
```
SendScaleoutPayloadsToProxy()   — 跨超组发送（remoteServerCount=0 时空操作）
BuildFanoutDescriptors()        — Fanout 描述构建（remoteServerCount=0 时空操作）
SyncAll<true>()
SendScaleupPayloads()           — 节点内发送（AIV0 发本卡，AIV1 发非本卡）
SyncAll<true>()
ClearReceivedScaleoutStatuses() — 清理（remoteServerCount=0 时空操作）
```

### 4.3 Direct 对比（不卡）
```
CalSendCnt → SyncAll → Communication(ExchangeCount + GetRecvCount) → SendPhase → SyncAll → WriteToRemoteWindow
```
- `ExchangeCount` 和 `GetRecvCount` 之间**没有 SyncAll**
- `WriteToRemoteWindow` 发完 final ready 后**没有 SyncAll，没有 Drain**，直接 return

## 5. 迄今为止的尝试

### 5.1 FP8 slot 大小统一（提交 `19b605167`）
**问题**：`CheckWinSize` 中 `dispatchReservedPerSlotBytes` 用 `MAX_OUT_DTYPE_SIZE=2`（固定），不含 FP8 scales，而 kernel 用 `perSlotBytes`（含 scales），导致 window 越界。
**修复**：dispatch/epilogue 改用 `info.cfg.perSlotBytes`，combine 改用 `AlignUpWin(hiddenAlign + topKAlign*2 + UB_ALIGN)`。
**结果**：BF16 也卡，排除 FP8 因素。

### 5.2 Combine slot 大小误用（提交 `49ddbce81`）
**问题**：`19b605167` 把 combine 的 `dispatchReservedPerSlotBytes` 改成了 `info.cfg.perSlotBytes`，但 combine 的 `perSlotBytes` 是 combine 自己的返回数据 slot（BF16/FP16，无 scales/topk meta），不是 dispatch 的 payload slot，导致三方 window 布局不一致。
**修复**：combine 恢复 `AlignUpWin(hiddenAlign + topKAlign*2 + UB_ALIGN)`。
**结果**：仍然卡死。

### 5.3 `ReadGmByPassDCache` 替换轮询（提交 `ed008d823`，已回退）
**问题**：怀疑 `DataCopy`（MTE2）读 count 状态区时 DCache 返回旧值。
**修复**：轮询改用 `ReadGmByPassDCache`（绕过 DCache）逐 rank 读取。
**结果**：仍然卡死，排除 DCache 读问题。但该改动有编译问题（`localCntStateWinAddr_` 未声明），可能未真正生效。

### 5.4 DCache invalidate 清理后（已回退）
**问题**：`GetRecvCount` 清理 count 状态区后 DCache 残留零值。
**修复**：清理 `DataCopy` 后逐 rank `DataCacheCleanAndInvalid`。
**结果**：未测试，分析后认为根因是跨 rank 轮次不同步，不是 DCache。

### 5.5 Final ready 每次 Drain（提交 `5d5fbdf8e`）
**问题**：`WriteFinalReadyStatusToRank` 的 remote `WriteNbi` 后没有立即 Drain。
**修复**：`WriteNbi` 后加 `hcomm_.Drain(commHandle)`。
**结果**：仍然卡死。

### 5.6 Payload 每次 Drain（提交 `5d5fbdf8e`）
**问题**：`SendDescriptorRange` 的 `WriteNbi<false>` 批量发，最后统一 Commit+Drain。
**修复**：每条 `WriteNbi<false>` 后立即 `Commit + Drain`。
**结果**：仍然卡死。

### 5.7 打印加详细地址和值
**内容**：
- `sendcount` / `sendcount done`：count 发送前后
- `recvcount poll`：逐 rank 状态值 `r0/r1`
- `sendscaleup`：每个 AIV 负责的 dstRank
- `payload done`：payload 发送完成
- `writeready local/remote`：final ready 写入的 src/dst/addr/val
- `finalready drain begin/done`：Drain 前后
- `epilogue poll addr` / `epilogue wait`：epilogue 轮询地址和逐 rank 值

**发现**：
- 地址匹配，dispatch 写入地址 = epilogue 轮询地址
- rank 1 每轮正常，rank 0 卡在 `GetRecvCount`
- rank 1 的 `srcAddr` 在两个值之间交替（可能因 `tpipe_->Reset()` 后 workspace 地址变化）
- 卡死轮次不固定

## 6. 当前分析

### 6.1 跨 rank 轮次不同步

**核心问题**：rank 0 第 N+1 轮 dispatch 已经开始（`GetRecvCount` 等待），但 rank 1 还在第 N 轮 epilogue。rank 1 的 count 还没发，rank 0 永远等不到。

**为什么 direct 不卡**：direct 的 `ExchangeCount` 和 `GetRecvCount` 之间没有 `SyncAll`，且 `WriteToRemoteWindow` 后没有 `SyncAll`，kernel 直接返回。ACLNN runtime 通过 stream 同步保证 HCOMM 完成。

**为什么 `RANK_NUM_PER_SERVER=1` 不卡**：count 走 inter-server HCOMM channel，payload 走 scaleout，不争用同一通道。

**为什么 `topK=1` 不卡**：payload 量小，HCOMM 操作少，竞争窗口窄，不容易触发。topK=2 时窗口变大，更容易触发轮次不同步。

### 6.2 排除项

- **DCache 不一致**：`ReadGmByPassDCache` 也卡，排除
- **HCOMM notify 资源耗尽**：卡死轮次不固定，换种子变化，排除
- **FP8 slot 大小越界**：已修复，BF16 也卡，排除
- **workspace/counter 累积**：每轮清零，排除
- **地址错误**：日志确认地址匹配，排除
- **WriteNbi/Drain 配对**：每次 Drain 仍卡，排除

### 6.3 未尝试的方向

1. **跨 rank 轮尾确认**：dispatch kernel 返回前加跨 rank 屏障，确保所有 rank 的上一轮 epilogue 都完成后才开始下一轮 dispatch。不复用 count 状态，新增独立的 epoch/done 状态区。

2. **去掉 `ExchangeCount` 和 `GetRecvCount` 之间的 `SyncAll`**：回到 direct 的模式，让 `ExchangeCount` 和 `GetRecvCount` 连续执行。

3. **`GetRecvCount` 前加 `DataCacheCleanAndInvalid`**：在轮询开始前 invalidate count 状态区的 DCache，确保第一轮读从 GM 取。

4. **count 交换改用 direct 的方式**：把 `WriteWithNotifyNbi` 改成 `WriteNbi`（发 count 数据）+ `Drain` + `WriteNbi`（发 notify 值），和 direct 完全一致。

## 7. 关键代码位置

| 位置 | 文件 | 行号 | 说明 |
|------|------|------|------|
| `GetRecvCount` | `moe_ep_dispatch_hybrid.h` | 838 | AIV0 轮询等 count，卡死位置 |
| `SendCountToRemoteRank` | `moe_ep_dispatch_hybrid.h` | 758 | HCOMM 发 count + notify |
| `PrepareDispatchPayloads` | `moe_ep_dispatch_hybrid.h` | 1907 | count 交换 + payload 构建流程 |
| `TransferDispatchPayloads` | `moe_ep_dispatch_hybrid.h` | 1962 | payload 发送 + final ready |
| `SendPayloadsToScaleupRank` | `moe_ep_dispatch_hybrid.h` | 1612 | 发 payload + final ready |
| `WriteFinalReadyStatusToRank` | `moe_ep_dispatch_hybrid.h` | 1665 | 写 final ready 状态 |
| `WaitDispatch` (epilogue) | `moe_ep_dispatch_epilogue.h` | 295 | epilogue 等 final ready |
| direct `Communication` | `moe_ep_dispatch.h` | 478 | direct 的 count 交换（对比） |
| direct `ProcessDirect` | `moe_ep_dispatch.h` | 795 | direct 的完整流程（对比） |
