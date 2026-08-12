# MoeEpCombine 批量 WriteWithNotify、无 CQE、无 Drain 改造说明

## 1. 文档目的

本文说明 `mc2/moe_ep_combine/op_kernel/moe_ep_combine.h` 中远端发送路径的批量化改造，重点回答以下问题：

- 修改前每个 token 如何发送，性能瓶颈在哪里；
- 修改后如何使用 `WriteWithNotifyNbiPrepare` 和 `WriteWithNotifyNbiPrepareEnd` 组织批次；
- 为什么新版本不产生 CQE，也不再调用 `Drain`；
- channel 切换、批缓冲区满和队列预算不足时如何处理；
- 主线新增的 flag/top-k weight notify 语义如何在冲突解决后得到保留；
- 改造后的正确性前提、资源开销和风险边界。

本次实现参考：

- [asc-comm PR 75：AscendC 批量处理接口实现](https://gitcode.com/cann/asc-comm/pull/75)
- [批量接口使用示例](https://gitcode.com/wuzhf9/hccl_test/blob/master/op_kernel.cpp)

对应开发分支为 `moe-ep-combine-batch-no-cqe`。本文以该分支 rebase 到主线后的实现为准。

## 2. 改造范围

本次只修改 kernel 头文件：

```text
mc2/moe_ep_combine/op_kernel/moe_ep_combine.h
```

没有修改 host tiling、workspace 计算、算子接口或接收阶段的数据布局。改造集中在 `MoeEpCombine::SendSlot()` 所使用的远端 Hcomm 发送机制。

本地 rank 的 DataCopy 路径保持不变。

## 3. 修改前的发送逻辑

### 3.1 远端 token 的地址和 notify

每个待发送 slot 根据源 token 位置计算远端数据和状态地址：

```cpp
sendTokenOffset = (srcTokenIdx * topK_ + srcTopKIdx) * perSlotBytes_;
recvStateOffset = (srcTokenIdx * topK_ + srcTopKIdx) * WIN_ADDR_ALIGN;
```

因此：

```text
远端 payload 地址 = rank 数据窗口 + combineDataWinOffset_ + sendTokenOffset
远端 notify 地址  = rank 状态窗口 + combineStateWinOffset_ + recvStateOffset
```

主线代码使用一个 64 位 `notifyValue` 同时携带完成 flag 和可选的 top-k weight：

```text
bit 63 ... bit 32: 完成 flag，值为 1
bit 31 ... bit  0: top-k weight 的 float32 原始比特；无 weight 时为 0
```

构造方式为：

```cpp
uint64_t notifyValue = 1ULL << 32;
notifyValue |= static_cast<uint64_t>(weightBits);
```

### 3.2 每个 token 独立提交并等待

修改前，远端路径对每个 token 执行一次：

```cpp
hcomm_.WriteWithNotifyNbi<true, PIPE_S, PIPE_MTE3, DEFAULT_WQE_CONFIG>(
    commHandle, remotePayloadAddr, localTokenAddr, payloadBytes,
    remoteNotifyAddr, notifyValue);
hcomm_.Drain(commHandle);
```

其时序为：

```text
token 0: 构造并提交 WriteWithNotify -> Drain 等待完成
token 1: 构造并提交 WriteWithNotify -> Drain 等待完成
token 2: 构造并提交 WriteWithNotify -> Drain 等待完成
...
```

原配置为 `.cqe = 1`，每个请求产生 CQE。`Drain` 会等待对应 channel 上已提交的操作完成，所以每个 token 都引入一次提交、完成事件和等待开销，也阻止同一 channel 上的多个 WQE 聚合提交。

## 4. 修改后的整体逻辑

修改后，单个远端 token 不再立即提交和等待，而是先把 WriteWithNotify 任务编码到 UB 批缓冲区：

```text
token 0 -> WriteWithNotifyNbiPrepare
token 1 -> WriteWithNotifyNbiPrepare
token 2 -> WriteWithNotifyNbiPrepare
...
批次结束 -> WriteWithNotifyNbiPrepareEnd<true>
```

新的主流程为：

```text
SendPhaseExpertToToken
  |
  +-- 遍历 metadata，筛选当前 AIV 负责的 token
  |
  +-- SendSlot
  |     |
  |     +-- 本地 rank：维持原 DataCopy + 状态写入逻辑
  |     |
  |     +-- 远端 rank：构造 notifyValue
  |              |
  |              +-- PrepareWriteWithNotify
  |                       |
  |                       +-- channel 变化：先 Flush
  |                       +-- Prepare 成功：计入当前批次
  |                       +-- buffer/queue 满：Flush 后重试
  |
  +-- 全部 token 扫描结束：FlushPreparedWrites
```

## 5. 批缓冲区设计

### 5.1 UB 大小

新增常量：

```cpp
HCOMM_BATCH_HEADER_BYTES = 32
HCOMM_BATCH_CAPACITY = 128
HCOMM_WRITE_NOTIFY_TASK_BYTES = 128
```

批缓冲区大小为：

```text
32 + 128 * 128 = 16416 字节
```

其中：

- 前 32 字节由 Hcomm 批量接口保存 batch header；
- 后续每 128 字节保存一个 WriteWithNotify prepare task；
- 理论上一个 UB batch 最多容纳 128 个远端 token 的 WriteWithNotify 任务；
- 实际单批数量还会受到 SQ 队列预算限制，可能小于 128。

### 5.2 初始化

`Init()` 中新增一个 `TPosition::VECOUT` 缓冲区：

```cpp
tpipe_->InitBuffer(hcommBatchBuf_, HCOMM_BATCH_BUFFER_BYTES);
hcommBatchTensor_ = hcommBatchBuf_.Get<uint8_t>();
Duplicate<uint8_t>(hcommBatchTensor_, 0, HCOMM_BATCH_HEADER_BYTES);
SyncFunc<AscendC::HardEvent::V_S>();
```

只需要清零 32 字节 header。任务区会在 Prepare 时覆盖，不要求预先全部清零。`V_S` 同步保证 Scalar 侧调用 Hcomm Prepare 前已经看到初始化后的 header。

## 6. 无 CQE 配置

原配置：

```cpp
static constexpr UrmaWqeEntry DEFAULT_WQE_CONFIG = {
    .odr = 5,
    .fence = 1,
    .se = 0,
    .cqe = 1,
    .inlineEn = 0,
};
```

修改后：

```cpp
static constexpr UrmaWqeEntry NO_CQE_WQE_CONFIG = {
    .odr = 5,
    .fence = 1,
    .se = 0,
    .cqe = 0,
    .inlineEn = 0,
};
```

只有 `cqe` 从 1 改为 0，其余顺序和 fence 配置保持不变。

`.cqe = 0` 表示这些 WQE 不请求生成完成队列项。这样可以避免 CQE 产生及 CQ 轮询开销，但发送侧也不能再通过 CQE 判断每个 WQE 是否已经远端完成。

本算子的完成可见性依赖 WriteWithNotify 的 notify 写入：接收端只有看到对应状态 slot 的完成 flag 后，才消费该 slot 的 payload。

## 7. PrepareWriteWithNotify 状态机

新增成员状态：

```cpp
uint64_t activeBatchHandle_{0};
uint32_t preparedWriteCount_{0};
```

含义如下：

- `activeBatchHandle_`：当前 UB batch 绑定的 channel handle；
- `preparedWriteCount_`：当前 batch 已成功 Prepare 的任务数；
- `preparedWriteCount_ == 0` 表示当前没有活动批次。

### 7.1 channel 变化

同一个 Hcomm batch 只能属于一个 channel。如果新任务的 `commHandle` 与活动批次不同：

```cpp
if (preparedWriteCount_ != 0 && activeBatchHandle_ != commHandle) {
    FlushPreparedWrites();
}
```

当前批次会先提交，然后新任务再进入新的 batch。这样不会把不同 channel 的 WQE 混入同一个 batch header。

### 7.2 Prepare

远端 token 使用：

```cpp
hcomm_.WriteWithNotifyNbiPrepare<NO_CQE_WQE_CONFIG>(
    commHandle,
    dst,
    src,
    len,
    notifyAddr,
    notifyValue,
    batchBuffer,
    HCOMM_BATCH_BUFFER_BYTES);
```

Prepare 只是在 UB 中构造任务描述，并不等价于远端完成。成功后记录：

```cpp
activeBatchHandle_ = commHandle;
++preparedWriteCount_;
```

### 7.3 buffer 或 queue 满

以下两个返回值属于可恢复的批次边界：

```text
HCOMM_PREPARE_BUFFER_FULL
HCOMM_PREPARE_QUEUE_FULL
```

处理方式是：

1. 断言当前批次非空；
2. 调用 `FlushPreparedWrites()` 提交已有任务；
3. 对当前任务重试一次 Prepare；
4. 如果重试仍失败，通过 `ascendc_assert` 报错。

这样当前任务不会因为批缓冲区边界而丢失。

### 7.4 其他错误

其他返回值不是正常批次边界，直接要求：

```cpp
ascendc_assert(ret == HCOMM_SUCCESS, ...);
```

这比旧代码忽略 `WriteWithNotifyNbi` 返回值更容易暴露非法地址、无效 channel 或接口状态错误。

## 8. FlushPreparedWrites 的提交语义

空批次直接返回：

```cpp
if (preparedWriteCount_ == 0) {
    return;
}
```

非空批次执行：

```cpp
hcomm_.WriteWithNotifyNbiPrepareEnd<true>(activeBatchHandle_, batchBuffer);
```

模板参数 `<true>` 表示 End 阶段直接 commit，即在把准备好的 WQE 发布到 SQ 后触发 doorbell，不需要再单独调用 `hcomm_.Commit()`。

提交成功后清空软件侧批次状态：

```cpp
activeBatchHandle_ = 0;
preparedWriteCount_ = 0;
```

批次会在三类位置结束：

1. 下一个远端任务切换到不同 channel；
2. Prepare 返回 buffer full 或 queue full；
3. `SendPhaseExpertToToken()` 完成全部 metadata 扫描。

metadata chunk 的边界本身不强制 Flush，因此同一 channel 的 batch 可以跨越两个 metadata chunk。

## 9. 为什么可以删除 Drain

改造后没有任何：

```cpp
hcomm_.Drain(commHandle);
```

原因是目标模式明确选择“提交后不在发送侧等待完成”：

- `PrepareEnd<true>` 负责发布 WQE 并 commit；
- `.cqe = 0` 不生成 CQE；
- 接收端依靠远端状态区中的 notify flag 判断 slot 是否可消费；
- token payload 的源地址位于 `xGm_`，在发送阶段后仍保持有效，不依赖短生命周期 UB 数据。

需要区分两个概念：

```text
PrepareEnd<true> 成功 = WQE 已发布并提交给设备
远端状态 flag 可见  = 对应 WriteWithNotify 已到达接收端可观察状态
```

`PrepareEnd<true>` 返回并不代表所有远端写已经完成。本改造将完成同步责任从发送侧 Drain 转移到了接收侧状态轮询协议。

## 10. flag 和 top-k weight 语义的保留

本分支 rebase 主线时，主线已把 top-k weight 从 payload 拼接方式改为 notify 携带方式。冲突解决后的实现保留了该逻辑。

远端路径仍先构造：

```cpp
uint64_t notifyValue = 1ULL << 32;
```

`HasTopkWeight == 1` 时，从 GM 读取 float32 的原始 32 位比特并写入低 32 位：

```cpp
uint64_t weight = static_cast<uint64_t>(ubWeightTensor(0));
notifyValue |= weight;
```

最终只是把调用入口从：

```cpp
WriteWithNotifyNbi(..., notifyValue);
Drain(...);
```

替换为：

```cpp
PrepareWriteWithNotify(..., notifyValue);
```

因此批量化没有把 notify 值退化为固定的 `1`，也没有丢失主线的 top-k weight 信息。

## 11. 本地 rank 路径保持不变

当 `srcRank == rankId_` 时不经过 Hcomm：

1. `CopyTokenToQueue()` 将 token 搬入队列；
2. `CopyTokenFromQueue()` 将 token 写入本地窗口；
3. 如果携带 top-k weight，将 weight 的 32 位原始值写入 `statusTensor_(0)`；
4. 使用 DataCopy 写入本地状态 slot。

本地路径不会进入 batch，也不会修改：

```text
activeBatchHandle_
preparedWriteCount_
```

因此本地 token 穿插在远端 token 之间时，不会破坏当前远端 batch。后续远端任务仍可继续追加到相同 channel 的活动批次。

## 12. 修改前后对比

| 维度 | 修改前 | 修改后 |
| --- | --- | --- |
| 远端 API | `WriteWithNotifyNbi` | `WriteWithNotifyNbiPrepare` |
| 批次提交 | 每 token 隐式提交 | `WriteWithNotifyNbiPrepareEnd<true>` 批量提交 |
| CQE | `.cqe = 1` | `.cqe = 0` |
| 完成等待 | 每 token `Drain` | 不调用 `Drain` |
| notify 内容 | flag + 可选 weight | 保持 flag + 可选 weight |
| payload 源 | `xGm_` | 保持 `xGm_` |
| 本地 rank | DataCopy | 不变 |
| channel 切换 | 每 token 独立，无 batch 状态 | 切换前 Flush 当前 batch |
| buffer/queue 满 | 不适用 | Flush 后重试当前任务 |
| 发送结束 | 最后一次 Drain 已完成 | 显式 Flush 剩余 Prepare 任务 |
| 新增 UB | 无 batch UB | 每 AIV 16416 字节 |

## 13. 时序示例

假设连续三个远端 token 使用同一 channel：

```text
AIV/Kernel             UB batch                  SQ/设备                  远端窗口
    |                      |                         |                         |
    |-- Prepare token0 --->| task0                   |                         |
    |-- Prepare token1 --->| task0, task1            |                         |
    |-- Prepare token2 --->| task0, task1, task2     |                         |
    |                      |                         |                         |
    |-- PrepareEnd<true> --|------------------------>| doorbell                |
    |                      |                         |-- payload0 ------------>|
    |                      |                         |-- notify0 ------------->|
    |                      |                         |-- payload1 ------------>|
    |                      |                         |-- notify1 ------------->|
    |                      |                         |-- payload2 ------------>|
    |                      |                         |-- notify2 ------------->|
    |                      |                         |                         |
    |-- 继续执行，不 Drain |                         |                         |
```

如果 token1 使用另一个 channel，则 token0 的 batch 会在 token1 Prepare 前提前 Flush。

## 14. 顺序与数据生命周期

### 14.1 payload 必须先于 notify 可见

接收端看到 notify flag 后会认为 slot 可消费，所以必须保证同一个 WriteWithNotify 操作中的 payload 写入先于 notify 对接收端可见。

本改造没有把 payload 和 notify 拆成两个独立 API，而是继续使用 Hcomm 的 WriteWithNotify prepare 接口。`odr=5` 和 `fence=1` 保持原值，单个任务的数据与通知顺序仍由 WriteWithNotify 语义保证。

### 14.2 源数据生命周期

无 Drain 后，提交函数返回时设备可能仍在读取源地址。当前远端 payload 源是 `xGm_` 中的 token，而不是即将被复用的临时 UB buffer，因此发送阶段结束后源地址仍有效。

如果未来把远端 payload 源改成循环复用的 GM staging slot 或短生命周期缓冲区，必须重新评估无 Drain 模式下的覆盖风险。

### 14.3 batch UB 生命周期

`WriteWithNotifyNbiPrepareEnd<true>` 在返回前完成 prepared WQE 从 UB 到 SQ 的发布流程。之后 batch header 可以被下一批复用。发送阶段末尾先 Flush，随后 `Process()` 才执行 `tpipe_->Reset()`，不会在 End 之前释放 batch UB。

## 15. 性能收益预期

假设一个 AIV 连续向同一 channel 发送 `N` 个远端 token，实际单批容量为 `B`：

```text
修改前：N 次 WriteWithNotify 提交 + N 次 Drain
修改后：N 次轻量 Prepare + ceil(N / B) 次 PrepareEnd 提交 + 0 次 Drain
```

预期收益主要来自：

- 减少 SQ 发布和 doorbell 次数；
- 消除逐 token CQE 生成；
- 消除逐 token Drain 等待；
- 允许设备端执行与 AIV 后续工作重叠。

收益取决于 token 的 channel 连续性。如果 token 在多个 channel 间频繁切换，`activeBatchHandle_` 会频繁变化，批量聚合效果会下降。

## 16. 风险与边界条件

### 16.1 无 CQE 模式不提供发送侧完成确认

`.cqe = 0` 后不能依赖 CQ 来确认每个任务完成。设备异常、远端错误或链路问题不能通过本路径的 CQE 反馈给发送 kernel。

### 16.2 无 Drain 模式需要依赖接收端协议

发送侧不会等待远端完成。正确性依赖接收端严格遵循“先观察状态 flag，再消费 payload”的协议。

### 16.3 跨批次 SQ 背压

Prepare 接口可以限制单个 batch 不超过当前计算出的 buffer/queue budget，但 `.cqe = 0` 且不 Drain 时，发送侧缺少基于完成事件的跨批次回收确认。

如果单个 kernel 在设备消费速度不足时连续提交大量 batch，必须结合目标平台 SQ 深度、Hcomm 实现和最大 token 数验证不会覆盖仍在执行的 SQ 项。这是无 CQE、无 Drain 方案最重要的压力测试项。

### 16.4 UB 占用

每个 AIV 新增 16416 字节 UB。需要与 metadata buffer、xQueue 和接收阶段临时 buffer 的峰值占用一起检查。当前 batch buffer 在发送阶段使用，接收阶段开始前会经过 `tpipe_->Reset()`。

### 16.5 SDK 版本依赖

代码依赖 asc-comm PR 75 引入的接口和返回码：

```text
WriteWithNotifyNbiPrepare
WriteWithNotifyNbiPrepareEnd
HCOMM_PREPARE_BUFFER_FULL
HCOMM_PREPARE_QUEUE_FULL
```

使用未包含这些接口的 CANN/Hcomm 版本会编译失败。

## 17. 建议验证项

### 17.1 编译检查

- 使用包含 asc-comm PR 75 的工具链编译 `moe_ep_combine`；
- 覆盖 `HasTopkWeight=0` 和 `HasTopkWeight=1`；
- 确认 16416 字节新增 UB 不触发 UB 超限。

### 17.2 功能检查

- 单 rank 本地路径；
- 两个及以上 rank 的远端发送；
- 有/无 top-k weight；
- token 数小于、等于和大于单 batch 容量；
- 多 channel 切换；
- metadata 跨 chunk；
- `actualA_ == 0` 等空输入边界。

### 17.3 状态内容检查

`HasTopkWeight=1` 时验证接收端状态 slot：

```text
高 32 位 == 1
低 32 位按 float32 bit pattern 还原后 == 发送端 top-k weight
```

### 17.4 压力与性能检查

- 大 token 数连续提交，确认没有 SQ 覆盖或远端超时；
- 比较修改前后的 doorbell、CQE、Drain 次数；
- 统计平均 batch 大小和 channel 切换导致的提前 Flush 次数；
- 对比 kernel 总耗时及发送/接收阶段重叠情况。

## 18. 当前静态验证结果

当前开发环境已完成：

- rebase 到最新 `upstream/master` 并解决 `SendSlot()` 冲突；
- `git diff --check` 通过；
- 确认代码中无冲突标记；
- 确认发送路径使用 `.cqe = 0`；
- 确认调用 `WriteWithNotifyNbiPrepareEnd<true>`；
- 确认 `moe_ep_combine.h` 中不再调用 `hcomm_.Drain()`；
- 确认主线的 `notifyValue = 1ULL << 32` 和 top-k weight 低 32 位逻辑仍存在。

当前 Windows 环境没有可用的 WSL/Ascend kernel 编译工具链，因此尚未完成目标设备编译和运行验证。
