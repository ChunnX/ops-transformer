# Double Buffer / Ping-Pong 实施规则

当优化目标是让 MTE2、MTE1、M、V、FIX 等不同流水段并行推进时读取本文件。这里记录通用方法，不绑定具体算子。

## 先判断是否值得开 DB

double buffer 适合这种结构：

- tile `i` 的计算正在消费 slot A；
- tile `i + 1` 的搬运或准备可以写入 slot B；
- 两个 slot 对应的物理存储不 alias；
- producer 和 consumer 属于可并行的 pipeline，且有明确事件同步。

以下情况通常不是有效 DB：

- 下一 tile 的输入依赖当前 tile 的输出，不能提前生产；
- 只有一个 tile，且没有可提前放入的 zero/identity/constant/下一消费者输入；
- 两个 buffer 名字不同，但底层来自同一段 L1/UB/L0；
- 每轮都清空、重建、写回完整临时区，开 DB 后固定开销抵消 overlap；
- 只把循环写成 `idx ^= 1`，但没有把下一轮 producer 安排到当前 consumer 之前或并行阶段中。

## 选择 DB 层级

先拆清楚要重叠哪一层，不要把 GM->L1 和 L1->L0 混成一个问题：

- GM->L1/UB：producer 常是 `DataCopy`/`DataCopyPad`，consumer 是 V/MTE1/MTE3/FIX 前的本地读取。
- L1->L0A/L0B：producer 是 `LoadData`，consumer 是 `Mmad`。
- L0C->L1/GM：producer 是 `Fixpipe`，consumer 是后续 MTE2/MTE1/V/M 或外部可见性。
- UB 内 V 计算：producer/consumer 都可能是 V，但仍可通过 `Buf1/Buf2` 和 sync flag 管理阶段边界。

若代码库已有封装，优先复用。例如标准 L0A/L0B MatMul buffer 可找 `BuffersPolicyDB` / `BufferManager`；手写特殊 layout、特殊清零、特殊事件时再落手写 ping-pong。

## 事件模型

每个 slot 至少要表达两个状态：

- `free`：consumer 已经不再使用该 slot，producer 可以覆盖。
- `ready`：producer 已经把本轮数据放好，consumer 可以读取。

L1->L0 + MMAD 的常见事件闭环：

```cpp
// 初始化：两个 slot 都先标记为 free。
SetFlag<HardEvent::M_MTE1>(event0);
SetFlag<HardEvent::M_MTE1>(event1);

for (uint32_t t = 0; t < tileCount; ++t) {
    uint32_t slot = t & 1U;
    event_t event = events[slot];

    // producer: MTE1 写 L0 slot。
    WaitFlag<HardEvent::M_MTE1>(event);
    LoadData(l0Slot[slot], l1Src[t], loadParams);
    SetFlag<HardEvent::MTE1_M>(event);

    // consumer: M 读 L0 slot。
    WaitFlag<HardEvent::MTE1_M>(event);
    Mmad(cL0, l0aSlot[slot], l0bSlot[slot], params);
    PipeBarrier<PIPE_M>();
    SetFlag<HardEvent::M_MTE1>(event);
}

// 退出或复用前 drain，避免下一段覆盖未消费 slot。
WaitFlag<HardEvent::M_MTE1>(event0);
WaitFlag<HardEvent::M_MTE1>(event1);
```

上面的 `SetFlag` 在 M pipeline 上排队，因此控制流进入下一轮后，MTE1 可以对另一个 free slot 发起下一次 `LoadData`，从而和上一轮 MMAD 重叠。不同流水段要替换成对应的 `HardEvent` 方向，不能照抄 `M_MTE1/MTE1_M`。

## 手写 DB 的落地步骤

1. 分配两个真实 slot。
   - L0A/L0B：要么 `TBuf` 大小乘 2 后用 offset 切分，要么两个独立 buffer，要么用已有 `BuffersPolicyDB`。
   - L1/UB：可用 `Buf1/Buf2` 或单 buffer 双 offset。
   - 检查字节数、对齐、生命周期和 `pipe->Reset()` 是否会覆盖。

2. 绑定 slot 到事件。
   - 固定事件 ID 可读性强，但要确认没有和同一 kernel 其他 DB 冲突。
   - 动态事件必须有完整 alloc/free/drain 生命周期；不要只因为“更通用”就换动态事件。
   - 一个 slot 的 free/ready 可以复用同一个 `event_t`，但 `HardEvent` 方向必须对应 producer/consumer。

3. 把生产函数和消费函数拆开。
   - `IssueLoad(slot)` 只负责等待 free、搬入、标记 ready。
   - `Consume(slot)` 只负责等待 ready、计算、标记 free。
   - 清零、identity、constant 预置如果属于本轮 operand，也放进 producer 阶段；不要在 consumer 等待后再做。

4. 安排 prime / steady / drain。
   - `prime`：先给第一个 slot 发 producer。
   - `steady`：消费当前 slot 时，尽早发下一 slot 的 producer，或让下一轮控制流在上一轮 M/V/FIX 尚未 drain 时发出搬运。
   - `drain`：循环结束等待两个 slot 回到 free，或者明确等待所有 in-flight event。

5. 保持数学和 layout 不变。
   - DB phase 只改调度和 buffer slot，不同时合并 `LoadData`、换 transpose、改 RHS/operand 来源。
   - 若还要合并搬运、改半块计算、改 resident 生命周期，拆到后续独立 phase。

## 常见错误

- 忘记初始化两个 slot 的 free flag，首轮 `WaitFlag` 卡死。
- 只释放了 slot 0，slot 1 第二次使用时卡死。
- producer 用 `SetFlag<MTE1_M>`，consumer 却 `WaitFlag<MTE2_V>` 等错方向。
- `LoadData` 写 slot B，但 `Mmad` 仍读 slot A 的固定指针。
- 两个 slot 共用同一段 L1/L0 offset，精度随机错或只在长序列错。
- 退出前未 drain，下一段 helper 复用 L0/L1 后覆盖 in-flight 数据。
- 当前 tile 的 L0B 依赖当前 tile 的 L0C->L1 结果，却尝试把 L0B 提前到本轮 MMAD 前；这类真依赖不能靠 DB 消除。

## 查参考实现的关键词

不要只搜 `doublebuffer`。在 Ascend C 代码里常见名字是：

- `DB`
- `Buf1` / `Buf2`
- `pingPong` / `PingPong`
- `BuffersPolicyDB`
- `SYNC_*_BUF1_FLAG` / `SYNC_*_BUF2_FLAG`
- 成对的 `EVENT_ID3` / `EVENT_ID4`

优先看含有完整 wait/load/ready/compute/free 闭环的代码；只有“组成 DB”的注释只能作为线索，不能直接当实现模板。
