# QuantBlockSparseAttn MXFP8 静态 Tensor 改造与同步设计

> 适用分支：`opt_tensor_gpt`  
> 基线提交：`b078d6a38b976a9ee145cfbb31bdda9ff81d0731`  
> 当前已验证提交：`a784dce56`  
> 硬件目标：Ascend 950 / Arch35  
> 改造范围：仅 QuantBlockSparseAttn 的 MXFP8 路径；普通 FP8 路径不在本文范围内。

本文记录 QBS MXFP8 从 `TPipe`、`TBuf/TQue`、`BufferManager` 动态管理改造成静态 `LocalTensor` 后的最终设计。重点说明静态地址、Event ID 分配、事件初始化与回收、AIC/AIV 跨核握手及完整任务流水，供后续功能扩展、性能优化和故障定位使用。

---

## 1. 结论与维护原则

当前实现遵循以下原则：

1. MXFP8 使用固定模板：`M=128`、`S2=512`、`D=128`、`DV=128`，布局为 TND Query + PA_BNSD KV。
2. UB、L1、L0A、L0B、L0C 全部通过固定地址构造 `LocalTensor`，不再依赖 `BufferManager`。
3. 所有本地 Event ID 和跨核 Event ID 均为编译期常量。
4. 可复用 buffer 使用“首轮 seed、热路径 Wait/Set、结尾 drain”的 token 生命周期。
5. BMM1/BMM2 共享 UB 使用 Mode 4 跨核双向握手；L1P 使用 AIV→AIC 单向 ready 握手。
6. MXFP8 路径当前没有任何 `PipeBarrier<...>()`。跨核 free 恢复为原 `Buffer::SetCrossCore()` 的 `CrossCoreSetFlag<MODE4, PIPE_V>` 语义。
7. 本地 Event ID 6/7 不使用。Q/K/V 的 L1 复用事件固定在 0～5。
8. 同一个数字可以出现在不同事件资源域；只有“同一事件类型或同一跨核计数器内”的重复 ID 才是冲突。
9. 修改 slot 数、`PRELOAD_N`、buffer 地址、事件 ID 或 Set/Wait 位置时，必须重新做完整生命周期审计和重复稳定性测试。

---

## 2. 改造文件与职责

| 文件 | 职责 |
|---|---|
| `op_kernel/arch35/quant_block_sparse_attn_mx_kernel.h` | MXFP8 kernel 入口、静态共享 UB/L1P 构造、三槽任务流水、跨核事件初始化和收尾 |
| `op_kernel/arch35/quant_block_sparse_attn_mx_block_cube.h` | AIC 侧 Q/K/V L1、L0A/B/C 静态布局，C1/C2 Matmul 与 Fixpipe |
| `op_kernel/arch35/quant_block_sparse_attn_mx_block_vec.h` | AIV 侧静态 UB 布局，V1 softmax/P 量化、V2 update、Output/LSE 写回 |
| `op_kernel/arch35/quant_block_sparse_attn_mx_static.h` | 容量、固定地址、全部 Event ID、跨核同步辅助函数 |
| `op_kernel/arch35/quant_block_sparse_attn_mx_static_matmul.h` | 不依赖 BufferManager 的静态 L0A/L0B Matmul 实现 |
| `op_kernel/arch35/quant_block_sparse_attn_mx_static_vec.h` | 使用 `LocalTensor` 参数的 MXFP8 V1 VF 包装 |
| `op_kernel/arch35/quant_block_sparse_attn_mx_static_init_output.h` | 不依赖动态队列的输出初始化实现 |
| `op_kernel/arch35/common/matmul.h` | 增加 `mxScaleASrcStride`，保证 C2 的 PScale 在 K tail 下仍按固定 256-column packed stride 读取 |

改造前后主要映射：

| 改造前 | 改造后 |
|---|---|
| `TPipe::InitBuffer` | `LocalTensor<uint8_t>(TPosition, constexpr_addr, constexpr_size)` |
| `TBuf/TQue` | 直接持有 `LocalTensor<T>` |
| `AllocTensor/FreeTensor` | 固定 slot + 显式 Event token |
| `BufferManager/BuffersPolicy` | 固定地址、固定 slot、显式 cursor |
| 动态 event 分配 | `quant_block_sparse_attn_mx_static.h` 中的 `constexpr TEventID` |
| `Buffer::Wait/Set` | 调用点附近的 `WaitFlag/SetFlag` |
| `Buffer::WaitCrossCore/SetCrossCore` | 明确的 Mode 4 `CrossCoreWaitFlag/CrossCoreSetFlag` |

---

## 3. 算法与任务流水

### 3.1 一个 logical task 的计算

每个 logical S2 task 最多拼接 512 个 KV token，并分成最多两个 256-token subLoop：

```text
C1: K[256,D] × Q^T[D,128] -> C1[256,128]
V1: C1 -> mask + online softmax -> P/PScale

两个 subLoop 合并后：
P[128,512] + PScale[128,512/32]

C2: P[128,512] × V[512,DV] -> C2[128,DV]
V2: C2 + 历史累计值 -> AttentionOut，并在最后一个 S2 task 写 LSE
```

两个 AIV 各处理一半 M 轴，即 `VEC_M_BASE=64` 行。

### 3.2 三槽延迟流水

`PRELOAD_N=2`，任务状态和 L1P 使用 `PRELOAD_N+1=3` 个 slot：

```text
loop t:
  当前任务 t       执行 C1/V1
  两拍前任务 t-2   执行 C2/V2
```

slot 选择：

```cpp
taskSlot = loop % 3;
bmm1Slot = bmm1Cursor;  // 1,0,1,0...
bmm1Cursor ^= 1;
```

L1P slot `t % 3` 在任务 `t` 的 V1 中写入，在 loop `t+2` 的 C2 中读取，并在任务 `t+3` 才被再次作为写目标。这里使用的是单向 ready 事件，没有 C2→V1 的 free ack，因此 `PRELOAD_N=2` 与三槽 L1P 是一个整体设计约束；修改延迟或槽数时必须重新验证 L1P 生命周期。

---

## 4. 静态内存布局

所有地址均为对应硬件存储空间内的 byte offset，不是 GM 地址。

### 4.1 容量常量

| 存储空间 | 容量常量 | 容量 |
|---|---|---:|
| UB | `UB_CAPACITY` | 256 KiB |
| L1 | `L1_CAPACITY` | 512 KiB |
| L0A | `L0A_CAPACITY` | 64 KiB |
| L0B | `L0B_CAPACITY` | 64 KiB |
| L0C | `L0C_CAPACITY` | 256 KiB |

容量和边界通过 `static_assert` 在编译期检查。

### 4.2 AIC/AIV 共享 UB

| 区域 | 地址常量 | 起始地址 | 大小常量 | 大小 | 生产者 | 消费者 |
|---|---|---:|---|---:|---|---|
| BMM1 UB slot 0 | `BMM1_UB_ADDR0` | 0 | `BMM1_UB_SLOT_BYTES` | 65,536 B | AIC Fixpipe | AIV V1 |
| BMM1 UB slot 1 | `BMM1_UB_ADDR1` | 65,536 | `BMM1_UB_SLOT_BYTES` | 65,536 B | AIC Fixpipe | AIV V1 |
| BMM2 UB | `BMM2_UB_ADDR` | 131,072 | `BMM2_UB_SLOT_BYTES` | 32,768 B | AIC Fixpipe | AIV V2 |
| AIV 私有 UB 起点 | `VEC_PRIVATE_UB_BASE` | 163,840 | — | — | — | — |

BMM1 每个 AIV 获得 `[64,256]` fp32 数据；BMM2 每个 AIV 获得 `[64,128]` fp32 数据。

### 4.3 AIV 私有 UB 最大布局

下表按 `HAS_LSE=true`、`HAS_ATTEN=true` 的最大配置计算。MXFP8 数据和 E8M0 scale 均按 1 byte 计算。

| Buffer | 起始地址 | 大小 | 用途 |
|---|---:|---:|---|
| `stage2OutBuf_` | 163,840 | 32,768 B | V2 `[64,128]` fp32 累加输出 |
| `stage1OutBuf_[0]` | 196,608 | 16,896 B | V1 P 的 DN packed 双缓冲 0 |
| `stage1OutBuf_[1]` | 213,504 | 16,896 B | V1 P 的 DN packed 双缓冲 1 |
| `pScaleSubLoopBuf_` | 230,400 | 512 B | 一个 256-column subLoop 的 PScale |
| `softmaxSumBuf_[3]` | 230,912 | 768 B | 三槽 softmax sum |
| `softmaxMaxBuf_[3]` | 231,680 | 768 B | 三槽 softmax max |
| `softmaxExpBuf_[3]` | 232,448 | 768 B | 三槽 softmax exp/update 状态 |
| `preLoopMaxBuf_` | 233,216 | 256 B | online softmax 临时状态 |
| `preLoopSumBuf_` | 233,472 | 256 B | online softmax 临时状态 |
| `firstLoopSumBuf_` | 233,728 | 256 B | online softmax 临时状态 |
| `softmaxLseBuf_` | 233,984 | 2,048 B | LSE 计算/写回工作区 |
| `attenMaskBuf_` | 236,032 | 16,384 B | `[256,64]` mask 拼接区 |
| `vselrIndexesBuf_` | 252,416 | 256 B | DN VF lane 重排索引 |
| 最大结束地址 | 252,672 | — | 剩余 9,472 B |

`initOutputScratch_` 从 `stage2OutBuf_` 的地址开始复用 8,160 B。它只在主循环前的 ClearOutput 阶段使用，因此与 Stage2 不同时存活。

`HAS_LSE=false` 或 `HAS_ATTEN=false` 时，后续地址由 `constexpr` 自动前移，但结束地址仍由 `static_assert` 检查。

### 4.4 共享 L1P

每个 L1P slot：

```text
P       = 128 × 512 × 1 B      = 65,536 B
PScale  = 128 × (512 / 32) B   =  2,048 B
总计                              67,584 B
```

| 区域 | 地址常量 | 起始地址 | 大小常量 | 大小 |
|---|---|---:|---|---:|
| L1P slot 0 | `L1_P_ADDR0` | 0 | `L1_P_SLOT_BYTES` | 67,584 B |
| L1P slot 1 | `L1_P_ADDR1` | 67,584 | `L1_P_SLOT_BYTES` | 67,584 B |
| L1P slot 2 | `L1_P_ADDR2` | 135,168 | `L1_P_SLOT_BYTES` | 67,584 B |
| AIC 私有 L1 起点 | `CUBE_PRIVATE_L1_BASE` | 202,752 | — | — |

### 4.5 AIC 私有 L1

| Buffer | slot 数 | 单 slot 大小 | 总大小 | 地址范围 |
|---|---:|---:|---:|---|
| Q + QScale | 2 | 16,896 B | 33,792 B | `[202752,236544)` |
| K + KScale | 2 | 33,792 B | 67,584 B | `[236544,304128)` |
| V + VScale | 2 | 33,792 B | 67,584 B | `[304128,371712)` |

L1 最大结束地址为 371,712，小于 524,288，剩余 140,576 B。

注意：最终实现的 K 是双缓冲，不是四缓冲。这样 Q/K/V 的 `MTE1_MTE2` 复用事件可以连续放在 ID 0～5，避开 6/7。

### 4.6 L0 布局

| 存储空间 | slot | 地址 | 单 slot 大小常量 | 单 slot 大小 |
|---|---:|---:|---|---:|
| L0A | 0/1 | 0 / 32,768 | `L0_SLOT_BYTES` | 32,768 B |
| L0B | 0/1 | 0 / 32,768 | `L0_SLOT_BYTES` | 32,768 B |
| L0C | 0/1 | 0 / 131,072 | `L0C_SLOT_BYTES` | 131,072 B |

L0A/L0B/L0C 均刚好使用对应容量，不能直接增加 slot 或扩大单 slot，除非同步调整模板或分块策略。

---

## 5. Event ID 的资源域

### 5.1 ID 不是全局唯一编号

以下写法中的数字 0 不代表同一个硬件 token：

```cpp
SetFlag<HardEvent::MTE3_V>(0);
SetFlag<HardEvent::V_MTE2>(0);
SetFlag<HardEvent::MTE1_MTE2>(0);
CrossCoreSetFlag<4, PIPE_V>(0);
```

当前代码把事件分成以下资源域：

1. Mode 4 CrossCore counter。
2. `MTE1_MTE2` / `MTE2_MTE1`。
3. `M_MTE1` / `MTE1_M`。
4. `FIX_M` / `M_FIX`。
5. `MTE3_V` / `V_MTE3`。
6. `V_MTE2` / `MTE2_V`。

同一数字可以在不同资源域复用。审计时必须同时写出 `HardEvent` 方向、ID 和 buffer 名，不能只写“event 0”。

### 5.2 Ready 与 Free 的方向

约定：

```text
Ready：生产者已经写完，消费者可以读。
Free ：消费者已经读完，生产者可以覆盖。
```

例如 MTE2 生产、MTE1 消费的 L1 buffer：

```text
MTE1_MTE2 = Free，MTE1 通知 MTE2 可以覆盖
MTE2_MTE1 = Ready，MTE2 通知 MTE1 可以读取
```

---

## 6. 跨核 Event ID 分配与工作方式

### 6.1 ID 分配

| 宏 | 基础 ID | AIV0/AIC counter | AIV1/AIC counter | Buffer | 方向 |
|---|---:|---:|---:|---|---|
| `CC_BMM1_EVENT_ID0` | 0 | 0 | 16 | BMM1 UB slot 0 | ready + free 双向 |
| `CC_BMM1_EVENT_ID1` | 1 | 1 | 17 | BMM1 UB slot 1 | ready + free 双向 |
| `CC_BMM2_EVENT_ID` | 2 | 2 | 18 | BMM2 UB | ready + free 双向 |
| `CC_L1P_READY_EVENT_ID0` | 5 | 5 | 21 | L1P slot 0 | AIV→AIC ready |
| `CC_L1P_READY_EVENT_ID1` | 8 | 8 | 24 | L1P slot 1 | AIV→AIC ready |
| `CC_L1P_READY_EVENT_ID2` | 9 | 9 | 25 | L1P slot 2 | AIV→AIC ready |

Mode 4 由 `CROSS_CORE_SYNC_MODE=4` 指定，AIV1 映射偏移由 `MX_AIV0_AIV1_OFFSET=16` 表示。一个逻辑 AIC 对应 AIV0 和 AIV1：

- AIC 发给 AIV0：使用基础 ID。
- AIC 发给 AIV1：使用基础 ID + 16。
- AIV0 发给 AIC：AIV0 使用基础 ID，AIC 等基础 ID。
- AIV1 发给 AIC：AIV1 仍使用基础 ID，硬件自动映射到 AIC 的基础 ID + 16。

因此 AIV 侧绝对不能手工给 `eventId` 加 16：

```cpp
// 正确：AIV0、AIV1 都执行
CrossCoreSetFlag<4, PIPE_V>(eventId);

// 错误：AIV1 手工 +16 会进入错误 counter
CrossCoreSetFlag<4, PIPE_V>(eventId + 16);
```

### 6.2 BMM1/BMM2 双向握手

每个共享 UB slot 的完整周期：

```text
初始化：
  AIV0 Set free(baseId)
  AIV1 Set free(baseId)        -- 硬件映射到 baseId+16

AIC：
  Wait free(baseId)
  Wait free(baseId+16)         -- 两个 AIV 都已读完/首轮可写
  Fixpipe -> shared UB
  Set ready(baseId)            -- 给 AIV0
  Set ready(baseId+16)         -- 给 AIV1

AIV0/AIV1：
  Wait ready(baseId)
  VF 读取 shared UB
  Set free(baseId)             -- AIV1 自动映射到 baseId+16

下一轮 AIC 再次 Wait 两个 free counter。
```

对应代码：

```cpp
AicWaitUbFree(id);   // PIPE_FIX，wait id 和 id+16
AicSetUbReady(id);   // PIPE_FIX，set id 和 id+16
AivWaitUbReady(id);  // PIPE_V，每个 AIV wait 自己映射后的 ready
AivSetUbFree(id);    // PIPE_V，每个 AIV set 自己映射后的 free
```

为什么同一个基础 ID 可以同时表示 ready 和 free：Mode 4 的 AIC→AIV 与 AIV→AIC 是方向独立的 counter；它们不是同一个计数器。但不同 buffer 仍必须使用不同基础 ID。

### 6.3 初始化和收尾为什么必要

第一次 AIC 写 BMM UB 前没有历史消费者，因此 `CrossCoreBufferInit()` 由两个 AIV 各产生一次初始 free token：

```cpp
AivSetUbFree(BMM1_ID0);
AivSetUbFree(BMM1_ID1);
AivSetUbFree(BMM2_ID);
```

正常循环中，每次 AIV 消费结果后产生下一次 free。最后一次消费产生的 free 不会被下一任务使用，所以 `CrossCoreBufferUninit()` 由 AIC 将三个 buffer 的最终 free token 全部消费：

```cpp
AicWaitUbFree(BMM1_ID0);
AicWaitUbFree(BMM1_ID1);
AicWaitUbFree(BMM2_ID);
```

对每个共享 UB slot：

```text
AIV free Set 次数 = 1 次初始化 + 该 slot 实际消费次数
AIC free Wait 次数 = 该 slot 实际生产次数 + 1 次收尾
```

由于每次生产都对应一次消费，两边最终相等。删除 seed 会导致首轮死锁；删除 drain 会让 counter 残留到后续 kernel 调用。

### 6.4 L1P 单向 ready

两个 AIV 分别写自己负责的 64 行 P/PScale。只有最后一个 subLoop 完成后，各 AIV 才执行：

```cpp
CrossCoreSetFlag<4, PIPE_MTE3>(l1PEventId);
```

AIC 的 C2 在读取 L1P 前等待两个 AIV：

```cpp
CrossCoreWaitFlag<4, PIPE_MTE1>(l1PEventId);
CrossCoreWaitFlag<4, PIPE_MTE1>(l1PEventId + 16);
```

该通道只传递 ready，不传递 free；slot 的安全复用依赖三槽任务调度。后续若把 `PRELOAD_N` 改为 1、增加并发或调整 C1/C2 先后顺序，应优先考虑为 L1P 增加反向 free 通道，而不是继续假设三槽天然安全。

---

## 7. AIC 本地 Event ID

### 7.1 Q/K/V L1 事件

| Buffer | slot | 常量 | ID | Free 事件 | Ready 事件 |
|---|---:|---|---:|---|---|
| Q | 0 | `L1_Q_EVENT_ID0` | 0 | `MTE1_MTE2` | `MTE2_MTE1` |
| Q | 1 | `L1_Q_EVENT_ID1` | 1 | `MTE1_MTE2` | `MTE2_MTE1` |
| K | 0 | `L1_K_EVENT_ID0` | 2 | `MTE1_MTE2` | `MTE2_MTE1` |
| K | 1 | `L1_K_EVENT_ID1` | 3 | `MTE1_MTE2` | `MTE2_MTE1` |
| V | 0 | `L1_V_EVENT_ID0` | 4 | `MTE1_MTE2` | `MTE2_MTE1` |
| V | 1 | `L1_V_EVENT_ID1` | 5 | `MTE1_MTE2` | `MTE2_MTE1` |

通用生命周期：

```text
InitBuffers: Set MTE1_MTE2(id)       -- seed free
MTE2:       Wait MTE1_MTE2(id)
MTE2:       GM -> L1
MTE2:       Set MTE2_MTE1(id)        -- ready
MTE1:       Wait MTE2_MTE1(id)
MTE1/M:     LoadData + Mmad
MTE1:       Set MTE1_MTE2(id)        -- free
Finalize:   Wait MTE1_MTE2(id)       -- drain final token
```

Q 的特殊点：一个 query block 的多个 S2 task 复用同一个 Q slot，只在第一个有效 S2 task 的 subLoop 0 搬入 Q/QScale，在最后一个有效 S2 task 的最后 subLoop 才释放 Q。中间复用路径使用一组 `MTE2_MTE1 Set/Wait` 维持当前统一控制流，但不会重复从 GM 搬 Q。

K 每个 C1 subLoop 搬一次并释放。V 每个 C2 的 K 分片搬一次并释放，tail 数据补零到 64-token 对齐。

### 7.2 L0A/L0B 事件

| Buffer | slot | 常量 | ID | Free | Ready |
|---|---:|---|---:|---|---|
| L0A | 0 | `L0A_EVENT_ID0` | 0 | `M_MTE1` | `MTE1_M` |
| L0A | 1 | `L0A_EVENT_ID1` | 1 | `M_MTE1` | `MTE1_M` |
| L0B | 0 | `L0B_EVENT_ID0` | 2 | `M_MTE1` | `MTE1_M` |
| L0B | 1 | `L0B_EVENT_ID1` | 3 | `M_MTE1` | `MTE1_M` |

`MxMatmulFullStatic()` 内部流程：

```text
Wait M_MTE1       -> 上一轮 Mmad 已释放 slot
LoadData L1->L0
Set MTE1_M
Wait MTE1_M       -> 数据已进入 L0
Mmad
Set M_MTE1        -> 释放 L0 slot
```

L0A 和 L0B cursor 独立维护，但当前每次 Matmul 都各前进一格。

### 7.3 L0C 事件

| Buffer | slot | 常量 | ID | Free | Ready |
|---|---:|---|---:|---|---|
| L0C | 0 | `L0C_EVENT_ID0` | 0 | `FIX_M` | `M_FIX` |
| L0C | 1 | `L0C_EVENT_ID1` | 1 | `FIX_M` | `M_FIX` |

流程：

```text
Wait FIX_M        -> 上一轮 Fixpipe 已释放 L0C
Mmad 写 L0C
Set M_FIX
Wait M_FIX        -> 等 Matmul 完成
等待共享 UB free
Fixpipe L0C->UB
Set FIX_M         -> 释放 L0C
```

初始化 seed `FIX_M`，结束时 drain `FIX_M`。

---

## 8. AIV 本地 Event ID

### 8.1 ID 分配表

`MTE3_V/V_MTE3` 资源域：

| Buffer | 常量 | ID | 含义 |
|---|---|---:|---|
| Stage1 slot 0 | `STAGE1_EVENT_ID0` | 0 | P 的 Vector 生产与 MTE3 写 L1 同步 |
| Stage1 slot 1 | `STAGE1_EVENT_ID1` | 1 | 同上 |
| PScale | `PSCALE_EVENT_ID` | 2 | PScale 的 Vector 生产与 MTE3 写 L1 同步 |
| Stage2 | `STAGE2_EVENT_ID` | 3 | V2 累加 UB 与 AttentionOut 写回同步 |
| LSE | `LSE_EVENT_ID` | 4 | LSE UB 与 GM 写回同步 |
| InitOutput free | `INIT_OUTPUT_FREE_EVENT_ID` | 5 | 初始化 scratch 的复用 token |
| InitOutput ready | `INIT_OUTPUT_READY_EVENT_ID` | 10 | Duplicate 后通知 MTE3 |

`V_MTE2/MTE2_V` 资源域：

| Buffer | 常量 | ID | 含义 |
|---|---|---:|---|
| Attention Mask | `MASK_EVENT_ID` | 0 | MTE2 搬入与 Vector 消费同步 |

### 8.2 Stage1 双缓冲

每个 Stage1 slot 的生命周期：

```text
InitEvents: Set MTE3_V(id)            -- 首轮可被 Vector 写
ProcessV1:  Wait MTE3_V(id)
VF:         生成 packed P
Set V_MTE3(id)
Wait V_MTE3(id)
MTE3:       P UB -> L1P
Set MTE3_V(id)                         -- MTE3 完成后释放 slot
Finalize:   Wait MTE3_V(id)
```

Stage1 slot 由 `runInfo.loop % 2` 选择。Event ID 必须与 slot 一一对应，不能用 task ID 直接取模到其他事件表。

### 8.3 PScale 单缓冲

PScale 每个 subLoop 都复用同一个 UB：

```text
seed MTE3_V(2)
Wait MTE3_V(2)
VF 生成 PScale
Set/Wait V_MTE3(2)
MTE3 写 L1P
Set MTE3_V(2)
最终 drain MTE3_V(2)
```

PScale 在 L1P 中始终按完整 512-column tile 排布。C2 的尾部 K 分片仍必须使用固定 256-column packed stride，不能根据 `realK` 改成紧凑 stride；当前通过 `MMParam::mxScaleASrcStride` 保证这一点。

### 8.4 Stage2 累加 buffer

Stage2 是单 buffer，当前每个 V2 task 都执行一次 token 周期：

```text
seed MTE3_V(3)
Wait MTE3_V(3)

第一个 S2 task：Adds(dst, bmm2, 0) 建立 V 流水上的 UB->UB 复制
中间 S2 task：FlashUpdateNew
最后 S2 task：FlashUpdateLastNew / LastDiv

若为最后 task：
  Set/Wait V_MTE3(3)
  MTE3 写 AttentionOut

Set MTE3_V(3)
最终 drain MTE3_V(3)
```

首个 S2 task 必须使用 Vector 指令 `Adds(..., 0)` 完成复制。静态 `LocalTensor` 不再携带原 TQue 的隐式依赖，不能随意改回无法确定执行流水的通用 UB→UB `DataCopy`。

### 8.5 LSE buffer

仅最后一个有效 S2 task写 LSE：

```text
seed MTE3_V(4)
Wait MTE3_V(4)
Vector 计算 LSE
Set/Wait V_MTE3(4)
MTE3 写 GM
Set MTE3_V(4)
最终 drain MTE3_V(4)
```

### 8.6 Attention Mask

```text
InitEvents: Set V_MTE2(0)             -- MTE2 首轮可写 mask UB
Wait V_MTE2(0)
MTE2 搬入/拼接实际需要的 mask segment
若发生 MTE2 搬运：Set/Wait MTE2_V(0)
VF 消费 mask
Set V_MTE2(0)                         -- 释放 mask UB
Finalize: Wait V_MTE2(0)
```

没有实际 MTE2 搬运的分支不会产生 `MTE2_V` token，但仍保持 mask UB 的复用窗口。

### 8.7 输出初始化与 SyncAll

ClearOutput 在主循环前执行：

```text
Set MTE3_V(INIT_OUTPUT_FREE=5)
Wait MTE3_V(5)
Duplicate scratch
Set/Wait V_MTE3(INIT_OUTPUT_READY=10)
循环 MTE3 写 GM
Set MTE3_V(5)

AttentionOut 和可选 LSE 初始化完成后：
Wait MTE3_V(5)
SyncAll()
```

普通计算事件必须在这个 `SyncAll()` 之后 seed，跨核 BMM free token 也必须在其后产生。否则 SyncAll 前产生的 token 可能与后续同步阶段交叉，造成首轮等待异常或残留计数。

---

## 9. 一次 Kernel 调用中的完整同步时序

### 9.1 初始化阶段

```text
1. InitConstInfo / InitMMResBuf
2. AIC InitBuffers
   - 构造 Q/K/V、L0A/B/C
   - seed L1、L0A/B、L0C 的 free token
3. AIV InitBuffers
   - 构造 Stage1/2、PScale、softmax、mask、LSE 等静态 UB
4. AIV ClearOutput
   - 初始化 GM output/LSE
   - SyncAll
5. AIV InitEvents
   - seed Stage1、PScale、Stage2、LSE、Mask free token
6. CrossCoreBufferInit
   - AIV0/AIV1 为 BMM1[2]、BMM2[1] seed free token
```

顺序不能变成 CrossCoreBufferInit→ClearOutput，因为 ClearOutput 内含 `SyncAll()`。

### 9.2 C1/V1 阶段

对每个 256-token subLoop：

```text
AIC:
  Q/K L1 free -> 搬 Q/K+scale -> ready
  L0A/B free -> LoadData 随路反量化 -> ready
  Mmad C1 -> L0C
  等 BMM1 UB 的两个 AIV free
  Fixpipe C1 -> 对应 BMM1 UB slot
  给两个 AIV发送 BMM1 ready

AIV0/AIV1:
  等各自 BMM1 ready
  可并行搬 mask
  VF: mask + softmax + P/PScale 量化
  直接在 PIPE_V 上发送 BMM1 free
  PScale/P 通过 V_MTE3 同步后由 MTE3 写入 L1P
  最后 subLoop 完成后发送 L1P ready
```

BMM1 free 发送前不再有 `PipeBarrier<PIPE_V>()`。当前 `AivSetUbFree()` 本身使用 `CrossCoreSetFlag<MODE4, PIPE_V>`，与改造前 `Buffer::SetCrossCore()` 相同，并已通过精度重复验证。

### 9.3 C2/V2 阶段

任务延迟两拍后：

```text
AIC:
  等两个 AIV 的 L1P ready
  V L1 free -> 搬 V/VScale -> ready
  L0A/B free -> LoadData 随路反量化 -> ready
  Mmad C2 -> L0C
  等 BMM2 UB 的两个 AIV free
  Fixpipe C2 -> BMM2 UB
  给两个 AIV发送 BMM2 ready

AIV0/AIV1:
  等各自 BMM2 ready
  Adds / FlashUpdate 消费 BMM2 UB
  直接在 PIPE_V 上发送 BMM2 free
  最后 S2 task 完成归一化并写 AttentionOut
  可选写 LSE
```

### 9.4 尾部排空阶段

主任务枚举结束后，`while (HasValidTask)` 继续调用 `ExecuteTask`，排空尚未执行 C2/V2 的最后两个 task。

随后：

```text
AIC FinalizeBuffers:
  drain Q/K/V、L0A/B、L0C 的最终 free token

AIV FinalizeBuffers:
  drain Stage1、PScale、Stage2、LSE、Mask 的最终 free token

CrossCoreBufferUninit:
  AIC drain BMM1[2]、BMM2[1] 的两个 AIV 最终 free counter
```

---

## 10. 已解决问题及根因

### 10.1 本地事件 ID 6/7 冲突

静态 Tensor/SyncAll 路径保留了部分事件资源。早期将更多 K slot 和其他 buffer 放到更高 ID 后，出现卡死或随机结果。最终方案：

```text
Q: 0/1
K: 2/3
V: 4/5
```

K 从四槽收敛到双槽，使所有 Q/K/V 复用事件落在 0～5。不要重新引入 6/7。

### 10.2 Mode 4 的 AIV1 ID 映射

Mode 4 中 AIV1 发出的基础 ID 会由硬件映射到 AIC 的 `ID+16`。AIV1 手工加 16 会导致错误 counter，表现为：

- Set/Wait 不匹配；
- counter 残留；
- AIC 永久等待；
- 错误提前释放共享 UB；
- 多次运行精度不同。

最终规则：AIV0/AIV1 都只传基础 ID；只有 AIC 显式访问两个方向时使用 `id` 与 `id+16`。

### 10.3 V 事件 8/9 与 BMM2 输出异常

历史现象是 LSE 100%，AttentionOut 约 74%～79% 且多次运行变化。LSE 只依赖 C1/softmax，而 Out 还依赖 V/C2，因此误差分布指向 V 搬运或 BMM2 生命周期。V 事件改为 4/5 后精度恢复稳定。

### 10.4 Stage2 首轮复制流水不明确

静态 Tensor 去掉 TQue 后，通用 `DataCopy` 的执行流水无法再依赖队列元数据推断。首个 S2 task 使用：

```cpp
Adds<MM_T>(stage2, bmm2, 0.0F, size);
```

它既完成数值等价复制，也明确处于 V 流水，使后续 FlashUpdate、归一化和跨核 free 保持同一流水顺序。

### 10.5 PScale K-tail stride

C2 的 PScale 在 L1 中始终按固定 256-column subLoop packed。最后一个 K 分片可能只有 64/128/192，但 scale 源 stride 不能跟随 `realK` 收缩。`MMParam::mxScaleASrcStride` 固定为：

```cpp
S2_SPLIT / MX_TOKEN_GROUP
```

否则 S2 tail 的 AttentionOut 会错，而 LSE 仍正确。

### 10.6 Tail 初始化

- V tail 补零到 64-token 对齐。
- PScale 未覆盖的尾部填充 E8M0 `1.0` 表示值。
- C2 packed 行数和 stride 保持固定模板约束，不能为了 tail 任意缩小布局。

### 10.7 最后有效 S2 task

sparse metadata 尾部可能包含负数或超出实际 KV 长度的 block。`FindLastValidS2Loop()` 查找最后一个真正贡献 token 的 S2 group，让 Q 释放、AttentionOut 归一化和 LSE 写回绑定到最后有效任务，而不是最后一个 metadata group。

### 10.8 多余 PIPE_V barrier

排查过程中曾在 BMM1/BMM2 free 前增加 `PipeBarrier<PIPE_V>()`。最终确认原 BufferManager 路径的 `SetCrossCore()` 也只是 `CrossCoreSetFlag<MODE4, PIPE_V>`，并无显式 barrier。保留正确 ID 布局后，两处 barrier 均已移除且精度正常。

---

## 11. 后续修改规范

### 11.1 新增或扩大 buffer

必须完成：

1. 在对应存储空间中计算固定地址和大小。
2. 检查与所有已有 buffer 的生命周期是否允许 alias。
3. 增加 `static_assert(end <= CAPACITY)`。
4. 若需要多 slot，明确 cursor 初值和轮转公式。
5. 为每个可复用 slot 分配独立事件，不能让两个同时在途的 slot 共用一个 ID。

### 11.2 新增事件的审计模板

每个新事件在评审说明中填写：

| 项目 | 必填内容 |
|---|---|
| Buffer | 被保护的准确 buffer 和 slot |
| 资源域 | CrossCore 或具体 `HardEvent` 对 |
| Event ID | 数字及是否连续 |
| Producer | 谁 Set、在哪条流水 |
| Consumer | 谁 Wait、在哪条流水 |
| Seed | 首轮 token 由谁产生 |
| Hot path | 每个 task/subLoop 的 Set/Wait 次数 |
| Drain | 最后一枚 token 由谁消费 |
| Tail/空任务 | 是否仍保持次数对称 |

### 11.3 Set/Wait 计数规则

对于循环复用 buffer，建议使用以下形式：

```text
Seed Set = 1
每次使用前 Wait = 1
每次使用后 Set = 1
Finalize Wait = 1
```

若实际使用次数为 N：

```text
总 Set  = 1 + N
总 Wait = N + 1
```

跨核双 AIV buffer 需要分别对 AIV0/AIV1 counter 做同样计算，不能只看基础 ID。

### 11.4 不允许的修改

未经完整验证，不要：

- 使用 Event ID 6/7；
- 在 AIV1 上手工执行 `eventId + 16`；
- 删除 InitEvents、FinalizeBuffers、CrossCoreBufferInit 或 CrossCoreBufferUninit；
- 把 BMM1/BMM2 的 ready/free ID 分配给其他共享 buffer；
- 修改 L1P 三槽数却保持 `PRELOAD_N=2` 的原假设；
- 把 Stage2 首轮 `Adds(...,0)` 改回流水不明确的复制；
- 根据 tail `realK` 压缩 PScale 的源 stride；
- 只跑一次精度就判断同步正确。

### 11.5 可以继续评估的优化

以下优化尚未纳入当前稳定版本：

1. Stage2 的 `MTE3_V` token 从每个 S2 task 缩减到每个 query 一次。
2. 合并 P 与 PScale 的两组 `V_MTE3` source-ready 同步。
3. 删除 Q 复用分支中没有真实 MTE2 搬运的合成 `MTE2_MTE1 Set/Wait`。
4. 若确认更高本地 Event ID 在目标编译器/硬件上安全，可评估恢复更多 K slot；不得占用 6/7。

每项必须单独提交、单变量测试，不能再次把 ID 调整和同步删减混在一个提交中。

---

## 12. 编译与回归建议

### 12.1 编译

```bash
bash build.sh --pkg \
  --ops="quant_block_sparse_attn,quant_block_sparse_attn_metadata" \
  --soc=ascend950 \
  --experimental
```

### 12.2 必测用例

| 用例 | 覆盖重点 |
|---|---|
| `STC_B1_BS64_STITCH_8_BLOCKS_MASK3` | 多 sparse block 拼接、mask、基础正确性 |
| `STC_B1_BS64_S2_TAIL_MASK3` | S2 tail、PScale stride、随机同步问题 |
| `STC_B1_BS128_S1_TAIL_GT_GQA10` | S1 tail、GQA、两 AIV 分工 |
| `STC_B1_BS128_MIN_PAGE_EQ_MASK3` | 小 page、mask、历史卡死路径 |
| `STC_B1_BS128_TWO_S2_LOOPS_REVERSE` | 多 S2 task、Q 复用、反序 sparse block、历史卡死路径 |

同步相关修改至少满足：

1. 全部用例 Out/LSE 精度通过。
2. S2 tail 连续运行 50 次结果完全稳定。
3. 两个历史卡死用例连续运行，无超时和 507015。
4. 多个用例连续在同一 device/process 中运行，检查事件 counter 是否跨调用残留。
5. 比较修改前后 median 与 P95 耗时；小于测试噪声的优化不应以增加同步复杂度为代价。

### 12.3 看到不同症状时的优先检查方向

| 症状 | 优先检查 |
|---|---|
| LSE 正确、Out 错误 | V/VScale、PScale stride、C2、BMM2 UB 生命周期 |
| LSE 和 Out 都错 | Q/K、C1、mask、softmax、BMM1 UB 生命周期 |
| 多次运行精度不同 | Event ID 复用、free 提前、未 drain counter、未初始化 tail |
| 永久卡死 | Seed 缺失、Set/Wait 次数不等、AIV1 `+16` 错误、空任务分支漏 Set |
| 507015 | 越界地址、错误 tensor position、无效 scale offset，或同步错误导致非法覆盖 |
| 只有 tail 错 | 对齐、padding、packed stride、最后有效 task 判断 |

---

## 13. 代码定位索引

| 主题 | 入口 |
|---|---|
| 地址和全部 Event ID | `quant_block_sparse_attn_mx_static.h` |
| 初始化总体顺序 | `QuantBlockSparseAttnMxKernel::Process()` |
| 共享 UB/L1P 构造 | `QuantBlockSparseAttnMxKernel::InitMMResBuf()` |
| 跨核 seed/drain | `CrossCoreBufferInit()` / `CrossCoreBufferUninit()` |
| 三槽调度 | `ExecuteTask()` / `ProcessMainLoop()` |
| AIC 本地 seed/drain | `QuantBlockSparseAttnMxBlockCube::InitBuffers()` / `FinalizeBuffers()` |
| C1 | `IterateBmm1()` |
| C2 | `IterateBmm2()` |
| L0A/B token | `MxMatmulFullStatic()` |
| AIV 本地 seed/drain | `QuantBlockSparseAttnMxBlockVec::InitEvents()` / `FinalizeBuffers()` |
| V1 | `ProcessVec1()` |
| V2 | `ProcessVec2()` |
| Mask | `CopyAttenMaskToUb()` |
| P/PScale 写 L1 | `CopyPToL1()` / `CopyPScaleToL1()` |
| Output/LSE 写回 | `CopyOutAttentionOut()` / `SoftmaxLseCopyOut()` |
| 输出初始化 | `MxStatic::InitOutput()` |

---

## 14. 参考资料

- Ascend C 静态 Tensor 编程：<https://www.hiascend.com/document/detail/zh/canncommercial/latest/programug/Ascendcopdevg/atlas_ascendc_10_00019.html>
- 本文设计基线：`b078d6a38b976a9ee145cfbb31bdda9ff81d0731`
- 静态 Tensor 首次改造提交：`443f48fb6`
- 当前去除冗余 PIPE_V barrier 后的提交：`a784dce56`

---

## 15. 最终检查清单

提交 MXFP8 静态 Tensor 相关修改前逐项确认：

- [ ] 没有修改普通 FP8 路径。
- [ ] 没有重新引入 `BufferManager`、动态 `TBuf/TQue` 或动态 Event ID。
- [ ] 所有新地址均为固定偏移，并有容量 `static_assert`。
- [ ] 共享 buffer 的 AIC/AIV 地址和大小完全一致。
- [ ] Event ID 记录了资源域、方向和 buffer，未使用 6/7。
- [ ] AIV1 没有手工 `+16`。
- [ ] 每个复用 token 均有 seed、热路径 Wait/Set 和 drain。
- [ ] 空任务、单 subLoop、双 subLoop、S1 tail、S2 tail 均不破坏计数。
- [ ] `SyncAll()` 之前没有产生供主循环消费的跨核 free token。
- [ ] BMM1/BMM2 free 前没有新增不必要的 `PipeBarrier`。
- [ ] PScale tail 仍使用固定 packed stride。
- [ ] 关键用例通过，并完成多次重复稳定性测试。
