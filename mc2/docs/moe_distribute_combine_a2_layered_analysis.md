# moe_distribute_combine_a2_layered.h 详细分析报告

*修订时间: 2026-04-21（补充"第X-Y行"格式行号引用）*

## 1. 文件概述

**文件路径**: `mc2/moe_distribute_combine_v2/op_kernel/moe_distribute_combine_a2_layered.h`
**命名空间**: `MoeDistributeCombineA2Impl`（第36行）
**核心类**: `MoeDistributeCombineA2Layered<ExpandXType, ExpandIdxType, ExpandXTransType>`（第64行定义）
**功能定位**: MoE (Mixture of Experts) 分布式 Combine 阶段的 A2 架构（多机）Kernel 实现，运行在昇腾 NPU 的 AIV（AI Vector）核心上。负责将 Dispatch 阶段分发到各卡的 token 经 Expert 计算后收集回来，执行加权求和并输出最终结果。

**核心职责：**
1. 将本地 token 通过 IPC 共享内存转发到机内目标 rank
2. 在机内对同一 token 的多专家结果进行加权求和
3. 将求和结果通过 RDMA 跨机发送到目标 rank
4. 在目标 rank 上再次汇总跨机结果，生成最终输出

**相关文件行号参考：**
- 类定义: 第64行 `MoeDistributeCombineA2Layered`，构造函数: 第98行
- 公有方法声明: `Init()` 第99-101行, `Process()` 第102行, `AIVRDMAPostSend()` 第103行
- 私有方法声明: 第106-116行
- 成员变量: 第118-194行
- `BuffInit()`: 第383-397行

---

## 2. 模板参数与类型系统

### 2.1 模板参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `ExpandXType` | 输入/输出数据类型 | `half`(FP16), `bfloat16_t`(BF16) |
| `ExpandIdxType` | 索引类型 | `int32_t` |
| `ExpandXTransType` | 传输/量化数据类型 | `int8_t`(动态量化) 或与 `ExpandXType` 相同(非量化) |

### 2.2 条件编译常量

```cpp
constexpr static bool DynamicQuant = std::is_same<ExpandXTransType, int8_t>::value;
```

当 `ExpandXTransType = int8_t` 时启用动态量化路径，token 数据在传输前量化为 int8 + per-group scale，接收后反量化还原。

### 2.3 OutputType 特化

| 数据类型 | 中间累加类型 | 说明 |
|---------|------------|------|
| `half` (FP16) | `half` | 直接累加 |
| `bfloat16_t` (BF16) | `float` | 提升精度到 FP32 累加，减少精度损失 |

---

## 3. 整体数据流与阶段划分

```
                     ┌───────────────┐
                     │   Process()   │
                     └───────┬───────┘
                             │
         ┌───────────────────┼────────────────────┐
         │                   │                    │
    阶段1: 机内IPC收集    阶段2: 跨机RDMA传输   阶段3: 跨机结果汇总
         │                   │                    │
    ┌────┴─────┐      ┌─────┴──────┐       ┌─────┴──────┐
    │ GM2IPC() │      │AlltoAll    │       │SumToServer │
    │ WaitIPC()│      │Server      │       │            │
    │SumToWindow│     │Dispatch()  │       │            │
    └──────────┘      │WaitDispatch│       └────────────┘
                      └────────────┘
```

### Process() 函数入口（第1050-1070行）

```
Process() 执行流程（第1050-1070行）:
├── GM2IPC()              ← 阶段1a: 将本地token写入IPC共享内存供同机其他rank读取（第415-478行）
├── WaitIPC()             ← 阶段1b: 等待同机所有rank完成IPC写入（第481-509行）
├── SumToWindow()         ← 阶段1c: 从IPC读取同机各rank的token，加权求和，结果写入RDMA发送窗口（第512-729行）
├── AlltoAllServerDispatch() ← 阶段2: 通过RDMA将汇总结果发送到目标服务器（第732-814行）
├── Preload()             ← 预加载跨机路由表(offsetOuter, countOuter)（第845-877行）
├── WaitDispatch()        ← 等待所有服务器的RDMA数据到达（第817-842行）
├── SumToServer()         ← 阶段3: 将来自各服务器的RDMA数据加权求和，写入最终输出（第880-1037行）
├── CopyPerformanceInfo() ← 写回性能调试信息（第1040-1047行）
└── hccl_.Finalize()      ← 清理HCCL资源
```

完整数据流:

```
                机内通信                    跨机通信
            ◄──────────────►          ◄──────────────────►

Rank 0 ──┐
Rank 1 ──┤     IPC共享内存           RDMA (RoCE V2)
Rank 2 ──┤   ──────────────►      ◄──────────────────────►
Rank 3 ──┤    GM2IPC 阶段           AlltoAllServerDispatch
Rank 4 ──┤    WaitIPC 阶段          WaitDispatch 阶段
Rank 5 ──┤    SumToWindow 阶段       SumToServer 阶段
Rank 6 ──┤
Rank 7 ──┘

expandX + scales
       │
       ▼
  ┌─────────┐     token = [data(H) + scale(32B)]
  │  GM2IPC  │──────────────────────────────────► IPC 共享内存
  └─────────┘                                     (每 rank 可见所有专家数据)
       │
       ▼
  ┌─────────┐
  │ WaitIPC  │  等待机内所有 rank 完成拷贝
  └─────────┘
       │
       ▼
  ┌──────────┐
  │SumToWindow│  读取 IPC 数据 → 加权求和 → 量化/非量化 → 写入发送窗口
  └──────────┘
       │
       ▼
  ┌───────────────────┐
  │AlltoAllServerDispatch│  发送窗口 ──RDMA──► 接收窗口（其他服务器）
  └───────────────────┘
       │
       ▼
  ┌────────────┐
  │WaitDispatch │  等待跨机 RDMA 完成
  └────────────┘
       │
       ▼
  ┌────────────┐
  │SumToServer  │  读取接收窗口 → 加权求和/反量化 → 最终输出
  └────────────┘
       │
       ▼
    expandOut
```

**核分工策略**（`aivNum_` 个核，典型值 32+）:

| 核范围 | 职责 |
|--------|------|
| `coreIdx_ < 32` | 机内 IPC Reduce (GM2IPC + WaitIPC + SumToWindow) |
| `coreIdx_ < 32 + serverNum_` | RDMA 跨机转发 (AlltoAllServerDispatch) |
| `coreIdx_ >= 32` | 阶段2/3期间空闲，SyncAll同步等待 |
| `coreIdx_ < 8` | Preload + SumToServer 跨机汇总 |

---

## 4. 核心常量（第66-81行）

| 常量 | 值 | 含义 |
|------|----|------|
| **`UB_ALIGN`**（第66行） | 32 | UB 内存对齐字节数 |
| **`BLOCK_SIZE`**（第67行） | 32 | 一个 datablock 的字节数 |
| **`B16_PER_BLOCK`**（第68行） | 16 | 每 block 可放的 16bit 元素数 |
| **`B32_PER_BLOCK`**（第69行） | 8 | 每 block 可放的 32bit 元素数 |
| **`SERVER_RANK_SIZE`**（第70行） | 8 | 每台服务器的 GPU 卡数（A2 = 8卡/机） |
| **`VEC_LEN`**（第71行） | 256 | 向量计算单元位宽（256 bit） |
| **`TBUF_SIZE`**（第72行） | 185 KB | UB临时缓冲区总大小 |
| **`IPC_REDUCE_USED_CORE_NUM`**（第73行） | 32 | 机内 IPC+Reduce 阶段使用的核数 |
| **`WEIGHT_VALUE_NUM`**（第74行） | 16 | token 尾部附加的 weight 值占用的空间 (16个 half/bf16 = 32字节) |
| **`GM2IPC_SYNC_FLAG`**（第75行） | 12345 | IPC同步完成标志基础值 |
| **`RDMA_TOKEN_ARRIVED_FLAG`**（第76行） | 123 | RDMA token到达标志 |
| **`RDMA_TOKEN_END_FLAG`**（第77行） | 321 | RDMA传输结束标志 |

---

## 5. 关键成员变量（第118-194行）

### 5.1 输入/输出 Global Tensor

| 变量 | 类型 | 说明 |
|------|------|------|
| `expandXGlobal_` | `GlobalTensor<ExpandXType>` | 输入: expandX (经Expert计算后的token数据) |
| `sendCountGlobal_` | `GlobalTensor<ExpandIdxType>` | 输入: sendCount (每个expert对各rank发送的token计数) |
| `expandScalesGlobal_` | `GlobalTensor<float>` | 输入: scales (token的加权系数) |
| `expandOutGlobal_` | `GlobalTensor<ExpandXType>` | 输出: 最终合并后的结果 |

### 5.2 IPC/RDMA 通信相关

| 变量 | 类型 | 用途 |
|------|------|------|
| `shareMemGlobal_` | `GlobalTensor<ExpandXType>` | IPC共享内存（读取同机其他rank写入的token数据） |
| `dstshareMemGlobal_` | `GlobalTensor<ExpandXType>` | IPC共享内存（写入本rank的token数据供其他rank读取） |
| `shareFlagGlobal_` | `GlobalTensor<uint64_t>` | IPC/RDMA同步标志 |
| `localOutWindow_` | `GlobalTensor<ExpandXType>` | 本rank的RDMA发送窗口（SumToWindow结果写入此处） |
| `localInWindow_` | `GlobalTensor<ExpandXTransType>` | 本rank的RDMA接收窗口（读取跨机RDMA到达的数据） |
| `localInScaleWindow_` | `GlobalTensor<ExpandXType>` | 接收窗口中的scale区域（量化模式） |
| `qp_info_` | `HcclAiRMAInfo*` | RDMA QP信息 |
| `readStateGlobal_` | `GlobalTensor<uint64_t>` | RDMA完成状态标志 |

### 5.3 路由元数据

| 变量 | 类型 | 用途 |
|------|------|------|
| `countInnerGlobal_` | `GlobalTensor<int16_t>` | 机内路由表（每个token被多少专家处理） |
| `offsetInnerGlobal_` | `GlobalTensor<int32_t>` | 机内路由表（token→全局偏移映射） |
| `countOuterGlobal_` | `GlobalTensor<int32_t>` | 跨机路由表（每个token被多少服务器处理） |
| `offsetOuterGlobal_` | `GlobalTensor<int32_t>` | 跨机路由表（token→跨机偏移映射） |

### 5.4 计算维度参数

| 变量 | 含义 |
|------|------|
| `axisBS_` | 本rank的batch size |
| `globalBs_` | 全局batch size = axisBS_ × worldSize_ |
| `axisH_` | 隐藏维度 (hidden dimension) |
| `axisK_` | TopK 路由数 |
| `aivNum_` | AIV 核心总数 |
| `worldSize_` | EP总rank数 |
| `rankId_` | 本rank ID |
| `coreIdx_` | 当前AIV核心ID |
| `serverNum_` | 服务器数 = worldSize / 8 |
| `moeExpertNum_` | 总MoE专家数 |
| `localMoeExpertNum_` | 每卡专家数 |
| `maxBs_` | 每rank最大token数 = globalBs / worldSize |
| `startRankId_` / `endRankId_` | 当前核负责的rank范围 |
| `sendRankNum_` | 当前核负责的rank数量 |
| `magicValue_` | Ping/Pong缓冲区选择标志 |

### 5.5 UB 缓冲区

| 变量 | 类型 | 说明 |
|------|------|------|
| `tBuf` | `TBuf<QuePosition::VECCALC>` | 主临时缓冲区（185 KB） |
| `rdmaInBuf_` / `rdmaInBuf2_` | `TBuf<TPosition::VECOUT>` | RDMA WQE 写入用的 32 字节缓冲 |
| `statusBuf_` | `TBuf<>` | 状态标志缓冲（worldSize_ × 32 字节） |
| `performanceInfoBuf_` | `TBuf<>` | 性能信息缓冲 |

---

## 6. 各阶段详细分析

### 6.1 Init() — 初始化（第285-380行）

**主要工作**:
1. 从 tilingData 提取计算维度参数 (bs, h, k, rankId, worldSize 等)
2. 根据数据类型(FP16/BF16)初始化量化粒度参数:
   - FP16: `SCALE_GRANU=16`, 即每16个元素共享一个scale
   - BF16: `SCALE_GRANU=8`, 即每8个元素共享一个scale
3. 初始化 HCCL 通信 (`hccl_.InitV2`)，获取 RDMA QP 信息
4. 通过 `addrInfo_.Init()` 计算所有 IPC/RDMA 地址偏移
5. 设置 Global Tensor 地址映射
6. `SplitCoreCal()`（第399-412行）计算每个核心负责的 rank 范围
7. 解析路由表地址（从 sendCount 的偏移区域解析 inner/outer 路由表）
8. 写入初始同步标志 `magicValue_` 到 RDMA flag 区

**量化参数计算**:
```
scaleNum = axisH / SCALE_GRANU          // scale的数量
scaleNumAlign = RoundUp(scaleNum, 32/sizeof(type))  // 对齐后的scale数量
repeatNum = CeilDiv(axisH, VEC_LEN/sizeof(type))    // BlockReduceMax的repeat次数
```

### 6.2 SplitCoreCal() — 核间任务划分（第399-412行）

将 `worldSize_` 个 rank 均匀分配给 `aivNum_` 个核:

采用 **均分 + 余数前分配** 策略:
- 每核处理 `sendRankNum_ = worldSize / aivNum` 个 rank
- 余数部分前 `remainderRankNum` 个核多处理一个 rank
- 每核负责 `[startRankId_, endRankId_)` 范围内的 rank

```
例: worldSize=64, aivNum=8
  core 0: rank [0, 9)   — 9 个 rank
  core 1: rank [9, 18)  — 9 个 rank
  ...
  core 7: rank [56, 64) — 8 个 rank
```

### 6.3 GM2IPC() — 数据写入IPC共享内存（第415-478行）

**目的**: 将本 rank 经 Expert 计算后的 token 数据写入 IPC 共享内存，供同机其他 rank 读取。

**流程**:
```
1. 从GM加载 sendCount 到UB (所有expert×worldSize的累计计数)
2. 从GM加载 expandScales 到UB (token的加权系数)
3. 双层循环: 遍历 localMoeExpert × [startRank, endRank):
   a. 计算 preCount (前一个expert+rank的累计token数)
   b. 计算 tokenNum = currentCount - preCount
   c. 对每个 token:
      - 从 expandXGlobal_ 读取 axisH_ 个数据
      - 从 expandScalesGlobal_ 读取 scale 值
      - 将 [data(H) + scale(32B)] 拼接为一个 token 结构
      - 写入 IPC 共享内存 dstshareMemGlobal_ (通过 addrInfo_.GetIpcDataAddr)
4. SyncAll 等待所有核完成
5. 前8个核分别写IPC同步标志 (GM2IPC_SYNC_FLAG + magicValue)
6. SyncAll 再次同步
```

**Token 在 IPC 中的结构**:
```
|<-- axisH × sizeof(ExpandXType) -->|<-- 32B (WEIGHT_VALUE_NUM × sizeof(ExpandXType)) -->|
|         token data                 |              scale value                            |
```

### 6.4 WaitIPC() — 等待IPC写入完成（第481-509行）

**流程**:
1. 前 8 个核(coreIdx < SERVER_RANK_SIZE=8)分别轮询来自同机 8 张卡的 IPC flag
2. 轮询条件: `flag >= GM2IPC_SYNC_FLAG + magicValue_`
3. 检测到后清零 flag，记录通信耗时
4. SyncAll 同步所有核

### 6.5 SumToWindow() — 机内加权求和并写入RDMA窗口（第512-729行）

**这是整个实现中最复杂的函数**，承担机内 Reduce + 可选动态量化 + RDMA窗口写入。

#### 整体流程:

```
1. 确定目标服务器 targetServerId 和目标 rank targetRankId
2. 从路由表加载机内 count/offset (countInnerGlobal_, offsetInnerGlobal_)
3. 筛选有效 token (count > 0)，按核交错分配避免链路冲突
4. 对每个 token 循环处理:
   a. 根据 offset 计算源 IPC 地址 (shareMemGlobal_)
   b. DataCopy 读取 token: [data(H) + weight(32B)]
   c. Cast 数据到 FP32
   d. 首次: Muls(scale × data)；后续: Axpy(scale × data + accumulated)
   e. 当一个 token 的所有 copyNum 都累加完:
      - 非量化路径: Cast FP32→ExpandXType, DataCopy 到 localOutWindow
      - FP16量化路径: Cast→Abs→BlockReduceMax→Muls(1/127)→Brcb→Div→Cast FP16→int8, 写入int8+scale
      - BF16量化路径: 类似但经过 FP32→BF16→int8 多级转换
   f. 写入 RDMA token到达标志
5. 写入 RDMA 结束标志
6. SyncAll
```

#### 量化路径对比:

| 步骤 | FP16 量化 | BF16 量化 | 非量化 |
|------|----------|----------|--------|
| 累加类型 | FP16 (Muls+Axpy替代为MulAddDst) | FP32 | FP32 (Muls+Add) |
| 求绝对值 | `Abs(sumHalf)` | `Abs(sumFloat)` | — |
| BlockReduceMax | FP16 粒度 g16 | FP32 粒度 g16 | — |
| Scale计算 | `max × 1/127` → FP16 | `max × 1/127` → FP32→BF16 | — |
| Scale广播 | `Brcb` 扩充到 H | `Brcb` 扩充到 H | — |
| 除法 | `data / scale` FP16 | `data / scale` FP32 | — |
| 量化 | FP16→int8 | FP32→FP16→int8 | — |
| 输出格式 | int8(H/2) + scale(scaleNumAlign) | int8(H/2) + scale(scaleNumAlign) | ExpandXType(H) |

**核间链路交错策略**:
```cpp
if ((i % coreNumPerServer) == (coreIdx_ % coreNumPerServer))
```
每 `coreNumPerServer` 个核交错处理 token，避免多个核访问同一 IPC 链路造成冲突。

### 6.6 AlltoAllServerDispatch() — 跨机RDMA传输（第732-814行）

**核分工**: `coreIdx_ ∈ [stepCoreNum, stepCoreNum + serverNum_)` 的核各负责一个目标服务器。

**流程**:
```
1. 确定目标服务器 checkServer 和目标 rank dstRankId
2. 计算每次RDMA传输的数据长度 copyLen_ (量化含scale，非量化不含)
3. 轮询 IPC token flag 等待 SumToWindow 完成:
   - RDMA_TOKEN_ARRIVED_FLAG: 有新token到达 → copySum++
   - RDMA_TOKEN_END_FLAG 或 copySum == maxBs_: 传输结束
4. 对到达的 token:
   - rankId != dstRankId: 调用 AIVRDMAPostSend() 发起 RDMA Write
   - rankId == dstRankId: 本机 rank，直接 GM→GM DataCopy（自环路径）
5. 所有token传输完成后，发送RDMA完成标志到远端
6. SyncAll
```

### 6.7 AIVRDMAPostSend() — RDMA Write 发送（第197-282行）

**底层 RoCEv2 RDMA 实现**，直接操作硬件 WQE (Work Queue Entry):

```
1. 根据 destRankId 计算目标 QP 的 Send Queue 上下文
2. 轮询 SQ 深度确保有空闲 WQE 槽位 (spin-loop)
3. 构造 WQE:
   - opcode = 0x3 (RDMA_WRITE)
   - owner_bit: 轮转标记
   - IBV_SEND_SIGNALED: 需要完成通知
   - 远端 rkey + VA (来自 memInfo table)
   - 源 lkey + VA
4. cacheWriteThrough 刷新 WQE 缓存
5. 更新 SQ head 指针
6. 写 DoorBell 触发硬件发送
```

### 6.8 Preload() — 预加载跨机路由表（第845-877行）

仅前 8 个核执行:
1. 将 axisBS_ 均匀分配给 8 个核（含余数处理）
2. 从 GM 加载 `offsetOuterGlobal_` 和 `countOuterGlobal_` 到 UB
3. 这些路由表将在 `SumToServer()` 中使用

### 6.9 WaitDispatch() — 等待RDMA数据到达（第817-842行）

**核分工**: `coreIdx_ < serverNum_` 且非本服务器的核执行:
1. 轮询 `GetLocalRecvBuffFlagAddr(srcRankId)` 处的 flag
2. 等待 `flag == sumTarget_` (即 magicValue_)
3. 记录通信耗时

### 6.10 SumToServer() — 跨机结果汇总（第880-1037行）

仅前 8 个核执行，将来自各服务器的 RDMA 数据累加并输出最终结果。

**三种路径**:

#### 非量化路径（FP16/BF16直接传输）
```
对每个token:
1. Duplicate(sumFloat, 0.0)  // 清零累加器
2. 循环 copyNum 次:
   a. 计算跨机数据在接收窗口中的偏移 offsetOnIpc
   b. 确定源 rank: srcServerId × SERVER_RANK_SIZE + rankId % SERVER_RANK_SIZE
   c. DataCopy 从 localInWindow_ 读取数据
   d. Cast ExpandXType → FP32
   e. Add 到累加器
3. Cast FP32 → ExpandXType
4. DataCopy 写入 expandOutGlobal_ (最终输出)
```

#### FP16 + int8 量化路径
```
对每个token:
1. Duplicate(sumFp16, 0)  // 清零FP16累加器
2. 循环 copyNum 次:
   a. 读取 int8 数据和 BF16 scale
   b. Cast int8 → FP16
   c. Brcb 将 scale 扩充到 H 个
   d. MulAddDst: FP16 × scale 累加
3. DataCopy 写入 expandOutGlobal_
```

#### BF16 + int8 量化路径
```
对每个token:
1. Duplicate(sumFloat, 0.0)  // 清零FP32累加器
2. 循环 copyNum 次:
   a. 读取 int8 数据和 BF16 scale
   b. Cast int8 → FP16 → FP32 (数据)
   c. Cast BF16 → FP32 (scale)
   d. Brcb 将 FP32 scale 扩充到 H 个
   e. MulAddDst: FP32 × scale 累加
3. Cast FP32 → BF16
4. DataCopy 写入 expandOutGlobal_
```

**量化模式下 SumToServer 的数据还原**:
```
int8 数据 × scale = 近似原始 FP16/BF16 值
多机累加: sum = Σ(data_int8_i × scale_i)
```

### 6.11 CopyPerformanceInfo()（第1040-1047行）

使用原子 Add 将各核心记录的性能数据写回 GM，供 host 端读取。

---

## 7. 通信模型与地址布局

### 7.1 三层通信架构

```
┌─────────────────────────────────────────────────────┐
│                    服务器 0                          │
│  ┌──────┐ ┌──────┐ ┌──────┐       ┌──────┐         │
│  │Rank 0│ │Rank 1│ │Rank 2│  ...  │Rank 7│         │
│  └──┬───┘ └──┬───┘ └──┬───┘       └──┬───┘         │
│     │  IPC    │  IPC    │    IPC      │             │
│     └─────────┴─────────┴────────────┘              │
│              共享内存 (ShareMem)                      │
└─────────────────────┬───────────────────────────────┘
                      │ RDMA (RoCEv2)
┌─────────────────────┴───────────────────────────────┐
│                    服务器 1                          │
│  ┌──────┐ ┌──────┐       ┌──────┐                  │
│  │Rank 8│ │Rank 9│  ...  │Rank15│                  │
│  └──────┘ └──────┘       └──────┘                  │
└─────────────────────────────────────────────────────┘
```

### 7.2 地址计算 (addrInfo_)（初始化于 `Init()` 第285-380行中的 `addrInfo_.Init()` 调用）

由 `MoeDistributeA2CombineAddrInfo` 提供，基于 HCCL Window 内存布局:

**IPC 地址**:
- `GetIpcDataAddr(dstRank, expertId, srcRank)` → 同机 dstRank 的 expertId 对应 IPC 数据区中 srcRank 的偏移位置

**RDMA 发送地址**:
- `GetLocalSendBuffDataAddr(dstRank)` → 本 rank 的 WindowOut 中，目标为 dstRank 所在服务器的数据区
- `GetLocalSendBuffFlagAddr()` → 本 rank 的 WindowOut 中，RDMA 标志区

**RDMA 接收地址**:
- `GetLocalRecvBuffDataAddr(srcRank)` → 本 rank 的 WindowIn 中，来自 srcRank 所在服务器的数据区
- `GetLocalRecvBuffFlagAddr(srcRank)` → 对应标志位

**IPC Token Flag**:
- `GetIpcTokenFlagAddr(serverId)` → IPC 区域中用于 SumToWindow 与 AlltoAllServerDispatch 之间同步的 per-server token 标志

### 7.3 路由表结构

路由表从 `sendCount` 的偏移区域解析:

```
sendCount 内存布局:
┌──────────────────────────────────────────────────┐
│ [0, worldSize × localMoeExpertNum)               │ 原始 sendCount
├──────────────────────────────────────────────────┤
│ [send_counts_inner_offset, ...)                   │ countInner (int16_t, 机内token计数)
├──────────────────────────────────────────────────┤
│ [offset_inner_offset, ...)                        │ offsetInner (int32_t, 机内token偏移)
├──────────────────────────────────────────────────┤
│ [send_counts_outer_offset, ...)                   │ countOuter (int32_t, 跨机token计数)
├──────────────────────────────────────────────────┤
│ [offset_outer_offset, ...)                        │ offsetOuter (int32_t, 跨机token偏移)
└──────────────────────────────────────────────────┘
```

---

## 8. 同步机制总结

| 同步方式 | 使用场景 | 标志/方法 | 参与者 | 代码位置 |
|----------|----------|-----------|--------|----------|
| `SyncAll<true>()` | 阶段间全局同步 | 屏障同步 | 所有 aivNum 个核 | 各阶段末尾 |
| `PipeBarrier<PIPE_ALL>` | 指令流水线同步 | 确保前后指令执行顺序 | 当前核 | 贯穿各阶段 |
| `SyncFunc<HardEvent::*>` | 数据搬运同步 | MTE2/MTE3/V 之间的事件同步 | 当前核 | 贯穿各阶段 |
| IPC Flag 轮询 | GM2IPC 完成通知 | `GM2IPC_SYNC_FLAG + magicValue` | 8 个核轮询 8 个 IPC flag | `WaitIPC()` 第481-509行 |
| RDMA Token Flag 轮询 | token 级流控 | `RDMA_TOKEN_ARRIVED/END_FLAG + magic` | AlltoAllServerDispatch 核轮询 | `AlltoAllServerDispatch()` 第732-814行 |
| RDMA Status 轮询 | 跨机同步 | `magicValue_` 写入 `readStateGlobal_` | serverNum_ 个核轮询 | `WaitDispatch()` 第817-842行 |

---

## 9. 数据类型与精度分析

### 累加精度选择

| 数据类型 | 机内Reduce累加类型 | 跨机Reduce累加类型 | 最终输出类型 |
|---------|-------------------|-------------------|------------|
| FP16 (非量化) | FP32 (Muls+Add) | FP32 (Add) | FP16 |
| BF16 (非量化) | FP32 (Muls+Add) | FP32 (Add) | BF16 |
| FP16 + int8量化 | FP16 (MulAddDst) | FP16 (MulAddDst) | FP16 |
| BF16 + int8量化 | FP32 (MulAddDst) | FP32 (MulAddDst) | BF16 |

### 动态量化流程 (以BF16为例)（参见 `SumToWindow` 第512-729行中的 BF16 量化路径）

```
累加完成后:
sumFloat(H个FP32值)
    → Abs → BlockReduceMax (每8个取最大值) → 得到 scaleNum 个最大值
    → × 1/127 → 得到 FP32 scale
    → Brcb 扩充到 H 个
    → sumFloat / scaleBroadcast → 归一化后的FP32值
    → Cast FP32→BF16 (scale)
    → Cast FP32→FP16 (数据)
    → Cast FP16→int8 (数据)
    输出: int8数据(H/2字节) + BF16 scale(scaleNumAlign × 2字节)
```

---

## 10. Ping/Pong 双缓冲机制

通过 `magicValue_` 的低位选择当前使用的缓冲区（Ping 或 Pong），支持 Dispatch/Combine 交替执行时复用同一块 IPC/RDMA 内存：

- 每次 Dispatch+Combine 轮次切换 `magicValue_`
- IPC flag、RDMA token flag、RDMA status 均携带 `magicValue_`
- 不同轮次的标志互不干扰，避免残留标志导致误判

---

## 11. UB 内存管理

使用 `TBuf<QuePosition::VECCALC>` (185KB, 第72行 `TBUF_SIZE`) 作为统一临时缓冲区，通过 `GetWithOffset` 手动管理偏移（缓冲区初始化参见 `BuffInit()` 第383-397行）:

### SumToWindow() 的 UB 布局

```
Offset 0 (TBUF_TEMP_OFFSET):
├── sendCountLocal      : RoundUp(moeExpertNum_, 8) × 4B
├── expandScalesLocal   : RoundUp(maxBs_, 32) × 4B
├── dataInLocal         : (axisH + 16) × sizeof(ExpandXType)
├── castInFloatLocal    : RoundUp(axisH×4, 256) bytes  (FP32/BF16量化复用)
├── sumFloatLocal       : axisH × 4B
├── [FP16量化路径]:
│   ├── sumHalfLocal    : axisH × 2B
│   ├── reduceMaxOut    : scaleNum × 2B
│   └── absScaleTensor  : RoundUp(axisH×2, 256) bytes (Brcb扩充用)
├── [BF16量化路径]:
│   ├── halfLocal       : axisH × 2B (中间转换)
│   └── reduceMaxFloat  : scaleNum × 4B
└── [end]
```

### SumToServer() 的 UB 布局

```
Offset 0 (Preload 预加载):
├── offsetReduceLocal_  : RoundUp(axisBS × serverNum, 32/4) × 4B
└── countReduceLocal_   : RoundUp(axisBS, 32/4) × 4B

Offset (baseBuffOffset):
├── [非量化]: sumFloatLocal + dataIn + castFp32
├── [FP16量化]: sumFp16 + dataInt8 + scaleData + castFp16 + scaleDup
└── [BF16量化]: sumFloat + dataInt8 + scaleBf16 + castHalf + castFp32 + castFp32scale + castFp32ScaleBrcb
```

**内存复用策略**: FP16 和 BF16 路径的 UB 从相同起始偏移 (`fp16baseBuffOffset` / `bfBaseBuffOffset`) 开始分配，因为它们通过 `if constexpr` 在编译期互斥。

---

## 12. 性能优化要点

1. **核心流水并行**: 32 核机内求和 + serverNum_ 核跨机 RDMA 分工并行（参见 `Process()` 第1050-1070行的核分工逻辑）
2. **核间链路交错**: `SumToWindow`（第512-729行）中按 `coreIdx_ % coreNumPerServer` 交错分配 token，避免多个核心竞争同一条 RDMA 链路
3. **量化减半传输量**: int8 量化将跨机 RDMA 数据量从 `H x 2B` 减少到 `H/2 x 1B + scale`，带宽减少 50% 以上（参见 `SumToWindow` 第512-729行量化路径）
4. **Ping/Pong 双缓冲**: 通过 `magicValue_` 低位选择缓冲区，支持 Dispatch/Combine 交替执行时复用内存
5. **流式 RDMA**: `AlltoAllServerDispatch`（第732-814行）中逐 token 检测并立即发送，无需等待全部 token 完成
6. **自环优化**: 同机 rank 的 RDMA 直接 DataCopy，不走 RDMA 网络（参见 `AlltoAllServerDispatch` 第732-814行）
7. **预加载分发表**: `Preload`（第845-877行）在 `WaitDispatch`（第817-842行）期间提前加载跨机偏移表，隐藏内存延迟
8. **BlockReduceMax 粒度**: FP16 按 16 个元素、BF16 按 8 个元素分组取最大值作为量化 scale（参见 `SumToWindow` 第512-729行）
9. **Brcb 广播**: 将少量 scale 值高效广播到 H 个位置，避免逐元素计算

---

## 13. 依赖关系图

```
moe_distribute_combine_a2_layered.h
├── kernel_operator.h / kernel_basic_intf.h  (CANN SDK)
├── kernel_tiling/kernel_tiling.h            (构建生成: Mc2InitTiling, Mc2CcTiling)
├── moe_distribute_combine_tiling.h          (Combine 专用 tiling 结构: MoeDistributeCombineA2Info 第21-41行, MoeDistributeCombineA2TilingData 第43-47行)
├── moe_distribute_base.h                    (HCCL 基础: HcclA2CombineOpParam, RDMA 结构)
├── mc2_kernel_utils.h                       (SyncFunc 工具)
└── moe_distribute_a2_base.h                 (A2 地址计算: MoeDistributeA2CombineAddrInfo)
```

**关联文件行号参考：**
- 非分层变体 `moe_distribute_combine_a2.h`: 类 `MoeDistributeCombineA2`（第92行），`Init()`（第223-300行），`Process()`（第671-682行）
- AiCPU 变体 `moe_distribute_combine_a2_layered_aicpu.h`: 类 `MoeDistributeCombineA2LayeredAicpu`（第37行），`Init()`（第207-321行），`Process()`（第709-721行）
- Tiling 结构 `moe_distribute_combine_tiling.h`: `MoeDistributeCombineA2Info`（第21-41行），`MoeDistributeCombineA2TilingData`（第43-47行）
- Kernel 入口 `moe_distribute_combine_v2.cpp`: 入口函数（第62行），A2 MTE 非量化实例化（第88-94行），A2 AICPU 非量化路径（第95-110行），A2 AICPU int8 量化路径（第111-123行）

---

## 14. 注意事项

1. **UB 空间约束**: 185 KB 临时 buffer（第72行 `TBUF_SIZE`）需容纳所有中间变量，`SumToWindow`（第512-729行）是空间压力最大的阶段。所有路径（量化/非量化 x FP16/BF16）的 UB 分配必须在 185KB 以内。
2. **量化精度**: BF16 路径在 FP32 域完成累加后再量化，比 FP16 直接累加精度更高。
3. **SyncAll 木桶效应**: 多处使用 `SyncAll<true>()` 作为全局同步屏障，所有核必须到达一致状态后才继续——最慢的核决定整体进度。
4. **轮询开销**: `WaitIPC`（第481-509行）和 `WaitDispatch`（第817-842行）使用忙等轮询（while true + DataCopy），在高负载下会占用 AIV 计算资源。
5. **死锁风险**: `AIVRDMAPostSend`（第197-282行）中的 SQ 溢出 spin-loop 依赖 host 超时机制保活，若无超时则可能死锁。
6. **核心数硬编码假设**: 硬编码 `IPC_REDUCE_USED_CORE_NUM=32`（第73行）做机内归约、`SERVER_RANK_SIZE=8`（第70行）做跨机汇总，需确保硬件配置匹配。
7. **路由表偏移耦合**: inner/outer 路由表的偏移计算依赖 sendCount 内存的精确布局，与 Dispatch 阶段的输出格式强耦合——`countInner`/`offsetInner`/`countOuter`/`offsetOuter` 由 Dispatch 阶段生成，必须保证一致性。参见 `Init()`（第285-380行）中的路由表解析逻辑。
8. **Magic Value 机制**: 通过 `magicValue_` 区分不同轮次的同步标志，避免残留标志导致误判。Magic 从 `addrInfo_.GetMagicValue()` 获取，基于 ping-pong 缓冲区选择。
