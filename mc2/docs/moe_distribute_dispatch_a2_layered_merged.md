# MoeDistributeDispatchA2Layered 详细分析报告

## 1. 文件概述

**文件路径**: `mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_a2_layered.h`

**功能定位**: 在 Ascend A2 (Atlas 910B) 硬件上实现 MoE (Mixture-of-Experts) 模型的 Token 分发（Dispatch）操作。该算子负责将输入 Token 按照路由表（expertIds）分发到对应的专家所在的设备上，并生成后续 Combine 阶段所需的归约索引表，是 MoE 推理/训练流程中的核心通信算子。

**核心设计思想**: 采用**分层通信架构**（Layered/Hierarchical），将多机通信分解为：
- **机间通信**（Inter-Server）：通过 RDMA（RoCEv2）完成跨服务器 Token 传输
- **机内通信**（Intra-Server）：通过 IPC（共享内存窗口映射）完成同服务器内跨卡 Token 传输

这种分层设计避免了所有 Rank 之间直接 RDMA 通信带来的资源膨胀问题（N^2 连接），将通信复杂度从 O(worldSize^2) 降低到 O(serverNum * SERVER_RANK_SIZE)。

**命名空间**: `MoeDistributeDispatchA2Impl`

**模板参数**:
| 参数 | 说明 |
|------|------|
| `XType` | 输入 Token 数据类型，通常为 `half`（FP16）或 `bfloat16_t`（BF16） |
| `ExpandXOutType` | 输出 Token 数据类型，量化模式下为 `int8_t`，非量化模式下与 `XType` 相同 |
| `StaticQuant` | 是否启用静态量化（编译期常量，用于优化分支消除） |
| `DynamicQuant` | 是否启用动态量化（运行时逐 Token 计算 scale 并量化为 INT8） |
| `IsSmoothScaleExist` | 是否存在 SmoothQuant 的平滑缩放因子 |

**关键常量**:
| 常量 | 值 | 含义 |
|-----|-----|------|
| `SERVER_RANK_SIZE` | 8 | 每台服务器的 NPU 卡数（A2 架构固定为 8） |
| `TBUF_SIZE` | 190 KB | UB（Unified Buffer）临时缓冲区大小 |
| `IPC_BUFF_ALIGN` | 512 | IPC 缓冲区对齐粒度 |
| `FLAG_SIZE` | 64 Byte | workspace 中 sendServerInfo 的存储槽大小（含 padding） |
| `flagLenInStruct_` | 32 Byte | TokenStruct 中 flag 字段的实际大小 |
| `SHOULD_SEND_FLAG_VALUE` | 0x0f0f0f0f | Token 有效标志（加上 magicVal 防止陈旧数据误判） |
| `END_OF_WRITE_FLAG_VALUE` | 0xffffffff | 发送结束标志 |

---

## 2. 整体架构与数据流

### 2.1 硬件拓扑假设

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           集群 (Cluster)                                │
│  ┌──────────────────┐  ┌──────────────────┐       ┌──────────────────┐ │
│  │    Server 0      │  │    Server 1      │  ...  │    Server N      │ │
│  │ ┌──────────────┐ │  │ ┌──────────────┐ │       │ ┌──────────────┐ │ │
│  │ │  Rank 0      │ │  │ │  Rank 8      │ │       │ │  Rank N*8    │ │ │
│  │ │  Rank 1      │ │  │ │  Rank 9      │ │       │ │  ...         │ │ │
│  │ │  ...         │ │  │ │  ...         │ │       │ │              │ │ │
│  │ │  Rank 7      │ │  │ │  Rank 15     │ │       │ │              │ │ │
│  │ └──────────────┘ │  │ └──────────────┘ │       │ └──────────────┘ │ │
│  └────────┬─────────┘  └────────┬─────────┘       └──────────────────┘ │
│           │ RDMA (RoCEv2)       │                                      │
│           └─────────────────────┘                                      │
│    每Server 8个Rank (SERVER_RANK_SIZE=8)                                │
│    机内通过 IPC（共享内存窗口映射）通信                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

- `SERVER_RANK_SIZE = 8`: 每台服务器8个Rank
- `serverNum = worldSize / 8`: 集群中服务器数量
- `localMoeExpertNum = moeExpertNum / worldSize`: 每个Rank拥有的MoE专家数
- `moeExpertNumInServer = 8 * localMoeExpertNum`: 每台Server内的专家总数

### 2.2 数据流全貌

```
                    ┌──────────────┐
                    │   输入数据    │
                    │  x, expertIds│
                    │  scales, ... │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ ReorderTokens│  阶段1: Token重组+打包
                    │   (全核并行)  │  将输入数据按TokenStruct打包到RDMA发送缓冲区
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  SyncAll     │  全局同步
                    └──────┬───────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
    ┌───────▼──────┐ ┌─────▼─────┐ ┌─────▼──────┐
    │SendDataToServer│ │CreateOuter│ │   Win2Ipc  │
    │ (跨机RDMA发送) │ │ReduceInfo │ │ (机内IPC转发)│
    │  核0~N-1执行  │ │ 核N执行   │ │ 核N+1执行   │
    └───────┬──────┘ └─────┬─────┘ └─────┬──────┘
            │              │              │
            │      ┌───────▼───────┐     │
            │      │CreateInner    │     │
            │      │ReduceInfo     │     │
            │      │(机内索引生成)  │     │
            │      └───────┬───────┘     │
            │              │              │
            └──────────────┼──────────────┘
                           │
                    ┌──────▼───────┐
                    │ SetIpcFlag   │  通知对端IPC数据就绪
                    │ WaitIpcFlag  │  等待本端IPC数据就绪
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Ipc2Out    │  阶段3: IPC缓冲区→输出
                    │  (全核并行)   │  写出expandXOut、weights、dynamicScales
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   CleanUp    │  清理缓冲区状态
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   输出数据    │
                    │expandXOut,   │
                    │expandScales, │
                    │dynamicScales │
                    │expertTokenNums│
                    │epRecvCounts   │
                    └──────────────┘
```

**核角色分配** (假设 `aivNum_` 个AIV核):
| AIV ID | 角色 |
|--------|------|
| 0 ~ serverNum-1 | 跨机RDMA发送到对应Server + Inner索引生成 |
| serverNum | Outer索引生成（仅1个核） |
| serverNum+1 ~ end | Win2Ipc机内IPC转发 |

示例（假设 `serverNum=2, aivNum=24`）:
```
┌────┬────┬────┬────────────────────────────────────────┐
│ 核0 │ 核1 │ 核2 │ 核3 ~ 核23                              │
│发送 │发送 │Outer│ Win2Ipc (机内IPC转发)                     │
│到S0 │到S1 │索引 │  核3-12: 处理Server 0 的数据              │
│+Inner│+Inner│生成│  核13-22: 处理Server 1 的数据            │
│到S0 │到S1 │    │  核23: 空闲 (如有余数)                    │
└────┴────┴────┴────────────────────────────────────────┘
```

---

## 3. 核心数据结构

### 3.1 TokenStruct（Token传输单元）

每个 Token 在通信过程中被封装为一个固定大小的结构体，这是RDMA传输中的基本数据单元：

```
偏移:    0          flagLen      +tokenLen      +expLen        +weightLen     +scaleLen
         │(32B)      │(H*dtype)   │(alignK*U32)  │(alignK*U32)  │(32B)        │
         ├───────────┼────────────┼──────────────┼──────────────┼─────────────┤
         │   Flag    │ Token数据  │  ExpertIds   │  Weights     │ Scale(量化) │
         │(magic+    │            │(对齐到32B)    │(对齐到32B)    │ (仅量化时)  │
         │0x0f0f0f0f)│            │              │              │             │
         └───────────┴────────────┴──────────────┴──────────────┴─────────────┘
         ←──────────────── tokenStructLen_ ──────────────────────────────────→
```

| 字段 | 偏移 | 大小 | 说明 |
|-----|------|------|------|
| `flag` | 0 | 32B (`flagLenInStruct_`) | 到达标志，值为 `SHOULD_SEND_FLAG_VALUE + magicVal_`，包含 `magicVal` 防止不同轮次数据混淆 |
| `token` | 32B | `H * sizeof(ExpandXOutType)` | Token 的实际数据（量化后为 INT8） |
| `expIds` | 32B + tokenLen | `alignK_ * sizeof(uint32)` | 该 Token 被路由到的 K 个专家 ID。使用 `alignK_`（K 向上取整到 8 的倍数，即 32B 对齐）而非实际 `axisK_`，以保证 UB 操作的对齐要求 |
| `weights` | expIds 之后 | `alignK_ * sizeof(uint32)` | 该 Token 对应 K 个专家的路由权重 |
| `scale` | weights 之后 | 32B (`scaleLenInStruct_`) | 动态量化时的反量化 scale（仅动态量化模式下有效） |

**Flag 保序机制（关键硬件约束）**:

Flag 必须位于 TokenStruct 最前端，且 Token 之间不连续存储（通过 `tokenGapInStruct_` 间隔）。原因：两条 ROCE 消息通过 PCIe 总线写入 GM 时，只有**第二条消息的第一个分片**的写操作与上一条消息保证保序，其余分片可能乱序到达。因此通过下一个 Token 的 flag 到到来校验上一个 Token 的数据是否完整。

`tokenGapInStruct_` 和 `infoGapInStruct_` 的计算：
```cpp
tokenGapInStruct_ = (tokenStructLen_ - tokenLenInStruct_) / UB_32B_ALIGN;  // DataCopyPad 的 dstGap
infoGapInStruct_ = (tokenStructLen_ - expLenInStruct_) / UB_32B_ALIGN;     // DataCopyPad 的 dstGap
```

### 3.2 HCCL 缓冲区布局

基于 `MoeDistributeA2Base::MoeDistributeA2DispatchAddrInfo` 管理的缓冲区：

**WindowIn（接收侧）**:
```
| Arrived Flag (512B * serverNum) | Inner Flag (512B * serverNum) | Inner Data | RDMA Data (按server划分) | ... | IPC Data (按expert和srcRank划分) | IPC Sync Flag | IPC TokenCnt |
```

**WindowOut（发送侧）**:
```
| Flag (32B) | Inner Data (按dstServer划分) | RDMA Data (连续存放所有Token) |
```

### 3.3 双缓冲 Ping/Pong 机制

通过 `magicValue_ & 0x1` 选择当前使用的缓冲区（Ping 或 Pong）。每次算子调用时 `magicValue_` 递增（存储在共享内存中），自动切换 Ping/Pong，避免前后两次调用之间的数据竞争。

### 3.4 epRecvCounts 输出布局

`epRecvCounts` 输出缓冲区按以下偏移存储 Combine 阶段的索引信息：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ epRecvCounts 缓冲区                                                          │
├───────────────────────────┬────────────────────────┬────────────────────────┤
│ ExpertTokenNums           │ InnerCnt + InnerOffset │ OuterCnt + OuterOffset │
│ (专家Token计数表)          │ (机内归约索引)          │ (机间归约索引)          │
│ 偏移: 0                   │ 偏移: cntOff           │ 偏移: outerCntOff      │
└───────────────────────────┴────────────────────────┴────────────────────────┘
```

各区域精确布局（带类型和维度）：

| 区域 | 类型 | 维度 | 偏移计算 |
|------|------|------|---------|
| ExpertTokenNums | int32 | `localExpNum * serverNum * SERVER_RANK_SIZE` | 0 |
| InnerCnt | int16 | `globalBs * serverNum` | `combineInnerCntOffset` |
| InnerOffset | int32 | `globalBs * axisK * serverNum` | `combineInnerCntIndexOffset` |
| OuterCnt | int32 | `axisBS` | `combineOuterCntOffset` |
| OuterOffset | int32 | `axisBS * serverNum` | `combineOuterCntIndexOffset` |

偏移量计算：
```cpp
combineInnerCntOffset     = localMoeExpertNum * serverNum * SERVER_RANK_SIZE * sizeof(int32_t);
combineInnerCntIndexOffset = combineInnerCntOffset + globalBs * serverNum * sizeof(int16_t);
combineOuterCntOffset      = combineInnerCntIndexOffset + globalBs * axisK * serverNum * sizeof(int32_t);
combineOuterCntIndexOffset = combineOuterCntOffset + axisBS * sizeof(int32_t);
```

---

## 4. 核心方法详解

### 4.1 Init() — 初始化（第197-318行）

**功能**: 从 Tiling 数据和 HCCL 上下文中提取参数，初始化 GlobalTensor、UB 缓冲区。

**关键流程**:
1. **Tiling 数据提取**: 解析 `MoeDistributeDispatchA2TilingData`，获取 BS、H、K、worldSize、expertNum 等
2. **HCCL 初始化**: 调用 `hccl_.InitV2()` 和 `SetCcTilingV2()` 初始化通信上下文
3. **Server 拓扑计算**: `serverId = rankId / 8`，`serverNum = worldSize / 8`
4. **地址信息初始化**: `addrInfo_.Init()` 设置 RDMA/IPC 地址映射，包括 Ping/Pong 双缓冲选择
5. **TokenStruct 尺寸计算**: 根据 H、K、数据类型计算各字段长度和对齐
6. **GlobalTensor 绑定**: 将输入/输出 GM 地址绑定到 GlobalTensor
7. **UB 缓冲区分配**:
   - `statusBuf_` (192B = 3 * FLAG_SIZE): 用于 `GetArrivedTokenInfo` 中的 3 个 flag 轮询（ArrivedFlag, CurrentTokenFlag, NextTokenFlag）
   - `rdmaInBuf_` / `rdmaInBuf2_` (各32B): RDMA WQE 写入辅助
   - `tBuf` (190KB): 主工作缓冲区
   - `performanceInfoBuf_`: 性能统计缓冲（可选）
8. **expertIds I16 预转换**: 在 tBuf 末尾预留空间存放 `int16_t` 版本的 expertIds
9. **MagicValue 初始化**: 从共享内存读取并递增，用于区分不同调用轮次。`aivId_ == 0` 的核同时负责初始化 RDMA 发送结束标志为 `END_OF_WRITE_FLAG_VALUE`

**设计要点**:
- Flag 置于 TokenStruct 最前端，利用硬件 ROCE 消息首分片保序特性进行正确性校验
- 使用 `magicVal_` 区分不同轮次的调用，防止残留数据干扰

### 4.2 Process() — 主流程入口（第1324-1354行）

```
Process() 流程:
┌─────────────────────┐
│ ReorderTokens()     │  Step 1: 所有AIV核并行重组Token
│ PipeBarrier         │
│ SyncAll             │  Step 2: 全局同步确保重组完成
│ ┌─────────────────┐ │
│ │核0~serverNum-1: │ │  Step 3a: 跨机RDMA发送 + 机内索引生成
│ │  SendDataToServer│ │         (跳过本Server)
│ │  CreateInner     │ │
│ │  ReduceInfo      │ │
│ ├─────────────────┤ │
│ │核serverNum:     │ │  Step 3b: 机间索引生成
│ │  CreateOuter     │ │
│ │  ReduceInfo      │ │
│ ├─────────────────┤ │
│ │核serverNum+1+:  │ │  Step 3c: 机内IPC转发
│ │  Win2Ipc()       │ │
│ └─────────────────┘ │
│ PipeBarrier + SyncAll│
│ SetIpcFlag          │  Step 4: 通知IPC数据就绪
│ WaitIpcFlag         │  Step 5: 等待对端IPC就绪
│ PipeBarrier + SyncAll│
│ Ipc2Out()           │  Step 6: IPC→最终输出
│ CleanUp()           │  Step 7: 清理缓冲区
│ SyncAll             │
│ CopyPerformanceInfo │  Step 8: 写出性能信息
│ hccl_.Finalize()    │
└─────────────────────┘
```

### 4.3 ReorderTokens() — Token 重组与打包（第686-798行）

**功能**: 所有 AIV 核并行将输入 Token 数据按 TokenStruct 格式打包到 RDMA 发送缓冲区。

**流程**:
1. **分核策略**: 将 `axisBS_` 个 Token 均匀分配到 `aivNum_` 个核，前 `remainderTokenNum` 个核多处理 1 个 Token
2. **ExpertIds 搬入**: 一次性将全部 `BS*K` 个 expertId 从 GM 搬入 UB，转换为 `int16_t` 节省空间
3. **分批处理**: 每批处理 `maxTokenNumInUB` 个 Token（受 UB 大小限制）
4. **逐 Token 打包**:
   - 动态量化: 先拷贝原始 Token，再通过 `QuantProcess()` 量化
   - 非量化: 直接带间隔拷贝到 TokenStruct 的 token 字段
   - 拷贝 ExpertIds 和 Weights 到 TokenStruct 对应字段
   - 设置 Flag 值 = `SHOULD_SEND_FLAG_VALUE + magicVal_`
5. **目标 Server 位图**: 根据 expertIds 计算每个 Token 需要发送到哪些 Server，记录到 `tokenAddrFlagStructGM_`
   ```cpp
   for (uint32_t i = 0; i < axisK_; i++) {
       uint32_t expertId = expertIdsI16Tensor_(startExpId + i);
       uint32_t dstServerId = expertId / moeExpertNumInServer_;
       sendServerInfo |= (1UL << dstServerId);
   }
   ```
6. **输出到 WinOut**: 将打包好的 TokenStruct 批量写入 RDMA 发送缓冲区

**动态量化路径** (`DynamicQuant=true`):
- 额外使用 `axisH * sizeof(float)` 空间存放 float 中间结果
- `QuantProcess()` 执行: float 转换 → ReduceMax/ReduceMin → 计算 dynamicScale → 缩放 → Cast 到 int8

### 4.4 QuantProcess() — 动态量化（第801-855行）

**功能**: 在 `DynamicQuant = true` 时，对每个 Token 进行动态量化。

**量化链路**: `XType → float32 → (scale) → int32 → half → ExpandXOutType(int8)`

**算法**:
1. 将 XType（FP16/BF16）Cast 为 float32
2. 对每个 Token：
   - **ReduceMax + ReduceMin + Abs + ReduceMax**: 找到绝对值最大值
   - **计算 scale**: `dynamicScale = 127.0 / maxAbsVal`
   - **缩放**: `Muls(tokenCastLt, tokenCastLt, dynamicScale, axisH_)`
   - **量化**: float32 → int32（四舍五入）→ half → ExpandXOutType（截断）
   - **记录 scale**: 将 `1.0 / dynamicScale`（反量化因子）写入 TokenStruct 的 scale 字段

### 4.5 SendDataToServer() — 跨机 RDMA 发送（第858-885行）

**功能**: 将本 Server 需要发送给指定目标 Server 的 Token 通过 RDMA 逐一发送。

**流程**:
1. 计算 `dstRankId`（目标 Server 上与本卡同位置的 Rank）
2. 从 `tokenAddrFlagStructGM_` 读取每个 Token 的 `sendServerInfo` bitmask
3. 对每个 Token 检查是否需要发送到 `destServerId`:
   - 若目标位图命中，通过 `AIVRDMAPostSend()` 发送该 Token 的 TokenStruct
   - 目标地址递增 `tokenStructLen_`，保证接收端 Token 连续排列
4. 发送结束 Flag (`END_OF_WRITE_FLAG_VALUE`)，通知接收端发送完成

### 4.6 AIVRDMAPostSend() — RDMA 发送原语（第321-406行）

**功能**: 底层 RDMA WRITE 操作，通过 RoCEv2 将数据从本卡发送到目标 Rank。直接操作 QP 上下文和 DoorBell，避免了 HCCL 层的开销，实现了用户态 RDMA 的完整流程。

**实现细节**:
1. **QP（Queue Pair）管理**: 从 `HcclAiRMAInfo` 获取目标 Rank 的 SQ（Send Queue）上下文
2. **SQ 空间检查**: 轮询 SQ 的 head/tail 指针，确保有可用 WQE，不会溢出 QP 深度
3. **WQE（Work Queue Element）构建**:
   - 操作码: `opcode = 0x3` (RDMA_WRITE)
   - **owner_bit**: 通过 `(curHead >> shift) & 1U` 取反，用于 SQ 环形缓冲区回绕时的正确性保证
   - 设置 `IBV_SEND_SIGNALED` 标志
   - SGE（Scatter/Gather Entry）: 1 个，包含源地址和长度
   - 远端地址: 通过 `mem_info_table` 获取目标 Rank 的 MR（Memory Region）key 和 VA
4. **Cache 刷新**: `cacheWriteThrough()` 确保 WQE 和 SGE 对 RNIC 可见
5. **DoorBell 机制**: 通过 DataCopy 将更新后的 head 指针写入 DoorBell 寄存器，通知 RNIC 处理新 WQE

### 4.7 CreateInnerReduceInfo() — 机内归约索引生成（第429-601行）

**功能**: 为后续 Combine 阶段生成**机内**（同一 Server 内 8 个 Rank 之间）的 Token 归约索引表，并通过 RDMA 发送给目标 Server。

**流程**:
1. **Inner 表构建**:
   - 统计目标 Server 的每个专家收到的 Token 数（`expCntMap`）
   - 对每个 Token，检查其 K 个 expertId 是否属于目标 Server 的专家范围
   - 若属于，生成 `innerOffset = offsetInExp + expIdInServer * globalBs`，并按 expIdInServer 排序（插入排序）
   - `innerCnt` 记录每个 Token 在本 Server 内有几个专家需要接收
2. **RDMA 发送 Inner 表**: 将 `{BS信息, innerCnt[], innerOffset[]}` 发送到目标 Server
3. **RDMA 接收 Inner 表**: 等待从对端 Server 接收对应的 Inner 表数据
4. **写入输出**: 将 Inner 表解析后写入 `epRecvCountsGM_` 的对应偏移位置

**Inner Reduce 表结构**:
```
| axisBS (int16) | innerCnt[BS] (int16) | innerOffset[BS * K] (int32) | innerCnt[BS] | innerOffset[BS * K] | ... |
                 ←────────── 第1批 ──────────→←────────── 第2批 ──────────→←── ... ──→
```

**分批处理**: 因 UB 空间有限，Inner 表按 `maxBSInUBForInner_` 分批处理，每批大小由 `PreloadForInner()` 计算。

### 4.8 CreateOuterReduceInfo() — 机间归约索引生成（第626-682行）

**功能**: 为 Combine 阶段生成**机间**（跨 Server）的 Token 归约索引表。仅由 1 个核（`aivId_ == serverNum`）执行。

**流程**:
1. 从 `tokenAddrFlagStructGM_` 读取每个 Token 的 Server 位图（由 ReorderTokens 写入）
2. 对每个 Token，根据 Server 位图计算:
   - `outerCnt`: 该 Token 需要发送到几个 Server
   - `outerOffset[]`: 在每个目标 Server 中的偏移（`serverId * axisBS + tokenCntServer[serverId]`）
3. 将结果写入 `epRecvCountsGM_` 的 Outer 区域

**分批处理**: 按 `maxBSInUBForOuter_` 分批处理，每批大小由 `PreloadForOuter()` 计算。

### 4.9 Win2Ipc() — 机内 IPC 转发（第1021-1123行）

**功能**: 接收来自跨机 RDMA 和本机的 Token，按目标 Expert 和目标 Rank 进行分类，写入 IPC 缓冲区。

**核分配**: `coresPerServer = (aivNum_ - serverNum - 1) / serverNum`，每个核负责处理一个源 Server 的数据。

**流程**:
1. **Token 接收**: 通过 `GetArrivedTokenInfo()`（跨机 Token）或 `GetSelfServerTokenInfo()`（本机 Token）逐个接收 Token
2. **轮转调度**: 使用 `justExpInfo` 标志实现 `coresPerServer` 个核之间的轮转:
   - **完整模式**（`justExpInfo = false`）: 读取完整 TokenStruct 并搬运到 IPC
   - **仅信息模式**（`justExpInfo = true`）: 只读取 expertIds 用于统计计数
   - 这两种模式在 Token 之间交替进行，实现了 Token 数据搬运和计数统计的流水线化
3. **按 Expert 分类**: 将 Token 写入目标 Rank 的 IPC 数据缓冲区（`addrInfo_.GetIpcDataAddr(targetRankId, targetExpId, fromRankId)`）
4. **Token 计数**: 统计每个 Expert 收到的 Token 数，分批发送到目标 Rank 的 IPC 计数缓冲区

### 4.10 GetArrivedTokenInfo() — 跨机 Token 到达检测（第941-989行）

**功能**: 检测从指定 Server 的 RDMA 接收缓冲区中是否已到达指定的 Token。

**返回状态**:
| 状态 | 值 | 含义 |
|------|---|------|
| `FINISH_STATUS` | 0 | 所有 Token 已处理完毕 |
| `WAIT_STATUS` | 1 | Token 未到达或发送未结束，继续等待 |
| `ARRIVAL_STATUS` | 2 | Token 已到达，数据已搬入 `localUB` |
| `SKIP_STATUS` | 3 | 本机 Token 不需要发送到当前 Server（仅 `GetSelfServerTokenInfo` 返回） |

**保序校验机制**: 同时检查 3 个 Flag（EndFlag, CurrentTokenFlag, NextTokenFlag），利用硬件 ROCE 保序特性:
- 若下一个 Token 的 Flag 已到达，则当前 Token 必然完整 → 直接处理
- 若下一个 Token 未到达，则检查 end-of-write 标志 → 若已结束且当前 Token 有效，处理最后一个 Token

### 4.11 SetIpcFlag() / WaitIpcFlag() — IPC 同步（第895-938行）

**功能**: 基于 IPC Flag 实现 Server 内部的 Rank 间同步:
- `SetIpcFlag()`: 前 8 个核（`aivId_ < SERVER_RANK_SIZE`）各自向同 Server 内对应 Rank 写入 `magicVal_` 标志
- `WaitIpcFlag()`: 等待同 Server 内对应 Rank 写入 Flag（轮询直到 Flag >= `magicVal_`）

这是机内转发完成后的全局同步点，确保所有 Token 数据都已写入 IPC 缓冲区后再进行下一步读取。

### 4.12 Ipc2Out() — IPC 数据到最终输出（第1126-1278行）

**功能**: 从 IPC 缓冲区读取分类好的 Token，写出最终输出张量。

**核分配**: `coresPerExp = aivNum / localMoeExpertNum`，每个核处理一个 Expert 的所有源 Rank 数据。每个核进一步按 `srCntPerCore` 分配源 Rank 行。

**流程**:
1. **Token 计数前缀和**: 从 IPC TokenCnt 区域读取所有源 Rank 的 Token 计数，计算前缀和
2. **按 Expert 统计**: 计算每个 Expert 的总 Token 数，写入 `expertTokenNumsOut`。
   - 支持 `expertTokenNumsType_` 两种计数模式:
     - `0`（cumsum）: 累计和模式
     - `1`（count）: 按专家独立计数模式
3. **Token 搬运**: 对每个源 Rank，分批将 Token 从 IPC 缓冲区搬入 UB:
   - 提取 Token 数据 → 写入 `expandXOut`
   - 匹配当前 Expert → 提取对应权重 → 写入 `expandScales`
   - 动态量化场景 → 提取 scale → 写入 `dynamicScalesOut`
4. **元数据输出** (核 0 负责):
   - Token 计数表 → `epRecvCountsGM_`
   - Expert 粒度 Token 计数 → `expertTokenNumsOutGM_`（int64）

### 4.13 CleanUp() — 缓冲区清理（第1281-1311行）

**功能**: 清理 RDMA 接收缓冲区和 Inner 表的 Flag，为下一轮调用做准备。
- 核 `serverNum - 1` 负责清理 RDMA Flag 和 Inner Flag（共 `serverNum * STATE_OFFSET` 大小）
- 所有核（`aivId_ < serverNum`）清理各自对应 Server 的 RDMA Data 区域的 Token flag（按 `maxBs_` 个 Token 的 flag 字段清理）

### 4.14 CopyPerformanceInfo() — 性能信息输出（第1314-1321行）

**功能**: 当 `performanceInfo != nullptr` 时，使用 `AtomicAdd` 聚合各核性能数据到输出张量。

---

## 5. 通信模式分析

### 5.1 两级通信架构

**第一级 (跨机 RDMA)**:
- 方向: 源 Server → 目标 Server
- 传输方式: RoCE V2 RDMA WRITE
- 数据粒度: 整个 Token（包含所有 K 个 expert 信息）
- 发送方: 核 0 ~ serverNum-1 分别负责发送到对应 Server
- 特点: Token 保持原始顺序，不按 Expert 拆分

**第二级 (机内 IPC)**:
- 方向: 同一 Server 内 Rank 之间
- 传输方式: 共享内存 IPC（Global Memory 直写）
- 数据粒度: 按目标 Expert 和目标 Rank 分类后的 Token
- 处理方: 核 serverNum+1 及之后的所有核

### 5.2 同步点

| 同步点 | 类型 | 作用 |
|--------|------|------|
| `SyncAll` after ReorderTokens | 全局 | 确保所有 Token 重组完成后再开始发送 |
| `SyncAll` after Send+Inner+Outer+Win2Ipc | 全局 | 确保所有跨机发送和 IPC 写入完成 |
| `SetIpcFlag + WaitIpcFlag` | Server 内 | IPC 数据就绪通知 |
| `SyncAll` after Ipc2Out | 全局 | 确保输出完成后再清理 |

---

## 6. 内存管理

### 6.1 UB (Unified Buffer) 分配

```
UB布局 (tBuf, 190KB):
┌────────────┬──────────────────────────────────┬─────────────────┐
│ 0 ~ 8KB    │ TBUF_TEMP_OFFSET (8KB) ~ ...     │ 末尾 ~ 190KB    │
│ (临时空间)  │ 主工作区: TokenStruct暂存、      │ expertIdsI16    │
│            │ Inner/Outer索引表构建、IPC缓冲     │ (axisBS*alignK  │
│            │                                  │  个int16_t)     │
└────────────┴──────────────────────────────────┴─────────────────┘
```

各函数在 tBuf 中动态分配子区域:
- `QuantProcess`: workLt (6KB) + maxLt (3 * float)
- `CreateInnerReduceInfo`: expCntMap + expertIds + innerCnt + innerOffset
- `CreateOuterReduceInfo`: tokenCntServer + sendTokenInfo + outerCnt + outerOffset
- `Win2Ipc`: tokenNumPerExp + localUB
- `Ipc2Out`: tokenCntByExpUB (4k~6k) + weightLt (2k~4k) + tokenCntUB (6k~8k) + localUB (8k~)

### 6.2 GM (Global Memory) 使用

| 区域 | 用途 |
|------|------|
| RDMA Send Buffer (WindowOut) | TokenStruct 打包后的发送缓冲区 |
| RDMA Recv Buffer (WindowIn) | 接收其他 Server 发来的 Token |
| IPC Data Buffer | 机内 Rank 间 Token 转发的共享缓冲区 |
| IPC Flag Buffer | 机内同步标志 |
| IPC Token Count Buffer | 机内 Token 计数 |
| tokenAddrFlagStructGM_ | 存储每个 Token 的目标 Server 位图（workspace） |
| epRecvCountsGM_ | Combine 阶段索引表输出 |
| expandXOut | 最终输出 Token 数据（按 expert 汇总） |
| expandScales | 最终输出路由权重 |

---

## 7. 关键设计模式与优化

### 7.1 双缓冲 Ping/Pong

通过 `magicValue_ & 0x1` 选择当前使用的缓冲区。每次算子调用时 `magicValue_` 递增，自动切换 Ping/Pong，避免前后两次调用之间的数据竞争。

### 7.2 MagicValue 防伪机制

所有 Token 的 flag 值为 `SHOULD_SEND_FLAG_VALUE + magicVal_`，其中 `magicVal_` 是全局递增的计数器，存储在共享内存中。这确保了不同轮次的数据不会混淆——接收端只接受带有当前轮次 magicVal 的 Token。

### 7.3 分批处理

所有操作（ReorderTokens、CreateInnerReduceInfo、Win2Ipc、Ipc2Out）都支持将大量数据分批处理，每批大小由 UB 剩余空间决定。这避免了对超大 BS 场景的 UB 溢出问题。

### 7.4 流水线化的 Token 到达检测

`Win2Ipc()` 中通过 `justExpInfo` 标志将 Token 处理分为两种模式：
- **完整模式**（`justExpInfo = false`）: 读取完整 TokenStruct 并搬运到 IPC
- **仅信息模式**（`justExpInfo = true`）: 只读取 expertIds 用于统计

这两种模式在 Token 之间交替进行，实现了 Token 数据搬运和计数统计的流水线化，减少完整 Token 搬运次数。

### 7.5 保序设计

RDMA WRITE 的消息分片可能乱序到达（只有第一条消息的首个分片保证保序）。设计上通过以下方式保证正确性：
- 每个 Token 的 flag 放在结构体最前端
- 检测下一个 Token 的 flag 到达来确认当前 Token 数据完整
- 结束标志 `END_OF_WRITE_FLAG_VALUE` 作为最后的同步点

### 7.6 RDMA 资源复用

`AIVRDMAPostSend()` 直接操作 QP 上下文和 DoorBell，避免了 HCCL 层的开销。SQ 的 head/tail 轮询确保不会溢出。WQE 使用 owner_bit 机制支持环形缓冲区的正确回绕。

### 7.7 性能采集

当 `performanceInfo` 不为空时:
- `WaitIpcFlag`: 记录机间 RDMA 通信耗时
- `Win2Ipc`: 记录 Token 处理时间（多个核中仅第一个核记录，避免重复）
- `CopyPerformanceInfo`: 使用 `AtomicAdd` 汇总各核性能数据到输出张量（每个 Rank 对应一个 int64 时间戳，低 32 位为微秒级耗时）

### 7.8 数据搬运优化

- 使用 `DataCopyPad` 进行带间隔的 2D 数据搬运，减少 UB-GM 搬运次数
- `ReorderTokens` 中 ExpertIds 一次性搬入，避免重复访问 GM
- `Ipc2Out` 中 weight 匹配采用线性搜索 + break 优化（匹配到当前 expert 后立即跳出）

---

## 8. 关键约束与注意事项

### 8.1 硬件约束
- Flag 必须在 TokenStruct 最前端，且 Token 之间不连续存储（ROCE 消息保序要求）
- RDMA WQE 深度有限，需要轮询 head/tail 避免 SQ 溢出
- UB 空间 190KB，需精心分配以容纳最大批次数据

### 8.2 对齐要求
- 所有内存操作按 32B 对齐（`UB_32B_ALIGN = 32`）
- ExpertIds 和 Weights 在 TokenStruct 中按 `alignK_`（K 向上取整到 8 的倍数）对齐
- IPC 缓冲区按 512B 对齐（`IPC_BUFF_ALIGN = 512`）

### 8.3 正确性保证
- `magicVal_` 用于区分不同调用轮次，防止旧数据干扰
- `END_OF_WRITE_FLAG_VALUE (0xffffffff)` 标志发送结束
- 多处 `PipeBarrier` 和 `SyncFunc` 确保数据一致性

---

## 9. 输出数据说明

| 输出张量 | 内容 | 用途 |
|---------|------|------|
| `expandXOut` | 分发后的 Token 数据（按 expert 汇总） | 后续 FFN 计算的输入 |
| `dynamicScalesOut` | 动态量化的反量化 scale | INT8 → FP16/BF16 反量化使用 |
| `expandScales` | 每个 Token 对应专家的路由权重 | Combine 阶段加权求和 |
| `expertTokenNumsOut` | 每个专家收到的 Token 总数（int64） | 辅助信息 |
| `epRecvCountsOut` | Inner/Outer Reduce 索引信息 | Combine 算子使用，用于反向路由 |

---

## 10. 与非 Layered 版本的差异

本文件（`_a2_layered.h`）相对于 `_a2.h` 的核心区别在于:

1. **分层通信**: 将跨机通信和机内通信分为两个独立阶段（SendDataToServer + Win2Ipc），而非一步到位
2. **索引预生成**: 显式生成 Inner/Outer 归约索引表供 Combine 阶段使用
3. **IPC 转发**: 引入 Win2Ipc 阶段，将 Token 按 Expert 精确分类到目标 Rank 的 IPC 缓冲区
4. **核角色细分**: 将 AIV 核分为发送组、索引生成组、IPC 转发组，各司其职

---

## 11. 函数调用关系

```
Process()
├── ReorderTokens()
│   └── QuantProcess() [仅DynamicQuant]
├── SendDataToServer(destServerId)
│   └── AIVRDMAPostSend() [多次]
├── CreateInnerReduceInfo(serverIdx)
│   ├── PreloadForInner()
│   └── AIVRDMAPostSend() [Inner表 + 结束符]
├── CreateOuterReduceInfo()
│   └── PreloadForOuter()
├── Win2Ipc()
│   ├── GetArrivedTokenInfo() [跨机Token]
│   ├── GetSelfServerTokenInfo() [本机Token]
│   └── GetExpRank()
├── SetIpcFlag()
├── WaitIpcFlag()
├── Ipc2Out()
├── CleanUp()
└── CopyPerformanceInfo()
```

---

## 12. 依赖关系

```
moe_distribute_dispatch_a2_layered.h
    ├── moe_distribute_a2_base.h              # HCCL 缓冲区地址管理
    │   └── MoeDistributeA2DispatchAddrInfo   # Dispatch 专用地址计算（含 Ping/Pong 选择）
    ├── moe_distribute_base.h                 # HCCL 底层数据结构
    │   ├── HcclA2CombineOpParam              # A2 通信参数
    │   ├── HcclAiRMAInfo                     # RDMA QP/Mem 信息
    │   └── cacheWriteThrough()               # Cache 刷新工具
    ├── mc2_kernel_utils.h                    # SyncFunc<> 同步原语、RecordRankCommDuration
    ├── moe_distribute_dispatch_tiling.h      # Tiling 数据结构
    │   └── MoeDistributeDispatchA2TilingData
    └── kernel_basic_intf.h / kernel_operator.h  # AscendC 基础 API（按 SDK 版本选择）
```
