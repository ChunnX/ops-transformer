# MoeDistributeDispatchA2Layered::Process() 函数深度分析

## 目录
1. [整体逻辑概述](#1-整体逻辑概述)
2. [集群拓扑与分层通信策略](#2-集群拓扑与分层通信策略)
3. [核心数据结构](#3-核心数据结构)
4. [内存布局详解](#4-内存布局详解)
5. [子流程详细分析](#5-子流程详细分析)
6. [时序流水与同步机制](#6-时序流水与同步机制)
7. [核间分工策略](#7-核间分工策略)
8. [开发注意事项](#8-开发注意事项)

---

## 1. 整体逻辑概述

### 1.1 算子功能

`MoeDistributeDispatchA2Layered` 实现了 MoE（Mixture of Experts）模型中 Token 的**分发（Dispatch）**逻辑。其核心任务是根据路由结果（expertIds），将每个 Token 发送到对应 Expert 所在的 Rank 上。

### 1.2 Process() 主流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Process() 主流程                               │
├─────────────────────────────────────────────────────────────────────────┤
│  阶段1: ReorderTokens()         所有核并行将Token打包成TokenStruct        │
│  ─────────────────────────────────────────────────────────────────────  │
│  阶段2-5: 按核分工并行处理                                                │
│    ├─ 核[0, serverNum): SendDataToServer + CreateInnerReduceInfo        │
│    ├─ 核[serverNum]:     CreateOuterReduceInfo                          │
│    └─ 核[serverNum+1,∞): Win2Ipc                                        │
│  ─────────────────────────────────────────────────────────────────────  │
│  Server内Rank间同步: SetIpcFlag + WaitIpcFlag                           │
│  ─────────────────────────────────────────────────────────────────────  │
│  阶段6: Ipc2Out()               所有核并行将IPC数据写入输出Tensor         │
│  ─────────────────────────────────────────────────────────────────────  │
│  阶段7: CleanUp()               清理缓冲区标志位                         │
│  ─────────────────────────────────────────────────────────────────────  │
│  收尾: CopyPerformanceInfo() + hccl_.Finalize()                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 代码实现

```cpp
__aicore__ inline void Process()
{
    if ASCEND_IS_AIV { // 仅AIV核心执行（排除AIC核心）
        // ========== 阶段1：所有核并行将Token打包成TokenStruct ==========
        ReorderTokens();
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();  // 全核同步：确保所有Token已打包完毕

        // ========== 阶段2-5：按核分工并行处理 ==========
        if(aivId_ < serverNum){
            // 前serverNum个核：跨Server发送Token + 构建Inner表
            if(aivId_ != serverId_){
                SendDataToServer(aivId_);  // 向远端Server发送Token
            }
            CreateInnerReduceInfo(aivId_); // 构建并交换Inner路由表
        } else if (aivId_ == serverNum) {
            // 第serverNum个核：构建Outer路由表
            CreateOuterReduceInfo();
        } else {
            // 其余核：从RDMA接收缓冲区分拣Token到IPC缓冲区
            Win2Ipc();
        }

        // ========== Server内Rank间同步 ==========
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        SetIpcFlag(IPC_FLAG_STEP_1);   // 通知Server内其他Rank
        WaitIpcFlag(IPC_FLAG_STEP_1);  // 等待Server内所有Rank就绪

        // ========== 阶段6：所有核并行将IPC数据写入输出Tensor ==========
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        Ipc2Out();

        // ========== 阶段7：清理缓冲区 ==========
        if (aivId_ < serverNum) {
            PipeBarrier<PIPE_ALL>();
            CleanUp();
        }

        // ========== 收尾 ==========
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        CopyPerformanceInfo();
        hccl_.Finalize();
    }
}
```

---

## 2. 集群拓扑与分层通信策略

### 2.1 集群拓扑结构

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              集群整体拓扑                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Server 0                Server 1                Server N-1                 │
│  ┌─────────┐            ┌─────────┐            ┌─────────┐                  │
│  │ Rank 0  │            │ Rank 8  │            │ Rank    │                  │
│  │ Rank 1  │            │ Rank 9  │            │ ...     │                  │
│  │ Rank 2  │            │ Rank 10 │            │         │                  │
│  │ Rank 3  │  RDMA      │ Rank 11 │  RDMA      │         │                  │
│  │ Rank 4  │◄──────────►│ Rank 12 │◄──────────►│         │                  │
│  │ Rank 5  │  跨Server  │ Rank 13 │  跨Server  │         │                  │
│  │ Rank 6  │            │ Rank 14 │            │         │                  │
│  │ Rank 7  │            │ Rank 15 │            │         │                  │
│  └─────────┘            └─────────┘            └─────────┘                  │
│       │                       │                      │                      │
│       └───────────────────────┴──────────────────────┘                      │
│                        IPC (Server内通信)                                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 关键常量定义

| 常量名 | 值 | 说明 |
|--------|-----|------|
| `SERVER_RANK_SIZE` | 8 | 每个Server内的Rank数量（8卡一组） |
| `UB_32B_ALIGN` | 32 | UB最小对齐粒度 |
| `TBUF_SIZE` | 190KB | 主工作UB缓冲区大小 |
| `FLAG_SIZE` | 64B | TokenStruct中flag字段大小 |
| `IPC_BUFF_ALIGN` | 512B | IPC缓冲区对齐粒度 |

### 2.3 分层通信策略（Layered = 先跨Server，再Server内）

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          分层通信策略                                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   第1层: 跨Server通信 (RDMA)                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  • 使用RoCE RDMA WRITE单边写操作                                      │  │
│   │  • 每个Rank向其他Server的"同位卡"发送Token                             │  │
│   │  • 发送目标: dstRankId = rankId_ % 8 + destServerId * 8              │  │
│   │  • 利用RoCE PCIE保序特性判断Token到达                                  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                              ↓                                             │
│   第2层: Server内通信 (IPC)                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  • 使用共享内存IPC机制                                                │  │
│   │  • Token按Expert分拣到对应Rank的IPC缓冲区                             │  │
│   │  • 通过SetIpcFlag/WaitIpcFlag进行Server内Rank间同步                   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心数据结构

### 3.1 TokenStruct 内存布局

TokenStruct 是 Token 在 RDMA 传输和 IPC 共享中的核心数据结构：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TokenStruct 内存布局                                │
├──────────┬────────────────────┬─────────────────────────────────────────────┤
│  字段    │       大小          │                  说明                        │
├──────────┼────────────────────┼─────────────────────────────────────────────┤
│  flag    │  32B (UB_32B_ALIGN)│ Token标志位，用于判断Token是否完整接收        │
│          │                    │ 值 = SHOULD_SEND_FLAG_VALUE + magicVal_      │
├──────────┼────────────────────┼─────────────────────────────────────────────┤
│  token   │  H * sizeof(dtype) │ Token特征数据                                │
│          │                    │ 非量化: 直接拷贝; 量化: 经QuantProcess处理     │
├──────────┼────────────────────┼─────────────────────────────────────────────┤
│  exp     │  alignK * 4B       │ K个Expert的ID（32B对齐）                     │
│          │                    │ alignK = RoundUp(K, 8)                       │
├──────────┼────────────────────┼─────────────────────────────────────────────┤
│  weight  │  alignK * 4B       │ K个Expert对应的权重（32B对齐）               │
├──────────┼────────────────────┼─────────────────────────────────────────────┤
│  scale   │  32B (UB_32B_ALIGN)│ 动态量化因子（仅DynamicQuant模式使用）       │
│          │                    │ 存放反量化因子 1.0/dynamicScale              │
└──────────┴────────────────────┴─────────────────────────────────────────────┘

总长度: tokenStructLen_ = 32 + H*sizeof(dtype) + alignK*8 + 32

【关键设计约束】
• flag 必须放在 TokenStruct 最前端
• 原因: RoCE消息经PCIE写入GM时，只有下一条消息的第一个分片与上一条消息保证写入顺序
• 接收方通过检测"下一个Token的flag是否到达"来判断"当前Token是否完整接收"
```

### 3.2 关键偏移量计算

```cpp
// 在 Init() 中计算
flagOffsetInStruct_ = 0;                              // flag始终在开头
tokenOffsetInStruct_ = flagLenInStruct_;              // token紧跟flag
expOffsetInStruct_ = tokenOffsetInStruct_ + tokenLenInStruct_;
weightOffsetInStruct_ = expOffsetInStruct_ + expLenInStruct_;
scaleOffsetInStruct_ = weightOffsetInStruct_ + weightLenInStruct_;

// 跨步拷贝参数（用于DataCopy跨步批量拷贝）
tokenGapInStruct_ = (tokenStructLen_ - tokenLenInStruct_) / UB_32B_ALIGN;
infoGapInStruct_ = (tokenStructLen_ - expLenInStruct_) / UB_32B_ALIGN;
```

### 3.3 路由表数据结构

#### 3.3.1 Inner表（Server内路由表）

Inner表描述"某个Server发来的Token，在目标Server内如何分配到各个Expert"。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Inner表结构                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  innerCnt[tokenIdx]: int16                                                  │
│    → 该Token在目标Server内有几个Expert需要接收                               │
│                                                                             │
│  innerOffset[tokenIdx * K + i]: int32                                       │
│    → 第i个Expert的偏移信息                                                   │
│    → 编码: expertInServer * globalBs + tokenIdxInExpert                     │
│    → 用于Combine阶段定位Token的来源                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 3.3.2 Outer表（跨Server路由表）

Outer表描述"本Rank的每个Token发往了哪些Server"。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Outer表结构                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  outerCnt[tokenIdx]: int32                                                  │
│    → 该Token发往了几个Server                                                │
│                                                                             │
│  outerOffset[tokenIdx * serverNum + i]: int32                               │
│    → 该Token在第i个目标Server中的到达顺序                                    │
│    → 编码: serverId * axisBS_ + tokenCntServer[serverId]                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 epRecvCounts 输出内存布局

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      epRecvCounts 内存布局                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Offset                                                                     │
│  0          combineInnerCntOffset          每个Expert每个Rank的Token计数    │
│             (前缀和)                                                        │
│                                                                             │
│  combine    combineInnerCntIndexOffset      Inner表Cnt部分                  │
│  InnerCnt                                                                   │
│  Offset                                                                     │
│                                                                             │
│  combine    combineOuterCntOffset           Inner表Offset部分               │
│  InnerCnt                                                                   │
│  IndexOffset                                                                │
│                                                                             │
│  combine    combineOuterCntIndexOffset      Outer表Cnt部分                  │
│  OuterCnt                                                                   │
│  Offset                                                                     │
│                                                                             │
│  combine    ...                             Outer表Offset部分               │
│  OuterCnt                                                                   │
│  IndexOffset                                                                │
└─────────────────────────────────────────────────────────────────────────────┘

偏移量计算:
combineInnerCntOffset = localMoeExpertNum_ * serverNum * SERVER_RANK_SIZE * sizeof(int32_t)
combineInnerCntIndexOffset = combineInnerCntOffset + globalBs_ * serverNum * sizeof(int16_t)
combineOuterCntOffset = combineInnerCntIndexOffset + globalBs_ * axisK_ * serverNum * sizeof(int32_t)
combineOuterCntIndexOffset = combineOuterCntOffset + axisBS_ * sizeof(int32_t)
```

---

## 4. 内存布局详解

### 4.1 UB（Unified Buffer）布局

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TBUF (190KB) 内存布局                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Offset 0: TBUF_TEMP_OFFSET (8KB)                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    临时数据区 (保留)                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Offset TBUF_TEMP_OFFSET:                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Token数据区                                        │   │
│  │                    大小: tokenUbSize_                                 │   │
│  │                    用于ReorderTokens/Win2Ipc/Ipc2Out                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Offset TBUF_TEMP_OFFSET + tokenUbSize_:                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    ExpertIds int16缓存区                              │   │
│  │                    大小: expertIdsSize                                │   │
│  │                    用于存储转换后的int16 ExpertIds                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 GM（Global Memory）缓冲区布局

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GM 缓冲区布局                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. RDMA发送缓冲区 (sendTokensU8Tensor_)                                    │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │  TokenStruct[0] | TokenStruct[1] | ... | TokenStruct[BS-1]     │    │
│     │  每个TokenStruct大小 = tokenStructLen_                          │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  2. RDMA接收缓冲区 (按源Rank分区)                                           │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │  Rank0区 | Rank1区 | ... | Rank7区 | Rank8区 | ...              │    │
│     │  每个Rank区大小 = maxBs_ * tokenStructLen_                       │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  3. IPC缓冲区 (按目标Expert和源Rank分区)                                    │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │  [Expert0][Rank0] | [Expert0][Rank1] | ... | [ExpertN][RankM]  │    │
│     │  每个分区大小 = maxBs_ * tokenStructLen_                         │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  4. workspace (Token目标Server位图)                                         │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │  sendServerInfo[0] | sendServerInfo[1] | ... | sendServerInfo[BS-1]│ │
│     │  每个元素大小 = FLAG_SIZE (64B)                                  │    │
│     │  内容: 64位位图，每位表示一个Server                               │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 子流程详细分析

### 5.1 阶段1: ReorderTokens()

**功能**: 所有AIV核心并行将Token打包成TokenStruct结构。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ReorderTokens() 流程                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入:                                                                      │
│    • expandXGM_: Token特征数据 [BS, H]                                      │
│    • expertIdsGMTensor_: Expert ID列表 [BS, K]                              │
│    • weightsGM_: Expert权重 [BS, K]                                         │
│                                                                             │
│  处理步骤:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. 分核: 将BS个Token均匀分配到aivNum_个核                            │   │
│  │    • 前remainderTokenNum个核多处理1个Token                           │   │
│  │                                                                     │   │
│  │ 2. 预取ExpertIds: GM -> UB (int32 -> int16节省空间)                 │   │
│  │                                                                     │   │
│  │ 3. 分批处理 (UB空间有限):                                           │   │
│  │    a. Token特征数据: GM -> UB -> TokenStruct.token字段              │   │
│  │       (DynamicQuant模式需经QuantProcess量化)                        │   │
│  │    b. Expert ID: GM -> TokenStruct.exp字段                          │   │
│  │    c. Weight: GM -> TokenStruct.weight字段                          │   │
│  │    d. 设置flag = SHOULD_SEND_FLAG_VALUE + magicVal_                 │   │
│  │    e. 计算目标Server位图 -> workspace                               │   │
│  │    f. TokenStruct: UB -> RDMA发送缓冲区                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  输出:                                                                      │
│    • sendTokensU8Tensor_: 打包好的TokenStruct数组                           │
│    • tokenAddrFlagStructGM_: 每个Token的目标Server位图                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**目标Server位图计算**:
```cpp
for (uint32_t i = 0; i < axisK_; i++) {
    uint32_t expertId = expertIdsI16Tensor_(startExpId + i);
    uint32_t dstServerId = expertId / moeExpertNumInServer_;
    sendServerInfo |= (1UL << dstServerId);  // 位图OR操作
}
```

### 5.2 阶段2: SendDataToServer()

**功能**: 通过RDMA将Token跨Server发送到目标Server。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SendDataToServer() 流程                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  执行条件: aivId_ < serverNum && aivId_ != serverId_                        │
│  (不需要给自己所在Server发送)                                                │
│                                                                             │
│  处理步骤:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. 读取workspace中所有Token的目标Server位图                          │   │
│  │                                                                     │   │
│  │ 2. 遍历所有Token:                                                   │   │
│  │    if (destServerInfo & destServerMask) != 0:                       │   │
│  │        AIVRDMAPostSend(TokenStruct)                                 │   │
│  │        dstRdmaAddr += tokenStructLen_                               │   │
│  │                                                                     │   │
│  │ 3. 发送结束标志: AIVRDMAPostSend(END_OF_WRITE_FLAG)                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  RDMA目标地址计算:                                                          │
│    dstRankId = rankId_ % SERVER_RANK_SIZE + destServerId * SERVER_RANK_SIZE│
│    (向对端Server的"同位卡"发送)                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 阶段3: CreateInnerReduceInfo()

**功能**: 构建Server内路由表（Inner表），用于Combine阶段的Token回收。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   CreateInnerReduceInfo() 流程                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  执行条件: aivId_ < serverNum                                                │
│  每个核负责一个目标Server的Inner表构建                                       │
│                                                                             │
│  处理步骤:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. 初始化:                                                          │   │
│  │    • expCntMap[expertNumInServer]: 统计每个Expert收到的Token数      │   │
│  │    • 将axisBS_写入Inner表头部                                       │   │
│  │                                                                     │   │
│  │ 2. 分批处理 (UB空间有限):                                           │   │
│  │    for batch in batches:                                            │   │
│  │        a. 从GM搬入ExpertIds                                         │   │
│  │        b. 遍历每个Token的K个Expert:                                 │   │
│  │           if Expert在目标Server内:                                  │   │
│  │              innerCnt[tokenIdx]++                                   │   │
│  │              innerOffset = expertInServer * globalBs + tokenIdx     │   │
│  │              expCntMap[expertInServer]++                            │   │
│  │        c. 将innerCnt/innerOffset写入RDMA发送缓冲区                  │   │
│  │                                                                     │   │
│  │ 3. 发送Inner表到目标Server (通过RDMA)                               │   │
│  │                                                                     │   │
│  │ 4. 等待目标Server发来的Inner表                                      │   │
│  │                                                                     │   │
│  │ 5. 将收到的Inner表写入epRecvCounts输出                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  innerOffset编码:                                                           │
│    = expertInServer * globalBs + tokenIdxInExpert                           │
│    用于Combine阶段快速定位Token的来源Expert                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.4 阶段4: CreateOuterReduceInfo()

**功能**: 构建跨Server路由表（Outer表），用于Combine阶段的跨Server回收。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   CreateOuterReduceInfo() 流程                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  执行条件: aivId_ == serverNum (仅第serverNum个核执行)                       │
│                                                                             │
│  处理步骤:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. 初始化tokenCntServer[serverNum]: 统计发往每个Server的Token数     │   │
│  │                                                                     │   │
│  │ 2. 分批处理:                                                        │   │
│  │    for batch in batches:                                            │   │
│  │        a. 从workspace读取Token的目标Server位图                      │   │
│  │        b. 遍历每个Token:                                            │   │
│  │           for serverId in [0, serverNum):                           │   │
│  │               if (destServerInfo & (1 << serverId)):                │   │
│  │                  outerCnt[tokenIdx]++                               │   │
│  │                  outerOffset = serverId * BS + tokenCntServer[...]  │   │
│  │                  tokenCntServer[serverId]++                         │   │
│  │        c. 写入epRecvCounts输出                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  outerOffset编码:                                                           │
│    = serverId * axisBS_ + tokenCntServer[serverId]                          │
│    用于Combine阶段定位Token的来源Server                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.5 阶段5: Win2Ipc()

**功能**: 从RDMA接收缓冲区按Expert分拣Token到IPC缓冲区。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Win2Ipc() 流程                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  执行条件: aivId_ > serverNum                                                │
│  多核协同: coresPerServer个核负责一个源Server的Token分拣                     │
│                                                                             │
│  核心思想:                                                                  │
│    收到的Token是按源Rank打包的，但Expert分布在Server内不同Rank上            │
│    本函数将Token按Expert分拣到对应Rank的IPC缓冲区                           │
│                                                                             │
│  处理步骤:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. 确定本核负责的源Server: fromServerId = logicAivId / coresPerServer│  │
│  │                                                                     │   │
│  │ 2. 主循环 (while tokenStatus != FINISH_STATUS):                     │   │
│  │    a. 获取Token:                                                    │   │
│  │       • 本Server: GetSelfServerTokenInfo()                          │   │
│  │       • 远端Server: GetArrivedTokenInfo()                           │   │
│  │    b. 解析Expert信息，筛选属于本Server的Expert                      │   │
│  │    c. 统计每个Expert的Token数 (tokenNumPerExp)                      │   │
│  │    d. 如果是本核轮次 (justExpInfo=false):                           │   │
│  │       将Token数据写入目标Rank的IPC缓冲区                            │   │
│  │                                                                     │   │
│  │ 3. 尾部: 将tokenNumPerExp写入对端Rank的IPC计数区                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  协同机制:                                                                  │
│    • 所有核都统计Expert Token计数                                           │
│    • 只有 tokenIdx % coresPerServer == localCoreId 的核负责搬运Token数据    │
│    • 其他核只读取Expert信息（justExpInfo=true，节省带宽）                   │
│                                                                             │
│  Token状态判断 (利用RoCE PCIE保序特性):                                     │
│    • ARRIVAL_STATUS: 下一个Token的flag已到达                                │
│    • WAIT_STATUS: 结束标志尚未到达                                          │
│    • FINISH_STATUS: 结束标志已到达且当前Token无数据                         │
│    • SKIP_STATUS: Token不发往本Server（仅本机自发自收）                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.6 阶段6: Ipc2Out()

**功能**: 从IPC缓冲区将Token写入最终输出Tensor。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Ipc2Out() 流程                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  执行条件: 所有AIV核心参与                                                   │
│  分核策略: 每coresPerExp个核处理一个本地Expert                               │
│                                                                             │
│  处理步骤:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. 从IPC计数区读取每个(Expert, srcRank)的Token计数                  │   │
│  │                                                                     │   │
│  │ 2. 计算前缀和，确定每个(Expert, srcRank)在输出Tensor中的位置        │   │
│  │                                                                     │   │
│  │ 3. 主循环 (遍历本核负责的每个(Expert, srcRank)对):                  │   │
│  │    for (Expert, srcRank) in my_task:                                │   │
│  │        a. 从IPC数据区搬Token到UB                                    │   │
│  │        b. Token特征数据 -> expandXOut                               │   │
│  │        c. 提取本Expert对应的权重 -> expandScales                    │   │
│  │        d. (DynamicQuant) 量化因子 -> dynamicScalesOut               │   │
│  │                                                                     │   │
│  │ 4. 核0负责输出统计信息:                                             │   │
│  │    • epRecvCounts: 每个(Expert, srcRank)的Token前缀和               │   │
│  │    • expertTokenNums: 每个Expert的Token总数                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  输出Tensor布局:                                                            │
│    expandXOut: Token按(Expert, srcRank)有序排列                             │
│                先按Expert分区，Expert内按srcRank排序                         │
│                                                                             │
│  权重提取:                                                                  │
│    遍历每个Token的K个Expert，找到匹配本Expert全局ID的权重                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.7 阶段7: CleanUp()

**功能**: 清理标志位和缓冲区，为下一轮调用做准备。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CleanUp() 流程                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  执行条件: aivId_ < serverNum                                                │
│  每个核清理一个Server对应的缓冲区                                            │
│                                                                             │
│  清理内容:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. 核0: 更新BufferId (双缓冲轮换)                                   │   │
│  │                                                                     │   │
│  │ 2. 最后一个核(serverNum-1):                                         │   │
│  │    • 清理所有Server的RDMA接收状态标志                               │   │
│  │    • 清理所有Server的Inner表接收状态标志                            │   │
│  │                                                                     │   │
│  │ 3. 每个核: 清理本核负责的Server对应的Token接收缓冲区中的flag字段    │   │
│  │    (跨步清零，只清flag部分)                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 时序流水与同步机制

### 6.1 完整时序图

```
时间轴
  │
  │  ┌─────────────────────────────────────────────────────────────────────┐
  │  │ 阶段1: ReorderTokens()                                              │
  │  │ 所有核并行执行                                                       │
  │  │ ├─ 核0: Token[0, N/aivNum)                                          │
  │  │ ├─ 核1: Token[N/aivNum, 2N/aivNum)                                  │
  │  │ └─ ...                                                              │
  │  └─────────────────────────────────────────────────────────────────────┘
  │                              │
  │                              ▼
  │                   ┌─────────────────────┐
  │                   │   SyncAll<true>()   │  全核同步点1
  │                   └─────────────────────┘
  │                              │
  │  ┌─────────────────────────────────────────────────────────────────────┐
  │  │ 阶段2-5: 按核分工并行                                                │
  │  │                                                                     │
  │  │ 核[0, serverNum)           │ 核[serverNum]      │ 核[serverNum+1,∞) │
  │  │ ├─ SendDataToServer()      │ CreateOuter        │ Win2Ipc()         │
  │  │ │  (RDMA跨Server)          │ ReduceInfo()       │ (接收+分拣)       │
  │  │ └─ CreateInnerReduceInfo() │                    │                   │
  │  └─────────────────────────────────────────────────────────────────────┘
  │                              │
  │                              ▼
  │                   ┌─────────────────────┐
  │                   │   SyncAll<true>()   │  全核同步点2
  │                   └─────────────────────┘
  │                              │
  │  ┌─────────────────────────────────────────────────────────────────────┐
  │  │ Server内Rank间同步                                                   │
  │  │ SetIpcFlag() ─────► 通知Server内其他Rank                            │
  │  │ WaitIpcFlag() ◄───── 等待Server内所有Rank就绪                       │
  │  └─────────────────────────────────────────────────────────────────────┘
  │                              │
  │                              ▼
  │                   ┌─────────────────────┐
  │                   │   SyncAll<true>()   │  全核同步点3
  │                   └─────────────────────┘
  │                              │
  │  ┌─────────────────────────────────────────────────────────────────────┐
  │  │ 阶段6: Ipc2Out()                                                    │
  │  │ 所有核并行执行，将IPC数据写入输出Tensor                              │
  │  └─────────────────────────────────────────────────────────────────────┘
  │                              │
  │  ┌─────────────────────────────────────────────────────────────────────┐
  │  │ 阶段7: CleanUp() (仅核[0, serverNum))                               │
  │  └─────────────────────────────────────────────────────────────────────┘
  │                              │
  │                   ┌─────────────────────┐
  │                   │   SyncAll<true>()   │  全核同步点4
  │                   └─────────────────────┘
  │                              │
  │  ┌─────────────────────────────────────────────────────────────────────┐
  │  │ 收尾: CopyPerformanceInfo() + hccl_.Finalize()                      │
  │  └─────────────────────────────────────────────────────────────────────┘
  ▼
```

### 6.2 同步原语说明

| 同步原语 | 作用域 | 说明 |
|----------|--------|------|
| `PipeBarrier<PIPE_ALL>` | 核内 | 流水线屏障，确保所有流水线阶段完成 |
| `SyncAll<true>` | 全核 | 全核同步，阻塞直到所有核到达同步点 |
| `SetIpcFlag` | Server内 | 设置IPC同步标志，通知Server内其他Rank |
| `WaitIpcFlag` | Server内 | 等待IPC同步标志，阻塞直到所有Rank就绪 |

### 6.3 同步流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Server内Rank间同步流程                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Rank 0              Rank 1              Rank 2              Rank 7         │
│    │                   │                   │                   │            │
│    │ Win2Ipc()         │ Win2Ipc()         │ Win2Ipc()         │ Win2Ipc()  │
│    │ 完成              │ 完成              │ 完成              │ 完成       │
│    │                   │                   │                   │            │
│    ▼                   ▼                   ▼                   ▼            │
│  SetIpcFlag ──────────────────────────────────────────────────►            │
│    │               (写入其他Rank的IPC同步区)                                │
│    │                   │                   │                   │            │
│    │◄────────────────────────────────── WaitIpcFlag            │            │
│    │                   │◄────────────────────────────────── WaitIpcFlag     │
│    │                   │                   │◄────────────────────────── ... │
│    │                   │                   │                   │            │
│    ▼                   ▼                   ▼                   ▼            │
│  所有Rank同步完成，继续执行Ipc2Out()                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 核间分工策略

### 7.1 阶段1 (ReorderTokens) 分核

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ReorderTokens 分核策略                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  均匀分配 + 余数处理                                                         │
│                                                                             │
│  假设: BS=100, aivNum=13                                                    │
│  • 基础Token数/核 = 100/13 = 7                                              │
│  • 余数 = 100%13 = 9                                                        │
│  • 前9个核处理8个Token，后4个核处理7个Token                                 │
│                                                                             │
│  核ID   │ Token范围      │ Token数                                          │
│  ───────┼────────────────┼────────                                          │
│  0      │ [0, 8)         │ 8                                                │
│  1      │ [8, 16)        │ 8                                                │
│  ...    │ ...            │ 8                                                │
│  8      │ [72, 80)       │ 8                                                │
│  9      │ [80, 87)       │ 7                                                │
│  10     │ [87, 94)       │ 7                                                │
│  11     │ [94, 100)      │ 6                                                │
│  12     │ (空)           │ 0                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 阶段2-5 分核

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       阶段2-5 分核策略                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  假设: serverNum=4, aivNum=13                                               │
│                                                                             │
│  核ID   │ 任务                                                              │
│  ───────┼──────────────────────────────────────────────────────────────────│
│  0      │ SendDataToServer(0) [跳过,本Server] + CreateInnerReduceInfo(0)   │
│  1      │ SendDataToServer(1) + CreateInnerReduceInfo(1)                   │
│  2      │ SendDataToServer(2) + CreateInnerReduceInfo(2)                   │
│  3      │ SendDataToServer(3) + CreateInnerReduceInfo(3)                   │
│  4      │ CreateOuterReduceInfo()                                          │
│  5-12   │ Win2Ipc() - 处理来自不同Server的Token分拣                         │
│                                                                             │
│  Win2Ipc详细分工:                                                           │
│  • coresPerServer = (13 - 4 - 1) / 4 = 2                                   │
│  • 核5-6: 处理来自Server0的Token                                            │
│  • 核7-8: 处理来自Server1的Token                                            │
│  • 核9-10: 处理来自Server2的Token                                           │
│  • 核11-12: 处理来自Server3的Token                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 阶段6 (Ipc2Out) 分核

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Ipc2Out 分核策略                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  假设: localMoeExpertNum=8, aivNum=13                                       │
│  • coresPerExp = 13 / 8 = 1                                                 │
│  • 每个核处理1个Expert（核12不参与）                                        │
│                                                                             │
│  核ID   │ Expert                                                            │
│  ───────┼──────────────────────────────────────────────────────────────────│
│  0      │ Expert 0                                                          │
│  1      │ Expert 1                                                          │
│  ...    │ ...                                                               │
│  7      │ Expert 7                                                          │
│  8-12   │ 不参与                                                            │
│                                                                             │
│  每个核处理所有(srcServer, srcRank)对的数据搬运                             │
│  • srCntPerExp = serverNum * SERVER_RANK_SIZE = 4 * 8 = 32                 │
│  • srCntPerCore = 32 / 1 = 32                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. 开发注意事项

### 8.1 内存对齐约束

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          内存对齐要求                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. UB对齐: 所有UB操作必须32B对齐                                           │
│     • tokenStructLen_ 必须是32的倍数                                        │
│     • DataCopy操作的数据量必须是32B的倍数                                   │
│                                                                             │
│  2. IPC缓冲区对齐: 512B对齐                                                 │
│     • addrInfo_.GetIpcDataAddr() 返回的地址512B对齐                        │
│                                                                             │
│  3. K值对齐:                                                                │
│     • alignK_ = RoundUp(K, 8)                                               │
│     • Expert ID和Weight数组长度必须是alignK_                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 RDMA发送注意事项

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RDMA发送关键点                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. SQ深度检查: 发送前必须检查SQ是否有空闲槽位                              │
│     while ((curHead - tail) >= QP_DEPTH - 1) { wait; }                     │
│                                                                             │
│  2. Cache刷新: WQE写入后必须刷新Cache                                       │
│     cacheWriteThrough(wqeAddr, wqeSize + sgeSize);                         │
│                                                                             │
│  3. Doorbell写入: 不能直接写MMIO，必须通过DataCopy                          │
│     DataCopyPad(DBGlobalTensor, ubLocal, copyParams);                      │
│                                                                             │
│  4. Token到达判断: 利用RoCE PCIE保序特性                                    │
│     • 检测"下一个Token的flag"是否到达来判断"当前Token"是否完整              │
│     • flag必须放在TokenStruct最前端                                         │
│                                                                             │
│  5. magicVal_机制: 每次调用递增，区分不同轮次                               │
│     magicVal_ = addrInfo_.UpdateAndGetMagicValue(tempLocal);               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.3 同步要点

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          同步机制要点                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. PipeBarrier vs SyncFunc:                                                │
│     • PipeBarrier<PIPE_ALL>: 核内流水线屏障                                 │
│     • SyncFunc<HardEvent::XXX_YYY>: 特定硬件事件同步                        │
│                                                                             │
│  2. SyncAll使用场景:                                                        │
│     • 阶段切换前，确保所有核完成上一阶段                                    │
│     • 使用SyncAll<true>表示需要同步                                         │
│                                                                             │
│  3. IPC同步:                                                                │
│     • SetIpcFlag/WaitIpcFlag仅前8个核执行                                   │
│     • 通过共享内存实现Server内Rank间同步                                    │
│                                                                             │
│  4. 轮询超时:                                                                │
│     • WaitIpcFlag轮询不做死循环保护                                         │
│     • Host侧会超时kill异常情况                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.4 数据搬运优化

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        数据搬运优化技巧                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 跨步拷贝 (Strided DataCopy):                                            │
│     • 用于从连续TokenStruct中提取特定字段                                   │
│     • DataCopyExtParams{rowCount, sizePerRow, gap, 0, 0}                   │
│     • gap = (structLen - fieldLen) / UB_32B_ALIGN                          │
│                                                                             │
│  2. 分批处理:                                                                │
│     • UB空间有限，大量数据需分批搬运                                        │
│     • maxTokenNumInUB = tokenUbSize_ / singleTokenUBSize                   │
│                                                                             │
│  3. 类型复用UB:                                                              │
│     • localUB.ReinterpretCast<TYPE>()                                       │
│     • 避免分配多个UB缓冲区                                                  │
│                                                                             │
│  4. Cache管理:                                                               │
│     • DataCacheCleanAndInvalid: 刷新Cache                                  │
│     • cacheWriteThrough: 写穿透模式                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.5 动态量化处理

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       动态量化 (DynamicQuant) 处理                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  量化流程 (fp16 -> int8):                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. Cast: fp16 -> float32                                            │   │
│  │ 2. ReduceMax/ReduceMin: 求最大/最小绝对值                            │   │
│  │ 3. dynamicScale = 127.0 / maxAbsVal                                 │   │
│  │ 4. Muls: float32 *= dynamicScale                                    │   │
│  │ 5. Cast: float32 -> int32 (四舍五入)                                │   │
│  │ 6. Cast: int32 -> half -> int8                                      │   │
│  │ 7. 写入TokenStruct.token字段                                        │   │
│  │ 8. 写入反量化因子 1.0/dynamicScale 到 TokenStruct.scale字段         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  注意事项:                                                                  │
│  • SetDeqScale(deqScale) 用于控制Cast精度                                   │
│  • scale字段固定32B（UB_32B_ALIGN）                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.6 常见问题排查

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          常见问题排查                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Token丢失/乱序:                                                         │
│     • 检查magicVal_是否正确递增                                             │
│     • 检查flag是否正确设置和清理                                            │
│     • 检查RDMA发送完成标志是否发送                                          │
│                                                                             │
│  2. 死锁:                                                                   │
│     • 检查SyncAll是否所有核都到达                                           │
│     • 检查WaitIpcFlag是否有对应的SetIpcFlag                                 │
│     • 检查RDMA接收缓冲区是否正确初始化                                      │
│                                                                             │
│  3. 数据错乱:                                                               │
│     • 检查TokenStruct各字段偏移是否正确                                     │
│     • 检查跨步拷贝的gap参数是否正确                                         │
│     • 检查IPC缓冲区地址计算是否正确                                         │
│                                                                             │
│  4. 性能问题:                                                               │
│     • 检查UB分批大小是否合理                                                │
│     • 检查是否有不必要的同步                                                │
│     • 检查RDMA SQ深度是否足够                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 附录: 关键函数调用关系

```
Process()
├── ReorderTokens()
│   └── QuantProcess() (DynamicQuant模式)
├── SendDataToServer(destServerId)
│   └── AIVRDMAPostSend()
├── CreateInnerReduceInfo(serverIdx)
│   ├── PreloadForInner()
│   └── AIVRDMAPostSend() (发送Inner表)
├── CreateOuterReduceInfo()
│   └── PreloadForOuter()
├── Win2Ipc()
│   ├── GetSelfServerTokenInfo() (本Server)
│   └── GetArrivedTokenInfo() (远端Server)
├── SetIpcFlag()
├── WaitIpcFlag()
├── Ipc2Out()
├── CleanUp()
└── CopyPerformanceInfo()
```

---

*文档生成时间: 2026-03-13*
*基于源文件: moe_distribute_dispatch_a2_layered.h*
