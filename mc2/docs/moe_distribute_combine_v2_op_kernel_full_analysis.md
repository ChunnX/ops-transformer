# moe_distribute_combine_v2/op_kernel 全面分析报告

## 1. 目录结构与文件职责

```
moe_distribute_combine_v2/op_kernel/
├── moe_distribute_combine_v2.cpp              # 内核入口，模板实例化与分发
├── moe_distribute_combine_v2_tiling_key.h     # TilingKey模板参数声明与编译期选择
├── moe_distribute_combine_v2_tiling.h         # A3/A5架构Tiling数据结构定义
├── moe_distribute_combine_tiling.h            # A2架构Tiling数据结构定义
├── moe_distribute_combine_v2.h                # A3架构MTE模式内核实现（主路径）
├── moe_distribute_combine_v2_quant.h          # Int8动态量化/反量化工具类
├── moe_distribute_combine_v2_layered.h        # A3架构分层(Hierarchy)模式内核实现
├── moe_distribute_combine_a2.h                # A2架构MTE模式内核实现
├── moe_distribute_combine_a2_layered.h        # A2架构分层+RDMA模式内核实现
├── moe_distribute_combine_a2_layered_aicpu.h  # A2架构分层+AiCPU通信模式
└── arch35/
    ├── moe_distribute_combine_arch35.h        # A5(arch35)架构内核实现
    └── moe_distribute_combine_v2_host_kfc.h   # A5 host侧KFC辅助
```

## 2. 模板参数与TilingKey编码

### 2.1 模板参数声明 (`moe_distribute_combine_v2_tiling_key.h`, 第30-38行)

| 参数位置 | 宏名 | 含义 | 可选值 |
|---------|------|------|--------|
| 1 | `TILINGKEY_TP` | 是否启用TP（张量并行）ReduceScatter | 0=禁用, 1=启用 |
| 2 | `TILINGKEY_QUANT_MODE` | 量化模式 | 0=NO_QUANT, 1=INT8_QUANT |
| 3 | `TILINGKEY_LAYERED_MODE` | 分层通信模式 | 0=MTE, 1=AICPU, 2=HIERARCHY |
| 4 | `ARCH_TAG` | 目标架构标签 | 0=A2, 1=A3, 2=A5 |

### 2.2 编译期实例化组合

**A2 (arch32) kernel侧:**
- (TP=0, NO_QUANT, MTE) → `MoeDistributeCombineA2` + `MoeDistributeCombineA2TilingData`
- (TP=0, NO_QUANT, AICPU) → `MoeDistributeCombineA2Layered` + `MoeDistributeCombineA2TilingData`
- (TP=0, INT8_QUANT, AICPU) → `MoeDistributeCombineA2Layered` + `MoeDistributeCombineA2TilingData`

**A3 kernel侧:**
- (TP=0/1, NO_QUANT/INT8_QUANT, MTE) → `MoeDistributeCombineV2` + `MoeDistributeCombineV2TilingData`
- (TP=0, NO_QUANT/INT8_QUANT, HIERARCHY) → `MoeDistributeCombineV2Layered` + `MoeDistributeCombineV2TilingData`

**A5 (arch35) kernel侧:**
- (TP=0, NO_QUANT, MTE) → `MoeDistributeCombineA5` + `MoeDistributeCombineV2TilingData`

## 3. Tiling数据结构

### 3.1 A2架构: `MoeDistributeCombineA2Info` (moe_distribute_combine_tiling.h, 第21-41行)

```
偏移   字段                    类型       说明
─────  ──────────────────────  ────────   ─────────────────────
0      epWorldSize             uint32     EP组网卡数
4      tpWorldSize             uint32     TP组网卡数
8      epRankId                uint32     本卡EP Rank
12     tpRankId                uint32     本卡TP Rank
16     expertSharedType        uint32     专家共享类型
20     sharedExpertRankNum     uint32     共享专家占用卡数
24     moeExpertNum            uint32     MoE专家总数
28     zeroExpertNum           uint32     零专家数量
32     copyExpertNum           uint32     拷贝专家数量
36     constExpertNum          uint32     常量专家数量
40     globalBs                uint32     全局Batch*SeqLen (BS * worldSize)
44     bs                      uint32     本卡Batch*SeqLen
48     k                       uint32     TopK值
52     h                       uint32     隐层维度
56     aivNum                  uint32     使用的AIV核数
60     totalUbSize             uint64     UB总大小
68     isTokenMask             bool       是否启用一维Token掩码
69     isExpertMask            bool       是否启用二维专家掩码
70     reserved[7]             int8[7]    内存对齐填充
```

### 3.2 A3/A5架构: `MoeDistributeCombineV2Info` (moe_distribute_combine_v2_tiling.h, 第22-55行)

在A2 Info基础上增加:
```
偏移   字段                    类型       说明
─────  ──────────────────────  ────────   ─────────────────────
...    expertShardType         uint32     (A2 expertSharedType的A3名称)
...    sharedExpertNum         uint32     共享专家数量（而非卡数）
...    moeExpertPerRankNum     uint32     每卡MoE专家数
...    a                       uint32     对齐维度(A3特有)
...    hasSharedExpertX        bool       是否有共享专家输入X
...    hasElasticInfo          bool       是否有弹性缩容信息
...    isPerformance           bool       是否记录性能信息
...    isMc2Context            bool       是否使用Mc2Context
...    totalWinSizeEp          uint64     EP通信Window总大小
...    totalWinSizeTp          uint64     TP通信Window总大小
...    armAvgFactor            float      AddRmsNorm的平均因子
...    epsilon                 float      AddRmsNorm的epsilon
...    bufferNum               uint32     缓冲区数量
```

## 4. 内核入口与分发 (`moe_distribute_combine_v2.cpp`)

### 4.1 全局入口函数

```
moe_distribute_combine_v2<HasTp, QuantMode, LayeredMode, ArchTag>(
    expandX, expertIds, assistInfoForCombine, epSendCount, scales,
    tpSendCount, xActiveMask, activationScale, weightScale, groupList,
    expandScales, sharedExpertX, elasticInfo, oriX, constExpertAlpha1,
    constExpertAlpha2, constExpertV, performanceInfo, XOut, workspaceGM, tilingGM)
```

第62-69行定义了全局入口，根据编译期模板参数分发到不同的实现类。

### 4.2 分发逻辑 (第72-146行)

```
┌─────────────────────────────────────────────────────────┐
│               REGISTER_TILING_DEFAULT                    │
│             (默认Tiling结构)                              │
├─────────────────────────────────────────────────────────┤
│  A5路径 (arch35)                                        │
│  ├─ MoeDistributeCombineA5<ExpandXType, int32_t>       │
├─────────────────────────────────────────────────────────┤
│  A2路径 (arch32)                                        │
│  ├─ NO_QUANT + MTE → MoeDistributeCombineA2            │
│  ├─ NO_QUANT + AICPU → MoeDistributeCombineA2Layered   │
│  │                       (根据dataplaneMode选择          │
│  │                        AICPU或AIV子路径)              │
│  └─ INT8_QUANT + AICPU → MoeDistributeCombineA2Layered │
├─────────────────────────────────────────────────────────┤
│  A3路径                                                 │
│  ├─ MTE → ExecMoeDistributeCombineV2<...>              │
│  │         (封装MoeDistributeCombineV2类)               │
│  └─ HIERARCHY → MoeDistributeCombineV2Layered          │
│                   (支持NO_QUANT/INT8_QUANT)              │
└─────────────────────────────────────────────────────────┘
```

## 5. A3架构 MTE模式: `MoeDistributeCombineV2` (moe_distribute_combine_v2.h)

### 5.1 模板参数

```cpp
template<typename ExpandXType, typename XType, typename ExpandIdxType,
         bool IsNeedReduceScatter, bool IsInt8Quant, bool HasAddRmsNorm>
```

| 参数 | 含义 |
|------|------|
| ExpandXType | 输入数据类型 (bf16/fp16) |
| XType | 输出/计算类型 |
| ExpandIdxType | 索引类型 (int32) |
| IsNeedReduceScatter | 是否需要TP ReduceScatter |
| IsInt8Quant | 是否启用Int8动态量化 |
| HasAddRmsNorm | 是否融合Add+RmsNorm |

### 5.2 整体处理流程 (Process(), 第1473-1485行)

```
Process()
  │
  ├─ if IsNeedReduceScatter:
  │    └─ ReduceScatterTrans()        // TP=2时，将远端TP数据搬入本卡window
  │
  ├─ BuffInit()                        // 初始化UB缓冲区
  │
  ├─ SetWaitTpStatusAndDisPatch()      // 等待TP就绪 + AlltoAll发送
  │    ├─ [TP] 设置状态标志 + 等待对端
  │    └─ ExpertAlltoAllDispatchCopyAdd()
  │         ├─ 读取expandIdx路由表
  │         ├─ 循环处理每个token:
  │         │    ├─ ExpertAlltoAllDispatchInnerCopyAdd()
  │         │    │    ├─ [RS] 从GM+TPWin读取 → Add → 写回Window
  │         │    │    ├─ [Quant] 从GM读取 → Int8量化 → 写回Window
  │         │    │    └─ [Normal] 从GM读取 → 直接写回Window
  │         │    └─ 写状态标志通知对端
  │         └─ 错位发送(epRankId偏移)避免热 contend
  │
  ├─ PipeBarrier<PIPE_ALL>()           // 确保AlltoAll完成
  │
  ├─ AlltoAllBuffInitAndMaskCal()       // 重置UB + 计算Mask
  │    ├─ AlltoAllCommBuffInit()        // 分配UB缓冲区
  │    ├─ [TokenMask] TokenMaskCalCnt() // 计算一维Mask
  │    ├─ [ExpertMask] ExpertMaskCalCnt() // 计算二维Mask
  │    ├─ [SpecialExpert] GenerateActiveMask() + MaskSpecialExpert()
  │    └─ [Performance] 初始化性能缓冲
  │
  └─ LocalWindowCopy()                 // 核心计算阶段
       ├─ 分核按activeMaskBsCnt划分
       ├─ ExpertScaleCopy()            // 预取专家权重scales
       ├─ while循环等待所有token完成:
       │    ├─ WaitDispatch()          // 轮询状态标志等待数据到达
       │    ├─ ProcessExpert()         // 处理各类专家
       │    │    ├─ ProcessMoeExpert()   // MoE专家: win→UB→cast→muls→add
       │    │    ├─ ProcessCopyExpert()  // 拷贝专家: oriX→cast→muls→add
       │    │    ├─ ProcessConstantExpert() // 常量专家: softmax(alpha1*x + alpha2*v)
       │    │    └─ 共享专家: win→dequant→add
       │    ├─ [AddRmsNorm] AddRmsNormAddCompute() + RmsNormCompute()
       │    └─ 结果搬出: cast→fp16 → DataCopy到XOut
       └─ [Performance] 原子写性能信息
```

### 5.3 Window地址计算 (第101-123行)

```
GetWinAddrByRankId(rankId, domain):
  ┌─ isMc2Context_:
  │    └─ mc2Context_->epHcclBuffer_[rankId] + STATE_SIZE + winDataSizeOffsetEp_
  ├─ domain == EP_DOMAIN:
  │    └─ GetBaseWindAddrByRankId(epWinContext_, rankId, epRankIdOriginal_)
  │         + winDataSizeOffsetEp_
  └─ domain == TP_DOMAIN:
       └─ GetBaseWindAddrByRankId(tpWinContext_, rankId, tpRankId_)
            + winDataSizeOffsetTp_

GetWinStateAddrByRankId(rankId, domain):
  ┌─ isMc2Context_:
  │    └─ mc2Context_->epHcclBuffer_[rankId] + winStatusOffset_
  ├─ EP_DOMAIN:
  │    └─ GetBaseWindStateAddrByRankId(epWinContext_, rankId, epRankIdOriginal_)
  │         + winStatusOffset_
  └─ TP_DOMAIN:
       └─ GetBaseWindStateAddrByRankId(tpWinContext_, rankId, tpRankId_)
            + winStatusOffset_
```

**Ping/Pong半区切换**: `winDataSizeOffsetEp_ = dataState_ * (totalWinSizeEp / 2UL)`，连续两次dispatch切换半区。`winStatusOffset_ = COMBINE_STATE_OFFSET + dataState_ * WIN_STATE_OFFSET`。

### 5.4 状态同步机制 (WaitDispatch, 第1009-1047行)

```
WaitDispatch(tokenIndex, performanceTimeStart, copyCount, beginIndex):
  1. 计算目标flag数量 (考虑ExpertMask和特殊专家)
  2. 读取Window状态区: stateGM = GetWinStateAddrByRankId(epRankIdOriginal_, EP_DOMAIN)
     + tokenIndex * flagRcvCount_ * stateOffset_
  3. DataCopy读入状态到UB
  4. Sum求和，检查是否所有flag都置位
  5. 若完成: 清零状态区(DataCopy写回stateResetTensor_)
  6. 若未完成: 返回false，由外层while循环重试
```

### 5.5 专家处理逻辑 (ProcessExpert, 第1275-1363行)

```
ProcessExpert(tokenIndex, processLen):
  ┌─ Duplicate(sumFloatBufLocal_, 0.0f, axisH_)    // 清零累加器
  │
  ├─ 如果无特殊专家:
  │    └─ for topKId in [0, axisK_):
  │         ├─ [ExpertMask] 检查mask跳过无效专家
  │         └─ ProcessMoeExpert(tokenIndexOffset, topKId, scaleVal)
  │
  ├─ 如果有特殊专家:
  │    └─ for topKId in [0, axisK_):
  │         ├─ 读取expert_id
  │         ├─ expert_id < moeExpertOriginalNum_ → ProcessMoeExpert()
  │         ├─ expert_id < ... + zeroExpertNum_ → 跳过 (零专家)
  │         ├─ expert_id < ... + copyExpertNum_ → ProcessCopyExpert()
  │         └─ expert_id < ... + constExpertNum_ → ProcessConstantExpert()
  │
  ├─ for topKId in [axisK_, axisK_ + sharedExpertNum_):
  │    └─ 从Window读取共享专家结果 → [dequant] → add累加
  │
  └─ [hasSharedExpertX] 从sharedExpertX读取 → add累加
```

### 5.6 常量专家处理 (ProcessConstantExpert, 第1143-1199行)

常量专家计算: `scale * (alpha1 * softmax(W1*x + W2*x) * x + alpha2 * softmax(...) * v)`

```
1. 读取oriX输入token → cast到float
2. 计算 alpha1 = Wc1 · x (内积) — CalConstExpertAlpha()
3. 计算 alpha2 = Wc2 · x (内积) — CalConstExpertAlpha()
4. softmax: exp(alpha - max) / sum(exp)
5. 读取常量专家向量v
6. 计算 alpha1 * x + alpha2 * v
7. 乘以专家权重scale → add到累加器
```

## 6. Int8动态量化 (`moe_distribute_combine_v2_quant.h`)

### 6.1 量化参数

| 数据类型 | 量化粒度 (SCALE_GRANU) | 每token scale数 |
|---------|----------------------|----------------|
| FP16    | 16元素/组             | axisH / 16     |
| BF16    | 8元素/组              | axisH / 8      |

量化缩放系数: `scaleValFloat_ = 1.0f / 127.0f`

### 6.2 Int8量化流水线 (Int8QuantProcess, 第63-88行)

```
FP16/BF16 Token → Abs → BlockReduceMax(每16/8个取最大值)
  → Muls(1/127) → Cast(FP16) → Brcb(扩展到H)
  → Div(data/scale) → Cast(FP16) → Cast(int8)
  → 拼接 [int8数据(axisH/2字节) + scale值] 写回Window
```

### 6.3 Int8反量化流水线 (Int8DequantProcess, 第90-108行)

```
[int8数据 + scale值] → Cast(int8→FP16/BF16) → Cast(→FP32)
  → Brcb(scale扩展) → Mul(data*scale) → Cast(FP32→输出类型)
```

### 6.4 量化/非量化路径对比表

| 维度 | 非量化 | Int8量化 |
|------|-------|---------|
| Window数据格式 | FP16/BF16 (H*sizeof(bf16/fp16)) | int8 + scale (H/2 + scaleNum*2 字节) |
| 发送带宽节省 | 无 | 约50% (H维度减半) |
| 量化阶段 | 无 | Abs→BlockReduceMax→Muls→Brcb→Div→Cast→int8 |
| 反量化阶段 | 无 | Cast→Brcb→Mul→Cast |
| 累加精度 | FP32 | FP32 (反量化后) |
| UB额外开销 | 无 | xAbsBuf + xMaxBuf + xScaleMulBuf (约3*H*4字节) |

## 7. A2架构 MTE模式: `MoeDistributeCombineA2` (moe_distribute_combine_a2.h)

### 7.1 处理流程 (Process, 第671-682行)

```
Process()
  ├─ CalXActiveMask()              // 计算激活Mask
  │    ├─ [TokenMask] TokenActiveMaskCalCnt()
  │    └─ [ExpertMask] 加载二维Mask
  ├─ AlltoAllDispatch()            // 数据分发
  │    ├─ 读取sendCount
  │    ├─ isSingleServer_?
  │    │    ├─ true: SingleServerDispatch()
  │    │    │    └─ 直接IPC写入对端Window + 写flag
  │    │    └─ false: ConstructBatchWriteInfo() → BatchWrite(HCCL) → MultiServerDispatch()
  │    └─ 切换bufferId (Ping/Pong)
  ├─ Preload()                     // 预加载计算所需数据
  │    ├─ 分核加载expertIds, expandScales, expandIdx片段
  │    ├─ 统计每个专家接收到的token数 (原子Add到GM)
  │    └─ 计算前缀和得到每个专家在Window中的偏移
  ├─ WaitDispatch()                // 等待所有rank数据到达
  │    └─ 轮询Window flag
  ├─ LocalWindowCopy()             // 核心计算
  │    └─ for token × topK:
  │         ├─ ProcessMoeAndCopyExpert() / ProcessConstantExpert()
  │         └─ 累加 → Cast → 输出
  ├─ CopyPerformanceInfo()
  └─ hccl_.Finalize()
```

### 7.2 单服务器 vs 多服务器分发

**单服务器 (SingleServerDispatch, 第396-423行)**:
- 直接IPC: expandX → UB → windowOut → 对端windowIn
- 写flag通知对端

**多服务器 (ConstructBatchWriteInfo + MultiServerDispatch, 第426-504行)**:
- 构造BatchWriteItem结构体 (srcAddr, dstAddr, length, dataType, dstRankId)
- 调用 `hccl_.BatchWrite<true>()` 批量RDMA发送
- 本地数据: outWindow → inWindow (本卡同号rank的转发)

## 8. A2分层模式: `MoeDistributeCombineA2Layered` (moe_distribute_combine_a2_layered.h)

### 8.1 处理流程 (Process, 第1050-1071行)

```
Process()
  ├─ GM2IPC()                      // 将expandX数据通过IPC发到同机器其他rank
  ├─ WaitIPC()                     // 等待所有IPC写入完成
  ├─ [core < 32] SumToWindow()     // 机内同号卡token累加 → 量化 → 写RDMA发送区
  ├─ [core < 32+serverNum] AlltoAllServerDispatch() // RDMA发送到跨机rank
  ├─ [core == 0] 切换bufferId + magicValue
  ├─ Preload()                     // 预加载outer路由表
  ├─ WaitDispatch()                // 等待跨机RDMA数据到达
  ├─ SumToServer()                 // 跨机token累加 → 输出
  ├─ CopyPerformanceInfo()
  └─ hccl_.Finalize()
```

### 8.2 分层通信架构

```
┌──────────────────────────────────────────────────────┐
│                    A2 分层通信架构                      │
├──────────────────────────────────────────────────────┤
│                                                      │
│  Server 0 (8卡)           Server 1 (8卡)             │
│  ┌─────────────────┐     ┌─────────────────┐        │
│  │ Rank0  Rank1... │     │ Rank8  Rank9... │        │
│  │  │      │       │     │  │      │       │        │
│  │  ▼      ▼       │     │  ▼      ▼       │        │
│  │ [IPC共享内存区]  │     │ [IPC共享内存区]  │        │
│  │   ↕             │     │   ↕             │        │
│  │ [机内累加]      │     │ [机内累加]      │        │
│  │   │             │     │   │             │        │
│  │   ▼             │     │   ▼             │        │
│  │ [RDMA发送区]    │←──→│ [RDMA发送区]    │        │
│  │   ↕             │ RDMA│   ↕             │        │
│  │ [RDMA接收区]    │     │ [RDMA接收区]    │        │
│  │   │             │     │   │             │        │
│  │   ▼             │     │   ▼             │        │
│  │ [跨机累加]      │     │ [跨机累加]      │        │
│  └─────────────────┘     └─────────────────┘        │
│                                                      │
│  第一层: GM2IPC → 机内AlltoAll (IPC)                 │
│  第二层: SumToWindow → 机内Reduce → 量化              │
│  第三层: AlltoAllServerDispatch → RDMA跨机AlltoAll    │
│  第四层: WaitDispatch → SumToServer → 跨机Reduce      │
└──────────────────────────────────────────────────────┘
```

### 8.3 RDMA发送实现 (AIVRDMAPostSend, 第197-282行)

直接操作RoCEv2硬件:
1. 构造WQE (Work Queue Element): opcode=RDMA_WRITE, 设置SGL
2. 写入发送队列SQ
3. cacheWriteThrough刷新WQE和SGE
4. 写DoorBell通知RNIC

### 8.4 核心分配策略

| 阶段 | 使用核数 | 分配逻辑 |
|------|---------|---------|
| GM2IPC | 全部aivNum核 | 按worldSize分核处理目标rank |
| SumToWindow | 前32核 (IPC_REDUCE_USED_CORE_NUM) | 按server分核，每server内按token流水 |
| AlltoAllServerDispatch | 32 ~ 32+serverNum核 | 每核处理一个目标server的RDMA |
| Preload/SumToServer | 前8核 (reduceCore) | 按BS/token分核 |

## 9. A3分层模式: `MoeDistributeCombineV2Layered` (moe_distribute_combine_v2_layered.h)

### 9.1 与A2分层模式的核心差异

| 维度 | A2 Layered | A3 Layered (V2) |
|------|-----------|----------------|
| 通信库 | HCCL AICPU模式 | HCCL标准模式 |
| IPC地址获取 | `hccl_.GetWindowsInAddr()` | `GetWindInAddrByRankId()` (直接访问HcclOpResParam) |
| 状态同步 | HCCL BatchWrite + int32状态 | RDMA + uint64 flag + magicValue |
| Window管理 | 固定bufferId | bufferId + magicValue双重控制 |
| 服务器规模 | SERVER_RANK_SIZE=8 | SERVER_RANK_SIZE=16 |
| RDMA数据区 | 100MB (RDMA_DATA_SIZE) | 400MB |
| IPC数据区偏移 | 4MB (IPC_DATA_OFFSET) | 4MB |
| UB大小 | TBUF_SIZE=185KB | TBUF_SIZE=185KB |

### 9.2 处理流程 (Process, 第1096-1117行)

```
Process()
  ├─ GM2IPC()                      // 同A2分层: IPC发送token+scale
  ├─ WaitIPC()                     // 等待IPC同步
  ├─ [core < 32] SumToWindow()     // 机内累加 + 量化 + 写发送区
  │    ├─ SumToWindowLocalTensor() // 分配UB LocalTensor
  │    ├─ 读取inner路由表 (countInner, offsetInner)
  │    ├─ 按token循环:
  │    │    ├─ ServerInAdd()       // 从IPC区读取 + scale乘加
  │    │    └─ [完成一个token] SumToWindowAdd()
  │    │         ├─ [DynamicQuant] DynamicQuantProcess()
  │    │         │    ├─ FP16: Cast→Abs→BlockReduceMax→Muls→Brcb→Div→Cast→int8
  │    │         │    └─ BF16: Abs→BlockReduceMax→Muls→Brcb→Div→Cast→Cast→int8
  │    │         └─ 写RDMA flag通知
  │    └─ SyncAll
  ├─ [core < 32+serverNum] AlltoAllServerDispatch() // RDMA跨机发送
  ├─ [core == 0] RDMADataSwitch()  // 切换magicValue和bufferId
  ├─ Preload()                     // 预加载outer路由表
  ├─ WaitDispatch()                // 等待跨机数据
  └─ SumToServer()                 // 跨机Reduce累加 → 输出
```

### 9.3 MagicValue机制

```cpp
// 读取magicValue (在IPC共享区末尾偏移MAGIC_OFFSET处)
magicValue_ = magicGlobal_.GetValue(MAGIC_OFFSET / sizeof(uint64_t));

// 所有flag都加上magicValue做隔离
rdmaFlagLocal(0) = RDMA_TOKEN_ARRIVED_FLAG + magicValue_;
rdmaFlagLocal(0) = RDMA_TOKEN_END_FLAG + magicValue_;
GM2IPC_SYNC_FLAG + magicValue_

// 每次完成后递增magicValue
magicGlobal_.SetValue(MAGIC_OFFSET / sizeof(uint64_t), magicValue_ + 1);
```

### 9.4 量化路径对比

**FP16量化 (DynamicQuantProcess, 第580-600行)**:
```
sumFloat(FP32) → Cast(FP16) → Abs → BlockReduceMax(g16) → Muls(1/127)
  → Brcb(扩展scale) → Div(data/scale) → Cast(int8)
  → DataCopy [int8数据 + scale值]
```

**BF16量化 (DynamicQuantProcess, 第601-623行)**:
```
sumFloat(FP32) → Abs → BlockReduceMax(FP32,g32) → Muls(1/127)
  → Brcb(扩展FP32 scale) → Div(data/scale)
  → Cast(scale: FP32→BF16) → Cast(data: FP32→BF16) → Cast(BF16→int8)
  → DataCopy [int8数据 + scale值]
```

**FP16反量化 (DynamicQuantSumToServer, 第996-1015行)**:
```
[int8数据 + scale值] → Cast(int8→FP16) → Brcb(scale扩展)
  → MulAddDst(data*scale, 累加) → DataCopy(FP16输出)
```

**BF16反量化 (DynamicQuantSumToServer, 第1017-1041行)**:
```
[int8数据 + scale值] → Cast(int8→FP16) → Cast(FP16→FP32)
  + Cast(scale:BF16→FP32) → Brcb(扩展FP32 scale)
  → MulAddDst(data*scale, 累加) → Cast(FP32→BF16) → DataCopy(BF16输出)
```

## 10. A5架构: `MoeDistributeCombineA5` (arch35/moe_distribute_combine_arch35.h)

### 10.1 处理流程 (Process, 第598-608行)

```
Process()
  ├─ PrepareSendData()              // 准备发送数据
  │    ├─ PrepareInit()             // 读取sendCount
  │    └─ SharedAndMoePrepare()     // 按rank构造发送数据到sendBuf
  │         ├─ SplitRank()          // 按worldSize分核
  │         ├─ HandleSharedAndMoeRank() // 处理共享/普通专家
  │         └─ CopyAndPadData()     // 写发送大小和偏移
  ├─ Communication()                // AlltoAllv通信
  │    ├─ [aivId==0] hccl_.AlltoAllvWrite()  // 使用CCU通信
  │    ├─ CalculateInit()           // 通信期间预加载 (掩盖延迟)
  │    └─ [aivId==0] hccl_.Wait()
  └─ Calculate()                    // 本地计算
       ├─ [TokenMask] TokenMaskCalCnt()
       ├─ 按token分核:
       │    ├─ CalculateMoeResult()  // MoE专家: recv → cast → muls → add
       │    ├─ 共享专家add
       │    ├─ [hasSharedExpertX] HandleSharedExpertX()
       │    └─ Cast → DataCopy输出
       └─ hccl_.Finalize()
```

### 10.2 A5特有设计

- **CCU通信**: 使用 `Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU>` 替代A2的AICPU模式
- **AlltoAllvWrite**: 直接调用HCCL AlltoAllv原语，无需手动RDMA
- **计算通信掩盖**: `CalculateInit()` 在 `hccl_.Wait()` 之前执行，用计算掩盖通信延迟
- **Ping/Pong缓冲区**: 通过状态值(0/1)切换recvBuf的上下半区 (第196-205行)
- **按token切分H维度**: 当activeMaskBsCnt < aivNum时，将H维度切分给多个核并行处理

## 11. UB内存管理

### 11.1 A3 MTE模式 UB布局 (AlltoAllCommBuffInit, 第686-716行)

```
┌──────────────────────────────────────────┐
│              UB (~256KB)                  │
├──────────────────────────────────────────┤
│ tokenBuf_         │ maxSizeTokenBuf      │  token临时缓冲 / mask
├───────────────────┼──────────────────────┤
│ rowTmpFloatBuf_   │ maxSizeRowTmpFloatBuf│  FP32中间结果
├───────────────────┼──────────────────────┤
│ mulBuf_           │ mulBufSize           │  乘法中间结果
├───────────────────┼──────────────────────┤
│ sumFloatBuf_      │ hFloatAlign32Size    │  累加缓冲(FP32)
├───────────────────┼──────────────────────┤
│ moeSumQueue_      │ hExpandXAlign32Size  │  MoE专家读取队列
├───────────────────┼──────────────────────┤
│ stateBuf_         │ flagRcvCount*512     │  状态读取
├───────────────────┼──────────────────────┤
│ stateSumBuf_      │ UB_ALIGN(32B)        │  状态求和
├───────────────────┼──────────────────────┤
│ stateResetBuf_    │ flagRcvCount*512     │  状态清零
├───────────────────┼──────────────────────┤
│ [Int8量化额外]    │                      │
│  xAbsBuf_         │ hFloatAlign256Size   │
│  xMaxBuf_         │ hFloatAlign256/8     │
│  xScaleMulBuf_    │ hFloatAlign256Size   │
│  winTpSendCountFloatBuf_ │ hFloatAlign32 │
├───────────────────┼──────────────────────┤
│ [AddRmsNorm额外]  │                      │
│  gammaBuf_        │ hExpandXAlign32Size  │
│  reduceFp32Buf_   │ 128*4 = 512B         │
└───────────────────┴──────────────────────┘
```

### 11.2 A2 Layered模式 UB布局 (BuffInit, 第383-397行)

```
┌──────────────────────────────────────────┐
│              UB                           │
├──────────────────────────────────────────┤
│ statusBuf_   │ worldSize * 32B           │  状态同步
├──────────────┼───────────────────────────┤
│ rdmaInBuf_   │ 32B                       │  RDMA DoorBell
├──────────────┼───────────────────────────┤
│ rdmaInBuf2_  │ 32B                       │  RDMA Head更新
├──────────────┼───────────────────────────┤
│ tBuf         │ 185KB                     │  通用计算缓冲
│  ├ sendCountLocal                         │  发送计数
│  ├ expandScalesLocal                      │  专家权重
│  ├ dataInLocal + inUbTemp                 │  token数据+scale
│  ├ castInFloatLocal                       │  FP32转换
│  ├ sumFloatLocal                          │  FP32累加
│  ├ [量化] reduceMaxOut/scaleData          │  量化中间
│  └ offsetReduce/countReduce              │  路由偏移
└──────────────────────────────────────────┘
```

## 12. 函数调用关系

```
moe_distribute_combine_v2 (内核入口)
  │
  ├── [A3+MTE] ExecMoeDistributeCombineV2
  │     └── MoeDistributeCombineV2::Init → Process
  │           ├─ ReduceScatterTrans
  │           ├─ BuffInit
  │           ├─ SetWaitTpStatusAndDisPatch
  │           │    └─ ExpertAlltoAllDispatchCopyAdd
  │           │         └─ ExpertAlltoAllDispatchInnerCopyAdd
  │           │              ├─ CustomAdd (BF16需FP32中间)
  │           │              ├─ [Quant] MoeDistributeCombineQuant::Int8QuantProcess
  │           │              └─ [Normal] DataCopy直接搬移
  │           ├─ AlltoAllBuffInitAndMaskCal
  │           │    ├─ TokenMaskCalCnt
  │           │    ├─ ExpertMaskCalCnt
  │           │    ├─ GenerateActiveMask → MaskAlign
  │           │    └─ MaskSpecialExpert
  │           └─ LocalWindowCopy
  │                ├─ ExpertScaleCopy
  │                ├─ WaitDispatch
  │                ├─ ProcessExpert
  │                │    ├─ ProcessMoeExpert / [dequant]
  │                │    ├─ ProcessCopyExpert
  │                │    └─ ProcessConstantExpert → CalConstExpertAlpha
  │                ├─ [AddRmsNorm] AddRmsNormAddCompute + AddRmsNormRmsNormCompute
  │                └─ CopyPerformanceInfo
  │
  ├── [A3+HIERARCHY] MoeDistributeCombineV2Layered::Init → Process
  │     ├─ GM2IPC
  │     ├─ WaitIPC
  │     ├─ SumToWindow → SumToWindowAdd → ServerInAdd / DynamicQuantProcess
  │     ├─ AlltoAllServerDispatch → WaitStatusAndDispatch
  │     ├─ RDMADataSwitch
  │     ├─ Preload
  │     ├─ WaitDispatch
  │     └─ SumToServer → DynamicQuantSumToServer
  │
  ├── [A2+MTE] MoeDistributeCombineA2::Init → Process
  │     ├─ CalXActiveMask → TokenActiveMaskCalCnt
  │     ├─ AlltoAllDispatch → SingleServerDispatch / ConstructBatchWriteInfo + MultiServerDispatch
  │     ├─ Preload
  │     ├─ WaitDispatch
  │     └─ LocalWindowCopy → ProcessMoeAndCopyExpert / ProcessConstantExpert
  │
  ├── [A2+AICPU] MoeDistributeCombineA2Layered::Init → Process
  │     ├─ GM2IPC → WaitIPC → SumToWindow → AlltoAllServerDispatch → Preload → WaitDispatch → SumToServer
  │     └─ AIVRDMAPostSend (直接RDMA操作)
  │
  ├── [A2+AICPU+DATAPLANE_AICPU] MoeDistributeCombineA2LayeredAicpu::Init → Process
  │     ├─ AlltoAllDispatch → SumToWindow → AlltoAllServerDispatch → SetStatus → Preload → WaitDispatch → SumToServer
  │     └─ BatchWrite (HCCL批量写)
  │
  └── [A5+MTE] MoeDistributeCombineA5::Init → Process
        ├─ PrepareSendData → SharedAndMoePrepare → HandleSharedAndMoeRank
        ├─ Communication → AlltoAllvWrite → CalculateInit → Wait
        └─ Calculate → HandleCalculateToken → CalculateMoeResult + HandleSharedExpertX
```

## 13. 依赖关系

```
moe_distribute_combine_v2.cpp
  ├── moe_distribute_combine_v2_tiling_key.h     # TilingKey宏定义
  ├── moe_distribute_combine_v2_tiling.h          # A3 Tiling结构
  ├── moe_distribute_combine_v2.h                 # A3 MTE实现
  │     ├── moe_distribute_combine_v2_quant.h     # 量化工具类
  │     ├── mc2_moe_context.h                     # MoE上下文
  │     ├── moe_distribute_v2_constant.h          # 常量定义
  │     ├── moe_distribute_base.h                 # Window操作基类
  │     ├── check_winsize.h                       # Window大小检查
  │     ├── moe_distribute_v2_base.h              # V2基类(InitWinState等)
  │     └── moe_distribute_elastic.h              # 弹性缩容
  ├── moe_distribute_combine_v2_layered.h          # A3 HIERARCHY实现
  │     ├── moe_distribute_base.h
  │     └── moe_distribute_v2_base.h
  ├── moe_distribute_combine_a2.h                  # A2 MTE实现
  │     ├── moe_distribute_base.h
  │     ├── mc2_kernel_utils.h                    # 核工具函数
  │     └── moe_distribute_combine_tiling.h       # A2 Tiling结构
  ├── moe_distribute_combine_a2_layered.h          # A2 RDMA分层实现
  │     ├── moe_distribute_base.h
  │     ├── mc2_kernel_utils.h
  │     └── moe_distribute_a2_base.h              # A2地址计算(AddrInfo)
  ├── moe_distribute_combine_a2_layered_aicpu.h    # A2 AiCPU分层实现
  │     ├── moe_distribute_base.h
  │     └── mc2_kernel_utils.h
  └── arch35/moe_distribute_combine_arch35.h       # A5 CCU实现
        └── mc2_kernel_utils.h
```

## 14. 关键约束与注意事项

1. **Window大小限制**: 需确保HCCL Window大小 >= `moeExpertNum * axisBS * (hExpandXTypeSize + extraInfoSize) + stateSize`，代码中有CheckWindowSize校验
2. **Ping/Pong缓冲**: 连续两次dispatch必须切换半区，否则数据竞争。A2 Layered使用bufferId^1切换，A3 Layered使用magicValue+1
3. **Cache一致性**: 所有状态标志写入后必须调用 `DataCacheCleanAndInvalid` 确保对端可见
4. **RDMA WQE深度**: A2Layered的AIVRDMAPostSend中需轮询SQ深度避免溢出
5. **量化粒度对齐**: FP16每16个元素一个scale，BF16每8个元素一个scale，axisH必须是granularity的整数倍
6. **TP=2限制**: ReduceScatterTrans仅适配tpWorldSize_=2的场景(第787行注释)
7. **核数约束**: A2 Layered的SumToWindow需要至少32核(IPC_REDUCE_USED_CORE_NUM)，SumToServer仅使用前8核
8. **专家类型顺序**: expert_id必须按 [moeExpert, zeroExpert, copyExpert, constExpert] 顺序编码
9. **A5计算通信掩盖**: CalculateInit在Wait之前执行，利用通信延迟掩盖计算开销
10. **A3 MTE模式状态标志**: 使用float类型(1.0)作为flag，通过Sum+范围判断确认所有flag到达，而非逐个检查

## 15. 同族变体核心差异总结

| 特性 | A3+MTE | A3+HIERARCHY | A2+MTE | A2+AICPU | A2+AICPU(Dataplane:AICPU) | A5 |
|------|--------|-------------|--------|----------|--------------------------|-----|
| 通信方式 | HCCL Window IPC/RDMA | 三层(IPC+机内Reduce+RDMA) | HCCL BatchWrite/IPC | 三层(IPC+机内Reduce+RDMA) | 三层(IPC+机内Reduce+HCCL) | HCCL CCU AlltoAllv |
| 量化支持 | Int8量化 | Int8量化 | 无 | Int8量化 | 无 | 无 |
| AddRmsNorm融合 | 支持 | 不支持 | 不支持 | 不支持 | 不支持 | 不支持 |
| TP ReduceScatter | 支持 | 不支持 | 不支持 | 不支持 | 不支持 | 不支持 |
| 特殊专家 | 零/拷贝/常量专家 | 无 | 零/拷贝/常量专家 | 无 | 无 | 无 |
| 共享专家 | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| 弹性缩容 | 支持 | 不支持 | 不支持 | 不支持 | 不支持 | 不支持 |
| Window管理 | dataState半区切换 | magicValue+bufferId | bufferId切换 | magicValue+bufferId | bufferId+状态值 | 状态值半区切换 |
| 状态标志类型 | float(1.0) | uint64+magic | uint32(0xFFFFFFFF) | uint64+magic | int32+sumTarget | int32状态 |
| 核心数量约束 | 全部核 | 32核累加+serverNum核RDMA+8核Reduce | 全部核 | 32核累加+serverNum核RDMA+8核Reduce | serverNum核累加+serverNum核RDMA+8核Reduce | 全部核 |
