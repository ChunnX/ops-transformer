# MoeDistributeDispatchV2 op_kernel 目录详细分析报告

## 1. 目录概述

**路径**: `mc2/moe_distribute_dispatch_v2/op_kernel/`

**功能定位**: MoE (Mixture of Experts) 分布式分发算子的设备侧 Kernel 实现。负责将本地 Token 按照路由表 (expertIds) 分发到 EP (Expert Parallelism) 域内各卡的 Window 共享内存区，支持 TP (Tensor Parallelism) 域的 AllGather 扩展。

**核心设计理念**:
- 基于硬件架构（A2/A3/A5）的模板编译特化
- IPC (进程间通信) + RDMA (远程直接内存访问) 的分层通信策略
- Ping/Pong 双缓冲机制避免跨调用数据竞争
- 支持 5 种量化模式 (UNQUANT / STATIC / PERTOKEN / PERGROUP / MX)

**文件列表** (20个文件):

| 文件名 | 职责 |
|--------|------|
| `moe_distribute_dispatch_v2.cpp` | Kernel 入口，模板分发 |
| `moe_distribute_dispatch_v2.h` | 主 Dispatch 类 (A3 MTE 路径) |
| `moe_distribute_dispatch_v2_full_mesh.h` | Full Mesh 通信变体 |
| `moe_distribute_dispatch_v2_layered.h` | A3 分层 (Hierarchy) 通信变体 |
| `moe_distribute_dispatch_v2_quant.h` | 量化辅助类 |
| `moe_distribute_dispatch_v2_tiling.h` | A3/A5 Tiling 数据结构 |
| `moe_distribute_dispatch_v2_tiling_key.h` | TilingKey 模板参数声明 |
| `moe_distribute_dispatch_tiling.h` | A2 Tiling 数据结构 |
| `moe_distribute_dispatch_a2.h` | A2 架构 Dispatch 类 |
| `moe_distribute_dispatch_a2_layered.h` | A2 分层通信变体 (AIV 模式) |
| `moe_distribute_dispatch_a2_layered_aicpu.h` | A2 分层通信变体 (AICPU 模式) |
| `moe_distribute_a2_base.h` | A2 地址管理基类与 HCCL Buffer 布局 |
| `moe_distribute_v2_base.h` | 通用 RDMA 发送与状态初始化 |
| `moe_distribute_v2_constant.h` | 全局常量定义 |
| `moe_distribute_v2_base.h` | Wait, 公共函数 |
| `moe_distribute_elastic.h` | 弹性缩放信息处理 |
| `moe_distribute_dispatch_v2_host_kfc.h` | Host 侧 Kernel Function Call 接口 |
| `quantize_functions.h` | A5 专用量化底层函数 (MX/PerGroup) |
| `common.h` | 对齐、Histogram、ReduceSum 工具函数 |
| `check_winsize.h` | Window 大小校验工具 |
| `arch35/moe_distribute_dispatch_arch35.h` | A5 (arch35) 架构专用 Dispatch 类 (CCU 路径) |

---

## 2. 整体架构与数据流

### 2.1 硬件拓扑

```
┌─────────────────────────────────────────────────────────────┐
│                        多机集群                               │
│  ┌─────────────────────┐   ┌─────────────────────┐          │
│  │     Server 0         │   │     Server 1         │   ...   │
│  │  ┌───┐ ┌───┐       │   │  ┌───┐ ┌───┐       │          │
│  │  │NPU0│ │NPU1│ ...  │   │  │NPU0│ │NPU1│ ...  │          │
│  │  └─┬─┘ └─┬─┘       │   │  └─┬─┘ └─┬─┘       │          │
│  │    │ IPC  │          │   │    │ IPC  │          │          │
│  │    └──┬───┘          │   │    └──┬───┘          │          │
│  │  ┌───┴───┐           │   │  ┌───┴───┐           │          │
│  │  │共享内存│           │   │  │共享内存│           │          │
│  │  └───────┘           │   │  └───────┘           │          │
│  └────────┬────────────┘   └────────┬────────────┘          │
│           │ RoCEv2 RDMA              │ RoCEv2 RDMA           │
│           └──────────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘

每个 NPU 芯片内部:
┌──────────────────────────────┐
│ AIV Core 0 ─── AIV Core 1    │  ← 向量计算核
│    ...        ...             │
│ AIV Core N                    │
│   ↕ UB (~256KB/core)          │
│   ↕ GM (HBM, 共享)            │
│   ↕ Window (HCCL 共享区域)    │
└──────────────────────────────┘
```

### 2.2 Kernel 变体分发关系

```
moe_distribute_dispatch_v2.cpp (入口)
│
├── A5 (arch35) 路径
│   ├── CommMode == CCU → MoeDistributeDispatchA5 (arch35/xxx.h)
│   └── CommMode == MTE
│       ├── FullMesh → MoeDistributeDispatchV2FullMesh
│       └── 非FullMesh → MoeDistributeDispatchV2
│
├── A3 (910_93) 路径
│   ├── FullMesh → MoeDistributeDispatchV2FullMesh
│   ├── 非FullMesh → MoeDistributeDispatchV2
│   └── Hierarchy → MoeDistributeDispatchV2Layered (A3专用)
│
└── A2 (910B/950) 路径
    ├── CommMode == MTE → MoeDistributeDispatchA2
    └── CommMode == AICPU
        ├── AICPU 模式 → MoeDistributeDispatchA2LayeredAicpu
        └── AIV 模式 → MoeDistributeDispatchA2Layered
```

### 2.3 核心数据流 (MoeDistributeDispatchV2 主路径)

```
输入: x[BS, H], expertIds[BS, K], xActiveMask[BS, K]
                    │
                    ▼
            ┌──────────────┐
            │ AlltoAllDispatch │
            │  1. TokenActiveMaskCal (可选)  │
            │  2. ExpertActiveMaskCal (可选) │
            │  3. ZeroComputeExpertMaskCal   │
            │  4. SendToSharedExpert (共享专家)│
            │  5. SendToMoeExpert (MoE专家)  │
            └──────┬───────┘
                   │ 写 Window 共享区
                   ▼
            ┌──────────────┐
            │  SetStatus   │
            │  写 flag 到对端 WindowState │
            └──────┬───────┘
                   │
                   ▼
            ┌──────────────┐
            │ WaitDispatch │
            │  轮询等待所有对端写完 flag │
            │  核间同步 token cnt │
            └──────┬───────┘
                   │
                   ▼
            ┌──────────────┐
            │ LocalWindowCopy │
            │  从 Window 拷贝到输出 GM │
            │  提取 expandIdx 三元组    │
            │  拷贝 dynamicScales      │
            └──────┬───────┘
                   │
                   ▼
            ┌──────────────────────────┐
            │ TP AllGather (可选)       │
            │  AllGatherSetStatusAndWait│
            │  AllgatherProcessOut      │
            │  AllGatherSetExpertTokenNumsAndTpRecvCount │
            └──────┬───────────────────┘
                   │
                   ▼
            ┌──────────────┐
            │SetExpertTokenNums│
            │  输出 expertTokenNums │
            └──────┬───────┘
                   │
                   ▼
输出: expandXOut, dynamicScalesOut, assistInfoOut(expandIdx),
      expertTokenNumsOut, epSendCountsOut, tpSendCountsOut
```

---

## 3. 核心数据结构

### 3.1 Tiling 数据结构

#### MoeDistributeDispatchV2TilingData (A3/A5 使用)

```
┌──────────────────────────────────────────────┐
│ MoeDistributeDispatchV2TilingData             │
│ ┌──────────────────────────────────────────┐ │
│ │ Mc2InitTiling mc2InitTiling              │ │  HCCL 初始化参数
│ ├──────────────────────────────────────────┤ │
│ │ Mc2CcTiling mc2CcTiling1                 │ │  EP 域通信参数
│ ├──────────────────────────────────────────┤ │
│ │ Mc2CcTiling mc2CcTiling2                 │ │  TP 域通信参数
│ ├──────────────────────────────────────────┤ │
│ │ MoeDistributeDispatchV2Info              │ │  算子参数
│ │   uint32_t epWorldSize                   │ │  EP域世界大小
│ │   uint32_t tpWorldSize                   │ │  TP域世界大小
│ │   uint32_t epRankId                      │ │  EP域当前rank
│ │   uint32_t tpRankId                      │ │  TP域当前rank
│ │   uint32_t expertShardType               │ │  专家分片类型
│ │   uint32_t sharedExpertNum               │ │  共享专家数量
│ │   uint32_t sharedExpertRankNum           │ │  共享专家占用的rank数
│ │   uint32_t moeExpertNum                  │ │  MoE专家总数
│ │   uint32_t quantMode                     │ │  量化模式 (0-4)
│ │   uint32_t globalBs                      │ │  全局batch = BS * worldSize
│ │   uint32_t bs                            │ │  本卡batch size
│ │   uint32_t k                             │ │  TopK 值
│ │   uint32_t h                             │ │  隐藏维度大小
│ │   uint32_t a                             │ │  TP域扩展参数
│ │   uint32_t aivNum                        │ │  使用的AIV核数
│ │   bool isTokenMask                       │ │  是否使用1维Token掩码
│ │   bool isExpertMask                      │ │  是否使用2维专家掩码
│ │   bool hasElasticInfo                    │ │  是否有弹性缩放信息
│ │   bool isPerformance                     │ │  是否采集性能数据
│ │   bool isQuant                           │ │  是否量化
│ │   bool isMc2Context                      │ │  是否使用Mc2Context
│ │   uint64_t totalUbSize                   │ │  UB总大小
│ │   uint64_t totalWinSizeEp                │ │  EP域Window大小
│ │   uint64_t totalWinSizeTp                │ │  TP域Window大小
│ │   uint32_t expertTokenNumsType           │ │  0: cumsum, 1: count
│ │   int32_t zeroComputeExpertNum           │ │  零计算专家数量
│ │   uint32_t maxSizeForUbBuffer            │ │  UB buffer最大大小
│ │   uint64_t scalesRow / scalesCol         │ │  Scales维度
│ │   uint32_t scalesTypeSize / scalesCount  │ │  Scales元信息
│ └──────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

#### MoeDistributeDispatchA2TilingData (A2 使用)

```
┌──────────────────────────────────────────────┐
│ MoeDistributeDispatchA2TilingData             │
│ ┌──────────────────────────────────────────┐ │
│ │ Mc2InitTiling mc2InitTiling              │ │
│ ├──────────────────────────────────────────┤ │
│ │ Mc2CcTiling mc2CcTiling                  │ │  单个通信域
│ ├──────────────────────────────────────────┤ │
│ │ MoeDistributeDispatchA2Info              │ │
│ │   (与V2Info类似但少了TP相关字段)          │ │
│ │   expertTokenNumsType, zeroComputeExpertNum│ │
│ │   maxMoeExpertNum                        │ │  A2独有
│ └──────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

### 3.2 TilingKey 编码 (6 维模板参数)

```
TILINGKEY_TPL_A2=0, TPL_A3=1, TPL_A5=2         ← 架构标签 (2bit)
TILINGKEY_NO_QUANT=0, STATIC=1, PERTOKEN=2,     ← 量化模式 (4bit)
  PERGROUP=3, MX=4
TILINGKEY_NO_FULLMESH=0, ENABLE=1, HIERARCHY=2  ← 全连接模式 (2bit)
TILINGKEY_TPL_MTE=0, AICPU=1, CCU=2             ← 通信模式 (2bit)
TILINGKEY_TP: bool                                ← 是否有TP域
TILINGKEY_SCALE: bool                             ← 是否有smooth scale
```

不同架构的有效组合见 `moe_distribute_dispatch_v2_tiling_key.h`，通过 `ASCENDC_TPL_SEL` 宏在编译时展开。

### 3.3 状态空间布局

```
GM 状态空间 (每张卡, 偏移起始于 WindowState 区域):
┌──────────────────────────────────────────────────┐
│ DISPATCH_STATE_WIN_OFFSET (768KB) + aivId * 512  │
│ ┌──────────────────────────────────────────────┐ │
│ │ [0] ZERONE_STATE (uint32)                    │ │  0/1 双缓冲选择
│ │ [1] OPOSITION (uint32) = 1                   │ │  运行位置标记
│ │ [2] TILING_EPRANKID (uint32)                 │ │  Tiling中的EP Rank
│ │ [3] MOE_NUM (uint32)                         │ │  MoE专家数
│ │ [4] TILING_WORLDSIZE (uint32)                │ │  Tiling中的世界大小
│ │ [5] GLOBALBS (uint32)                        │ │  全局Batch大小
│ │ [6-7] reserved                               │ │
│ │ [8-9] HCCL_DFX: epRankId, worldSize          │ │  HCCL实际rank信息
│ └──────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────┘

通信状态区 (每对 <srcRank, dstExpert> 占 32B):
┌──────────────────────────────┐
│ [0]   flag (float, 1.0=完成) │  标志位
│ [1]   tokenCount (int32)     │  发送的token数
│ [2-7] reserved (对齐到32B)   │
└──────────────────────────────┘
```

### 3.4 Window 数据区布局

```
Window 数据区 (EP域, 每张卡):
┌──────────────────────────────────────────────────────────────────┐
│ dataState_ * (totalWinSizeEp / 2)  ← Ping/Pong 半区选择         │
│ + axisMaxBS * (K + sharedExpertNum) * hSizeAlignCombine          │
│                       ← 偏移过 Combine 数据区                    │
│                                                                  │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 按专家排列: expert 0 区 | expert 1 区 | ... | expert N 区    │ │
│ │                                                              │ │
│ │ 每个专家区 (expertPerSizeOnWin = axisMaxBS * hAlignWinSize): │ │
│ │ ┌─────────────────────────────────────────────────────────┐ │ │
│ │ │ Token 0: [数据H | Scale | 三元组(rank,token,topK)]      │ │ │
│ │ │ Token 1: [数据H | Scale | 三元组]                       │ │ │
│ │ │ ...                                                      │ │ │
│ │ │ Token axisMaxBS-1: [数据H | Scale | 三元组]              │ │ │
│ │ └─────────────────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘

其中每个 Token 布局:
┌──────────────────────────────────────────────────────┐
│ [0, hAlignSize)        : XType 输入数据 (H个元素)     │
│ [hAlignSize, hOutSizeAlign) : Scale/量化参数          │
│ [hOutSizeAlign, +12)   : 三元组 (rankId, tokenId, topKId) │
│ 对齐到 512B (hAlignWinSize)                           │
└──────────────────────────────────────────────────────┘
```

---

## 4. 核心类详解

### 4.1 MoeDistributeDispatchV2 (主类, `moe_distribute_dispatch_v2.h`)

**模板参数**:
| 参数 | 类型 | 含义 |
|------|------|------|
| XType | typename | 输入数据类型 (BF16/FP16) |
| ExpandXOutType | typename | 输出数据类型 (BF16/FP16/INT8/FP8) |
| QuantMode | int32_t | 量化模式 (0=不量化, 1=静态, 2=perToken动态, 3=perGroup, 4=MX) |
| IsSmoothScaleExist | bool | 是否有平滑缩放因子 |
| IsNeedAllgather | bool | 是否需要TP域AllGather |

**关键成员**:
- 28 个 GlobalTensor (GM 侧输入/输出/Window)
- 20 个 LocalTensor (UB 侧临时存储)
- 19 个 TBuf/TQue (UB 缓冲区管理)
- `MoeDistributeDispatchV2Quant` 量化实例
- `MoeDistributeElastic` 弹性缩放实例

**核心方法详解**:

#### Init (第287-543行)
1. 获取 tiling 参数，解析 rank/worldSize/expertNum 等
2. 获取 HCCL Window 上下文 (EP/TP 两个通信域)
3. 校验 WindowSize 是否越界
4. 初始化状态空间 (Ping/Pong)
5. 计算专家分布: sharedExpertRankNum / moeExpertRankNum / moeExpertNumPerRank
6. 分配 UB buffer (根据量化模式选择不同分配策略)
7. 计算各核心分配的任务量

#### AlltoAllDispatch (第902-946行)
核心发送逻辑:
1. 可选的 TokenActiveMaskCal — 计算1维活跃 Token 掩码
2. 可选的 ExpertActiveMaskCal — 计算2维专家掩码
3. ZeroComputeExpertMaskCal — 零计算专家剪枝
4. **SendToSharedExpert** — 后部 AIV 核向共享专家卡发送数据
5. **SendToMoeExpert** — 前部 AIV 核向 MoE 专家卡发送数据

核心分配策略:
- 共享专家卡: 全部核用于给 MoE 专家发送数据
- MoE 专家卡: 部分核用于共享专家发送 (sharedUsedAivNum)，部分用于 MoE 专家发送 (moeUsedAivNum)

#### SendToMoeExpert (第689-733行)
1. 通过 SplitToCore 分配每个 AIV 核处理的 Token 范围
2. 遍历分配的 Token:
   - 读取 expertIds 确定目标专家
   - CalTokenSendExpertCnt 计算该专家之前已发送的 Token 数 (用于确定写入偏移)
   - 计算目标 rankId = dstExpertId / moeExpertNumPerRank_ + sharedExpertRankNum_
   - 支持弹性缩放 (isScalingDownFlag_) 重映射 rank
   - ProcessToken: 拷贝+量化+三元组填充+写入 Window

#### WaitDispatch (第1119-1162行)
1. BufferInit: 重新分配 UB buffer
2. 轮询等待: 循环读取 WindowState 的 flag，直到所有对端都写完 1.0
3. WaitDispatchClearStatus: 清除状态区
4. SyncCntOnCore: 核间同步 token 计数到 workspace

#### LocalWindowCopy (第1228-1293行)
1. RunPosRecord: 维测打点
2. GetCumSum: 从 workspace 获取前序核的累计 token 数
3. 遍历每个专家:
   - 从 Window 读取 token 数据到 UB
   - 提取 expandIdx 三元组 (rankId, tokenId, topKId)
   - 可选: 拷贝 dynamicScales 到输出
   - 可选: 拷贝到 TP AllGather Window
   - 拷贝 expandX 到最终输出
4. 更新 sendCountsOut

### 4.2 MoeDistributeDispatchV2FullMesh (`moe_distribute_dispatch_v2_full_mesh.h`)

**与主类的核心差异**:
- 使用 CumSum API 进行前缀和计算，替代手动的核间同步计数
- SplitBlock 分块策略，将 expert 划分为 512B 的块进行处理
- 更精细的接收端处理: CheckDataArriveWithFlag 逐块检查数据到达
- 支持基于 flag 的数据到达检查，而非全局 ReduceSum 等待
- 更复杂的 UB 布局，使用 `MAX_UB_SIZE = 190KB` 接近 UB 上限

**通信策略**: Full Mesh 模式，每个 rank 直接向所有其他 rank 写入 Window，无需中间节点转发。

### 4.3 MoeDistributeDispatchA2 (`moe_distribute_dispatch_a2.h`)

**A2 架构特化**:
- 使用 `HcclA2CombineOpParam` 管理窗口
- 使用 `Hccl<HCCL_SERVER_TYPE_AICPU>` HCCL 封装类
- BatchWrite 机制批量写入 Window (ConstructBatchWriteInfo)
- IndexSort: 对 token 按目标 rank 排序以优化连续写入
- 单机场景 (isSingleServer) 快速路径: SingleServerSendToMoeExpert
- A2 Tiling 数据结构 (无 TP 域参数)

### 4.4 A2 Layered 变体

#### MoeDistributeDispatchA2Layered (`moe_distribute_dispatch_a2_layered.h`)

**分层通信架构 (A2 AIV 模式)**:
- 使用 `MoeDistributeA2DispatchAddrInfo` 地址管理器
- TokenStruct 数据结构打包: 数据+expertId+weight+scale+flag 一体化传输
- 三阶段通信:
  1. ReorderTokens: 按 server 重排 token
  2. Win2Ipc: 写入 IPC 区域 (同 server 内)
  3. Ipc2Out: 从 IPC 读取并输出

#### MoeDistributeDispatchA2LayeredAicpu (`moe_distribute_dispatch_a2_layered_aicpu.h`)

**AICPU 数据面模式**: 当 HCCL 上下文指示 AICPU 模式时使用，数据面由 AICPU 管理，Kernel 仅负责数据准备和结果处理。

### 4.5 MoeDistributeDispatchV2Layered (`moe_distribute_dispatch_v2_layered.h`)

**A3 分层通信变体 (Hierarchy 模式)**:
- 用于多机多卡分层通信
- IPC + RDMA 双通道:
  - 同 server 内: IPC (共享内存直接写入)
  - 跨 server: RDMA WRITE (通过 QP/WQE)
- 详细的状态机: FINISH_STATUS / WAIT_STATUS / ARRIVAL_STATUS / SKIP_STATUS
- Inner/Outer Reduce: 两级 reduce 计算 expert 接收的 token 数

### 4.6 MoeDistributeDispatchA5 (arch35) (`arch35/moe_distribute_dispatch_arch35.h`)

**A5 架构 (CCU 路径)**:
- 使用 HCCL 库的 `lib/hccl/hccl.h` 接口
- 三阶段流水: TokenScatter → Communication → TokenGather
- 支持所有量化模式 (UNQUANT/STATIC/PERTOKEN/PERGROUP/MX)
- 独立的 QuantProcess 实现 (不依赖 DispatchV2Quant)

---

## 5. 通信模式分析

### 5.1 三种通信路径

| 通信模式 | 适用场景 | 实现类 | 通信协议 |
|---------|---------|--------|---------|
| MTE (Mesh) | 单机或全连接多机 | DispatchV2 / FullMesh | IPC 写 Window |
| AICPU (Layered) | A2 多机分层 | A2Layered / A2LayeredAicpu | IPC + RDMA |
| CCU (HCCL) | A5 单机 | DispatchA5 | HCCL 库封装 |
| Hierarchy | A3 多机分层 | V2Layered | IPC + RDMA |

### 5.2 IPC 写入机制 (主路径)

```
发送端 (本卡 AIV Core):                      接收端 (对卡):
┌──────────────────┐                       ┌──────────────────┐
│ 1. ProcessToken   │                       │                  │
│    量化+三元组    │                       │                  │
│ 2. DataCopyPad    │── Window GM ────────▶ │ WaitDispatch     │
│    写入对卡Window │    (共享内存)          │  轮询 flag       │
│ 3. SetStatus      │── State GM ────────▶ │ LocalWindowCopy  │
│    写flag=1.0     │                       │  Window→输出     │
└──────────────────┘                       └──────────────────┘
```

### 5.3 RDMA 写入机制 (A2/A3 Layered 路径)

```
本卡 AIV:                              目标卡 GM:
┌──────────────────┐                   ┌──────────────────┐
│ AIVRDMAPostSend   │                   │                  │
│ 1. 构造 WQE:      │                   │                  │
│   - opcode=0x3    │                   │                  │
│   - destAddr      │  RDMA WRITE ────▶ │ 数据到达          │
│   - messageLen    │  (RoCEv2)         │                  │
│ 2. cache flush    │                   │                  │
│ 3. PostDoorBell   │─── DB ──────────▶ │ 通知处理          │
│   [0:23] wqn      │                   │                  │
│   [32:47] sqHead  │                   │                  │
│   [48:50] sl      │                   │                  │
└──────────────────┘                   └──────────────────┘
```

RDMA WQE 结构 (32B + SGE 16B):
```
偏移  字段               值
0x00  Control            opcode=3(RDMA_WRITE) | signaled=1 | owner_bit
0x04  Message Length     messageLen
0x08  imm_data           0
0x0C  Flags              num_sge=1 << 24
0x10  start_sge_idx      0
0x14  R_Key              目标端注册内存Key
0x18  Remote VA           destDmaAddr
--- SGE (16B) ---
0x20  SGE Length          messageLen
0x24  L_Key              本地注册内存Key
0x28  Local VA            srcDmaAddr
```

### 5.4 A2 HCCL Buffer 布局

详见 `moe_distribute_a2_base.h` 中的 ASCII 文档。关键区域:

| 区域 | 偏移 | 用途 |
|------|------|------|
| Ping RDMA Arrived Flag | 0 | 跨机到达标志 |
| Ping RDMA Inner Flag | A1 | 机内中间标志 |
| Ping RDMA Inner Data | 2*A1 | 机内中间数据 |
| Ping RDMA Data | 1MB | 跨机 Token 数据 |
| IPC Data | W/2-2MB 区域 | 机内 Token 数据 (按 expert*rank 排列) |
| IPC Dispatch Sync Flag | W/2-1MB | Dispatch 同步标志 |
| Magic Value | W/2MB - 128*32B | 双缓冲 bufferId = magicValue & 0x1 |
| Pong RDMA (同上结构) | W/2MB | 第二缓冲 |

---

## 6. 内存管理

### 6.1 UB 分配布局 (MoeDistributeDispatchV2, 非量化场景)

```
UB (~170KB 可用)
┌─────────────────────────────────────────────────────┐
│ expertIdsBuf_    : BS * K * sizeof(int32)  ≈ 32KB   │  expertIds 输入
│ gatherMaskTBuf_  : max(hFp32, expertIdsSize, ...)   │  通用临时空间
│   ├→ gatherMaskTensor_ (uint32)                      │
│   └→ workLocalTensor_ (float)                        │
│ statusBuf_       : Ceil(totalExpert/aivNum, 8) * 32B │  发送状态
│ [可选] validBsIndexTBuf_ : BS * 4B 对齐              │  有效bs索引
│ [可选] validExpertIndexBuf_ : expertIdsSize           │  有效专家索引
│ dstExpBuf_       : maxSize                           │  专家ID临时
│ subExpBuf_        : maxSize                           │  专家ID差值
│ xQueue_          : BUFFER_NUM * hOutAlignUbSize      │  数据搬运队列
│                   (≈ 7KB * 2 + 32B + 12B)            │
├─────────────────────────────────────────────────────│
│ BufferInit 之后的 UB 布局:                           │
│ waitStatusBuf_   : max(recStatusNumPerCore*32, 256)  │  等待状态
│ gatherMaskOutBuf_: max(recStatusNumCore, recvBlockNum)*4│ Gather输出
│ sumCoreBuf_      : aivNum * 32B                      │  核间求和
│ sumLocalBuf_     : aivNum * 32B                      │  本地求和
│ sumContinueBuf_  : aivNum * 4B                       │  连续求和
│ scalarBuf_       : 96B                               │  标量辅助
│ xQueue_          : BUFFER_NUM * hOutAlignUbSize      │  数据队列
│ tokenNumBuf_     : moeExpertNumPerRank * 8B 对齐      │  Token计数
│ statusBuf_       : moeExpertNumPerRank * epWorldSize*32│ 状态
│ workLocalBuf_    : epWorldSize * 4B 对齐到32B         │  工作空间
└─────────────────────────────────────────────────────┘
```

### 6.2 GM 区域使用

| 区域 | 地址来源 | 用途 |
|------|---------|------|
| 输入 x | 参数 x | Token 数据 [BS, H] |
| 输入 expertIds | 参数 expertIds | 路由表 [BS, K] |
| 输入 scales | 参数 scales | 平滑缩放因子 |
| 输入 xActiveMask | 参数 xActiveMask | Token 活跃掩码 |
| Window EP | HCCL Window (EP域) | 跨卡共享数据区 |
| Window TP | HCCL Window (TP域) | TP AllGather 数据区 |
| WindowState EP | Window + STATE_WIN_OFFSET | 通信状态/flag |
| Workspace | 参数 workspaceGM | 核间同步计数 |
| expandXOut | 参数 expandXOut | 最终输出 Token |
| dynamicScalesOut | 参数 dynamicScalesOut | 量化 Scale 输出 |
| assistInfoOut | 参数 expandIdxOut | 三元组 (rank, token, topK) |
| expertTokenNumsOut | 参数 expertTokenNumsOut | 每个专家的 Token 数 |
| epSendCountsOut | 参数 sendCountsOut | EP 发送计数 |
| tpSendCountsOut | 参数 tpSendCountsOut | TP 发送计数 |

### 6.3 Ping/Pong 双缓冲机制

```
dataState_ 的取值: 0 或 1 (通过 InitWinState 翻转)

Window 数据区选择:
  winDataSizeOffsetEp_ = dataState_ * (totalWinSizeEp_ / 2)
                       + axisMaxBS * (K + sharedExpertNum) * hSizeAlignCombine

  → dataState_ = 0: 使用前半区
  → dataState_ = 1: 使用后半区

Window 状态区选择:
  statusSpaceGm_ = WindowStateAddr + dataState_ * WIN_STATE_OFFSET (384KB)

每次调用时 dataState_ 翻转，与 Combine 算子交替使用前后半区:
  Dispatch 用后半区 (偏移过 Combine 的数据)
  Combine 用前半区
```

---

## 7. 量化与精度分析

### 7.1 量化模式对比

| 量化模式 | 输入类型 | 输出类型 | Scale 来源 | 粒度 |
|---------|---------|---------|-----------|------|
| UNQUANT (0) | BF16/FP16 | 同输入 | 无 | - |
| STATIC_QUANT (1) | BF16/FP16 | INT8 | 外部提供 scales 参数 | 1 (全局) 或 H (per-channel) 或 expert*H |
| PERTOKEN_DYNAMIC_QUANT (2) | BF16/FP16 | INT8/FP8_E5M2/FP8_E4M3 | 运行时 ReduceMax 计算 | 每 Token 一个 scale |
| PERGROUP_DYNAMIC_QUANT (3) | BF16/FP16 | FP8_E5M2/FP8_E4M3 | 运行时 per-128 计算 | 每 128 元素一个 scale |
| MX_QUANT (4) | BF16/FP16 | FP8_E5M2/FP8_E4M3 | 运行时 max-exp 计算 | 每 32 元素一个 scale |

### 7.2 动态量化流水线 (PERTOKEN_DYNAMIC_QUANT, INT8 输出)

```
BF16/FP16 Token [H]
     │
     ▼ Cast → FP32
FP32 Token [H]
     │
     ├─[有smoothScale]─▶ Mul(smoothScales) ─▶ FP32 (缩放后)
     │
     ▼ Abs → FP32
FP32 Abs [H]
     │
     ▼ ReduceMax → FP32 (maxVal)
     │
     ▼ dynamicScale = 127.0 / maxVal
     │
     ▼ Muls(dynamicScale) → FP32
     │
     ▼ Cast(FP32→INT32, RINT) → INT32
     ▼ SetDeqScale(1.0), Cast(INT32→FP16, ROUND) → FP16
     ▼ Cast(FP16→INT8, TRUNC) → INT8
     │
     ▼ 输出: INT8 Token [H] + float(1/dynamicScale) 作为 dynamicScale
```

### 7.3 MX 量化流水线 (仅 A5)

```
BF16/FP16 Token [H]
     │
     ▼ ComputeMaxExp (每32个一组, 取最大指数)
     │  → uint16 maxExp[numScales]
     │
     ▼ ComputeScale (maxExp → fp8_e8m0_t scales)
     │  → uint16 mxScale[Align2(Ceil32(H))]
     │
     ▼ ComputeData (原始数据 / halfScale → FP8)
     │  → FP8_E5M2 或 FP8_E4M3
     │
     ▼ 输出: FP8 Token[Align256(H)] + fp8_e8m0_t scales[Align2(Ceil32(H))]
```

---

## 8. 地址计算接口 (A2 架构)

`MoeDistributeA2AddrInfo` 和 `MoeDistributeA2DispatchAddrInfo` 提供了完整的地址计算接口:

### 8.1 基类方法 (MoeDistributeA2AddrInfo)

| 方法 | 签名 | 语义 |
|------|------|------|
| GetLocalSendBuffFlagAddr | `()` | 本卡发送缓冲区的 flag 地址 |
| GetRemoteRecvBuffFlagAddr | `(dstRankId)` | 目标卡的接收 flag 地址 |
| GetRemoteRecvBuffDataAddr | `(dstRankId)` | 目标卡的接收数据地址 |
| GetLocalRecvBuffFlagAddr | `(srcRankId)` | 本卡接收某源卡的 flag 地址 |
| GetLocalRecvBuffDataAddr | `(srcRankId)` | 本卡接收某源卡的数据地址 |
| GetIpcDataAddr | `(dstRankId, localExpertId, srcRankId)` | IPC 数据区地址 (按 expert+srcRank 索引) |
| GetMagicValue | `()` | 当前 magic 值 (用于 Ping/Pong) |

### 8.2 Dispatch 专用方法 (MoeDistributeA2DispatchAddrInfo)

| 方法 | 语义 |
|------|------|
| GetLocalSendBuffDataAddr | 发送缓冲区数据 (连续累加) |
| GetLocalSendBuffInnerDataAddr(dstServerId) | 发送缓冲区 Inner 表 (按 server) |
| GetRemoteRecvBuffInnerFlagAddr(dstServerId) | 目标卡 Inner 表 flag |
| GetRemoteRecvBuffInnerDataAddr(dstServerId) | 目标卡 Inner 表数据 |
| GetRemoteIpcSyncFlagAddr(dstRankId) | 目标卡 IPC 同步标志 |
| GetRemoteIpcTokenCntAddr(dstRankId, targetExpId, srcRankId) | 目标卡 IPC token 计数 |
| GetLocalIpcSyncFlagAddr(srcRankId) | 本卡 IPC 同步标志 |
| GetLocalIpcTokenCntAddr | 本卡 IPC token 计数起始 |

### 8.3 MagicValue 与缓冲选择

```
UpdateMagicValue:
  1. 从 IPC 区域读取当前 magicValue (uint64)
  2. magicValue += 1
  3. 写回 IPC 区域
  4. bufferId = magicValue & 0x1  →  选择 Ping(0) 或 Pong(1) 半区
  5. rdmaFlagAddrStart_ = bufferId ? winSize/2 : 0
```

---

## 9. 关键设计模式与优化

### 9.1 多核负载均衡 (SplitToCore)

```cpp
// 每个核心的任务分配:
sendTokenNum = totalTasks / aivNum;     // 基本份额
remainder = totalTasks % aivNum;         // 余数
// 前remainder个核多处理1个任务
if (aivId < remainder) {
    sendTokenNum += 1;
    startTokenId += aivId;
} else {
    startTokenId += remainder;
}
```

### 9.2 Token 发送偏移计算 (CalTokenSendExpertCnt)

为了确定每个 token 写入 Window 的确切位置，需要计算同一目标专家之前的 token 数:
1. Duplicate: 将目标专家 ID 填充到临时 tensor
2. Sub: expertIds - dstExpertId，得到差值
3. Abs: 取绝对值
4. Mins: 差值为0的得到1，否则为0
5. ReduceSum: 统计等于目标专家 ID 的数量

### 9.3 零计算专家剪枝

当存在 `zeroComputeExpertNum > 0` 时:
1. CompareScalar: 筛掉 expertId >= moeExpertNum 的零计算专家
2. 与之前的 mask 进行 AND 操作
3. GatherMask: 压缩得到有效专家索引

### 9.4 弹性缩放支持 (MoeDistributeElastic)

从 elasticInfo 输入读取:
- `[0]`: isScalingDown 标志
- `[1]`: 新的 epWorldSize
- `[2]`: 新的 sharedExpertRankNum
- `[3]`: 新的 moeExpertNum
- `[4, 4+epWorldSizeOriginal)`: rank 重映射表
- `[4+epWorldSizeOriginal, 4+2*epWorldSizeOriginal)`: 完整 rank 重映射表

---

## 10. 关键约束与注意事项

### 10.1 硬件约束

| 约束 | 值 | 说明 |
|------|-----|------|
| UB 大小 | ~170-190 KB | 由 tiling 确保 totalUsedUB_ 不超限 |
| Window 地址对齐 | 512B | WIN_ADDR_ALIGN |
| 状态空间对齐 | 32B | UB_ALIGN |
| DataCopy 最小粒度 | 32B | DataCopy 要求 32 字节对齐 |
| A2 单 server rank 数 | 8 | SERVER_RANK_SIZE |
| AIV 核最大数 | 由 tiling 决定 | aivNum_ |

### 10.2 正确性保证

1. **Flag 写入顺序**: 状态 flag 放在每个 32B 块的首位，利用 ROCE 的 32B 原子写入保证一致性
2. **全局同步**: `SyncAll<true>()` 确保所有核完成发送后再进入等待阶段
3. **Cache 一致性**: RDMA 写入前通过 `cacheWriteThrough` 刷新 WQE/SGE 的 cache
4. **Pipeline Barrier**: 各阶段间使用 `PipeBarrier<PIPE_ALL>()` 保证操作完成
5. **Window Size 校验**: `CheckWindowSize` 在 Init 时检查 HCCL 分配的 Window 是否满足 tiling 要求

### 10.3 Timeout 检测

```
TIMEOUT_DETECTION_THRESHOLD = 50000
CYCLES_PER_US = 50
→ 超时阈值 = 50000 / 50 = 1000 us = 1 ms
```

---

## 11. 与同族变体的核心差异

### 11.1 算法路径对比

| 特性 | DispatchV2 (主) | FullMesh | Layered (A3) | A2 | A2 Layered | A5 |
|------|----------------|----------|-------------|-----|-----------|-----|
| 架构 | A3 | A3/A5 | A3 | A2 | A2 | A5 |
| 通信协议 | IPC Window | IPC Window | IPC+RDMA | BatchWrite Window | IPC+RDMA | HCCL 库 |
| 同步机制 | ReduceSum轮询 | CumSum+Flag | IPC Flag | ReduceSum | IPC Flag | HCCL同步 |
| TP AllGather | 支持 | 支持 | 不支持 | 不支持 | 不支持 | 不支持 |
| 量化模式 | 全部5种 | 全部5种 | PerToken | 静态/PerToken | 静态/PerToken | 全部5种 |
| 共享专家 | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| 弹性缩放 | 支持 | 支持 | 不支持 | 不支持 | 不支持 | 不支持 |
| 单机快速路径 | 无 | 无 | 无 | 有 | 无 | 无 |

### 11.2 通信策略差异

| 场景 | 主路径 | Layered 路径 |
|------|--------|-------------|
| 同 server 内 rank | 直接写 Window (IPC) | 写 IPC Data 区 + IPC Flag 同步 |
| 跨 server | 写 Window (RDMA 或 HCCL 路由) | RDMA WRITE 直接写入目标卡 |
| 多机分层 | 不适用 | Inner (同 server) + Outer (跨 server) 两级 Reduce |

---

## 12. 函数调用关系

### 12.1 MoeDistributeDispatchV2 调用树

```
moe_distribute_dispatch_v2()  [入口]
│
├── MoeDistributeDispatchV2::Init()
│   ├── CheckWindowSize()
│   ├── InitWinState()           [moe_distribute_v2_base.h]
│   ├── MoeDistributeElastic::SetElasticInitParams()
│   ├── MoeDistributeElastic::InitElasticInfoTensor()
│   ├── MoeDistributeElastic::InitElasticInfo()
│   ├── MoeDistributeDispatchV2Quant::QuantInit()
│   ├── MoeDistributeDispatchV2Quant::SetQuantInitParams()
│   └── TPipe::InitBuffer() x N
│
└── MoeDistributeDispatchV2::Process()
    ├── AlltoAllDispatch()
    │   ├── TokenActiveMaskCal()
    │   │   ├── DataCopyPad (xActiveMask)
    │   │   ├── Cast → Sum → 活跃token计数
    │   │   └── CalValidBSCnt() + CalValidExpIdx()
    │   ├── ExpertActiveMaskCal()
    │   ├── ZeroComputeExpertMaskCal()
    │   │   └── MaskZeroComputeExpert()
    │   ├── SendToSharedExpert()
    │   │   └── ProcessToken() x N
    │   │       ├── DataCopyPad (x)
    │   │       ├── QuantProcess() [量化]
    │   │       ├── FillTriple()
    │   │       └── DataCopyPad (→Window)
    │   └── SendToMoeExpert()
    │       ├── CalTokenSendExpertCnt()
    │       └── ProcessToken() x N
    │
    ├── SetStatus()
    │   ├── CalTokenSendExpertCnt() x N
    │   ├── SyncAll()
    │   └── DataCopy (→WindowState)
    │
    ├── WaitDispatch()
    │   ├── BufferInit()
    │   ├── DataCopy (轮询 WindowState) [循环]
    │   ├── PerformanceInfoPerRank() [可选]
    │   ├── WaitDispatchClearStatus()
    │   ├── SyncCntOnCore()
    │   │   ├── GatherMask + Sum
    │   │   └── DataCopy (→Workspace)
    │   └── SyncAll()
    │
    ├── LocalWindowCopy()
    │   ├── RunPosRecord()
    │   ├── GetCumSum()
    │   │   ├── DataCopy (←Workspace)
    │   │   └── GatherMask + Sum
    │   ├── DataCopyPad (Window→UB) x N
    │   ├── DataCopyPad (expandIdx) x N
    │   ├── CopyScalesToOut() x N
    │   ├── DataCopyPad (expandXOut) x N
    │   └── DataCopyPad (sendCountsOut)
    │
    ├── [可选 TP] AllGatherSetStatusAndWait()
    ├── [可选 TP] AllgatherProcessOut()
    ├── [可选 TP] AllGatherSetExpertTokenNumsAndTpRecvCount()
    │
    └── SetExpertTokenNums()
        ├── DataCopy (←WindowState)
        ├── ReduceSum (每专家)
        └── DataCopyPad (→expertTokenNumsOut)
```

---

## 13. 依赖关系

```
moe_distribute_dispatch_v2.cpp
├── moe_distribute_dispatch_v2.h
│   ├── moe_distribute_dispatch_v2_tiling.h
│   ├── moe_distribute_v2_constant.h
│   ├── moe_distribute_dispatch_v2_quant.h
│   │   ├── moe_distribute_v2_constant.h
│   │   ├── moe_distribute_v2_base.h
│   │   └── quantize_functions.h [仅A5]
│   │       └── common.h
│   ├── moe_distribute_elastic.h
│   ├── moe_distribute_v2_base.h
│   │   ├── moe_distribute_v2_constant.h
│   │   └── common/op_kernel/moe_distribute_base.h
│   ├── check_winsize.h
│   ├── common/op_kernel/mc2_moe_context.h
│   └── common/op_kernel/moe_distribute_base.h
├── moe_distribute_dispatch_v2_full_mesh.h
│   ├── adv_api/math/cumsum.h
│   ├── adv_api/reduce/sum.h
│   ├── moe_distribute_dispatch_v2_tiling.h
│   └── common/op_kernel/mc2_kernel_utils.h
├── moe_distribute_dispatch_v2_layered.h
│   ├── moe_distribute_dispatch_v2_tiling.h
│   └── common/op_kernel/moe_distribute_base.h
├── moe_distribute_dispatch_v2_tiling_key.h
├── moe_distribute_dispatch_a2.h [仅A2]
│   ├── moe_distribute_dispatch_tiling.h
│   └── common/op_kernel/mc2_kernel_utils.h
├── moe_distribute_dispatch_a2_layered.h [仅A2]
│   ├── moe_distribute_dispatch_tiling.h
│   ├── moe_distribute_a2_base.h
│   └── common/op_kernel/mc2_kernel_utils.h
├── moe_distribute_dispatch_a2_layered_aicpu.h [仅A2]
└── arch35/moe_distribute_dispatch_arch35.h [仅A5]
    ├── lib/hccl/hccl.h
    ├── common.h
    ├── quantize_functions.h
    └── adv_api/reduce/sum.h
```

---

*报告生成时间: 2026-04-17*
*分析范围: mc2/moe_distribute_dispatch_v2/op_kernel/ 全部 20 个文件*
