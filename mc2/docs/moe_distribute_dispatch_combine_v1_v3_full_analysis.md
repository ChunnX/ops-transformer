# moe_distribute_dispatch/combine (V1 & V3) 全面分析报告

## 1. 算子概述

`moe_distribute_dispatch` 和 `moe_distribute_combine` 是 MOE 推理中 Token-Expert 路由与聚合的核心配对算子。Dispatch 将 Token 按路由表分发到各 Expert 所在 Rank，Combine 收集 Expert 计算结果并按 gating scale 加权聚合。

- **V1**: 原生实现，支持 910B (A2) / 910_93 (A3) / 950 (A5) 三平台，自有 Kernel 类 + Window P2P 通信
- **V3**: 接口升级版本，仅支持 910_93 (A3)，**所有计算逻辑完全委托给 V2 实现**，支持 FP8/MX 量化、mc2Context、弹性训练等新特性

```
MOE 推理数据流:

Dispatch 阶段:
  x[BS,H] + expert_ids[BS,K]
      → [路由表查找] → [量化(可选)] → [AlltoAll写入Window] → [SetStatus通知]
      → [WaitDispatch等待] → [LocalWindowCopy重排]
      → expand_x[按Expert分组], expand_idx, expert_token_nums, ...

Combine 阶段:
  expand_x[A,H] + expert_scales[BS,K]
      → [AlltoAll写入Window] → [SetStatus] → [WaitDispatch]
      → [LocalWindowCopy] → [ExpertAlltoAllDispatchCopyAdd: 加权聚合]
      → x_out[BS,H]
```

---

## 2. 目录结构

```
mc2/moe_distribute_dispatch/                     # V1 Dispatch
├── op_kernel/
│   ├── moe_distribute_dispatch.cpp              # Kernel入口 (177行): 6模板参数, A2/A3/A5分发
│   ├── moe_distribute_dispatch.h                # 核心实现 (52,138B): MoeDistributeDispatch类
│   └── moe_distribute_dispatch_tiling_key.h     # 模板Key (101行): 6参数编码, 7组合
├── op_host/
│   ├── moe_distribute_dispatch_def.cpp          # 定义 (125行): 5输入/7输出/12属性
│   ├── moe_distribute_dispatch_infershape.cpp   # Shape推导 (13,369B, 第154行InferShape, 第207-230行InferDataType, 第233-235行注册)
│   └── op_tiling/
│       ├── moe_distribute_dispatch_tiling.cpp   # Tiling (1077行): A2/A3/A5三路径
│       └── moe_distribute_dispatch_tiling_a2a3.h # Tiling委托 (34行)
├── op_api/
│   ├── aclnn_moe_distribute_dispatch.h          # ACLNN接口
│   └── aclnn_moe_distribute_dispatch.cpp
└── config/{ascend910b,ascend910_93,ascend950}/   # 编译配置

mc2/moe_distribute_combine/                      # V1 Combine
├── op_kernel/
│   ├── moe_distribute_combine.cpp               # Kernel入口 (111行): 4模板参数
│   ├── moe_distribute_combine.h                 # 核心实现 (42,245B): MoeDistributeCombine类
│   └── moe_distribute_combine_tiling_key.h      # 模板Key (75行): 4参数编码, 4组合
├── op_host/
│   ├── moe_distribute_combine_def.cpp           # 定义 (143行): 10输入/1输出/14属性
│   ├── moe_distribute_combine_infershape.cpp   # Shape推导 (第31-56行InferShape, 第67-69行注册)
│   └── op_tiling/
│       ├── moe_distribute_combine_tiling.cpp    # Tiling (985行)
│       └── moe_distribute_combine_tiling_a2a3.h
└── op_api/

mc2/moe_distribute_dispatch_v3/                  # V3 Dispatch (薄封装)
├── op_kernel/
│   └── moe_distribute_dispatch_v3.cpp           # 委托V2 (99行)
├── op_host/
│   ├── moe_distribute_dispatch_v3_def.cpp       # 定义 (139行): 8输入/7输出/16属性
│   ├── moe_distribute_dispatch_v3_infershape.cpp # Shape推导 (19,068B, 第120-302行InferShape, 第366-368行注册)
│   └── op_tiling/
│       └── moe_distribute_dispatch_v3_tiling.cpp # 委托V2 Tiling (107行)
└── op_api/
    └── aclnn_moe_distribute_dispatch_v5.cpp     # 外部API名V5, 内部委托V3

mc2/moe_distribute_combine_v3/                   # V3 Combine (薄封装)
├── op_kernel/
│   └── moe_distribute_combine_v3.cpp            # 委托V2 (82行)
├── op_host/
│   ├── moe_distribute_combine_v3_def.cpp        # 定义 (178行): 18输入/1输出/17属性
│   └── op_tiling/
│       └── moe_distribute_combine_v3_tiling.cpp # 委托V2 Tiling (104行)
└── op_api/
    └── aclnn_moe_distribute_combine_v5.cpp      # 外部API名V5, 内部委托V3
```

---

## 3. Host 侧分析

### 3.1 V1 Dispatch 定义

`moe_distribute_dispatch_def.cpp`（第19行类定义；第22-46行输入张量；第48-75行输出张量；第77-89行属性；第122行 `OP_ADD` 注册）

```
输入 (5):
  x              REQUIRED  [BS, H]          BF16/FP16      输入Token特征
  expert_ids     REQUIRED  [BS, K]          INT32          TopK Expert ID
  scales         OPTIONAL  [专家数, ?]       FLOAT          静态量化scale
  x_active_mask  OPTIONAL  [BS]             BOOL           Token激活mask
  expert_scales  OPTIONAL  [BS, K]          FLOAT          Gating scale

输出 (7):
  expand_x         REQUIRED  [A, H]         BF16/FP16/INT8 按Expert重排的Token
  dynamic_scales   REQUIRED  [A]            FLOAT          动态量化scale
  expand_idx       REQUIRED  [BS*K]         INT32          每Token在Expert内序号
  expert_token_nums REQUIRED [localExpertNum] INT64         每Expert的Token计数
  ep_recv_count    REQUIRED  [1D]           INT32          EP域接收计数
  tp_recv_count    REQUIRED  [1D]           INT32          TP域接收计数
  expand_scales    REQUIRED  [1D]           FLOAT          展开的量化scale

属性 (12):
  group_ep             REQUIRED  String   EP通信域名称
  ep_world_size        REQUIRED  Int      EP域大小
  ep_rank_id           REQUIRED  Int      本卡EP Rank ID
  moe_expert_num       REQUIRED  Int      MoE专家总数
  group_tp             OPTIONAL  String   TP通信域 (默认空=无TP)
  tp_world_size        OPTIONAL  Int(0)   TP域大小
  tp_rank_id           OPTIONAL  Int(0)   TP Rank ID
  expert_shard_type    OPTIONAL  Int(0)   专家分片类型
  shared_expert_num    OPTIONAL  Int(1)   共享专家数量
  shared_expert_rank_num OPTIONAL Int(0)  共享专家卡数
  quant_mode           OPTIONAL  Int(0)   量化模式 (0/1/2)
  global_bs            OPTIONAL  Int(0)   全局BatchSize

MC2配置: HcclGroup({"group_ep", "group_tp"})
平台: ascend950, ascend910_93, ascend910b
```

### 3.2 V1 Combine 定义

`moe_distribute_combine_def.cpp`（第19行类定义；第22-87行输入张量；第89-93行输出张量；第95-108行属性；第141行 `OP_ADD` 注册）

```
输入 (10):
  expand_x        REQUIRED  [A, H]         BF16/FP16/INT32  Expert计算结果
  expert_ids      REQUIRED  [BS, K]        INT32
  expand_idx      REQUIRED  [BS, K]        INT32
  ep_send_counts  REQUIRED  [1D]           INT32            EP发送计数
  expert_scales   REQUIRED  [BS, K]        FLOAT            Gating scale
  tp_send_counts  OPTIONAL  [1D]           INT32            TP发送计数
  x_active_mask   OPTIONAL  [BS]           BOOL
  activation_scale OPTIONAL [?]            FLOAT            激活量化scale
  weight_scale    OPTIONAL  [?]            FLOAT            权重量化scale
  group_list      OPTIONAL  [1D]           INT64
  expand_scales   OPTIONAL  [1D]           FLOAT

输出 (1):
  x               REQUIRED  [BS, H]        BF16/FP16        最终Combine结果

属性 (14):
  与Dispatch基本一致，额外: out_dtype(0), comm_quant_mode(0), group_list_type(0)

MC2配置: HcclGroup({"group_ep", "group_tp"})
平台: ascend950, ascend910_93, ascend910b
```

### 3.3 V3 Dispatch 定义

`moe_distribute_dispatch_v3_def.cpp`（第19行类定义；第22-65行输入张量；第67-100行输出张量；第102-118行属性；第136行 `OP_ADD` 注册）

```
输入 (8):
  context          REQUIRED  [1D]          INT32            MC2持久化上下文
  x                REQUIRED  [BS, H]       BF16/FP16/FP8    (支持19种数据类型)
  expert_ids       REQUIRED  [BS, K]       INT32
  scales           OPTIONAL  [?, ?]        FLOAT/FP8        (支持15种类型)
  x_active_mask    OPTIONAL  [BS]          BOOL
  expert_scales    OPTIONAL  [BS, K]       FLOAT
  elastic_info     OPTIONAL  [1D]          INT32            弹性训练信息
  performance_info OPTIONAL  [1D]          INT64            性能采集

输出 (7):
  expand_x                  REQUIRED  [A, H]       BF16/INT8/FP8
  dynamic_scales            REQUIRED  [A]           FLOAT/FP8
  assist_info_for_combine   REQUIRED  [1D]          INT32         ← 新增: Combine辅助信息
  expert_token_nums         REQUIRED  [localExpert] INT64
  ep_recv_count             REQUIRED  [1D]          INT32
  tp_recv_count             REQUIRED  [1D]          INT32
  expand_scales             REQUIRED  [1D]          FLOAT

属性 (16):
  无 group_ep/group_tp (通过context管理通信)
  新增: ccl_buffer_size(REQUIRED), comm_alg, zero_expert_num, copy_expert_num, const_expert_num, y_dtype

平台: 仅 ascend910_93
```

### 3.4 V3 Combine 定义

`moe_distribute_combine_v3_def.cpp`（第19行类定义；第22-134行输入张量；第136-140行输出张量；第142-158行属性；第176行 `OP_ADD` 注册）

```
输入 (18):
  context                REQUIRED  INT32            MC2持久化上下文
  expand_x               REQUIRED  BF16/FP16/INT32  Expert计算结果
  expert_ids             REQUIRED  INT32
  assist_info_for_combine REQUIRED INT32            来自Dispatch V3
  ep_send_counts         REQUIRED  INT32
  expert_scales          REQUIRED  FLOAT
  tp_send_counts         OPTIONAL  INT32
  x_active_mask          OPTIONAL  BOOL
  activation_scale       OPTIONAL  FLOAT
  weight_scale           OPTIONAL  FLOAT
  group_list             OPTIONAL  INT64
  expand_scales          OPTIONAL  FLOAT
  shared_expert_x        OPTIONAL  BF16/FP16        ← 新增: 共享专家独立输入
  elastic_info           OPTIONAL  INT32            ← 新增: 弹性训练
  ori_x                  OPTIONAL  BF16/FP16        ← 新增: 原始输入(残差)
  const_expert_alpha_1   OPTIONAL  BF16/FP16        ← 新增: 常量专家权重1
  const_expert_alpha_2   OPTIONAL  BF16/FP16        ← 新增: 常量专家权重2
  const_expert_v         OPTIONAL  BF16/FP16        ← 新增: 常量专家值
  performance_info       OPTIONAL  INT64            ← 新增: 性能采集

输出 (1):
  x               REQUIRED  [BS, H]        BF16/FP16

属性 (17): 与V3 Dispatch基本一致
平台: 仅 ascend910_93
```

### 3.5 V1 Tiling 分析

**Dispatch Tiling** `moe_distribute_dispatch_tiling.cpp`（第1024-1038行顶层分发器；第691-721行 A3/A5 实现；第964-1022行 A2 实现；第1074-1076行注册）:

```
MoeDistributeDispatchTilingFunc():
  ├── socVersion == "Ascend910B"
  │   └── MoeDistributeDispatchA2TilingFuncImpl()
  │       └── 检查 HCCL_INTRA_PCIE_ENABLE / HCCL_INTRA_ROCE_ENABLE
  │           → 选择 BatchWrite 通信 (fullmesh)
  └── 其他 (910_93/950)
      └── MoeDistributeDispatchA3A5TilingFuncImpl()
          ├── 校验属性: epWorldSize∈[2,384], moeExpertNum∈(0,512]
          ├── 计算moeExpertPerRankNum
          ├── 计算TilingKey: 从 TP/Quant/Scale/FullMesh/CommMode/ArchTag 组合
          ├── Window大小校验
          ├── SetBlockDim(aivNum), SetScheduleMode(batch)
          └── Workspace = 32MB
```

**Combine Tiling** (`moe_distribute_combine_tiling.cpp:921-962`):

```
MoeDistributeCombineTilingFunc():
  ├── expandX.dtype == DT_INT32? → 走INT8量化路径
  ├── socVersion == "Ascend910B"
  │   └── MoeDistributeCombineA2TilingFuncImpl()
  └── 其他
      └── MoeDistributeCombineA3A5TilingFuncImpl()
          ├── EP通信: AlltoAll (fullmesh;pairwise)
          ├── TP通信: ReduceScatter (ring)
          └── A2额外: BatchWrite (fullmesh)
```

### 3.6 V3 Tiling 委托机制

**Dispatch V3** `moe_distribute_dispatch_v3_tiling.cpp`（第51-84行委托V2实现；第93-95行注册）:
```cpp
MoeDistributeDispatchV3TilingFunc():
  DispatchV2Config config;
  config.isMc2Context = true;
  config.contextIndex = 0;     // V3的context在位置0
  config.xIndex = 1;           // V3的x在位置1 (V2中x在位置0)
  config.expertIdsIndex = 2;   // 索引映射...
  // ... 将V3的16个属性映射到V2的位置
  MoeDistributeDispatchA3TilingFuncImplPublic(context, config);  ← 完全委托V2
```

**Combine V3** `moe_distribute_combine_v3_tiling.cpp`（第46-92行委托V2实现；第101-103行注册）:
```cpp
MoeDistributeCombineV3TilingFunc():
  CombineV2Config config;
  config.isMc2Context = true;
  config.hasAddRmsNorm = false;
  config.contextIndex = 0;
  config.expandXIndex = 1;
  config.assistInfoIndex = 3;       // assist_info_for_combine
  config.sharedExpertXIndex = 12;   // shared_expert_x
  config.oriXIndex = 14;            // ori_x
  config.constExpertAlpha1Index = 15;
  // ... 将V3的18个输入映射到V2位置
  MoeDistributeCombineV2TilingFuncNew(context, config);  ← 完全委托V2
```

---

## 4. Kernel 侧分析

### 4.1 V1 Dispatch Kernel 入口

`moe_distribute_dispatch.cpp`（第43-46行 Kernel 入口函数；第58-79行 A5 路径分发；第84-88行 A3 路径分发；第91-96行 A2 路径分发）

```
模板参数:
  HasTp (bool):     是否有TP维度
  QuantMode (uint): 0=不量化, 1=静态量化, 2=逐Token量化
  ScaleMode (bool):  是否有smoothScale
  FullMesh (uint):  全mesh模式
  CommMode (uint):  0=MTE, 1=AICPU, 2=CCU
  ArchTag (uint):   0=A2, 1=A3, 2=A5

Kernel入口分发:
  ArchTag == A5 (950):
    ├── BF16/FP16 + 无量化:    MoeDistributeDispatch<BF16, BF16, false, false, ...>
    └── INT8 + 静态量化/逐Token: MoeDistributeDispatch<BF16, INT8, true/false, ...>
  ArchTag == A3 (910_93):
    └── 同A5路径
  ArchTag == A2 (910B):
    ├── CommMode==MTE:    MoeDistributeDispatchA2 (来自V2)
    ├── CommMode==AICPU:
    │   ├── DataplaneMode==AICPU: MoeDistributeDispatchA2LayeredAicpu
    │   └── DataplaneMode==MTE:   MoeDistributeDispatchA2Layered
    └── (其他CommMode): 默认路径
```

### 4.2 V1 Dispatch 核心类

`moe_distribute_dispatch.h`（52,138字节，核心实现类 `MoeDistributeDispatch`）

```
MoeDistributeDispatch<XType, ExpandXOutType,
                      StaticQuant, DynamicQuant,
                      IsSmoothScaleExist, IsNeedAllgather>

常量:
  BUFFER_NUM = 2              # 双缓冲
  STATE_OFFSET = 512          # 状态间距
  STATE_SIZE = 1MB            # 状态区大小
  SELF_STATE_OFFSET = 256KB   # 本卡状态偏移
  COMM_NUM = 2                # EP + TP 两个通信域

Init():
  ├── HCCL上下文: GetHcclContext<HCCL_GROUP_ID_0/1>()
  ├── 计算地址偏移 (与auxiliary_ops相同的Window布局)
  ├── SplitToCore(): 分核计算
  └── UB缓冲区初始化

Process():
  ├── AlltoAllDispatch(): Token分发核心
  │   ├── CalTokenSendExpertCnt(): 计算每Expert发送的Token数
  │   ├── SendToMoeExpert(): 按Expert将Token写入Window
  │   │   └── x[BS,H] → Window Data Area (per-rank, per-expert)
  │   ├── SendToSharedExpert(): 共享专家Token写入
  │   ├── QuantProcess(): [量化模式] BF16/FP16 → INT8
  │   │   └── 计算abs/max, 量化, 输出dynamic_scales
  │   └── SetStatus(): 通知所有目标Rank数据已就绪
  ├── WaitDispatch(): 自旋等待所有Rank完成发送
  │   ├── DataCopy(状态) → GatherMask → Sum
  │   └── 等待 sumOfFlag ≈ epWorldSize
  ├── LocalWindowCopy(): 从Window拷贝接收到的数据到输出GM
  │   └── 按 expert_token_nums 重排到 expand_x
  ├── [IsNeedAllgather] AllGatherSetStatusAndWait():
  │   └── TP=2场景: 额外的AllGather跨TP Rank同步
  └── UpdateTokenNumsOut(): 写出expert_token_nums, ep_recv_count等
```

### 4.3 V1 Combine Kernel 入口

`moe_distribute_combine.cpp`（第42-47行 Kernel 入口函数；第58-69行 A5/A3 路径分发；第72-78行 A2 路径分发）

```
模板参数:
  HasTp (bool):       是否有TP维度
  QuantMode (uint):   0=不量化, 1=INT8量化
  LayeredMode (uint): 0=MTE, 1=AICPU
  ArchTag (uint):     0=A2, 1=A3, 2=A5

分发逻辑类似Dispatch:
  A5/A3 → MoeDistributeCombine<ExpandXType, int32_t, HasTp, QuantMode==INT8>
  A2    → MoeDistributeCombineA2 / A2Layered / A2LayeredAicpu (来自V2)
```

### 4.4 V1 Combine 核心类

`moe_distribute_combine.h`（42,245字节，核心实现类 `MoeDistributeCombine`）

```
MoeDistributeCombine<ExpandXType, ExpandIdxType,
                     IsNeedReduceScatter, IsQuant>

常量:
  VEC_LEN = 256         # 向量长度
  SCALE_PARAM = 127.0   # INT8量化参数
  BLOCK_NUM = 8         # Block数

Process():
  ├── [IsNeedReduceScatter] ReduceScatterTrans()
  │   └── TP=2场景: ReduceScatter跨TP域同步
  ├── AlltoAllBuffInit(): UB队列初始化
  │   ├── readStateBuf_     ← 状态等待
  │   ├── expertIdsBuf_     ← expert_ids
  │   ├── expandScalesBuf_  ← expert_scales
  │   ├── rowTmpFloatBuf_   ← 中间计算 (float)
  │   ├── sumFloatBuf_      ← 累加缓冲
  │   ├── mulBuf_           ← 乘法缓冲
  │   └── [IsQuant] xAbsBuf_, xInt8Buf_, xMaxBuf_ ← 量化
  ├── SetStatus(): 设置本卡EP状态标志
  ├── WaitDispatch(): 等待所有EP Rank的Dispatch完成
  ├── LocalWindowCopy(): 从Window拷贝Expert结果
  ├── ExpertAlltoAllDispatchCopyAdd(): 核心加权聚合
  │   ├── 对每个Token:
  │   │   ├── 读取expandIdx, expertId, scale
  │   │   ├── 从Window读取对应Expert输出 (1 token)
  │   │   ├── Cast BF16/FP16 → Float
  │   │   ├── Muls(× scale) → 加权
  │   │   └── Add → sumBuf (累加K个Expert)
  │   └── Cast Float → 输出类型, 写出
  ├── [IsQuant] DequantProcess():
  │   └── INT8 → BF16/FP16 反量化
  └── 输出 x[BS, H]
```

### 4.5 V3 Kernel (委托V2)

**Dispatch V3** `moe_distribute_dispatch_v3.cpp`（第35-39行 Kernel 入口函数；第46-96行 A3 路径分发，委托V2实现）:
```cpp
__global__ __aicore__ void moe_distribute_dispatch_v3(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales,
    GM_ADDR xActiveMask, GM_ADDR expertScales, GM_ADDR elasticInfo,
    GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, ...)
{
    // 仅A3架构
    if (FullMesh) {
        MoeDistributeDispatchV2FullMesh<...>(...);  // 来自V2
    } else {
        MoeDistributeDispatchV2<...>(...);           // 来自V2
    }
}
```

**Combine V3** `moe_distribute_combine_v3.cpp`（第62-69行 Kernel 入口函数；第74-79行 A3 路径分发，委托V2实现）:
```cpp
void ExecMoeDistributeCombineV3(...) {
    MoeDistributeCombineV2<ExpandXType, XType, int32_t,
                           HasTp, QuantMode, false>(...);  // 来自V2
}
```

### 4.6 模板Key编码

**Dispatch TilingKey** `moe_distribute_dispatch_tiling_key.h`（第35-46行 `ASCENDC_TPL_ARGS_DECL` 声明；第47-99行 `ASCENDC_TPL_SEL` 模板选择）：
```
6位参数: TP(bool,1bit) | QuantMode(4bit) | Scale(bool,1bit) | FullMesh(2bit) | CommMode(2bit) | ArchTag(2bit)

编码值:
  TILINGKEY_TPL_A2 = 0, TILINGKEY_TPL_A3 = 1, TILINGKEY_TPL_A5 = 2
  NO_QUANT=0, STATIC_QUANT=1, PERTOKEN_QUANT=2, PERGROUP_QUANT=3, MX_QUANT=4
  MTE=0, AICPU=1, CCU=2

7种组合: A5×{无量化,静态,逐Token} + A3×{无量化} + A2×{MTE,AICPU}
```

**Combine TilingKey** `moe_distribute_combine_tiling_key.h`（第29-37行 `ASCENDC_TPL_ARGS_DECL` 声明；第38-73行 `ASCENDC_TPL_SEL` 模板选择）:
```
4位参数: TP(bool) | QuantMode(2bit: NO_QUANT=0, INT8_QUANT=1) | LayerMode(2bit) | ArchTag(2bit)

4种组合: A5×{无量化} + A3×{无量化} + A2×{MTE} + A2×{AICPU_INT8}
```

---

## 5. Host-Kernel 接口

### 5.1 Tiling数据结构

V1使用 `MoeDistributeDispatchTilingData` (A3/A5) 或 `MoeDistributeDispatchA2TilingData` (A2):

```
TilingData 内存布局:
┌───────────────────────────────────────────────────────────────┐
│ 偏移 0x0                                                      │
│ ┌─────────────────────────────────────────────────────────┐   │
│ │ Mc2InitTiling (HCCL初始化)                               │   │
│ │   ├── rankId, rankDim, rankSize                         │   │
│ │   ├── hcclSqrBufOffset, hcclSqrBufSize                  │   │
│ │   └── groupInfo (EP/TP域配置)                            │   │
│ ├─────────────────────────────────────────────────────────┤   │
│ │ Mc2CcTiling (通信配置)                                   │   │
│ │   ├── commAlgo: FULL_MESH / DOUBLE_RING / SWITCH_WING   │   │
│ │   ├── cclBufferOffset, cclBufferSize                    │   │
│ │   └── commMode: MTE / AICPU / CCU                       │   │
│ ├─────────────────────────────────────────────────────────┤   │
│ │ MoeDistributeInfo (业务参数)                             │   │
│ │   ├── bs, h, k, expertNum                               │   │
│ │   ├── epWorldSize, epRankId                             │   │
│ │   ├── quantMode, scaleMode                              │   │
│ │   ├── axisMaxBS (最大BS上界，决定Window槽位大小)          │   │
│ │   └── expertPerSizeOnWin (每Expert Window槽位字节数)     │   │
│ └─────────────────────────────────────────────────────────┘   │
│ 总大小: ~61KB (Dispatch) / ~55KB (Combine)                    │
└───────────────────────────────────────────────────────────────┘
```

V3使用 `MoeDistributeDispatchV2TilingData`，通过config映射后完全复用V2结构，总大小约 ~5.3KB / ~6.1KB（更紧凑的布局）。

### 5.2 HCCL上下文

V1通过 `GetHcclContext<HCCL_GROUP_ID_0>()` (EP域) 和 `HCCL_GROUP_ID_1` (TP域) 获取:
```
HcclCombinOpParam 结构:
┌────────────────────────────────────────────────────────────┐
│ windowsIn[epWorldSize]                                     │
│   每个元素指向对应Rank的Window数据区起始地址 (GM_ADDR)      │
│   用于: AlltoAll接收缓冲 → 本地拷贝源                      │
├────────────────────────────────────────────────────────────┤
│ windowsOut[epWorldSize]                                    │
│   每个元素指向对应Rank的Window状态区起始地址 (GM_ADDR)      │
│   用于: SetStatus(写flag) / WaitDispatch(轮询flag)         │
├────────────────────────────────────────────────────────────┤
│ URMA SQ/CQ 资源                                            │
│   SQ (Submission Queue): 远端写入请求队列                  │
│   CQ (Completion Queue): 写入完成通知队列                  │
│   用于A2 BatchWrite通信路径                                │
└────────────────────────────────────────────────────────────┘
```

V3通过 `mc2Context` 输入张量传递，在V2内部解析。mc2Context 是由 `moe_distribute_dispatch_setup` 算子预先分配的持久化资源，包含 HCCL 句柄和 Window 地址。

### 5.3 Window共享内存布局

Window 是 HCCL 提供的跨Rank共享内存区域，所有Rank通过 URMA/RDMA 直接读写远端 Window。

```
Window 空间总布局 (每个Rank视角):
┌──────────────────────────────────────────────────────────────────┐
│ 偏移 0x0                                                          │
│ ┌────────────────────────────────────────────────────────────┐   │
│ │ 数据区 (双缓冲: Buffer 0)                                   │   │
│ │   大小 = epWorldSize × localExpertNum × expertPerSizeOnWin │   │
│ │   expertPerSizeOnWin = axisMaxBS × H × sizeof(type)        │   │
│ │                                                            │   │
│ │   ┌─────────────┬─────────────┬─────────┬─────────────┐   │   │
│ │   │ Rank 0       │ Rank 1       │ ...     │ Rank N-1    │   │   │
│ │   │  Expert 0    │  Expert 0    │         │  Expert 0   │   │   │
│ │   │  Expert 1    │  Expert 1    │         │  Expert 1   │   │   │
│ │   │  ...         │  ...         │         │  ...        │   │   │
│ │   │  Expert E-1  │  Expert E-1  │         │  Expert E-1 │   │   │
│ │   └─────────────┴─────────────┴─────────┴─────────────┘   │   │
│ │   每个Expert槽位: axisMaxBS × H 个元素                      │   │
│ └────────────────────────────────────────────────────────────┘   │
│ 偏移 = dataBufferSize                                             │
│ ┌────────────────────────────────────────────────────────────┐   │
│ │ 数据区 (双缓冲: Buffer 1)                                   │   │
│ │   布局与 Buffer 0 完全相同                                  │   │
│ │   通过 dataState_ (0/1) 自动切换                            │   │
│ └────────────────────────────────────────────────────────────┘   │
│ 偏移 = 2 × dataBufferSize                                        │
│ ┌────────────────────────────────────────────────────────────┐   │
│ │ 状态区 (~1MB)                                               │   │
│ │   每个远端Rank占 512B 间距 (moeSendNum > 512时为 256B)      │   │
│ │   存储 flag = 1.0f 表示该Rank数据已就绪                     │   │
│ │                                                            │   │
│ │   ┌──────────┬──────────┬─────────┬──────────┐           │   │
│ │   │ Rank 0    │ Rank 1    │ ...     │ Rank N-1  │           │   │
│ │   │ flag(4B)  │ flag(4B)  │         │ flag(4B)  │           │   │
│ │   │ pad(508B) │ pad(508B) │         │ pad(508B) │           │   │
│ │   └──────────┴──────────┴─────────┴──────────┘           │   │
│ └────────────────────────────────────────────────────────────┘   │
│ 偏移 = STATE_WIN_OFFSET (~950KB)                                  │
│ ┌────────────────────────────────────────────────────────────┐   │
│ │ 本卡状态标识 (512B / core)                                  │   │
│ │   存储双缓冲索引 dataState_ (0 或 1)                        │   │
│ │   各核独立读写，用于确定当前应使用 Buffer 0 还是 Buffer 1    │   │
│ └────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘

数据写入流程 (Dispatch 阶段):
  HostRank[i] → SendToMoeExpert()
    → 计算目标偏移: winOffset = rankIdx × localExpertNum × expertPerSizeOnWin
                     + expertIdx × expertPerSizeOnWin
                     + tokenIdx × H
    → DataCopy(localToken, windowGM + winOffset)
  HostRank[i] → SetStatus(flag=1.0f)
    → 写入远端Rank的Window状态区对应位置
```

---

## 6. 数据流

### 6.1 V1 Dispatch 完整数据流

```
┌──────────────────────────────────────────────────────────────────────┐
│ x[BS,H] + expert_ids[BS,K]                                          │
│     │                                                                │
│     ▼                                                                │
│ ┌─────────────────────────────┐                                     │
│ │ CalTokenSendExpertCnt()     │ ← 统计每Expert需发送的Token数        │
│ └────────────┬────────────────┘                                     │
│              │                                                       │
│     ┌────────┴────────┐                                             │
│     ▼                 ▼                                              │
│ [量化模式]        [非量化模式]                                        │
│ BF16→INT8         直接BF16                                           │
│ dynamic_scales                                                        │
│     │                 │                                              │
│     └────────┬────────┘                                             │
│              ▼                                                       │
│ ┌─────────────────────────────┐    ┌───────────────────────────┐   │
│ │ SendToMoeExpert()           │    │ SendToSharedExpert()      │   │
│ │ 按Expert写入Window各Rank   │    │ 共享专家Token写入Window   │   │
│ │ expandX → WinIn[rank][ep]  │    │                           │   │
│ └────────────┬────────────────┘    └────────────┬──────────────┘   │
│              └──────────┬───────────────────────┘                   │
│                         ▼                                           │
│              SetStatus(flag=1.0f)                                    │
│              → WinOut[远端Rank]状态区                                │
│                         │                                           │
│              WaitDispatch()                                          │
│              ← polling 本卡WinOut状态区                              │
│              (等待所有远端Rank发送完成)                               │
│                         │                                           │
│              LocalWindowCopy()                                       │
│              WinIn[本卡] → expand_x[A,H] (按Expert重排)             │
│                         │                                           │
│              [HasTp] AllGatherSetStatusAndWait()                     │
│              TP=2场景额外同步                                        │
│                         ▼                                           │
│ 输出: expand_x, expand_idx, expert_token_nums,                      │
│       ep_recv_count, tp_recv_count, expand_scales                   │
└──────────────────────────────────────────────────────────────────────┘
```

### 6.2 V1 Combine 完整数据流

```
┌──────────────────────────────────────────────────────────────────────┐
│ expand_x[A,H] + expert_scales[BS,K] + expert_ids[BS,K]             │
│     │                                                                │
│     ▼                                                                │
│ [HasTp] ReduceScatterTrans() ← TP域ReduceScatter                    │
│     │                                                                │
│     ▼                                                                │
│ SetStatus() → 通知所有Rank本卡数据就绪                                │
│     │                                                                │
│ WaitDispatch() ← 等待所有Rank的expand_x写入Window完成                │
│     │                                                                │
│ LocalWindowCopy() ← Window数据拷贝                                   │
│     │                                                                │
│ ExpertAlltoAllDispatchCopyAdd():                                     │
│   对每个Token t ∈ [0, BS):                                          │
│     清零 sumFloatBuf[H]                                              │
│     对每个Expert k ∈ [0, K):                                        │
│       indexCount = expandIdx[t*K + k]                                │
│       expertId = expertIds[t*K + k]                                  │
│       scale = expertScales[t*K + k]                                  │
│       winAddr = epWindowGM + expertOffset(expertId) + indexCount*H   │
│       DataCopy: Win[winAddr] → UB (1 token)                         │
│       Cast: BF16/FP16 → Float                                       │
│       Muls: × scale                                                  │
│       Add: → sumFloatBuf (累加)                                      │
│     Cast: Float → BF16/FP16                                          │
│     DataCopy: sumFloatBuf → xOut[t*H]                                │
│     │                                                                │
│ [IsQuant] DequantProcess(): INT8 → BF16/FP16                        │
│     ▼                                                                │
│ 输出: x[BS, H]                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 7. 性能设计

### 7.1 双缓冲零拷贝通信

与auxiliary_ops相同的双缓冲策略:
- Window数据区分为Buffer 0/1，通过 `dataState_` 自动切换
- 状态区 `STATE_WIN_OFFSET` 存储per-core的双缓冲标识
- 避免了额外的数据拷贝开销

### 7.2 向量化状态轮询

`WaitDispatch` 使用 GatherMask + Sum 替代标量逐个轮询:
```
一次处理 sendRankNum 个状态:
  DataCopy(状态) → GatherMask(过滤有效) → Sum(累加) → 标量比较
```

### 7.3 多核负载均衡

- **Rank维分核**: `SplitToCore()` 将epWorldSize均匀分配到aivNum个核
- **Token维分核**: Combine阶段额外按BS维度分配token给各核
- 核心数偏移: `(coreIdx + epRankId) % aivNum` 避免不同Rank争抢同一SQ/CQ

### 7.4 TP域优化

- Dispatch: AllGather跨TP Rank同步 (仅 TP=2 时触发)
- Combine: ReduceScatter跨TP域聚合
- 两个通信域 (EP/TP) 使用不同的HCCL_GROUP_ID，互不干扰

### 7.5 量化路径优化

- **静态量化**: 预计算scale，直接量化写入Window
- **逐Token量化**: 每Token独立计算max值，输出dynamic_scales
- Combine端: DequantProcess 反量化后参与加权聚合

### 7.6 Batch调度模式

所有算子设置 `SetScheduleMode(1)` (batch mode)，所有AIV核同时启动。

---

## 8. 依赖关系

```
moe_distribute_dispatch (V1):
  ├── common/op_kernel/mc2_kernel_utils.h           ← 通信原语
  ├── common/op_kernel/moe_distribute_base.h        ← HcclCombinOpParam
  ├── common/op_host/op_tiling/moe_tiling_base.h    ← MOE Tiling基类
  ├── [A2路径] moe_distribute_dispatch_v2/           ← V2 Kernel类和Tiling
  └── kernel_tiling/kernel_tiling.h                  ← Mc2InitTiling/Mc2CcTiling

moe_distribute_combine (V1):
  ├── common/op_kernel/mc2_kernel_utils.h
  ├── common/op_kernel/moe_distribute_base.h
  ├── common/op_host/op_tiling/moe_tiling_base.h
  ├── [A2路径] moe_distribute_combine_v2/            ← V2 Kernel类和Tiling
  └── kernel_tiling/kernel_tiling.h

moe_distribute_dispatch_v3:
  └── moe_distribute_dispatch_v2/                    ← 完全委托V2
      ├── op_kernel/ (MoeDistributeDispatchV2, MoeDistributeDispatchV2FullMesh)
      └── op_host/op_tiling/ (MoeDistributeDispatchA3TilingFuncImplPublic)

moe_distribute_combine_v3:
  └── moe_distribute_combine_v2/                     ← 完全委托V2
      ├── op_kernel/ (MoeDistributeCombineV2)
      └── op_host/op_tiling/ (MoeDistributeCombineV2TilingFuncNew)

跨算子数据传递:
  dispatch.expand_x → [Expert计算] → combine.expand_x
  dispatch.ep_recv_count → combine.ep_send_counts
  dispatch.expand_idx → combine.expand_idx
  dispatch.expand_scales → combine.expand_scales (量化场景)

V3额外:
  dispatch_v3.assist_info_for_combine → combine_v3.assist_info_for_combine
```

### 8.1 V1 vs V3 对比总结

| 特性 | V1 | V3 |
|------|----|----|
| 平台 | 910B/910_93/950 | 仅910_93 |
| Kernel实现 | 自有类 (52KB+42KB) | 委托V2 (无自有.h) |
| Tiling实现 | 自有 (~1077行+985行) | 委托V2 (107行+104行) |
| 通信管理 | HcclGroup字符串 | mc2Context张量 |
| 数据类型 | BF16/FP16/INT8 | +FP8/HF8等19种 |
| 量化模式 | NO/STATIC/PERTOKEN | +PERGROUP/MX |
| Dispatch输入数 | 5 | 8 |
| Combine输入数 | 10 | 18 |
| 共享专家 | 内置处理 | 独立输入shared_expert_x |
| 常量专家 | 不支持 | alpha_1/alpha_2/v |
| 弹性训练 | 不支持 | elastic_info |
| 性能采集 | 不支持 | performance_info |
| 辅助信息 | 无 | assist_info_for_combine |
| ACLNN API | aclnnMoeDistributeDispatch | aclnnMoeDistributeDispatchV5 |
| Tiling大小 | ~61KB/55KB | ~5.3KB/6.1KB |

---

*分析完成时间: 2026-04-20，修订时间: 2026-04-21（补充行号引用、修正章节编号为x.y格式）*
