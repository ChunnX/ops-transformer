# MOE 辅助算子全面分析报告

## 1. 算子概述

本报告覆盖 MOE (Mixture of Experts) 分布式推理中的5个辅助算子，它们共同构成 EP (Expert Parallelism) 场景下 Token-Expert 路由与聚合的完整生命周期。这些算子将传统 `moe_distribute_dispatch`/`moe_distribute_combine` 的功能拆分为 setup-teardown 对，支持在通信阶段之间插入自定义 Expert 计算逻辑。

```
完整调用时序:

  [moe_update_expert]           ← Expert ID 重映射 (EPLB) + 可选剪枝
          │
  [dispatch_setup]              ← 量化输入 + 准备通信元数据
          │
    (HCCL AllToAllV / URMA)     ← 跨卡 Token 分发
          │
  [dispatch_teardown]           ← 接收数据 + Token 按 Expert 重排
          │
    [Expert 计算: GMM/FFN]      ← 用户自定义 Expert 层计算
          │
  [combine_setup]               ← 量化 Expert 输出 + URMA 发送
          │
    (URMA WriteWithNotify)      ← 跨卡 Expert 结果回传
          │
  [combine_teardown]            ← 接收 + 反量化 + 加权聚合 → 最终输出
```

平台支持: ascend910_93 (arch35), ascend950 (arch32)。

---

## 2. 目录结构

```
mc2/
├── moe_update_expert/                    # Expert ID 重映射算子
│   ├── op_kernel/
│   │   ├── moe_update_expert.cpp         # Kernel入口 (64行): 模板分发
│   │   └── moe_update_expert.h           # 核心类 (450行): EPLB查表+剪枝
│   ├── op_host/
│   │   ├── moe_update_expert_def.cpp     # 算子定义 (79行): 5输入/2输出/3属性
│   │   └── op_tiling/                    # Tiling实现
│   └── op_api/                           # ACLNN接口
│
├── moe_distribute_dispatch_setup/        # Dispatch 准备阶段
│   ├── op_kernel/
│   │   ├── moe_distribute_dispatch_setup.cpp         # Kernel入口 (27行)
│   │   └── moe_distribute_dispatch_setup_tiling.h    # Tiling数据结构 (46行)
│   ├── op_host/
│   │   ├── moe_distribute_dispatch_setup_def.cpp     # 算子定义 (88行)
│   │   └── op_tiling/
│   │       ├── moe_distribute_dispatch_setup_tiling.cpp          # Tiling注册 (43行)
│   │       ├── moe_distribute_dispatch_setup_tiling_base.cpp     # Tiling基类 (~803行)
│   │       ├── arch32/..._tiling_arch32.cpp                      # A3平台Tiling
│   │       └── arch35/..._tiling_arch35.h                        # A5平台Tiling
│   └── op_api/
│
├── moe_distribute_dispatch_teardown/     # Dispatch 拆解阶段
│   ├── op_kernel/
│   │   ├── moe_distribute_dispatch_teardown.cpp         # Kernel入口 (27行)
│   │   └── moe_distribute_dispatch_teardown_tiling.h    # Tiling数据结构 (46行)
│   ├── op_host/
│   │   ├── moe_distribute_dispatch_teardown_def.cpp     # 算子定义 (88行)
│   │   └── op_tiling/
│   │       ├── moe_distribute_dispatch_teardown_tiling.cpp
│   │       ├── moe_distribute_dispatch_teardown_tiling_base.cpp  # (~777行)
│   │       ├── arch32/...
│   │       └── arch35/...
│   └── op_api/
│
├── moe_distribute_combine_setup/         # Combine 准备阶段
│   ├── op_kernel/
│   │   ├── moe_distribute_combine_setup.cpp                  # Kernel入口 (51行)
│   │   ├── moe_distribute_combine_setup_tiling_data.h        # Tiling数据 (60行)
│   │   ├── moe_distribute_combine_setup_tiling_key.h         # 模板Key (30行)
│   │   ├── moe_distribute_base.h                             # HCCL基类
│   │   └── arch35/moe_distribute_combine_setup_arch35.h      # A5核心实现 (~485行)
│   ├── op_host/
│   │   ├── moe_distribute_combine_setup_def.cpp              # 算子定义 (84行)
│   │   ├── moe_distribute_combine_setup_infershape.cpp       # Shape推导 (151行)
│   │   └── op_tiling/
│   │       ├── moe_distribute_combine_setup_tiling_base.cpp   # (~708行)
│   │       └── arch32/arch35/
│   └── op_api/
│       └── aclnn_moe_distribute_combine_setup.h              # ACLNN API (104行)
│
├── moe_distribute_combine_teardown/      # Combine 拆解阶段
│   ├── op_kernel/
│   │   ├── moe_distribute_combine_teardown.cpp               # Kernel入口 (49行)
│   │   ├── moe_distribute_combine_teardown_tiling_data.h     # Tiling数据 (47行)
│   │   ├── moe_distribute_combine_teardown_tiling_key.h      # 模板Key (30行)
│   │   └── arch35/moe_distribute_combine_teardown_arch35.h   # A5核心实现 (~396行)
│   ├── op_host/
│   │   ├── moe_distribute_combine_teardown_def.cpp           # 算子定义 (97行)
│   │   ├── moe_distribute_combine_teardown_infershape.cpp    # Shape推导 (147行)
│   │   └── op_tiling/
│   │       └── moe_distribute_combine_teardown_tiling_base.cpp
│   └── op_api/
│       └── aclnn_moe_distribute_combine_teardown.h           # ACLNN API (82行)
```

---

## 3. Host 侧分析

### 3.1 moe_update_expert 定义

`moe_update_expert_def.cpp`（第19行类定义；第22-46行输入张量；第48-55行输出张量；第57-59行属性；第77行 `OP_ADD` 注册）

```
输入:
  x                  REQUIRED  [BS, K]   INT32/INT64     Expert ID序列
  eplb_table         REQUIRED  [专家数, 物理设备数] INT32  EPLB重映射表
  expert_scales      OPTIONAL  [BS, K]   FLOAT           Gating scale (剪枝用)
  pruning_threshold  OPTIONAL  scalar    FLOAT           剪枝阈值系数
  active_mask        OPTIONAL  [BS, K]   BOOL            已激活的Expert mask

输出:
  balanced_expert_ids  REQUIRED  [BS, K]  INT32/INT64   重映射后的物理Expert ID
  balanced_active_mask REQUIRED  [BS, K]  BOOL           剪枝后的激活mask

属性:
  local_rank_id    REQUIRED  Int    当前Rank在EP域中的ID
  world_size       REQUIRED  Int    EP域总卡数
  balance_mode     REQUIRED  Int    0=RankID模式, 1=TokenID模式
```

平台: ascend910_93, ascend950。数据类型支持 INT32/INT64 Expert ID。

### 3.2 moe_distribute_dispatch_setup 定义

`moe_distribute_dispatch_setup_def.cpp`（第19行类定义；第23-42行输入张量；第44-55行输出张量；第57-67行属性；第86行 `OP_ADD` 注册）

```
输入:
  x              REQUIRED  [BS, H]    BF16/FP16    输入Token特征
  expert_ids     REQUIRED  [BS, K]    INT32        每个Token的TopK Expert ID
  scales         OPTIONAL  [专家数, H] FLOAT       静态量化scale
  x_active_mask  OPTIONAL  [BS]       BOOL         Token激活mask

输出:
  y              REQUIRED  [BS*(K+sharedExpertNum), tokenMsgSize]  BF16/FP16/INT8  发送数据
  expand_idx     REQUIRED  [BS*K]     INT32        每个Token在其Expert内的序号
  comm_cmd_info  REQUIRED  [1D]       INT32        HCCL通信命令元数据

属性:
  group_ep             REQUIRED  String   EP通信域名称
  ep_world_size        REQUIRED  Int      EP域大小
  ep_rank_id           REQUIRED  Int      本卡Rank ID
  moe_expert_num       REQUIRED  Int      MoE专家总数
  expert_shard_type    OPTIONAL  Int(0)   专家分片类型
  shared_expert_num    OPTIONAL  Int(1)   共享专家数量 [0, 4]
  shared_expert_rank_num OPTIONAL Int(0)  共享专家卡数
  quant_mode           OPTIONAL  Int(0)   量化模式: 0=不量化, 2=动态量化
  global_bs            OPTIONAL  Int(0)   全局BatchSize (0=均分)
  comm_type            OPTIONAL  Int(0)   通信类型: 0=SDMA
  comm_alg             OPTIONAL  String("") 通信算法
```

MC2配置: `HcclGroup({"group_ep"})`, 仅支持 ascend950。

### 3.3 moe_distribute_dispatch_teardown 定义

`moe_distribute_dispatch_teardown_def.cpp`（第19行类定义；第23-42行输入张量；第44-53行输出张量；第55-66行属性；第85行 `OP_ADD` 注册）

```
输入:
  x              REQUIRED  [BS, H]    BF16/FP16/INT8   原始输入
  y              REQUIRED  [BS*(K+sharedExpertNum), tokenMsgSize]  通信结果
  expert_ids     REQUIRED  [BS, K]    INT32
  comm_cmd_info  REQUIRED  [1D]       INT32            来自setup的通信命令

输出:
  expand_x                 REQUIRED  [A, H]     BF16/FP16/INT8  按Expert重排的Token
  dynamic_scales           REQUIRED  [A]        FLOAT            动态量化scale
  assist_info_for_combine  REQUIRED  [A*128]    INT32            Combine辅助信息
  expert_token_nums        REQUIRED  [localExpertNum] INT64      每个Expert的Token计数

属性: 与dispatch_setup基本一致，额外增加 expert_token_nums_type(0或1)
```

### 3.4 moe_distribute_combine_setup 定义

`moe_distribute_combine_setup_def.cpp`（第19行类定义；第23-40行输入张量；第41-50行输出张量；第52-62行属性；第81行 `OP_ADD` 注册）

```
输入:
  expand_x               REQUIRED  [A, H]     BF16/FP16    Expert计算结果
  expert_ids             REQUIRED  [BS, K]    INT32
  assist_info_for_combine REQUIRED [A*128]    INT32        来自dispatch_teardown

输出:
  quant_expand_x         REQUIRED  [A, hs]    INT8         量化后数据
  comm_cmd_info          REQUIRED  [1D]       INT32        通信命令

属性: 增加 comm_quant_mode(预留, 当前仅支持0)
```

### 3.5 moe_distribute_combine_teardown 定义

`moe_distribute_combine_teardown_def.cpp`（第20行类定义；第24-63行输入张量；第64行输出张量；第65-75行属性；第94行 `OP_ADD` 注册）

```
输入:
  expand_x       REQUIRED  [A, H]      BF16/FP16    Expert输出
  quant_expand_x REQUIRED  [A, hs]     INT8          量化数据
  expert_ids     REQUIRED  [BS, K]     INT32
  expand_idx     REQUIRED  [BS, K]     INT32         每Token的Expert内序号
  expert_scales  REQUIRED  [BS, K]     FLOAT         Gating scale
  comm_cmd_info  REQUIRED  [1D]        INT32
  x_active_mask  OPTIONAL  [BS]        BOOL          预留
  shared_expert_x OPTIONAL  [?, H]     BF16/FP16    共享专家输出

输出:
  x              REQUIRED  [BS, H]     BF16/FP16    最终Combine结果
```

### 3.6 InferShape 分析

**combine_setup InferShape** (`moe_distribute_combine_setup_infershape.cpp` 第70行 `InferShape` 入口；第119-127行 `quantExpandX` shape 推导；第129-132行 `commCmdInfo` shape 推导；第148-150行注册）:
- `quantExpandX`: `[A, hs]`，其中 `hs = Align(Align(H, 32) + Align(H, 8)/8*sizeof(float), 512)`
  - 每个Token消息包含: Align(H, 32)字节的量化数据 + 对应的scale空间，整体512对齐
- `commCmdInfo`: `[(A + epWorldSize) * 16]`
  - 包含发送和接收两个维度的通信命令，每条命令16字节

**combine_teardown InferShape** (`moe_distribute_combine_teardown_infershape.cpp` 第59行 `InferShape` 入口；第118-124行 `x` shape 推导；第144-146行注册）:
- `x_out`: `[BS, H]` — 恢复为原始Token维度
- 数据类型: 与输入 `expand_x` 保持一致

### 3.7 Tiling 分析

所有setup/teardown算子的Tiling遵循统一模式，以 `MoeDistributeDispatchSetupTilingBase` 为典型代表:

**Tiling流程** (`moe_distribute_dispatch_setup_tiling_base.cpp` 第767-802行主Tiling实现；第680-700行 `SetTilingKey`；第702-734行 `CheckHcclBuffSize`）:
```
MoeDistributeDispatchSetupTilingFuncImpl():
  ├── GetRequiredAttrAndSetTilingData()    ← 获取group_ep/epWorldSize/epRankId/moeExpertNum
  │   └── CheckRequiredAttrValue()         ← 范围校验: epWorldSize∈[2,384], moeExpertNum∈(0,512]
  ├── GetOptionalAttrAndSetTilingData()    ← expert_shard_type/quant_mode/global_bs等
  │   ├── CheckOptionalAttrValue()         ← expertShardType=0, quantMode∈{0,2}, globalBs∈[0,BS*epWorldSize]
  │   └── CheckSharedExpertAttrValue()     ← sharedExpertNum<=sharedExpertRankNum, 整除性
  ├── GetComplexAttrAndSetTilingData()     ← 计算moeExpertPerRankNum
  │   └── moeExpertNum % (epWorldSize - sharedExpertRankNum) == 0
  ├── CheckTensorDataType()                ← x:BF16/FP16, expertIds:INT32, y:与x相同(非量化)或INT8(量化)
  ├── CheckTensorDim()                     ← x:[BS,H], expertIds:[BS,K], y:[A,tokenMsgSize], expandIdx:[BS*K]
  ├── CheckTensorShapeRelation()           ← x.dim0 == expertIds.dim0 == BS
  ├── CheckTensorShapeSizeAndSetTilingData()  ← H∈[1024,8192], BS∈(0,512], K∈(0,min(16,moeExpertNum)]
  │   └── CheckComplexTensorShapeSize()    ← y.dim0 == BS*(K+sharedExpertNum)
  ├── CheckCalcTensorShapeSizeAndSetTilingData()  ← expandIdx=BS*K, commCmdInfo=(BS*(K+shared)+epWorld*local)*16
  ├── CheckHcclBuffSize()                  ← Window空间充足性校验
  ├── SetPlatformInfo()                    ← blockDim=aivNum, UB size, scheduleMode=batch
  ├── SetWorkspace()                       ← 32MB (16MB系统 + 16MB SDMA)
  └── SetTilingKey()                       ← 4位十进制: 千位=1, 十位=scales, 个位=quantMode
```

**TilingKey编码** (dispatch_setup, `moe_distribute_dispatch_setup_tiling_base.cpp` 第680-700行）:
```
千位: 固定为1
百位: 0
十位: 0=无scales, 1=有scales (smoothScale)
个位: quantMode (0=不量化, 2=动态量化)

例: 1000 = 无量化无scales, 1010 = 无量化有scales, 1002 = 动态量化无scales
```

**TilingKey编码** (dispatch_teardown, `moe_distribute_dispatch_teardown_tiling_base.cpp` 第683-695行）:
```
基础值10000 + quantMode + (isSharedExpert ? 1000 : 0)
```

**combine_setup/teardown TilingKey** 使用模板参数编码:
- `moe_distribute_combine_setup_tiling_key.h`（第21-23行 `ASCENDC_TPL_ARGS_DECL`；第27-29行 `ASCENDC_TPL_SEL`）
- `moe_distribute_combine_teardown_tiling_key.h`（第21-23行 `ASCENDC_TPL_ARGS_DECL`；第27-29行 `ASCENDC_TPL_SEL`）
```cpp
ASCENDC_TPL_BOOL_DECL(TILINGKEY_TP, 0, 1)  // 是否有TP维
ASCENDC_TPL_BOOL_SEL(TILINGKEY_TP, 0)       // 当前仅支持 TP=0
```

**HCCL Buffer Size校验** (dispatch_setup, `moe_distribute_dispatch_setup_tiling_base.cpp` 第702-734行）:
```
所需Window大小 = 2 × localExpertNum × maxBS × epWorldSize × align
                + (K + sharedExpertNum) × maxBS × Align(2H, 512)

其中 align = CeilAlign(CeilAlign(2H, 32) + 44, 512)
```

**各算子Tiling注册入口**:
- `moe_update_expert_tiling.cpp`（第506-512行Tiling入口函数；第523-525行注册）
- `moe_distribute_dispatch_setup_tiling.cpp`（第25-28行Tiling入口；第40-42行注册）
- `moe_distribute_combine_setup_tiling.cpp`（第27-28行模板声明；第30-37行Tiling入口；第47-49行注册）
- `moe_distribute_combine_teardown_tiling.cpp`（第27-28行模板声明；第30-37行Tiling入口；第46-48行注册）

**各算子Tiling数据结构**:
- `moe_update_expert_tiling.h`（第23-33行 `MoeUpdateExpertTilingData` 结构体）
- `moe_distribute_dispatch_setup_tiling.h`（第18-39行 `MoeDistributeDispatchSetupInfo`；第41-45行 `TilingData`）
- `moe_distribute_dispatch_teardown_tiling.h`（第18-39行信息结构体；第41-45行 `TilingData`）
- `moe_distribute_combine_setup_tiling_data.h`（第22-39行信息结构体；第41-45行 `TilingData`；第48-57行 `BatchWriteItem`）
- `moe_distribute_combine_teardown_tiling_data.h`（第21-38行信息结构体；第40-44行 `TilingData`）

**combine_setup/teardown Tiling基类实现**:
- `moe_distribute_combine_setup_tiling_base.cpp`（第655-690行主Tiling实现；第578-586行 `SetTilingKey`）
- `moe_distribute_combine_teardown_tiling_base.cpp`（第670-704行主Tiling实现；第601-609行 `SetTilingKey`）

---

## 4. Kernel 侧分析

### 4.1 moe_update_expert Kernel

**入口** (`moe_update_expert.cpp` 第35-63行）:
```cpp
template <int32_t BalanceMode, int32_t ExpertScalesDataType>
__aicore__ void moe_update_expert(GM_ADDR expertIds, GM_ADDR eplbTable, ...)
// BalanceMode: 0=RANK_ID_BALANCING_MODE, 1=TOKEN_ID_BALANCING_MODE
// ExpertScalesDataType: 控制剪枝分支
```

**Tiling模板参数** (`moe_update_expert_tiling_key.h` 第26-29行 `ASCENDC_TPL_ARGS_DECL`；第30-42行 `ASCENDC_TPL_SEL`）

**核心类** (`moe_update_expert.h` 第44-449行 `MoeUpdateExpert` 类；第123-177行 `Init()`；第424-448行 `Process()`）:
```
MoeUpdateExpert<DataType, ScalesType, EnablePruning>
  ├── Init()         ← GM buffer初始化, 获取tiling参数
  ├── Process()      ← 主流程入口
  │   ├── [EnablePruning] ProcessPruning()
  │   │   ├── CalculateUnroll()  ← 批量读取expert_scales
  │   │   ├── 计算scale_sum = Σ(scale_i)
  │   │   ├── threshold = pruning_threshold × scale_sum
  │   │   ├── Compare(scale >= threshold) → 生成active_mask
  │   │   └── 输出 balanced_active_mask[BS, K]
  │   └── [RANK_ID_MODE]:
  │       physical_id = eplb_table[expertId * stride + rankId]
  │   [TOKEN_ID_MODE]:
  │       physical_id = eplb_table[expertId * stride + tokenId % expertNum]
  ├── CalculateUnroll()  ← UB内unroll读取辅助函数
  └── Calculate()        ← EPLB表查找核心计算
```

**关键实现**:
- EPLB表查找是纯标量操作 (每个element一次查表), 无向量化优化
- 剪枝使用 `Compare` 指令生成布尔mask, 支持 UB → UB 的 in-place 操作
- `CalculateUnroll` 使用模板参数展开循环减少分支开销

### 4.2 moe_distribute_dispatch_setup/teardown Kernel

**dispatch_setup Kernel入口** (`moe_distribute_dispatch_setup.cpp` 第21-27行）:
```cpp
extern "C" __global__ __aicore__ void moe_distribute_dispatch_setup(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask,
    GM_ADDR YOut, GM_ADDR expandIdxOut, GM_ADDR commCmdInfoOut,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeDispatchSetupTilingData);
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchSetupTilingData, tilingData, tilingGM);
}
```

注意: dispatch_setup/teardown 的Kernel入口只解析tiling数据，实际的Kernel逻辑由 `REGISTER_TILING_DEFAULT` 宏内部调度，真正的实现在 arch-specific 的代码中。

**Tiling数据结构** (`moe_distribute_dispatch_setup_tiling.h` 第18-39行 `MoeDistributeDispatchSetupInfo` 结构体；第41-45行 `TilingData` 封装）:
```cpp
struct MoeDistributeDispatchSetupInfo {
    uint32_t epWorldSize;        // EP域大小
    uint32_t epRankId;           // 本卡Rank ID
    uint32_t expertShardType;    // 专家分片类型
    uint32_t sharedExpertNum;    // 共享专家数量
    uint32_t sharedExpertRankNum;// 共享专家卡数
    uint32_t moeExpertNum;       // MoE专家总数
    uint32_t moeExpertPerRankNum;// 每卡MoE专家数
    uint32_t quantMode;          // 量化模式
    uint32_t globalBs;           // 全局BS
    uint32_t bs, k, h;           // 问题规模
    uint32_t aivNum;             // AIV核数
    uint32_t sdmaUsedStreamPerCore; // 每核SDMA流数
    bool isQuant, isActiveMask;  // 开关
    uint64_t totalUbSize;        // UB总大小
    uint64_t totalWinSize;       // Window总大小
};
```

**dispatch_teardown Kernel入口** (`moe_distribute_dispatch_teardown.cpp` 第21-27行）: 类似结构，额外有 `expertTokenNumsType` 字段。

### 4.3 moe_distribute_combine_setup Kernel (arch35)

**入口** (`moe_distribute_combine_setup.cpp` 第31-51行）:
```cpp
template <bool HasTp>
__global__ __aicore__ void moe_distribute_combine_setup(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine,
    GM_ADDR quantExpandX, GM_ADDR commCmdInfoOut,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);  // 纯Vector核
    // ...
    MoeDistributeCombineSetup<DTYPE_EXPAND_X, int32_t> op;
    op.Init(expandX, expertIds, assistInfoForCombine, quantExpandX,
            commCmdInfoOut, workspaceGM, &pipe, &tilingData,
            mc2InitTiling, mc2CcTiling);
    op.Process();
}
```

**核心类** (`moe_distribute_combine_setup_arch35.h` 第48-485行 `MoeDistributeCombineSetup` 类）:
```
MoeDistributeCombineSetup<ExpandXType, ExpandIdxType>

常量:
  STATE_OFFSET = 512          // 状态空间偏移
  STATE_SIZE = 1MB             // 状态区总大小
  WIN_STATE_OFFSET = 350KB    // Window中状态区偏移
  STATE_WIN_OFFSET = 950KB    // 状态区中Window偏移
  STATE_COUNT_THRESHOLD = 512 // 状态数阈值(决定状态间距)

Init():
  ├── 读取HCCL上下文: hcclContext_ = GetHcclContext<HCCL_GROUP_ID_0>()
  ├── 双缓冲区状态切换: dataState_ ← Win状态区的0/1标识, 然后翻转
  ├── 计算地址偏移:
  │   ├── winDataSizeOffset_ = dataState × (totalWinSize / 2)   ← 双缓冲数据区
  │   ├── stateOffset_ = (moeSendNum > 512) ? 256 : 512        ← 状态间距
  │   ├── epDataOffsetOnWin_ = rankId × moeExpertPerRank × expertPerSize  ← 数据区偏移
  │   └── epStateOffsetOnWin_ = rankId × stateOffset_           ← 状态区偏移
  ├── OOM检查: OOMCheckAddrRange()
  └── SplitCoreCal(): 按epWorldSize分核 (coreIdx + epRankId偏移避免冲突)

Process():
  ├── BuffInit()                  ← UB队列初始化
  │   ├── assistInfoQueue_        ← moeSendNum × sizeof(int32)
  │   ├── URMA通信buf: sqInfo(80B), cqInfo(48B), doorBell, cqe(256×32B)
  │   ├── templateSqeBuf_         ← 96B SQE模板
  │   ├── tokenSqeBuf_            ← 96B × moeExpertPerRankNum
  │   └── expertTokenTmpQueue_    ← expertPerSizeOnWin_ (本卡回环用)
  ├── InitCqeStatus()             ← 首次通信时初始化CQE为0xFF
  ├── GenerateCommWriteWithNotifySQE()  ← 生成URMA WQE模板
  └── Communication()             ← 核心通信流程
      ├── AssistInfoLocalCopy()   ← 从GM拷贝assistInfoForCombine到UB
      ├── CurRankComm()           ← 本卡回环: GM → UB → WinIn (DataCopy)
      │   └── 写状态flag 0x3f800000 (1.0f) 到WinOut状态区
      └── 对每个远端Rank循环:
          ├── UrmaInit()          ← 加载SQ/CQ, 更新PI/CI, PollCQ, 更新WQE模板
          ├── SendPerExpert()     ← 按Expert组装SQE:
          │   ├── 读取assistInfo获取每Expert的preCount和curTokenNum
          │   ├── srcAddr = expandXGM + preCount × H_size
          │   ├── dstAddr = GetWinAddrByRankId(epIdx, expertIdx) + epDataOffsetOnWin_
          │   └── 组装 WriteWithNotifySQE
          ├── 最后一个SQE设置flag=0x3f800000保证强保序
          ├── PutCommNotifySQE()   ← 下发SQE到硬件
          ├── SendJFSDoorBell()    ← 敲门铃触发发送
          └── UpdatePICI()         ← 更新生产者/消费者索引
```

**URMA通信机制**:
```
  ┌─────────────────────┐        URMA WriteWithNotify        ┌─────────────────────┐
  │   本卡 (Rank R)      │ ──────────────────────────────→   │   远端卡 (Rank P)    │
  │                     │                                     │                     │
  │  expandX → SQE组装  │    数据: expandX → WinIn[P]        │  WinIn[R] ← 数据    │
  │  DoorBell触发       │    通知: flag → WinOut[P]状态区     │  WinOut[R] ← flag   │
  └─────────────────────┘                                     └─────────────────────┘

SQE (Send Queue Element) 结构:
  - dataSrcAddr: 源地址 (expandX中对应token)
  - dataDstAddr: 目的地址 (远端WinIn)
  - length: 数据长度
  - notifyAddr: 通知地址 (远端WinOut状态区)
  - notifyData: 通知值 (0x3f800000 = 1.0f)
  - cqe: 是否生成完成事件
```

### 4.4 moe_distribute_combine_teardown Kernel (arch35)

**核心类** (`moe_distribute_combine_teardown_arch35.h` 第40-396行 `MoeDistributeCombineTeardown` 类；第241-296行 `WaitDispatch`）:

```
MoeDistributeCombineTeardown<ExpandXType, ExpandIdxType>

Init():
  ├── 读取HCCL上下文 + 双缓冲状态切换 (与setup类似)
  ├── 计算地址:
  │   ├── expertPerSizeOnWin_ = axisMaxBS × H_type_size  ← 每Expert在Win中的空间
  │   ├── sharedExpertDataSizeOffset_ = moeExpertPerRank × sharedExpertRank × expertPerSize
  │   └── winDataSizeOffset_ = dataState × (totalWinSize / 2)
  ├── SplitCoreCal()    ← Rank维分核
  └── TokenSplitCoreCal() ← Token维分核 (beginIndex_, endIndex_)

Process():
  ├── AlltoAllBuffInit()    ← UB队列初始化
  │   ├── indexCountsQueue_  ← BS×K × sizeof(int32)
  │   ├── expertIdsQueue_    ← BS×K × sizeof(int32)
  │   ├── expandScalesQueue_ ← BS×K × sizeof(float)
  │   ├── moeSumQueue_       ← H_type_size (单个token)
  │   ├── rowTmpFloatBuf_    ← H_float_size
  │   ├── sumFloatBuf_       ← H_float_size (累加缓冲)
  │   ├── statusBuf_         ← sendRankNum × 32B (状态等待)
  │   └── gatherMaskOutBuf_  ← epWorldSize × sizeof(float)
  ├── WaitDispatch()         ← 等待dispatch完成
  │   ├── 循环polling本卡WinOut状态区
  │   ├── DataCopy状态到UB → GatherMask → Sum
  │   ├── 等待 sumOfFlag ≈ sendRankNum (所有远端均已写入)
  │   └── 清除状态区 (写0)
  ├── SyncAll()              ← 全核同步
  ├── CopyIn()               ← 从GM拷贝expandIdx/expertIds/expertScales到UB
  └── LocalWindowCopy()      ← 核心聚合逻辑
      ├── 对每个Token (beginIndex_ ~ endIndex_):
      │   ├── 清零 sumFloatBuf_
      │   └── 对每个Expert (K个):
      │       ├── 读取 indexCount, moeExpert, scaleVal
      │       ├── 计算Win地址: epWindowGM_ + sharedExpertOffset + expertIdx×expertPerSize + indexCount×H
      │       ├── DataCopy Win → UB (1 token)
      │       ├── Cast BF16/FP16 → Float
      │       ├── Muls(× scaleVal)
      │       └── Add → sumFloatBuf_ (累加)
      ├── Cast Float → BF16/FP16
      └── DataCopy → expandOutGlobal_ (写出)
```

**WaitDispatch同步机制** (`moe_distribute_combine_teardown_arch35.h` 第241-296行）:
```
                   状态区布局 (每个Rank):
  ┌───────────────────────────────────────────────────────────┐
  │ Rank0状态 │ Rank1状态 │ ... │ RankN状态 │ (stride=stateOffset) │
  └───────────────────────────────────────────────────────────┘

  WaitDispatch():
  1. DataCopy: 将 startRankId ~ endRankId 的状态从GM拷到UB
  2. GatherMask: 过滤有效状态
  3. Sum: 累加所有有效状态值
  4. 等待 sum ≈ sendRankNum (所有远端卡已写入1.0f)
  5. 清除状态区 (写0, 为下一轮准备)
```

---

## 5. Host-Kernel 接口

### 5.1 Tiling数据传递

所有算子统一使用 `Mc2InitTiling` + `Mc2CcTiling` + `{Op}Info` 三段式Tiling数据:

```
┌──────────────────────────────────────────────────┐
│ Mc2InitTiling    │ Mc2CcTiling │ {Op}Info         │
│ (HCCL初始化)     │ (通信配置)  │ (算子特有参数)     │
└──────────────────────────────────────────────────┘
                         ↓ tilingGM
              Kernel通过 GET_TILING_DATA_WITH_STRUCT 解析
```

### 5.2 HCCL上下文传递

Combine Setup/Teardown 通过 `GetHcclContext<HCCL_GROUP_ID_0>()` 获取 `HcclCombinOpParam`:
```cpp
struct HcclCombinOpParam {
    GM_ADDR* windowsIn[epWorldSize];    // 各Rank的Window输入区地址
    GM_ADDR* windowsOut[epWorldSize];   // 各Rank的Window输出区(状态区)地址
    // URMA相关: SQ/CQ基址, PI/CI状态
};
```

### 5.3 Window共享内存布局

```
Window地址空间 (每个Rank):
┌─────────────────────────────────────────────────────────────┐
│ 数据区 (双缓冲)                                              │
│ ┌───────────────────────────┬───────────────────────────┐  │
│ │ Buffer 0 (dataState=0)   │ Buffer 1 (dataState=1)    │  │
│ │ 各Expert的Token数据        │ 各Expert的Token数据         │  │
│ └───────────────────────────┴───────────────────────────┘  │
│                                                             │
│ 状态区 (1MB)                                                │
│ ┌─────────────────────────────────────────────────────────┐│
│ │ Dispatch状态 │ Combine状态(offset=0)                      ││
│ │ 每个远端Rank一个flag │ WinState区域 (350KB)                ││
│ └─────────────────────────────────────────────────────────┘│
│ Win状态标识 (950KB偏移处, 512B/core)                         │
│ ┌─────────────────────────────────────────────────────────┐│
│ │ core0: dataState │ core1: dataState │ ...               ││
│ └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 数据流

### 6.1 Dispatch阶段完整数据流

```
输入: x[BS, H], expert_ids[BS, K]
           │
    ┌──────┴──────┐
    │ dispatch_   │
    │   setup     │
    └──────┬──────┘
           │
    y[BS*(K+sharedExpertNum), tokenMsgSize]  → HCCL AllToAllV →
           │
    ┌──────┴──────┐
    │ dispatch_   │
    │  teardown   │
    └──────┬──────┘
           │
    expand_x[A, H] (按Expert分组)
    assist_info_for_combine[A*128]
    expert_token_nums[localExpertNum]
```

### 6.2 Combine阶段完整数据流

```
输入: expand_x[A, H] (Expert计算后)
           │
    ┌──────┴──────┐
    │  combine_   │  1. AssistInfoLocalCopy: 读assistInfoForCombine到UB
    │   setup     │  2. CurRankComm: 本卡数据 GM→UB→WinIn
    │  (arch35)   │  3. Communication: 对每个远端Rank:
    └──────┬──────┘     a. UrmaInit: 加载SQ/CQ
           │            b. SendPerExpert: 按Expert组装SQE并下发
           │            c. DoorBell触发发送
    ┌──────┴──────┐
    │  combine_   │  1. WaitDispatch: polling状态区等待所有Rank发送完成
    │  teardown   │  2. CopyIn: 读expandIdx/expertIds/expertScales到UB
    │  (arch35)   │  3. LocalWindowCopy: 对每个Token:
    └──────┬──────┘     a. 清零sumBuf
           │            b. 对K个Expert: Win→UB→Cast→Muls(scale)→Add
           │            c. Cast回原类型→写出
           │
    x_out[BS, H] (最终Combine结果)
```

### 6.3 assist_info_for_combine 编码

```
assist_info_for_combine [A * 128]:
  - 前128个元素: 第0个Expert在所有Rank上的累计Token数
    [rank0_count, rank1_count, ..., rankN_count, padding...]
  - 第128~255个元素: 第1个Expert在所有Rank上的累计Token数
  - ...
  - 用于combine_setup确定每个Expert发给每个Rank多少token
```

---

## 7. 性能设计

### 7.1 双缓冲零拷贝通信

Combine Setup/Teardown 使用Window共享内存进行跨Rank通信，避免数据拷贝:
- `dataState_` 标识当前使用Buffer 0还是Buffer 1
- 每次通信自动切换缓冲区，实现流水线双缓冲
- 状态区 `STATE_WIN_OFFSET` 保存双缓冲标识 (per-core, 512B)

### 7.2 URMA硬件加速

Combine Setup使用URMA (User-space Remote Memory Access) 硬件通信:
- `WriteWithNotifySQE`: 单条SQE同时完成数据写入和通知
- 最后一个SQE设置 `notifyData=0x3f800000` 实现强保序
- `JFSDoorBell` 触发硬件发送，减少CPU参与

### 7.3 多核负载均衡

**Rank维分核** (`SplitCoreCal`):
```
sendRankNum = epWorldSize / aivNum
余数部分由前几个核多承担一个Rank
startRankId按(coreIdx + epRankId) % aivNum偏移，避免不同卡上的核争抢同一Rank的SQ/CQ
```

**Token维分核** (`TokenSplitCoreCal`, combine_teardown):
```
tokenPerAiv = BS / aivNum
余数部分由前几个核多处理一个Token
```

### 7.4 状态轮询优化

`WaitDispatch` 使用向量化Polling替代标量轮询:
```
DataCopy(状态) → GatherMask(过滤) → Sum(累加) → 标量比较
一次处理 sendRankNum 个状态，而非逐个读取
状态间距根据 moeSendNum 动态调整: >512时用256B间距，否则512B
```

### 7.5 Batch调度模式

所有setup/teardown算子设置 `SetScheduleMode(1)` (batch mode):
- 所有AIV核同时启动，无需等待前序核完成
- 适合通信类算子的全核并行特性

### 7.6 Tiling参数范围约束

| 参数 | 范围 | 说明 |
|------|------|------|
| epWorldSize | [2, 384] | EP域大小 |
| moeExpertNum | (0, 512] | MoE专家总数 |
| H | [1024, 8192] | 隐藏层维度 |
| BS | (0, 512] | BatchSize |
| K | (0, min(16, moeExpertNum)] | TopK |
| sharedExpertNum | [0, 4] | 共享专家数 |
| quantMode | {0, 2} | 0=不量化, 2=动态量化 |

---

## 8. 依赖关系

```
moe_update_expert:
  ├── common/op_kernel/  ← 无特殊依赖，独立实现

moe_distribute_dispatch_setup:
  ├── common/op_host/op_tiling/moe_tiling_base.h   ← MOE Tiling基类
  └── common/op_kernel/mc2_kernel_utils.h           ← MC2 Kernel工具

moe_distribute_dispatch_teardown:
  ├── common/op_host/op_tiling/moe_tiling_base.h
  └── common/op_kernel/mc2_kernel_utils.h

moe_distribute_combine_setup:
  ├── common/op_host/op_tiling/moe_tiling_base.h
  ├── common/op_kernel/mc2_kernel_utils.h           ← URMA通信原语
  ├── common/op_kernel/moe_distribute_base.h        ← HcclCombinOpParam定义
  ├── kernel_tiling/kernel_tiling.h                  ← Mc2InitTiling/Mc2CcTiling
  └── basic_api/kernel_basic_intf.h                  ← AscendC基础接口

moe_distribute_combine_teardown:
  ├── moe_distribute_combine_setup/op_kernel/moe_distribute_base.h  ← 共享HCCL上下文定义
  ├── common/op_kernel/mc2_kernel_utils.h
  └── kernel_tiling/kernel_tiling.h

跨算子依赖:
  dispatch_teardown的输出 assist_info_for_combine → combine_setup的输入
  dispatch_setup的输出 comm_cmd_info → dispatch_teardown的输入
  combine_setup的输出 comm_cmd_info → combine_teardown的输入
```

### Setup/Teardown 对比 V1 Dispatch/Combine

| 特性 | V1 Dispatch/Combine | Setup/Teardown |
|------|---------------------|----------------|
| 粒度 | 单一大算子 | 4阶段分离 (可插入自定义计算) |
| 通信 | Window P2P + Flag同步 | URMA WriteWithNotify / HCCL AllToAllV |
| 量化 | 内嵌 (静态/动态) | 量化路径 (BF16/FP16 → INT8) |
| 同步 | Flag-based轮询 | URMA CQE + Sum-based状态轮询 |
| 双缓冲 | 无 | dataState_自动切换 (setup写/teardown读) |
| 平台 | 910B/910_93/950 | 仅910_93/950 |
| 灵活性 | 单次调用完成全流程 | 可在dispatch和combine间插入Expert计算 |

---

*分析完成时间: 2026-04-20，修订时间: 2026-04-21（补充行号引用、修正章节编号为x.y格式）*
