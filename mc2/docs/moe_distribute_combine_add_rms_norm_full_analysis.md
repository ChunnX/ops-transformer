# moe_distribute_combine_add_rms_norm 算子全面分析报告

## 1. 算子概述

`moe_distribute_combine_add_rms_norm` 是 `moe_distribute_combine_v2` 的 **AddRmsNorm 融合变体**。在标准 Combine (Expert 结果 AlltoAll 收集 + 加权聚合) 的基础上，融合残差加法和 RMS 归一化，将三阶段操作合并为单次算子调用。

### 1.1 算子语义

```
输入:  expand_x [A*epWorldSize, H]   (Expert计算结果)
       expert_scales [BS, K]           (Top-K门控权重)
       residual_x [BS, 1, H]           (残差输入)
       gamma [H]                       (RMSNorm缩放因子)
       norm_eps (float, 默认1e-6)
计算:
  combined_result = MOE_Combine(expand_x, expert_ids, assist_info, ep_send_counts, expert_scales)
                   (AlltoAll收集 + scale加权聚合, 结果 [BS, H])
  y   = combined_result + residual_x                     (残差加法)
  rstdOut = 1 / sqrt(mean(y^2) + epsilon)               (RMS倒数)
  x   = y * rstdOut * gamma                              (RMSNorm输出)
输出:  y [BS, 1, H], rstdOut [BS, 1, 1], x [BS, 1, H]
```

### 1.2 设计模式

**纯薄封装 (Thin Wrapper)** — 所有层完全委托给 `moe_distribute_combine_v2`:

| 层 | 委托方式 | 委托目标 |
|----|---------|---------|
| Kernel | 复用类实例 | `MoeDistributeCombineV2` (模板参数 `HasAddRmsNorm=true`) |
| Tiling | 一行调用 | `MoeDistributeCombineV2TilingFunc(context)` |
| ACLNN V1 | 补 nullptr | Base -> `aclnnInnerMoeDistributeCombineAddRmsNorm` |
| ACLNN V2 | 直接传递 | Base -> `aclnnInnerMoeDistributeCombineAddRmsNorm` |

ARN 计算逻辑并非独立实现，而是已内嵌于 `MoeDistributeCombineV2` 类中，通过编译期模板参数 `HasAddRmsNorm` 控制是否启用。本算子的内核入口仅将 `HasAddRmsNorm` 硬编码为 `true` 并传入 `residualX`/`gamma` 地址。

---

## 2. 目录结构与文件职责

```
moe_distribute_combine_add_rms_norm/
├── op_kernel/
│   └── moe_distribute_combine_add_rms_norm.cpp         :78  内核入口(委托V2+HasAddRmsNorm=true)
├── op_host/
│   ├── moe_distribute_combine_add_rms_norm_def.cpp      :193 算子定义(19输入/3输出/17属性)
│   ├── moe_distribute_combine_add_rms_norm_infershape.cpp:100 Shape推导+DataType推导
│   └── op_tiling/
│       └── moe_distribute_combine_add_rms_norm_tiling.cpp:62  Tiling(一行委托V2)
├── op_api/
│   ├── aclnn_moe_distribute_combine_add_rms_norm.h       :101 V1 ACLNN API声明(14必选输入)
│   ├── aclnn_moe_distribute_combine_add_rms_norm.cpp     :63  V1实现(补nullptr调用Base)
│   ├── aclnn_moe_distribute_combine_add_rms_norm_v2.h    :116 V2 ACLNN API声明(全部输入)
│   ├── aclnn_moe_distribute_combine_add_rms_norm_v2.cpp  :64  V2实现(直调Base)
│   ├── moe_distribute_combine_add_rms_norm_base.h         :67  Base声明(参数校验+外部链接)
│   └── moe_distribute_combine_add_rms_norm_base.cpp       :149 Base实现(校验+Inner调用)
├── op_graph/
│   ├── moe_distribute_combine_add_rms_norm_proto.h       :114 图模式算子原型(REG_OP)
│   └── moe_distribute_combine_add_rms_norm_gen_task.cpp       图模式gen_task
└── tests/
    ├── ut/op_api/test_aclnn_moe_distribute_combine_add_rms_norm.cpp
    ├── ut/op_host/test_moe_distribute_combine_add_rms_norm_infershape.cpp
    ├── ut/op_host/arch32/test_moe_distribute_combine_add_rms_norm_tiling.cpp
    └── ut/op_kernel/test_moe_distribute_combine_add_rms_norm.cpp
```

---

## 3. Host 侧分析

### 3.1 算子定义

**文件**: `moe_distribute_combine_add_rms_norm_def.cpp`（第19行 `OpDef` 子类定义，第21-188行构造函数）

**输入张量** (19个: 7必选 + 12可选，第22-135行声明):

| 索引 | 名称 | 必选/可选 | 数据类型 | 维度 | 说明 |
|------|------|----------|---------|------|------|
| 0 | expand_x | REQUIRED | BF16 | 2D [A*epWorldSize, H] | Expert计算结果 |
| 1 | expert_ids | REQUIRED | INT32 | 2D [BS, K] | 每个token的Top-K专家索引 |
| 2 | assist_info_for_combine | REQUIRED | INT32 | 1D [A*128] | Combine辅助信息(Dispatch输出) |
| 3 | ep_send_counts | REQUIRED | INT32 | 1D | EP域发送计数(Dispatch输出) |
| 4 | expert_scales | REQUIRED | FLOAT32 | 2D [BS, K] | Expert门控权重 |
| 5 | **residual_x** | REQUIRED | BF16 | 3D [BS, 1, H] | 残差输入 (ARN独有) |
| 6 | **gamma** | REQUIRED | BF16 | 1D [H] | RMSNorm缩放因子 (ARN独有) |
| 7 | tp_send_counts | OPTIONAL | INT32 | 1D | TP域发送计数 |
| 8 | x_active_mask | OPTIONAL | BOOL | 1D | Token活跃掩码 |
| 9 | activation_scale | OPTIONAL | FLOAT32 | 1D | 量化激活缩放(预留) |
| 10 | weight_scale | OPTIONAL | FLOAT32 | 2D | 量化权重缩放(预留) |
| 11 | group_list | OPTIONAL | INT64 | 1D | GMM分组列表(预留) |
| 12 | expand_scales | OPTIONAL | FLOAT32 | 1D | 展开缩放因子 |
| 13 | shared_expert_x | OPTIONAL | BF16 | 2D | 共享专家结果 |
| 14 | elastic_info | OPTIONAL | INT32 | 1D | 弹性信息 |
| 15 | ori_x | OPTIONAL | BF16 | 2D | 原始输入(常量/拷贝专家用) |
| 16 | const_expert_alpha_1 | OPTIONAL | BF16 | 1D | 常量专家alpha1 |
| 17 | const_expert_alpha_2 | OPTIONAL | BF16 | 1D | 常量专家alpha2 |
| 18 | const_expert_v | OPTIONAL | BF16 | 1D | 常量专家向量v |

**输出张量** (3个，第137-151行声明):

| 索引 | 名称 | 数据类型 | 维度 | 说明 |
|------|------|---------|------|------|
| 0 | y | BF16 | 3D [BS, 1, H] | 残差加法结果: combine_output + residual_x |
| 1 | rstdOut | FLOAT32 | 3D [BS, 1, 1] | RMS倒数: 1/sqrt(mean(y^2) + eps) |
| 2 | x | BF16 | 3D [BS, 1, H] | RMSNorm最终输出: y * rstd * gamma |

**属性** (17个: 4必选 + 13可选，第153-171行声明):

| 序号 | 名称 | 类型 | 必选/可选 | 默认值 | 说明 |
|------|------|------|----------|--------|------|
| 0 | group_ep | String | REQUIRED | - | EP通信域名 |
| 1 | ep_world_size | Int | REQUIRED | - | EP域大小 (8/16/32/64/128/144/256/288) |
| 2 | ep_rank_id | Int | REQUIRED | - | EP域本卡ID [0, epWorldSize) |
| 3 | moe_expert_num | Int | REQUIRED | - | MOE专家数量 [1, 512] |
| 4 | group_tp | String | OPTIONAL | "" | TP通信域名 |
| 5 | tp_world_size | Int | OPTIONAL | 0 | TP域大小 [0, 2] |
| 6 | tp_rank_id | Int | OPTIONAL | 0 | TP域本卡ID |
| 7 | expert_shard_type | Int | OPTIONAL | 0 | 专家共享类型(仅支持0) |
| 8 | shared_expert_num | Int | OPTIONAL | 0 | 共享专家数量 [0, 1] |
| 9 | shared_expert_rank_num | Int | OPTIONAL | 0 | 共享专家卡数 |
| 10 | global_bs | Int | OPTIONAL | 0 | 全局批大小 |
| 11 | out_dtype | Int | OPTIONAL | 0 | 输出数据类型(预留) |
| 12 | comm_quant_mode | Int | OPTIONAL | 0 | 通信量化模式(预留) |
| 13 | group_list_type | Int | OPTIONAL | 0 | groupList格式(预留) |
| 14 | comm_alg | String | OPTIONAL | "" | 通信算法(预留) |
| 15 | **norm_eps** | Float | OPTIONAL | 1e-6 | RMSNorm的epsilon (ARN独有) |
| 16 | zero_expert_num | Int | OPTIONAL | 0 | 零专家数量 |
| 17 | copy_expert_num | Int | OPTIONAL | 0 | 拷贝专家数量 |
| 18 | const_expert_num | Int | OPTIONAL | 0 | 常量专家数量 |

**平台支持**: `ascend950`, `ascend910_93` (不支持 `ascend910b`, 见 base.cpp 第77-79行校验)

**AICore配置** (def.cpp 第173-183行):
```
DynamicCompileStaticFlag = true
DynamicFormatFlag = true
DynamicRankSupportFlag = true
DynamicShapeSupportFlag = true
PrecisionReduceFlag = true
jitCompile = static_true
multiKernelSupportDynamicGraph = multi_kernel
MC2().HcclGroup({"group_ep", "group_tp"})
```

### 3.2 Shape 推导

**文件**: `moe_distribute_combine_add_rms_norm_infershape.cpp`（第51-99行）

**核心逻辑** (infershape.cpp 第51-82行): 以 `residual_x` (输入索引5) 的 shape 为基准推导三个输出:

```cpp
// 行57: 以residual_x为shape基准
const gert::Shape* residualX = context->GetInputShape(IDX_FIVE);  // IDX_FIVE=5
size_t dimNum = residualX->GetDimNum();

// 行67-70: y 输出shape
x1OutShape->SetDimNum(IDX_THREE);                       // 3维
x1OutShape->SetDim(IDX_ZERO, residualX->GetDim(IDX_ZERO));   // dim0 = BS
x1OutShape->SetDim(IDX_ONE, IDX_ONE);                         // dim1 = 1
x1OutShape->SetDim(IDX_TWO, residualX->GetDim(dimNum - 1));   // dim2 = H

// 行71-74: rstdOut 输出shape
rstdOutShape->SetDimNum(IDX_THREE);
rstdOutShape->SetDim(IDX_ZERO, residualX->GetDim(IDX_ZERO));  // dim0 = BS
rstdOutShape->SetDim(IDX_ONE, IDX_ONE);                        // dim1 = 1
rstdOutShape->SetDim(IDX_TWO, IDX_ONE);                        // dim2 = 1

// 行75-78: x 输出shape (同y)
x2OutShape->SetDimNum(IDX_THREE);
x2OutShape->SetDim(IDX_ZERO, residualX->GetDim(IDX_ZERO));
x2OutShape->SetDim(IDX_ONE, IDX_ONE);
x2OutShape->SetDim(IDX_TWO, residualX->GetDim(dimNum - 1));
```

**Shape 推导汇总**:

```
residual_x: [BS, 1, H]  (输入)
     ↓
y:       [BS, 1, H]     (输出0, 同residual_x)
rstdOut: [BS, 1, 1]     (输出1)
x:       [BS, 1, H]     (输出2, 同residual_x)
```

**DataType 推导** (infershape.cpp 第84-95行): 固定类型，不依赖输入。

```
y  → DT_BF16
rstdOut → DT_FLOAT
x  → DT_BF16
```

**前置检查** (infershape.cpp 第29-49行):
- 验证 context 非空
- 验证前 3 个输出 shape 指针非空
- 验证前 8 个输入 shape 指针非空

### 3.3 Tiling 策略

**文件**: `moe_distribute_combine_add_rms_norm_tiling.cpp`（第46-61行）

**完全委托** — Tiling 函数体仅一行调用:

```cpp
// 行46-49: Tiling函数
static ge::graphStatus MoeDistributeCombineAddRmsNormTilingFunc(gert::TilingContext* context)
{
    ge::graphStatus ret = optiling::MoeDistributeCombineV2TilingFunc(context);
    return ret;
}
```

**TilingParse** (第52-57行): 空实现，不做任何额外解析:
```cpp
struct MoeDistributeCombineAddRmsNormCompileInfo {};
ge::graphStatus TilingParseForMoeDistributeCombineAddRmsNorm(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}
```

**TilingKey**: 与 `moe_distribute_combine_v2` 完全共用同一套 TilingKey 枚举 (见 `moe_distribute_combine_v2_tiling_key.h`)。TilingKey 编码由4个模板参数组成:
- `TILINGKEY_TP`: 是否有TP域 (0/1)
- `TILINGKEY_QUANT_MODE`: 量化模式 (NO_QUANT=0 / INT8_QUANT=1)
- `TILINGKEY_LAYERED_MODE`: 分层模式 (MTE=0 / AICPU=1 / HIERARCHY=2)
- `ARCH_TAG`: 架构标签 (A2=0 / A3=1 / A5=2)

### 3.4 ACLNN API

#### 3.4.1 V1 API

**文件**: `aclnn_moe_distribute_combine_add_rms_norm.cpp`（第27-59行）

V1 API 有 14 个必选输入 (不含弹性/常量/拷贝专家相关参数), 缺少的 5 个可选参数以 `nullptr` 填充:

```cpp
// 行43-49: V1调用Base，补充nullptr给V2新增参数
aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSizeBase(
    expandX, expertIds, assistInfoForCombine, epSendCounts, expertScales, residualX, gamma,
    tpSendCountsOptional, xActiveMaskOptional, activationScaleOptional, weightScaleOptional,
    groupListOptional, expandScalesOptional, sharedExpertXOptional,
    nullptr,    // elasticInfoOptional
    nullptr,    // oriXOptional
    nullptr,    // constExpertAlpha1Optional
    nullptr,    // constExpertAlpha2Optional
    nullptr,    // constExpertVOptional
    groupEp, epWorldSize, epRankId, moeExpertNum, groupTp, tpWorldSize, tpRankId,
    expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs, outDtype,
    commQuantMode, groupListType, commAlg, normEps,
    0, 0, 0,    // zeroExpertNum=0, copyExpertNum=0, constExpertNum=0
    yOut, rstdOut, xOut, workspaceSize, executor);
```

执行阶段 (第53-58行) 直接委托 Base:
```cpp
aclnnStatus aclnnMoeDistributeCombineAddRmsNorm(void *workspace, uint64_t workspaceSize,
                                                  aclOpExecutor *executor, aclrtStream stream)
{
    return aclnnMoeDistributeCombineAddRmsNormBase(workspace, workspaceSize, executor, stream);
}
```

#### 3.4.2 V2 API

**文件**: `aclnn_moe_distribute_combine_add_rms_norm_v2.cpp`（第28-60行）

V2 API 包含全部 19 个输入和 3 个专家数量属性, 直接传递给 Base, 无参数填充:

```cpp
// 行43-50: V2直调Base，所有参数原样传递
aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSizeBase(
    expandX, expertIds, assistInfoForCombine, epSendCounts, expertScales, residualX, gamma,
    tpSendCountsOptional, xActiveMaskOptional, activationScaleOptional, weightScaleOptional,
    groupListOptional, expandScalesOptional, sharedExpertXOptional,
    elasticInfoOptional, oriXOptional, constExpertAlpha1Optional,
    constExpertAlpha2Optional, constExpertVOptional,
    groupEp, ..., normEps, zeroExpertNum, copyExpertNum, constExpertNum,
    yOut, rstdOut, xOut, workspaceSize, executor);
```

#### 3.4.3 Base 层

**文件**: `moe_distribute_combine_add_rms_norm_base.cpp`（第92-148行）

Base 层职责:
1. **平台检查** (第110行): 检测是否为 `ascend910b`，若为 910B 则拒绝 (第77-79行)
2. **参数校验** (第111-113行): `CombineArnCheckParams()` 做空指针、通信域名长度检查
3. **Workspace 计算** (第119-126行): 调用外部链接函数 `aclnnInnerMoeDistributeCombineAddRmsNormGetWorkspaceSize()`
4. **执行** (第131-145行): 设置 HCCL Server Type 为 MTE，调用 `aclnnInnerMoeDistributeCombineAddRmsNorm()`

```cpp
// 行139-142: 执行阶段
if (NnopbaseSetHcclServerType) {
    NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
}
ret = aclnnInnerMoeDistributeCombineAddRmsNorm(workspace, workspaceSize, executor, stream);
```

---

## 4. Kernel 侧分析

### 4.1 内核入口

**文件**: `moe_distribute_combine_add_rms_norm.cpp`（第65-78行内核入口，第38-51行 `ExecMoeDistributeCombineAddRmsNorm` 内部执行函数）

内核入口是一个模板函数，与 `moe_distribute_combine_v2` 的入口结构一致，但核心区别在于:

1. **调用 `ExecMoeDistributeCombineAddRmsNorm`** (第38-51行) 而非 `ExecMoeDistributeCombineV2`
2. `HasAddRmsNorm` 模板参数硬编码为 `true` (第74行)
3. 额外传入 `residualX` 和 `gamma` 地址 (第47行)

```cpp
// 行65-69: 内核签名 (与V2对比，增加了residualX和gamma参数位置)
template<bool HasTp, uint8_t QuantMode, uint8_t LayeredMode, uint8_t ArchTag>
__global__ __aicore__ void moe_distribute_combine_add_rms_norm(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine, GM_ADDR epSendCount,
    GM_ADDR scales, GM_ADDR residualX, GM_ADDR gamma,    // <-- residual和gamma在scales之后
    GM_ADDR tpSendCount, GM_ADDR xActiveMask, ...)
```

```cpp
// 行38-51: 内部执行函数
template <CombineMC2TypeClass>
__aicore__ inline void ExecMoeDistributeCombineAddRmsNorm(
    GM_ADDR expandX, ..., GM_ADDR residualX, GM_ADDR gamma, ..., GM_ADDR YOut,
    GM_ADDR dynamicScaleOut, GM_ADDR XOut, GM_ADDR workspaceGM, GM_ADDR tilingGM, TPipe* pipePtr)
{
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineV2TilingData, tilingData, tilingGM);
    MoeDistributeCombineV2<CombineMC2TypeFunc> op;
    op.Init(
        nullptr, expandX, expertIds, assistInfoForCombine, epSendCount, tpSendCount,
        residualX, gamma,    // <-- 传入residual和gamma
        scales, xActiveMask, sharedExpertX, elasticInfo, oriX,
        constExpertAlpha1, constExpertAlpha2, constExpertV,
        nullptr, YOut, dynamicScaleOut, XOut,   // YOut=rstdOut映射, dynamicScaleOut在此为rstdOut
        workspaceGM, pipePtr, &tilingData);
    op.Process();
}
```

**模板参数展开** (第74行):
```cpp
ExecMoeDistributeCombineAddRmsNorm<
    DTYPE_EXPAND_X, DTYPE_X, int32_t,
    HasTp,                         // IsNeedReduceScatter
    QuantMode == TILINGKEY_INT8_QUANT,  // IsInt8Quant
    true                           // HasAddRmsNorm <-- 硬编码为true
>(...);
```

**TilingKey 说明** (第54-63行):
```
A3 tilingkey为3位十进制数:
  第1位(个位): 占位
  第2位(十位): 占位
  第3位(百位): 是否做TP域AllGather (0=不做, 1=做)
  第4位(千位): 占位
  第5位(万位): 占位
```

### 4.2 核心类: MoeDistributeCombineV2

**文件**: `moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2.h`（第50-297行）

`MoeDistributeCombineV2` 是一个编译期多态模板类，通过 6 个模板参数控制行为:

```cpp
// v2.h 第46-47行
#define CombineMC2TypeClass typename ExpandXType, typename XType, typename ExpandIdxType, \
                             bool IsNeedReduceScatter, bool IsInt8Quant, bool HasAddRmsNorm
#define CombineMC2TypeFunc ExpandXType, XType, ExpandIdxType, IsNeedReduceScatter, IsInt8Quant, HasAddRmsNorm
```

**HasAddRmsNorm 影响的行为**:

| 位置 | HasAddRmsNorm=false (普通Combine) | HasAddRmsNorm=true (ARN融合) |
|------|-----------------------------------|------------------------------|
| `InitInputAndOutput()` 第362-367行 | 不绑定 residualX/gamma/yOut/rstdOut | 绑定这4个GlobalTensor |
| `InitAttrs()` 第465-468行 | 不读取 armAvgFactor/epsilon | 从tiling读取 armAvgFactor_ 和 epsilon_ |
| `AlltoAllCommBuffInit()` 第705-709行 | 不分配 gammaBuf/reduceFp32Buf | 分配 gammaBuf_ 和 reduceFp32Buf_ |
| `LocalWindowCopy()` 第1401-1405行 | 不加载gamma | 从GM加载gamma到UB (gammaBuf_) |
| `LocalWindowCopy()` 第1446-1449行 | 不执行Add | 执行 `AddRmsNormAddCompute()` |
| `LocalWindowCopy()` 第1456-1460行 | 不执行RMSNorm | 执行 `AddRmsNormRmsNormCompute()` |

### 4.3 Process 流程

**文件**: `moe_distribute_combine_v2.h`（第1473-1485行 `Process` 函数）

```
Process():
  ├── [AIV] ReduceScatterTrans()           // TP域数据搬运(仅IsNeedReduceScatter)
  ├── BuffInit()                           // 分配UB缓冲区(分核+队列初始化)
  ├── SetWaitTpStatusAndDisPatch()         // TP同步+EP AlltoAll发送
  ├── PipeBarrier<PIPE_ALL>()             // 确保AlltoAll完成
  ├── AlltoAllBuffInitAndMaskCal()         // 接收缓冲区初始化+Mask计算
  └── LocalWindowCopy()                    // 核心处理循环
      ├── 加载gamma到UB (仅HasAddRmsNorm)
      ├── ExpertScaleCopy()                // 加载expert_scales
      ├── while (tokenNumCompleted != tokenPerAivNum):
      │   ├── for curIdx in [begin, end):
      │   │   ├── WaitDispatch(tokenIndex)     // 轮询HCCL状态等待数据到达
      │   │   ├── ProcessExpert(tokenIndex)     // 加权聚合各Expert结果
      │   │   │   ├── ProcessMoeExpert()        // 标准MOE专家: scale*expert -> 累加
      │   │   │   ├── ProcessCopyExpert()       // 拷贝专家: scale*oriX -> 累加
      │   │   │   └── ProcessConstantExpert()   // 常量专家: softmax(alpha)*x+alpha*v -> 累加
      │   │   ├── [HasAddRmsNorm] AddRmsNormAddCompute()    ← ARN阶段1: 残差加法
      │   │   ├── 结果搬出 (combine_output -> GM)
      │   │   └── [HasAddRmsNorm] AddRmsNormRmsNormCompute() ← ARN阶段2: RMSNorm
      │   └── tokenNumCompleted++
      └── [可选] 性能信息写回
```

### 4.4 AddRmsNormAddCompute — 残差加法

**文件**: `moe_distribute_combine_v2.h`（第1050-1063行）

计算 `y = combine_output + residual_x`:

```cpp
// 第1050-1063行
AddRmsNormAddCompute(tokenIndex, tokenOffset, numCol, ...):
    // 从GM加载residual_x
    x2 = tokenBuf_.Get<XType>();
    DataCopyPad(x2, residualXGM_[tokenIndex * axisH_ + tokenOffset], ...);
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    // BF16 -> FP32 转换
    Cast(x2TmpFloatLocal, x2, AscendC::RoundMode::CAST_NONE, numCol);
    PipeBarrier<PIPE_V>();

    // FP32加法: sumFloatBufLocal += x2TmpFloatLocal
    // sumFloatBufLocal 即 ProcessExpert 累加的 combine_output (FP32)
    AscendC::Add(addOutTmpFloatLocal, x1TmpFloatLocal, x2TmpFloatLocal, numCol);
    // addOutTmpFloatLocal == sumFloatBufLocal (原地)
```

### 4.5 AddRmsNormRmsNormCompute — RMS归一化

**文件**: `moe_distribute_combine_v2.h`（第1066-1112行）

计算 `rstd = 1/sqrt(mean(y^2) + eps)`, 然后 `x = y * rstd * gamma`:

```
AddRmsNormRmsNormCompute():
  ├── 计算 rstd:
  │   ├── Mul(sqx, xFp32, xFp32, numCol)           // y^2
  │   ├── Muls(sqx, sqx, armAvgFactor_, numCol)     // y^2 / H  (armAvgFactor = 1/H)
  │   ├── ReduceSum(sqx, sqx, reduceBufLocal, numCol) // Σ(y^2/H) → 标量
  │   ├── Adds(sqx, sqx, epsilon_, 1)                // + ε
  │   ├── Sqrt(sqx, sqx, 1)                          // sqrt(sum + ε)
  │   ├── Div(reduceBufLocal, ONE/sqx, 1)            // rstd = 1 / sqrt(...)
  │   └── DataCopyPad(rstdOutGlobal_[tokenIndex], reduceBufLocal)  // 写出rstd
  │
  └── 计算 x = y * rstd * gamma:
      ├── Muls(xFp32, xFp32, rstdValue, numCol)     // y * rstd
      ├── Cast(yLocal, xFp32, ROUND_RINT)            // FP32 -> BF16 (写y)
      ├── Cast(xFp32, yLocal, CAST_NONE)             // BF16 -> FP32 (再转回)
      ├── Cast(sqx, gammaLocal, CAST_NONE)           // gamma BF16 -> FP32
      ├── Mul(xFp32, xFp32, sqx, numCol)             // * gamma
      ├── Cast(yLocal, xFp32, ROUND_RINT)            // FP32 -> BF16 (写x)
      └── DataCopyPad(yOutGlobal_[...], yLocal)      // 写出x (复用yOut地址->实际是xOut)
```

**注意**: 代码中 `yOutGlobal_` 在 `HasAddRmsNorm` 模式下实际写入的是 RMSNorm 输出 `x`，而 `expandOutGlobal_` 写入的是 combine 原始输出 (在 AddRmsNormAddCompute 之前已写出)。

### 4.6 数据结构

#### 4.6.1 Tiling 数据结构

**文件**: `moe_distribute_combine_v2_tiling.h`（第22-61行）

```
MoeDistributeCombineV2TilingData:
├── Mc2InitTiling mc2InitTiling                // MC2初始化参数
├── Mc2CcTiling mc2CcTiling1                   // 通信参数1 (EP域)
├── Mc2CcTiling mc2CcTiling2                   // 通信参数2 (TP域)
└── MoeDistributeCombineV2Info moeDistributeCombineV2Info:
    ├── epWorldSize         // EP域大小
    ├── tpWorldSize         // TP域大小
    ├── epRankId            // EP域本卡ID
    ├── tpRankId            // TP域本卡ID
    ├── moeExpertNum        // 总MOE专家数
    ├── moeExpertPerRankNum // 每卡MOE专家数
    ├── zeroExpertNum       // 零专家数
    ├── copyExpertNum       // 拷贝专家数
    ├── constExpertNum      // 常量专家数
    ├── bs                  // 批大小
    ├── k                   // Top-K数
    ├── h                   // 隐藏维度
    ├── aivNum              // AIV核数
    ├── totalUbSize         // 总UB大小
    ├── totalWinSizeEp      // EP Window大小
    ├── totalWinSizeTp      // TP Window大小
    ├── armAvgFactor        // 1/H (ARN平均因子, HasAddRmsNorm时有效)
    ├── epsilon             // RMSNorm epsilon (HasAddRmsNorm时有效)
    ├── isTokenMask         // 是否有一维mask
    ├── isExpertMask        // 是否有二维mask
    ├── hasSharedExpertX    // 是否有共享专家
    ├── hasElasticInfo      // 是否有弹性信息
    ├── isMc2Context        // 是否使用MC2上下文
    └── bufferNum           // 缓冲区数量
```

#### 4.6.2 ARN 相关成员变量

**文件**: `moe_distribute_combine_v2.h`（第141-144行、第233-234行、第250-251行）

```
GlobalTensor<XType>  residualXGM_;     // 残差输入 GM 地址
GlobalTensor<XType>  gammaGM_;         // gamma GM 地址
GlobalTensor<XType>  yOutGlobal_;      // y输出 GM 地址 (残差加法结果)
GlobalTensor<float>  rstdOutGlobal_;   // rstd输出 GM 地址

TBuf<> gammaBuf_;                      // gamma UB缓冲区
TBuf<TPosition::VECCALC> reduceFp32Buf_; // ReduceSum辅助缓冲区 (128字节)

float armAvgFactor_;                   // 1/H, 用于mean(y^2)计算
float epsilon_;                        // RMSNorm epsilon
```

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据传递

```
Host (Tiling阶段):
  MoeDistributeCombineAddRmsNormTilingFunc(context)
  └── MoeDistributeCombineV2TilingFunc(context)  // 委托V2
      └── 生成 MoeDistributeCombineV2TilingData
          ├── mc2InitTiling
          ├── mc2CcTiling1, mc2CcTiling2
          └── moeDistributeCombineV2Info
              ├── armAvgFactor = 1.0 / H     (仅ARN场景)
              ├── epsilon = norm_eps          (仅ARN场景)
              └── ...其他参数

Kernel (执行阶段):
  REGISTER_TILING_DEFAULT(MoeDistributeCombineV2TilingData)
  GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineV2TilingData, tilingData, tilingGM)
  └── 读取 tilingData.moeDistributeCombineV2Info.armAvgFactor / epsilon
```

### 5.2 GM 地址映射

```
内核入口参数 → MoeDistributeCombineV2::Init() 参数映射:

  expandX               → expandXGM_        [输入] Expert计算结果
  expertIds             → expertIdsGM_       [输入] Expert索引
  assistInfoForCombine  → expandIdxGM_       [输入] Combine辅助信息
  epSendCount           → epSendCountGM_     [输入] EP发送计数
  tpSendCount           → tpSendCountGM_     [输入] TP发送计数
  residualX             → residualXGM_       [输入] 残差输入 (ARN专用)
  gamma                 → gammaGM_           [输入] RMSNorm缩放 (ARN专用)
  scales                → expertScalesGM_    [输入] Expert权重
  xActiveMask           → xActiveMaskGM_     [输入] Token掩码
  sharedExpertX         → sharedExpertXGM_   [输入] 共享专家结果
  elasticInfo           → elasticInfoGM_     [输入] 弹性信息
  oriX                  → oriXGM_            [输入] 原始输入
  constExpertAlpha1     → constExpertAlpha1GM_ [输入] 常量专家alpha1
  constExpertAlpha2     → constExpertAlpha2GM_ [输入] 常量专家alpha2
  constExpertV          → constExpertVGM_    [输入] 常量专家向量
  YOut                  → expandOutGlobal_   [输出] combine结果 (= x输出)
  dynamicScaleOut       → rstdOutGlobal_     [输出] rstd输出 (ARN专用)
  XOut                  → yOutGlobal_        [输出] RMSNorm结果 (ARN专用)
```

---

## 6. 数据流

### 6.1 Combine+ARN 端到端数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│ GM (Global Memory)                                                      │
│                                                                         │
│ ┌──────────────────┐  ┌──────────┐  ┌──────────┐                       │
│ │expand_x          │  │expert_   │  │ep_send_  │                       │
│ │[A*W,H] BF16      │  │ids [BS,K]│  │counts    │                       │
│ │(各rank的Expert   │  │INT32     │  │INT32     │                       │
│ │ 计算结果)        │  │          │  │          │                       │
│ └────────┬─────────┘  └────┬─────┘  └────┬─────┘                       │
│          │                  │              │                              │
│          ▼                  ▼              ▼                              │
│ ┌──────────────────────────────────────────────────────┐                │
│ │ Phase 1: AlltoAll Dispatch + Wait                     │                │
│ │ SetWaitTpStatusAndDisPatch():                         │                │
│ │   1. [TP域] 设置状态标志 + 等待对端TP状态               │                │
│ │   2. ExpertAlltoAllDispatchCopyAdd():                 │                │
│ │      读取expandIdx → 按rank错位发送到Window区           │                │
│ │      写状态标志通知数据已就绪                            │                │
│ └───────────────────────────┬──────────────────────────┘                │
│                             │                                           │
│ ┌───────────────────────────▼──────────────────────────┐                │
│ │ Phase 2: Wait + Aggregate (per token)                 │                │
│ │ LocalWindowCopy():                                    │                │
│ │   WaitDispatch(): 轮询Window状态区等待所有Expert到达    │                │
│ │   ProcessExpert():                                    │                │
│ │     for topk in 0..K:                                 │                │
│ │       load expert_result[BS,K+shared,H] from Window   │                │
│ │       sumFloatBuf += expert_scales[topk] * expert_res  │                │
│ │     [sharedExpertX]: sumFloatBuf += shared_expert_x    │                │
│ │     result = sumFloatBuf  [FP32, axisH_]              │                │
│ └───────────────────────────┬──────────────────────────┘                │
│                             │                                           │
│ ┌───────────────────────────▼──────────────────────────┐                │
│ │ Phase 3: AddRmsNorm (HasAddRmsNorm=true 时执行)       │                │
│ │                                                        │                │
│ │ ┌─── AddRmsNormAddCompute() ───────────────────────┐  │                │
│ │ │  residualXGM_ ──DataCopy──→ UB(x2)              │  │                │
│ │ │  sumFloatBuf (combine)   ──→ x1TmpFloatLocal    │  │                │
│ │ │  Cast(BF16→FP32): x2 → x2TmpFloatLocal          │  │                │
│ │ │  Add: sumFloatBuf = x1 + x2  (FP32残差加法)      │  │                │
│ │ └──────────────────────────────────────────────────┘  │                │
│ │                                                        │                │
│ │  写出 combine+add 结果: sumFloatBuf → Cast(BF16)       │                │
│ │  → DataCopyPad(expandOutGlobal_)  ← 即输出 x         │                │
│ │                                                        │                │
│ │ ┌─── AddRmsNormRmsNormCompute() ───────────────────┐  │                │
│ │ │  sumFloatBuf (= y, FP32)                         │  │                │
│ │ │  → Mul(y, y)           // y^2                     │  │                │
│ │ │  → Muls(1/H)           // mean(y^2)              │  │                │
│ │ │  → ReduceSum           // Σ                       │  │                │
│ │ │  → Adds(epsilon)       // + ε                     │  │                │
│ │ │  → Sqrt                // √(...)                  │  │                │
│ │ │  → Div(1/sqrt)         // rstd                    │  │                │
│ │ │  → DataCopy(rstdOutGlobal_)  // 写出rstd         │  │                │
│ │ │  → Muls(rstd)          // y * rstd                │  │                │
│ │ │  → gammaBuf → Cast(FP32)                          │  │                │
│ │ │  → Mul(gamma)          // y * rstd * gamma        │  │                │
│ │ │  → Cast(BF16) → DataCopyPad(yOutGlobal_)          │  │                │
│ │ │                            // 即输出 x (RMSNorm)  │  │                │
│ │ └──────────────────────────────────────────────────┘  │                │
│ └────────────────────────────────────────────────────────┘                │
│                                                                         │
│ ┌──────────┐  ┌──────────┐  ┌────────────┐                             │
│ │y [BS,1,H]│  │rstdOut   │  │x [BS,1,H]  │                             │
│ │BF16      │  │[BS,1,1]  │  │BF16        │                             │
│ │(add结果) │  │FP32      │  │(RMSNorm结果│                             │
│ └──────────┘  └──────────┘  └────────────┘                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 与标准 Combine 的对比数据流

```
标准 moe_distribute_combine_v2:
  Expert结果 → AlltoAll → 加权聚合 → combine_output [BS, H]
  └── 输出: x [BS, 1, H] (聚合结果)

moe_distribute_combine_add_rms_norm:
  Expert结果 → AlltoAll → 加权聚合 → combine_output [BS, H]
      ↓                                                   ↓
  [不变]                                            残差加法 y = combine + residual_x
      ↓                                                   ↓
  [不变]                                              RMSNorm x = y / RMS(y) * gamma
      ↓
  额外输出: y [BS,1,H] (残差加法), rstdOut [BS,1,1], x [BS,1,H] (归一化)

额外内存开销: +1个gammaBuf(H*2B) + 1个reduceFp32Buf(128B) UB空间
```

### 6.3 UB 内存布局 (ARN模式)

```
┌── UB (User Buffer) ──────────────────────────────────────────────┐
│                                                                    │
│ [通用Combine部分]                                                  │
│ ├── tokenBuf_        : max(activeMaskAlign, hExpandXAlign32)     │
│ ├── rowTmpFloatBuf_  : max(activeMaskAlignHalf, hFloatAlign32)   │
│ ├── mulBuf_          : max(hFloatAlign256, bsKFloatAlign)        │
│ ├── sumFloatBuf_     : hFloatAlign32          ← combine累加器    │
│ ├── moeSumQueue_     : bufferNum * hExpandXAlign32               │
│ ├── indexCountsBuf_  : sendCntNum * EXPAND_IDX_INFO * 4B         │
│ ├── stateBuf_        : flagRcvCount * STATE_OFFSET               │
│ ├── stateSumBuf_     : UB_ALIGN                                   │
│ └── stateResetBuf_   : flagRcvCount * STATE_OFFSET               │
│                                                                    │
│ [ARN专用部分, HasAddRmsNorm=true 时分配]                          │
│ ├── gammaBuf_        : hExpandXAlign32       ← gamma (BF16)      │
│ └── reduceFp32Buf_   : 128字节              ← ReduceSum辅助      │
│                                                                    │
│ gammaBuf_ 和 reduceFp32Buf_ 的分配位于 AlltoAllCommBuffInit()     │
│ 行705-709, 在其他通用Buffer之后分配                               │
└────────────────────────────────────────────────────────────────────┘
```

---

## 7. 性能设计

### 7.1 通信-计算流水线

ARN 融合的核心性能优势在于消除 combine 输出到 add_rms_norm 之间的 GM 读写往返:

```
非融合 (3次独立算子调用):
  Combine算子 → 结果写GM → 调度开销 → Add算子读GM → 结果写GM → 调度开销 → RMSNorm读GM
  ╎__3次GM写出+2次GM读入+2次算子调度__╎

融合 (单次算子调用):
  Combine → [原地Add+RMSNorm] → 结果写GM
  ╎__1次GM写出, 无额外调度开销__╎
```

### 7.2 ARN 计算流水线

在 `LocalWindowCopy()` 中，ARN 与 Combine 聚合形成 per-token 的紧凑流水线:

```
per-token 流水线:
  WaitDispatch → ProcessExpert → AddRmsNormAddCompute → Cast→GM → AddRmsNormRmsNormCompute → GM
  [轮询等待]    [scale加权累加]  [残差加法]              [写出y]  [RMSNorm]                 [写出x+rstd]
                                                                      
  UB空间复用: sumFloatBuf 既作为Combine累加器，又作为Add和RMSNorm的FP32输入，
              避免额外的UB分配。
```

### 7.3 FP32 精度保障

BF16 算术精度不足 (仅8位尾数)，ARN 所有中间计算均在 FP32 下进行:

```
combine_output (FP32累加, 无精度损失)
    ↓ Add (FP32)
y = combine + residual_x  (FP32)
    ↓ RMSNorm (FP32)
    Mul(y, y) → Muls(1/H) → ReduceSum → Adds(eps) → Sqrt → Div(1/sqrt)
    ↓
rstd = 1 / sqrt(mean(y^2) + eps)  (FP32, 写出为FP32)
    ↓
y * rstd * gamma → Cast(BF16) → 写出  (仅在最终输出时截断)
```

### 7.4 UB 额外开销分析

ARN 增加的 UB 消耗:

| 缓冲区 | 大小 | 用途 |
|--------|------|------|
| gammaBuf_ | Ceil(H * 2B, 32) * 32 | 存储gamma因子 (BF16) |
| reduceFp32Buf_ | 128B | ReduceSum辅助空间 |

gamma 在 token 循环外一次性加载 (第1403-1404行)，所有 token 共享同一份 gamma UB 数据。

---

## 8. 依赖关系

### 8.1 内部依赖

```
moe_distribute_combine_add_rms_norm
├── moe_distribute_combine_v2/ (委托关系)
│   ├── op_kernel/moe_distribute_combine_v2.h              ← MoeDistributeCombineV2 类
│   ├── op_kernel/moe_distribute_combine_v2_tiling.h       ← Tiling数据结构
│   ├── op_kernel/moe_distribute_combine_v2_tiling_key.h   ← TilingKey枚举
│   ├── op_kernel/moe_distribute_combine_tiling.h          ← A2 Tiling数据结构
│   ├── op_kernel/moe_distribute_combine_v2.cpp            ← V2内核入口(A2/A3/A5分发)
│   ├── op_kernel/moe_distribute_combine_v2_quant.h        ← INT8量化
│   ├── op_kernel/moe_distribute_combine_v2_layered.h      ← 分层模式
│   ├── op_kernel/moe_distribute_combine_a2.h              ← A2架构实现
│   ├── op_kernel/moe_distribute_combine_a2_layered.h      ← A2分层实现
│   ├── op_host/op_tiling/moe_distribute_combine_v2_tiling.cpp  ← V2 Tiling函数
│   └── op_host/op_tiling/moe_distribute_combine_tiling_helper.h ← Tiling辅助
├── moe_distribute_dispatch_v2/ (运行时依赖)
│   ├── op_kernel/check_winsize.h                          ← Window大小校验
│   ├── op_kernel/moe_distribute_v2_constant.h             ← 常量定义
│   └── op_kernel/moe_distribute_v2_base.h                 ← Dispatch基类
├── mc2/common/
│   ├── op_kernel/mc2_moe_context.h                        ← Mc2MoeContext结构
│   ├── op_kernel/moe_distribute_base.h                    ← MOE分布式基类
│   ├── op_kernel/mc2_common_def.h                         ← 通信算法枚举
│   ├── op_host/op_tiling/mc2_tiling_utils.h               ← 公共Tiling工具
│   └── op_host/op_tiling/mc2_tiling_struct.h              ← MC2ServerCfg等
└── 外部链接
    ├── aclnnInnerMoeDistributeCombineAddRmsNormGetWorkspaceSize (Workspace计算)
    ├── aclnnInnerMoeDistributeCombineAddRmsNorm (执行)
    └── NnopbaseSetHcclServerType (HCCL Server类型设置)
```

### 8.2 外部依赖

- **HCCL**: AlltoAll 通信 (EP域 + TP域), Window IPC, 状态同步
- **AscendC Runtime**: TPipe, TBuf, TQue, SyncFunc, DataCopyPad, Cast, Add, Mul, ReduceSum 等
- **matmul_intf**: LIB库加载 (`#include "lib/matmul_intf.h"` 在 kernel 入口)
- **kernel_tiling**: REGISTER_TILING_DEFAULT, GET_TILING_DATA_WITH_STRUCT 宏

---

## 9. 函数调用关系总览

```
moe_distribute_combine_add_rms_norm (内核入口, op_kernel/*.cpp 第65行)
├── REGISTER_TILING_DEFAULT(MoeDistributeCombineV2TilingData)
├── TPipe pipe;
└── ExecMoeDistributeCombineAddRmsNorm<..., HasAddRmsNorm=true>(...)
    ├── GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineV2TilingData)
    └── MoeDistributeCombineV2<..., HasAddRmsNorm=true>::
        ├── Init(...)
        │   ├── InitInputAndOutput()
        │   │   ├── [HasAddRmsNorm] residualXGM_, gammaGM_, yOutGlobal_, rstdOutGlobal_ 绑定
        │   │   └── expandXGM_, expertIdsGM_, ... 绑定
        │   ├── InitAttrs()
        │   │   ├── InitTilingAttrs(): axisBS_, axisH_, axisK_, ...
        │   │   ├── [HasAddRmsNorm] armAvgFactor_ = tilingData.armAvgFactor
        │   │   └── [HasAddRmsNorm] epsilon_ = tilingData.epsilon
        │   ├── SplitCoreCal(): 分核计算
        │   └── [IsNeedReduceScatter] TpGroupInit()
        │
        └── Process()
            ├── [IsNeedReduceScatter] ReduceScatterTrans()
            ├── BuffInit(): 分核UB分配
            ├── SetWaitTpStatusAndDisPatch()
            │   ├── [TP域] 状态同步
            │   └── ExpertAlltoAllDispatchCopyAdd(): AlltoAll发送
            ├── PipeBarrier<PIPE_ALL>()
            ├── AlltoAllBuffInitAndMaskCal()
            │   ├── AlltoAllCommBuffInit()
            │   │   ├── [HasAddRmsNorm] gammaBuf_, reduceFp32Buf_ 分配
            │   │   └── 其他Buffer分配
            │   ├── [isTokenMask] TokenMaskCalCnt()
            │   ├── [isExpertMask] ExpertMaskCalCnt()
            │   └── [enableSpecialExpert] MaskSpecialExpert()
            │
            └── LocalWindowCopy()
                ├── [HasAddRmsNorm] gamma 加载到 UB
                ├── ExpertScaleCopy()
                └── while (token循环):
                    ├── WaitDispatch(): 轮询状态
                    ├── ProcessExpert(): 加权聚合
                    │   ├── ProcessMoeExpert(): 标准专家
                    │   ├── ProcessCopyExpert(): 拷贝专家
                    │   └── ProcessConstantExpert(): 常量专家
                    ├── [HasAddRmsNorm] AddRmsNormAddCompute()
                    │   └── combine_output + residual_x
                    ├── Cast(FP32→BF16) + DataCopyPad(expandOutGlobal_)  → 输出x
                    └── [HasAddRmsNorm] AddRmsNormRmsNormCompute()
                        ├── 计算 rstd = 1/sqrt(mean(y^2)+eps)
                        ├── DataCopyPad(rstdOutGlobal_)  → 输出rstdOut
                        └── x = y * rstd * gamma → DataCopyPad(yOutGlobal_)  → 输出x

Tiling (op_host/op_tiling/*.cpp 第46行):
  MoeDistributeCombineAddRmsNormTilingFunc(context)
  └── optiling::MoeDistributeCombineV2TilingFunc(context)  ← 完全委托

ACLNN V1 (op_api/aclnn_*_v1.cpp 第27行):
  aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSize()
  └── aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSizeBase(nullptr*5)
      ├── CombineArnCheckParams(): 平台+参数校验
      └── aclnnInnerMoeDistributeCombineAddRmsNormGetWorkspaceSize()

ACLNN V2 (op_api/aclnn_*_v2.cpp 第28行):
  aclnnMoeDistributeCombineAddRmsNormV2GetWorkspaceSize()
  └── aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSizeBase(全参数)
      ├── CombineArnCheckParams(): 平台+参数校验
      └── aclnnInnerMoeDistributeCombineAddRmsNormGetWorkspaceSize()
```

---

*分析完成时间: 2026-04-20（修订: 2026-04-21 统一行号引用格式为"第X-Y行"）*
