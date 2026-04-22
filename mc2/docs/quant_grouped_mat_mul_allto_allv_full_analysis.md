# quant_grouped_mat_mul_allto_allv 算子全面分析报告

## 1. 算子概述

`quant_grouped_mat_mul_allto_allv` 是 MOE Combine 阶段的 **量化 GroupedMatMul + AllToAllV 通信融合**算子（算子定义类见 `_def.cpp` 第19行，构造函数第22-169行）。采用 mc2_templates Stage 化架构，数据流方向为 **先计算后通信**（GMM → Unpermute → AllToAllV），与兄弟算子 `allto_allv_quant_grouped_mat_mul`（先通信后计算）互为对偶。

支持 PerTensor/PerBlock/MX 量化模式定义，当前仅 **TT（PerTensor×PerTensor）路径**完全实现（`tiling_key.h` 第60-186行定义了18种 `ASCENDC_TPL_SEL` 模板组合，`ASCENDC_TPL_ARGS_DECL` 声明于第37-54行）。输入为 HIFLOAT8 量化数据 + FP32 缩放因子，输出为 FP16/BF16 反量化结果。可选并行执行共享专家（Shared Expert）的独立 MatMul。

**平台**: 仅 `ascend950`（arch35/A5），通信使用 CCU Server。

```
数据流概览:
                                    可选并行
gmm_x [A,H1] ──┐                    mm_x [BS,H2] ──┐
gmm_wt [e,H1,N1]┤ ×gmmXScale×gmmWtScale            │ ×mmXScale×mmWtScale
gmmXScl / gmmWtScl┤                              mmXScl/mmWtScl┤
                ↓                                    ↓
         量化 GMM (per expert)                   量化 MM
                ↓                                    ↓
         Unpermute + AllToAllV ──→ y [BSK,N1]    mm_y [BS,N2]
```

---

## 2. 目录结构与文件职责

```
quant_grouped_mat_mul_allto_allv/
├── op_kernel/
│   ├── quant_grouped_mat_mul_allto_allv.cpp           (117行) 内核入口，模板实例化+Scheduler驱动
│   │   └── 入口签名第56-63行, GMMATAVType结构第47-53行, REGISTER_TILING_DEFAULT第75行, GET_TILING_DATA第76行
│   ├── quant_grouped_mat_mul_allto_allv_tiling_key.h  (189行) 18种模板组合(TPL_ARGS_DECL/TPL_SEL)
│   │   └── ASCENDC_TPL_ARGS_DECL第37-54行, ASCENDC_TPL_SEL第60-186行
│   └── arch35/
│       └── quant_grouped_mat_mul_allto_allv_tiling.h  (74行)  QuantGmmA2avTilingData 结构定义
│           └── GmmA2avWorkspaceInfo第48-52行, QuantGmmA2avTilingData第54-72行
├── op_host/
│   ├── quant_grouped_mat_mul_allto_allv_def.cpp       (173行) 算子定义: 11输入/2输出/14属性
│   │   └── 类定义第19行, 构造函数第22-169行, 输入第24-122行, 输出第124-139行, 属性第141-155行, OP_ADD第172行
│   ├── quant_grouped_mat_mul_allto_allv_infershape.cpp (212行) Shape推导+校验
│   ├── op_tiling/
│   │   ├── quant_grouped_mat_mul_allto_allv_tiling.cpp      (51行)  Tiling分发入口
│   │   ├── quant_grouped_mat_mul_allto_allv_tiling_base.h   (151行) 基类+枚举+常量
│   │   ├── quant_grouped_mat_mul_allto_allv_tiling_base.cpp (128行) 基类实现+GetQuantMode
│   │   └── arch35/
│   │       ├── tt_quant_grouped_mat_mul_allto_allv_tiling.h   (137行) TT模式Tiling类声明
│   │       ├── tt_quant_grouped_mat_mul_allto_allv_tiling.cpp (877行) TT模式完整校验+Tiling
│   │       ├── quant_grouped_mat_mul_allto_allv_tiling_adapter.h   (60行)  Mc2GroupedQbmmTiling适配器
│   │       └── quant_grouped_mat_mul_allto_allv_tiling_adapter.cpp (109行) 适配器实现
│   ├── config/ascend950/
│   │   └── quant_grouped_mat_mul_allto_allv_binary.json (259行) 预编译二进制配置(FP16/BF16)
│   └── op_api/  → 链接到 op_api/ 目录
├── op_api/
│   ├── aclnn_quant_grouped_mat_mul_allto_allv.h  (116行) ACLNN C-API声明
│   └── aclnn_quant_grouped_mat_mul_allto_allv.cpp (397行) ACLNN C-API实现+参数校验
├── op_graph/
│   ├── quant_grouped_mat_mul_allto_allv_proto.h             (101行) 图模式算子原型
│   ├── quant_grouped_mat_mul_allto_allv_gen_task_training.cpp (82行)  图模式GenTask
│   └── fallback_quant_grouped_mat_mul_allto_allv.cpp        (144行) 回退路径→非量化版本
├── examples/
│   └── test_aclnn_quant_grouped_mat_mul_allto_allv.cpp (362行) 多线程8卡E2E示例
├── tests/ut/
│   ├── op_api/test_aclnn_quant_grouped_mat_mul_allto_allv.cpp (1251行) ACLNN API UT
│   └── op_host/arch35/test_quant_gmm_alltoallv_tiling.cpp    (690行)  Tiling UT
└── docs/
    └── aclnnQuantGroupedMatMulAlltoAllv.md (747行) API文档

依赖的共享模板 (allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/):
├── common/a2av_common_tiling.h            (65行)  HcclA2avTilingInfo/TaskTilingInfo
├── communication/hccl_a2av_op.h           (214行) HcclA2avOp AllToAllV封装
├── compute/quant_grouped_matmul.h         (182行) QuantGroupedMatmul 计算封装
└── scheduler/gmm_a2av_scheduler.h         (66行)  GmmA2avScheduler 调度器

模板参数声明与选择宏 (`tiling_key.h`):
├── ASCENDC_TPL_ARGS_DECL  第37-54行  声明5个编译期模板参数
└── ASCENDC_TPL_SEL         第60-186行 18种模板组合实例化选择

内核关键行号 (`quant_grouped_mat_mul_allto_allv.cpp`):
├── GMMATAVType struct      第47-53行
├── 入口签名                 第56-63行
├── REGISTER_TILING_DEFAULT 第75行
└── GET_TILING_DATA         第76行
```

---

## 3. Host 侧分析

### 3.1 算子定义 (`_def.cpp`:173行)

**文件**: `op_host/quant_grouped_mat_mul_allto_allv_def.cpp`（类定义见第19行，构造函数第22-169行）

**输入张量** (11个, `_def.cpp` 第24-122行):

| 序号 | 名称 | 类型 | 维度 | 必选 | 说明 |
|------|------|------|------|------|------|
| 0 | gmm_x | HIFLOAT8/FP16/BF16 | [A, H1] | 是 | 路由专家GMM输入, A=本卡token总数 |
| 1 | gmm_weight | HIFLOAT8/FP16/BF16 | [e, H1, N1] | 是 | 路由专家量化权重, e=本卡专家数 |
| 2 | send_counts_tensor | INT32/INT64 | [ep*epWorldSize] | 否 | 每卡发送token数(当前必须为null) |
| 3 | recv_counts_tensor | INT32/INT64 | [ep*epWorldSize] | 否 | 每卡接收token数(当前必须为null) |
| 4 | mm_x | HIFLOAT8/FP16/BF16 | [BS, H2] | 否 | 共享专家输入 |
| 5 | mm_weight | HIFLOAT8/FP16/BF16 | [H2, N2] | 否 | 共享专家权重 |
| 6 | gmmXScaleOptional | FLOAT32 | - | 否 | GMM输入量化缩放因子 |
| 7 | gmmWeightScaleOptional | FLOAT32 | - | 否 | GMM权重量化缩放因子 |
| 8 | mmXScaleOptional | FLOAT32 | - | 否 | MM输入量化缩放因子 |
| 9 | mmWeightScaleOptional | FLOAT32 | - | 否 | MM权重量化缩放因子 |
| 10 | commQuantScaleOptional | FLOAT32 | - | 否 | 通信量化缩放(当前未启用) |

**输出张量** (2个, `_def.cpp` 第124-139行):

| 序号 | 名称 | 类型 | 维度 | 必选 | 说明 |
|------|------|------|------|------|------|
| 0 | y | FP16/BF16 | [BSK, N1] | 是 | AllToAllV融合输出, BSK=sum(recvCounts) |
| 1 | mm_y | FP16/BF16 | [BS, N2] | 否 | 共享专家输出 |

**关键属性** (14个, `_def.cpp` 第141-155行):
- `group` (string): HCCL通信组名
- `ep_world_size` (int): EP域总卡数
- `send_counts` / `recv_counts` (listInt): 每专家发送/接收token数
- `trans_gmm_weight` / `trans_mm_weight` (bool): 权重转置标志
- `gmm_x_quant_mode` / `gmm_weight_quant_mode` (int): GMM量化模式(0=None, 1=PerTensor)
- `mm_x_quant_mode` / `mm_weight_quant_mode` (int): MM量化模式
- `comm_quant_mode` (int): 通信量化模式(当前必须=0)
- `group_size` (int): 量化分组大小(当前必须=0)
- `y_dtype` / `mm_dtype` (int): 输出数据类型

**平台配置** (`_def.cpp` 第157-168行):
```
ascend950:
  DynamicCompileStaticFlag = true
  DynamicFormatFlag = true
  multiKernelSupportDynamicGraph
  MC2 HCCL group = {"group"}
```

### 3.2 Tiling 策略

#### 3.2.1 基类与枚举 (`tiling_base.h`:151行, `tiling_base.cpp`:128行)

**输入/输出/属性索引枚举** (`tiling_base.h` 第25-60行):
```
GmmA2AvInputTensorIndex:  IDX_GMM_X=0 .. IDX_COMM_QUANT_SCALE=10 (11项)
GmmA2AvOutputTensorIndex: IDX_Y=0, IDX_MM_Y=1
GmmA2AvAttrIndex:         ATTR_GROUP=0 .. ATTR_MM_DTYPE=13 (14项)
```

**量化模式枚举** (`tiling_base.h` 第62-78行):
```
QuantizationMode: NONE=0, PERTENSOR=1, PERCHANNEL=2, PERTOKEN=3,
                  PERGROUP=4, PERBLOCK=5, MX=6, PERTOKEN_DYNAMIC=7
QuantModePair:    NONE=0, TT=1, KC=2, ERROR=3
```

**关键常量** (`tiling_base.h` 第80-129行):
```
MAX_EXPERT_NUM=256, MAX_EXPERT_NUM_PER_RANK=32, MAX_BSK_VALUE=52428800
MAX_SHARED_H_SHAPE_SIZE=12288, HCCL_CMD_ALLTOALLV=8
COMM_TILE=8, CUBE_BLOCK=16, SUPPORT_RANK_SIZE={2,4,8,16,32,64,128,256}
```

**GetQuantMode()** (`tiling_base.cpp` 第97-126行): 根据 gmm/mm 的 x_quant_mode 和 weight_quant_mode 返回配对:
- 两者都=0 → `QUANT_PAIR_NONE`
- 两者都=1 → `QUANT_PAIR_TT`
- 其他 → `QUANT_PAIR_ERROR`

#### 3.2.2 TT模式Tiling (`tt_*_tiling.h`:137行, `tt_*_tiling.cpp`:877行)

**类**: `TTQuantGroupedMatmulAllToAllvTiling : public QuantGmmAlltoAllvTilingBase`

**参数结构** (`tt_*_tiling.h` 第35-77行):
```cpp
struct QuantGmmAlltoAllvParamsInfo {
    // GMM参数
    int64_t A, H1, ep, BsK, N1;       // gmmX=[A,H1], gmmWeight=[ep,H1,N1], y=[BsK,N1]
    int64_t expertNum;                  // = ep * epWorldSize (总专家数)
    // MM参数
    int64_t Bs, H2, N2;               // mmX=[Bs,H2], mmWeight=[H2,N2], mmY=[Bs,N2]
    // 标志
    bool hasSharedMm;
    int8_t gmmQuantMode, mmQuantMode;
    bool isGmmWeightTrans, isMmWeightTrans;
    // 数据类型
    ge::DataType gmmXDtype, gmmWeightDtype, yDtype, mmXDtype, mmWeightDtype, mmYDtype;
    int64_t coreNums;
    int64_t epWorldSize;
};
```

**校验链** (`tt_*_tiling.cpp`):
```
GetShapeAttrsInfo() [第54-66行]
  → CheckOpInputSingleParamsTensor*() [第69-135行]   send/recv必须null, scale非空
  → CheckAndSetLocalParamsGmm() [第137-185行]         HIFLOAT8输入, FP16/BF16输出
  → CheckAndSetLocalParamsMm() [第187-240行]          可选MM一致性
  → CheckAndSetLocalParamsAttr() [第242-278行]        量化模式+转置标志
  → CheckFormat() [第280-327行]                       全部FORMAT_ND
  → CheckParamsRelationGmm() [第354-401行]            scale=FP32, K/N维度匹配
  → CheckParamsRelationMm() [第403-455行]             MM scale/维度校验
  → CheckParamsAttrEpAndSetLocalParams() [第457-499行] epWorldSize∈SUPPORT, expertNum≤256
  → CheckAndSetSendRecvCountsAttr() [第501-541行]     sendCounts总和=A, recvCounts总和=BsK
  → CheckLocalParams() [第543-573行]                  H1<65536, BsK<52428800, N1<65536, K∈[2,8]
```

**IsCapable()** (`tt_*_tiling.cpp` 第616-626行): 仅 `QUANT_PAIR_TT` 返回 true。

**Tiling核心逻辑**:
- `SetTilingCommonInfo()` (`tt_*_tiling.cpp` 第646-664行): 填充 taskTilingInfo，mainLoopExpertNum=tailLoopExpertNum=1，totalLoopCount=ep
- `SetGmmA2avWorkspaceInfo()` (`tt_*_tiling.cpp` 第666-680行):
  ```
  wsGmmOutputSize      = A * N1 * sizeof(FP16/BF16) 对齐到512字节
  wsGmmComputeWorkspaceSize     = 1MB (内含64B ptrTable + expertNum*8B groupList)
  wsSharedGmmComputeWorkspaceSize = 1MB
  ```
- `DoQuantGMMTiling()` (`tt_*_tiling.cpp` 第682-705行): 通过 `QuantGroupedMatmulAllToAllvAdapter` 计算路由专家和共享专家的GMM Tiling
- `SetHcclTiling()` (`tt_*_tiling.cpp` 第707-731行): HCCL AllToAllV配置，command=8，config="AlltoAll=level0:fullmesh;level1:pairwise"
- `PostTiling()` (`tt_*_tiling.cpp` 第829-851行): 拷贝tiling数据，设置blockDim=aicCoreNum
- `GetTilingKey()` (`tt_*_tiling.cpp` 第863-871行): `GET_TPL_TILING_KEY(hasSharedMm, isGmmWeightTrans, isMmWeightTrans, gmmQuantSuit, mmQuantSuit)`

#### 3.2.3 Tiling适配器 (`tiling_adapter.h`:60行, `tiling_adapter.cpp`:109行)

**类**: `QuantGroupedMatmulAllToAllvAdapter : public Mc2GroupedQbmmTiling`（第三方grouped_matmul Tiling封装）

```
SetCommonInputParams() [第27-46行]:
  kernelType=0, splitItem=2(单张量输出), actType=0, ND格式, 无bias
SetGroupExpertInputParameters() [第48-73行]:
  M=gmmX_epSize, K=H1, N=N1, groupNum=1, groupType=SPLIT_M
SetSharedExpertInputParameters() [第75-101行]:
  M=Bs, K=H2, N=N2, groupNum=1
Process() [第103-108行]:
  → DoOpTiling() + DoLibApiTiling()
```

### 3.3 Shape推导 (`_infershape.cpp`:212行)

**文件**: `op_host/quant_grouped_mat_mul_allto_allv_infershape.cpp`

| 函数 | 行号 | 功能 |
|------|------|------|
| `CheckDims()` | 第45-72行 | 校验gmmX=2D, gmmWeight=3D, K维度匹配(考虑转置) |
| `CheckDimsOptional()` | 第74-101行 | 校验mmX=2D, mmWeight=2D, K维度匹配 |
| `InferGMMOutputShape()` | 第103-127行 | y=[bsk, n1], bsk=sum(recvCounts) |
| `InferMMOutputShape()` | 第129-154行 | mm_y=[bs, n2] |
| `InferShape/DataType` | 第156-208行 | 主入口，调用上述函数 |

### 3.4 ACLNN API (`aclnn_*.h`:116行, `aclnn_*.cpp`:397行)

**两阶段API**:

```cpp
// Phase 1: 计算Workspace (aclnn_*.h 第90-98行)
aclnnResult aclnnQuantGroupedMatMulAlltoAllvGetWorkspaceSize(
    aclnnTensorList* gmmX, aclnnTensorList* gmmWeight,           // 路由专家输入
    aclnnTensor* sendCountsTensor, aclnnTensor* recvCountsTensor, // 通信计数
    aclnnTensorList* mmX, aclnnTensorList* mmWeight,             // 共享专家输入(可选)
    aclnnTensorList* gmmXScale, aclnnTensorList* gmmWeightScale, // GMM量化缩放
    aclnnTensorList* mmXScale, aclnnTensorList* mmWeightScale,   // MM量化缩放
    aclnnTensor* commQuantScale,                                  // 通信量化缩放
    uint64_t group, int64_t epWorldSize,                          // HCCL组配置
    aclnnIntArray* sendCounts, aclnnIntArray* recvCounts,         // 属性形式计数
    int64_t transGmmWeight, int64_t transMmWeight,               // 转置标志
    int64_t gmmXQuantMode, ..., int64_t mmDtype,                  // 量化+输出类型
    aclnnTensor* y, aclnnTensor* mmY,                            // 输出
    uint64_t* workspaceSize, aclnnOpExecutor** executor);

// Phase 2: 执行 (aclnn_*.h 第109-110行)
aclnnResult aclnnQuantGroupedMatMulAlltoAllv(
    void* workspace, uint64_t workspaceSize,
    aclnnOpExecutor* executor, aclrtStream stream);
```

**参数校验链** (`aclnn_*.cpp`):
```
CheckNotNull() [第70-85行]        gmmX/gmmWeight/y必选非空
CheckMmOptionalConsistency() [第87-103行]  mmX/mmWeight/mmY全空或全非空
CheckNullStatus() [第106-121行]   sendCounts/recvCounts tensor必须null, group非空
CheckSendAndRecv() [第123-146行]  数组非空
CheckNotEmptyTensor() [第148-205行] 无零维张量(除gmmX dim0允许=0)
CheckQuantMode() [第284-310行]    mode=0则scale必须null, mode=1则scale必须非空
CheckParams() [第313-348行]       汇总校验, commQuantMode必须=0
```

**执行** (`aclnn_*.cpp` 第386-396行): arch3510 设置 HCCL Server 为 CCU，调用 inner API。

---

## 4. Kernel 侧分析

### 4.1 内核入口 (`quant_grouped_mat_mul_allto_allv.cpp`:117行)

**内核函数签名** (`quant_grouped_mat_mul_allto_allv.cpp` 第56-63行):
```cpp
__global__ __aicore__ void quant_grouped_mat_mul_allto_allv(
    // 5个编译期模板参数
    bool TILINGKEY_COMPUTE_MATMUL,          // 是否有共享专家MM
    bool TILINGKEY_GROUPED_MATMUL_TRANS,    // GMM权重转置
    bool TILINGKEY_MATMUL_TRANS,            // MM权重转置
    uint8_t TILINGKEY_GMM_QUANT_MODE,       // GMM量化模式
    uint8_t TILINGKEY_SHARED_MM_QUANT_MODE, // MM量化模式
    // 16个GM地址参数
    GM_ADDR gmmx, gmmweight, sendCountsTensor, recvCountsTensor,
            mmx, mmweight, gmmxScale, gmmWeightScale,
            mmxScale, mmWeightScale, commQuantScale,
            y, mmy, workspace, tiling, ...)
```

**内核任务类型** (`quant_grouped_mat_mul_allto_allv.cpp` 第65行): `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` — AIC+Vector混合内核。

**组件组装** (`quant_grouped_mat_mul_allto_allv.cpp` 第86-96行):
```
HcclOpType     = HcclA2avOp<DTYPE_Y, false>     // false=计算后通信
GmmASWKernelType = Mc2GmmASWKernel (from 3rd)    // 底层GMM ASW内核
ComputeOpType  = QuantGroupedMatmul<..., false, false>  // 路由专家, isLocal=false, isA2avGmm=false
SharedGmmType  = QuantGroupedMatmul<..., true, false>   // 共享专家, isLocal=true, isA2avGmm=false
SchedulerType  = GmmA2avScheduler<HcclOpType, ComputeOpType, SharedGmmType, TILINGKEY_COMPUTE_MATMUL>
```

**初始化流程** (`quant_grouped_mat_mul_allto_allv.cpp` 第98-116行):
```
1. HCCL初始化: 从tiling获取hcclInitTiling和a2avCcTiling偏移
2. 路由专家GMM初始化:
   - gmmArrayAddr_ 通过嵌套tiling宏获取
   - workspace偏移: wsGmmOutputSize(存放groupList)
3. 共享专家MM初始化:
   - mmArrayAddr_ 通过嵌套tiling宏获取
   - workspace偏移: wsGmmOutputSize + wsGmmComputeWorkspaceSize
4. 创建Scheduler并调用Process()
```

### 4.2 mc2_templates 架构

本算子使用 `allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/` 下的通用模板组件:

```
组件关系:
┌─────────────────────────────────────────────────────────────────┐
│  GmmA2avScheduler<HcclOpType, ComputeOpType, SharedType, MM>  │
│  (scheduler/gmm_a2av_scheduler.h 第66行)                       │
│                                                                 │
│  Process():                                                     │
│  ├── [IsNeedMM] SharedComputeOp.Process(0)  ─── 并行共享专家    │
│  │                                          (实际是串行执行)     │
│  ├── for expertIdx in [0, ep):                                  │
│  │   ├── ComputeOp.Process(expertIdx)  ←── QuantGroupedMatmul │
│  │   │   ├── UpdateAddr(expertIdx)                               │
│  │   │   ├── BuildPtrTable()                                     │
│  │   │   └── Mc2GmmASWKernel::Init+Process (量化GMM)            │
│  │   └── HcclOp.Launch(expertIdx, 1)  ←── HcclA2avOp           │
│  └── HcclOp.WaitAll() + End()                                   │
└─────────────────────────────────────────────────────────────────┘
```

**注意**: GmmA2avScheduler::Process() 的实际执行模式是 **串行计算后通信**: 对每个expert先计算再启动通信，各expert间不重叠。共享专家在所有路由expert之后执行。

### 4.3 HcclA2avOp 通信封装 (`hccl_a2av_op.h` 第214行)

```
HcclA2avOp<DTYPE_Y, commBeforeComputeFlag=false>

Init():
  设置HCCL上下文, rankId, rankDim, e, H1, N1, send/recv缓冲区

Launch(startExpertIdx, expertNum):
  按专家聚合每卡send/recv token数
  → LaunchCommAfterCompute() (本算子路径)
    axis=N1, 计算每专家batch的send/recv偏移
    → hccl_.AlltoAllV<true>()

通信服务: arch3510使用 CCU Server, 其他使用 AICPU Server
```

### 4.4 QuantGroupedMatmul 计算封装 (`quant_grouped_matmul.h` 第182行)

```
QuantGroupedMatmul<TilingData, GmmTilingData,
                   xType, wType, scaleType, yType, wFormat,
                   aTrans, bTrans, isLocal, isA2avGmm>

Init():
  设置GM全局缓冲区, 从recvCnt/sendCnt计算expertTokenNum
  分配ptrTable和groupList的GM空间

Process(expertIdx):
  ├── 跳过零token专家
  ├── UpdateAddr(expertIdx)     推进x/y/weight偏移
  ├── BuildPtrTable(dataAddr, slotIdx)  构建16字节间接寻址槽
  │     槽格式: [byteOffset(8B), dataAddr(8B)]
  │     4个槽: x, w, scaleB, y
  └── Mc2GmmASWKernel::Init+Process  调用第三方量化GMM内核
```

### 4.5 TilingKey与模板组合 (`tiling_key.h`:189行)

**5个模板参数** (`tiling_key.h` 第37-54行):
```
TILINGKEY_COMPUTE_MATMUL          (bool)   ← 共享专家MM开关
TILINGKEY_GMM_WEIGHT_TRANS        (bool)   ← GMM权重转置
TILINGKEY_SHARED_MM_WEIGHT_TRANS  (bool)   ← MM权重转置
TILINGKEY_GMM_QUANT_MODE          (uint:4) ← GMM量化模式(0=None, 1=TT)
TILINGKEY_SHARED_MM_QUANT_MODE    (uint:4) ← MM量化模式
```

**18种组合** (`tiling_key.h` 第60-186行): 按COMPUTE_MATMUL × TRANS × QUANT_MODE 全排列:
- 组合1-4: 无共享专家(COMPUTE_MATMUL=0), GMM_QUANT=TT, 变化TRANS
- 组合5-10: 有共享专家(COMPUTE_MATMUL=1), GMM_QUANT=TT+MM_QUANT=TT, 变化TRANS
- 组合11-18: 有/无共享专家, GMM_QUANT=NONE+MM_QUANT=NONE, 变化TRANS

### 4.6 Tiling数据结构 (`arch35/quant_grouped_mat_mul_allto_allv_tiling.h`:74行)

```cpp
struct GmmA2avWorkspaceInfo {                          // 第48-52行
    uint64_t wsGmmOutputSize;              // 路由专家GMM输出缓冲区
    uint64_t wsGmmComputeWorkspaceSize;     // 路由专家GMM工作区 (1MB)
    uint64_t wsSharedGmmComputeWorkspaceSize; // 共享专家GMM工作区 (1MB)
};

struct QuantGmmA2avTilingData {                        // 第54-72行
    HcclA2avTilingInfo hcclA2avTiling;     // HCCL初始化+CC Tiling
    GmmA2avWorkspaceInfo workspaceInfo;     // 三段Workspace布局
    TaskTilingInfo taskTilingInfo;          // BSK,H1,N1,ep,send/recv counts
    GMMQuantTilingData sharedGmmTiling;    // 共享专家GMM Tiling
    GMMQuantTilingData gmmBaseTiling;      // 路由专家GMM Tiling
    bool isPermuteOut;                     // 输出是否需要排列
};
```

**TaskTilingInfo** (`a2av_common_tiling.h`):
```
BSK, BS, H1, H2, A, N1, N2    ← 形状参数
epWorldSize, e                  ← EP域配置
mainLoopExpertNum=1             ← 每轮1个专家(串行)
tailLoopExpertNum=1
totalLoopCount=ep
sendCnt[256], recvCnt[256]     ← 每专家发送/接收计数
```

---

## 5. Host-Kernel 接口

### 5.1 TilingKey 编码

```
GET_TPL_TILING_KEY(hasSharedMm, isGmmWeightTrans, isMmWeightTrans,
                   gmmQuantSuit, mmQuantSuit)

十进制编码位:
  bit4: hasSharedMm (TILINGKEY_COMPUTE_MATMUL)
  bit3: isGmmWeightTrans
  bit2: isMmWeightTrans
  bit1-0: gmmQuantSuit × mmQuantSuit 组合

典型值:
  Key=137 → hasSharedMm=1, gmmTrans=0, mmTrans=0, gmmQuant=TT, mmQuant=TT
```

### 5.2 Workspace 布局

```
Workspace 总布局:
┌──────────────────────────────────────────────────────────────────┐
│ libApiWorkspace    (由HCCL框架分配)                              │
├──────────────────────────────────────────────────────────────────┤
│ gmmOutputBuffer    A*N1*elemSize, 对齐到512字节                  │
│  [路由专家GMM计算中间结果, 也存放groupList]                       │
├──────────────────────────────────────────────────────────────────┤
│ gmmComputeWorkspace    1MB                                      │
│  ├── ptrTable      64字节 (4个×16字节间接寻址槽)                  │
│  └── groupList     expertNum*8字节 (每专家token偏移)              │
├──────────────────────────────────────────────────────────────────┤
│ sharedGmmComputeWorkspace  1MB                                  │
│  ├── ptrTable      64字节                                       │
│  └── groupList     1*8字节 (共享专家单组)                         │
└──────────────────────────────────────────────────────────────────┘
```

### 5.3 Tiling数据传递

Host → Kernel 通过 `__gm__ uint8_t* tiling` 指针传递 `QuantGmmA2avTilingData`:
```
Host (PostTiling, `tt_*_tiling.cpp` 第829-851行):
  localTilingData_ → context → tiling data buffer

Kernel (`quant_grouped_mat_mul_allto_allv.cpp` 第98-116行):
  嵌套宏 GET_NESTED_TILING_DATA_MEMBER_ADDR 解析子结构偏移
  → hcclA2avTiling  → HCCL初始化
  → gmmBaseTiling   → 路由专家GMM配置
  → sharedGmmTiling → 共享专家MM配置
  → workspaceInfo   → Workspace分段偏移
  → taskTilingInfo  → sendCnt/recvCnt数组
```

---

## 6. 数据流

### 6.1 完整数据流图

```
                           Host侧
  ┌──────────────────────────────────────────────────────────────┐
  │ aclnnQuantGroupedMatMulAlltoAllvGetWorkspaceSize()          │
  │  → 参数校验 → Tiling(校验+计算) → 分配Workspace              │
  │  → 注册TilingKey → 设置blockDim                              │
  └───────────────────────────┬──────────────────────────────────┘
                              ↓
                           Kernel侧
  ┌──────────────────────────────────────────────────────────────┐
  │ quant_grouped_mat_mul_allto_allv<K1,K2,K3,Q1,Q2>(...)      │
  │                                                              │
  │  ┌─ GmmA2avScheduler::Process() ─────────────────────────┐  │
  │  │                                                        │  │
  │  │  [可选] Shared Expert:                                 │  │
  │  │  mm_x[BS,H2] × mm_wt[H2,N2] × mmXScale × mmWtScale  │  │
  │  │       → QuantGroupedMatmul → mm_y[BS,N2]              │  │
  │  │                                                        │  │
  │  │  for expertIdx = 0..ep-1:                              │  │
  │  │  ┌──────────────────────────────────────────────────┐  │  │
  │  │  │ Expert i:                                         │  │  │
  │  │  │ gmm_x偏移[expertTokenNum,H1]                     │  │  │
  │  │  │   × gmm_wt[i,H1,N1] × gmmXScale × gmmWtScale   │  │  │
  │  │  │   → Mc2GmmASWKernel → GMM结果[expertTokenNum,N1] │  │  │
  │  │  │       ↓                                           │  │  │
  │  │  │ HcclA2avOp::Launch(i, 1)                         │  │  │
  │  │  │   AllToAllV发送: 本expert结果→对应rank的y区域     │  │  │
  │  │  └──────────────────────────────────────────────────┘  │  │
  │  │                                                        │  │
  │  │  HcclA2avOp::WaitAll() + End()                         │  │
  │  └────────────────────────────────────────────────────────┘  │
  │                                                              │
  │  输出: y[BSK,N1] (AllToAllV聚合结果)                        │
  │       mm_y[BS,N2] (共享专家结果, 可选)                       │
  └──────────────────────────────────────────────────────────────┘
```

### 6.2 AllToAllV通信模式

```
Rank 0                          Rank 1
┌──────────┐                   ┌──────────┐
│ Expert 0  │──sendCounts[0]──→│ y区[0,*] │
│ Expert 1  │──sendCounts[1]──→│ y区[1,*] │
│ Expert 2  │──sendCounts[2]──→│ y区[2,*] │
│ Expert 3  │──sendCounts[3]──→│ y区[3,*] │
└──────────┘                   └──────────┘
       ↕ AllToAllV (HCCL CCU)        ↕
┌──────────┐                   ┌──────────┐
│ y区[0,*] │←──recvCounts[0]──│ Expert 0 │
│ y区[1,*] │←──recvCounts[1]──│ Expert 1 │
│ y区[2,*] │←──recvCounts[2]──│ Expert 2 │
│ y区[3,*] │←──recvCounts[3]──│ Expert 3 │
└──────────┘                   └──────────┘

配置: level0=fullmesh, level1=pairwise
命令: HCCL_CMD_ALLTOALLV (8)
```

### 6.3 量化GMM计算流

```
HIFLOAT8 gmm_x [expertTokenNum, H1]
HIFLOAT8 gmm_weight [H1, N1]
     ↓
Mc2GmmASWKernel (3rd/grouped_matmul 量化内核)
     ├── Cube: HIFLOAT8 × HIFLOAT8 → FP32 累加
     ├── Vector: FP32 × gmmXScale × gmmWeightScale → FP32
     └── Vector: Cast FP32 → FP16/BF16 输出
     ↓
FP16/BF16 结果 [expertTokenNum, N1] → 送入AllToAllV
```

---

## 7. 性能设计

### 7.1 串行计算-通信流水线

当前实现为 **per-expert 串行模式**（mainLoopExpertNum=1）:
```
时间线:
Expert 0: [GMM Compute] [Sync] [AllToAllV Launch]
Expert 1:                [GMM Compute] [Sync] [AllToAllV Launch]
  ...
Expert e-1:                                    [GMM Compute] [Sync] [AllToAllV Launch]
Shared MM:                                                          [MM Compute]
                                                                    [WaitAll] [End]
```

### 7.2 Workspace优化

- GMM输出缓冲区对齐到512字节，避免非对齐访问
- 每个专家的计算工作区固定1MB，内含ptrTable和groupList
- ptrTable采用16字节间接寻址槽设计，支持快速地址切换

### 7.3 HCCL通信优化

- CCU Server: arch35/950上使用CCU硬件通信加速器，减少CPU开销
- 配置: `level0:fullmesh` 低延迟直连 + `level1:pairwise` 大规模分对
- AllToAllV按expert粒度聚合token，减少小消息数量

---

## 8. 依赖关系

### 8.1 内部依赖

```
quant_grouped_mat_mul_allto_allv
├── mc2_templates (共享模板, 位于 allto_allv_quant_grouped_mat_mul/)
│   ├── common/a2av_common_tiling.h         HcclA2avTilingInfo, TaskTilingInfo
│   ├── communication/hccl_a2av_op.h        HcclA2avOp (AllToAllV封装)
│   ├── compute/quant_grouped_matmul.h      QuantGroupedMatmul (量化GMM封装)
│   └── scheduler/gmm_a2av_scheduler.h      GmmA2avScheduler (调度器)
├── 3rd/grouped_matmul
│   ├── Mc2GmmASWKernel                     底层量化GMM内核
│   └── Mc2GroupedQbmmTiling                GMM Tiling适配器
├── mc2/common/
│   ├── op_kernel/mc2_common_def.h          通信算法枚举
│   └── op_host/op_tiling/mc2_tiling_struct.h MC2 tiling基础结构
└── HCCL SDK
    ├── Hccl<CCU> (arch3510)                CCU硬件通信
    └── hccl_.AlltoAllV<true>()             AllToAllV原语
```

### 8.2 与 grouped_mat_mul_allto_allv 的核心差异

| 特性 | grouped_mat_mul_allto_allv | quant_grouped_mat_mul_allto_allv |
|------|---------------------------|--------------------------------|
| 架构 | 单一大类 (~数千行) | mc2_templates Stage 化组件 |
| 输入数据类型 | FP16/BF16 | HIFLOAT8 (量化) |
| 量化支持 | 不支持 | PerTensor×PerTensor |
| 缩放因子 | 无 | gmmXScale/gmmWeightScale等4个 |
| 通信量化 | 不支持 | commQuantScale (未启用) |
| 平台 | 910_93 + 950 | 仅 950 |
| HCCL Server | AICPU(910_93)/CCU(950) | CCU (950) |
| TilingKey数 | 8种 | 18种 |
| Tiling适配器 | 直接Tiling | Mc2GroupedQbmmTiling (3rd) |
| 回退路径 | - → fallback调用非量化版本 |
| 执行模式 | 可能有overlap | 串行per-expert |

### 8.3 外部约束

```
参数约束范围:
  epWorldSize ∈ {2, 4, 8, 16, 32, 64, 128, 256}
  expertNum = ep × epWorldSize ≤ 256
  expertPerRank ∈ [1, 32]
  H1 < 65536
  N1 < 65536
  BsK < 52428800 (52M tokens)
  H2 ≤ 12288
  K = ep ∈ [2, 8]
  commQuantMode = 0 (当前唯一)
  所有张量 FORMAT_ND
  量化模式: gmm_x_quant + gmm_weight_quant 均需=1 (PerTensor)
```

---

*分析完成时间: 2026-04-21*
