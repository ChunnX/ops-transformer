# FlashInferAttention GQA NoQuant Kernel (arch35) 深度分析

> 作者：资深算子开发工程师
> 日期：2026-05-06
> 版本：v1.0

---

## 1. 概述

`fia_kernel_noquant_gqa.h` 是 FusedInferAttentionScore 算子在 Ascend NPU arch35 架构上的**非量化 GQA（Grouped Query Attention）核心 Kernel 模板**，采用 **AIC（Cube）+ AIV（Vector）异构流水** 架构，通过预加载任务队列实现计算与通信的深度流水化，最大化 NPU 计算单元利用率。

### 1.1 核心特征

| 特征 | 描述 |
|------|------|
| **架构代际** | arch35 (Ascend 910B/910C 系列) |
| **核类型** | AIC (Cube) + AIV (Vector) 异构协同 |
| **CV比例** | 1 AIC : 2 AIV (默认) |
| **计算精度** | 中间结果 float32，输入 FP16/BF16 |
| **预加载深度** | PRELOAD_N=2 (C1 C1 C1 C2) |
| **支持特性** | GQA、PSE、AttenMask、Rope、PageAttention、FlashDecode、SystemPrefix、LeftPadding |

### 1.2 文件索引

| 文件 | 行数 | 职责 |
|------|------|------|
| `attention/common/op_kernel/arch35/fia_kernel_noquant_gqa.h` | 834 | 主Kernel类：任务调度、流水编排 |
| `attention/common/op_kernel/arch35/fia_block_cube_noquant_gqa.h` | 943 | Cube子模块：Gm→L1搬运、BMM1/BMM2 |
| `attention/common/op_kernel/arch35/fia_block_vec_noquant_gqa.h` | 1540 | Vec子模块：Softmax、FlashUpdate、输出 |
| `attention/common/op_kernel/arch35/fia_block_vec_flashdecode.h` | 869 | FlashDecode：跨核归约后处理 |
| `attention/common/op_kernel/arch35/fia_template_dispatcher.h` | 74 | 模板展开、TilingKey→模板实例化 |
| `attention/common/op_host/arch35/fia_tiling_nonquant_gqa.cpp` | ~1100 | Host侧Tiling实现 |
| `attention/common/op_host/arch35/fia_tiling_nonquant_gqa.h` | 137 | Tiling类声明 |
| `attention/common/op_host/arch35/fia_tiling_nonquant_mla.h` | 146 | MLA模式Tiling（另一路径） |
| `attention/common/op_kernel/arch35/fia_tiling_data_noquant_gqa.h` | 44 | TilingData结构定义 |
| `attention/common/op_kernel/arch35/fia_public_define_arch35.h` | 271 | 公共数据结构(RunInfoX, ConstInfo等) |
| `attention/common/op_host/split_core_v2.h` | 264 | 分核算法接口 |
| `attention/common/op_kernel/buffer_manager.h` | 61 | L1/L0/UB内存池管理 |
| `attention/common/op_kernel/buffer_mix_core.h` | 216 | 跨核同步Buffer封装 |

---

## 2. 系统级架构总览

### 2.1 端到端数据流

```
Host (Tiling)                              Device (Kernel)
┌─────────────────────┐                   ┌──────────────────────────────┐
│ TilingInfo输入      │                   │ fia_template_dispatcher.h     │
│ ├─ Input shape       │                   │  └─ PARSE_PARAMS_NoQuant()   │
│ ├─ Layout/Config     │   ──TilingData──> │     └─ CubeBlock类型确定     │
│ ├─ Mask/Sparse info  │                   │     └─ VecBlock类型确定      │
│ └─ ActualSeqLens     │                   │     └─ Kernel实例化          │
└─────────────────────┘                   │     └─ op.Init() + Process() │
                                           └──────────────────────────────┘
                                                    │
                                    ┌───────────────┴───────────────┐
                                    │ FlashAttentionNoQuantGqaKernel │
                                    │        ::Process()             │
                                    └───────────────┬───────────────┘
                                           ┌────────┴────────┐
                                    FlashAttention()    FlashDecode()
                                    (AIC + AIV)         (AIV only)
```

### 2.2 三级流水线架构

```
          Tiling         →    Preload Queue    →    AIC/AIV Execute
     (Host, 编译期确定)      (Device, 运行时调度)     (NPU硬件计算)

  ┌──────────────┐       ┌──────────────────┐    ┌─────────────────┐
  │ SplitCore    │       │ taskRunInfo[0..2]│    │ AIC: CubeBlock  │
  │ 分核切分     │  ───> │ 3槽位环形队列    │───>│     BMM1/BMM2   │
  │ FAMetaData   │       │ PRELOAD_N=2      │    │ AIV: VecBlock   │
  │ TilingKey    │       │                  │    │     Softmax/Upd │
  └──────────────┘       └──────────────────┘    └─────────────────┘
```

---

## 3. Tiling 切分策略（Host侧）

### 3.1 基本块尺寸确定

Tiling 的入口是 `FiaTilingNonQuantArch35::DoOpTiling()`，位于 `fia_tiling_nonquant_gqa.cpp:105`。其中 `AdjustSinnerAndSouter()` 负责确定 S1 外循环基本块（mBaseSize）和 S2 内循环基本块（s2BaseSize）。

```cpp
// fia_tiling_nonquant_gqa.cpp:269-316
void FiaTilingNonQuantArch35::AdjustSinnerAndSouter()
{
    // 默认值
    sOuterFactor_ = SOUTER_64;   // AIC视角基本块 = 64 × CV_RATIO = 128
    sInnerFactor_ = SINNER_128;

    // 条件1: D≤128, S1≤64, S2>128, sparse预/后token和不超128, 且非量化
    if (fiaInfo_->vHeadDim <= DSIZE_128 && ...) {
        bool checkSparseMode = (sparseMode != 2 && preTokens + nextTokens > 128);
        if (checkDtype && checkQueryAndValueS && checkSparseMode) {
            sOuterFactor_ = SOUTER_32;   // 32 × 2 = 64
            sInnerFactor_ = SINNER_256;
        }
    }
    // 条件2: BSH/BSND/TND layout, D≤256
    else if (fiaInfo_->vHeadDim > DSIZE_128 && fiaInfo_->s1Size != 1) {
        if (fiaInfo_->vHeadDim <= DSIZE_256) {
            sOuterFactor_ = SOUTER_32;
            sInnerFactor_ = SINNER_256;
        } else {
            sOuterFactor_ = SOUTER_64;
            sInnerFactor_ = SINNER_128;
        }
    }
    // 条件3: S1==1 (IFA场景), D>128
    else if (fiaInfo_->s1Size == 1 && fiaInfo_->vHeadDim > DSIZE_128) {
        sOuterFactor_ = SOUTER_64;
        sInnerFactor_ = SINNER_128;
    }
}
```

**编译期映射为模板参数**：

```cpp
// fia_template_dispatcher.h:49-50
using CubBlock = BaseApi::FANoQuantGqaBlockCube<...,
    s1TemplateType,   // = mBaseSize = sOuterFactor × CV_RATIO
    s2TemplateType,   // = s2BaseSize = sInnerFactor
    dTemplateType,    // = Aligned DSize
    dVTemplateType    // = Aligned DVSize
>;
```

基本块尺寸选取表：

| 场景条件 | sOuterFactor | mBaseSize(AIC视角) | sInnerFactor | s2BaseSize |
|----------|:-----------:|:-----------------:|:------------:|:----------:|
| D≤128, S1≤64, S2>128, 稀疏范围≤128 | 32 | 64 | 256 | 256 |
| BSH/TND, D≤256 | 32 | 64 | 256 | 256 |
| D>128, S1==1 (IFA) | 64 | 128 | 128 | 128 |
| **默认（大部分场景）** | **64** | **128** | **128** | **128** |

### 3.2 分核算法 (split_core_v2)

分核算法位于 `split_core_v2.h`，通过 `SplitCore()` 函数将 BN2×GS1×S2 三维任务空间沿三级轴切分到多个 AIC 核。

#### 3.2.1 输入数据结构

```cpp
// split_core_v2.h:68-94
struct BaseInfo {
    uint32_t bSize;                       // Batch数
    uint32_t n2Size;                      // Head数 (N2 = num_heads / gSize for GQA)
    uint32_t gSize;                       // Group数
    uint32_t s1Size;                      // Q序列长度
    uint32_t s2Size;                      // KV序列长度
    bool isS1G;                           // S1G布局标记 (BSH/S1G为true)
    std::vector<int64_t> actualSeqS1Size; // 每个Batch的S1实际长度
    std::vector<int64_t> actualSeqS2Size; // 每个Batch的S2实际长度
    int64_t preToken;                     // 前置token数 (Mask用)
    int64_t nextToken;                    // 后置token数 (Mask用)
    // ...
};

struct SplitParam {
    uint32_t mBaseSize;       // S1G基本块大小 (= sOuterFactor × CV_RATIO)
    uint32_t s2BaseSize;      // S2基本块大小 (= sInnerFactor)
    bool streamK;             // 是否StreamK模式
};
```

#### 3.2.2 输出数据结构

```cpp
// split_core_v2.h:121-133
struct FAMetaData {
    uint32_t usedCoreNum;                  // 实际使用的AIC核数
    std::vector<uint32_t> bN2End;          // 每个核的BN2结束下标
    std::vector<uint32_t> mEnd;            // 每个核的GS1结束下标
    std::vector<uint32_t> s2End;           // 每个核的S2结束下标
    std::vector<uint32_t> firstFdDataWorkspaceIdx; // FD workspace偏移
    FlashDecodeResult fdRes;               // FlashDecode归约信息
};
```

#### 3.2.3 分核结果写入TilingData

分核结果通过 `SetSplitOutput()` 写入 `FusedInferAttentionScoreTilingData.fiaMetaData`，存储在 GM 上，Kernel 初始化时从 GM 读取。

```cpp
// fia_tiling_nonquant_gqa.cpp:502-539
void FiaTilingNonQuantArch35::SetSplitOutput(const split_core_v2::FAMetaData &splitRes)
{
    for (size_t i = 0; i < NPU_AIC_CORE_NUM; ++i) {
        tilingData_.fiaMetaData.FAMetadata[i][FA_CORE_ENABLE_INDEX] =
            (i < splitRes.usedCoreNum) ? 1 : 0;

        // 起始位置 (第i核): 前一个核的结束位置, 首个核从0开始
        tilingData_.fiaMetaData.FAMetadata[i][FA_BN2_START_INDEX] =
            (i == 0) ? 0 : splitRes.bN2End[i - 1];
        tilingData_.fiaMetaData.FAMetadata[i][FA_M_START_INDEX] =
            (i == 0) ? 0 : splitRes.mEnd[i - 1];
        tilingData_.fiaMetaData.FAMetadata[i][FA_S2_START_INDEX] =
            (i == 0) ? 0 : splitRes.s2End[i - 1];

        // 结束位置
        tilingData_.fiaMetaData.FAMetadata[i][FA_BN2_END_INDEX] = splitRes.bN2End[i];
        tilingData_.fiaMetaData.FAMetadata[i][FA_M_END_INDEX] = splitRes.mEnd[i];
        tilingData_.fiaMetaData.FAMetadata[i][FA_S2_END_INDEX] = splitRes.s2End[i];

        // FD workspace索引
        tilingData_.fiaMetaData.FAMetadata[i][FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] =
            splitRes.firstFdDataWorkspaceIdx[i];
    }
}
```

### 3.3 TilingKey 生成

Tiling计算结果通过 TilingKey 编码为编译期模板参数，实现**零开销分支**。

```cpp
// fia_tiling_nonquant_gqa.cpp:752-798
void FiaTilingNonQuantArch35::GenTilingKey()
{
    UpdateTilingKeyInfo();  // 填充 tilingKeyInfo_ 各字段

    tilingKey_ = GET_TPL_TILING_KEY(
        tilingKeyInfo_.inputLayout,     // BSH/TND/BNSD等
        tilingKeyInfo_.config,          // S1Aligned128_S2Aligned128_DAligned128_DVAligned128等
        tilingKeyInfo_.pseMode,         // PSE模式
        tilingKeyInfo_.quantMode,       // NoQuant
        tilingKeyInfo_.hasAttenMask,    // 使能AttenMask
        tilingKeyInfo_.hasRope,         // 使能Rope
        tilingKeyInfo_.kvLayoutType,    // KV存储布局 (BNSD/PA/TensorList)
        tilingKeyInfo_.isFd,            // FlashDecode使能
        tilingKeyInfo_.emptyTensor,     // 空tensor标记
        tilingKeyInfo_.maskMode,        // Mask模式
        tilingKeyInfo_.matmulMode,      // MatMul模式
        tilingKeyInfo_.enableKvPrefix,  // KV前缀使能
        tilingKeyInfo_.enableS1OutSplit // S1外切分使能
    );
}
```

---

## 4. 模板调度与实例化

### 4.1 dispatcher 入口

```cpp
// fia_template_dispatcher.h:21-71
template <typename INPUT_T, typename OUT_T,
          uint8_t inOutLayoutType, uint16_t config, uint8_t pseMode,
          uint8_t quantMode, bool hasAttenMask, bool hasRope,
          uint8_t KvLayoutType, bool isFd, bool emptyTensor, ...>
inline __aicore__ void run_fia_noquant_gqa_kernel(
    __gm__ uint8_t *query, __gm__ uint8_t *key, /* ... 所有输入指针 ... */,
    __gm__ uint8_t *tiling)
{
    // 1. 解析TilingKey → 各模板参数枚举值
    PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, ...);

    // 2. 确定BMM2输出位置 (UB或GM)
    constexpr TPosition bmm2OutPos = GetC2Position(/*...*/);
    constexpr bool bmm2Write2Ub = bmm2OutPos == TPosition::VECCALC;

    // 3. 实例化Cube/Vec Block
    using CubBlock = BaseApi::FANoQuantGqaBlockCube<..., bmm2Write2Ub, ...>;
    using VecFaBlock = BaseApi::FANoQuantGqaBlockVec<..., bmm2Write2Ub, ...>;
    using VecFdBlock = BaseApi::FiaBlockVecFlashDecode<...>;
    using Kernel = FlashAttentionNoQuantGqaKernel<CubBlock, VecFaBlock, VecFdBlock>;

    // 4. 提取TilingData
    GET_TILING_DATA_MEMBER(FusedInferAttentionScoreTilingData, baseTiling, baseTilingIn, tiling);
    const NoQuantTilingArch35 *__restrict tilingData = &baseTilingIn;
    __gm__ uint8_t *fiaMetaData =
        tiling + offsetof(FusedInferAttentionScoreTilingData, fiaMetaData);

    // 5. Init + Process
    Kernel op;
    op.Init(query, key, /*...*/ fiaMetaData, tilingData, &tPipe);
    op.Process();
}
```

**关键设计**：所有条件分支（layout、D size、bmm2输出位置、dn模式等）均在编译期展开为模板参数，Kernel运行时零分支开销。

---

## 5. 主Kernel流程 (FlashAttentionNoQuantGqaKernel)

### 5.1 类层次结构

```cpp
// fia_kernel_noquant_gqa.h:40-831
template <typename CubeBlockType, typename VecFaBlockType, typename VecFdBlockType>
class FlashAttentionNoQuantGqaKernel {
    // === 编译期常量 (从Cube/Vec Block提取) ===
    static constexpr uint32_t mBaseSize = CubeBlockType::mBaseSize;
    static constexpr uint32_t s2BaseSize = CubeBlockType::s2BaseSize;
    static constexpr uint32_t PRELOAD_N = 2;    // Preload深度
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 1; // 3个槽位

    // === Buffer管理 ===
    BufferManager<BufferType::GM>  gmBufferManager;
    BufferManager<BufferType::UB>  ubBufferManager;
    BufferManager<BufferType::L1>  l1BufferManager;
    BuffersPolicy3buff<..., L1>    l1PBuffers;     // L1上的P矩阵(3buffer轮转)
    BuffersPolicyDB<..., UB>       bmm1Buffers;    // UB上BMM1结果(DoubleBuffer)
    BuffersPolicyDB<..., UB>       bmm2Buffers;    // UB上BMM2结果(DoubleBuffer)
    BuffersPolicy3buff<..., GM>    bmm2ResGmBuffers; // GM上BMM2结果(3buffer轮转)

    // === 子模块引用 ===
    CubeBlockType cubeBlock;
    VecFaBlockType vecFaBlock;
    VecFdBlockType vecFdBlock;
    const NoQuantTilingArch35 *__restrict tilingData;
    TPipe *pipe;
};
```

### 5.2 初始化流程 (`Init`)

```cpp
// fia_kernel_noquant_gqa.h:124-186
__aicore__ inline void Init(/*...所有输入指针...*/)
{
    // Step 1: 设置 tilingData 指针和 fiaMetaDataGm (全局对象)
    this->pipe = tPipe;
    this->tilingData = tiling;
    fiaMetaDataGm.SetGlobalBuffer((__gm__ uint32_t *)fiaMetaData,
        NPU_AIC_CORE_NUM * FA_METADATA_SIZE + NPU_AIV_CORE_NUM * FD_METADATA_SIZE);

    // Step 2: 从TilingData填充全部ConstInfo (轴长度、Mask参数、PSE参数等)
    InitConstInfo(queryPaddingSize, kvPaddingSize, actualSharedPrefixLen);

    // Step 3: 初始化Q/KV实际序列长度解析器
    actualSeqLengthsGmQ.SetGlobalBuffer(...);
    qActSeqLensParser.Init(actualSeqLengthsGmQ, ...);
    actualSeqLengthsGm.SetGlobalBuffer(...);
    kvActSeqLensParser.Init(actualSeqLengthsGm, ...);

    // Step 4: 初始化矩阵乘结果Buffer (UB/L1/GM)
    InitMMResBuf(workspace);

    // Step 5: AIV初始化Vec子模块
    if ASCEND_IS_AIV {
        vecFaBlock.InitVecBlock(tPipe);
        vecFaBlock.InitVecInput(/*...*/);
        vecFaBlock.ClearOutput();  // 初始化输出为0
    }

    // Step 6: AIC初始化Cube子模块
    if ASCEND_IS_AIC {
        cubeBlock.InitCubeBlock(tPipe, &l1BufferManager);
        cubeBlock.InitCubeInput(query, key, value, /*...*/);
    }

    // Step 7: FlashDecode初始化 (仅AIV)
    if constexpr (FLASH_DECODE) {
        if ASCEND_IS_AIV {
            vecFdBlock.InitParams();
            vecFdBlock.InitGlobalTensor(/*...*/);
        }
    }
}
```

### 5.3 主循环：任务调度与预加载流水

这是整个Kernel最核心的调度逻辑：

```cpp
// fia_kernel_noquant_gqa.h:397-443
__aicore__ inline void FlashAttention()
{
    // 边界检查：当前核是否被使能
    if (constInfo.aicIdx >= constInfo.coreNum) return;

    RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE]; // 3个槽位
    uint32_t bN2Cur = constInfo.bN2Start;  // 当前BN2坐标
    uint32_t gS1Cur = constInfo.gS1OStart; // 当前GS1坐标
    uint32_t s2Cur = constInfo.s2OStart;   // 当前S2坐标

    if (enableS1OutSplit) {
        ClacS1OutSplitLoopTimes();  // S1外切分需预计算循环次数
    }

    bool shouldDispatchTask = true;
    bool shouldExecuteTask = false;
    while (shouldDispatchTask || shouldExecuteTask) {
        // === 分发阶段 ===
        shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
        if (shouldDispatchTask) {
            TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
            if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                CreateTask(createdTaskCount, bN2Cur, gS1Cur, s2Cur, taskRunInfo);
                createdTaskCount++;
                UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
            }
            else if (taskDealMode == TASK_DEAL_MODE::DEAL_ZERO) {
                vecFaBlock.DealZeroActSeqLen(bN2Cur, gS1Cur, s2Cur);
                UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                continue;  // 不占用执行槽位
            }
        }

        // === 执行阶段 ===
        shouldExecuteTask = ShouldExecuteTask(taskRunInfo);
        if (shouldExecuteTask) {
            ExecuteTask(executedTaskCount, taskRunInfo);
            executedTaskCount++;
        }
    }
}
```

#### 5.3.1 预加载流水时序

```
时间轴 →
    ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
调度 │C0 │C1 │C2 │C3 │C4 │C5 │C6 │C7 │   │   │  CreateTask
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
Cube │   │   │M1 │M2 │M3 │M4 │M5 │M6 │M7 │   │  ComputeMm1 + ComputeMm2
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
Vec  │   │   │   │   │V1 │V2 │V3 │V4 │V5 │V6 │  ComputeVec1 + ComputeVec2
    └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘

    PRELOAD_N = 2:
      - C0完成后2轮 (C0→M0), M2才对应V0 (C0→M0→V0 延迟2轮)
      - 写入 taskRunInfo[loop % 3]
      - 执行时, C_L: dispatch第L个, M_(L-2): cube第L-2轮, V_(L-4): vec第L-4轮
```

#### 5.3.2 执行轮次详解 (`ExecuteTask`)

```cpp
// fia_kernel_noquant_gqa.h:619-641
__aicore__ inline void ExecuteTask(uint64_t loop,
                                    RunInfoX taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
{
    RunInfoX &runInfo0 = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE];
    RunInfoX &runInfoNegN =
        taskRunInfo[(loop - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE];

    // 当前轮: BMM1或Vec1
    if (runInfo0.isValid) {
        if ASCEND_IS_AIC {
            ComputeMm1(runInfo0);   // Cube: Q×K^T
        } else {
            ComputeVec1(runInfo0);  // Vec: Scale+Mask+Softmax
        }
    }

    // 前PRELOAD_N轮: BMM2或Vec2
    if (loop >= PRELOAD_N) {
        if (runInfoNegN.isValid) {
            if ASCEND_IS_AIC {
                ComputeMm2(runInfoNegN);  // Cube: P×V
            } else {
                ComputeVec2(runInfoNegN); // Vec: FlashUpdate+Output
            }
            runInfoNegN.isValid = false;  // 标记已消费
        }
    }
}
```

**流水对应关系**：
```
loop=0: M0(C0)  + V-2(N/A)
loop=1: M1(C1)  + V-1(N/A)
loop=2: M2(C2)  + V0(C0)  ← 开始并行
loop=3: M3(C3)  + V1(C1)
loop=4: M4(C4)  + V2(C2)
...
```

### 5.4 任务创建 (`CreateTask → CalcParams`)

每个任务的 `RunInfoX` 包含执行所需的全部信息：

```cpp
// fia_public_define_arch35.h:71-103
struct RunInfoX {
    uint32_t loop;               // 全局任务编号
    uint32_t mloop;              // S1G外循环编号 (同一GS1的S2数)
    bool isValid;                // 任务有效性
    bool isFirstS2Loop;          // 是否某GS1行的首个S2循环 (决定是否搬运Q)
    bool isLastS2Loop;           // 是否某GS1行的末个S2循环 (决定是否写LSE)

    uint32_t bIdx;               // Batch索引
    uint32_t n2Idx;              // Head(N2)索引
    uint32_t gS1Idx;             // GS1拼接轴起始
    uint32_t s2Idx;              // S2轴起始

    uint64_t actS1Size;          // 当前head S1实际大小
    uint64_t actS2Size;          // 当前head S2实际大小
    uint32_t actMSize;           // 有效M轴长度 (= min(mBaseSize, 剩余))
    uint32_t actSingleLoopS2Size;// 有效S2长度 (= min(s2BaseSize, 剩余))
    uint32_t actVecMSize;        // VEC视角M轴长度 (subBlock拆分后)
    uint32_t vecMbaseIdx;        // VEC对应M轴起始位置
    bool isS2SplitCore;          // S2跨核切分标记
    uint32_t faTmpOutWsPos;      // FD中间结果workspace位置索引
};
```

关键计算逻辑 (`CalcParams`)：

```cpp
// fia_kernel_noquant_gqa.h:679-765
__aicore__ inline void CalcParams(uint64_t loop, uint32_t bN2Cur,
                                   uint32_t gS1Cur, uint32_t s2Cur,
                                   RunInfoX &info)
{
    // 1. 从BN2→(bIdx, n2Idx) 解码
    info.bIdx = bN2Cur / constInfo.n2Size;
    info.n2Idx = bN2Cur % constInfo.n2Size;

    // 2. GS1起始位置编码
    info.gS1Idx = gS1Cur * mBaseSize;   // GS1拼接轴地址

    // 3. 从GS1地址解码为(s1Idx, gIdx) — 取决于Layout
    if constexpr (LAYOUT_Q == LAYOUT_BSH) { // S1G layout
        info.s1Idx = info.gS1Idx / constInfo.gSize;
    } else {                                // GS1 layout
        info.s1Idx = info.gS1Idx % actSeqLensQ;
    }

    // 4. S2起始位置
    info.s2Idx = s2Cur * s2BaseSize;

    // 5. 有效M轴长度 (尾块处理)
    info.actMSize = mBaseSize;
    if (((gS1Cur + 1) * mBaseSize) > (actSeqLensQ * constInfo.gSize)) {
        info.actMSize = actSeqLensQ * constInfo.gSize - gS1Cur * mBaseSize;
    }

    // 6. 有效S2长度 (尾块处理)
    info.actSingleLoopS2Size = s2BaseSize;
    if (((s2Cur + 1) * s2BaseSize) > actSeqLensKv) {
        info.actSingleLoopS2Size = actSeqLensKv - s2Cur * s2BaseSize;
    }

    // 7. VEC视角M轴拆分 (CV_RATIO=2时, V0和V1各处理一半)
    info.actVecMSize = (info.actMSize + 1) >> 1; // 一半
    info.vecMbaseIdx = 0;
    if (constInfo.subBlockIdx == 1) {   // V1处理另一半
        info.vecMbaseIdx = info.actVecMSize;
        info.actVecMSize = info.actMSize - info.actVecMSize;
    }

    // 8. S2跨核切分判断
    // isS2SplitCore: 当前任务的S2轴被跨核切分, 需要FlashDecode归约

    // 9. Sparse模式下 preTokens/nextTokens 计算
    GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv,
                          info.preTokensLeftUp, info.nextTokensLeftUp);
}
```

### 5.5 任务有效性判断 (`GetTaskDealMode`)

```cpp
// fia_kernel_noquant_gqa.h:467-520
__aicore__ inline TASK_DEAL_MODE GetTaskDealMode(
    uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
{
    // 1. Batch切换时更新actSeqLens
    uint32_t bIdx = bN2Cur / constInfo.n2Size;
    if (isFirstTask || prevBIdx != bIdx) {
        actSeqLensKv = kvActSeqLensParser.GetActualSeqLength(bIdx)
                       + constInfo.actualKVPrefixSize;
        actSeqLensQ = qActSeqLensParser.GetActualSeqLength(bIdx);
    }

    // 2. 零序列长度 → 结果置零
    uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
    uint64_t gS1LoopTimes =
        (actSeqLensQ * constInfo.gSize + mBaseSize - 1) / mBaseSize;
    if (s2LoopTimes == 0 || gS1LoopTimes == 0) {
        return (gS1Cur == 0 && s2Cur == 0) ? DEAL_ZERO : SKIP;
    }

    // 3. Padding越界检查
    if ((isQHasLeftPadding && actSeqLensQ + queryRightPaddingSize > s1Size) ||
        (isKVHasLeftPadding && actSeqLensKv + kvRightPaddingSize > s2Size)) {
        return DEAL_ZERO;
    }

    // 4. 计算有效S2范围 (Mask/Sparse剪枝)
    if constexpr (!HAS_MASK) {
        CalcCurS2StartEndNoSparse(bN2Cur, gS1Cur);
    } else {
        CalcCurS2StartEndWithSparse(bN2Cur, gS1Cur);
    }

    // 5. S2范围外 → 跳过
    if (s2Cur < curS2Start || s2Cur >= curS2End) return SKIP;

    return CREATE_TASK;
}
```

### 5.6 轴遍历顺序 (`UpdateAxisInfo`)

```cpp
// fia_kernel_noquant_gqa.h:767-799
__aicore__ inline void UpdateAxisInfo(TASK_DEAL_MODE taskDealMode,
                                       uint32_t &bN2Cur, uint32_t &gS1Cur,
                                       uint32_t &s2Cur)
{
    // S2+1: 内循环推进
    if (s2Cur + 1 < s2LoopTimes) {
        s2Cur++;
        return;
    }

    // S2走完 → GS1+1, S2归零
    s2Cur = 0;
    if (gS1Cur + 1 < gS1LoopTimes) {
        gS1Cur++;
        return;
    }

    // GS1走完 → BN2+1, GS1归零
    gS1Cur = 0;
    bN2Cur++;
}
```

**遍历顺序**：`BN2 (粗) → GS1 (中) → S2 (细)`

---

## 6. Cube 子模块 (FANoQuantGqaBlockCube)

### 6.1 L1 空间分配

```cpp
// fia_block_cube_noquant_gqa.h:156-174
__aicore__ inline void InitLocalBuffer(TPipe *tPipe)
{
    if constexpr (dBaseSize > 256) {
        // D>256: L1上切K轴, Q全量驻留(BMM1), V全量驻留(BMM2)
        constexpr uint32_t mm1LeftSize  = mBaseSize * dBaseSize * sizeof(Q_T);
        constexpr uint32_t mm1RightSize = s2BaseSize * 128 * sizeof(KV_T);
        constexpr uint32_t mm2RightSize = s2BaseSize * 128 * sizeof(KV_T);
        l1QBuffers.Init((*l1BufferManagerPtr), mm1LeftSize);
        l1KBuffers.Init((*l1BufferManagerPtr), mm1RightSize);
        l1VBuffers.Init((*l1BufferManagerPtr), mm2RightSize);
    } else {
        // D≤256: L1全量加载Q/K/V
        constexpr uint32_t mm1LeftSize  = mBaseSize * dBaseSize * sizeof(Q_T);
        constexpr uint32_t mm1RightSize = dBaseSize * s2BaseSize * sizeof(KV_T);
        constexpr uint32_t mm2RightSize = dVBaseSize * s2BaseSize * sizeof(KV_T);
        l1QBuffers.Init((*l1BufferManagerPtr), mm1LeftSize);
        l1KBuffers.Init((*l1BufferManagerPtr), mm1RightSize);
        l1VBuffers.Init((*l1BufferManagerPtr), mm2RightSize);
    }
    // Total L1 ≤ 512KB (524288 B)
}
```

**L1 Buffer类型选择（编译期）**：

```cpp
// fia_block_cube_noquant_gqa.h:41-52
// Q缓冲区: D>256用SingleBuffer, 否则用DoubleBuffer
template <uint32_t dBaseSize>
struct QL1BuffSel {
    using Type = std::conditional_t<(dBaseSize > 256),
        BuffersPolicySingleBuffer<BufferType::L1>,
        BuffersPolicyDB<BufferType::L1>>;
};

// KV缓冲区: s2BaseSize=256且D>128用SingleBuffer
template <uint32_t s2BaseSize, uint32_t dBaseSize>
struct KVL1BuffSel {
    using Type = std::conditional_t<(s2BaseSize == 256 && dBaseSize > 128),
        BuffersPolicySingleBuffer<BufferType::L1>,
        BuffersPolicyDB<BufferType::L1>>;
};
```

### 6.2 L0A/B/C 空间分配

```cpp
// fia_block_cube_noquant_gqa.h:177-189
// L0A: 64KB → 32KB DoubleBuffer
l0aBufferManager.Init(tPipe, 65536);
mmL0ABuffers.Init(l0aBufferManager, 32 * 1024);

// L0B: 64KB → 32KB DoubleBuffer
l0bBufferManager.Init(tPipe, 65536);
mmL0BBuffers.Init(l0bBufferManager, 32 * 1024);

// L0C: 256KB → 条件选择 4buffer(64KB each) 或 DB(128KB each)
l0cBufferManager.Init(tPipe, 262144);
if constexpr (mBaseSize * s2BaseSize * 4 <= L0C_SIZE/4 &&
              mBaseSize * dVBaseSize * 4 <= L0C_SIZE/4) {
    mmL0CBuffers.Init(l0cBufferManager, L0C_SIZE/4 * 1024); // 4 buffer
} else {
    mmL0CBuffers.Init(l0cBufferManager, L0C_SIZE/2 * 1024); // Double buffer
}
```

### 6.3 BMM1 = Q × K^T (三条计算路径)

#### 路径 1: D>256 → `IterateBmm1NdL1SplitK`

```cpp
// fia_block_cube_noquant_gqa.h:613-677
// 针对 D>256 场景：Q全量驻留L1(SingleBuffer), K每次搬运128列
__aicore__ inline void IterateBmm1NdL1SplitK(
    MM1_DBUF_T &outputBuf, RunInfoX &runInfo)
{
    constexpr uint32_t baseK = 128;  // 每轮加载128列K
    uint32_t kLoops = (constInfo.dSize + baseK - 1) / baseK;

    // 循环K轴切片
    for (uint32_t k = 0; k < kLoops; k++) {
        realK = (k == kLoops - 1) ? dSize % baseK : baseK;

        // 搬运Q切片（仅首轮，后续Q驻留）
        if (unlikely(runInfo.isFirstS2Loop)) {
            CopyQuerySlice(mm1ATensor[k * l1BaseKOffset],
                           k * baseK, realK, runInfo);
        }

        // 搬运K切片
        mm1B = l1KBuffers.Get();
        CopyKeySlice(mm1BTensor, k * baseK, realK, runInfo);

        // MatMul Q[:, k*128:(k+1)*128] × K[(k*128:(k+1)*128), :]
        MatmulFull<..., 128, 128, baseK, ...>(
            mm1A[k * l1BaseKOffset], mm1B, ...);
    }
}
```

**L1布局示意**：
```
L1 Q:  [mBaseSize × dBaseSize]  ← 全量驻留 (SingleBuffer)
L1 K:  [s2BaseSize × 128]       ← 每次搬运 (DoubleBuffer, 释放复用)
```

#### 路径 2: useDn模式 → `IterateBmm1Dn` / `IterateBmm1DnSplitK`

Dn模式（转置模式）：K作为左矩阵，Q作为右矩阵，在L0上切D。

```cpp
// fia_block_cube_noquant_gqa.h:680-727 (SplitK) / 743-787 (全载)
// 关键差异: 矩阵角色互换
MatmulK<KV_T, Q_T, T, 128, 128, 128, ABLayout::MK, ABLayout::KN>(
    mm1A.GetTensor<KV_T>(),    // K作左矩阵
    mm1B.GetTensor<Q_T>(),     // Q作右矩阵
    ...);
// Fixpipe 使用 dualDstCtl=2 (沿N方向拆分)
```

#### 路径 3: 默认路径 → `IterateBmm1NdL0Split`

```cpp
// fia_block_cube_noquant_gqa.h:549-610
// D≤256: Q全量驻留L1, K全量搬运L1, L0上切D
__aicore__ inline void IterateBmm1NdL0Split(
    MM1_DBUF_T &outputBuf, RunInfoX &runInfo)
{
    // 1. 获取Q (首轮搬运, 后续复用Pre)
    Buffer<BufferType::L1> mm1A;
    if (unlikely(runInfo.isFirstS2Loop)) {
        mm1A = l1QBuffers.Get();
        CopyQueryTile(mm1ATensor, runInfo); // GM→L1搬运Q
        mm1A.Set<HardEvent::MTE2_MTE1>();   // 通知搬运完成
    } else {
        mm1A = l1QBuffers.GetPre();         // 复用上一轮的Q
    }

    // 2. 搬运K到L1
    Buffer<BufferType::L1> mm1B = l1KBuffers.Get();
    CopyKeyTile(mm1BTensor, runInfo);       // GM→L1搬运K

    // 3. 等待L1 Q/K就绪
    mm1A.Wait<HardEvent::MTE2_MTE1>();
    mm1B.Wait<HardEvent::MTE2_MTE1>();

    // 4. MatMul on L0
    Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
    if constexpr (dBaseSize > 128) {
        MatmulK<..., 128, 128, 128, ...>(
            mm1A, mm1B, mmL0ABuffers, mmL0BBuffers, mm1ResL0C, param);
    } else {
        MatmulBase<..., 128, 128, dBaseSize, ...>(
            mm1A, mm1B, ...);
    }

    // 5. Fixpipe L0C→UB (NZ→ND转排)
    FixpipeMm1(outputBuf.template GetTensor<T>(),
               mm1ResL0C.GetTensor<T>(), runInfo);

    // 6. 跨核通知AIV
    outputBuf.SetCrossCore();
}
```

**同步事件链（默认BMM1）**：

```
MTE2(GM→L1搬运K) ──┐
                   ├──> Wait MTE2_MTE1 ──> MatMul ──> M_FIX ──> FIX_M
MTE1(GM→L1搬运Q) ──┘                                           │
                                                                ↓
                                                          Wait M_FIX
                                                                │
                                                                ↓
                                                          Fixpipe L0C→UB
                                                                ↓
                                                          SetCrossCore(AIV)
```

### 6.4 BMM2 = P × V

#### 路径选择

```cpp
// fia_block_cube_noquant_gqa.h:789-802
__aicore__ inline void IterateBmm2(
    mm2ResPos &outputBuf, MM2_ABUF_POLICY_T &inputBuf, RunInfoX &runInfo)
{
    if constexpr (dVTemplateType > 256 || dTemplateType > 256) {
        IterateBmm2L1SplitN(outputBuf, inputBuf, runInfo);
    } else {
        IterateBmm2l0Split(outputBuf, inputBuf, runInfo);
    }
}
```

#### 默认路径: `IterateBmm2l0Split`

```cpp
// fia_block_cube_noquant_gqa.h:885-938
__aicore__ inline void IterateBmm2l0Split(
    mm2ResPos &outputBuf, MM2_ABUF_POLICY_T &inputBuf, RunInfoX &runInfo)
{
    // 1. 等待P矩阵在L1就绪 (由AIV Vec1写入)
    Buffer<...> mm2A = inputBuf.Get();
    mm2A.WaitCrossCore();

    // 2. 搬运V到L1
    Buffer<...> mm2B = l1VBuffers.Get();
    CopyValueTile(mm2BTensor, runInfo);

    // 3. MatMul P × V
    Buffer<...> mm2ResL0C = mmL0CBuffers.Get();
    MatmulN<..., 128, 128, s2BaseSize, ...>(mm2A, mm2B, ...);

    // 4. Fixpipe
    FixpipeMm2PartialN(outputBuf, mm2ResL0C, constInfo.dSizeV, runInfo);

    // 5. 跨核通知AIV
    outputBuf.SetCrossCore();
}
```

#### BMM2结果存储位置决策

```cpp
// fia_template_dispatcher.h:35-47
constexpr TPosition bmm2OutPos = GetC2Position(/*...*/);
constexpr bool bmm2Write2Ub = bmm2OutPos == TPosition::VECCALC;

// bmm2Write2Ub=true:  BMM2结果 Fixpipe→UB → Vec直接读取
// bmm2Write2Ub=false: BMM2结果 Fixpipe→GM workspace → Vec通过DataCopy读取
```

### 6.5 Rope 拼接搬运

```cpp
// fia_block_cube_noquant_gqa.h:282-323 (Query with Rope)
__aicore__ inline void CopyQuerySlice(
    const LocalTensor<Q_T> &dstTensor, uint32_t dOffset,
    uint32_t dRealSize, RunInfoX &runInfo)
{
    // 计算Rope与非Rope的交界点
    if constexpr (hasRope) {
        // dOffset ≤ dSize ≤ dOffset+dRealSize 时存在Rope跨越
        nopeDealSize = dSize - dOffset;       // 非Rope部分
        ropeDealSize = dOffset + dRealSize - Align(dSize, 16U); // Rope部分
    }

    // 搬运非Rope部分: 从 queryGm 搬运
    if (nopeDealSize > 0) {
        copyQueryGmToL1(l1Tensor, queryGm, gmCoord);
    }

    // 搬运Rope部分: 从 queryRopeGm 搬运 (追加到nope后)
    if (ropeDealSize > 0) {
        uint32_t offset = Align(nopeDealSize, 16U) * Align(runInfo.actMSize, 16U);
        copyQueryGmToL1(l1Tensor, queryRopeGm, gmCoord_rope);
    }
}
```

**Rope内存布局示意**：
```
L1上的Q行:
┌──────────────────────┬─────────────────┐
│  dSize列 (nope部分)   │  dSizeRope列     │
│  (从queryGm搬运)      │  (从queryRopeGm) │
└──────────────────────┴─────────────────┘
EFFECTIVE D = dSize + dSizeRope (MatMul使用全量)
```

### 6.6 SystemPrefix 拼接待搬运

```cpp
// fia_block_cube_noquant_gqa.h:416-459
template <bool VALUE = false>
__aicore__ inline void CopyKVSliceWithPrefix(
    const LocalTensor<KV_T> &dstTensor, uint32_t dOffset,
    uint32_t dRealSize, RunInfoX &runInfo)
{
    // 计算当前S2块中Prefix与Normal KV的重叠区间
    uint32_t prefixDealSize =
        (actualKVPrefixSize > s2StartIdx) ? (actualKVPrefixSize - s2StartIdx) : 0;
    uint32_t normalDealSize =
        (s2EndIdx > actualKVPrefixSize) ? (s2EndIdx - actualKVPrefixSize) : 0;

    // 搬运Prefix部分
    if (prefixDealSize > 0) {
        copyKvGmToL1(l1Tensor, valuePrefixGm, gmCoord);  // 从prefix tensor搬运
    }

    // 搬运Normal KV部分
    if (normalDealSize > 0) {
        l1Tensor.tensor = dstTensor[offset_after_prefix];
        copyKvGmToL1(l1Tensor, valueGm, gmCoord_normal);  // 从主KV tensor搬运
    }
}
```

**Prefix拼接后的L1布局**：
```
S2轴→
┌─────────────────┬─────────────────────┐
│ Prefix (s2条)    │ Normal KV (s2BaseSize - s2条) │
│ 从keyPrefixGm    │ 从keyGm             │
└─────────────────┴─────────────────────┘
```

---

## 7. Vec 子模块 (FANoQuantGqaBlockVec)

### 7.1 UB 空间分配

```cpp
// fia_block_vec_noquant_gqa.h:1249-1327
__aicore__ inline void InitLocalBuffer()
{
    // === Softmax相关 (256B × 3 × 3 = 2.25KB) ===
    SoftmaxInitBuffer();  // max[3] + sum[3] + exp[3], each 256B

    // === Stage1输出Queue (BMM1结果→L1中转) ===
    if constexpr (s2BaseSize == 256 && mBaseSize == 128) {
        tPipe->InitBuffer(stage1OutQue[0], 1, 33024); // ~32KB
        tPipe->InitBuffer(stage1OutQue[1], 1, 33024);
    } else {
        tPipe->InitBuffer(stage1OutQue[0], 1, 16896); // ~16.5KB
        tPipe->InitBuffer(stage1OutQue[1], 1, 16896);
    }

    // === Stage2结果Buffer (Vec2累加结果) ===
    tPipe->InitBuffer(stage2OutBuf, 64 * dTemplateAlign64 * sizeof(T));

    // === AttenMask Queue (8KB × 2) ===
    if constexpr (hasAtten) {
        tPipe->InitBuffer(attenMaskInQue[0], 1, 8192);
        tPipe->InitBuffer(attenMaskInQue[1], 1, 8192);
    }

    // === mm2InBuf (仅BMM2在GM时, 从GM→UB搬运用) ===
    if constexpr (!bmm2Write2Ub) {
        tPipe->InitBuffer(mm2InBuf, 32768); // 32KB
    }

    // === 事件ID分配 ===
    mte3ToVId[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    vToMte3Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
}
```

**UB内存池布局概览**：

```
UB (上界):
├── stage2OutBuf         (64 × dTemplateAlign64 × sizeof(T))
├── stage1OutQue[0..1]   (16896~33024 B × 2)
├── attenMaskInQue[0..1] (8192 B × 2, 可选)
├── mm2InBuf             (32768 B, BMM2在GM时)
├── softmaxMaxBuf[0..2]  (256 B × 3)
├── softmaxSumBuf[0..2]  (256 B × 3)
├── softmaxExpBuf[0..2]  (256 B × 3)
├── commonTBuf           (512 B)
├── maxBrdcst + sumBrdcst (FD用)
├── lseTmpBuff + softmaxLseQueue (LSE用)
└── sinkQue + postQuantXxxQue (可选)
```

### 7.2 Vec1 阶段 (Scale + Mask + Softmax)

#### Nd 路径（默认, `ProcessVec1Nd`）

```cpp
// fia_block_vec_noquant_gqa.h:535-687
__aicore__ inline void ProcessVec1Nd(
    Buffer<..., L1> &outputBuf, Buffer<..., UB> &bmm1ResBuf,
    RunInfoX runInfo)
{
    // 1. 等待AIC BMM1结果就绪
    bmm1ResBuf.WaitCrossCore();

    // 2. 准备PSE数据 (Optional)
    if constexpr (hasPse) {
        pseUb = this->pseInQue.DeQue<pseShiftType>();
        copyPSEGmToUb(pseShiftUbTensor, pseShiftGmTensor, pseCoord);
    }

    // 3. 拷贝AttenMask到UB (Optional)
    if constexpr (hasAtten) {
        attenMaskUb = this->attenMaskInQue[...].AllocTensor<uint8_t>();
        AttentionmaskCopyIn(attenMaskUb, attenMaskUbPre, attenMaskGmInt, maskInfo);
    }

    // 4. 主计算: Scale + AttenMask + PSE + Softmax
    //    根据actSingleLoopS2Size分派到不同模板:
    //    EQ_128 / GT_0_AND_LTE_64 / GT_64_AND_LTE_128 / GT_128_AND_LTE_256
    ProcessVec1Vf<..., s2SizeBranch>(stage1CastTensor,
        vselrIndexesBuf, sumUb, maxUb, mmRes, expUb,
        attenMaskUb, pseUb, dropMaskUb,
        apiTmpBuffer, actVecMSize, actSingleLoopS2Size,
        scaleValue, negativeFloatScalar);

    // 5. DataCopy UB→L1 (写入P矩阵供BMM2使用)
    DataCopy(mm2AL1Tensor[vecBaseIdx * stride],
             stage1CastTensor,
             {blockCount, dealSize, srcStride, dstStride});

    // 6. 跨核通知AIC: P矩阵就绪
    outputBuf.SetCrossCore();

    // 7. UpdateExpSumAndExpMax (非首轮S2时更新Running Max/Sum)
    if (!runInfo.isFirstS2Loop) {
        UpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, ...);
    }

    // 8. 末轮S2: 写出LSE / Sink / FD中间结果
    if (unlikely(runInfo.isLastS2Loop)) {
        SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
    }
}
```

**DataCopy 参数含义**：
```cpp
DataCopy(mm2AL1Tensor[offset], stage1CastTensor,
    {blockCount, dealSize, srcStride, dstStride});
//   blockCount: s2BaseSize / 16 个连续块
//   dealSize:   actVecMSize 个元素/块
//   srcStride:  源UB上两行间隔 = vec1Srcstride - actVecMSize
//   dstStride:  目标L1上两行间隔 = mBaseSize - actVecMSize
```

#### Dn 路径 (`ProcessVec1Dn`)

与Nd路径的关键差异：
- Softmax 输入矩阵为 K×M（转置），而非 M×K
- Fixpipe 使用 `dualDstCtl=2`（沿N方向拆分）
- DataCopy 到 L1 时按 subBlock 偏移写入

### 7.3 Vec2 阶段 (FlashUpdate + 输出)

#### 路径 1: BMM2结果在UB (`ProcessVec2OnUb`)

```cpp
// fia_block_vec_noquant_gqa.h:689-729
__aicore__ inline void ProcessVec2OnUb(
    Buffer<..., UB> &bmm2ResBuf, RunInfoX runInfo)
{
    bmm2ResBuf.WaitCrossCore();

    LocalTensor<T> mmRes = bmm2ResBuf.GetTensor<T>();

    if (unlikely(runInfo.isFirstS2Loop)) {
        // 首轮S2: 直接拷贝BMM2结果
        DataCopy(vec2ResUb, mmRes, vec2CalcSize);
    } else {
        // 非首轮: 累积更新
        // vec2Res = exp × vec2Res + bmm2Res
        FlashUpdateNew<T, ...>(vec2ResUb, mmRes, vec2ResUb,
                               expUb, actVecMSize, dTemplateAlign64);
    }

    bmm2ResBuf.SetCrossCore();

    if (runInfo.isLastS2Loop) {
        if (isFirstS2Loop) {
            // 仅一轮S2 → 直接除sum
            LastDivNew(vec2ResUb, sumUb, actVecMSize, dTemplateAlign64);
        }
        // 最终输出: Cast + PostQuant + RowInvalid + CopyOut
        CopyOutAttentionOut(runInfo, vec2ResUb, 0, actVecMSize);
    }
}
```

#### 路径 2: BMM2和累加结果都在GM (`ProcessVec2DSplit`)

```cpp
// fia_block_vec_noquant_gqa.h:812-931
// 适用于splitD场景: D轴跨核切分, 需要在GM间做累加
__aicore__ inline void ProcessVec2DSplit(
    GlobalTensor<T> &mmRes, const RunInfoX &runInfo)
{
    uint32_t subMBase = 8192 / dBasicBlock;  // 每次处理32KB
    int64_t vec2LoopLimit = CeilDiv(actVecMSize, subMBase);

    for (int64_t mOIdx = 0; mOIdx < vec2LoopLimit; mOIdx++) {
        // 1. DataCopy GM→UB: 从GM workspace搬运BMM2结果
        DataCopy(bmm2Ub, mmRes[offset], vec2CalcSize);

        // 2. DataCopy GM→UB: 从GM workspace搬运上一轮累加结果
        DataCopy(vec2ResUb, vec2ResGm[row][offset], vec2CalcSize);

        // 3. FlashUpdate (UB上)
        FlashUpdateNew(vec2ResUb, bmm2Ub, vec2ResUb, expUb,
                       subMRealsize, dBasicBlock);

        // 4. DataCopy UB→GM: 写回累加结果（下一轮用）
        DataCopy(vec2ResGm[row][offset], vec2ResUb, vec2CalcSize);
    }
}
```

#### 路径 3: BMM2在GM, 累加在UB (`ProcessVec2_C2GmUpdateUb`)

```cpp
// fia_block_vec_noquant_gqa.h:1140-1229
// BMM2结果从GM搬到UB, 在UB上做FlashUpdate
```

### 7.4 FlashDecode 归约流程

当 S2 轴被跨核切分时（`isS2SplitCore=true`），单个 batch/head 的完整 Attention 计算分布在多个核上，每个核只计算了部分 KV 的 Attention，需要在 FA 结束后进行**归约**。

```cpp
// fia_kernel_noquant_gqa.h:801-820
__aicore__ inline void FlashDecode()
{
    // 从GM读取FD分核元数据
    FDparamsX fdParams = {
        fdCoreEnable,      // 当前AIV是否使能
        fdBN2Idx,           // 归约任务的BN2索引
        fdMIdx,             // 归约任务的GS1索引
        fdS2SplitNum,      // S2跨核切分份数(需归约的份数)
        mStart, mLen,      // 当前AIV负责的M轴起止
        fdWorkspaceIdx      // workspace索引
    };
    vecFdBlock.FlashDecode(fdParams);
}
```

FlashDecode 核心逻辑（位于 `fia_block_vec_flashdecode.h`）：
1. 从 GM workspace 读取各核的**部分 max**和**部分 sum**
2. **Reduce Max**: 归约得到全局 max
3. **Reduce Sum**: 利用全局 max 重新计算 sum
4. **更新Attention输出**: 用归约后的 max/sum 做最终 FlashUpdate
5. **后量化 + 输出到 GM**

### 7.5 Softmax mloop 对冲机制

```cpp
// PRELOAD_N+1 = 3个buffer对冲
softmaxMaxBuf[3]  // 索引 = mloop % 3
softmaxSumBuf[3]
softmaxExpBuf[3]

// Vec1阶段写入: index = mloop % 3
LocalTensor<float> sumUb = softmaxSumBuf[runInfo.mloop % 3].Get<float>();

// Vec2阶段读取: index = (loop - PRELOAD_N) % 3
LocalTensor<T> expUb = softmaxExpBuf[runInfo.loop % 3].Get<T>();

// 对冲原理:
//   mloop=0 → buf[0]; mloop=1 → buf[1]; mloop=2 → buf[2]
//   mloop=3 → buf[0] (此时buf[0]已被消费完毕)
//   PRELOAD_N=2保证了消费在第mloop+2轮发生，第mloop轮写入,
//   3个buffer足够: 写入(mloop) → 等待 → 消费(mloop+2) → 释放
```

**对冲时序示意**：
```
mloop:   0       1       2       3       4       5
写入:   buf[0]  buf[1]  buf[2]  buf[0]  buf[1]  buf[2]
消费:   --      --      buf[0]  buf[1]  buf[2]  buf[0]
                 (loop  (loop   (loop   (loop   (loop
                  =2)     =3)     =4)     =5)     =6)
```

---

## 8. 同步控制与流水

### 8.1 AIC↔AIV 跨核同步

使用 **核间 ShareBuffer + Flag** 机制实现 AIC (Cube核心) 与 AIV (Vector核心) 之间的数据传递：

```
AIC ──Flipflow──> AIV:
  BMM1结果: Fixpipe(L0C→UB) → bmm1Buffers.SetCrossCore()
  AIV side: bmm1Buffers.WaitCrossCore()

AIV ──DateCopy──> AIC:
  P矩阵: DataCopy(UB→L1) → l1PBuffers.SetCrossCore()
  AIC side: l1PBuffers.Get() → WaitCrossCore()

AIC ──Fixpipe──> AIV (BMM2):
  BMM2结果: Fixpipe(L0C→UB/GM) → bmm2Buffers.SetCrossCore()
  AIV side: bmm2Buffers.WaitCrossCore()
```

### 8.2 Cube 内部流水事件链

以默认 BMM1 路径 (`IterateBmm1NdL0Split`) 为例：

```
时间 ───────────────────────────────────────────────────────>

MTE2 (搬运K到L1):
  [l1KBuffers.Get] → Wait MTE1_MTE2 → CopyKeyTile → Set MTE2_MTE1

                                   ↓
                              Wait MTE2_MTE1 (Q也需就绪)
                                   ↓
Cube (MatMul):
  [mmL0CBuffers.Get] → Wait FIX_M → MatmulN → Set M_FIX

                                   ↓
                              Wait M_FIX
                                   ↓
Fixpipe (L0C→UB):
  FixpipeMm1 (NZ→ND 转排)

                                   ↓
跨核通知:
  outputBuf.SetCrossCore()
```

**关键驻留优化**：Q矩阵在 S2 外层循环中**驻留 L1**，仅首次循环搬运，后续通过 `l1QBuffers.GetPre()` 复用。

```cpp
// fia_block_cube_noquant_gqa.h:551-560
if (unlikely(runInfo.isFirstS2Loop)) {
    mm1A = l1QBuffers.Get();
    CopyQueryTile(mm1ATensor, runInfo);  // 首次搬运Q
    mm1A.Set<HardEvent::MTE2_MTE1>();
} else {
    mm1A = l1QBuffers.GetPre();          // 复用上一轮的Q
    mm1A.Set<HardEvent::MTE2_MTE1>();    // 重新通知(状态机重置)
}
```

### 8.3 BMM1结果在UB上的 DoubleBuffer 管理

```cpp
// 初始化
InitMMResBuf:
  ubBufferManager.Init(pipe, mm1ResultSize * 2 + mm2ResultSize * 2);
  bmm1Buffers.Init(ubBufferManager, mm1ResultSize);  // DB自动分配2份
  if ASCEND_IS_AIV {
      bmm1Buffers.Get().SetCrossCore(); // AIV设置跨核同步开关
      bmm1Buffers.Get().SetCrossCore();
  }
```

**DB 轮转**：
```
AIC写入:  bmm1Buffers.Get()  → db[0]  // 第0轮写db[0]
           bmm1Buffers.Get()  → db[1]  // 第1轮写db[1]
           bmm1Buffers.Get()  → db[0]  // 第2轮写db[0], 此时db[0]已被AIV消费

AIV读取:  bmm1ResBuf.WaitCrossCore()  // 等待当前buffer就绪
           ...
           bmm1ResBuf.SetCrossCore()   // 释放(通知AIC可覆盖)
```

---

## 9. 关键数据结构

### 9.1 ConstInfo 公共常量

```cpp
// fia_public_define_arch35.h:105-215
struct CommonConstInfo {
    // === 轴长度 ===
    uint32_t bSize;            // Batch数
    uint64_t s1Size, s2Size;  // 序列长度
    uint32_t dSize, dSizeV;   // Head Dim
    uint32_t dBasicBlock;      // D对齐到64的块
    uint32_t dSizeRope;        // Rope D
    uint32_t gSize;            // Group数
    uint32_t n2Size;           // Head数

    // === 任务起止 (从GM MetaData读取) ===
    uint32_t bN2Start, bN2End;
    uint32_t gS1OStart, gS1OEnd;
    uint32_t s2OStart, s2OEnd;

    // === Mask/Sparse ===
    uint32_t sparseMode;
    int64_t preTokens, nextTokens;
    float scaleValue;

    // === 核信息 ===
    uint32_t aicIdx, aivIdx;
    uint8_t subBlockIdx;      // AIC下第几号AIV (0或1)
    uint32_t coreNum;
};
```

### 9.2 TilingData 结构

```cpp
// fia_tiling_data_noquant_gqa.h:23-42
class NoQuantTilingArch35 {
    FiaBaseParams         fiaBaseParams;          // 基础轴参数
    FiaAttenMaskParams    fiaAttenMaskParams;      // Mask参数
    FiaPseParams          fiaPseParams;            // PSE参数
    FiaSystemPrefixParams fiaSystemPrefixParams;   // 前缀参数
    FiaPageAttentionParams fiaPageAttentionParams; // PageAttention参数
    FiaLeftPaddingParams  fiaLeftPaddingParams;    // 左Padding参数
    FiaPostQuantParams    fiaPostQuantParams;      // 后量化参数
    FiaWorkspaceParams    fiaWorkspaceParams;      // Workspace参数
    FiaS1OuterSplitCoreParams fiaS1OuterSplitCoreParams; // S1外切参数
    FiaEmptyTensorParams  fiaEmptyTensorParams;    // 空Tensor参数
};

class FusedInferAttentionScoreTilingData {
    NoQuantTilingArch35 baseTiling;    // 基础Tiling数据
    FiaMetaData         fiaMetaData;   // FA+FD分核元数据
};
```

---

## 10. Workspace 空间计算

```cpp
// fia_tiling_nonquant_gqa.cpp:829-871
void FiaTilingNonQuantArch35::CalcWorkspaceSize()
{
    // 1. 系统基础workspace
    workspaceSize_ = sysWorkspaceSize;

    // 2. BMM2中间结果 (D>128或Dn模式D>192)
    if ((!dnFlag_ && dSize > DSIZE_128) || (dnFlag_ && dSize > DSIZE_192)) {
        // BMM2结果: mSize × dVBasicBlock × sizeof(float)
        // 3份: 3 × numBlocks (环形队列)
        workspaceSize_ += bmm2Bytes * 3 * numBlocks_;

        // Vec2累加结果 (DV>256时额外分配)
        if (dVBasicBlock > DSIZE_256) {
            workspaceSize_ += vec2Bytes * 3 * numBlocks_;
        }
    }

    // 3. FlashDecode归约中间结果 (仅FD场景)
    if (flashDecodeFlag_) {
        uint32_t faTmpAttenGmSize = numBlocks_ * 2 * mSize * dSize;
        uint32_t fatmpResLseGmSize = numBlocks_ * 2 * mSize * 8;
        workspaceSize_ += (faTmpAttenGmSize + 2 * fatmpResLseGmSize) * sizeof(float);
    }
}
```

**Workspace 布局**：
```
GM Workspace:
├── System sys workspace
├── BMM2 Res Ring Buffers (3 × numBlocks × mSize × dVBasicBlock × sizeof(T))
│   ├── Core 0: buf[0], buf[1], buf[2]
│   ├── Core 1: buf[0], buf[1], buf[2]
│   └── ...
├── Vec2 Acc Ring Buffers (DV>256, 3 × numBlocks × mSize × dVBasicBlock × sizeof(T))
├── FD attenOut workspace (numBlocks × 2 × mSize × dSize × sizeof(float))
│   └── 每个核最多2次写到workspace
├── FD sum workspace (numBlocks × 2 × mSize × 8 × sizeof(float))
└── FD max workspace (numBlocks × 2 × mSize × 8 × sizeof(float))
```

---

## 11. 总结

### 11.1 核心设计理念

| 设计点 | 实现 | 收益 |
|--------|------|------|
| **编译期模板特化** | TilingKey→PARSE_PARAMS→constexpr分支 | 零运行时分支开销 |
| **Preload任务队列** | PRELOAD_N=2, 3槽位环形队列 | Cube/Vec流水线隐藏延迟 |
| **Q驻留L1** | S2外层循环复用Q, 仅首次搬运 | 减少GM→L1带宽消耗 |
| **AIC/AIV异构协同** | Cube做MatMul, Vector做Softmax/Update | 最大化各计算单元利用率 |
| **Softmax三buffer对冲** | mloop%3轮转 | 保证WebSocket和Vec2之间流水不阻塞 |
| **L0切分多路径** | L0切K(L1SplitK) / L0切D(L0Split) / 全载 | Dsize自适应高效计算 |

### 11.2 计算路径决策树

```
输入 TilingKey
  │
  ├─ emptyTensor? → 空算子快速路径
  │
  ├─ Quant Mode?
  │   ├─ NoQuant     → FANoQuantGqaBlockCube/Vec (本文分析)
  │   ├─ AntiQuant   → FAFullQuantGqaBlockCube/Vec
  │   └─ FullQuant   → FAFullQuantGqaBlockCube/Vec
  │
  ├─ MLA Mode?
  │   └─ ROPE_SPLIT_D512 → FiaTilingNonQuantMlaArch35 (另一路径)
  │
  └─ BMM2输出位置?
      ├─ VECCALC (UB) → bmm2Write2Ub=true
      └─ GM           → bmm2Write2Ub=false
```

### 11.3 BMM1路径选择

```
dBaseSize?
├─ dBaseSize > 256 → IterateBmm1NdL1SplitK (L1切K)
│
├─ useDn?
│   ├─ dBaseSize > 128 → IterateBmm1DnSplitK (转置+L0切D)
│   └─ dBaseSize ≤ 128 → IterateBmm1Dn (转置全载)
│
└─ 默认 → IterateBmm1NdL0Split (Q驻留+L0切D)
```

### 11.4 Vec2路径选择

```
BMM2结果位置?
├─ bmm2Write2Ub=true → ProcessVec2OnUb (所有数据在UB)
│
└─ bmm2Write2Ub=false
    ├─ splitD=true  → ProcessVec2DSplit (结果+累加都在GM)
    └─ splitD=false → ProcessVec2_C2GmUpdateUb (BMM2在GM, 累加在UB)
```
