# Flash Attention Kernel层架构图

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                        Flash Attention Kernel层架构                                              │
│                        attention/flash_attn/op_kernel/                                           │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

## 1. Kernel入口层 (flash_attn.cpp)

┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│  __global__ __aicore__ void flash_attn(...)                                                    │
│                                                                                                │
│  4字段TilingKey → 模板参数推导:                                                                │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐        │
│  │ template <uint8_t inOutLayoutType, uint8_t KvLayoutType,                          │        │
│  │           bool hasAttenMask, uint8_t config>                                       │        │
│  │                                                                                    │        │
│  │ TilingKey字段解析:                                                                 │        │
│  │ • InOutLayoutType (8-bit)  → BSND/BNSD/TND/BNSD_BSND                             │        │
│  │ • KvLayoutType (8-bit)     → NO_PA/PA_BBND/PA_BNBD                                │        │
│  │ • HasAttenMask (1-bit)     → false/true                                          │        │
│  │ • Config (1-bit)           → (S1=128,S2=128) 或 (S1=64,S2=256)                  │        │
│  └────────────────────────────────────────────────────────────────────────────────────┘        │
│                                                                                                │
│  Dtype分发:                                                                                    │
│  #if (ORIG_DTYPE_Q == DT_BF16)                                                                │
│      flash_attn_kernel_run<bfloat16_t, bfloat16_t, ...>(...)                                  │
│  #elif (ORIG_DTYPE_Q == DT_FLOAT16)                                                           │
│      flash_attn_kernel_run<half, half, ...>(...)                                              │
│  #endif                                                                                        │
│                                                                                                │
│  REGISTER_TILING_DEFAULT(FlashAttnTilingData)                                                 │
│  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)  // Cube+Vector混合核                       │
│  dc_preload(metadata, 0)  // 预加载metadata                                                    │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
                                        ↓
                                        
## 2. 类型推导层 (flash_attn_entry_regbase.h)

┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│  flash_attn_kernel_run<INPUT_T, OUT_T, inOutLayoutType, KvLayoutType,                         │
│                        hasAttenMask, config>(...)                                             │
│                                                                                                │
│  Step 1: Layout解析                                                                            │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐        │
│  │ constexpr inputLayoutType  = InOutLayoutTypeValue[inOutLayoutType][0]            │        │
│  │ constexpr outputLayoutType = InOutLayoutTypeValue[inOutLayoutType][1]            │        │
│  │                                                                                    │        │
│  │ InOutLayoutTypeValue[5][2]:                                                       │        │
│  │ [0] → (BSND, BSND)                                                                │        │
│  │ [1] → (BNSD, BNSD)                                                                │        │
│  │ [2] → (TND,  TND)                                                                 │        │
│  │ [3] → (BNSD, BSND)                                                                │        │
│  │ [4] → (NTD,  NTD)                                                                 │        │
│  └────────────────────────────────────────────────────────────────────────────────────┘        │
│                                                                                                │
│  Step 2: Config解析 (D=128固定)                                                               │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐        │
│  │ ConfigValue[2]数组:                                                               │        │
│  │ Config=0 → (S1=128, S2=128, D=128, DV=128)  // sOuter=64, sInner=128            │        │
│  │ Config=1 → (S1=64,  S2=256, D=128, DV=128)  // sOuter=32, sInner=256            │        │
│  └────────────────────────────────────────────────────────────────────────────────────┘        │
│                                                                                                │
│  Step 3: 编译期常量推导                                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐        │
│  │ bmm2OutPos  = GetC2Position(dVTemplateType, UbOutCondition, ...)                 │        │
│  │ useDn       = IsDn(false, hasAttenMask, dTemplateType, ...)                      │        │
│  │ bmm2Write2Ub = (bmm2OutPos == VECCALC)                                            │        │
│  │ splitD      = (dVTemplateType > DTemplateType::Aligned256)                       │        │
│  └────────────────────────────────────────────────────────────────────────────────────┘        │
│                                                                                                │
│  Step 4: Block类型定义                                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐        │
│  │ using CubeBlock = FANoQuantGqaBlockCube<INPUT_T, float,                          │        │
│  │                                          inputLayoutType, s1TemplateType,         │        │
│  │                                          s2TemplateType, dTemplateType,           │        │
│  │                                          dVTemplateType, KvLayoutType,            │        │
│  │                                          useDn, bmm2Write2Ub, splitD>            │        │
│  │                                                                                    │        │
│  │ using VecFaBlock = FANoQuantGqaBlockVec<INPUT_T, float, OUT_T,                   │        │
│  │                                           inputLayoutType, outputLayoutType,      │        │
│  │                                           s1TemplateType, s2TemplateType,         │        │
│  │                                           dTemplateType, dVTemplateType,          │        │
│  │                                           hasAttenMask, false, KvLayoutType,     │        │
│  │                                           useDn, bmm2Write2Ub, splitD>          │        │
│  │                                                                                    │        │
│  │ using VecFdBlock = FiaBlockVecFlashDecode<INPUT_T, float, OUT_T,                 │        │
│  │                                           inputLayoutType, outputLayoutType,      │        │
│  │                                           s1TemplateType, s2TemplateType,         │        │
│  │                                           dTemplateType, dVTemplateType,          │        │
│  │                                           hasAttenMask, false, KvLayoutType,     │        │
│  │                                           useDn, bmm2Write2Ub, splitD>          │        │
│  └────────────────────────────────────────────────────────────────────────────────────┘        │
│                                                                                                │
│  Step 5: AIC/AIV分离编译                                                                       │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐        │
│  │ // 使用Dummy Block避免交叉编译不需要的代码                                          │        │
│  │ #ifdef __DAV_C310_CUBE__                                                          │        │
│  │     using Kernel = FlashAttentionNoQuantGqaKernel<CubeBlock,                     │        │
│  │                                                   VecBlockDummy,                  │        │
│  │                                                   VecBlockDummy>                  │        │
│  │ #else  // Vector核                                                                │        │
│  │     using Kernel = FlashAttentionNoQuantGqaKernel<CubeBlockDummy,                │        │
│  │                                                   VecFaBlock,                     │        │
│  │                                                   VecFdBlock>                     │        │
│  │ #endif                                                                            │        │
│  └────────────────────────────────────────────────────────────────────────────────────┘        │
│                                                                                                │
│  Step 6: Kernel实例化与执行                                                                    │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐        │
│  │ TPipe tPipe;                                                                       │        │
│  │ Kernel op;                                                                         │        │
│  │ op.Init(query, key, value, attenMask, cuSeqLensQ, cuSeqLensKv,                   │        │
│  │         blocktable, learnableSink, softmaxLse, attentionOut,                      │        │
│  │         workspace, runtimeMetaData, sequsedQ, sequsedKv,                          │        │
│  │         tilingData, &tPipe);                                                      │        │
│  │ op.Process();                                                                      │        │
│  └────────────────────────────────────────────────────────────────────────────────────┘        │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
                                        ↓

## 3. GQA Kernel模板架构 (flash_attn_kernel_base_gqa_template.h)

┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│  FlashAttentionNoQuantGqaKernel<CubeBlock, VecFaBlock, VecFdBlock>                             │
│                                                                                                │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐        │
│  │ 继承: FlashAttentionGqaContext (共享数据类)                                        │        │
│  │ 组合模式:                                                                          │        │
│  │ ├─ cubeProc: FlashAttentionCubeProcessor<CubeData>                               │        │
│  │ └─ vectorProc: FlashAttentionVectorProcessor<VectorData>                         │        │
│  └────────────────────────────────────────────────────────────────────────────────────┘        │
│                                                                                                │
│  核心方法:                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐        │
│  │ Init():                                                                            │        │
│  │ ├─ InitConstInfoWithPA() → 初始化aicIdx/aivIdx/dBasicBlock                       │        │
│  │ ├─ InitMMResBuf(workspace) → 初始化L1/UB Buffer                                   │        │
│  │ ├─ InitQCuSeqLensParser() → 初始化Query序列长度Parser                            │        │
│  │ ├─ InitKvCuSeqLensParser() → 初始化KV序列长度Parser                              │        │
│  │ │                                                                                  │        │
│  │ │ if ASCEND_IS_AIC {                                                              │        │
│  │ │     InitCube(query, key, value, blockTable)                                     │        │
│  │ │ } else {                                                                        │        │
│  │ │     vectorProc.vectorData.isFd = metadata[1]                                   │        │
│  │ │     if (isFd) InitFdMetaDataGm()                                               │        │
│  │ │     InitVector(cuSeqLensQ, cuSeqLensKv, attenMask, ...)                        │        │
│  │ │ }                                                                               │        │
│  │ └─────────────────────────────────────────────────────────────────────────────┐  │        │
│  │ Process():                                                                    │  │        │
│  │ ├─ if ASCEND_IS_AIV { CrossCoreBufferInit() }                                │  │        │
│  │ ├─ for (sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {            │  │        │
│  │ │       if ASCEND_IS_AIC {                                                    │  │        │
│  │ │           cubeProc.ProcessCube(sectionIdx)                                  │  │        │
│  │ │       } else {                                                              │  │        │
│  │ │           vectorProc.ProcessVector(sectionIdx)                              │  │        │
│  │ │       }                                                                     │  │        │
│  │ │   }                                                                         │  │        │
│  │ └─ if ASCEND_IS_AIC { CrossCoreBufferUnInit() }                              │  │        │
│  │ └─────────────────────────────────────────────────────────────────────────────┘  │        │
│  └────────────────────────────────────────────────────────────────────────────────────┘        │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
                                        ↓

## 4. 共享Context层 (flash_attn_kernel_base_gqa_common.h)

┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│  FlashAttentionGqaContext (共享数据类 - 无模板参数)                                             │
│                                                                                                │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐        │
│  │ Buffer管理器:                                                                      │        │
│  │ ├─ gmBufferManager  (GM内存)                                                     │        │
│  │ ├─ ubBufferManager  (UB内存)                                                     │        │
│  │ └─ l1BufferManager  (L1内存)                                                     │        │
│  │                                                                                    │        │
│  │ Buffer策略类:                                                                      │        │
│  │ ├─ bmm2ResGmBuffers: BuffersPolicy3buff<GM, CROSS_CORE_SYNC_FORWARD>            │        │
│  │ ├─ bmm1Buffers:     BuffersPolicyDB<UB, CROSS_CORE_SYNC_BOTH>                   │        │
│  │ ├─ bmm2Buffers:     BuffersPolicyDB<UB, CROSS_CORE_SYNC_BOTH>                   │        │
│  │ └─ l1PBuffers:      BuffersPolicy3buff<L1, CROSS_CORE_SYNC_FORWARD>             │        │
│  │                                                                                    │        │
│  │ GlobalTensor:                                                                      │        │
│  │ ├─ cuSeqLensGmQ   (Query累积序列长度)                                            │        │
│  │ ├─ cuSeqLensGmKv  (KV累积序列长度)                                               │        │
│  │ ├─ seqUsedGmQ     (Query实际序列长度)                                            │        │
│  │ ├─ seqUsedGmKv    (KV实际序列长度)                                               │        │
│  │ ├─ faMetaDataGm   (Flash Attention Metadata)                                     │        │
│  │ └─ keyPtr         (Key指针，用于FlashDecode)                                     │        │
│  │                                                                                    │        │
│  │ Tiling数据:                                                                        │        │
│  │ ├─ tilingData: const FlashAttnNoQuantTilingArch35*                              │        │
│  │ ├─ pipe: TPipe*                                                                   │        │
│  │ └─ constInfo: ConstInfo_t<NO_QUANT>                                              │        │
│  │                                                                                    │        │
│  │ Scheduler变量:                                                                     │        │
│  │ ├─ sectionNum_, coreGS1Loops, frontGS1Count                                      │        │
│  │ ├─ createdTaskCount, executedTaskCount                                           │        │
│  │ ├─ bN2Start_, bN2End_, gS1OStart_, gS1OEnd_, s2OStart_, s2OEnd_                 │        │
│  │ └─ actSeqLensKv, actSeqLensQ, s1SizeCurrBatch                                    │        │
│  └────────────────────────────────────────────────────────────────────────────────────┘        │
│                                                                                                │
│  核心方法:                                                                                      │
│  ├─ InitConstInfo() → 初始化aicIdx/aivIdx/dBasicBlock                                          │
│  ├─ GetFAMetaDataIndex(coreIdx, metaIdx, sectionIdx) → 计算Metadata索引                        │
│  └─ GetFASectionInfo(sectionIdx) → 从Metadata读取BN2/GS1/S2的Start/End                         │
└───────────────────────────────────────────────────────────────────────────────────────────────┘

## 5. Cube核处理器 (flash_attn_kernel_base_gqa_cube.h)

┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│  FlashAttentionCubeProcessor<CubeData>                                                         │
│                                                                                                │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐        │
│  │ 数据成员:                                                                          │        │
│  │ ├─ cubeData: FlashAttentionCubeData<CubeBlockType>                               │        │
│  │ │   ├─ cubeBlock: FANoQuantGqaBlockCube实例                                      │        │
│  │ │   ├─ qCuSeqLensParser, kvCuSeqLensParser                                       │        │
│  │ │   ├─ qSeqUsedParser, kvSeqUsedParser                                           │        │
│  │ │   └─ ctx: FlashAttentionGqaContext指针                                         │        │
│  │ └─────────────────────────────────────────────────────────────────────────────┐  │        │
│  │ │ RunInfoXCube taskRunInfoCube[PRELOAD_TASK_CACHE_SIZE]  // 预加载任务缓存     │  │        │
│  │ │ PRELOAD_N = 2, PRELOAD_TASK_CACHE_SIZE = 3                                   │  │        │
│  │ └─────────────────────────────────────────────────────────────────────────────┘  │        │
│  └────────────────────────────────────────────────────────────────────────────────────┘        │
│                                                                                                │
│  核心方法 (if ASCEND_IS_AIC):                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐        │
│  │ ProcessCube(sectionIdx):                                                          │        │
│  │ ├─ cubeBlock.InitBuffers()                                                       │        │
│  │ ├─ cubeBlock.AllocEventID()                                                      │        │
│  │ ├─ if (aicIdx < coreNum) FlashAttentionCube(sectionIdx)                          │        │
│  │ └─ cubeBlock.FreeEventID()                                                       │        │
│  │                                                                                    │        │
│  │ FlashAttentionCube(sectionIdx):                                                   │        │
│  │ ├─ GetFASectionInfo(sectionIdx) → 读取BN2/GS1/S2范围                            │        │
│  │ ├─ for (uint32_t bN2Idx = bN2Start_; bN2Idx < bN2End_; bN2Idx++) {              │        │
│  │ │       for (uint32_t gS1Idx = gS1OStart_; gS1Idx < gS1OEnd_; gS1Idx++) {       │        │
│  │ │           CreateTaskCube() → 创建任务                                         │        │
│  │ │           if (ShouldExecuteTaskCube()) ExecuteTaskCube()                      │        │
│  │ │       }                                                                       │        │
│  │ │   }                                                                           │        │
│  │                                                                                    │        │
│  │ ExecuteTaskCube(loop, taskRunInfoCube[]):                                         │        │
│  │ ├─ for (uint32_t taskIdx = 0; taskIdx < PRELOAD_TASK_CACHE_SIZE; taskIdx++) {   │        │
│  │ │       if (taskRunInfoCube[taskIdx].isValid) {                                 │        │
│  │ │           ComputeMm1Cube(taskRunInfoCube[taskIdx])                            │        │
│  │ │           ComputeMm2Cube(taskRunInfoCube[taskIdx])                            │        │
│  │ │       }                                                                       │        │
│  │ │   }                                                                           │        │
│  │                                                                                    │        │
│  │ ComputeMm1Cube(runInfoCube):                                                      │        │
│  │ ├─ CalcParamsCube() → 计算bIdx, n2Idx, gS1Idx, s2Idx, actMSize等                │        │
│  │ ├─ cubeBlock.IterateBmm1(bmm1Buffers, runInfo)                                  │        │
│  │ │   → Q × K^T 矩阵乘, 输出到bmm1Buffers                                         │        │
│  │ └─────────────────────────────────────────────────────────────────────────────┐  │        │
│  │ ComputeMm2Cube(runInfoCube):                                                 │  │        │
│  │ ├─ cubeBlock.IterateBmm2(bmm2Buffers, runInfo)                               │  │        │
│  │ │   → AttnScore × V 矩阵乘, 输出到bmm2Buffers                                 │  │        │
│  │ └─ if (!BMM2_TOUB) 输出到GM workspace                                         │  │        │
│  │ └─────────────────────────────────────────────────────────────────────────────┘  │        │
│  └────────────────────────────────────────────────────────────────────────────────────┘        │
│                                                                                                │
│  辅助方法:                                                                                      │
│  ├─ ShouldDispatchTaskCube() → 是否分发任务                                                    │        │
│  ├─ ShouldExecuteTaskCube() → 是否执行任务                                                     │        │
│  ├─ GetTaskDealModeCube() → 任务处理模式(SKIP/EXECUTE/CREATE)                                  │        │
│  ├─ CalcCurS2StartEndNoSparseCube() → 计算S2范围(无稀疏)                                       │        │
│  ├─ CalcCurS2StartEndWithSparseCube() → 计算S2范围(有稀疏Mask)                                 │        │
│  └─ UpdateAxisInfoCube() → 更新循环索引                                                        │        │
└───────────────────────────────────────────────────────────────────────────────────────────────┘

## 6. Vector核处理器 (flash_attn_kernel_base_gqa_vector.h)

┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│  FlashAttentionVectorProcessor<VectorData>                                                     │
│                                                                                                │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐        │
│  │ 数据成员:                                                                          │        │
│  │ ├─ vectorData: FlashAttentionVectorData<VecFaBlock, VecFdBlock, InputT,         │        │
│  │ │                LayoutQ, PageAttention>                                          │        │
│  │ │   ├─ vecFaBlock: FANoQuantGqaBlockVec实例                                      │        │
│  │ │   ├─ vecFdBlock: FiaBlockVecFlashDecode实例                                    │        │
│  │ │   ├─ qCuSeqLensParser, kvCuSeqLensParser                                       │        │
│  │ │   ├─ qSeqUsedParser, kvSeqUsedParser                                           │        │
│  │ │   ├─ isFd: FlashDecode启用标志                                                 │        │
│  │ │   └─ ctx: FlashAttentionGqaContext指针                                         │        │
│  │ └─────────────────────────────────────────────────────────────────────────────┐  │        │
│  │ │ RunInfoXVec taskRunInfoVec[PRELOAD_TASK_CACHE_SIZE]  // 预加载任务缓存       │  │        │
│  │ └─────────────────────────────────────────────────────────────────────────────┘  │        │
│  └────────────────────────────────────────────────────────────────────────────────────┘        │
│                                                                                                │
│  核心方法 (if ASCEND_IS_AIV):                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐        │
│  │ ProcessVector(sectionIdx):                                                        │        │
│  │ ├─ if (aicIdx < coreNum) FlashAttentionVector(sectionIdx)                        │        │
│  │ ├─ else FlashDecodeVector(sectionIdx)                                            │        │
│  │                                                                                    │        │
│  │ FlashAttentionVector(sectionIdx):                                                 │        │
│  │ ├─ GetFASectionInfo(sectionIdx) → 读取BN2/GS1/S2范围                            │        │
│  │ ├─ for (uint32_t bN2Idx = bN2Start_; bN2Idx < bN2End_; bN2Idx++) {              │        │
│  │ │       for (uint32_t gS1Idx = gS1OStart_; gS1Idx < gS1OEnd_; gS1Idx++) {       │        │
│  │ │           CreateTaskVector() → 创建任务                                       │        │
│  │ │           if (ShouldExecuteTaskVector()) ExecuteTaskVector()                  │        │
│  │ │       }                                                                       │        │
│  │ │   }                                                                           │        │
│  │                                                                                    │        │
│  │ ExecuteTaskVector(loop, taskRunInfoVec[]):                                        │        │
│  │ ├─ for (uint32_t taskIdx = 0; taskIdx < PRELOAD_TASK_CACHE_SIZE; taskIdx++) {   │        │
│  │ │       if (taskRunInfoVec[taskIdx].isValid) {                                  │        │
│  │ │           ComputeVec1Vector(taskRunInfoVec[taskIdx])                          │        │
│  │ │           ComputeVec2Vector(taskRunInfoVec[taskIdx])                          │        │
│  │ │       }                                                                       │        │
│  │ │   }                                                                           │        │
│  │                                                                                    │        │
│  │ ComputeVec1Vector(runInfoVec):                                                    │        │
│  │ ├─ CalcParamsVector() → 计算loop, bIdx, n2Idx, actS1Size, actS2Size等           │        │
│  │ ├─ vecFaBlock.ProcessVec1(bmm1Buffers, runInfo)                                 │        │
│  │ │   → Softmax + LSE计算, 输出到bmm2Buffers                                       │        │
│  │ │   ├─ SoftmaxFlashV2(bmm1Buffers)                                              │        │
│  │ │   ├─ Mul(bmm2Buffers, softmaxMax, softmaxSum)                                 │        │
│  │ │   └─ 计算log-sum-exp (LSE)                                                    │        │
│  │ └─────────────────────────────────────────────────────────────────────────────┐  │        │
│  │ ComputeVec2Vector(runInfoVec):                                              │  │        │
│  │ ├─ vecFaBlock.ProcessVec2(bmm2Buffers, runInfo)                             │  │        │
│  │ │   → Output写入 + LSE输出                                                   │  │        │
│  │ │   ├─ if (bmm2Write2Ub) 从UB读取bmm2结果                                    │  │        │
│  │ │   ├─ else 从GM workspace读取bmm2结果                                       │  │        │
│  │ │   ├─ Cast<float→INPUT_T>()                                                 │  │        │
│  │ │   └─ WriteOutput(attentionOutGm)                                           │  │        │
│  │ │   ├─ if (isSoftmaxLseEnable) WriteLSE(softmaxLseGm)                        │  │        │
│  │ │   └─────────────────────────────────────────────────────────────────────┐  │  │        │
│  │ │ FlashDecodeVector(sectionIdx):                                          │  │  │        │
│  │ │ ├─ vecFdBlock.InitParams()                                              │  │  │        │
│  │ │ ├─ vecFdBlock.ProcessFlashDecode()                                      │  │  │        │
│  │ │ │   → 多Token并行解码                                                    │  │  │        │
│  │ │ │   ├─ 从GM读取bmm2Res                                                   │  │  │        │
│  │ │ │   ├─ Softmax + Scale                                                   │  │  │        │
│  │ │ │   ├─ Accumulate Output                                                 │  │  │        │
│  │ │ │   └─ Write to attentionOutGm                                          │  │  │        │
│  │ │ └─────────────────────────────────────────────────────────────────────┘  │  │        │
│  │ └─────────────────────────────────────────────────────────────────────────────┘  │        │
│  └────────────────────────────────────────────────────────────────────────────────────┘        │
│                                                                                                │
│  辅助方法:                                                                                      │
│  ├─ CrossCoreBufferInit() → 初始化Cube-Vector核间同步Buffer                                    │        │
│  ├─ ShouldDispatchTaskVector() → 是否分发任务                                                  │        │
│  ├─ ShouldExecuteTaskVector() → 是否执行任务                                                   │        │
│  ├─ GetTaskDealModeVector() → 任务处理模式                                                    │        │
│  ├─ CalcCurS2StartEndNoSparseVector() → 计算S2范围                                            │        │
│  ├─ CalcCurS2StartEndWithSparseVector() → 计算S2范围                                          │        │
│  └─ UpdateAxisInfoVector() → 更新循环索引                                                     │        │
└───────────────────────────────────────────────────────────────────────────────────────────────┘

## 7. Block实现层

### 7.1 Cube Block (flash_attn_block_cube_noquant_gqa.h)

┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│  FANoQuantGqaBlockCube<INPUT_T, MM_T, layout, s1TemplateType, s2TemplateType,                  │
│                        dTemplateType, dVTemplateType, KvLayoutType,                            │
│                        useDn, bmm2Write2Ub, splitD>                                            │
│                                                                                                │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐        │
│  │ 编译期常量:                                                                        │        │
│  │ ├─ mBaseSize  = (uint32_t)s1TemplateType  // 64/128                              │        │
│  │ ├─ s2BaseSize = (uint32_t)s2TemplateType  // 128/256                             │        │
│  │ ├─ dBaseSize  = (uint32_t)dTemplateType  // 128                                  │        │
│  │ ├─ dVBaseSize = (uint32_t)dVTemplateType // 128                                  │        │
│  │ ├─ USE_DN     = useDn                                                            │        │
│  │ ├─ BMM2_TOUB  = bmm2Write2Ub                                                    │        │
│  │ └─ PAGE_ATTENTION = (KvLayoutType != 0)                                         │        │
│  │                                                                                    │        │
│  │ Buffer选择策略 (模板推导):                                                         │        │
│  │ ├─ QL1BuffSel<INPUT_T, dBaseSize>::Type                                         │        │
│  │ ├─ KVL1BuffSel<INPUT_T, s2BaseSize, dBaseSize>::Type                            │        │
│  │ ├─ L0ABuffSel<INPUT_T>::Type                                                    │        │
│  │ ├─ L0BBuffSel<INPUT_T>::Type                                                    │        │
│  │ └─ L0CBuffSel<MM_T>::Type                                                       │        │
│  │                                                                                    │        │
│  │ GlobalTensor格式推导:                                                             │        │
│  │ ├─ GetQueryGmFormat<LAYOUT>() → BSNGD/SBNGD/BNGSD/TNGD/NGTD                     │        │
│  │ └─ GetKVGmFormat<LAYOUT, KvLayoutType>() → BSND/PA_BnBsND/PA_BnNBsD/PA_NZ       │        │
│  └────────────────────────────────────────────────────────────────────────────────────┘        │
│                                                                                                │
│  核心方法:                                                                                      │
│  ├─ InitCubeBlock(pipe, l1BufferManager, query, key, value, blockTable, ...)                   │        │
│  │   → 初始化所有GM Tensor和L1/L0A/L0B/L0C Buffer                                               │        │
│  ├─ IterateBmm1(bmm1Buffers, runInfo)                                                          │        │
│  │   → Q × K^T 矩阵乘                                                                            │        │
│  │   ├─ LoadQToL1(qGm, l1QBuffer)                                                               │        │
│  │   ├─ LoadKToL1(kGm, l1KBuffer)                                                               │        │
│  │   ├─ MatMul<Q×K>(l1QBuffer, l1KBuffer, l0CBuffer)                                            │        │
│  │   └─ FixPipe(l0CBuffer, bmm1Buffers) → 输出到UB                                              │        │
│  ├─ IterateBmm2(bmm2Buffers, runInfo)                                                          │        │
│  │   → AttnScore × V 矩阵乘                                                                     │        │
│  │   ├─ LoadVToL1(vGm, l1VBuffer)                                                               │        │
│  │   ├─ if (BMM2_TOUB) MatMul<Score×V>(bmm1Buffers, l1VBuffer, l0CBuffer)                       │        │
│  │   │   else MatMul<Score×V>(bmm1BuffersGm, l1VBuffer, l0CBuffer)                              │        │
│  │   └─ FixPipe(l0CBuffer, bmm2Buffers) → 输出到UB或GM                                          │        │
│  └─ InitBuffers(), AllocEventID(), FreeEventID()                                               │        │
└───────────────────────────────────────────────────────────────────────────────────────────────┘

### 7.2 Vector Block (flash_attn_block_vec_noquant_gqa.h)

┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│  FANoQuantGqaBlockVec<INPUT_T, MM_T, OUTPUT_T, layout, outLayout,                              │
│                       s1TemplateType, s2TemplateType,                                          │
│                       dTemplateType, dVTemplateType,                                          │
│                       hasAtten, hasDrop, KvLayoutType,                                         │
│                       useDn, bmm2Write2Ub, splitD>                                             │
│                                                                                                │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐        │
│  │ 编译期常量:                                                                        │        │
│  │ ├─ mBaseSize  = (uint32_t)s1TemplateType  // 64/128                              │        │
│  │ ├─ s2BaseSize = (uint32_t)s2TemplateType  // 128/256                             │        │
│  │ ├─ HAS_MASK   = hasAttenMask                                                    │        │
│  │ ├─ PRELOAD_N  = 2                                                               │        │
│  │ ├─ MASK_LAYOUT = SG/GS (根据layout推导)                                         │        │
│  │ └─ BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0                              │        │
│  │                                                                                    │        │
│  │ GlobalTensor成员:                                                                 │        │
│  │ ├─ attenMaskGm (if HAS_MASK)                                                    │        │
│  │ ├─ softmaxLseGm (if isSoftmaxLseEnable)                                         │        │
│  │ ├─ attentionOutGm                                                               │        │
│  │ ├─ softmaxMaxGm, softmaxSumGm                                                   │        │
│  │ ├─ accumOutGm (FlashDecode)                                                     │        │
│  │ └─ learnableSinkGm (if learnableSinkFlag)                                       │        │
│  │                                                                                    │        │
│  │ Buffer成员:                                                                        │        │
│  │ ├─ softmaxMaxUb, softmaxSumUb, softmaxExpUb                                    │        │
│  │ ├─ lseUb, vec1ResUb                                                              │        │
│  │ ├─ vec2ResUb, vec2ResTmpUb                                                      │        │
│  │ └─ bmm2ResUb (if bmm2Write2Ub)                                                  │        │
│  └────────────────────────────────────────────────────────────────────────────────────┘        │
│                                                                                                │
│  核心方法:                                                                                      │
│  ├─ InitVecBlock(pipe, cuSeqLensQ, cuSeqLensKv, attenMask, ...)                               │        │
│  │   → 初始化所有GM Tensor和UB Buffer                                                           │        │
│  ├─ ProcessVec1(bmm1Buffers, runInfo)                                                          │        │
│  │   → Softmax + LSE计算                                                                        │        │
│  │   ├─ if (HAS_MASK) ApplyAttenMask(bmm1Buffers, attenMaskGm)                                 │        │
│  │   ├─ SoftmaxFlashV2(bmm1Buffers, softmaxMaxUb, softmaxSumUb, softmaxExpUb)                  │        │
│  │   ├─ Div(bmm2Res, softmaxSumUb) → Scale by LSE                                              │        │
│  │   ├─ Mul(bmm2Buffers, softmaxMaxUb) → Rescale                                               │        │
│  │   └─ CalcLSE(softmaxMaxUb, softmaxSumUb, lseUb)                                             │        │
│  ├─ ProcessVec2(bmm2Buffers, runInfo)                                                          │        │
│  │   → Output写入 + LSE输出                                                                     │        │
│  │   ├─ if (bmm2Write2Ub) Copy(bmm2ResUb, bmm2Buffers)                                         │        │
│  │   │   else Copy(bmm2ResGm, bmm2Buffers)                                                     │        │
│  │   ├─ Cast<bmm2Res, OUTPUT_T>() → FP16/BF16转换                                              │        │
│  │   ├─ WriteOutput(attentionOutGm, bIdx, n2Idx, gS1Idx)                                       │        │
│  │   ├─ if (isSoftmaxLseEnable) WriteLSE(softmaxLseGm, lseUb)                                  │        │
│  │   └─ if (learnableSinkFlag) AddSink(attentionOutGm, learnableSinkGm)                        │        │
│  ├─ ClearOutput() → 清零attentionOutGm                                                         │        │
│  └─ SetCuSeqLensParser(parser) → 设置序列长度Parser                                            │        │
└───────────────────────────────────────────────────────────────────────────────────────────────┘

### 7.3 FlashDecode Block (flash_attention_noquant_block_vec_flashdecode.h)

┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│  FiaBlockVecFlashDecode<INPUT_T, MM_T, OUTPUT_T, layout, outLayout,                            │
│                         s1TemplateType, s2TemplateType,                                        │
│                         dTemplateType, dVTemplateType,                                         │
│                         hasAtten, hasDrop, KvLayoutType,                                       │
│                         useDn, bmm2Write2Ub, splitD>                                           │
│                                                                                                │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐        │
│  │ FlashDecode用于多Token并行解码:                                                   │        │
│  │ • 将S2拆分到多个Vector核并行处理                                                 │        │
│  │ • 每个核处理一部分KV序列                                                         │        │
│  │ • 最后累加所有核的部分结果                                                       │        │
│  │                                                                                    │        │
│  │ Metadata结构:                                                                      │        │
│  │ ├─ fdCoreEnable → 核启用标志                                                     │        │
│  │ ├─ fdBN2Idx → Batch/Head索引                                                    │        │
│  │ ├─ fdMIdx → Token索引                                                           │        │
│  │ ├─ fdS2SplitNum → S2拆分数量                                                    │        │
│  │ ├─ mStart, mLen → Token范围                                                     │        │
│  │ └─ fdWorkspaceIdx → Workspace索引                                               │        │
│  └────────────────────────────────────────────────────────────────────────────────────┘        │
│                                                                                                │
│  核心方法:                                                                                      │
│  ├─ SetTilingData(tilingData) → 设置Tiling参数                                                │        │
│  ├─ InitParams() → 初始化FlashDecode参数                                                      │        │
│  ├─ InitGlobalTensor(softmaxFDMaxGm, softmaxFDSumGm, accumOutGm, attentionOutGm, keyPtr)      │        │
│  ├─ SetFdMetaDataGm(fdMetaDataPtr, sectionNum_) → 设置Metadata                                │        │
│  ├─ ProcessFlashDecode()                                                                       │        │
│  │   → 多Token并行解码                                                                          │        │
│  │   ├─ for (sectionIdx < sectionNum_) {                                                       │        │
│  │   │       GetFDSectionInfo(sectionIdx) → 读取fdBN2Idx, fdMIdx, fdS2SplitNum                 │        │
│  │   │       for (fdBN2Idx, fdMIdx) {                                                          │        │
│  │   │           LoadPartialBmm2Res(workspace, fdWorkspaceIdx)                                │        │
│  │   │           Softmax + Scale                                                               │        │
│  │   │           Accumulate(accumOutGm, partialRes)                                            │        │
│  │   │       }                                                                                 │        │
│  │   │   }                                                                                     │        │
│  │   ├─ FinalAccumulate(accumOutGm, softmaxFDMaxGm, softmaxFDSumGm)                            │        │
│  │   └─ WriteToOutput(attentionOutGm)                                                          │        │
│  └─ SetCuSeqLensParsers(qParser, kvParser) → 设置Parser                                        │        │
└───────────────────────────────────────────────────────────────────────────────────────────────┘

## 8. 数据结构层 (utils/flash_attn_common_def.h)

┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│  RunInfoXCube (28字节, Cube核专用)                                                              │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐        │
│  │ uint32_t: bIdx, n2Idx, gS1Idx, s2Idx, actMSize, actSingleLoopS2Size              │        │
│  │ bool: isValid, isFirstS2Loop, isLastS2Loop, isChangeBatch                         │        │
│  │                                                                                    │        │
│  │ 字段排布 (按对齐大小排序):                                                         │        │
│  │ • 6个uint32 (24字节)                                                              │        │
│  │ • 4个bool (4字节)                                                                 │        │
│  │ • 总计28字节, 无填充                                                              │        │
│  │                                                                                    │        │
│  │ 删除的16个无用字段 (相比原始RunInfoX):                                             │        │
│  │ • loop, mloop, s1Idx, actS1Size, actS2Size                                       │        │
│  │ • actMSizeAlign32, actVecMSize, vecMbaseIdx                                     │        │
│  │ • isS2SplitCore, faTmpOutWsPos                                                   │        │
│  │ • preTokensLeftUp, nextTokensLeftUp, gIdx                                       │        │
│  │ • qPaddingBeginOffset, kvPaddingBeginOffset                                     │        │
│  └────────────────────────────────────────────────────────────────────────────────────┘        │
├───────────────────────────────────────────────────────────────────────────────────────────────┤
│  RunInfoXVec (96字节, Vector核专用)                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐        │
│  │ uint64_t: actS1Size, actS2Size, preTokensLeftUp, nextTokensLeftUp                 │        │
│  │ int64_t: preTokensLeftUp, nextTokensLeftUp                                        │        │
│  │ uint32_t: actVecMSize, actSingleLoopS2Size, loop, mloop,                          │        │
│  │           actSingleLoopS2SizeAlign, bIdx, n2Idx, gS1Idx, gIdx, s2Idx,             │        │
│  │           actMSize, actMSizeAlign32, vecMbaseIdx, faTmpOutWsPos                   │        │
│  │ bool: isValid, isFirstS2Loop, isLastS2Loop, isS2SplitCore                         │        │
│  │                                                                                    │        │
│  │ 字段排布优化:                                                                       │        │
│  │ • 8字节字段放前面 (消除填充)                                                      │        │
│  │ • 4字节热点字段紧凑排列 (actVecMSize访问46次)                                    │        │
│  │ • 1字节字段放末尾                                                                  │        │
│  │                                                                                    │        │
│  │ 优化效果:                                                                          │        │
│  │ • 原始104字节 → 96字节 (节省8字节)                                               │        │
│  │ • 填充空洞: 11字节 → 0字节                                                        │        │
│  │                                                                                    │        │
│  │ 删除的4个无用字段:                                                                 │        │
│  │ • isChangeBatch, s1Idx, qPaddingBeginOffset, kvPaddingBeginOffset                │        │
│  └────────────────────────────────────────────────────────────────────────────────────┘        │
├───────────────────────────────────────────────────────────────────────────────────────────────┤
│  ConstInfo_t<FiaKernelType::NO_QUANT>                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐        │
│  │ struct CommonConstInfo {                                                          │        │
│  │     uint32_t aicIdx;      // Cube核索引                                           │        │
│  │     uint32_t aivIdx;      // Vector核索引                                         │        │
│  │     uint8_t subBlockIdx;  // 子块索引                                             │        │
│  │     uint32_t dBasicBlock; // D对齐大小                                            │        │
│  │ }                                                                                  │        │
│  └────────────────────────────────────────────────────────────────────────────────────┘        │
├───────────────────────────────────────────────────────────────────────────────────────────────┤
│  ConfigValue[2]数组 (Config → S1/S2/D/DV映射)                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐        │
│  │ Config=0: {s1=Aligned128, s2=Aligned128, d=Aligned128, dv=Aligned128}              │        │
│  │ Config=1: {s1=Aligned64,  s2=Aligned256, d=Aligned128, dv=Aligned128}              │        │
│  │                                                                                    │        │
│  │ S1TemplateType: 16/32/64/128/256/NotAligned                                       │        │
│  │ S2TemplateType: 16/32/64/128/256/512/1024/NotAligned                             │        │
│  │ DTemplateType:  16/32/48/64/80/96/128/160/192/256/512/576/NotAligned             │        │
│  └────────────────────────────────────────────────────────────────────────────────────┘        │
├───────────────────────────────────────────────────────────────────────────────────────────────┤
│  InOutLayoutTypeValue[5][2] (Q_Out Layout → Input/Output映射)                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐        │
│  │ [0] BSND     → (BSND, BSND)                                                       │        │
│  │ [1] BNSD     → (BNSD, BNSD)                                                       │        │
│  │ [2] TND      → (TND,  TND)                                                        │        │
│  │ [3] BNSD_BSND → (BNSD, BSND)  // 输入BNSD，输出BSND                              │        │
│  │ [4] NTD_NTD  → (NTD,  NTD)                                                        │        │
│  └────────────────────────────────────────────────────────────────────────────────────┘        │
└───────────────────────────────────────────────────────────────────────────────────────────────┘

## 9. 核间同步机制

┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│  Cube-Vector核间同步 (CROSS_CORE_SYNC)                                                         │
│                                                                                                │
│  Buffer同步策略:                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐        │
│  │ • bmm1Buffers:  BuffersPolicyDB<UB, CROSS_CORE_SYNC_BOTH>                        │        │
│  │   → Cube写入, Vector读取, 双向同步                                               │        │
│  │                                                                                    │        │
│  │ • bmm2Buffers:  BuffersPolicyDB<UB, CROSS_CORE_SYNC_BOTH>                        │        │
│  │   → Cube写入, Vector读取, 双向同步                                               │        │
│  │                                                                                    │        │
│  │ • bmm2ResGmBuffers: BuffersPolicy3buff<GM, CROSS_CORE_SYNC_FORWARD>              │        │
│  │   → Cube写入, Vector读取, 前向同步                                               │        │
│  │                                                                                    │        │
│  │ • l1PBuffers:   BuffersPolicy3buff<L1, CROSS_CORE_SYNC_FORWARD>                  │        │
│  │   → 前向同步                                                                      │        │
│  └────────────────────────────────────────────────────────────────────────────────────┘        │
│                                                                                                │
│  Metadata同步:                                                                                 │
│  ├─ faMetaDataGm: Cube写入BN2/GS1/S2范围, Vector读取                                           │        │
│  └─ fdMetaDataGm: FlashDecode核读取任务分配                                                   │        │
│                                                                                                │
│  同步流程:                                                                                      │
│  ├─ Cube核:                                                                                     │
│  │   ├─ ProcessCube() → ComputeMm1/ComputeMm2 → 写入bmm1Buffers/bmm2Buffers                    │        │
│  │   └─ CrossCoreBufferUnInit() → 释放同步Buffer                                                │        │
│  └─ Vector核:                                                                                   │
│  │   ├─ CrossCoreBufferInit() → 初始化同步Buffer                                                 │        │
│  │   └─ ProcessVector() → ComputeVec1/ComputeVec2 → 读取bmm1Buffers/bmm2Buffers                │        │
└───────────────────────────────────────────────────────────────────────────────────────────────┘

## 10. 整体数据流

┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│  Kernel执行流程                                                                                │
│                                                                                                │
│  1. Kernel入口 (flash_attn.cpp)                                                                │
│     ├─ TilingKey解析 → 4字段模板参数                                                          │
│     ├─ Dtype分发 → BF16/FP16                                                                  │
│     └─ 调用flash_attn_kernel_run()                                                            │
│                                                                                                │
│  2. 类型推导 (flash_attn_entry_regbase.h)                                                      │
│     ├─ Layout解析 → inputLayout/outputLayout                                                  │
│     ├─ Config解析 → S1/S2/D/DV                                                                │
│     ├─ 编译期常量推导 → bmm2OutPos, useDn, bmm2Write2Ub, splitD                               │
│     ├─ Block类型定义 → CubeBlock, VecFaBlock, VecFdBlock                                      │
│     └─ AIC/AIV分离编译 → Kernel实例化                                                         │
│                                                                                                │
│  3. Kernel初始化 (FlashAttentionNoQuantGqaKernel.Init())                                      │
│     ├─ InitConstInfoWithPA() → aicIdx/aivIdx                                                  │
│     ├─ InitMMResBuf() → L1/UB Buffer初始化                                                    │
│     ├─ InitCuSeqLensParser() → Parser初始化                                                   │
│     ├─ if ASCEND_IS_AIC { InitCube() }                                                        │
│     └ else { InitVector() + FlashDecode初始化 }                                               │
│                                                                                                │
│  4. Kernel执行 (FlashAttentionNoQuantGqaKernel.Process())                                     │
│     ├─ for (sectionIdx < sectionNum_) {                                                       │
│     │       if ASCEND_IS_AIC { ProcessCube() }                                                │
│     │       else { ProcessVector() }                                                          │
│     │   }                                                                                     │
│                                                                                                │
│  5. Cube核循环:                                                                               │
│     ├─ for (bN2Idx, gS1Idx) {                                                                 │
│     │       ├─ GetFASectionInfo() → 读取BN2/GS1/S2范围                                        │
│     │       ├─ CreateTaskCube() → 创建任务                                                    │
│     │       ├─ ComputeMm1Cube() → Q×K^T                                                      │
│     │       ├─ ComputeMm2Cube() → Score×V                                                    │
│     │       └─ 输出到bmm1Buffers/bmm2Buffers                                                  │
│     │   }                                                                                     │
│                                                                                                │
│  6. Vector核循环:                                                                              │
│     ├─ for (bN2Idx, gS1Idx) {                                                                 │
│     │       ├─ GetFASectionInfo() → 读取BN2/GS1/S2范围                                        │
│     │       ├─ CreateTaskVector() → 创建任务                                                  │
│     │       ├─ ComputeVec1Vector() → Softmax + LSE                                            │
│     │       ├─ ComputeVec2Vector() → Output写入                                               │
│     │       └─ (可选) FlashDecodeVector() → 多Token并行解码                                  │
│     │   }                                                                                     │
│                                                                                                │
│  7. 核间同步:                                                                                  │
│     ├─ Cube写入bmm1Buffers/bmm2Buffers → 同步信号                                             │
│     └─ Vector等待同步 → 读取bmm1Buffers/bmm2Buffers                                           │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
```