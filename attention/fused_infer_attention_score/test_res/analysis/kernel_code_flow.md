# FIA NoQuant GQA Kernel 详细代码流程分析

> 当前测试用例: B=2, N2=4, G=2, S1=32, S2=9856, D=128, FP16, BNSD layout
> 模板参数: quantMode=31, isReconstructTemp=true, fd=1, pse=9, mask=0, rope=0
> 架构: 28 AIC + 56 AIV, blockDim=84, s2BaseSize=128, mBaseSize=128, dBaseSize=128, dVBaseSize=128

## 一、Kernel 代码流程

### 1.1 完整调用链总览

```
fused_infer_attention_score_apt.cpp
  └─ fused_infer_attention_score<>()
       │  printf: [FIA KERNEL] PFA dispatcher, blk=...
       │
       └─ prompt_flash_attention_entry_regbase.h
            └─ prompt_flash_attention_FIAS_regbase<>()
                 │  isReconstructTemp=true, FP16
                 │
                 └─ fia_template_dispatcher.h
                      └─ run_fia_noquant_gqa_kernel<half, half, ...>()
                           │  PARSE_PARAMS_NoQuant → tile 尺寸
                           │  AIC/AIV 双路 Block 选择
                           │  Inst FlashAttentionNoQuantGqaKernel
                           │
                           ├─ op.Init(Q,K,V,...)
                           │    ├─ InitConstInfo()
                           │    ├─ InitMMResBuf()
                           │    ├─ vecFaBlock.InitVecBlock()   [AIV]
                           │    ├─ cubeBlock.InitCubeBlock()   [AIC]
                           │    └─ vecFdBlock.Init{Params,GlobalTensor}() [AIV,FD]
                           │
                           └─ op.Process()
                                ├─ CrossCoreBufferInit()
                                ├─ FlashAttention()         ← 主计算循环
                                │    └─ while(dispatch||execute):
                                │         CreateTask → ExecuteTask
                                │         └─ AIC: MM1→MM2  AIV: Vec1→Vec2
                                ├─ CrossCoreBufferUnInit()
                                └─ FlashDecode()           [AIV, FD]
```

### 1.2 入口分发

**文件**: `attention/fused_infer_attention_score/op_kernel/fused_infer_attention_score_apt.cpp`

```cpp
#define FullQuantTiling 15
template <uint8_t inOutLayoutType, uint16_t config, uint8_t pseMode, uint8_t quantMode,
          bool hasAttenMask, bool hasRope, uint8_t KvLayoutType, bool isFd,
          bool emptyTensor, bool enableKVPrefix, bool enableS1OutSplit, bool isReconstructTemp>
__global__ __aicore__ void fused_infer_attention_score(...)
{
    printf("[FIA KERNEL] %s dispatcher, blk=%u lay=%u cfg=%u pse=%u quant=%u "
           "mask=%u rope=%u kvL=%u fd=%u empty=%u kvpfx=%u s1os=%u reconst=%u\n", ...);

    if (quantMode >= FullQuantTiling) {
        // PFA 模板 (Prompt Flash Attention)
        prompt_flash_attention_FIAS_regbase<...>(...);
    } else {
        // IFA 模板 (Incremental Flash Attention)
        incre_flash_attention_FIAS_regbase<...>(...);
    }
}
```

- `quantMode=31 >= 15` → 走 PFA 路径
- `emptyTensor=false` → 不做空 tensor 短路返回
- printf 输出: 13 个模板参数的完整编码

### 1.3 PFA Regbase 入口

**文件**: `attention/prompt_flash_attention/op_kernel/arch35/prompt_flash_attention_entry_regbase.h`

```cpp
template<...>
inline __aicore__ void prompt_flash_attention_FIAS_regbase(...)
{
    if constexpr (emptyTensor == true) {
        INVOKE_PFA_ZERO_OP_IMPL_V2(half);
        return;
    }
    #if (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT16 && ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
        if constexpr (isReconstructTemp == true) {
            run_fia_noquant_gqa_kernel<half, half, inOutLayoutType, config, ...>(...);
        } else {
            PARSE_PARAMS_NoQuant(...);
            INVOKE_PFA_NOQUANT_GENERAL_OP_IMPL_ASCEND950_FA_BASEAPI(...);
        }
    #endif
}
```

- `isReconstructTemp=true` + FP16 → `run_fia_noquant_gqa_kernel<half, half, ...>`

### 1.4 模板分发器 - 组装 Kernel 类型

**文件**: `attention/common/op_kernel/arch35/fia_template_dispatcher.h`

```cpp
template <typename INPUT_T, typename OUT_T, uint8_t inOutLayoutType, uint16_t config, ...>
inline __aicore__ void run_fia_noquant_gqa_kernel(...)
{
    // 1. 解析 config 字段 → tile 尺寸枚举
    PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, ...);
    //   → s1TemplateType=S128, s2TemplateType=S128, dTemplateType=D128, dVTemplateType=D128
    //   → inputLayout=BNSD, outputLayout=BNSD

    // 2. 编译期决策
    constexpr bool useDn = false;
    constexpr bool bmm2Write2Ub = false;
    constexpr bool splitD = false;

    // 3. 按 g_coreType 选 Block 类型 (AIC/AIV 双路)
    using CubBlock = std::conditional<g_coreType==AIC,
        FANoQuantGqaBlockCube<...>, FANoQuantGqaBlockCubeDummy<...>>::type;
    using VecFaBlock = std::conditional<g_coreType==AIV,
        FANoQuantGqaBlockVec<...>, FANoQuantGqaBlockVecDummy<...>>::type;
    using VecFdBlock = std::conditional<g_coreType==AIV,
        FiaBlockVecFlashDecode<...>, FiaBlockVecFlashDecodeDummy<...>>::type;

    // 4. 实例化 Kernel 类
    using Kernel = FlashAttentionNoQuantGqaKernel<CubBlock, VecFaBlock, VecFdBlock>;

    // 5. 解析 TilingData → NoQuantTilingArch35
    GET_TILING_DATA_MEMBER(FusedInferAttentionScoreTilingData, baseTiling, baseTilingIn, tiling);
    const NoQuantTilingArch35 *tilingData = &baseTilingIn;

    // 6. 初始化 + 执行
    Kernel op;
    op.Init(...);
    op.Process();
}
```

**AIC/AIV 双路设计原理**: Ascend 每个 core 要么是 Cube(AIC) 要么是 Vector(AIV)。编译期用 `g_coreType` + `std::conditional` 在 AIC 上实例化真实的 CubeBlock + 空的 VecBlock，AIV 上反之。所有 core 编译同一份代码但实际执行不同分支。

### 1.5 Init 初始化

**文件**: `attention/common/op_kernel/arch35/fia_kernel_noquant_gqa.h`

#### Init() 主函数

```cpp
void Init(query, key, value, pse, attenMask, actualSeqLengths, actualSeqLengthsKv,
          blockTable, queryPaddingSize, kvPaddingSize, postQuantScale, postQuantOffset,
          keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope,
          learnableSink, softmaxLse, attentionOut, workspace, fiaMetaData, tiling, tPipe)
{
    printf("[FIA KERNEL] noquant_gqa Init enter: blk=%u isAIC=%d isAIV=%d\n", ...);
    InitConstInfo(queryPaddingSize, kvPaddingSize, actualSharedPrefixLen);
    InitMMResBuf(workspace);

    if (ASCEND_IS_AIV) { vecFaBlock.InitVecBlock(tPipe, pse, ...); vecFaBlock.ClearOutput(); }
    if (ASCEND_IS_AIC) { cubeBlock.InitCubeBlock(tPipe, &l1BufferManager, query, key, value, ...); }
    if constexpr (FLASH_DECODE) {
        if (ASCEND_IS_AIV) { vecFdBlock.InitParams(); vecFdBlock.InitGlobalTensor(...); }
    }
    printf("[FIA KERNEL] noquant_gqa Init done: blk=%u B=%u N2=%u S1=%u S2=%u D=%u G=%u coreNum=%u\n", ...);
}
```

#### InitConstInfo() - 加载 Tiling 参数

从 `NoQuantTilingArch35` 解析所有维度/属性到 `ConstInfoX`:

| 来源 | 字段 | 说明 |
|------|------|------|
| fiaBaseParams | bSize, n2Size, gSize, s1Size, s2Size, dSize, dSizeV | 张量维度 |
| fiaBaseParams | actualSeqLenSize, actualSeqLenKVSize | 实际序列长度 |
| fiaBaseParams | scaleValue, coreNum, outputLayout | 缩放因子, 核数, 输出布局 |
| fiaBaseParams | keyStrides, valueStrides | KV 跨 batch/head 步长 |
| fiaAttenMaskParams | sparseMode, preTokens, nextTokens | Mask 稀疏模式 |
| fiaAttenMaskParams | isRowInvalidOpen, isExistRowInvalid | 行掩码开关 |
| fiaPseParams | pseShiftByBatch, pseS1Size, pseS2Size | PSE 参数 |
| fiaSystemPrefixParams | prefixSeqInnerSize, isActualSharedPrefixLenNull | 共享前缀 |
| fiaPageAttentionParams | maxBlockNumPerBatch, blockSize, paLayoutType | PA 参数 |
| fiaLeftPaddingParams | isQHasLeftPadding, isKVHasLeftPadding | 左填充 |
| fiaPostQuantParams | isPostQuantPerChnl, isPostQuantBF16 | 后量化 |
| fiaWorkspaceParams | accumOutSize, logSumExpSize | 工作空间 |
| fiaS1OuterSplitCoreParams | enableS1OutSplit, totalSize | S1 外切 |
| fiaMetaData (GM) | bN2Start/End, gS1OStart/End, s2OStart/End | **每核算力分配** |
| fiaMetaData (GM) | coreFirstTmpOutWsPos | FlashDecode 工作空间索引 |

#### InitMMResBuf() - 分配缓冲区

```
L1  Buffer: l1PBuffers (3-buff, mBaseSize × s2BaseSize × sizeof(T))
            ↓ 存放 P = softmax(Q×K^T×scale) 矩阵

UB  Buffer: bmm1Buffers (DB, mBaseSize/CV_RATIO × s2BaseSize × sizeof(T))
            ↓ 存放 Q×K^T 结果 (BMM1), AIC写→AIV读

UB  Buffer: bmm2Buffers (DB) 或 跳过
            ↓ BMM2结果缓存 (仅 bmm2Write2Ub=true)

GM  Buffer: bmm2ResGmBuffers (3-buff, mBaseSize × dBasicBlock × sizeof(T))  [仅 !bmm2Write2Ub]
            ↓ 存放 P×V 结果 (BMM2), AIC写→AIV读, 每核独立地址
```

### 1.6 Process 主控

```cpp
void Process()
{
    printf("[FIA KERNEL] noquant_gqa Process enter: blk=%u coreNum=%u aicIdx=%u fd=%d\n", ...);

    if (constInfo.aicIdx < constInfo.coreNum) {
        CrossCoreBufferInit();        // AIV: 标记 bmm1/2Buf 为跨核同步
        vecFaBlock.InitBuffers();     // AIV: 分配 softmax/vec/tscm 缓冲
        cubeBlock.InitBuffers();      // AIC: 分配 L0 matmul 缓冲
        AllocEventID();

        FlashAttention();             // ★ 核心计算 ★

        FreeEventID();
        CrossCoreBufferUnInit();      // AIC: WaitCrossCore 回收
    }

    if constexpr (FLASH_DECODE) {
        if (ASCEND_IS_AIV) {
            printf("[FIA KERNEL] noquant_gqa FlashDecode start: blk=%u\n", ...);
            FlashDecode();
            printf("[FIA KERNEL] noquant_gqa FlashDecode end: blk=%u\n", ...);
        }
    }
    printf("[FIA KERNEL] noquant_gqa Process end: blk=%u\n", ...);
}
```

### 1.7 FlashAttention - 任务调度主循环

```cpp
void FlashAttention()
{
    printf("[FIA KERNEL] noquant_gqa FA enter: blk=%u "
           "bN2=[%u,%u) M=[%u,%u) s2=[%u,%u) enableS1OS=%d\n", ...);

    RunInfoX taskRunInfo[3] = {};     // 3槽预取缓存 (PRELOAD_N=2)
    uint32_t bN2Cur = constInfo.bN2Start;
    uint32_t gS1Cur = constInfo.gS1OStart;
    uint32_t s2Cur  = constInfo.s2OStart;

    while (shouldDispatchTask || shouldExecuteTask) {
        shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
        if (shouldDispatchTask) {
            TASK_DEAL_MODE mode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
            switch (mode) {
            case CREATE_TASK:
                CreateTask(createdTaskCount, bN2Cur, gS1Cur, s2Cur, taskRunInfo);
                createdTaskCount++;
                UpdateAxisInfo(mode, bN2Cur, gS1Cur, s2Cur);
                break;
            case DEAL_ZERO:
                vecFaBlock.DealZeroActSeqLen(bN2Cur);
                UpdateAxisInfo(mode, bN2Cur, gS1Cur, s2Cur);
                continue;
            default: // SKIP / NOT_START / S2_END
                UpdateAxisInfo(mode, bN2Cur, gS1Cur, s2Cur);
                continue;
            }
        }
        shouldExecuteTask = ShouldExecuteTask(taskRunInfo);
        if (shouldExecuteTask) {
            ExecuteTask(executedTaskCount, taskRunInfo);
            executedTaskCount++;
        }
    }
}
```

#### GetTaskDealMode() 核心逻辑

```
1. 换 batch 时读取 actSeqLensQ[], actSeqLensKV[]
2. 计算: gS1LoopTimes = ceil(actSeqLensQ×G/mBaseSize), s2LoopTimes = ceil(actSeqLensKV/s2BaseSize)
3. 零序列 → DEAL_ZERO
4. Padding溢出 → DEAL_ZERO
5. CalcCurS2StartEndNoSparse() / CalcCurS2StartEndWithSparse() 计算有效S2范围
6. s2Cur在区间外 → NOT_START / SKIP / S2_END
7. s2Cur在区间内 → CREATE_TASK
```

#### CreateTask → CalcParams()

填充 `RunInfoX` 结构体:

```cpp
struct RunInfoX {
    uint64_t loop, mloop;
    uint32_t bIdx, n2Idx, gS1Idx, s1Idx, s2Idx;
    uint64_t actS1Size, actS2Size, actMSize, actSingleLoopS2Size;
    int64_t  preTokensLeftUp, nextTokensLeftUp;
    int64_t  qPaddingBeginOffset, kvPaddingBeginOffset;
    bool     isFirstS2Loop, isLastS2Loop, isS2SplitCore;
    uint32_t vecMbaseIdx, actVecMSize;    // AIV 子块分配
    uint32_t faTmpOutWsPos;
    bool     isValid;
};
```

#### 流水线调度 (PRELOAD_N=2)

```
Loop 0: 创建Task0 → MM1/Vec1
Loop 1: 创建Task1 → MM1/Vec1
Loop 2: 创建Task2 → MM1/Vec1  +  MM2/Vec2(Task0)  ← 重叠!
Loop 3: 创建Task3 → MM1/Vec1  +  MM2/Vec2(Task1)
...
  Loop N: Task[N] 的 MM1/Vec1  +  Task[N-2] 的 MM2/Vec2
```

### 1.8 BMM1: Q×K^T

**文件**: `attention/common/op_kernel/arch35/fia_block_cube_noquant_gqa.h`

```
IterateBmm1NdL0Split() 流程:
  1. 加载 Q tile: GM→UB→L0A, Q[bIdx,n2Idx][gS1+0..mBaseSize][0..dSize], 16B对齐
  2. 加载 K tile: GM→UB→L0B, K[bIdx,n2Idx×G][s2Cur..s2BaseSize][0..dSize]
     [RoPE] vector RoPE 处理K
  3. MatMul: Q[mBaseSize×dBaseSize] × K^T[dBaseSize×s2BaseSize] × scale → S
  4. 结果写入 bmm1Buffers (UB), SetCrossCore()→AIV
```

### 1.9 Vec1: Softmax + PSE + Mask

**文件**: `attention/common/op_kernel/arch35/fia_block_vec_noquant_gqa.h`

```
ProcessVec1Nd() 流程:
  1. WaitCrossCore, 从UB读S矩阵, 转置
  2. 按 s2BaseSize 分派 VF API
  3. ProcessVec1Vf():
     Step 1: Online Softmax Flash V2 (Mul+Select+SoftmaxFlash+Cast+NZ)
     Step 2: Apply PSE (if pseMode!=NONE): pseShift GM add
     Step 3: Apply Mask (if hasAttenMask): sparse Band/Causal/Tree
     Step 4: Apply Dropout (if hasDrop)
     Step 5: 写P到L1 outputBuf, SetCrossCore()→AIC
```

### 1.10 BMM2: P×V

**文件**: `attention/common/op_kernel/arch35/fia_block_cube_noquant_gqa.h`

```
IterateBmm2l0Split() 流程:
  1. 加载 P tile: L1→L0A, P[mBaseSize×s2BaseSize]
  2. 加载 V tile: GM→UB→L0B, V[bIdx,n2Idx×G][s2Cur..s2BaseSize][0..dVBaseSize]
     [KV Prefix] 先加载prefix V
  3. MatMul: P[mBaseSize×s2BaseSize] × V[s2BaseSize×dVBaseSize] → O_patch
  4. 结果: BMM2_TOUB=true→bmm2Buffers(UB), false→bmm2ResGmBuffers(GM 3-buff)
```

### 1.11 Vec2: 输出累加与写回

**文件**: `attention/common/op_kernel/arch35/fia_block_vec_noquant_gqa.h`

```
ProcessVec2_C2GmUpdateUb() 流程:
  1. DataCopy: GM→UB (O_patch)
  2. VF FlashUpdate: softmax rescale + 累加 (利用max/sum修正)
  3. VF Div+Cast: O = accumulated_sum / softmax_sum × scale
  4. [postQuant] 量化 (FP16→INT8)
  5. 写回: isLastS2Loop → attentionOut GM; 中间 → accumOut workspace
```

### 1.12 FlashDecode - 跨核算力归约

**文件**: `attention/common/op_kernel/arch35/fia_block_vec_flashdecode.h`

仅在 AIV + `fdCoreEnable=1` 时执行:

```
1. 从 fiaMetaData 读 FD 参数: fdBN2Idx, fdMIdx, fdS2SplitNum, mStart, mLen, workspaceIdx
2. 遍历 S2 切块: 从 accumOut 读 partial O, 从 softmaxFDMax/SumGm 读 max/sum
3. VF FlashDecode: rescale + sum weighted partials
4. 最终 rescale: O = sum / sum_global × scale
5. 写入 attentionOut GM; [LSE] 写 softmaxLse; [Sink] 合并 learnableSink
```

### 1.13 Buffer 与 AIC/AIV 通信架构

```
┌──────────────────────────────────────────────────────────────┐
│                        GM (HBM)                               │
│  Q[b,n2,G×S1,D]  K[b,n2,G×S2,D]  V[b,n2,G×S2,DV]           │
│  attentionOut[b,n2,G×S1,DV]  accumOut[...]  softmaxLse[...]  │
│  fiaMetaData [FA: AIC任务分配] [FD: AIV归约参数]             │
└─────┬──────────────┬──────────────────┬──────────────────────┘
      ▼              ▲                  ▼
┌──────────┐  ┌──────────┐  ┌──────────────────┐
│ AIC Core │  │ AIV Core │  │ 共享 Buffer       │
│ L0A:Q/P  │──│ L0C:scal │  │ L1:l1PBuf[3]     │
│ L0B:K/V  │  │ VF:      │  │ UB:bmm1Buf[DB]   │
│ MMAD:    │  │ Softmax  │  │ UB:bmm2Buf[DB]   │
│ QK^T,PV  │  │ PSE+Mask │  │ GM:bmm2ResGm[3]  │
└──────────┘  └──────────┘  └──────────────────┘
      │           │                  │
      └───────────┴──────────────────┘
         CrossCore Sync: Set/Wait (UB→L1→UB)
```

### 1.14 用例参数统计

| 参数 | 值 | 说明 |
|------|------|------|
| B (batch) | 2 | |
| N2 (kv_heads) | 1 | |
| G (q_heads/kv_heads) | 2 | GQA ratio=2 |
| S1 (q_seq) | 1 | decode 场景 |
| S2 (kv_seq) | 1024 | KV cache 长度 |
| D (head_dim) | 128 | |
| usedCoreNum | 28 | |
| blockDim | 28 AIC + 56 AIV | 每 AIC 配 2 AIV |
| mBaseSize | 128 | M 方向 tile |
| s2BaseSize | 128 | S2 方向 tile |
| dBaseSize | 128 | D 方向 tile |
| Workspace | ~18MB | GM 工作空间 |

**每核任务量**: B×N2×G = 4 行, 每行 ceil(1024/128)=8 个 S2 切块, 共 32 任务, 28 核分配.

---

## 二、内存分配

### 2.1 总 Workspace (Host 分配)

**文件**: `fia_tiling_nonquant_gqa.cpp:819 CalcWorkspaceSize`

```cpp
workspaceSize = sysWorkspaceSize                          // 系统 (~16MB)
              + (bmm2Bytes + vec2Bytes) × 3 × numBlocks  // BMM2/vec2 GM缓冲
              + (accumOut + 2×logSumExp) × sizeof(float)  // FlashDecode
```

本用例各变量:

| 项目 | 条件 | 计算 | 结果 |
|------|------|------|------|
| sysWorkspaceSize | 固定 | ~16MB | ~16MB |
| bmm2Bytes | dSize=128, !>DSIZE_128 | 0 | 0 |
| vec2Bytes | dVBasicBlock=128 | 0 | 0 |
| accumOut | fd=true | numBlocks×2×mSize×dSize×4 | 1.75MB |
| logSumExp×2 | fd=true | numBlocks×2×mSize×8×4×2 | 0.22MB |
| **总计** | | | **~18MB (18841600 bytes)** |

> mSize(host)=sOuterFactor_×cvRatio=32×2=64, dSize=128
> bmm2Bytes=0 → 本用例 BMM2_TOUB=true

### 2.2 硬件内存规格 (Ascend 950 / arch35)

| 层级 | 容量/核 | 总计(84核) | 延迟 | 用途 |
|------|---------|-----------|------|------|
| L0A (AIC) | 64KB | 1.75MB | 极低 | Cube MatMul 左矩阵 Q/P |
| L0B (AIC) | 64KB | 1.75MB | 极低 | Cube MatMul 右矩阵 K/V |
| L0C (AIV) | 256KB | 14MB | 极低 | Vector 标量计算 |
| UB (Unified) | ~248KB | ~20.3MB | 低 | 通用向量缓冲,跨核共享 |
| L1 | 512KB | 42MB | 中 | 片上缓存 DataCopy |
| L2 | 128MB | 统一 | 高 | 片外缓存 |
| GM (HBM) | — | 设备决定 | 最高 | 全局内存 |

### 2.3 每核 Buffer 分配

#### L1 (512KB pool)

| Buffer | 类型 | 计算 | 大小 |
|--------|------|------|------|
| l1PBuffers[0] | 3-buff | mBaseSize×s2BaseSize×sizeof(half) | 128×128×2=32KB |
| l1PBuffers[1] | 3-buff | 同上 | 32KB |
| l1PBuffers[2] | 3-buff | 同上 | 32KB |
| **已用** | | | **96KB (19%)** |
| 剩余 | | | 416KB |

#### UB (~248KB pool, BMM2_TOUB=true)

| Buffer | 类型 | 计算 | 大小 |
|--------|------|------|------|
| bmm1Buf[0] | DB | mBaseSize/CV_RATIO×s2BaseSize×sizeof(float) | 64×128×4=32KB |
| bmm1Buf[1] | DB | 同上 | 32KB |
| bmm2Buf[0] | DB | mBaseSize/CV_RATIO×dVBaseSize×sizeof(float) | 64×128×4=32KB |
| bmm2Buf[1] | DB | 同上 | 32KB |
| **已分配** | | | **128KB** |
| VF scratch | | | ~64KB |
| TSCM | | | ~32KB |
| Event/overhead | | | ~24KB |
| **总占用** | | | **~248KB (100%)** |

#### GM Workspace 布局

```
Offset 0:
  ┌─────────────────────────────────┐
  │     sysWorkspace (~16MB)        │  系统: events, TSCM, runtime
  ├─────────────────────────────────┤
  │     accumOut (1.75MB)           │  FD partial O (float)
  │     28×2×64×128×4 bytes        │
  ├─────────────────────────────────┤
  │     softmaxFDSum (0.11MB)       │  FD sum (float)
  ├─────────────────────────────────┤
  │     softmaxFDMax (0.11MB)       │  FD max (float)
  └─────────────────────────────────┘
Offset 18841600
```

### 2.4 Cube MatMul 内存数据流

**BMM1 (Q×K^T)**:
```
Q[128×128×2=32KB, half] →UB → L0A  |  K^T[128×128×2=32KB, half] →UB → L0B
                                    └→ S[64×128×4=32KB, float] → UB bmm1Buf
```

**BMM2 (P×V)**:
```
P[64×128×4=32KB, float] → L0A      |  V[128×128×2=32KB, half] → UB → L0B
                                    └→ O[64×128×4=32KB, float] → UB bmm2Buf
```

### 2.5 FlashDecode 内存数据

| 数据 | 来源 | 大小 (per core) |
|------|------|----------------|
| softmaxFDMax | GM workspace | 2×64×8×4 = 4KB (float) |
| softmaxFDSum | GM workspace | 4KB (float) |
| accumOut partial | GM workspace | 2×64×128×4 = 64KB (float) |
| attentionOut | GM output | 64×128×2 = 16KB (half) |
| sink (条件) | GM | 16KB (half) |
| softmaxLse (条件) | GM | 64×4 = 256B (float) |

### 2.6 内存全景图

```
┌──────────────────────────────────────────────────────────────┐
│                       GM (HBM)                                │
│  Q:1KB  K:512KB  V:512KB  Out:1KB  Workspace:18MB           │
└──┬───────┬───────┬──────────────────────────────────────────┘
   │       │       │
   ▼       ▼       ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│     AIC Core (×28)       │  │     AIV Core (×56)       │
│                          │  │                          │
│  L1  512KB   (19% used)  │  │  L1  512KB   (19% used)  │
│  L0A 64KB    (Q/P tile)  │  │  L0C 256KB   (scalar)   │
│  L0B 64KB    (K/V tile)  │  │                          │
│  UB  ~248KB  (100% used)  │  │  UB  ~248KB  (100% used)  │
│  ├ bmm1Buf[2] 64KB       │  │  ├ bmm1Buf[2] 64KB       │
│  ├ bmm2Buf[2] 64KB       │  │  ├ bmm2Buf[2] 64KB       │
│  └ TSCM       ~32KB      │  │  ├ VF scratch  ~64KB     │
│                          │  │  └ Event etc   ~24KB     │
└──────────────────────────┘  └──────────────────────────┘
         ↕ CrossCore Sync (UB/L1)
```

### 2.7 关键内存决策

| 决策 | 条件 | 本用例 | 影响 |
|------|------|--------|------|
| BMM2_TOUB | dV≤192, 无PSE/Mask | true | BMM2存UB, 免GM来回 |
| splitD | dV>256 | false | D不拆分 |
| useDn | Mask/PSE/大S1 | false | 不用Downward |
| CV_RATIO | arch35 | 2 | UB存储粒度 |
| bmm1 DB | 128/2×128×4 | 32KB | QK^T缓存 |
| l1PBuf | 128×128×2 | 32KB | P矩阵缓存 |

### 2.8 内存瓶颈分析

```
UB 是最大瓶颈:
  已用: bmm1×2(64) + bmm2×2(64) + VF(64) + TSCM(32) = 224KB / 248KB → 90%+
  剩余 ~24KB → 无法支持更大 tile (如S2=256)

L1 有充足余量:
  已用: l1PBuf×3 = 96KB / 512KB → 19%
  剩余 416KB → 可做预取优化, 或用L1分担UB压力

GM 带宽:
  BMM1: Q(32KB) + K(32KB) = 64KB/task
  BMM2: V(32KB) = 32KB/task
  每核 ~6 FA tasks + 2 FD tasks → ~0.5MB/核
  28核 → ~14MB GM读写总量
```
---

## 三、四阶段内存加载与分块策略

### 3.1 单 Task 定义

一个 `task` (RunInfoX) 处理: **1个 bN2行 × 1个 M切块 × 1个 s2 tile**

```cpp
// fia_kernel_noquant_gqa.h:701 CalcParams
info.bIdx   = bN2Cur / constInfo.n2Size;   // batch索引
info.n2Idx  = bN2Cur % constInfo.n2Size;   // head索引
info.gS1Idx = gS1Cur * mBaseSize;          // G×S1起始偏移
info.s2Idx  = s2Cur * s2BaseSize;          // S2 token起始偏移
info.actMSize              = ...;          // 当前M块实际行数
info.actSingleLoopS2Size   = ...;          // 当前S2块实际列数
```

本用例中各维度 block 拆分:

```
B=2, N2=4 → 8个 bN2行
gS1Size=64 < mBaseSize=128 → 1个 M切块
S2=9856, s2BaseSize=128 → 77个 s2 tile

每 bN2 行: 77 tasks (77个 s2 tile)
总计: 8 × 77 = 616 tasks, 28核分担 → ~22 tasks/核
```

---

### 3.2 Stage 1: BMM1 (Q×K^T)——GM→L1→L0A/L0B→MatMul→UB

#### L1 Buffer 分配

**文件** `fia_block_cube_noquant_gqa.h`

```cpp
// line 42-52: Q/K/V 的 L1 Buffer 类型选择
struct QL1BuffSel {
    using Type = std::conditional_t<(dBaseSize > 256),
        BuffersPolicySingleBuffer<BufferType::L1>,    // D>256: 单buf + 驻留
        BuffersPolicyDB<BufferType::L1>>;              // D≤256: DB 2槽  ← 本用例
};

struct KVL1BuffSel {
    using Type = std::conditional_t<(s2BaseSize == 256 && dBaseSize > 128),
        BuffersPolicySingleBuffer<BufferType::L1>,    // 大tile: 单buf
        BuffersPolicyDB<BufferType::L1>>;              // 常规: DB 2槽  ← 本用例
};

// line 175-181: 本用例 InitBuffers (D=128 ≤ 256 → else分支)
constexpr uint32_t mm1LeftSize  = mBaseSize * dBaseSize * sizeof(half)  = 128×128×2 = 32KB;  // Q tile
constexpr uint32_t mm1RightSize = dBaseSize * s2BaseSize * sizeof(half) = 128×128×2 = 32KB;  // K tile
l1QBuffers.Init(*l1BufferManagerPtr, 32KB);   // DB 2槽, Q
l1KBuffers.Init(*l1BufferManagerPtr, 32KB);   // DB 2槽, K
```

#### Q 从 GM 搬运到 L1

**文件** `fia_block_cube_noquant_gqa.h`

```cpp
// line 579-587: Q tile 搬运, 首次搬 vs 复用
if (unlikely(runInfo.isFirstS2Loop)) {
    mm1A = l1QBuffers.Get();          // 第1个s2 tile: Get()取新槽, GM→L1搬运
    CopyQueryTile(mm1ATensor, runInfo);
} else {
    mm1A = l1QBuffers.GetPre();       // 后续s2 tile: GetPre()取上次的槽, 复用!
    // ← 不搬运! Q驻留L1
}

// line 352-360: CopyQueryTile → CopyQuerySlice
// line 308-350: CopyQuerySlice
void CopyQuerySlice(dstTensor, dOffset, dRealSize, runInfo) {
    FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};
    GmCoord gmCoord{.bIdx = runInfo.bIdx, .n2Idx = runInfo.n2Idx,
                    .gS1Idx = runInfo.gS1Idx, .dIdx = dOffset,
                    .gS1DealSize = runInfo.actMSize, .dDealSize = nopeDealSize};
    copyQueryGmToL1(l1Tensor, queryGm, gmCoord);  // GM Q → L1 (DataCopy)
}
```

**搬运规模**: 每 s2 tile 第一行时搬 Q[64行×128列×2B]=16KB (实际搬满 mBaseSize=128 行=32KB, 无效行填 0)
**复用**: 同一 bN2 行跨 s2 tile 时不再搬 Q

#### K 从 GM 搬运到 L1 (每次 s2 tile 都要搬)

```cpp
// line 589-597
Buffer<BufferType::L1> mm1B = l1KBuffers.Get();  // 每次s2 tile都Get
mm1B.Wait<HardEvent::MTE1_MTE2>();
CopyKeyTile(mm1BTensor, runInfo);                 // K tile: 32KB

// line 426-433: CopyKeyTile → CopyKeySlice
void CopyKeySlice(dstTensor, dOffset, dRealSize, runInfo) {
    GmKvCoord gmCoord{.bIdx = ..., .n2Idx = runInfo.n2Idx,
                      .s2Idx = runInfo.s2Idx,      // ← 每个s2 tile不同!
                      .s2DealSize = runInfo.actSingleLoopS2Size, .dDealSize = dRealSize};
    copyKvGmToL1(l1Tensor, keyGm, gmCoord);         // GM K → L1
}
```

#### MatMul

```cpp
// line 601-620
Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
MMParam param = MakeMMParam(actMSize, actSingleLoopS2Size, dSize, false, true);

// D≤128 → full tile matmul
MatmulBase<Q_T, KV_T, T, 128, 128, dBaseSize, ABLayout::MK, ABLayout::KN>(
    mm1A.GetTensor<Q_T>(),   // L1 Q
    mm1B.GetTensor<KV_T>(),  // L1 K
    mmL0ABuffers, mmL0BBuffers,
    mm1ResL0C.GetTensor<T>(), param);   // L0C → S[128×128]

// line 629-634: Fixpipe L0C→UB + cross-core sync
outputBuf.WaitCrossCore();
FixpipeMm1(outputBuf.GetTensor<T>(), mm1ResL0C.GetTensor<T>(), runInfo);
outputBuf.SetCrossCore();     // 通知AIV: BMM1结果就绪
```

**Q复用策略总结**:
```
Task0(s2 tile 0): l1QBuffers.Get()    → slot A: GM→L1, 搬Q[128×128]
Task1(s2 tile 1): l1QBuffers.GetPre() → slot A: 复用, 不搬
Task2(s2 tile 2): l1QBuffers.GetPre() → slot A: 复用, 不搬
换bN2行时:        l1QBuffers.Get()    → slot B: 新Q
```

---

### 3.3 Stage 2: Vec1 (Online Softmax)——UB→寄存器→L1

**全线 UB→寄存器, 无 GM 搬运**

**文件** `fia_block_vec_noquant_gqa.h`

```cpp
// line 217-233: ProcessVec1 入口
void ProcessVec1(outputBuf, bmm1ResBuf, runInfo) {
    bmm1ResBuf.WaitCrossCore();           // 等AIC完成BMM1
    if (useDn) ProcessVec1Dn(...);
    else       ProcessVec1Nd(...);        // 本用例走这里
}

// line 484-641: ProcessVec1Nd
void ProcessVec1Nd(outputBuf, bmm1ResBuf, runInfo) {
    // 申请 UB 中的临时缓冲区
    LocalTensor<float> sumUb = softmaxSumBuf[mloop % 3].Get<float>();
    LocalTensor<float> maxUb = softmaxMaxBuf[mloop % 3].Get<float>();
    LocalTensor<float> expUb = softmaxExpBuf[loop  % 3].Get<T>();

    // 读取 BMM1 结果 (已在 UB 中, cross-core sync 后可用)
    LocalTensor<T> mmRes = bmm1ResBuf.GetTensor<T>();   // S矩阵[64行×128列]

    auto stage1CastTensor = stage1OutQue[stage1Offset].AllocTensor<INPUT_T>();

    // isFirstS2Loop → NoUpdate, else → Update (在线修正)
    if (isFirstS2Loop)
        ProcessVec1Vf<isUpdate=false>(mmRes, ...);   // 初值
    else
        ProcessVec1Vf<isUpdate=true>(mmRes, ...);    // 在线修正

    bmm1ResBuf.SetCrossCore();       // 释放UB, AIC可写下一个BMM1结果

    // P(half)写入L1
    DataCopy(mm2AL1Tensor[offset], stage1CastTensor, dataCopyParams);
    outputBuf.SetCrossCore();        // 通知AIC: P就绪

    if (!isFirstS2Loop)
        UpdateExpSumAndExpMax(...);  // 在线修正 running max/sum
}
```

#### AIV 行分块 (Row-Loop 策略)

**文件** `fia_kernel_noquant_gqa.h:701 CalcParams`

```cpp
// line 762-771: M 维度的 AIV 子块拆分
info.actMSize = min(gS1Size - gS1Cur*mBaseSize, mBaseSize);  // = 64行
info.actVecMSize = (info.actMSize + 1) >> 1;                  // = 32行 (一半)
info.vecMbaseIdx = 0;                             // AIV0 从第0行开始
if (constInfo.subBlockIdx == 1) {
    info.vecMbaseIdx = info.actVecMSize;           // AIV1 从第32行开始
    info.actVecMSize = info.actMSize - info.actVecMSize;  // 剩余32行
}
```

**AIV0 处理行 0..31, AIV1 处理行 32..63**, 并行执行。

#### Online Softmax 寄存器级循环

**文件** `vf_basic_block_aligned128_no_update.h:28 ProcessVec1NoUpdateImpl128VF`

```cpp
// Phase 1: 逐行求 local_max (m=32次循环, line 105-193)
for (uint16_t i = 0; i < m; ++i) {
    LoadAlign(vreg_input_x,        srcUb + i * s2BaseSize);         // 前64列
    LoadAlign(vreg_input_x_unroll, srcUb + i * s2BaseSize + 64C);  // 后64列
    Muls(vreg_input_x, vreg_input_x, dScale);        // ×scale

    Max(vreg_max_tmp, vreg_input_x, vreg_input_x_unroll);  // 前后64取大
    Reduce<MAX>(vreg_input_max, vreg_max_tmp);              // 64→1归约
    StoreUnAlign(POST_MODE_UPDATE) → tmpMaxUb[i];
}
// Phase 2: 逐行 ExpSub + Cast (m=32次循环, line 201-343)
LocalMemBar();
for (uint16_t i = 0; i < m; ++i) {
    LoadAlign<BRC_B32>(vreg_max_brc, maxUbStart + i);     // 广播max到64 lane
    LoadAlign<DINTLV_B32>(vreg_input_x, vreg_input_x_unroll, srcUb + i*128);

    ExpSub(vreg_exp_even, vreg_input_x, vreg_max_brc);          // 前64列 exp
    ExpSub(vreg_exp_odd,  vreg_input_x_unroll, vreg_max_brc);  // 后64列 exp

    Add(vreg_exp_sum, vreg_exp_even, vreg_exp_odd);       // 前后64求和
    Reduce<SUM>(vreg_exp_sum, vreg_exp_sum);               // 64→1归约
    StoreUnAlign(POST_MODE_UPDATE) → expSumUb[i];

    Cast<half, float>(vreg_exp_f16, vreg_exp_even/odd);   // float→half
    Or(interleave even/odd);
    StoreAlign<DATA_BLOCK_COPY>(dstTensor[i], vreg_exp_f16); // P[i] → UB
}
```

#### Update 路径: local vs running 一行指令比较

**文件** `vf_basic_block_aligned128_update.h:28 ProcessVec1UpdateImpl128VF`

```cpp
// line 201-206: 32行 local_max vs running_max, 1条向量指令
LoadAlign(vreg_in_max, inMaxUb);                              // running_max[0..31]
LocalMemBar();
LoadAlign(vreg_input_max, tmpMaxUb2);                         // local_max[0..31]
Max(vreg_max_new, vreg_input_max, vreg_in_max, preg_all);    // ★ 32次比较, 1条指令 ★
    // new_max[i] = max(local_max[i], running_max[i])
StoreAlign(DIST_NORM_B32)(tmpMaxUb2, vreg_max_new);          // 存回merged_max

// line 219-230: 用 merged_max 做 ExpSub (逐行循环)
for (uint16_t i = 0; i < m; ++i) {
    LoadAlign<BRC_B32>(vreg_max, tmpMaxUb2 + i);       // 广播第i行merged_max
    ExpSub(vreg_exp_even, vreg_input_x, vreg_max);      // exp(S - merged_max[i])
    ...
}
```

#### UpdateExpSumAndExpMax: 在线 sum 修正

**文件** `vf_mul_sel_softmaxflashv2_cast_nz.h:152 UpdateExpSumAndExpMaxImplVF`

```cpp
// line 189-213: 在线修正 (useNz=false)
LoadAlign(vreg_max, tmpMaxUb);              // 新max (当前s2块的local_max)
LoadAlign(vreg_in_max, inMaxUb);            // 旧max (前面s2块的running_max)
ExpSub(vreg_exp_max, vreg_in_max, vreg_max);// α = exp(old_max - new_max)
StoreAlign(expMaxUb, vreg_exp_max);          // 保存修正因子

LoadAlign(vreg_in_exp_sum, inExpSumUb);      // 旧sum
LoadAlign(vreg_exp_sum_brc, tmpExpSumUb);    // 新sum
Mul(vreg_exp_sum_update, vreg_exp_max, vreg_in_exp_sum); // α × old_sum
Add(vreg_exp_sum_update, vreg_exp_sum_update, vreg_exp_sum_brc); // + new_sum
StoreAlign(expSumUb, vreg_exp_sum_update);   // 更新 running_sum
```

---

### 3.4 Stage 3: BMM2 (P×V)——L1→L0A/L0B→MatMul→UB

#### V 从 GM 搬运到 L1 (每次 s2 tile)

**文件** `fia_block_cube_noquant_gqa.h`

```cpp
// line 907-953: IterateBmm2l0Split
Buffer<BufferType::L1, CROSS_CORE_SYNC_FORWARD> mm2A = inputBuf.Get();  // L1 P
mm2A.WaitCrossCore();                           // 等AIV写完P

Buffer<BufferType::L1> mm2B = l1VBuffers.Get();  // L1 V
mm2B.Wait<HardEvent::MTE1_MTE2>();
CopyValueTile(mm2BTensor, runInfo);              // V tile: 32KB, 每个s2 tile都搬

// line 435-438: CopyValueTile
void CopyValueTile(dstTensor, runInfo) {
    CopyValueSlice(dstTensor, 0, constInfo.dSizeV, runInfo);
}
// line 410-423: CopyValueSlice
void CopyValueSlice(dstTensor, dOffset, dRealSize, runInfo) {
    GmKvCoord gmCoord{.bIdx=..., .n2Idx=..., .s2Idx=runInfo.s2Idx,
                      .s2DealSize=runInfo.actSingleLoopS2Size, .dDealSize=dRealSize};
    copyKvGmToL1(l1Tensor, valueGm, gmCoord);    // GM V → L1
}

// line 932-938: MatMul P × V
MatmulFull<Q_T, KV_T, T, 128, 128, 128, ABLayout::MK, ABLayout::KN>(
    mm2A.GetTensor<Q_T>(),   // L1 P[128×128]
    mm2BTensor,              // L1 V[128×128]
    mmL0ABuffers, mmL0BBuffers,
    mm2ResL0C.GetTensor<T>(), param);  // O[128×128] → L0C
```

**P 加载**: L1 l1PBuffers → L0A (3buff, AIV 通过 cross-core sync 写入)
**V 加载**: GM → L1 → L0B, 每次 s2 tile 重新搬运 32KB (不同 s2 段)
**MatMul 规模**: P[mBaseSize=128, s2BaseSize=128] × V[s2BaseSize=128, dVBaseSize=128]

---

### 3.5 Stage 4: Vec2 (Output 累加与写回)——三条路径 + M维切块 + GM/UB双缓冲

**文件** `fia_block_vec_noquant_gqa.h`

Vec2 负责将 BMM2 的 O_patch 累加到最终输出。有三条路径和两层切块。

#### 3.5.1 入口分发

```cpp
// line 1037-1051: 三条路径分发
void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfoX runInfo) {
    bmm2ResBuf.WaitCrossCore();         // 等AIC完成BMM2

    if constexpr (bmm2Write2Ub) {
        ProcessVec2OnUb(bmm2ResBuf, runInfo);      // 路径1: BMM2结果在UB
    } else if constexpr (splitD) {
        GlobalTensor<T> mmRes = bmm2ResBuf.GetTensor<T>();
        ProcessVec2DSplit(mmRes, runInfo);          // 路径2: D维拆分, 需回写GM
    } else {
        GlobalTensor<T> mmRes = bmm2ResBuf.GetTensor<T>();
        ProcessVec2_C2GmUpdateUb(mmRes, runInfo);   // 路径3: 本用例, BMM2→GM, O累加在UB, 最终写GM
    }
}
```

本用例: **路径3** (`ProcessVec2_C2GmUpdateUb`), BMM2_TOUB=false, splitD=false, D=128.

#### 3.5.2 三层数据结构

**文件** `fia_block_vec_noquant_gqa.h:643-683, 1053-1142`

```
┌─────────────────────────────────────────────────────────────────┐
│                          GM                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ O_patch (输入)│  │ accumOut     │  │ attentionOut (输出)   │  │
│  │ [64×128]     │  │ [flashDecode]│  │ [b×n1×G×S1×D]       │  │
│  │ float, 32KB  │  │ float        │  │ half, 最终输出        │  │
│  └──────┬───────┘  └──────────────┘  └──────────▲───────────┘  │
│         │ DataCopy                                   │            │
├─────────┼────────────────────────────────────────────┼───────────┤
│         ▼             UB                            │            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │            │
│  │ mm2InBuf     │  │ stage2OutBuf │  │ softmax* │  │            │
│  │ [32KB]       │  │ [64×128]     │  │ Buf [256B│  │            │
│  │ O_patch临时  │  │ O累加器      │  │ each]    │  │            │
│  └──────┬───────┘  └──────▲───────┘  │ expMax   │  │            │
│         │                 │          │ expSum   │  │            │
│         └── FlashUpdate ──┘          └──────────┘  │            │
│            + LastDiv                                 │            │
│                                                     │ Cast+Copy  │
│                                                     ▼            │
│                                              ┌──────────────┐    │
│                                              │ attenOut UB  │────┘
│                                              │ (half, 8KB)  │
│                                              └──────────────┘
└─────────────────────────────────────────────────────────────────┘
```

| Buffer | 位置 | 大小 | 用途 |
|--------|------|------|------|
| `mm2InBuf` | UB | 32KB | O_patch 从GM搬入的暂存区 |
| `stage2OutBuf` | UB | 64×128×4=32KB | **O 累加器**——跨s2 tile的 accumulated O |
| `softmaxExpBuf[loop%3]` | UB | 256B | exp 修正因子 (从Vec1传下来的 max 修正) |
| `softmaxSumBuf[mloop%3]` | UB | 256B | running softmax sum (Vec1传下) |
| `softmaxMaxBuf[mloop%3]` | UB | 256B | running max (用于RowInvalid) |
| `attenOut UB` | UB | 64×128×2=16KB | Cast后 half 格式的输出暂存 |

#### 3.5.3 M 维度切块——subMBase 循环

**本用例**: 所有行一次性处理 (subMBase=64 > actVecMSize=32)

但代码设计了通用的 M 维切块:

```cpp
// line 1056-1057: M维切块参数计算
uint32_t subMBase = 8192 / dTemplateAlign64;   // 8192/128 = 64行 (UB一次处理上限)
int64_t vec2LoopLimit = CeilDiv(runInfo.actVecMSize, subMBase);  // ceil(32/64)=1

for (int64_t mOIdx = 0; mOIdx < vec2LoopLimit; mOIdx++) {
    int32_t mIdx = mOIdx * subMBase;          // 当前M块起始行
    uint32_t subMRealsize = subMBase;          // 当前M块实际行数
    if (mOIdx == vec2LoopLimit - 1)
        subMRealsize = runInfo.actVecMSize - mIdx;  // 尾块: 剩余行数

    // ... 处理 subMRealsize 行 × dTemplateAlign64 列
}
```

**切块逻辑**: UB 只能存 `8192字节 / (128×4) ≈ 64行` 的 float 数据。行数超过 64 时, 分多轮处理。本用例 32 行, 一轮搞定。

#### 3.5.4 O_patch 从 GM 搬到 UB——双缓冲流水

```cpp
// line 1062-1063: 硬件事件, GM→UB 和 UB→VEC 双缓冲
event_t mte2ToV = FetchEventID(HardEvent::MTE2_V);    // MTE2→VEC: DMA完成, Vector可读
event_t vToMte2 = FetchEventID(HardEvent::V_MTE2);    // V→MTE2: Vector用完, DMA可写

// line 1075-1104: 双缓冲流水
SetFlag<V_MTE2>(vToMte2);                              // ① 通知DMA: 可以写下一块
WaitFlag<V_MTE2>(vToMte2);                             // ② 等DMA确认

// ③ DMA搬运: GM O_patch → UB mm2InBuf (bmm2Ub)
if (dSizeV == dTemplateAlign64) {                      // D=128=128 → 连续搬运
    DataCopy(bmm2Ub,
        mmRes[subBlockIdx * (actMSize-actVecMSize) * dSizeV + mm2ResInnerOffset],
        subMRealsize * dTemplateAlign64 * sizeof(float));  // 32行×128列×4B = 16KB
} else {
    DataCopyPad(bmm2Ub, mmRes[offset], dataCopyParams); // D不对齐: 跳读搬
}

SetFlag<MTE2_V>(mte2ToV);                              // ④ 通知Vector: DMA完成, 数据就绪
WaitFlag<MTE2_V>(mte2ToV);                             // ⑤ 等DMA确认

// ⑥ Vector处理: bmm2Ub → stage2OutBuf (累加)
```

**双缓冲运作**:
```
时间 →
DMA:    [搬O_patch_N到bmm2Ub] ─等─ [搬O_patch_N+1]  ─等─
Vector: ─等─ [处理bmm2Ub]      ─等─ [处理bmm2Ub]     ─等─
         ←─ MTE2_V 事件同步 ─→          ←─ V_MTE2 事件同步 ─→
```

#### 3.5.5 核心累加逻辑——三条分支

**分支 A: 首个 s2 tile (isFirstS2Loop=true)**

```cpp
// line 1109-1110
if (unlikely(runInfo.isFirstS2Loop)) {
    DataCopy(vec2ResInner, bmm2Ub, vec2CalcSize);
    // stage2OutBuf = O_patch (直接拷贝, 不累加)
}
```

`stage2OutBuf` 直接初始化为第一个 s2 tile 的 O_patch。

**分支 B: 中间 s2 tile (非首非尾)**

```cpp
// line 1111-1119
LocalTensor<T> expUb = softmaxExpBuf[loop % 3].Get<T>();
FlashUpdateNew<T, INPUT_T, OUTPUT_T, 128, false, false>(
    vec2ResInner,    // dst: stage2OutBuf (累加结果)
    bmm2Ub,          // cur: O_patch (当前s2 tile的BMM2结果)
    vec2ResInner,    // pre: stage2OutBuf (之前s2 tile的累加结果)
    expUb, expUb,    // expMax: α修正因子
    subMRealsize, 128, 0.0F, 1.0f);
```

**寄存器公式** (`vf_flashupdate_new.h:26 FlashUpdateBasicVF`):

```
for i in [0, m):                    // 逐行
    Load BRC_B32 expMax[i]          // 广播 α = exp(old_max - new_max)
    for j in [0, dLoops):           // 逐列块 (D=128 → dLoops=2, 每次64浮点)
        Load pre[i*D + j*64]        // 旧O (前面s2块的累加)
        Load cur[i*D + j*64]        // 新O (当前s2块的 O_patch)
        Mul oldScaled = pre × α     // 旧O × 修正因子
        Add acc = oldScaled + cur   // + 新O
        Store → dst[i*D + j*64]     // 存回累加器
```

**物理含义**: `O_acc = O_acc_old × exp(old_max - new_max) + O_new`

旧的最大值被新的更大值超越时, 旧累加需要衰减 (exp(old-new) < 1)。新旧最大值相等时, α = exp(0) = 1, 旧累加不变。

**分支 C: 最后一个 s2 tile (isLastS2Loop=true)**

```cpp
// line 1120-1127
LocalTensor<float> sumUb = softmaxSumBuf[mloop % 3].Get<float>();
FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, 128, false, false>(
    vec2ResInner,    // dst
    bmm2Ub,          // cur: O_patch
    vec2ResInner,    // pre: 累加O
    expUb, expUb,    // expMax
    sumUb,           // expSum: running softmax sum
    subMRealsize, 128, 0.0F, 1.0f);
```

**寄存器公式** (`vf_flashupdate_new.h:201 FlashUpdateLastBasicVF`):

```
for i in [0, m):
    Load BRC_B32 expMax[i]          // α = exp(old_max - new_max)
    Load BRC_B32 expSum[i]          // running softmax sum
    for j in [0, dLoops):
        Load pre[i*D + j*64], Load cur[i*D + j*64]
        Mul oldScaled = pre × α
        Add acc = oldScaled + cur
        Div final = acc / expSum    // ← 多一步除法: 归一化
        Store → dst[i*D + j*64]
```

**物理含义**: `O_final = (O_acc_old × α + O_new) / sum_total`

最后一个 s2 tile 时除以最终的 softmax sum, 完成 FlashAttention 的在线归一化。

#### 3.5.6 最终输出写回——Cast + DataCopy to GM

**仅 isLastS2Loop 时执行**:

```cpp
// line 1130-1138
if (unlikely(runInfo.isLastS2Loop)) {
    // 特殊情况: 只有1个s2 tile (首=尾)
    if (unlikely(runInfo.isFirstS2Loop)) {
        LocalTensor<float> sumUb = softmaxSumBuf[mloop % 3].Get<float>();
        LastDivNew<T, INPUT_T, OUTPUT_T, 128, false>(
            vec2ResInner, vec2ResInner, sumUb,
            subMRealsize, 128, 0.0F);
        // vec2ResInner = vec2ResInner / sumUb  (仅1个tile时没有累加, 直接除)
    }
    CopyOutAttentionOut(runInfo, vec2ResInner, mIdx, subMRealsize);
}
```

`CopyOutAttentionOut` 的调用链 (`line 847-941`):

```
CopyOutAttentionOut(runInfo, vec2ResInner, mIdx, subMRealsize)  [line 847]
  │
  ├── if (FLASH_DECODE && isS2SplitCore):
  │     Bmm2ResForFDCopyOut  → 写 partial O 到 accumOut GM workspace
  │
  └── else:
        Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStart, mDealSize)  [line 685]
          │
          ├── RowInvalid(vec2ResUb, ...)               [line 697, 892]
          │     └── 检查无效行 (稀疏mask), 置为-invalid
          │
          ├── Cast<float, half>(attenOut, vec2ResUb, mDealSize × 128) [line 698]
          │     └── float O → half O, 32行×128列 = 8KB
          │
          ├── PostQuant(runInfo, attenOut, vec2ResUb, ...)  [line 700, 709]
          │     └── (仅 INT8 输出, 本用例跳过)
          │
          ├── SetFlag<V_MTE3> / WaitFlag<V_MTE3>         [line 703-704]
          │     └── 双缓冲: Vector→MTE3 事件同步
          │
          └── Bmm2DataCopyOutTrans(runInfo, attenOut, mStart, mDealSize) [line 706, 928]
                │
                GmCoord{.bIdx, .n2Idx, .gS1Idx=vecMbaseIdx+mStart,
                        .gS1DealSize=mDealSize, .dDealSize=dSizeV}
                CopyAttentionOut: UB → GM (half, 8KB)
```

#### 3.5.7 寄存器级完整流程 (本用例: m=32, d=128, 非尾块)

```
┌───────────────────────────────────────────────────────────────┐
│  每个 s2 tile 的 Vec2 寄存器操作 (FlashUpdateNew)              │
│  m=32行, D=128, dTemplateAlign64=128, dLoops=128/64=2          │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  for i in [0..31]:                  // 逐行循环                │
│    ┌──────────────────────────────────────────────────┐       │
│    │ Load BRC_B32:                                  │       │
│    │   vreg_exp_max ← expMaxUb[i]                    │  ← 广播 α 到 64 lane │
│    │                                                 │       │
│    │ for j in [0..1]:              // 128列=2×64列   │       │
│    │   ┌────────────────────────────────────────┐    │       │
│    │   │ Load: vreg_pre ← preUb[i*128 + j*64] │    │  ← 旧累加O │
│    │   │ Load: vreg_cur ← curUb[i*128 + j*64] │    │  ← 新O_patch │
│    │   │ Mul:  vreg_mul = α × vreg_pre        │    │  ← 旧O衰减 │
│    │   │ Add:  vreg_add = vreg_mul + vreg_cur │    │  ← 累加新O │
│    │   │ Store: dstUb[i*128+j*64] = vreg_add  │    │  ← 存回累加器 │
│    │   └────────────────────────────────────────┘    │       │
│    └──────────────────────────────────────────────────┘       │
│                                                                │
│  总计: 32行 × 2列块 = 64次循环, 每次3条向量指令                 │
└───────────────────────────────────────────────────────────────┘
```

#### 3.5.8 完整执行状态机

```
  ┌──── 1个 s2 tile 的 Vec2 执行 ────┐
  │                                  │
  │ ① WaitCrossCore                  │ 等AIC完成BMM2
  │ ② WaitFlag<MTE3_V>              │ 等上轮输出完成
  │ ③ for each M子块:               │ subMBase=64, 仅1轮
  │      SetFlag<V_MTE2>            │  通知DMA: 可以搬
  │      WaitFlag<V_MTE2>           │  等DMA确认
  │      DataCopy GM→UB (O_patch)   │  GM O_patch → bmm2Ub, 16KB
  │      SetFlag<MTE2_V>            │  通知Vector: 数据就绪
  │      WaitFlag<MTE2_V>           │  等DMA搬运完成
  │      if (isFirstS2Loop):        │
  │          DataCopy O_patch→累加器│  直接拷贝, 初始化为第一个O
  │      elif (!isLastS2Loop):      │
  │          FlashUpdateNew         │  在线累加: acc=acc×α+cur
  │      else:                      │
  │          FlashUpdateLastNew     │  最终累加: acc=(acc×α+cur)/sum
  │          if (isFirstS2Loop):    │
  │              LastDivNew         │  仅1个tile: cur/sum
  │          Cast float→half        │  8KB
  │          CopyAttentionOut UB→GM │  attentionOut GM
  │ ④ SetFlag<MTE3_V>              │ 通知下一轮
  └──────────────────────────────────┘
```

#### 3.5.9 三种场景的数学公式

| 场景 | 条件 | 寄存器操作 | 公式 |
|------|------|-----------|------|
| 首个s2 tile | isFirstS2Loop | DataCopy | `O_acc = O_patch` |
| 中间s2 tile | !isFirst && !isLast | FlashUpdateNew | `O_acc = O_acc × α + O_new` |
| 末个s2 tile | isLastS2Loop | FlashUpdateLastNew | `O_final = (O_acc × α + O_new) / sum` |
| 仅1个s2 tile | isFirst && isLast | LastDivNew | `O_final = O_patch / sum` |

其中 `α = exp(old_running_max - new_running_max)` 是从 Vec1 传下来的修正因子, 复用 Vec1 已经算好的值（存在 `softmaxExpBuf` 中），Vec2 不重复计算。

---

### 3.6 四阶段拆分的代码位置索引

| 阶段 | 操作 | 文件 | 行号 | 说明 |
|------|------|------|------|------|
| BMM1 | Q L1类型选择 | `fia_block_cube_noquant_gqa.h` | 42-44 | D≤256→DB, D>256→Single+驻留 |
| BMM1 | K L1类型选择 | `fia_block_cube_noquant_gqa.h` | 48-51 | 大tile→Single, 常规→DB |
| BMM1 | InitBuffers | `fia_block_cube_noquant_gqa.h` | 162-182 | 分配 L1 Q/K/V 槽位 |
| BMM1 | Q搬运决策 | `fia_block_cube_noquant_gqa.h` | 579-587 | isFirstS2Loop→Get搬, else→GetPre复用 |
| BMM1 | CopyQueryTile | `fia_block_cube_noquant_gqa.h` | 352-360 | Q GM→L1 |
| BMM1 | CopyKeyTile | `fia_block_cube_noquant_gqa.h` | 426-433 | K GM→L1 |
| BMM1 | MatMul | `fia_block_cube_noquant_gqa.h` | 617-619 | Q×K^T → L0C |
| BMM1 | Fixpipe → UB | `fia_block_cube_noquant_gqa.h` | 631 | L0C→UB + scale |
| Vec1 | ProcessVec1 | `fia_block_vec_noquant_gqa.h` | 217-233 | WaitCrossCore入口 |
| Vec1 | ProcessVec1Nd | `fia_block_vec_noquant_gqa.h` | 484-641 | No更新 / Update 分发 |
| Vec1 | M 维度拆分 (AIV子块) | `fia_kernel_noquant_gqa.h` | 762-771 | actVecMSize = 32行/子核 |
| Vec1 | NoUpdate VF | `vf_basic_block_aligned128_no_update.h` | 28-346 | local_max + ExpSub + Cast |
| Vec1 | Update VF | `vf_basic_block_aligned128_update.h` | 28-~350 | local vs running Max比较 |
| Vec1 | Max(local, running) | `vf_basic_block_aligned128_update.h` | 201-206 | 32行比较, 1条向量指令 |
| Vec1 | UpdateExpSumAndExpMax | `vf_mul_sel_softmaxflashv2_cast_nz.h` | 152-214 | sum在线修正 |
| Vec1 | P→L1 | `fia_block_vec_noquant_gqa.h` | 623-632 | DataCopy + SetCrossCore |
| BMM2 | P加载 (L1→L0A) | `fia_block_cube_noquant_gqa.h` | 909, 911 | WaitCrossCore |
| BMM2 | CopyValueTile | `fia_block_cube_noquant_gqa.h` | 435-438, 917 | V GM→L1 |
| BMM2 | MatMul P×V | `fia_block_cube_noquant_gqa.h` | 932-938 | P×V → L0C |
| Vec2 | ProcessVec2入口分发 | `fia_block_vec_noquant_gqa.h` | 1037-1051 | UB/GM/splitD三路径 |
| Vec2 | ProcessVec2_C2GmUpdateUb | `fia_block_vec_noquant_gqa.h` | 1053-1142 | GM→UB→累加→写回 (M切块循环) |
| Vec2 | M维切块subMBase | `fia_block_vec_noquant_gqa.h` | 1056-1057 | subMBase=8192/D, vec2LoopLimit |
| Vec2 | GM→UB双缓冲 | `fia_block_vec_noquant_gqa.h` | 1075-1105 | V_MTE2/MTE2_V事件, DataCopy+SetFlag |
| Vec2 | FlashUpdateNew (中间累加) | `vf_flashupdate_new.h` | 26-68 | O_acc = O_acc×α + O_new (BFV) |
| Vec2 | FlashUpdateLastNew (末块累加+除) | `vf_flashupdate_new.h` | 201-258 | O_final = (O_acc×α+O_new)/sum (LastBFV) |
| Vec2 | LastDivNew (仅1个tile) | `vf_flashupdate_new.h` | 399-451 | O_final = O_cur / sum (DivVF) |
| Vec2 | Cast + CopyOut | `fia_block_vec_noquant_gqa.h` | 685-706, 847-941 | float→half, DataCopy→GM |

### 3.7 与 A3 V 核 Row-Loop 策略对比

| 维度 | 本代码 | A3 原生策略 | 相同点 |
|------|--------|-----------|--------|
| M 维度拆分 | 64行→AIV0(32)+AIV1(32) | V核内按行循环 | 行分块+循环 |
| 单次处理 | 1行/迭代 (for loop) | 1行/迭代 | 相同 |
| 列拆分 | 128→even(64)+odd(64) | 按向量宽度(64)拆 | 硬件决定 |
| 寄存器并行 | 64 float/指令 | 同 | 同 |
| 跨s2在线修正 | 1条Max指令并行32行比较 | FlashAttention标准 | FIA特殊 |
| Q复用 | L1 DB + GetPre不复搬 | 通常同 | 同 |
| P流水 | L1 3buff支持AIC/AIV流水 | 通常同 | FIA特殊 |

**本质相同**: 行分块 + 向量寄存器并行 + 循环迭代。
**FIA 特殊**: (1) AIV双子核拆分 (2) Q跨s2 L1驻留 (3) P三缓冲支持AIC/AIV流水线 (4) 1条Max向量指令完成全部32行local vs running比较。
```
