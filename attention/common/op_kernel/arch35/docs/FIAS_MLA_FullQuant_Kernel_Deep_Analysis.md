# FlashAttentionScore Infer MLA FullQuant Kernel (arch35) 深度分析

> 作者：资深算子开发工程师
> 日期：2026-05-08
> 版本：v1.0

---

## 1. 概述

`flash_attention_score_kernel_infer_mla_fullquant.h` 是 FlashAttentionScore 算子在 Ascend NPU arch35 架构上的**推理 MLA（Multi-head Latent Attention）全量化 Kernel 模板**，采用 **CRTP 基类 + 4 阶段流水 + Cub/Vec 双引擎异构** 架构，在 arch35（Ascend 950DT/950PR）芯片上实现极致性能。

### 1.1 核心特征

| 特征 | 描述 |
|------|------|
| **架构代际** | arch35 (DAV_3510, Ascend 950DT/950PR) |
| **核类型** | AIC (Cube) + AIV (Vector) 异构协同 |
| **编程模型** | Regbase (MicroAPI 直接寄存器编程) |
| **量化模式** | FullQuant — INT8/FP8 全量化推理 |
| **MLA特征** | QK HeadDim = 512/576, RopeDim = 64, KV压缩表示 |
| **流水深度** | 4 阶段 (BMM1→Vec1→BMM2→Vec2)，runInfo[4] 环形缓冲 |
| **支持特性** | GQA、PSE、AttenMask、Rope、PageAttention、FlashDecode、KV Prefix、Antiquant、PostQuant、Attention Sink |
| **模板架构** | CRTP (ChildClass→BaseClass) + if constexpr 编译期路由 |

### 1.2 MLA (Multi-head Latent Attention) 简介

MLA 是 DeepSeek-V2/V3 引入的关键技术，核心是 **低秩键值联合压缩**：

```
传统 Attention：Q[g, s, D] × K[g, s, D]^T
MLA Attention： Q[g, s, D] × (K_compress[s, D_c] · W_UK[g, D_c, D])^T
                       + Q_rope[g, s, D_r] × K_rope[s, D_r]^T
```

其中：
- D=512 或 576（全量 head dim），D_r=64（rope head dim）
- D_c << D（压缩维度，典型 128）
- W_UK 是上投影矩阵（运行时反量化得到），由 fullquant 量化为低精度
- Q 也被分解为 nope（非rope）和 rope 两部分，拼接后形成有效 D

### 1.3 文件索引

| 文件 | 行数 | 职责 |
|------|------|------|
| `attention/common/op_kernel/arch35/flash_attention_score_kernel_infer_mla_fullquant.h` | 283 | 主 Kernel 类：流水编排、任务调度 |
| `attention/common/op_kernel/arch35/flash_attention_score_kernel_base_fullquant.h` | 711 | CRTP 基类：通用 Init/Process/Tiling 解析 |
| `attention/common/op_kernel/arch35/flash_attention_score_kernel_base.h` | 692 | 非量化基类（本 Kernel 通过继承链复用） |
| `attention/common/op_kernel/arch35/flash_attention_score_block_cube_mla_fullquant.h` | 1014 | Cube 子模块：MLA 全量化 BMM1/BMM2 |
| `attention/common/op_kernel/arch35/flash_attention_score_block_vec_infer_mla_fullquant.h` | 968 | Vec 推理子模块：Vec1/Vec2/Softmax/FD |
| `attention/common/op_kernel/arch35/flash_attention_score_block_vec_base_fullquant.h` | 1817 | Vec 基础子模块：Mask/Softmax/DataCopy |
| `attention/common/op_kernel/arch35/flash_attention_score_antiquant_processor.h` | ~800 | Antiquant 反量化处理器 |
| `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h` | ~500 | Vec 侧反量化 block |
| `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_cube.h` | ~400 | Cube 侧反量化 block |
| `attention/common/op_kernel/arch35/flash_attention_score_common_regbase.h` | 209 | Regbase 公共常量/枚举 |
| `attention/common/op_kernel/arch35/flash_attention_score_tiling_regbase.h` | 383 | Tiling 数据结构定义 |
| `attention/common/op_kernel/arch35/util_regbase.h` | 529 | Regbase 工具类型 (RunInfo/ConstInfo/模板宏) |
| `attention/common/op_kernel/arch35/infer_flash_attention_comm.h` | 167 | 推理公共定义 (模板宏展开体系) |
| `attention/common/op_kernel/arch35/infer_flash_attention_kvcache.h` | 897 | KV Cache 查询/Padding/Offset 计算 |
| `attention/common/op_kernel/arch35/attenmask.h` | ~600 | AttenMask 压缩/偏移/VF 合并 |
| `attention/common/op_kernel/arch35/vf/vf_flash_decode.h` | ~600 | FlashDecode 归约 VF 实现 |
| `attention/common/op_kernel/arch35/vf/vf_basic_block_utils.h` | 130 | Vector Function 公共工具/宏 |
| `attention/common/op_host/arch32/fia_tiling_nonquant_mla.cpp` | 602 | MLA 非量化 Tiling（arch32 注册参考） |

---

## 2. 系统级架构总览

### 2.1 CRTP 类继承层次

```
FlashAttentionScoreKernelBase<Child, Cube, Vec>       ← 最顶层基类（复用）
    │
FlashAttentionScoreKernelBaseFullquant<Child, Cube, Vec>    ← Fullquant 扩展
    │
FlashAttentionScoreKernelInferMlaFullquant<CubeBlockType, VecBlockType>  ← 推理 MLA 特化
```

CRTP 模式使得 `BaseClass` 可以调用 `GetDerived()` 获取 `ChildClass` 的成员，消除虚函数开销。

```cpp
// flash_attention_score_kernel_infer_mla_fullquant.h:26-31
template <typename CubeBlockType, typename VecBlockType>
class FlashAttentionScoreKernelInferMlaFullquant 
    : public FlashAttentionScoreKernelBaseFullquant<
        FlashAttentionScoreKernelInferMlaFullquant<CubeBlockType, VecBlockType>, 
        CubeBlockType, VecBlockType> {
    using BaseClass = FlashAttentionScoreKernelBaseFullquant<...>;
};
```

### 2.2 四级流水线架构

MLA FullQuant Kernel 实现 4 阶段软件流水，每阶段滞后 1 slot：

```
Timeline  ────────────────────────────────────────────────────────►
Slot  0: [BMM1(0)] → [Vec1(0)] → [BMM2(0)] → [Vec2(0)]
Slot  1:             [BMM1(1)] → [Vec1(1)] → [BMM2(1)] → [Vec2(1)]
Slot  2:                         [BMM1(2)] → [Vec1(2)] → [BMM2(2)]
Slot  3:                                     [BMM1(3)] → [Vec1(3)]

         runInfo[0]  runInfo[1]  runInfo[2]  runInfo[3]  ← 4 槽环形缓冲
```

每个 slot 对应一个 `RunInfo<isInfer>` 上下文，存储该迭代的全部参数。

```cpp
// flash_attention_score_kernel_infer_mla_fullquant.h:212-249
for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
    if (notLastThreeLoop) {
        RunInfo<isInfer> &runInfo1 = runInfo[taskId & 3];  // 当前槽位
        this->SetRunInfo(runInfo1, runParam, taskId, ...);
        if ASCEND_IS_AIC {
            this->cubeBlock.IterateBmm1(...);               // BMM1: Q × K^T
        }
    }
    if (taskId > 0 && notLastTwoLoop) {
        if ASCEND_IS_AIV {
            auto &runInfo3 = runInfo[(taskId + 3) & 3];    // Lag 1: Vec1
            this->vecBlock.ProcessVec1(...);
        }
    }
    if (taskId > 1 && notLast) {
        if ASCEND_IS_AIC {
            RunInfo<isInfer> &runInfo2 = runInfo[(taskId + 2) & 3];  // Lag 2: BMM2
            this->cubeBlock.IterateBmm2(...);
        }
    }
    if (taskId > 2) {
        if ASCEND_IS_AIV {
            RunInfo<isInfer> &runInfo3 = runInfo[(taskId + 1) & 3];  // Lag 3: Vec2
            this->vecBlock.ProcessVec2(...);
        }
    }
    ++taskId;
}
```

### 2.3 流水排空机制

末尾 Batch 的 GS1 循环额外执行 3 轮排空（`:171-196`）：

```cpp
int64_t tempGS1End = lastBN ? (runParam.s1LoopTimes + 3) : runParam.s1LoopTimes;
for (int64_t gS1Index = gS1StartIdx; gS1Index < tempGS1End; ++gS1Index) {
    // 通过 extraGS1 判断：-1 = isLastBmm1, 0 = notLastThreeLoop=false, 
    // 1 = notLastTwoLoop=false, 2 = notLast=false
    // S2 循环内通过 notLastThreeLoop/notLastTwoLoop/notLast 逐级停止
}
```

+3 排空对应 4 阶段流水（BMM1→Vec1→BMM2→Vec2），确保管道中的最后一个任务完全排空。

---

## 3. Tiling 切分策略

### 3.1 多级切块维度

MLA 场景涉及 S1(Q)、S2(KV)、G(Group) 三级切分：

| 切分维度 | 切分依据 | 基本块粒度 | 决定因素 |
|---------|---------|-----------|---------|
| **S2 内循环** | `s2TemplateType` | 128 | KV 序列长度 / PageAttention Block Size |
| **S1 外循环** | `s1TemplateType` | 64 或 128 | Q 序列长度 / gSize / D 维容量 |
| **G 轴** | `gSize` | 1~128 | Group 数切分 (MLA g<128 可全载，g=128 需要切分) |

### 3.2 S1 基本块与 GS1 合轴

MLA 场景中，Q 的有效 S 维 = gSize × s1Size（BSH/BSND layout），G 和 S1 在 AIC 视角被合并为 GS1 轴：

```cpp
// infer_flash_attention_kvcache.h (ComputeS1LoopInfo)
if (hasRope && dTemplateType == DTemplateType::Aligned576) {
    s1LoopTimes = CeilDiv(runParam.actualS1Size, s1BaseSize);  // MLA 特化
} else if (constInfo.isGqa) {
    s1LoopTimes = CeilDiv(runParam.actualS1Size * constInfo.gSize, s1BaseSize);  // GQA GS1 合轴
}
```

**MLA 与 GQA GS1 合轴的关键区别**：MLA 场景 gSize×s1Size 就是 Q 的 M 维总大小（无需再乘 gSize），因为 dTemplateType=Aligned576 表示 MLA。

### 3.3 S2 基本块与 Pre/Next Token 裁剪

S2 方向具备 preToken / nextToken 基于 Sparse Attention Mask 的裁剪：

```cpp
// infer_flash_attention_kvcache.h (ComputeS2LoopInfo)
if (hasRope && dTemplateType == DTemplateType::Aligned576) {
    // MLA: Q 的 g×s1 直接映射到 KV 的 s2
    sInnerFirstToken = ClipSInnerTokenCube(
        (cubeSOuterOffset - preTokensPerBatch) / constInfo.gSize, 0, actualS2Size);
    runParam.s2LineEndIdx = ClipSInnerTokenCube(
        CeilDiv(cubeSOuterOffset + nextTokensPerBatch + s1RealSize, constInfo.gSize), 
        0, actualS2Size);
}
```

### 3.4 分核算法（Host 侧）

分核使用 `SplitCore()` 引擎，定义在 `split_core_v2.h`：

```
输入：
  BaseInfo: b×n2×g×s1×s2 × layout+actualSeqLens
  SplitParam: mBaseSize, s2BaseSize

算法：
  for each batch:
    for each head (n2):
      for each GS1 block:
        计算该块的 S2 裁剪范围 (preToken/nextToken)
        按 s2BaseSize 切分成 S2 子任务
  
  贪心分配到每个 Core：按 CurrentCost + TaskCost 取最小值的 Core
  输出每个 Core 的 [bnStartIdx, bnEndIdx, gS1StartEndIdx, s2StartEndIdx]
```

### 3.5 TilingKey 与模板展开

Kernel 入口通过 TilingKey 在编译期确定所有模板参数：

```cpp
// infer_flash_attention_comm.h 中的模板宏体系
#define CHILD_SPEC_TEMPLATE_ANTI \
    template <typename Q_T, typename KV_T, typename T, typename OUTPUT_T, \
    ImplModeEnum implMode, LayOutTypeEnum layout, \
    S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, \
    DTemplateType dTemplateType, DTemplateType dVTemplateType, \
    PseTypeEnum pseMode, AntiquantTypeEnum antiquantMode, \
    bool hasAtten, bool hasDrop, bool hasRope, \
    bool isInfer, bool isPa, bool isFd, bool enableKVPrefix>
```

模板参数涵盖：数据类型 (Q_T/KV_T/OUTPUT_T)、布局 (BSH/BNSD/TND)、基本块 (S1_64/128、S2_128/256)、PSE 模式、量化模式、AttenMask、Rope、推理标记、PageAttention、FlashDecode、KV Prefix 等。

---

## 4. 主 Kernel 流程

### 4.1 初始化 (`InitUniqueConstInfo` + `InitUniqueRunInfo`)

Kernel 初始化分两个阶段执行：

#### 阶段 1: 常量初始化 (`InitUniqueConstInfo`)

```cpp
// flash_attention_score_kernel_infer_mla_fullquant.h:44-86
void InitUniqueConstInfo() {
    // FlashDecode: 计算 splitKVNum 和 sInnerLoopSize
    if constexpr (isFd) {
        this->constInfo.splitKVNum = this->sharedParams.splitKVNum;
        this->constInfo.sInnerLoopSize = CeilDiv(s2Size, splitKVNum);
    }
    // PostQuant: 后量化参数
    if constexpr (POST_QUANT) {
        this->constInfo.isPostQuantPerChnl = sharedParams.isPostQuantPerChnl;
        this->constInfo.isPostQuantBF16 = sharedParams.isPostQuantBF16;
    }
    // MLA 关键维度推导
    this->constInfo.n2GD = n2G * dSize;          // N2×G×D
    this->constInfo.gDR = gSize * dSizeRope;      // G×D_Rope
    this->constInfo.n2GDR = n2Size * gDR;         // N2×G×DR
    this->constInfo.mm1RopeKa = dSizeRope;        // BMM1 的 Rope K 维 = DR
    this->constInfo.mm1Ka = dSize;                // BMM1 的非 Rope K 维 = D
}
```

#### 阶段 2: 运行时初始化 (`InitUniqueRunInfo`)

```cpp
void InitUniqueRunInfo(const RunParamStr<isInfer> &runParam, RunInfo<isInfer> &runInfo) {
    InitTaskParamByRun(...);  // 初始化任务参数
    ComputeOffset(runParam, constInfo,
        runInfo.s2LoopCount + runInfo.s2StartIdx / constInfo.s2BaseSize, runInfo);
}
```

### 4.2 主循环流程 (`ProcessMainLoop`)

```cpp
// flash_attention_score_kernel_infer_mla_fullquant.h:98-255
void ProcessMainLoop() {
    // 1. 确定当前 Core 处理的 Batch/Head 范围
    bnStartIdx = sharedParams.bnStartIdx;
    gS1StartIdx = sharedParams.multiCoreInnerOffset;
    bnEndIdx = lastCore ? (bSize * n2Size * headNumRatio) : sharedParams.bnEndIdx;

    RunInfo<isInfer> runInfo[4];  // 4 槽环形缓冲
    
    for (bnIdx = bnStartIdx; bnIdx < bnEndIdx; ++bnIdx) {
        // 2. 计算当前 Batch 的参数
        ComputeParamBatch(runParam, constInfo, attenMaskInfo, ...);
        ComputeS1LoopInfo(runParam, constInfo, lastBN, nextGs1Idx);

        // 3. GS1 循环 (含 +3 排空)
        int64_t tempGS1End = lastBN ? (s1LoopTimes + 3) : s1LoopTimes;
        for (gS1Index = gS1StartIdx; gS1Index < tempGS1End; ++gS1Index) {
            // 4. S2 循环 — 4 阶段流水
            for (s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
                // Stage 0: BMM1 (AIC)
                // Stage 1: Vec1 (AIV, Lag 1)
                // Stage 2: BMM2 (AIC, Lag 2)
                // Stage 3: Vec2 (AIV, Lag 3)
            }
        }
    }
}
```

### 4.3 任务创建 (`ComputeAxisIdxByBnAndGs1`)

将 BN2 复合索引反解为 (bIdx, n2Idx, gIdx, s1Idx)：

```cpp
void ComputeAxisIdxByBnAndGs1(int64_t bnIndex, int64_t gS1Index, RunParamStr &runParam) {
    runParam.goIdx = gS1Index / constInfo.s1OuterSize;  // G 索引
    runParam.s1oIdx = gS1Index % constInfo.s1OuterSize; // S1 索引
    // boIdx 和 n2oIdx 由 bnIndex 反解：boIdx = bnIdx / (n2Size × headNumRatio)
}
```

### 4.4 Vec 侧 M 轴拆分

每个 AIC Core 下有 2 个 AIV Core（CV_RATIO=2），通过 `subBlockIdx` 将 M 轴对半拆分：

```cpp
// flash_attention_score_block_vec_base_fullquant.h (ProcessVec1Nd)
runParam.halfS1RealSize = (s1RealSize + 1) >> 1;  // 一半
if (constInfo.subBlockIdx == 1) {
    runParam.halfS1RealSize = s1RealSize - halfS1RealSize;  // V1 处理另一半
    runParam.sOuterOffset += firstHalfS1RealSize;
}
```

---

## 5. Cube 子模块 (FABlockCubeMlaFullquant)

### 5.1 L1 空间分配

MLA 场景下，Q 矩阵 D=512（含 Rope=64），需要约 mBaseSize × 512 × 2B ≈ 128KB L1 存放 Q 全量。K/V 采用分块搬运：

```cpp
// flash_attention_score_block_cube_mla_fullquant.h
// L1 分配策略：
// Q_L1:   mBaseSize × dSize × sizeof(Q_T)  ≈ 128KB (single buffer, S2 复用)
// K_L1:   s2BaseSize × dSize × sizeof(KV_T) ≈ 64KB  (double buffer, BMM1用)
// V_L1:   s2BaseSize × dSizeV × sizeof(KV_T) ≈ 64KB (double buffer, BMM2用)
// Total L1 ≤ 256KB (MLA 模式保留余地给 Rope 拼接)
```

### 5.2 MLA BMM1 = Q × K^T

MLA 的 BMM1 有两条路径：FP8 量化和 INT8 量化。

#### INT8 路径 (`IterateBmm1MLAFullQuantInt8`)

```cpp
// flash_attention_score_block_cube_mla_fullquant.h
// BMM1 = Q [mSize, D+R] × K^T [s2Size, D+R]
// 其中 Q = [Q_nope(D) | Q_rope(R)], K = [K_compressed(D) | K_rope(R)]
// D=512, R=dSizeRope=64, 实际 K=576

void IterateBmm1MLAFullQuantInt8(Buffer<...> &outputBuf, RunInfo &runInfo, ConstInfo &constInfo) {
    // 1. Q 搬运到 L1（仅首次 s2Loop）
    if (unlikely(runInfo.isFirstS2Loop)) {
        // Q_nope 部分：从 queryGm 搬运
        CopyQueryNopeToL1(l1QBuffer, runInfo);
        // Q_rope 部分：从 queryRopeGm 搬运，追加到 Q_nope 后
        CopyQueryRopeToL1(l1QBuffer, runInfo);
    }

    // 2. K 搬运到 L1（每次 s2Loop 都需要）
    // K_nope 部分：从 keyGm（含 pageTable 转换）搬运
    CopyKeyNopeToL1(l1KBuffer, runInfo);
    // K_rope 部分：从 keyRopeGm 搬运
    CopyKeyRopeToL1(l1KBuffer, runInfo);

    // 3. Mmad INT8: 硬件 INT8 乘 → INT32 累加
    //    结果 Fixpipe → FP16 (在 L1 的 BMM1 输出区)
    MatmulK<KV_T, Q_T, ..., 128, 128, baseK, ...>(...);
    FixpipeMm1(output, l0c, runInfo);
}
```

**MLA Rope 拼接的 L1 布局**：

```
Q 在 L1: ┌─────────────────────┬──────────────┐
         │  Q_nope (D=512 col)  │ Q_rope (64)  │
         │  from queryGm        │ from ropeGm  │
         └─────────────────────┴──────────────┘
         ← Align(D, 16) = 512 →│← Align(R, 16) = 64 →│
         Effective MM K = 576

K 在 L1: ┌─────────────────────┬──────────────┐
         │  K_nope (D=512 col)  │ K_rope (64)  │
         │  from keyGm          │ from ropeGm  │
         └─────────────────────┴──────────────┘
```

### 5.3 MLA BMM2 = P × V

MLA 的 BMM2 使用 `MatmulN`（不同于 BMM1 的 `MatmulK`），P 矩阵经过 Vec1 处理后在 L1 就绪：

```cpp
void IterateBmm2MLAFullQuant(Buffer<...> &outputBuf, ..., RunInfo &runInfo, ConstInfo &constInfo) {
    // 1. P 矩阵已在 L1 就绪（由 Vec1 写入）
    Buffer mm2A = l1PBuffers.Get();
    mm2A.WaitCrossCore();       // 等待 AIV Vec1 写入完成

    // 2. V 搬运到 L1
    Buffer mm2B = l1VBuffers.Get();
    CopyValueSlice(mm2BTensor, runInfo);

    // 3. MatMul: P [mSize, s2Size] × V [s2Size, dV]
    MatmulN<KV_T, ..., 128, 128, s2BaseSize, ...>(mm2A, mm2B, ...);

    // 4. Fixpipe L0C→输出 (UB 或 GM)
    FixpipeMm2(output, l0c, runInfo);
}
```

### 5.4 Anti-quant 模式的 Cube 侧处理

全量化模式下，Q 和 K/V 已在 Host 侧预先量化为 INT8。Cube 侧的 Mmad 直接处理 INT8 输入。反量化在 IFA_PRE 模式中处理。

对于 `IterateBmm1MLAFullQuantInt8`，Q 和 K 均已被反量化处理器预先处理，Mmad 直接使用 INT8 输入：
- Q：已由 `AntiquantProcessor` 从 FP16→INT8 量化
- K：已由 `AntiquantProcessor` 从 compressed→INT8 反量化

### 5.5 PageAttention 适配

MLA PageAttention 场景下，KV Cache 通过 `blockTableGm` 间接寻址：

```cpp
// 每次 s2Loop 的 K 搬运时
if constexpr (isPa) {
    uint64_t blockIdOffset = curSeqIdx / kvCacheBlockSize;
    uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);
    uint64_t keyOffset = idInBlockTable * kvCacheBlockSize * headDim * kvHeadNum;
    // + n2oIdx 偏移 + block 内 remainRowCnt 偏移
}
```

---

## 6. Vec 子模块 (FABlockVecInferMlaFullquant + Base)

### 6.1 UB 空间分配

MLA FullQuant 的 UB 分配包括以下关键区域：

```
UB (总计 ≈ 192KB):
├── BMM1 结果 Queue (stage1OutQue[0..1])
│   └── mBaseSize/2 × s2BaseSize × sizeof(float)  ≈ 64KB × 2
├── AttnMask Buffer (可选)
│   └── s2BaseSize/8 × mBaseSize/2  ≈ 8KB × 2
├── PostQuant Buffer (可选)
│   └── mBaseSize/2 × dV × sizeof(int8_t)  ≈ 32KB
├── BMM2 输入暂存 (BMM2 在 GM 时)
│   └── mBaseSize/2 × dV × sizeof(float)  ≈ 32KB
├── Vec2 累加结果暂存
│   └── mBaseSize/2 × dV × sizeof(float)  ≈ 32KB
├── Softmax state (max/sum/exp)
│   └── mBaseSize × sizeof(float) × 3  ≈ 1.5KB
└── LSE/Sink/FD 专用暂存
    └── mBaseSize × 8 × sizeof(float)  ≈ 2KB
```

### 6.2 Vec1 阶段 (Softmax + AttnMask + PSE)

MLA FullQuant 的 Vec1 有三条计算路径：

#### 路径 1: Nz 格式 (`ProcessVec1Nz`)

HIFLOAT8 格式的 BMM1 结果，在 Softmax 前需反量化：

```cpp
void ProcessVec1Nz(Buffer<..., L1> &outputBuf, Buffer<..., UB> &bmm1ResBuf, ...) {
    if constexpr (hasAtten) {
        // 拷贝 AttnMask → UB
        AttenmaskCopyIn(attenMaskUb, ...);
    }
    if constexpr (hasPse) {
        // 拷贝 PSE → UB
    }
    // HIFLOAT8 → FP32 反量化 + Scale + Softmax + FP32→FP16
    ProcessVec1NzVf(stage1Cast, vselrIndexesBuf, softmaxMaxUb, softmaxSumUb,
                    mmRes, attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer,
                    actVecMSize, actSingleLoopS2Size, scaleValue, ...);
    // 写出到 L1 (供 BMM2 使用)
    DataCopy(mm2AL1Tensor, stage1Cast, ...);
    outputBuf.SetCrossCore();
}
```

#### 路径 2: Dn 格式 (`ProcessVec1Dn`)

Dn 格式（转置）的 BMM1 结果，FP8 scale 在反量化前应用：

```cpp
void ProcessVec1Dn(...) {
    // 1. 拷贝 FP8 Scale → UB
    CopyFp8Scale(scaleUb, scaleGm);
    // 2. Scale × BMM1 + Softmax + Cast
    ProcessVec1DnVf(...);
    // 3. 写出到 L1
}
```

#### 路径 3: Nd 格式 (`ProcessVec1Nd`) — 默认路径

标准 FP16 格式的 BMM1 结果，包含最完整的 AttnMask/PSE/Dropout 处理：

```cpp
void ProcessVec1Nd(Buffer<..., L1> &outputBuf, Buffer<..., UB> &bmm1ResBuf, 
                   const RunInfo &runInfo, const ConstInfo &constInfo) {
    // 1. 等待 AIC BMM1 结果就绪
    bmm1ResBuf.WaitCrossCore();

    // 2. 获取 PSE 数据
    if constexpr (hasPse) { CopyPse(pseUb, runInfo, constInfo); }

    // 3. 拷贝 AttnMask → UB
    if constexpr (hasAtten) { MlaAttenMaskCopyIn(attenMaskUb, ..., constInfo); }

    // 4. 拷贝 上一轮 DropMask (hasDrop)
    if constexpr (hasDrop) { CopyDropMask(dropMaskUb, ...); }

    // 5. 主计算: Scale + Mask + PSE + SoftmaxFlashV2
    //    SoftmaxInvalidLineCheck: 检测全无效行
    //    MlaScaleValueProcess: MLA 专用 Scale Handler
    MulsAndSelectAndSoftmaxFlashV2AndCast<VEC_T, ..., actVecMSize, s2BaseSize>(...);

    // 6. 写出 P 矩阵到 L1 → AIC BMM2 可用
    DataCopy(mm2AL1Tensor, stage1Cast, ...);
    outputBuf.SetCrossCore();

    // 7. 更新 Running Softmax Max/Sum
    if (!runInfo.isFirstS2Loop) {
        UpdateExpSumAndExpMax(sumUb, maxUb, expUb, ...);
    }

    // 8. 末轮写出 LSE
    if (runInfo.isLastS2Loop) {
        SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
    }
}
```

#### MLA 专用 AttenMask 拷贝 (`MlaAttenMaskCopyIn`)

MLA 场景下 AttenMask 分 "GS1 合轴副本" 和 "按需紧凑拷贝"：

```cpp
// flash_attention_score_block_vec_base_fullquant.h
void MlaAttenMaskCopyIn(...) {
    if (constInfo.isPfaGS1Merge) {
        // GS1 合轴: gSize 行都相同, 只拷贝 1 行然后 DataCopy 复用
        BoolCopyInRegbase(maskUb, maskGm, maskOffset, 1, s2Size, ...);
        for (uint32_t i = 1; i < actVecMSize; i++) {
            DataCopy(maskUb[i * s2BaseSize], maskUb, s2BaseSize);
        }
    } else {
        MlaBoolCopyInRegbase(maskUb, maskGm, maskOffset, actVecMSize, s2Size, ...);
    }
}
```

### 6.3 Vec2 阶段 (FlashUpdate + 输出)

BMM2 结果有三种输出路径：

| 路径 | BMM2 输出位置 | 累加位置 | 使用场景 |
|------|-------------|---------|---------|
| `ProcessVec2OnUb` | UB | UB | D ≤ 128, BMM2 结果小 |
| `ProcessVec2DSplit` | GM | GM | D > 256, splitD 场景 |
| `ProcessVec2` | GM | UB | 默认: BMM2 在 GM, 累加在 UB |

#### 默认路径 (`ProcessVec2`)

```cpp
void ProcessVec2(Buffer<..., GM> &bmm2ResGmBuffers, RunInfo &runInfo, ...) {
    // 循环 M 轴小块处理
    for (mOIdx = 0; mOIdx < vec2LoopLimit; mOIdx++) {
        // 1. DataCopy GM→UB: BMM2 结果
        DataCopy(bmm2Ub, bmm2ResGm[offset], vec2CalcSize);

        if (runInfo.isFirstS2Loop) {
            // 首轮: 直接赋值
            DataCopy(vec2ResUb, bmm2Ub, vec2CalcSize);
        } else {
            // 非首轮: FlashUpdate
            FlashUpdateNew(vec2ResUb, bmm2Ub, vec2ResUb, 
                          expUb, subMRealsize, dBasicBlock);
        }

        if (runInfo.isFirstS2Loop && runInfo.isLastS2Loop) {
            // 仅一轮 S2 (No Multi-pass): 除 sum
            LastDivNew(vec2ResUb, sumUb, subMRealsize, dBasicBlock);
        }
    }

    if (runInfo.isLastS2Loop) {
        // 最终归一化 + 后量化 + 输出到 GM
        OutputAttentionOut(runInfo, vec2ResUb, actVecMSize);
    }
}
```

### 6.4 输出路径 (CopyOutAttentionOut)

MLA 场景下输出要经过：
1. Cast FP32 → OUT_T (FP16/BF16/INT8)
2. RowInvalid（将无效 M 行置零）
3. DataCopyExt（支持 BNSD→BSND 或 BSH→NBSD 等转排布局）
4. PostQuant（可选的后量化）

```cpp
void CopyOutAttentionOut(RunInfo &runInfo, LocalTensor<T> &attenOutUb, 
                         uint32_t actVecMSize, const ConstInfo &constInfo) {
    // 1. Format 输出：对齐 + 填充
    if (constInfo.transposeLayout == BNSD_BSND) {
        MlaTransposeDataCopyOut(attenOutUb, attenOutGm, runInfo);
    } else if (constInfo.transposeLayout == BSND_NBSD || ...) {
        MlaTranspose2DataCopyOut(attenOutUb, attenOutGm, runInfo);
    } else {
        Bmm2DataCopyOut(attenOutUb, attenOutGm, runInfo);
    }

    // 2. PostQuant (INT8 输出)
    if constexpr (POST_QUANT) {
        PostQuant(attenOutUb, attenOutGm, runInfo);
    }
}
```

### 6.5 Attention Sink 支持

MLA 推理引入 Sink Token（可学习注意力池化），Vec1 阶段特殊处理：

```cpp
void Vec1SinkCompute(runInfo, ...) {
    // 1. 从 sinkGm 拷贝 Sink Token 到 UB
    CopySinkIn(sinkUb, sinkGm, runInfo);
    // 2. 将 Sink 作为额外的 S2 行参与 Softmax
    SoftmaxWithSink(mmResUB, sinkUb, ...);
}
```

### 6.6 FlashDecode 归约

FlashDecode 模式下，单个 (B,N2) 对的 KV 被跨核切分（splitKVNum 份），多核各自计算 BMM1→Vec1→BMM2→Vec2 后，需要 AIV 统一归约：

```cpp
void FlashDecodeCompute(ConstInfo &constInfo, ...) {
    SyncAll();

    for (uint32_t splitKVIndex = 0; splitKVIndex < constInfo.splitKVNum; splitKVIndex++) {
        // 1. 从 workspace 拷贝各核的 accumOut + LSE (Max+Sum)
        CopyAccumOutIn(accumOutLocal, lseLocal, splitKVIndex);

        // 2. Combine LSE: ComputeScaleValue (Max→Exp→Sum→Div)
        ComputeScaleValue(dealRowCount, lseLocal, splitKVIndex);

        // 3. Reduce: dst = dst × oldLse + accumOut × newLse
        ReduceFinalRes(dstLocal, lseLocal, accumOutLocal, dealRowCount, splitKVIndex);
    }

    // 4. 后处理: Cast + PostQuant + DataCopy 最终输出
    CopyFinalResOut(dstLocal, attenOutGm, dealRowCount);
}
```

---

## 7. Antiquant 全量化处理器

### 7.1 量化模式

MLA FullQuant 支持三种量化粒度：

| 模式 | 描述 | 反量化参数 |
|------|------|-----------|
| **PerChannel** | 每 head 每 channel 独立 scale | `keyAntiQuantScale[n2][d]` |
| **PerToken** | 每 token 独立 scale | `keyAntiQuantScale[n2][seq]` |
| **PerChannel+PerToken** | 两层 scale 级联 | outer × inner |

### 7.2 Cube 侧反量化 (FABlockCubeMlaFullquant)

Cube 侧在 BMM1 之前将 Q 和 K 从 INT8 反量化为 FP16：

```cpp
// BMM1 前：将 Q 从 INT8 反量化到 L0（或 L1 的 FP16 区）
// Q' = Q_int8 × scale_Q  (逐元素，使用 Fixpipe 的 DEQF16)
// BMM1 后：Fixpipe 将 INT32 结果转换为 FP16
// S = S_int32 × scale_S (=1/1024)
```

### 7.3 Vec 侧反量化与量化

Vec1 阶段处理浮点 Softmax 后，将 P 矩阵重新量化为低精度：

```cpp
// P_fp32 → 归一化 → P_fp16 (Muls + Div) → P_int8
// 使用 RowMuls 逐行应用 scale
#define USE_MLA_FULLQUANT_V1_P(vreg_exp, vreg_rowmax_p, MaskReg) \
    do {                                                         \
        Muls(vreg_exp, vreg_exp, fp8e4m3MaxValue, MaskReg);      \
        Div(vreg_exp, vreg_exp, vreg_rowmax_p, MaskReg);         \
    } while (0)
```

反量化 Scale 缓存策略（在 UB 上预留 antiqScaleUb）：

```cpp
// Vec1 处理时将 scale 存入 UB 局部
CopyAntiquantScale(antiqScaleUb, keyAntiQuantScaleGm, kvAntiQuantOffset);
// Vec2 处理时再加 V 的 scale
Muls(bmm2ResUb, bmm2ResUb, scaleC2, vec2ComputeSize);
VecMulMat(bmm2ResUb, antiqScaleUb, bmm2ResUb, dealRowCount, columnCount, actualColumnCount);
```

---

## 8. 同步控制与流水

### 8.1 Cube 内部流水事件链

BMM1 的内部事件链（`IterateBmm1MLAFullQuantInt8`）：

```
GM→L1 Q_rope (MTE2) ─┐
GM→L1 Q_nope (MTE2) ─┤
                      ├──> Wait MTE2_MTE1 ──> Q 就绪
GM→L1 K_rope (MTE2) ─┐
GM→L1 K_nope (MTE2) ─┤
                      ├──> Wait MTE2_MTE1 ──> K 就绪
                      │
                      ↓
L1→L0 LoadData (MTE1→M)
                      ↓
Mmad (M→FIX)
                      ↓
Fixpipe (FIX→UB/GM/L1)
```

BMM2 的内部事件链类似，但 P 矩阵由 AIV Vec1 通过 DataCopy 写入 L1。

### 8.2 AIC↔AIV 跨核同步

```cpp
// AIC BMM1 完成 → 通知 AIV Vec1
mm1ResGmBuffers.SetCrossCore();  // AIC 侧
mm1ResGmBuffers.WaitCrossCore(); // AIV 侧

// AIV Vec1 完成 → 通知 AIC BMM2 (P 写入 L1)
l1PBuffers.SetCrossCore();      // AIV 侧
l1PBuffers.Get();                // AIC 侧 (触发 WaitCrossCore)

// AIC BMM2 完成 → 通知 AIV Vec2
mm2ResGmBuffers.SetCrossCore();  // AIC 侧
mm2ResGmBuffers.WaitCrossCore(); // AIV 侧
```

### 8.3 多 Batch 间的 Q 驻留策略

每个 BN2 切换时，Q 需要重新搬运；但同一 BN2 内的 S2 循环中，Q 驻留 L1：

```cpp
// flash_attention_score_kernel_infer_mla_fullquant.h (ProcessMainLoop)
for (bnIdx = ...) {
    ComputeParamBatch(...);  // ← 计算新 Batch 的 Q 地址
    
    for (gS1Index = ...) {
        for (s2LoopCount = ...) {
            // Q_nope + Q_rope 搬运：仅 s2LoopCount==0 时执行
            // (通过 runInfo.isFirstS2Loop 控制)
            cubeBlock.IterateBmm1(...);
        }
    }
}
```

---

## 9. 关键数据结构

### 9.1 RunParamStr（运行时参数）

```cpp
// util_regbase.h
struct RunParamStr {
    int64_t boIdx, n2oIdx, goIdx, s1oIdx;  // 多级索引
    int64_t s1RealSize;                      // 实际 S1 大小
    int64_t s2LineEndIdx;                    // 有效 S2 结束位置
    int64_t s2LineStartIdx;                  // 有效 S2 开始位置
    int64_t queryLeftPaddingSize;           // Q 左 padding
    int64_t kvLeftPaddingSize;              // KV 左 padding
    uint64_t qBOffset, qRopeBOffset;        // Q 基址偏移
    uint64_t keyCoreOffset, valueCoreOffset; // KV 基址偏移
    uint64_t attentionOutOffset;            // 输出基址偏移
    int64_t actualS1Size, actualS2Size;      // 实际序列长度
    int64_t s1LoopTimes, s2LoopEndIdx;       // 循环次数
    int64_t halfS1RealSize;                 // 当前 AIV 的 M 大小
    int64_t sOuterOffset, cubeSOuterOffset;  // 偏移量
    // ...
};
```

### 9.2 RunInfo（每迭代上下文）

```cpp
// util_regbase.h
struct RunInfo {
    uint32_t loop, s2LoopCount, s2StartIdx, s2LoopEndIdx;  // 循环状态
    bool isFirstS2Loop, isLastS2Loop;                       // 首/末轮标记
    int64_t s1RealSize, s1RealSizeAlign32;                  // 有效 S1
    int64_t s1RealSizeV;                                    // VEC 视角 S1
    int64_t actualS1Size, actualS2Size;                     // 实际序列长度
    uint64_t vecCoreOffset;                                 // VEC Core 内部偏移
    // ... (20+ 字段)
};
```

### 9.3 ConstInfo（编译期 + 运行时常量）

```cpp
// util_regbase.h
struct ConstInfo {
    uint32_t bSize, n2Size, gSize, s1Size, s2Size;  // 轴长
    uint32_t dSize, dSizeV, dSizeRope;              // 维度
    uint32_t splitKVNum;                            // FlashDecode 分份数
    uint32_t sInnerLoopSize;                        // FD 的单核 S2 大小
    bool isRowInvalid, isGqa, isPfaGS1Merge;
    bool isKvContinuous, isQHasLeftPadding, isKVHasLeftPadding;
    uint8_t subBlockIdx;                            // AIC 下第几号 AIV
    uint32_t transposeLayout;                       // 输出转排模式
    uint64_t n2GD, gDR, n2GDR;                     // MLA 专用偏移
    uint32_t mm1RopeKa, mm1Ka;                     // BMM1 的 K 轴维度
    uint32_t headNumRatio;                          // Head 号比例
    // ... (30+ 字段)
};
```

---

## 10. AttenMask 压缩与计算模式

### 10.1 压缩模式枚举

```cpp
enum class AttenMaskCompressMode {
    NO_COMPRESS_MODE = 0,            // 无压缩
    LEFT_UP_CAUSAL_MODE = 1,        // 左下三角 (Causal)
    RIGHT_DOWN_CAUSAL_MODE = 2,     // 右上三角
    BAND_MODE = 3,                   // 带状 (pre+next tokens)
    PREFIX_MODE = 4,                 // Prefix 组合
    RIGHT_DOWN_CAUSAL_BAND_MODE = 5,// Causal + Band
    BAND_LEFT_UP_CAUSAL_MODE = 6,   // Band + Causal
};
```

### 10.2 压缩模式 → 计算模式转换

```cpp
// attenmask.h: GetAttenMaskComputeMode
// 输入: 压缩模式 + S1/S2 偏移
// 输出: 
//   CAUSAL_OR_NEXT_ONLY_MODE:  仅处理三角/next部分, pre部分跳过
//   PRE_ONLY_MODE:             仅处理 pre 部分
//   PRE_AND_NEXT_MODE:         pre+next 都处理
//   NO_NEED_COMPUTE_MODE:      mask 全为0, 跳过计算
//   PREFIX_COMPUTE_MODE:       处理 prefix+三角交错区域
//   PREFIX_N_COMPUTE_MODE:     仅处理 prefix 矩形部分
```

这种**两级模式转换**在编译期消除分支：外层 `GetAttenMaskComputeMode` 判断后，内层调用对应的专用 Mask Copy 函数。

### 10.3 GS1 合轴下的 Mask 复用

MLA 场景下，当 `isPfaGS1Merge=true` 时，gSize 行共享同一 Mask。优化的 `BoolCopyInRegbase` 函数只拷贝第 1 行，后续行通过 `DataCopy` 原地复制：

```cpp
// attenmask.h: BoolCopyInRegbase
if constexpr (isInfer && isGqa) {
    dataCopyParams.blockCount = 1;
    for (uint32_t i = 0; i < s1Size; ++i) {
        DataCopy(dstTensor[i * s2BaseSize], srcTensor[srcOffset], dataCopyParams);
    }
}
```

### 10.4 Regbase Mask 合并（Band + Prefix）

Arch35 使用 `__simd_vf__` Vector Function 直接在寄存器上合并两种 Mask：

```cpp
// attenmask.h: MergeBandVF
__simd_vf__ void MergeBandVF(uint64_t maskPreUb, uint64_t maskNextUb, uint16_t loopCount) {
    for (uint16_t i = 0; i < loopCount; ++i) {
        LoadAlign(vreg_pre, (__ubuf__ uint32_t*&)maskPreUb + i * 64);
        LoadAlign(vreg_next, (__ubuf__ uint32_t*&)maskNextUb + i * 64);
        Xor(vreg_not, vreg_pre, vreg_xor, preg_all);  // NOT pre
        Or(vreg_or, vreg_not, vreg_next, preg_all);    // NOT(pre) OR next
        StoreAlign((__ubuf__ uint32_t*&)maskNextUb + i * 64, vreg_or, preg_all);
    }
}
```

一步完成 NOT+OR 合并，延迟 ≈ 3 cycle/32B (vs DataCopy+Muls+Add 的 ~15 cycle/32B)。

---

## 11. 优化技术深度分析

### 11.1 CRTP 编译期多态

**原理**：基类 `FlashAttentionScoreKernelBaseFullquant` 接受 `ChildClass` 模板参数，通过 `GetDerived()` 调用子类方法。所有虚函数开销在编译期消除。

**与硬件关系**：Arch35 的指令 Issue 宽度有限，CRTP 零开销多态比虚函数表 + 间接跳转节省 ≈3-5 cycles/call。

**代码实现**：
```cpp
// flash_attention_score_kernel_base_fullquant.h
template <typename ChildClass, ...>
class FlashAttentionScoreKernelBaseFullquant {
    ChildClass& GetDerived() { return static_cast<ChildClass&>(*this); }
    void Process() { GetDerived().Process(); }  // 编译期 resolved
};
```

### 11.2 Arch35 Regbase 直接寄存器编程

**原理**：Arch35（Ascend 950DT/950PR）新增 Regbase 编程模式，允许直接操作 Vector 寄存器（vreg），绕过传统的 AscendC Pipe/Queue API。

**与硬件关系**：Arch35 Vector 引擎有 32 个 256-bit 向量寄存器。Regbase 模式比 Pipe API 延迟低约 30-40%。

**代码实现**：
```cpp
// vf_flash_decode.h
__simd_vf__ void ReduceFinalRes_0_VF(__ubuf__ T* dstUb, __ubuf__ T* lseUb, 
                                      __ubuf__ T* accumOutUb, ...) {
    MicroAPI::RegTensor<T> vregDst;       // 直接映射到物理寄存器
    MicroAPI::RegTensor<T> vregLse;
    MicroAPI::LoadAlign<T>(vregLse, lseUb + offset);  // UB→寄存器
    MicroAPI::Mul<T>(vregAccumOut, vregLse, vregAccumOut, pregTailN);
    MicroAPI::Add<T>(vregDst, vregDst, vregAccumOut, pregTailN);
    MicroAPI::StoreAlign<T>(dstUb, vregDst, pregTailN);  // 寄存器→UB
}
```

### 11.3 4 阶段流水 + 排空 + Q 驻留

**原理**：
- 4 阶段（BMM1→Vec1→BMM2→Vec2）各滞后 1 slot
- `runInfo[4]` 环形缓冲，index = taskId & 3
- 末尾 Batch GS1 循环多执行 3 轮排空
- Q 矩阵在 L1 驻留，多个 S2 循环复用

**效果**：将 Cube 的 MatMul 延迟与 Vector 的 Softmax/反量化延迟完全重叠。Q 驻留节省 (sInnerLoopTimes - 1) × mBaseSize × dSize 的 GM→L1 带宽。

### 11.4 多路径 Vec1/Vec2 编译期分派

**原理**：Vec1 有 Nz/Dn/Nd 三条路径，Vec2 有 OnUb/DSplit/GMUpdate 三条路径，通过 `if constexpr` 在编译期选择。

```cpp
if constexpr (useNz) {
    ProcessVec1Nz(...);
} else if constexpr (useDn) {
    ProcessVec1Dn(...);  // FP8 scale 特殊处理
} else {
    ProcessVec1Nd(...);  // 标准 FP16 + Mask + PSE
}
```

**效果**：不同量化格式的代码在编译期完全隔离，避免运行时的 if-else 分支预测失败。

### 11.5 MLA Rope 拼接搬运

**原理**：MLA 的 Q 矩阵由 nope (512 cols) + rope (64 cols) 拼接而成。Copy 时先搬运 nope 部分（从 queryGm），再搬运 rope 部分（从 queryRopeGm），追加到 L1 的同一行末尾。

**效果**：避免 Host 侧的显式拼接 kernel，一次 MTE2 搬运完成 Q 的构建。

### 11.6 AttenMask 压缩模式快速判定

**原理**：压缩模式（Causal/Band/Prefix）下，按 S1/S2 偏移计算 delta，判断当前块是否需要 Mask，避免 GM→UB 的无效搬运。

**效果**：在 Causal+Prefill 场景，约 50% 的 S2 块不需要 Mask 处理。

### 11.7 FlashDecode 分粒度归约

**原理**：FlashDecode 模式下多个 AIV Core 各自计算部分 Attention 结果，最后通过 `ComputeScaleValue`（Max → Exp → Sum → Div）和 `ReduceFinalRes`（加权累加）归约。

**亮点**：
- `ComputeScaleValue_8`/`ComputeScaleValue_16` 两个特化模板，按 dealRowCount≤8 或 ≤16 分派，充分利用 Arch35 的 256-bit 寄存器宽度
- `ReduceFinalRes_0` 和 `ReduceFinalRes_Rest` 分离：首轮直接赋值，后续轮 Load→Mul→Add→Store

### 11.8 模板宏展开体系

**原理**：`infer_flash_attention_comm.h` 定义了宏展开体系，将模板参数列表声明为可重用的宏：

```cpp
#define CUBE_BLOCK_TRAITS_TYPE_FIELDS(X) \
    X(INPUT_T) X(T) X(OUTPUT_T)

#define CUBE_BLOCK_TRAITS_CONST_FIELDS(X) \
    X(implMode, ImplModeEnum, AA_HIGH_PRECISION) \
    X(layout, LayOutTypeEnum, None) \
    X(s1TemplateType, S1TemplateType, Aligned128) \
    // ... (14+ 常量模板参数)

#define TEMPLATES_DEF \
    template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM) \
              CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = true>
```

**效果**：类型参数和常量参数集中管理，新增模板参数只需修改宏定义，避免了大量重复的类型声明。

### 11.9 反量化 Scale 缓存

**原理**：MLA 反量化需要 Key Scale 和 Value Scale。在 UB 上预分配 `antiqScaleUb` 和 `antiqOffsetUb`：

```cpp
// flash_attention_score_block_vec_infer_mla_fullquant.h: InitGlobalBuffer
tPipe->InitBuffer(antiqScaleUb, headDim * sizeof(float));   // ~512B
tPipe->InitBuffer(antiqOffsetUb, headDim * sizeof(float));  // ~512B
```

- Vec1 阶段：拷贝 Key 反量化 Scale
- Vec2 阶段：拷贝 Value 反量化 Scale，直接 `VecMulMat` 完成 Scale 乘

**效果**：Scale 驻留 UB 避免重复从 GM 搬运。

### 11.10 GS1 Merge 下的 Vec 负载均衡

**原理**：GQA 场景下 Q 的 G×S1 被合并为 GS1 轴，两个 AIV Core（subBlockIdx=0/1）各负责一半：

```cpp
if (subBlockIdx == 0) {
    // Core 0: 处理上半段 GS1 → 对应 G0..G(gSize/2-1) × S1
    runParam.halfS1RealSize = (s1RealSize + 1) >> 1;
} else {
    // Core 1: 处理下半段 GS1
    runParam.halfS1RealSize = s1RealSize - halfS1RealSize;
    runParam.sOuterOffset += firstHalfS1RealSize;
}
```

**效果**：完美均衡两个 AIV 的 M 轴负载。

---

## 12. 量化精度与数值处理

### 12.1 FP8 反量化 Scale 精度

MLA FullQuant 的 FP8 反量化使用 `fp8e4m3MaxValue = 448.0f` 作为归一化因子：

```cpp
// vf_basic_block_utils.h
constexpr float fp8e4m3MaxValue = 448.0f;
#define USE_MLA_FULLQUANT_V1_P(vreg_exp, vreg_rowmax_p, MaskReg) \
    do { Muls(vreg_exp, vreg_exp, fp8e4m3MaxValue, MaskReg); \
         Div(vreg_exp, vreg_exp, vreg_rowmax_p, MaskReg); } while (0)
```

**原理**：FP8 E4M3 的最大可表示值为 448。将 Softmax 的 exp 值归一化到 [0, 448] 范围，再做 FP32→FP16→FP8 量化，保证量化误差最小。

### 12.2 反量化 RowMax 保护

在反量化 INT8→FP32 时，对逐行 RowMax 做 `Max(1e-6)` 保护，防止除零：

```cpp
// flash_attention_score_antiquant_processor.h
AbsRowMax(rowMaxUb, srcUb, dbSize, dealRowCount);
// 后续: Max(rowMaxUb, 1e-6) 保护
```

---

## 13. Workspace 计算

MLA FullQuant 的 Workspace 空间 = 基础部分 + BMM2 中间结果 + FlashDecode 部分：

```
workspace = sysWorkspace
          + bmm2IntRes:    3 × BM × mSize × dV × sizeof(int32_t)
          + vec2AccRes:    3 × BM × mSize × dV × sizeof(float)  (DV>256 时)
          + vec1IntRes:    3 × BM × mSize × actualS2 × sizeof(int8_t)
          + qPreProcRes:   BM × mSize × dV × sizeof(int8_t)  (ANTIQUANT 时)
          + FlashDecode:
            + accumOut:    numBlocks × splitKVNum × mSize × dV × sizeof(float)
            + lseMax:      numBlocks × splitKVNum × mSize × sizeof(float)
            + lseSum:      numBlocks × splitKVNum × mSize × sizeof(float)
```

> BM = PRE_LOAD_NUM (MLA = 2)，numBlocks = usedCoreNum_

---

## 14. 总结

### 14.1 核心设计理念

| 设计点 | 实现 | 收益 |
|--------|------|------|
| **CRTP 编译期多态** | BaseClass<Child> 模板 | 零虚函数开销，全 inline |
| **4 阶段流水** | runInfo[4] + Lag 1/2/3 | Cube/Vec 延迟完全重叠 |
| **Q 驻留 L1** | isFirstS2Loop 控制 | 减少 (n-1)/n 的 Q 搬运 |
| **Regbase 寄存器编程** | __simd_vf__ + MicroAPI | 延迟降 30-40% |
| **多路径编译期分派** | if constexpr(Nz/Dn/Nd) | 零运行时分支 |
| **AttenMask 压缩判定** | delta 计算 + skip | 节省 50% Mask 处理 |
| **MLA Rope 拼接** | nope+rope 顺序搬运 | Host 侧零开销 |
| **反量化 Scale 驻留 UB** | antiqScaleUb + antiqOffsetUb | 消除重复 GM 搬运 |
| **FlashDecode 分粒度归约** | ComputeScaleValue_8/16 | 寄存器利用率最大化 |
| **GS1 Merge 均衡** | subBlockIdx 对半拆分 | 完美负载均衡 |

### 14.2 路径决策树

```
MLA FullQuant Kernel
├─ BMM1 路径
│   ├─ isFp8?  → IterateBmm1MLAFullQuant (FP8量化)
│   └─ isInt8? → IterateBmm1MLAFullQuantInt8 (INT8量化)
│       ├─ isPa? → blockTable indirection
│       └─ 直连 → keyGm[offset]
├─ Vec1 路径
│   ├─ useNz?   → ProcessVec1Nz (HIFLOAT8反量化)
│   ├─ useDn?   → ProcessVec1Dn (FP8 Scale处理)
│   └─ default  → ProcessVec1Nd (FP16+Mask+PSE+Dropout)
├─ BMM2 路径
│   ├─ IterateBmm2MLAFullQuant (V搬运+P×V+Fixpipe)
│   └─ bmm2Write2Ub? → UB/Fixpipe vs GM/Fixpipe
├─ Vec2 路径
│   ├─ BMM2 on UB?  → ProcessVec2OnUb
│   ├─ splitD?      → ProcessVec2DSplit
│   └─ default      → ProcessVec2 (GM→UB)
└─ 输出
    ├─ POST_QUANT?  → fdPostQuant / PostQuantPerChnl
    ├─ transposeLayout → MlaTransposeDataCopyOut
    └─ default       → Bmm2DataCopyOut
```

### 14.3 与 DD Preload Kernel 的关键差异

| 方面 | DD Preload (arch32) | MLA FullQuant (arch35) |
|------|---------------------|----------------------|
| **硬件** | Ascend 910B (DAV_2201) | Ascend 950 (DAV_3510) |
| **编程模型** | AscendC Pipe API | Regbase MicroAPI |
| **MLA** | 不支持 | D=512/576 + Rope Split |
| **流水深度** | 4 阶段 (V0→C1→V1→C2→V2) | 4 阶段 (BMM1→Vec1→BMM2→Vec2) |
| **同步方式** | CrossCoreSetFlag (硬件 Flag) | Buffer::SetCrossCore() (ShareBuffer) |
| **RunInfo 槽位** | EXTRA_NUM = 6 | runInfo[4] |
| **排空** | 尾 Batch 多 2 轮 S2 | 尾 Batch 多 3 轮 GS1 |
| **量化** | Anti-quant (PerChannel/PerToken) | FullQuant (INT8/FP8) |
| **Sink** | 不支持 | Vec1SinkCompute |
| **PostQuant** | 不支持 | PostQuantPerChnl/PerTensor |
