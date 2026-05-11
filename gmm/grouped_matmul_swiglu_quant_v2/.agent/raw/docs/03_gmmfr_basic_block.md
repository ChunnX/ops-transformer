# GMMFRWeightQuantVcvBasicBlock 代码走读

## 1. 概述

`GMMFRWeightQuantVcvBasicBlock` 是 GMMFR（Grouped Matmul Finalize Routing）算子在 Arch35 架构上的 BasicBlock 层实现，负责 **AIV（Vector Core）和 AIC（Cube Core）异构协同**完成一个 BasicBlock 的完整计算流程。该算子的核心场景是 **MX A8W4**（MX 格式反量化 + INT8 左矩阵 × INT4 右矩阵），融合了 Weight 反量化、Matmul、Logits 乘法和 Routing Scatter-Add 四个阶段。

**异构分工**：
- **AIC（Cube Core）**：负责 MMA（Matrix Multiply-Accumulate）计算，即 `x @ weight` 的矩阵乘法
- **AIV（Vector Core）**：负责 Weight 反量化（INT4→INT8）、Bias 处理、Logits 乘法、Routing 写回

**关键设计**：AIC 和 AIV 通过 `CrossCoreSetFlag` / `CrossCoreWaitFlag` 实现流水线同步，AIV 先反量化 weight 供 AIC 做 MMA，AIC 算完后通过 fixpipe 将结果输出到 UB，AIV 再做后处理。

---

## 2. 类定义分析

### 2.1 模板参数

```cpp
template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType,
          typename perTokenScaleType, typename biasType, typename yType, typename sharedInputDType,
          typename logitsType, typename rowIndexType, bool hasBias, const WqmmConfig &wqmmConfig,
          const VecAntiQuantConfig &vecConfig>
```

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `xType` | 左矩阵（激活值）数据类型 | `fp8_e4m3fn_t` |
| `wType` | 右矩阵（权重）数据类型 | `int4b_t` |
| `antiQuantScaleType` | 反量化 scale 类型 | `float` |
| `scaleType` | 量化 scale 类型 | `int64_t` |
| `perTokenScaleType` | per-token scale 类型（MX scale A） | `float` |
| `biasType` | Bias 数据类型 | `half` |
| `yType` | 输出数据类型 | `float` |
| `sharedInputDType` | 共享输入数据类型 | `half` |
| `logitsType` | Logits 数据类型 | `float` |
| `rowIndexType` | Routing 索引类型 | `int64_t` |
| `hasBias` | 是否有 Bias | `true` / `false` |
| `wqmmConfig` | Weight Quant Matmul 配置（含反量化类型、Cube 格式等） | `MXA8W4_NZNK` |
| `vecConfig` | Vector 反量化配置（UB buffer 数量等） | `VEC_ANTIQUANT_CONFIG_DYNAMIC` |

### 2.2 成员变量

```cpp
// 核心计算单元
GMMFRWeightQuantVecCompute<...> vecCompute_;   // AIV 侧：反量化 + Logits + Routing
GMMFRWeightQuantCubeCompute<...> cubeCompute_; // AIC 侧：MMA 计算

// 循环索引
uint64_t cvLoopIdx_ = 0;        // 跨 BasicBlock 的 K 循环全局索引
uint64_t rlLoopIdx_ = 0;        // Routing/Logits 的 double-buffer 索引

// L1 Buffer（AIC/AIV 共享）
uint64_t weightL1DbOffset_ = 0; // Weight L1 double-buffer 偏移
uint64_t biasL1DbOffset_ = 0;   // Bias L1 double-buffer 偏移
LocalTensor<xType> weightL1_;   // Weight 在 L1 的存储（AIV 反量化写入，AIC MMA 读取）
LocalTensor<biasType> biasL1_;  // Bias 在 L1 的存储
LocalTensor<float> ubOutputF32Buffer_; // UB 上的 F32 输出 buffer（AIC fixpipe 输出，AIV 后处理读取）
```

### 2.3 AIC↔AIV 同步方法

同步基于 `CrossCoreSetFlag` / `CrossCoreWaitFlag` 实现，使用 `SYNC_MODE4` 模式。同步标志定义在 `tool.h`：

```cpp
static constexpr uint64_t SYNC_AIV_AIC_FLAG = 8;       // AIV→AIC: weight 反量化完成
static constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;       // AIC→AIV: MMA 计算完成，可继续搬运
static constexpr uint64_t SYNC_AIC_FIX_AIV_VF_FLAG = 3; // AIC→AIV: fixpipe 输出完成
static constexpr uint64_t SYNC_AIV_MTE3_AIC_FIX_FLAG = 4; // AIV→AIC: 后处理完成，可 fixpipe
static constexpr uint64_t FLAG_ID_MAX = 16;
```

**四个同步原语**：

```cpp
// AIV 通知 AIC（单 flag）
SetAivToAic<PIPE>(syncFlag)
    → CrossCoreSetFlag<SYNC_MODE4, PIPE>(syncFlag)

// AIC 等待 AIV（双 flag 机制：先等 FLAG_ID_MAX+syncFlag，再等 syncFlag）
WaitAivToAic<PIPE>(syncFlag)
    → CrossCoreWaitFlag<SYNC_MODE4, PIPE>(syncFlag + FLAG_ID_MAX)
    → CrossCoreWaitFlag<SYNC_MODE4, PIPE>(syncFlag)

// AIC 通知 AIV（双 flag 机制）
SetAicToAiv<PIPE>(syncFlag)
    → CrossCoreSetFlag<SYNC_MODE4, PIPE>(syncFlag + FLAG_ID_MAX)
    → CrossCoreSetFlag<SYNC_MODE4, PIPE>(syncFlag)

// AIV 等待 AIC（单 flag）
WaitAicToAiv<PIPE>(syncFlag)
    → CrossCoreWaitFlag<SYNC_MODE4, PIPE>(syncFlag)
```

> **双 flag 机制说明**：AIC→AIV 和 AIV→AIC 的等待/通知使用不对称设计。AIC 通知 AIV 时设置两个 flag（`syncFlag` 和 `syncFlag+16`），AIV 等待时只等 `syncFlag`；AIV 通知 AIC 时只设置一个 flag，AIC 等待时先等 `syncFlag+16` 再等 `syncFlag`。这种设计确保跨 core 同步的可靠性和流水线正确性。

---

## 3. Init() 分析

```cpp
void Init(uint64_t antiQuantGroupSize, __gm__ yType *y, float sharedInputWeight)
```

### 3.1 L1 Buffer 分配

L1（512KB）在 AIC 和 AIV 之间共享，分配策略如下：

```
L1 布局（有 Bias 场景）:
┌──────────────────────────────────────────────────────────────────┐
│ WeightL1_P0(64KB) │ Bias_P0(4KB) │ ScaleAL1_P0(32KB) │ ScaleBL1_P0(32KB) │ AL1_P0(124KB) │
│ AL1_P1(124KB)     │ ScaleBL1_P1(32KB) │ ScaleAL1_P1(32KB) │ Bias_P1(4KB) │ WeightL1_P1(64KB) │
└──────────────────────────────────────────────────────────────────┘
```

- **Weight L1**：`weightL1_` 从 L1 起始地址开始，占满 512KB。`weightL1DbOffset_` 指向 P0 区域的末尾（`L1_SIZE * KBUnit - 256*256`），即 256×256 的 weight 块大小
- **Bias L1**：`biasL1_` 紧接 Weight 之后，`biasL1DbOffset_` 指向 P0 Bias 区域的末尾
- **Scale A/B L1** 和 **A L1**：由 `cubeCompute_.MxA8W4Init()` 在 AIC 侧进一步分配

### 3.2 UB Buffer 分配

```cpp
constexpr uint64_t ubOffset = GetGmmFRMxA8W4BufferInfo<vecConfig>().weightLowbitTotalSize;
constexpr uint64_t highBitSize = GetGmmFRMxA8W4BufferInfo<vecConfig>().weightLowbitTotalSize;
ubOutputF32Buffer_ = LocalTensor<float>(TPosition::LCM, ubOffset, highBitSize);
```

`GetGmmFRMxA8W4BufferInfo()` 返回 UB buffer 布局信息：

| Buffer | 大小 | 用途 |
|--------|------|------|
| weightLowBit | 80KB | 存放 INT4 低比特 weight（MTE2 搬运） |
| weightHighBit | 160KB (4-buffer) | 存放反量化后的 INT8 高比特 weight |
| bias | 1KB | 存放 Bias |
| biasRescale | 1KB | 存放 rescale 后的 Bias |
| logits | 1KB (2-buffer) | 存放 Logits |
| rowIndex | 2KB (2-buffer) | 存放 Routing 索引 |

`ubOutputF32Buffer_` 复用 `weightLowbitTotalSize` 大小的空间，位于 UB 的 `weightLowbitTotalSize` 偏移处，用于接收 AIC fixpipe 输出的 F32 结果。

### 3.3 AIC/AIV 初始化分流

```cpp
if ASCEND_IS_AIC {
    cubeCompute_.MxA8W4Init(l1RemainSize, l1StartSize, biasL1DbOffset_, biasL1_);
    SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);  // 通知 AIV：AIC 初始化完成
    SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);  // 双 flag 确保可靠性
} else {
    vecCompute_.Init(sharedInputWeight, y);       // AIV 初始化 UB buffer 和 EventID
}
```

AIC 侧初始化 L1 上的 Scale A/B L1 和 A L1 空间，并设置 MTE1↔MTE2 的 HardEvent。AIV 侧初始化 UB 上的 weight/bias/logits/rowIndex buffer 和 V↔MTE2、V↔MTE3 的同步 EventID。

---

## 4. InitAtomicGm() 分析

```cpp
void InitAtomicGm(uint64_t initSize, uint64_t sharedInputStartSize,
                  uint64_t sharedInputSize, __gm__ sharedInputDType *shareInputAddr)
```

### 4.1 GM 清零逻辑

```cpp
uint64_t yGmOffset = GetBlockIdx() * initZeroBufferSize;
InitGmZeroWithIterate(yGmOffset, initSize, sharedInputStartSize, sharedInputSize);
```

`InitGmZeroWithIterate` 将 GM 上的输出区域清零。由于输出区域中间可能穿插 sharedInput 区域，需要分段处理：

```
GM 输出布局:
┌──────────────────────┬──────────────────────┬──────────────────────┐
│   MoE 专家输出区域    │  SharedInput 区域     │   MoE 专家输出区域    │
│   (需要清零)          │  (跳过，后续填充)      │   (需要清零)          │
└──────────────────────┴──────────────────────┴──────────────────────┘
```

清零通过 `vecCompute_.InitGmToZero()` 实现：在 VF 上用 `Duplicate(0)` 生成零向量，通过 MTE3 写出到 GM。

### 4.2 SharedInput 初始化搬运

```cpp
for (uint64_t sharedInputGmOffset = GetBlockIdx() * mte2BufferSize;
     sharedInputGmOffset < sharedInputSize;
     sharedInputGmOffset += GetBlockNum() * mte2BufferSize) {
    vecCompute_.WaitVToMTE2();
    vecCompute_.CopyShareInputGmToUb(...);  // GM→UB: 加载 sharedInput
    vecCompute_.CopyShareInputUbToGm(...);  // UB→GM: 乘 sharedInputWeight 后写出
    vecCompute_.SetVToMTE2();
}
```

多 core 并行搬运 sharedInput：每个 core 从 `GetBlockIdx() * bufferSize` 开始，步长为 `GetBlockNum() * bufferSize`。搬运流程：
1. **GM→UB**：MTE2 将 sharedInput（BF16）加载到 UB 的 weightLowBit buffer
2. **VF 处理**：`CastAndMulWithSharedWeightVf` 将 BF16→F32，乘以 `sharedInputWeight`，写入 weightHighBit buffer
3. **UB→GM**：MTE3 将处理后的 F32 结果写出到 GM 的 sharedInput 区域

---

## 5. ComputeBasicBlock() 分析

```cpp
void ComputeBasicBlock(const BasicBlockOffsetParam &curOffsetParam,
                       const BasicBlockOffsetParam &lastOffsetParam)
{
    if ASCEND_IS_AIV {
        IterateNzNkWithAiv(curOffsetParam, lastOffsetParam);
    } else {
        IterateNzNkWithKAic(curOffsetParam, lastOffsetParam);
    }
}
```

`BasicBlockOffsetParam` 包含当前 BasicBlock 的维度信息：

```cpp
struct BasicBlockOffsetParam {
    uint64_t mL1Size;   // M 方向 L1 载入大小
    uint64_t kaL1Size;  // K 方向 A 矩阵 L1 载入大小
    uint64_t kbL1Size;  // K 方向 B 矩阵 L1 载入大小
    uint64_t nL1Size;   // N 方向 L1 载入大小
    uint64_t mOffset;   // M 方向全局偏移
    uint64_t nOffset;   // N 方向全局偏移
    uint64_t mSize;     // M 方向总大小
    uint64_t kSize;     // K 方向总大小
    uint64_t nSize;     // N 方向总大小
    uint64_t kAlign;    // K 方向对齐值
    uint64_t nAlign;    // N 方向对齐值
    GM_ADDR yGmAddr;    // 输出 GM 地址
};
```

---

## 6. IterateNzNkWithAiv() 详细分析（AIV 路径）

AIV 路径采用**三段式流水线结构**：Pass1（预加载）→ PhaseB（后处理上一个 BasicBlock）→ Pass2（继续加载）。

### 6.1 整体流程

```
IterateNzNkWithAiv 流程:
┌─────────────────────────────────────────────────────────────────┐
│ Pass1: VecComputeNzNkWithStartLimit(0, DOUBLE_BUFFER_NUM * kbL1Size) │
│   → 预加载前 2 个 K 块的 weight，触发反量化流水线                    │
├─────────────────────────────────────────────────────────────────┤
│ PhaseB: 处理上一个 BasicBlock 的后处理（cvLoopIdx_ > 0 时）        │
│   → 等待 rowIndex/logits 搬运完成                                  │
│   → 两 core 切 M：CopyRowIndexLogitsGmToUb                        │
│   → MulLogits() + RoutingYToGm()                                  │
├─────────────────────────────────────────────────────────────────┤
│ Pass2: VecComputeNzNkWithStartLimit(kMte2Offset, kSize)           │
│   → 继续加载剩余 K 块的 weight                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 VecComputeNzNkWithStartLimit() — Weight 反量化主循环

```cpp
void VecComputeNzNkWithStartLimit(uint64_t &kMte2Offset, uint64_t kMte2Limit,
                                  const BasicBlockOffsetParam &curOffsetParam)
{
    uint64_t kMte2BaseSize = curOffsetParam.kbL1Size / DOUBLE_BUFFER_NUM;

    for (; kMte2Offset < kMte2Limit; kMte2Offset += curOffsetParam.kbL1Size, cvLoopIdx_++) {
        // Step 1: 等待 VF 完成上一轮，MTE2 buffer 可用
        vecCompute_.WaitVToMTE2();

        // Step 2: 计算 bias 搬运参数（两 core 切 N）
        uint64_t biasL1Offset = GetSubBlockIdx() * MX_BIAS_SINGLE_VECTOR_SIZE;
        uint64_t biasGmOffset = curOffsetParam.nOffset + biasL1Offset;
        // ... 边界处理 ...

        // Step 3: 计算 K 方向搬运参数（两 core 切 K）
        uint64_t kL1Offset = GetSubBlockIdx() * kMte2BaseSize;
        uint64_t mte2RealK = /* 边界处理 */;

        // Step 4: MTE2 搬运 weight(INT4) + bias 到 UB
        vecCompute_.CopyGmToUb(mte2RealK, kMte2Offset, kL1Offset,
                               biasGmOffset, biasRealSize, curOffsetParam);

        // Step 5: 等待 AIC 完成上一轮 MMA（AIC→AIV 同步点）
        WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);

        // Step 6: VF 反量化 weight(INT4→INT8) + bias rescale，写出到 L1
        vecCompute_.WeightAntiQuantComputeNzNk(
            mte2RealK, biasRealSize, kMte2Offset,
            weightL1_[(cvLoopIdx_ & 1) * weightL1DbOffset_ + ...],
            biasL1_[(cvLoopIdx_ & 1) * biasL1DbOffset_ + biasL1Offset],
            curOffsetParam);

        // Step 7: 通知 AIC weight 反量化完成（AIV→AIC 同步点）
        SetAivToAic<PIPE_MTE3>(SYNC_AIV_AIC_FLAG);

        // Step 8: 释放 MTE2 buffer
        vecCompute_.SetVToMTE2();
    }
}
```

**两 core 切分策略**：
- **K 方向**：`kMte2BaseSize = kbL1Size / 2`，SubBlockIdx=0 处理前半段 K，SubBlockIdx=1 处理后半段
- **N 方向（Bias）**：`biasL1Offset = SubBlockIdx * 128`，SubBlockIdx=0 处理前半段 N，SubBlockIdx=1 处理后半段

**Weight 反量化核心**（`WeightAntiQuantComputeNzNk` → `AntiQuantProcessNzMxA8W4` → `FrAntiQuantMxA8W4NzNkVf`）：

VF 上执行 INT4→INT8 的位操作反量化：
1. 从 UB 加载 INT4 weight（`DIST_US_B8` 模式，每个字节解包为两个 4bit 值）
2. 通过 `ShiftRight` + `ShiftLeft` + `Select` + `And` 提取高/低 4bit 并符号扩展为 INT8
3. 写出到 UB 的 weightHighBit buffer（4-buffer 跳写避免 bank 冲突）
4. 通过 MTE3 将反量化后的 INT8 weight 写出到 L1 的 `weightL1_`

### 6.3 PhaseB：后处理阶段

```cpp
if (curCvLoopIdx > 0) {  // 非首个 BasicBlock 时执行
    // 两 core 切 M
    uint64_t mSingleVec = CeilDivide(curOffsetParam.mL1Size, 2UL);
    vecCompute_.CopyRowIndexLogitsGmToUb(
        curOffsetParam.mOffset + GetSubBlockIdx() * mSingleVec,
        GetSubBlockIdx() == 0 ? mSingleVec : curOffsetParam.mL1Size - mSingleVec,
        rlLoopIdx_);

    // 同步：通知 AIC 可以 fixpipe，等待 AIC fixpipe 完成
    SetAivToAic<PIPE_MTE3>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
    WaitAicToAiv<PIPE_V>(SYNC_AIC_FIX_AIV_VF_FLAG);

    // MulLogits: y *= logits * 64.0
    vecCompute_.MulLogits(ubOutputF32Buffer_, lastBasicBlockMSize, lastOffsetParam, rlLoopIdx_);

    // RoutingYToGm: scatter-add 写回 GM
    vecCompute_.RoutingYToGm(lastBasicBlockMSize, ubOutputF32Buffer_, lastOffsetParam, rlLoopIdx_);
}
```

**MulLogits 实现**（`FrMulLogitsVf`）：
```cpp
// 对每个 M 行，每个 N 段（每段 8 个 float）：
yFp32 = yFp32 * 64.0f;   // MX 格式的 scale 补偿
yFp32 = yFp32 * logits;  // 乘以对应 token 的 logits
```

**RoutingYToGm 实现**：
```cpp
SetAtomicAdd<float>();  // 开启 atomic add 模式
for (mIdx = 0; mIdx < mRealSize; mIdx++) {
    int64_t mGmOffset = rowIndex[...].GetValue(mIdx);
    if (mGmOffset < 0) continue;  // 跳过无效索引（padding token）
    // scatter-add: yFp32Global[mGmOffset * nSize + nOffset] += ubOutputF32Buffer[mIdx * 256]
    DataCopyPad2D(yFp32Global_[mGmOffset * nSize + nOffset], ubOutputF32Buffer[mIdx * 256], ...);
}
SetAtomicNone();
```

> **关键**：Routing 使用 `SetAtomicAdd` 模式，多个 expert 的输出可能 scatter 到同一个 token 位置，通过 GM 的 atomic add 保证累加正确性。

---

## 7. IterateNzNkWithKAic() 详细分析（AIC 路径）

AIC 路径负责 MMA 计算，与 AIV 通过同步点形成流水线。

### 7.1 整体流程

```
IterateNzNkWithKAic 流程:
┌─────────────────────────────────────────────────────────────────┐
│ PhaseB (cvLoopIdx_ > 0): 处理上一个 BasicBlock 的 fixpipe 输出    │
│   → WaitAivToAic<PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG)          │
│   → GetTensorC(ubOutputF32Buffer_, lastOffsetParam)              │
│   → SetAicToAiv<PIPE_FIX>(SYNC_AIC_FIX_AIV_VF_FLAG)             │
├─────────────────────────────────────────────────────────────────┤
│ K 循环: 对每个 kbL1Size 块执行 MMA                               │
│   for (kbL1Offset = 0; kbL1Offset < kSize; kbL1Offset += kbL1Size) │
│     → 搬运 MX Scale A/B (GM→L1)                                   │
│     → 搬运 A 矩阵 + Bias (GM→L1)                                  │
│     → Pad L1 对齐                                                 │
│     → WaitAivToAic (等待 AIV 反量化完成)  ★ 同步点                │
│     → LaunchMatmul (MMA 计算)                                     │
│     → SetAicToAiv (通知 AIV 可继续搬运)  ★ 同步点                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 K 循环详细流程

```cpp
for (uint64_t kbL1Offset = 0; kbL1Offset < curOffsetParam.kSize;
     kbL1Offset += curOffsetParam.kbL1Size, cvLoopIdx_++) {

    // 1. 等待 Scale MTE1→MTE2 完成（double-buffer 同步）
    cubeCompute_.WaitScaleMTE1ToMTE2(kbL1Offset);

    // 2. 搬运 MX Scale A/B 到 L1（Dn2Nz 格式转换）
    cubeCompute_.CopyMxScaleGmToL1(curOffsetParam, kbL1Offset);

    // 3. 等待 A 矩阵 MTE1→MTE2 完成
    cubeCompute_.WaitMTE1ToMTE2(kbL1Offset, curOffsetParam);

    // 4. 搬运 A 矩阵 + Bias 到 L1（Nd2Nz 格式转换）
    cubeCompute_.CopyAAndBiasGmToL1(curOffsetParam, kbL1Offset, cvLoopIdx_);

    // 5. 如果 K 不对齐，Pad L1 尾部为零
    cubeCompute_.PadL1IfNotAlign(weightL1_, weightL1DbOffset_, cvLoopIdx_,
                                 kbL1RealSize, curOffsetParam.nL1Size);

    // ★ 同步点：等待 AIV 完成 weight 反量化
    WaitAivToAic<PIPE_MTE1>(SYNC_AIV_AIC_FLAG);

    // 6. 启动 MMA 计算（MTE1→L0A/L0B→Mmad→L0C 流水线）
    cubeCompute_.LaunchMatmul(
        weightL1_[(cvLoopIdx_ & 1) * weightL1DbOffset_],
        kbL1Offset, kbL1RealSize, cvLoopIdx_, curOffsetParam);

    // 7. 设置 MTE1→MTE2 完成标志（释放 double-buffer）
    cubeCompute_.SetMTE1ToMTE2(kbL1Offset, curOffsetParam);
    cubeCompute_.SetScaleMTE1ToMTE2(kbL1Offset, curOffsetParam);

    // ★ 同步点：通知 AIV 可以继续搬运下一块 weight
    SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
}
```

### 7.3 LaunchMatmul — MMA 计算核心

```cpp
void LaunchMatmul(weightL1, kbOffset, kbL1RealSize, cvLoopIdx, param) {
    // L0C double-buffer 管理
    if (kbOffset == 0) {
        // 检查 L0C 空间是否足够，不够则回绕
        if (l0cOffset_ + nextL0cBufSize > L0C_BUF_SIZE) l0cOffset_ = 0;
    }

    // K 方向 L0 循环（每次处理 kL0Size=128 或 256）
    for (l1KOffset = 0; l1KOffset < kbL1RealSize; l1KOffset += kL0Size) {
        // L0A: 加载 A 矩阵 + MX Scale A（L1→L0A）
        LoadAAndScaleL1ToL0(l0a_[loopId * L0_BUF_OFFSET_B8], aL1_[...], mxScaleAL1_[...], params);

        // L0B: 加载 B 矩阵（weight）+ MX Scale B（L1→L0B）
        LoadBAndScaleL1ToL0(l0b_[loopId * L0_BUF_OFFSET_B8], weightL1[...], mxScaleBL1_[...], params);

        // Mmad: 矩阵乘加（首次 K 循环加载 Bias 到 BiasTable）
        DoMmadCompute(loopId, cvLoopIdx, params);
    }
}
```

**DoMmadCompute 流程**：
1. 首次 K 循环时，加载 Bias 到 BiasTable（L0 C2）
2. `SetFlag<MTE1_M>` + `WaitFlag<MTE1_M>`：等待 L0A/L0B 加载完成
3. `MmadCompute(l0c_, l0a_, l0b_, biasTable_, params)`：执行 MMA 指令
4. `SetFlag<M_MTE1>`：通知 MTE1 可以继续加载下一块

### 7.4 GetTensorC — Fixpipe 输出

```cpp
void GetTensorC(LocalTensor<yType> &yUb, const BasicBlockOffsetParam &param) {
    FixL0CToDstParams fixParams;
    fixParams.mL0Size = param.mL1Size;
    fixParams.nL0Size = param.nL1Size;
    fixParams.outNSize = 256;  // 输出 N stride 固定 256
    FixL0CToDst(yUb, l0c_[l0cOffset_], fixParams);
    // 更新 L0C offset 为下一个 BasicBlock 腾出空间
    l0cOffset_ += CeilAlign(mL0Size, BLOCK_CUBE) * CeilAlign(nL0Size, BLOCK_CUBE);
}
```

Fixpipe 将 L0C 上的累加结果（F32）搬运到 UB 的 `ubOutputF32Buffer_`，供 AIV 后续做 MulLogits 和 Routing。

---

## 8. End() 分析

```cpp
void End(const BasicBlockOffsetParam &lastOffsetParam)
```

### 8.1 AIC 侧

```cpp
if ASCEND_IS_AIC {
    if (cvLoopIdx_ > 0) {
        // 最后一个 BasicBlock 的 fixpipe 输出
        WaitAivToAic<PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
        cubeCompute_.GetTensorC(ubOutputF32Buffer_, lastOffsetParam);
        SetAicToAiv<PIPE_FIX>(SYNC_AIC_FIX_AIV_VF_FLAG);
    }
    // 等待所有 MTE1→MTE2 和 M→MTE1 流水线完成
    cubeCompute_.EndSync();
}
```

### 8.2 AIV 侧

```cpp
else {
    if (cvLoopIdx_ > 0) {
        // 最后一个 BasicBlock 的后处理
        SetAivToAic<PIPE_MTE3>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
        WaitAicToAiv<PIPE_V>(SYNC_AIC_FIX_AIV_VF_FLAG);

        vecCompute_.MulLogits(ubOutputF32Buffer_, lastBasicBlockMSize, lastOffsetParam, rlLoopIdx_ - 1);
        vecCompute_.RoutingYToGm(lastBasicBlockMSize, ubOutputF32Buffer_, lastOffsetParam, rlLoopIdx_ - 1);
    }
    // 等待 AIC 完成最后两次搬运
    WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
    WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
    // 等待所有 V↔MTE3 和 V↔MTE2 流水线完成
    vecCompute_.End();
}
```

---

## 9. AIV 路径流程图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           AIV 路径 (IterateNzNkWithAiv)                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─ Pass1: 预加载 ──────────────────────────────────────────────────────┐   │
│  │  for (k = 0; k < 2*kbL1Size; k += kbL1Size):                         │   │
│  │    WaitVToMTE2()                                                      │   │
│  │    CopyGmToUb()          ← MTE2: weight(INT4) + bias GM→UB           │   │
│  │    WaitAicToAiv(FLAG9)   ← ★ 等 AIC 完成上一轮 MMA                    │   │
│  │    WeightAntiQuant()     ← VF: INT4→INT8 反量化, MTE3→L1              │   │
│  │    SetAivToAic(FLAG8)    ← ★ 通知 AIC weight 就绪                     │   │
│  │    SetVToMTE2()                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│  ┌─ PhaseB: 后处理（cvLoopIdx_ > 0 时） ─────────────────────────────────┐   │
│  │  WaitSToMTE2() / WaitFrVToMTE2()                                      │   │
│  │  CopyRowIndexLogitsGmToUb() ← MTE2: rowIndex + logits GM→UB          │   │
│  │  SetMte2ToS()                                                         │   │
│  │  SetAivToAic(FLAG4)         ← ★ 通知 AIC 可以 fixpipe                 │   │
│  │  WaitAicToAiv(FLAG3)        ← ★ 等 AIC fixpipe 完成                   │   │
│  │  MulLogits()                ← VF: y *= logits * 64.0                  │   │
│  │  RoutingYToGm()             ← MTE3: scatter-add 写回 GM               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│  ┌─ Pass2: 继续加载 ────────────────────────────────────────────────────┐   │
│  │  for (k = 2*kbL1Size; k < kSize; k += kbL1Size):                     │   │
│  │    (同 Pass1)                                                         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. AIC 路径流程图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           AIC 路径 (IterateNzNkWithKAic)                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─ PhaseB: fixpipe 输出（cvLoopIdx_ > 0 时） ───────────────────────────┐   │
│  │  WaitAivToAic(FLAG4)      ← ★ 等 AIV 后处理就绪                       │   │
│  │  GetTensorC()             ← Fixpipe: L0C→UB (F32)                     │   │
│  │  SetAicToAiv(FLAG3)       ← ★ 通知 AIV fixpipe 完成                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│  ┌─ K 循环 ─────────────────────────────────────────────────────────────┐   │
│  │  for (kbL1Offset = 0; kbL1Offset < kSize; kbL1Offset += kbL1Size):   │   │
│  │    WaitScaleMTE1ToMTE2()                                              │   │
│  │    CopyMxScaleGmToL1()    ← MTE1: MX Scale A/B GM→L1 (Dn2Nz)         │   │
│  │    WaitMTE1ToMTE2()                                                   │   │
│  │    CopyAAndBiasGmToL1()   ← MTE1: A矩阵 + Bias GM→L1 (Nd2Nz)         │   │
│  │    PadL1IfNotAlign()      ← K 不对齐时尾部填零                        │   │
│  │    WaitAivToAic(FLAG8)    ← ★ 等 AIV weight 反量化完成                │   │
│  │    LaunchMatmul()         ← MMA: L1→L0A/L0B→Mmad→L0C                 │   │
│  │    SetMTE1ToMTE2()                                                    │   │
│  │    SetScaleMTE1ToMTE2()                                               │   │
│  │    SetAicToAiv(FLAG9)     ← ★ 通知 AIV 可继续搬运                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. AIC↔AIV 同步点汇总

```
时间线（单个 BasicBlock 内，K 循环的一次迭代）:

AIV (Vector Core)                          AIC (Cube Core)
     │                                          │
     │  CopyGmToUb (MTE2: weight→UB)            │  CopyMxScaleGmToL1 (MTE1: scale→L1)
     │  WaitAicToAiv(FLAG9) ←────────────────── │  CopyAAndBiasGmToL1 (MTE1: A→L1)
     │  WeightAntiQuant (VF: INT4→INT8)         │  PadL1IfNotAlign
     │  MTE3→L1 (weight→L1)                     │  WaitAivToAic(FLAG8) ←──────────┐
     │  SetAivToAic(FLAG8) ──────────────────→  │  LaunchMatmul (MMA)              │
     │  SetVToMTE2                              │  SetAicToAiv(FLAG9) ──────────→  │
     │                                          │                                 │
     │  (下一轮迭代)                             │  (下一轮迭代)                    │
     │                                          │                                 │
     │  ─── BasicBlock 边界 ───                  │  ─── BasicBlock 边界 ───         │
     │                                          │                                 │
     │  CopyRowIndexLogitsGmToUb                │  WaitAivToAic(FLAG4) ←────────┐ │
     │  SetAivToAic(FLAG4) ──────────────────→  │  GetTensorC (fixpipe: L0C→UB) │ │
     │  WaitAicToAiv(FLAG3) ←─────────────────  │  SetAicToAiv(FLAG3) ────────→ │ │
     │  MulLogits + RoutingYToGm                │                               │ │
```

| 同步标志 | 方向 | Pipe | 含义 |
|----------|------|------|------|
| `SYNC_AIV_AIC_FLAG (8)` | AIV→AIC | MTE1/MTE3 | AIV 完成 weight 反量化，AIC 可以开始 MMA |
| `SYNC_AIC_AIV_FLAG (9)` | AIC→AIV | MTE1/MTE3 | AIC 完成 MMA，AIV 可以继续搬运下一块 weight |
| `SYNC_AIV_MTE3_AIC_FIX_FLAG (4)` | AIV→AIC | MTE3/FIX | AIV 后处理就绪，AIC 可以 fixpipe 输出 |
| `SYNC_AIC_FIX_AIV_VF_FLAG (3)` | AIC→AIV | FIX/V | AIC fixpipe 完成，AIV 可以做 MulLogits + Routing |

---

## 12. 关键设计要点

### 12.1 Double Buffer 机制

- **L1 Weight**：`weightL1DbOffset_` 将 L1 分为 P0/P1 两个半区，`cvLoopIdx_ & 1` 选择当前使用的半区
- **L1 Bias**：`biasL1DbOffset_` 同理
- **UB weightHighBit**：4-buffer（QUADRUPLE_BUFFER_NUM=4），VF 反量化输出和 MTE3 搬运到 L1 之间流水
- **UB weightLowBit**：2-buffer（ubMte2BufferNum=2），MTE2 搬运和 VF 消费之间流水
- **L0A/L0B**：2-buffer，MTE1 加载和 MMA 计算之间流水

### 12.2 两 Core 切分

每个 BasicBlock 由 2 个 SubBlock（AIV 的 sub-core）协同处理：
- **M 方向切分**（Routing/Logits 阶段）：`mSingleVec = CeilDivide(mL1Size, 2)`，SubBlockIdx=0 处理前一半，SubBlockIdx=1 处理后一半
- **K 方向切分**（Weight 反量化阶段）：`kMte2BaseSize = kbL1Size / 2`，两个 SubBlock 各处理一半 K
- **N 方向切分**（Bias 搬运）：`biasL1Offset = SubBlockIdx * 128`，两个 SubBlock 各处理一半 N

### 12.3 MX 格式 Scale 处理

MX 格式的 scale 以 group size=32 为单位。在 AIC 侧，MX Scale A（per-token scale）和 MX Scale B（per-channel scale）以 `MX_SCALE_K_L1_SIZE=4096` 字节为单位分块搬运到 L1，通过 `Dn2Nz` 格式转换适配 Cube 的 NZ 格式要求。在 MMA 计算时，L0A/L0B 加载同时携带对应的 scale 数据。

### 12.4 Atomic Add 的 Routing

Routing 阶段使用 `SetAtomicAdd<float>()` 开启 GM 的 atomic add 模式。多个 expert 的输出可能 scatter 到同一个 token 位置（rowIndex 相同），通过 GM 硬件 atomic add 保证累加正确性。`rowIndex < 0` 的条目表示 padding token，直接跳过。

---

## 13. 调用时序总结

```
完整 BasicBlock 生命周期:

1. Init()              — 分配 L1/UB buffer，初始化同步 EventID
2. InitAtomicGm()      — GM 清零 + sharedInput 初始化
3. UpdateGlobalAddr()  — 更新 GM 地址指针
4. ComputeBasicBlock() — 执行 AIV/AIC 异构计算
   ├── BasicBlock 0: 仅 Pass1（预加载 weight），无 PhaseB
   ├── BasicBlock 1: Pass1 + PhaseB(处理 BB0 结果) + Pass2
   └── BasicBlock N: Pass1 + PhaseB(处理 BB{N-1} 结果) + Pass2
5. End()               — 最后一个 BasicBlock 的 PhaseB + 流水线收尾
```

首个 BasicBlock（`cvLoopIdx_ == 0`）没有 PhaseB 阶段，因为还没有 MMA 结果需要后处理。从第二个 BasicBlock 开始，Pass1 预加载当前 BB 的 weight，PhaseB 处理上一个 BB 的 MMA 结果（MulLogits + Routing），Pass2 继续加载当前 BB 剩余 weight，形成**计算与后处理的流水线重叠**。