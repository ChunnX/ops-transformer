# GMMFR CubeCompute 层代码走读

## 1. 概述

`GMMFRWeightQuantCubeCompute` 是 GMMFR（GroupedMatmulFinalizeRouting）算子在 AIC（AI Core）上执行 MMA 矩阵乘计算的核心类。它负责 MxA8W4（MX Scale + A8W4）场景下的 Cube 计算流水线：从 GM 搬运 A 矩阵和 MX Scale 到 L1，再从 L1 加载到 L0，启动 MMAD 计算，最后通过 Fixpipe 将 L0C 结果输出到 UB。

该类位于 `WeightQuantBatchMatmulV2::Arch35` 命名空间，与 `GMMFRWeightQuantVcvBasicBlock`（VCV 主控类）配合工作——VCV 层在 AIC 侧调用 CubeCompute 完成 MMA 计算，在 AIV 侧调用 VecCompute 完成反量化和 Routing。

---

## 2. 类定义分析

### 2.1 模板参数

```cpp
template <typename xType, typename biasType, typename antiQuantScaleType,
          typename perTokenScaleType, typename yType, bool hasBias,
          const WqmmConfig &wqmmConfig>
class GMMFRWeightQuantCubeCompute { ... };
```

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `xType` | A 矩阵数据类型 | `fp8_e4m3fn_t` |
| `biasType` | Bias 数据类型 | `float` |
| `antiQuantScaleType` | 反量化 Scale 类型（ScaleB） | `float` |
| `perTokenScaleType` | Per-Token Scale 类型（ScaleA） | `float` |
| `yType` | 输出类型 | `float` |
| `hasBias` | 是否有 Bias | `true` / `false` |
| `wqmmConfig` | 配置结构体 | `MXA8W4_NZNK` |

### 2.2 成员变量分析

#### L0 Buffer（AIC 内部高速缓存）

```cpp
// L0A: A2 位置，大小 = 64 * KB_UNIT / sizeof(L0DataType)
// 对于 int8: KB_UNIT = 1024, 所以 64*1024/1 = 65536 元素
// 双缓冲: [0, 32KB) 和 [32KB, 64KB)
LocalTensor<L0DataType> l0a_{TPosition::A2, 0, 64 * GetKBUnit<int8_t>() / sizeof(L0DataType)};

// L0B: B2 位置，大小同 L0A
LocalTensor<L0DataType> l0b_{TPosition::B2, 0, 64 * GetKBUnit<int8_t>() / sizeof(L0DataType)};

// L0C: CO1 位置，大小 = 256 * KB_UNIT / sizeof(float)
// 256*1024/4 = 65536 个 float 元素 = 256KB
// 用于存放 MMA 累加结果
LocalTensor<float> l0c_{TPosition::CO1, 0, 256 * GetKBUnit<int8_t>() / sizeof(float)};

// BiasTable: C2 位置，大小 = 4 * KB_UNIT / sizeof(float)
// 4*1024/4 = 1024 个 float 元素 = 4KB
// 用于存放 Bias 数据供 MMAD 使用
LocalTensor<float> biasTable_{TPosition::C2, 0, 4 * GetKBUnit<int8_t>() / sizeof(float)};
```

**L0 Buffer 布局图：**

```
L0 空间布局:
┌─────────────────────────────────────────────────────────────┐
│ L0A (A2)    │ L0B (B2)    │ L0C (CO1)   │ BiasTable (C2)  │
│ 64KB        │ 64KB        │ 256KB       │ 4KB             │
│ [DB0|DB1]   │ [DB0|DB1]   │ 可存放多个   │ [DB0|DB1]       │
│ 32KB each   │ 32KB each   │ M×N 分块     │ 2KB each        │
└─────────────────────────────────────────────────────────────┘
```

关键常量：
- `L0_BUF_NUM = 2`：L0A/L0B 双缓冲数量
- `L0_BUF_OFFSET_B8 = 32 * 1024`：每个 L0A/L0B 缓冲区的偏移（32KB，单位 B8）
- `BIAS_TABLE_OFFSET_B32 = 512`：BiasTable 双缓冲偏移（512 个 float = 2KB）
- `L0C_BUF_SIZE = 256 * 256`：L0C 总容量（256×256 个 float 元素）

#### L1 Buffer（AIC 与 AIV 共享的片上存储）

```cpp
LocalTensor<xType> aL1_;              // A 矩阵在 L1 的缓冲区（双缓冲）
uint64_t aL1DbOffset_;                // A 矩阵双缓冲偏移

LocalTensor<biasType> biasL1_;        // Bias 在 L1 的缓冲区（双缓冲）
uint64_t biasL1DbOffset_;             // Bias 双缓冲偏移

LocalTensor<fp8_e8m0_t> mxScaleAL1_;  // ScaleA (per-token scale) 在 L1 的缓冲区
uint64_t mxScaleAL1DbOffset_;         // ScaleA 双缓冲偏移

LocalTensor<fp8_e8m0_t> mxScaleBL1_;  // ScaleB (anti-quant scale) 在 L1 的缓冲区
uint64_t mxScaleBL1DbOffset_;         // ScaleB 双缓冲偏移
```

#### 其他成员变量

```cpp
int8_t aL1DbNum_;              // A 矩阵双缓冲数量（GMM 动态场景设为 DOUBLE_BUFFER_NUM=2）
uint64_t aL1Count_;            // A 分段全载计数（GMM 动态场景暂关闭，设为 0）
uint64_t aL1MaxHalfCount_;     // A 分段全载最大半计数（GMM 动态场景暂关闭，设为 0）
uint64_t mxScaleBufIdx_;       // MX Scale 双缓冲索引
uint64_t aL1BufIdx_;           // A 矩阵 L1 双缓冲索引
uint64_t l0LoopIdx_;           // L0 循环索引（用于 L0A/L0B 双缓冲切换）
uint64_t l0cOffset_;           // L0C 写入偏移（支持 L0C 存放多个 M×N 分块结果）

// Event ID 数组（双缓冲各一个）
AscendC::TEventID cubeEventIdsMxScaleMte1ToMte2_[DOUBLE_BUFFER_NUM];  // Scale MTE1→MTE2 同步
AscendC::TEventID cubeEventIdsMte1ToMte2_[DOUBLE_BUFFER_NUM];         // A 矩阵 MTE1→MTE2 同步

// Global Memory 地址
GlobalTensor<xType> xGlobal_;                    // A 矩阵 GM 地址
GlobalTensor<biasType> biasGlobal_;              // Bias GM 地址
GlobalTensor<fp8_e8m0_t> mxScaleAGlobal_;        // ScaleA (per-token scale) GM 地址
GlobalTensor<fp8_e8m0_t> mxScaleBGlobal_;        // ScaleB (anti-quant scale) GM 地址
GlobalTensor<yType> yGlobal_;                    // 输出 GM 地址
```

---

## 3. MxA8W4Init() 分析

`MxA8W4Init` 在 VCV 层的 `Init()` 中被调用，负责从 L1 剩余空间中分配 CubeCompute 所需的各个缓冲区。

### 3.1 调用上下文

```cpp
// gmm_fr_weight_quant_vcv_basic_block.h: Init()
weightL1_ = LocalTensor<xType>(TPosition::TSCM, 0, L1_SIZE_BYTE / sizeof(xType));
// Weight 占用 L1 前 128KB（256×256 双缓冲）
weightL1DbOffset_ = L1_SIZE * GetKBUnit<xType>() - MXA8W4_WEIGHT_SIZE;
uint64_t l1RemainSize = L1_SIZE_BYTE - MXA8W4_WEIGHT_SIZE * DOUBLE_BUFFER_NUM;  // 512KB - 128KB = 384KB
uint64_t l1StartSize = MXA8W4_WEIGHT_SIZE;  // 从 64KB 位置开始分配

// 如果有 Bias，先分配 Bias 空间
if constexpr (hasBias) {
    biasL1DbOffset_ = (l1RemainSize - MX_BIAS_L1_SIZE) / sizeof(biasType);
    l1RemainSize -= DOUBLE_BUFFER_NUM * MX_BIAS_L1_SIZE;  // 减去 8KB
    l1StartSize += MX_BIAS_L1_SIZE;                        // 偏移 4KB
}

// 然后调用 CubeCompute 的 MxA8W4Init
cubeCompute_.MxA8W4Init(l1RemainSize, l1StartSize, biasL1DbOffset_, biasL1_);
```

### 3.2 内部分配逻辑

```cpp
__aicore__ inline void MxA8W4Init(uint64_t l1RemainSize, uint64_t l1StartSize,
                                   uint64_t mxBiasL1DbOffset, const LocalTensor<biasType> &biasL1)
{
    biasL1_ = biasL1;
    biasL1DbOffset_ = mxBiasL1DbOffset;
    aL1Count_ = 0;         // GMM 动态场景暂时关闭 A 分段全载策略
    aL1MaxHalfCount_ = 0;
    aL1DbNum_ = DOUBLE_BUFFER_NUM;

    // 1. 分配 mxScaleAL1_（ScaleA / per-token scale）
    mxScaleAL1_ = LocalTensor<fp8_e8m0_t>(TPosition::TSCM, l1StartSize,
                                           l1RemainSize / sizeof(fp8_e8m0_t));
    mxScaleAL1DbOffset_ = l1RemainSize - MX_SCALE_L1_SIZE;
    l1RemainSize -= DOUBLE_BUFFER_NUM * MX_SCALE_L1_SIZE;  // 减去 64KB
    l1StartSize += MX_SCALE_L1_SIZE;                        // 偏移 32KB

    // 2. 分配 mxScaleBL1_（ScaleB / anti-quant scale）
    mxScaleBL1_ = LocalTensor<fp8_e8m0_t>(TPosition::TSCM, l1StartSize,
                                           l1RemainSize / sizeof(fp8_e8m0_t));
    mxScaleBL1DbOffset_ = l1RemainSize - MX_SCALE_L1_SIZE;
    l1RemainSize -= DOUBLE_BUFFER_NUM * MX_SCALE_L1_SIZE;  // 减去 64KB
    l1StartSize += MX_SCALE_L1_SIZE;                        // 偏移 32KB

    // 3. 分配 aL1_（A 矩阵），剩余空间全部给 A
    aL1_ = LocalTensor<xType>(TPosition::TSCM, l1StartSize, l1RemainSize);
    aL1DbOffset_ = l1RemainSize >> 1;  // 双缓冲：剩余空间的一半

    // 4. 初始化 Event ID
    for (uint64_t i = 0; i < DOUBLE_BUFFER_NUM; i++) {
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_MTE1_TO_MTE2 + i);       // A 搬运完成
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_SCALE_MTE1_TO_MTE2 + i); // Scale 搬运完成
        SetFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + i);             // MMA 计算完成
    }
}
```

### 3.3 L1 Buffer 分配图

`MX_SCALE_L1_SIZE = 32 * GetKBUnit<xType>() * sizeof(xType)`，对于 fp8（1 字节），`GetKBUnit<fp8_e4m3fn_t>() = 1024`，所以 `MX_SCALE_L1_SIZE = 32KB`。

#### 有 Bias 场景（hasBias = true）

```
L1 空间布局 (512KB):
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ WeightL1 │ WeightL1 │ BiasL1   │ BiasL1   │ ScaleAL1 │ ScaleAL1 │ ScaleBL1 │ ScaleBL1 │ AL1      │ AL1      │
│ P0       │ P1       │ P0       │ P1       │ P0       │ P1       │ P0       │ P1       │ P0       │ P1       │
│ 64KB     │ 64KB     │ 4KB      │ 4KB      │ 32KB     │ 32KB     │ 32KB     │ 32KB     │ 124KB    │ 124KB    │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
0          64KB       128KB      132KB      136KB      168KB      200KB      232KB      264KB      388KB      512KB
```

#### 无 Bias 场景（hasBias = false）

```
L1 空间布局 (512KB):
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ WeightL1 │ WeightL1 │ ScaleAL1 │ ScaleAL1 │ ScaleBL1 │ ScaleBL1 │ AL1      │ AL1      │
│ P0       │ P1       │ P0       │ P1       │ P0       │ P1       │ P0       │ P1       │
│ 64KB     │ 64KB     │ 32KB     │ 32KB     │ 32KB     │ 32KB     │ 128KB    │ 128KB    │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
0          64KB       128KB      160KB      192KB      224KB      256KB      384KB      512KB
```

### 3.4 Event ID 初始化

初始化时将所有 Event 置为完成状态（SetFlag），确保首次 WaitFlag 不会阻塞：

| Event ID | 含义 | 用途 |
|----------|------|------|
| `EVENT_ID_MTE1_TO_MTE2 + i` (3, 4) | A 矩阵 MTE1→MTE2 搬运完成 | 同步 A 矩阵从 GM→L1 的搬运 |
| `EVENT_ID_SCALE_MTE1_TO_MTE2 + i` (5, 6) | Scale MTE1→MTE2 搬运完成 | 同步 Scale 从 GM→L1 的搬运 |
| `EVENT_ID_M_TO_MTE1 + i` (3, 4) | MMA 计算完成 | 同步 MMA 计算→MTE1 搬运的依赖 |

---

## 4. UpdateGlobalAddr() 分析

```cpp
__aicore__ inline void UpdateGlobalAddr(__gm__ xType *x, __gm__ yType *y,
                                         __gm__ biasType *bias,
                                         __gm__ antiQuantScaleType *antiquantScale,
                                         __gm__ uint64_t *quantScale,
                                         __gm__ perTokenScaleType *perTokenScale)
{
    xGlobal_.SetGlobalBuffer(x);       // A 矩阵 GM 地址
    yGlobal_.SetGlobalBuffer(y);       // 输出 GM 地址
    if constexpr (hasBias) {
        biasGlobal_.SetGlobalBuffer(bias);  // Bias GM 地址
    }

    // 关键映射：
    // mxScaleAGlobal_ ← perTokenScale（per-token 的 MX Scale，对应 A 矩阵的 M 方向）
    // mxScaleBGlobal_ ← antiquantScale（per-channel 的反量化 Scale，对应 B 矩阵的 N 方向）
    mxScaleAGlobal_.SetGlobalBuffer(perTokenScale);
    mxScaleBGlobal_.SetGlobalBuffer(antiquantScale);
}
```

**mxScaleAGlobal_ 和 mxScaleBGlobal_ 的含义：**

| 变量 | 对应参数 | 含义 | 作用维度 |
|------|----------|------|----------|
| `mxScaleAGlobal_` | `perTokenScale` | A 矩阵的 MX Scale（per-token），每个 token 一个 scale 值 | M 方向（每行一个） |
| `mxScaleBGlobal_` | `antiquantScale` | B 矩阵（Weight）的反量化 Scale（per-channel），每个 channel 一个 scale 值 | N 方向（每列一个） |

在 MX 量化格式中，A 矩阵使用 fp8_e4m3fn 格式，每 32 个元素共享一个 fp8_e8m0 的 scale 值。ScaleA 沿 K 方向分组（每 32 个元素一组），ScaleB 沿 K 方向分组（每 32 个元素一组）。

---

## 5. CopyMxScaleGmToL1() 分析

### 5.1 函数签名与触发条件

```cpp
__aicore__ inline void CopyMxScaleGmToL1(const BasicBlockOffsetParam &param, uint64_t kbL1Offset)
{
    // 只在 kbL1Offset 对齐 MX_SCALE_K_L1_SIZE (4096) 时触发搬运
    if (kbL1Offset % MX_SCALE_K_L1_SIZE != 0) {
        return;
    }
    ...
}
```

`MX_SCALE_K_L1_SIZE = 4096`，即 Scale 在 L1 中覆盖 K 方向 4096 个元素。由于 MX_GROUPSIZE=32，这对应 4096/32=128 个 scale 值。Scale 搬运的粒度比 A/B 矩阵搬运的粒度（`kbL1Size`）更大，因此不是每次 K 循环都搬运 Scale。

### 5.2 搬运参数计算

```cpp
uint64_t kSizeAligned = CeilAlign(param.kSize, K_ALIGNMENT64);  // K 方向 64 对齐
uint64_t scaleKGmSize = kSizeAligned / MX_GROUPSIZE;            // GM 上 K 方向的 scale 个数

// L1 上标准搬运的 scale 个数
uint64_t scaleKL1StandardLen = MX_SCALE_K_L1_SIZE / MX_GROUPSIZE;  // 4096/32 = 128

// 实际搬运的 scale 个数（处理尾部不足一块的情况）
uint64_t scaleKL1RealSize = (kbL1Offset + MX_SCALE_K_L1_SIZE) > kSizeAligned ?
                                (kSizeAligned - kbL1Offset) / MX_GROUPSIZE :
                                scaleKL1StandardLen;
```

### 5.3 ScaleA 搬运（per-token scale）

```cpp
// 配置 Dn2Nz 参数
Dn2NzParams scaleAdn2NzParams;
ConfigScaleDn2NzParams(param.mL1Size, scaleKGmSize, scaleKL1StandardLen, scaleKL1RealSize, scaleAdn2NzParams);

// 计算 GM 偏移：mOffset * scaleKGmSize 定位到当前 M 分块的起始行
// kbL1Offset / MX_GROUPSIZE 定位到当前 K 分块
int64_t scaleAGmOffset = param.mOffset * scaleKGmSize + kbL1Offset / MX_GROUPSIZE;

// 构造 GM tensor（以 half 视角，因为 Dn2Nz 搬运以 16bit 为单位）
GlobalTensor<half> f16ScaleAGlobal;
f16ScaleAGlobal.SetGlobalBuffer((__gm__ half *)mxScaleAGlobal_[scaleAGmOffset].GetPhyAddr(),
                                (param.mL1Size * scaleKL1RealSize) >> 1);

// 目标 L1 地址（双缓冲选择）
auto f16ScaleALocal = mxScaleAL1_[(mxScaleBufIdx_ & 1) * mxScaleAL1DbOffset_]
                          .template ReinterpretCast<half>();

DataCopy(f16ScaleALocal, f16ScaleAGlobal, scaleAdn2NzParams);
```

### 5.4 ScaleB 搬运（anti-quant scale）

```cpp
// 配置 Dn2Nz 参数（使用 nL1Size 而非 mL1Size）
Dn2NzParams scaleBdn2NzParams;
ConfigScaleDn2NzParams(param.nL1Size, scaleKGmSize, scaleKL1StandardLen, scaleKL1RealSize, scaleBdn2NzParams);

// 计算 GM 偏移：nOffset * scaleKGmSize 定位到当前 N 分块
int64_t scaleBGmOffset = param.nOffset * scaleKGmSize + kbL1Offset / MX_GROUPSIZE;

GlobalTensor<half> f16ScaleBGlobal;
f16ScaleBGlobal.SetGlobalBuffer((__gm__ half *)mxScaleBGlobal_[scaleBGmOffset].GetPhyAddr(),
                                (param.nL1Size * scaleKL1RealSize) >> 1);

auto f16ScaleBLocal = mxScaleBL1_[(mxScaleBufIdx_ & 1) * mxScaleBL1DbOffset_]
                          .template ReinterpretCast<half>();

DataCopy(f16ScaleBLocal, f16ScaleBGlobal, scaleBdn2NzParams);
```

### 5.5 ScaleA 与 ScaleB 的区别

| 维度 | ScaleA (mxScaleA) | ScaleB (mxScaleB) |
|------|-------------------|-------------------|
| 语义 | Per-token scale（A 矩阵每行的 scale） | Anti-quant scale（Weight 反量化 scale） |
| 行数 | `param.mL1Size`（M 方向分块大小） | `param.nL1Size`（N 方向分块大小） |
| GM 偏移 | `param.mOffset * scaleKGmSize` | `param.nOffset * scaleKGmSize` |
| 数据类型 | `fp8_e8m0_t`（搬运时视为 half） | `fp8_e8m0_t`（搬运时视为 half） |

### 5.6 ConfigScaleDn2NzParams 参数详解

```cpp
__aicore__ inline void ConfigScaleDn2NzParams(uint64_t rowNum,        // 矩阵行数（M 或 N 方向）
                                               uint64_t scaleKGmSize,  // GM 上 K 方向 scale 总数
                                               uint64_t scaleKL1Stride,// L1 上标准 scale 个数（128）
                                               uint64_t scaleKL1RealSize,// 实际搬运 scale 个数
                                               Dn2NzParams &dn2NzParams)
{
    dn2NzParams.dnNum = 1;
    dn2NzParams.dValue = rowNum;  // 行数
    // 列数：scale 个数除以 SCALE_COPY_GROUP_SIZE(2)，因为 B16 搬 B8 需要除以 2 向上取整
    dn2NzParams.nValue = CeilDivide(scaleKL1RealSize, SCALE_COPY_GROUP_SIZE);
    dn2NzParams.srcDnMatrixStride = SCALE_COPY_DEFAULT_STRIDE;  // 0
    // 源矩阵一行所含 B16 元素个数
    dn2NzParams.srcDValue = CeilDivide(scaleKGmSize, SCALE_COPY_GROUP_SIZE);
    // 目标矩阵行方向两个相邻分形起始地址间隔（单位 32B）
    dn2NzParams.dstNzC0Stride = CeilDivide(MX_SCALE_K_L1_SIZE,
                                            MX_GROUPSIZE * SCALE_COPY_GROUP_SIZE);
    // 目标矩阵列方向两个相邻分形起始地址间隔（单位 32B）
    dn2NzParams.dstNzNStride = SCALE_COPY_DEFAULT_N_STRIDE;  // 1
    dn2NzParams.dstNzMatrixStride = SCALE_COPY_DEFAULT_STRIDE;  // 0
}
```

Dn2Nz 搬运将 GM 上的 ND 格式数据搬运到 L1 上的 NZ 格式。Scale 数据在 GM 上是 `[rowNum, scaleKGmSize]` 的 ND 矩阵，搬运到 L1 后变为 NZ 分形格式，便于后续 LoadData2DMx 加载到 L0。

---

## 6. CopyAAndBiasGmToL1() 分析

### 6.1 A 矩阵 ND→NZ 转换搬运

```cpp
__aicore__ inline void CopyAAndBiasGmToL1(const BasicBlockOffsetParam &param,
                                           int64_t kaGmOffset, uint64_t cvLoopIdx)
{
    // 只在 kaGmOffset 对齐 kaL1Size 时触发搬运
    if (kaGmOffset % param.kaL1Size != 0) {
        return;
    }

    // 计算实际搬运的 K 方向大小（处理尾部不足一块的情况）
    int64_t kaL1RealSize = (kaGmOffset + param.kaL1Size) >= param.kSize ?
                               param.kSize - kaGmOffset : param.kaL1Size;

    // 配置 Nd2Nz 参数
    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;                    // ND 矩阵数量
    nd2nzParams.nValue = param.mL1Size;       // N 方向大小（即 M 方向，命名历史原因）
    nd2nzParams.dValue = kaL1RealSize;        // D 方向大小（即 K 方向实际大小）
    nd2nzParams.srcDValue = param.kSize;      // 源矩阵 K 方向总大小
    nd2nzParams.srcNdMatrixStride = 0;        // 源 ND 矩阵间 stride
    nd2nzParams.dstNzC0Stride = CeilAlign(nd2nzParams.nValue,
                                           static_cast<uint16_t>(BLOCK_CUBE));  // 目标 NZ C0 stride
    nd2nzParams.dstNzNStride = 1;             // 目标 NZ N stride
    nd2nzParams.dstNzMatrixStride = 0;        // 目标 NZ 矩阵间 stride

    // 执行搬运：GM 上 [kaGmOffset + mOffset*kSize] 位置开始
    DataCopy(aL1_[(aL1BufIdx_ & 1) * aL1DbOffset_],
             xGlobal_[kaGmOffset + param.mOffset * param.kSize],
             nd2nzParams);

    // 如果 K 方向不对齐 64，填充尾部
    PadL1IfNotAlign(aL1_, aL1DbOffset_, aL1BufIdx_, kaL1RealSize, param.mL1Size);
}
```

### 6.2 Nd2NzParams 参数含义

| 参数 | 含义 | 本例取值 |
|------|------|----------|
| `ndNum` | ND 矩阵数量 | 1 |
| `nValue` | N 方向大小（对应 M 维度） | `param.mL1Size` |
| `dValue` | D 方向大小（对应 K 维度实际搬运量） | `kaL1RealSize` |
| `srcDValue` | 源矩阵 D 方向总大小 | `param.kSize` |
| `srcNdMatrixStride` | 源 ND 矩阵间 stride | 0 |
| `dstNzC0Stride` | 目标 NZ 格式 C0 方向 stride | `CeilAlign(mL1Size, BLOCK_CUBE)` |
| `dstNzNStride` | 目标 NZ 格式 N 方向 stride | 1 |
| `dstNzMatrixStride` | 目标 NZ 矩阵间 stride | 0 |

Nd2Nz 搬运将 GM 上 `[mL1Size, kSize]` 的 ND 矩阵中的 `[mL1Size, kaL1RealSize]` 子块搬运到 L1，并转换为 NZ 分形格式。NZ 格式是 Cube 单元所需的矩阵存储格式，将矩阵按 16×16 的分形（fractal）重新排列。

### 6.3 PadL1IfNotAlign() 填充逻辑

```cpp
__aicore__ inline void PadL1IfNotAlign(const LocalTensor<xType> &l1Buf,
                                        uint64_t dbOffset, uint64_t dbIdx,
                                        uint64_t realSize, uint64_t dimSize)
{
    // 当 K 方向实际大小不是 64 的倍数时，需要填充尾部
    if (realSize % K_ALIGNMENT64 != 0) {
        uint64_t alignDimSize = CeilAlign(dimSize, static_cast<uint64_t>(BLOCK_CUBE));
        // 定位到需要填充的起始位置
        LocalTensor<uint32_t> fillTensor =
            l1Buf[(dbIdx & 1) * dbOffset + alignDimSize * realSize]
                .template ReinterpretCast<uint32_t>();
        // 填充 alignDimSize 个 32bit 零值
        FillL1WithZero(fillTensor, alignDimSize);
    }
}
```

当 K 方向实际搬运量不是 64 对齐时（如尾部不足一块），Cube 单元仍需要完整的 64 对齐数据。因此将尾部不足的部分用 0 填充，确保 MMA 计算时不会读到脏数据。

---

## 7. LaunchMatmul() 分析

`LaunchMatmul` 是 MMA 计算的核心调度函数，负责将 L1 上的 A、B、Scale 数据加载到 L0，并启动 MMAD 计算。

### 7.1 函数签名

```cpp
__aicore__ inline void LaunchMatmul(const LocalTensor<xType> &weightL1,  // Weight（B 矩阵）在 L1 的地址
                                     int64_t kbOffset,                    // 当前 K 方向全局偏移
                                     uint64_t kbL1RealSize,               // 当前 K 方向 L1 实际大小
                                     uint64_t cvLoopIdx,                  // CV 循环索引
                                     const BasicBlockOffsetParam &param);
```

### 7.2 流水同步

```cpp
// MTE2→MTE1 同步：确保 Weight 反量化（AIV 侧）已完成
SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID_MTE2_TO_MTE1);
WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID_MTE2_TO_MTE1);
```

### 7.3 L0CopyAndCalcParams 参数准备

```cpp
L0CopyAndCalcParams l0CopyAndCalcParams;
l0CopyAndCalcParams.mL0Size = param.mL1Size;   // M 方向 L0 计算大小 = L1 载入大小
l0CopyAndCalcParams.mL1Size = param.mL1Size;   // M 方向 L1 载入大小
l0CopyAndCalcParams.nL0Size = param.nL1Size;   // N 方向 L0 计算大小 = L1 载入大小
l0CopyAndCalcParams.nL1Size = param.nL1Size;   // N 方向 L1 载入大小
l0CopyAndCalcParams.kL1Size = param.kaL1Size;  // K 方向 L1 载入大小
l0CopyAndCalcParams.scaleKL1Size = MX_SCALE_K_L1_SIZE;  // Scale K 方向 L1 大小 = 4096

// K 方向 L0 计算大小：当 M 和 N 都 ≤128 时用 256，否则用 128
uint64_t kL0Size = (param.mL1Size <= 128 && param.nL1Size <= 128) ? 256 : 128;
```

### 7.4 L0C 偏移管理

```cpp
if (kbOffset == 0) {
    // 每个新的 K 大循环开始时，检查 L0C 剩余空间
    uint64_t nextL0cBufSize = CeilAlign(l0CopyAndCalcParams.mL0Size, BLOCK_CUBE) *
                              CeilAlign(l0CopyAndCalcParams.nL0Size, BLOCK_CUBE);
    if (l0cOffset_ + nextL0cBufSize > L0C_BUF_SIZE) {
        l0cOffset_ = 0;  // 空间不足，回绕到起始位置
    }
}
```

L0C 总容量为 `256×256` 个 float 元素。当多个 M×N 分块的结果需要同时存放在 L0C 中时（流水线需要），通过 `l0cOffset_` 管理写入位置，避免覆盖尚未 Fixpipe 输出的数据。

### 7.5 K 循环中的 L0 加载策略

```cpp
for (uint64_t l1KOffset = 0; l1KOffset < kbL1RealSize; l1KOffset += kL0Size) {
    bool isLastL1K = l1KOffset + kL0Size >= kbL1RealSize;
    l0CopyAndCalcParams.isFirstKLoop = (l1KOffset + kbOffset) == 0;  // 全局首次 K 循环
    l0CopyAndCalcParams.isLastKLoop = l1KOffset + kbOffset + kL0Size >= param.kSize;  // 全局末次 K 循环

    uint64_t realL0k = isLastL1K ? kbL1RealSize - l1KOffset : kL0Size;
    l0CopyAndCalcParams.kL0Size = realL0k;

    uint64_t loopId = l0LoopIdx_ % L0_BUF_NUM;  // L0A/L0B 双缓冲索引

    // 等待上一次 MMA 计算完成（确保 L0A/L0B 可以被覆盖）
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + loopId);

    // 加载 A 矩阵和 ScaleA 到 L0A
    LoadAAndScaleL1ToL0(
        l0a_[loopId * L0_BUF_OFFSET_B8],                                    // L0A 目标地址
        aL1_[aL1Offset + mL1AlignSize * ((l1KOffset + kbOffset) % param.kaL1Size)],  // L1 A 源地址
        mxScaleAL1_[(mxScaleBufIdx_ & 1) * mxScaleAL1DbOffset_ +
                    BLOCK_CUBE * ((l1KOffset + kbOffset) % MX_SCALE_K_L1_SIZE / MX_GROUPSIZE)],  // L1 ScaleA 源地址
        l0CopyAndCalcParams);

    // 加载 B 矩阵（Weight）和 ScaleB 到 L0B
    LoadBAndScaleL1ToL0(
        l0b_[loopId * L0_BUF_OFFSET_B8],                                    // L0B 目标地址
        weightL1[nL1AlignSize * l1KOffset],                                  // L1 B 源地址
        mxScaleBL1_[(mxScaleBufIdx_ & 1) * mxScaleBL1DbOffset_ +
                    BLOCK_CUBE * ((l1KOffset + kbOffset) % MX_SCALE_K_L1_SIZE / MX_GROUPSIZE)],  // L1 ScaleB 源地址
        l0CopyAndCalcParams);

    // 启动 MMAD 计算
    DoMmadCompute(loopId, cvLoopIdx, l0CopyAndCalcParams);
}
```

**K 循环加载策略示意：**

```
L1 上的 A 矩阵: [mL1Size × kaL1Size]
L1 上的 B 矩阵: [nL1Size × kbL1RealSize]

每次 L0 加载:
  A: [mL1Size × kL0Size] 子块 → L0A
  B: [nL1Size × kL0Size] 子块 → L0B
  ScaleA: 对应 K 范围的 scale → L0A (通过 LoadData2DMx)
  ScaleB: 对应 K 范围的 scale → L0B (通过 LoadData2DMx)

L0A/L0B 双缓冲:
  loopId=0 → L0A[0:32KB], L0B[0:32KB]
  loopId=1 → L0A[32KB:64KB], L0B[32KB:64KB]
```

---

## 8. DoMmadCompute() 分析

```cpp
__aicore__ inline void DoMmadCompute(uint64_t loopId, uint64_t cvLoopIdx,
                                      const L0CopyAndCalcParams &l0CopyAndCalcParams)
{
    // 如果有 Bias 且是首次 K 循环，加载 Bias 到 BiasTable
    if constexpr (hasBias) {
        if (l0CopyAndCalcParams.isFirstKLoop) {
            LoadBiasToBt(biasTable_[loopId * BIAS_TABLE_OFFSET_B32],
                         biasL1_[(cvLoopIdx & 1) * biasL1DbOffset_],
                         l0CopyAndCalcParams);
        }
    }

    // MTE1→M 同步：确保 L0A/L0B 加载完成
    SetFlag<HardEvent::MTE1_M>(EVENT_ID_MTE1_TO_M);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID_MTE1_TO_M);

    // 启动 MMAD 计算
    if constexpr (hasBias) {
        if (l0CopyAndCalcParams.isFirstKLoop) {
            // 首次 K 循环：带 Bias 的 MMAD，cmatrixInitVal 由 MmadCompute 内部设为 true
            MmadCompute(l0c_[l0cOffset_],
                        l0a_[loopId * L0_BUF_OFFSET_B8],
                        l0b_[loopId * L0_BUF_OFFSET_B8],
                        biasTable_[loopId * BIAS_TABLE_OFFSET_B32],
                        l0CopyAndCalcParams);
        } else {
            // 非首次 K 循环：不带 Bias 的累加
            MmadCompute(l0c_[l0cOffset_],
                        l0a_[loopId * L0_BUF_OFFSET_B8],
                        l0b_[loopId * L0_BUF_OFFSET_B8],
                        l0CopyAndCalcParams);
        }
    } else {
        // 无 Bias：直接 MMAD
        MmadCompute(l0c_[l0cOffset_],
                    l0a_[loopId * L0_BUF_OFFSET_B8],
                    l0b_[loopId * L0_BUF_OFFSET_B8],
                    l0CopyAndCalcParams);
    }

    // 通知 MTE1：MMA 计算完成，L0A/L0B 可以被下一轮覆盖
    SetFlag<HardEvent::M_MTE1>(EVENT_ID_M_TO_MTE1 + loopId);
    l0LoopIdx_++;
}
```

**Bias 加载时机：** Bias 只在首次 K 循环（`isFirstKLoop == true`）时加载。此时 MMAD 使用带 Bias 的重载版本，将 Bias 作为初始值写入 L0C。后续 K 循环使用不带 Bias 的版本，在 L0C 上累加。

---

## 9. GetTensorC() 分析

```cpp
// 版本 1：输出到指定 UB tensor
__aicore__ inline void GetTensorC(LocalTensor<yType> &yUb, const BasicBlockOffsetParam &param)
{
    FixL0CToDstParams fixL0CToDstParams;
    fixL0CToDstParams.mL0Size = param.mL1Size;   // M 方向实际计算大小
    fixL0CToDstParams.nL0Size = param.nL1Size;   // N 方向实际计算大小
    fixL0CToDstParams.outNSize = 256;            // 输出 N 方向总大小（固定 256）
    fixL0CToDstParams.splitNSize = 0;            // 不切分
    fixL0CToDstParams.dstNdStride = 0;

    // Fixpipe: L0C → UB（ROW_MAJOR 格式）
    FixL0CToDst(yUb, l0c_[l0cOffset_], fixL0CToDstParams);

    // 更新 l0cOffset_：为下一个 M×N 分块预留空间
    l0cOffset_ += CeilAlign(fixL0CToDstParams.mL0Size, BLOCK_CUBE) *
                  CeilAlign(fixL0CToDstParams.nL0Size, BLOCK_CUBE);
}
```

**FixL0CToDstParams 参数：**

| 参数 | 含义 | 取值 |
|------|------|------|
| `mL0Size` | M 方向实际计算大小 | `param.mL1Size` |
| `nL0Size` | N 方向实际计算大小 | `param.nL1Size` |
| `outNSize` | 输出 N 方向总大小（用于 dstStride） | 256 |
| `splitNSize` | N 方向切分大小（0 = 不切分） | 0 |
| `dstNdStride` | 多矩阵输出时相邻 DN 矩阵偏移 | 0 |

**l0cOffset_ 管理：** 每次 Fixpipe 输出后，`l0cOffset_` 递增当前分块的对齐大小。当新一轮 K 循环开始时（`kbOffset == 0`），如果剩余空间不足，`l0cOffset_` 回绕到 0。这允许 L0C 中同时存放多个 M×N 分块的结果，支持流水线操作。

---

## 10. Cube Compute Tools 分析

### 10.1 LoadAAndScaleL1ToL0

```cpp
template <typename DstType, typename SrcType, typename XScaleType>
__aicore__ inline void LoadAAndScaleL1ToL0(
    const LocalTensor<DstType> &aL0Tensor,       // L0A 目标
    const LocalTensor<SrcType> &aL1Tensor,       // L1 A 矩阵源
    const LocalTensor<XScaleType> &scaleL1Tensor,// L1 ScaleA 源
    const L0CopyAndCalcParams &l0CopyAndCalcParams)
{
    uint64_t mL0AlignSize = CeilAlign(l0CopyAndCalcParams.mL0Size, BLOCK_CUBE);
    uint64_t mL1AlignSize = CeilAlign(l0CopyAndCalcParams.mL1Size, BLOCK_CUBE);

    // A 矩阵的 2D 加载参数
    AscendC::LoadData2DParamsV2 aL0Load2dParams;
    aL0Load2dParams.mStartPosition = 0;
    aL0Load2dParams.kStartPosition = 0;
    aL0Load2dParams.mStep = mL0AlignSize / BLOCK_CUBE;  // M 方向步长（以分形为单位）
    aL0Load2dParams.kStep = CeilDiv(CeilAlign(kL0Size, K_ALIGNMENT64), GetC0Size<SrcType>());
    aL0Load2dParams.srcStride = CeilDiv(mL1AlignSize * ONE_BLK_SIZE, L0AB_OPERATE_UNIT);
    aL0Load2dParams.dstStride = CeilDiv(mL0AlignSize * ONE_BLK_SIZE, L0AB_OPERATE_UNIT);

    // ScaleA 的 MX 加载参数
    AscendC::LoadData2DMxParams aL0Load2dMx;
    aL0Load2dMx.xStartPosition = 0;
    aL0Load2dMx.yStartPosition = 0;
    aL0Load2dMx.xStep = mL0AlignSize / BLOCK_CUBE;  // M 方向步长
    // yStep: K 方向跨过的 32B 分形个数
    // kL0Size / 64: 因为每 64 个 K 元素对应 2 个 e8m0 scale（32B 一个分形）
    aL0Load2dMx.yStep = CeilDiv(CeilAlign(kL0Size, K_ALIGNMENT64), 64UL);
    aL0Load2dMx.srcStride = CeilDiv(scaleKL1Size, 64UL);
    aL0Load2dMx.dstStride = aL0Load2dMx.yStep;

    // 联合加载：A 矩阵数据 + Scale 数据 → L0A
    AscendC::LoadData(aL0Tensor, aL1Tensor, scaleL1Tensor, aL0Load2dParams, aL0Load2dMx);
}
```

**LoadData2DMx 机制：** `LoadData` 的三参数版本同时加载矩阵数据和 MX Scale 数据到 L0。硬件在加载 A 矩阵数据的同时，将对应的 Scale 数据也加载到 L0A 的 Scale 区域，供 MMAD 指令直接使用。

### 10.2 LoadBAndScaleL1ToL0

```cpp
template <typename DstType, typename SrcType, typename WScaleType>
__aicore__ inline void LoadBAndScaleL1ToL0(
    const LocalTensor<DstType> &bL0Tensor,
    const LocalTensor<SrcType> &bL1Tensor,
    const LocalTensor<WScaleType> &scaleL1Tensor,
    const L0CopyAndCalcParams &l0CopyAndCalcParams)
{
    uint64_t nL0AlignSize = CeilAlign(l0CopyAndCalcParams.nL0Size, BLOCK_CUBE);
    uint64_t nL1AlignSize = CeilAlign(l0CopyAndCalcParams.nL1Size, BLOCK_CUBE);

    // B 矩阵的 2D 加载参数（与 A 对称，使用 N 方向替代 M 方向）
    AscendC::LoadData2DParamsV2 bL0Load2dParams;
    bL0Load2dParams.mStartPosition = 0;
    bL0Load2dParams.kStartPosition = 0;
    bL0Load2dParams.mStep = nL0AlignSize / BLOCK_CUBE;
    bL0Load2dParams.kStep = CeilDiv(CeilAlign(kL0Size, K_ALIGNMENT64), GetC0Size<SrcType>());
    bL0Load2dParams.srcStride = CeilDiv(nL1AlignSize * ONE_BLK_SIZE, L0AB_OPERATE_UNIT);
    bL0Load2dParams.dstStride = CeilDiv(nL0AlignSize * ONE_BLK_SIZE, L0AB_OPERATE_UNIT);

    // ScaleB 的 MX 加载参数
    AscendC::LoadData2DMxParams bL0Load2dMx;
    bL0Load2dMx.xStartPosition = 0;
    bL0Load2dMx.yStartPosition = 0;
    bL0Load2dMx.xStep = nL0AlignSize / BLOCK_CUBE;
    bL0Load2dMx.yStep = CeilDiv(CeilAlign(kL0Size, K_ALIGNMENT64), 64UL);
    bL0Load2dMx.srcStride = CeilDiv(scaleKL1Size, 64UL);
    bL0Load2dMx.dstStride = bL0Load2dMx.yStep;

    AscendC::LoadData(bL0Tensor, bL1Tensor, scaleL1Tensor, bL0Load2dParams, bL0Load2dMx);
}
```

**LoadA 与 LoadB 的差异：**

| 维度 | LoadA | LoadB |
|------|-------|-------|
| 行方向 | M（`mL0Size`, `mL1Size`） | N（`nL0Size`, `nL1Size`） |
| 列方向 | K（`kL0Size`） | K（`kL0Size`） |
| Scale 语义 | Per-token scale（沿 M 方向） | Anti-quant scale（沿 N 方向） |

### 10.3 MmadCompute（无 Bias）

```cpp
template <typename DstType, typename SrcAType, typename SrcBType>
__aicore__ inline void MmadCompute(const LocalTensor<DstType> &cL0Tensor,
                                    const LocalTensor<SrcAType> &aL0Tensor,
                                    const LocalTensor<SrcBType> &bL0Tensor,
                                    const L0CopyAndCalcParams &l0CopyAndCalcParams)
{
    AscendC::MmadParams mmadParams;
    mmadParams.m = CeilAlign(l0CopyAndCalcParams.mL0Size, 2UL);  // M 方向 2 对齐
    mmadParams.n = l0CopyAndCalcParams.nL0Size;                   // N 方向
    mmadParams.k = CeilAlign(l0CopyAndCalcParams.kL0Size, K_ALIGNMENT64);  // K 方向 64 对齐
    mmadParams.cmatrixInitVal = l0CopyAndCalcParams.isFirstKLoop;  // 首次 K 循环初始化 L0C
    mmadParams.disableGemv = true;                                 // 禁用 GEMV 模式
    mmadParams.unitFlag = l0CopyAndCalcParams.isLastKLoop ?
                              UNIT_FALG_UPDATE : UNIT_FALG_CHECK_ONLY;  // 末次 K 循环更新标记

    AscendC::Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
}
```

### 10.4 MmadCompute（带 Bias）

```cpp
template <typename DstType, typename SrcAType, typename SrcBType, typename BiasType>
__aicore__ inline void MmadCompute(const LocalTensor<DstType> &cL0Tensor,
                                    const LocalTensor<SrcAType> &aL0Tensor,
                                    const LocalTensor<SrcBType> &bL0Tensor,
                                    const LocalTensor<BiasType> &biasTable,
                                    const L0CopyAndCalcParams &l0CopyAndCalcParams)
{
    AscendC::MmadParams mmadParams;
    mmadParams.m = CeilAlign(l0CopyAndCalcParams.mL0Size, 2UL);
    mmadParams.n = l0CopyAndCalcParams.nL0Size;
    mmadParams.k = CeilAlign(l0CopyAndCalcParams.kL0Size, K_ALIGNMENT64);
    mmadParams.disableGemv = true;
    mmadParams.cmatrixInitVal = false;  // 带 Bias 时由 Bias 初始化，不自动清零
    mmadParams.unitFlag = l0CopyAndCalcParams.isLastKLoop ?
                              UNIT_FALG_UPDATE : UNIT_FALG_CHECK_ONLY;

    AscendC::Mmad(cL0Tensor, aL0Tensor, bL0Tensor, biasTable, mmadParams);
}
```

**unitFlag 的含义：**

| 值 | 常量 | 含义 |
|----|------|------|
| 2 | `UNIT_FALG_CHECK_ONLY` | 只检查标记位，不更新（非末次 K 循环） |
| 3 | `UNIT_FALG_UPDATE` | 检查并更新标记位（末次 K 循环，通知下游可以消费 L0C） |

### 10.5 FixL0CToDst

```cpp
template <typename DstType, typename SrcType, bool outSplitN = false>
__aicore__ inline void FixL0CToDst(const LocalTensor<DstType> &outTensor,
                                    const LocalTensor<SrcType> &cL0Tensor,
                                    const FixL0CToDstParams &fixL0CToDstParams)
{
    AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixParams;
    fixParams.params.ndNum = 1;
    fixParams.mSize = CeilAlign(fixL0CToDstParams.mL0Size, 2UL);
    fixParams.nSize = CeilAlign(fixL0CToDstParams.nL0Size, BLOCK_CUBE);
    fixParams.srcStride = CeilAlign(fixL0CToDstParams.mL0Size, BLOCK_CUBE);
    fixParams.dstStride = FIXP_DST_STRIDE;  // 256
    fixParams.unitFlag = UNIT_FALG_UPDATE;
    fixParams.quantPre = QuantMode_t::NoQuant;  // 不做量化
    fixParams.dualDstCtl = 1;                   // 双目标控制

    AscendC::Fixpipe<DstType, SrcType, CFG_ROW_MAJOR_UB>(
        outTensor, cL0Tensor, fixParams);
}
```

Fixpipe 将 L0C 中的数据以 ROW_MAJOR 格式输出到 UB，不做量化转换。`dualDstCtl = 1` 表示启用双目标控制，支持将 L0C 切分为两路输出到 UB 的不同位置。

### 10.6 LoadBiasToBt

```cpp
template <typename DstType, typename SrcType>
__aicore__ inline void LoadBiasToBt(const LocalTensor<DstType> &biasTable,
                                     const LocalTensor<SrcType> &biasL1,
                                     const L0CopyAndCalcParams &l0CopyAndCalcParams)
{
    AscendC::DataCopy(biasTable, biasL1,
        {1, static_cast<uint16_t>(CeilDivide(l0CopyAndCalcParams.nL0Size * sizeof(float), 64UL)), 0, 0});
}
```

将 Bias 从 L1 加载到 L0 的 BiasTable 区域。搬运量为 `nL0Size * sizeof(float)` 字节，以 64B 为单位。

---

## 11. 完整 MMA 计算流水线

### 11.1 流水线阶段

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AIC Cube 计算流水线                           │
├──────────┬──────────┬──────────┬──────────┬──────────┬─────────────┤
│  MTE1    │  MTE1    │  MTE1    │   MMA    │  Fixpipe │  下一阶段    │
│ Scale    │ A 矩阵   │ MTE2→M   │  计算    │  L0C→UB  │  (AIV VF)   │
│ GM→L1    │ GM→L1    │  同步    │          │          │             │
├──────────┼──────────┼──────────┼──────────┼──────────┼─────────────┤
│CopyMxScale│CopyAAnd │SetFlag/  │DoMmad    │GetTensorC│ vecCompute  │
│GmToL1    │BiasGmToL1│WaitFlag  │Compute   │          │ 消费 UB     │
└──────────┴──────────┴──────────┴──────────┴──────────┴─────────────┘
```

### 11.2 同步事件流

```
AIC 侧 (IterateNzNkWithKAic):
  for each kbL1Offset:
    1. WaitScaleMTE1ToMTE2(kbL1Offset)     ← 等待上一轮 Scale 搬运完成
    2. CopyMxScaleGmToL1(param, kbL1Offset) ← 搬运 Scale GM→L1
    3. WaitMTE1ToMTE2(kbL1Offset, param)    ← 等待上一轮 A 矩阵搬运完成
    4. CopyAAndBiasGmToL1(...)              ← 搬运 A 矩阵 GM→L1
    5. WaitAivToAic<PIPE_MTE1>(...)         ← 等待 AIV 侧 Weight 反量化完成
    6. LaunchMatmul(...)                     ← 启动 MMA 计算
       ├── SetFlag<MTE2_MTE1> / WaitFlag<MTE2_MTE1>
       ├── for each l1KOffset:
       │     ├── WaitFlag<M_MTE1>           ← 等待上一轮 MMA 完成
       │     ├── LoadAAndScaleL1ToL0        ← L1→L0A
       │     ├── LoadBAndScaleL1ToL0        ← L1→L0B
       │     ├── SetFlag<MTE1_M> / WaitFlag<MTE1_M>
       │     ├── MmadCompute                ← MMA 计算
       │     └── SetFlag<M_MTE1>            ← 通知 MMA 完成
       └── end for
    7. SetMTE1ToMTE2(kbL1Offset, param)     ← 通知 A 矩阵搬运完成
    8. SetScaleMTE1ToMTE2(kbL1Offset, param)← 通知 Scale 搬运完成
    9. SetAicToAiv<PIPE_MTE1>(...)          ← 通知 AIV 可以继续搬运
  end for

  (跨 CV 循环)
  WaitAivToAic<PIPE_FIX>(...)               ← 等待 AIV 消费完上一轮 UB
  GetTensorC(ubOutputF32Buffer_, lastOffsetParam) ← Fixpipe L0C→UB
  SetAicToAiv<PIPE_FIX>(...)                ← 通知 AIV UB 已就绪
```

### 11.3 双缓冲机制总结

| 层级 | 缓冲区 | 双缓冲数量 | 切换索引 |
|------|--------|-----------|----------|
| L1 | WeightL1 | 2 | `cvLoopIdx_ & 1` |
| L1 | BiasL1 | 2 | `cvLoopIdx_ & 1` |
| L1 | ScaleAL1 | 2 | `mxScaleBufIdx_ & 1` |
| L1 | ScaleBL1 | 2 | `mxScaleBufIdx_ & 1` |
| L1 | AL1 | 2 | `aL1BufIdx_ & 1` |
| L0 | L0A | 2 | `l0LoopIdx_ % 2` |
| L0 | L0B | 2 | `l0LoopIdx_ % 2` |
| L0 | BiasTable | 2 | `loopId` |

### 11.4 关键常量速查

| 常量 | 值 | 含义 |
|------|-----|------|
| `MX_GROUPSIZE` | 32 | MX Scale 分组大小 |
| `MX_SCALE_K_L1_SIZE` | 4096 | Scale L1 覆盖的 K 方向元素数 |
| `MX_SCALE_L1_SIZE` | 32KB | 单块 Scale L1 空间 |
| `K_ALIGNMENT64` | 64 | K 方向对齐粒度 |
| `BLOCK_CUBE` | 16 | Cube 分形大小 |
| `L0AB_OPERATE_UNIT` | 512B | L0 搬运单位 |
| `L0_BUF_OFFSET_B8` | 32KB | L0A/L0B 双缓冲偏移 |
| `BIAS_TABLE_OFFSET_B32` | 512 (2KB) | BiasTable 双缓冲偏移 |
| `L0C_BUF_SIZE` | 256×256 | L0C 总容量 |
| `FIXP_DST_STRIDE` | 256 | Fixpipe 输出 N 方向 stride |