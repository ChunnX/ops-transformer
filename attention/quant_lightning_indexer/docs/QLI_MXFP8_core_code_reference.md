# QLI MXFP8 核心代码参考

> 本文档整理了 QLI（Quant Lightning Indexer）算子中 MXFP8 量化的全部核心代码，按数据流顺序组织，附带本地代码跳转链接。

---

## 目录

- [一、架构总览](#一架构总览)
- [二、算子注册 — def.cpp](#二算子注册--defcpp)
- [三、Tiling 校验 — tiling.cpp](#三tiling-校验--tilingcpp)
- [四、类型体系 — common.h](#四类型体系--commonh)
- [五、Kernel 分发 — quant_lightning_indexer.cpp](#五kernel-分发--quant_lightning_indexercpp)
- [六、核间调度 — kernel.h](#六核间调度--kernelh)
- [七、Cube 核矩阵乘 — service_cube.h](#七cube-核矩阵乘--service_cubeh)
- [八、Vector 核后处理 — service_vector.h](#八vector-核后处理--service_vectorh)
- [九、VF 底层函数 — vector1.h](#九vf-底层函数--vector1h)
- [十、代码位置速查表](#十代码位置速查表)

---

## 一、架构总览

### 1.1 MXFP8 核心改动：Scale 从 Vector 核迁移到 Cube 核

```
┌─────────────────────────────────────────────────────────────────────┐
│                        非 MXFP8 流程                                │
│                                                                     │
│  Cube: Q × K → int32/float                                         │
│  Vector: weights × qScale → QK × result → accum × kScale → score   │
│  (scale 在 Vector 核 UB 中逐元素乘)                                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        MXFP8 流程                                   │
│                                                                     │
│  Cube: Q × K × qScale × kScale → float  (mmad_mx 硬件融合)          │
│  Vector: weights × QK → Relu → ReduceSum → score                   │
│  (scale 已在 Cube 核由 mmad_mx 融合，Vector 核无需处理 scale)         │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 数据类型

| 角色 | 类型 | 说明 |
|------|------|------|
| Query/Key 数据 | `fp8_e4m3fn_t` | 8位浮点，4位指数+3位尾数 |
| Scale | `fp8_e8m0_t` | 8位指数，无尾数，表示2的幂次 |
| Weight | `bfloat16_t` | 不量化，保持 bf16 |
| QK 矩阵乘输出 | `float` | mmad_mx 输出 fp32 |
| 量化粒度 | 每32个D维度元素一个scale | `MXFP8_BLOCK_SIZE = 32` |

### 1.3 数据流全链路

```
                     HOST SIDE                          DEVICE SIDE (ascend950)
                     =========                          =======================

def.cpp:                                               kernel.cpp:
  ascend950 config position 4:                           ORIG_DTYPE_QUERY_DEQUANT_SCALE == DT_FLOAT8_E8M0
  query/key = DT_FLOAT8_E4M3FN                            → QLIType<fp8_e4m3fn_t, fp8_e4m3fn_t, ...,
  weights = DT_BF16                                          bfloat16_t, fp8_e8m0_t, float32_t>
  qScale/kScale = DT_FLOAT8_E8M0  ← KEY                    → isMxFp8 = true

tiling.cpp:                                            kernel.h (QLIPreload):
  isMxFp8Scale = (scaleType == DT_FLOAT8_E8M0)           AIC: qScaleGm/kScaleGm → matmulService
  qScale shape: [B,S1,N1,D/32]                           AIV: NO qScale/kScale buffers
  kScale shape: [B,S2,N2,D/32]
  kScale stride: blockSize * (headDim/32)                service_cube.h (QLIMatmul):
                                                           L1: LoadQScaleToL1 / LoadKScaleToL1
                                                               (fp8_e8m0 packed as bf16, Dn2Nz copy)
                                                           L0: LoadData Mx variant
                                                               (mx_fp8_e4m3_t + fp8_e8m0_t scale)
                                                           Mmad: mx_fp8_e4m3_t → mmad_mx
                                                               → Q × K × qScale × kScale = float

                                                         service_vector.h (QLIVector):
                                                           NO qScale/kScale UB buffers
                                                           Calls BatchMulWeightAndReduceSum_MxFP8

                                                         vf/vector1.h (MxFP8 variants):
                                                           weights only (no qScale pre-multiply)
                                                           Relu + MulAddDst (no kScale post-multiply)
                                                           → bf16 → sortable uint16 keys
```

---

## 二、算子注册 — def.cpp

> **文件路径**: [`op_host/quant_lightning_indexer_def.cpp`](../op_host/quant_lightning_indexer_def.cpp)

### 2.1 ascend950 类型位置说明

MXFP8 是 ascend950 配置的 **第5个类型位置（index 4）**。

```cpp
// ascend950 类型位置说明:
// 位置0: fp8_e4m3 + bf16 + float32 scale
// 位置1: hifloat8 + bf16 + float32 scale
// 位置2: fp8_e4m3 + float16 + float16 scale
// 位置3: int8 + float16 + float16 scale
// 位置4: fp8_e4m3 + bf16 + fp8_e8m0 scale (MXFP8)
```
> [def.cpp L84-89](../op_host/quant_lightning_indexer_def.cpp#L84)

### 2.2 MXFP8 类型注册

```cpp
aicore_config_95.Input("query")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN,
               ge::DT_INT8, ge::DT_FLOAT8_E4M3FN})       // ← 位置4: MXFP8
    ...
aicore_config_95.Input("weights")
    .ParamType(REQUIRED)
    .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16,
               ge::DT_FLOAT16, ge::DT_BF16})              // ← 位置4: bf16
    ...
aicore_config_95.Input("query_dequant_scale")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16,
               ge::DT_FLOAT16, ge::DT_FLOAT8_E8M0})       // ← 位置4: fp8_e8m0 ★ KEY
    ...
aicore_config_95.Input("key_dequant_scale")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16,
               ge::DT_FLOAT16, ge::DT_FLOAT8_E8M0})       // ← 位置4: fp8_e8m0 ★ KEY
```
> [def.cpp L90-114](../op_host/quant_lightning_indexer_def.cpp#L90)

### 2.3 硬件配置注册

```cpp
this->AICore().AddConfig("ascend950", aicore_config_95);
this->AICore().AddConfig("ascend910_96", aicore_config_95);
```
> [def.cpp L140-141](../op_host/quant_lightning_indexer_def.cpp#L140)

---

## 三、Tiling 校验 — tiling.cpp

> **文件路径**: [`op_host/quant_lightning_indexer_tiling.cpp`](../op_host/quant_lightning_indexer_tiling.cpp)

### 3.1 数据类型校验

MXFP8 类型组合校验：`(bf16, fp8_e8m0, fp8_e8m0)` 作为合法组合。

```cpp
if (inputQType_ == ge::DT_FLOAT8_E4M3FN) {
    OP_CHECK_IF((weightsType_ != ge::DT_BF16 || inputQueryScaleType_ != ge::DT_FLOAT) && 
                (weightsType_ != ge::DT_FLOAT16|| inputQueryScaleType_ != ge::DT_FLOAT16) &&
                (weightsType_ != ge::DT_BF16 || inputQueryScaleType_ != ge::DT_FLOAT8_E8M0),
        OP_LOGE(opName_, "When input query and key are float8_e4m3, "
                "the data type of the input weights, query_dequant_scale and key_dequant_scale "
                "must be (bfloat16, float32, float32), (float16, float16, float16) or "
                "(bfloat16, float8_e8m0, float8_e8m0)."),
        return ge::GRAPH_FAILED);
}
```
> [tiling.cpp L286-294](../op_host/quant_lightning_indexer_tiling.cpp#L286)

### 3.2 MXFP8 Scale Shape 校验

MXFP8 scale 与 data **维度数相同**，最后一维为 `D/32`。

```cpp
bool isMxFp8Scale = (inputQueryScaleType_ == ge::DT_FLOAT8_E8M0);

if (isMxFp8Scale) {
    // MXFP8: qScale shape [B, S1, N1, D/32]，与 query 维度数相同
    OP_CHECK_IF(qDequantScaleShapeDim != qShapeDim, ...);
    OP_CHECK_IF(kDequantScaleShapeDim != kShapeDim, ...);
    // 检查前 N-1 维相同
    for (uint32_t i = 0; i < (qShapeDim - 1); i++) {
        OP_CHECK_IF(dimValueQueryScale != dimValueQuery, ...);
    }
    // 检查最后一维 = headDim / 32
    uint32_t qScaleLastDim = opParamInfo_.query_dequant_scale.shape->GetStorageShape().GetDim(qShapeDim - 1);
    uint32_t expectedQLastDim = headDim_ / 32;
    OP_CHECK_IF(qScaleLastDim != expectedQLastDim, ...);
    // kScale 同理...
} else {
    // 非 MXFP8: scale 维度数比 data 少 1
    OP_CHECK_IF(qDequantScaleShapeDim != (qShapeDim - 1), ...);
}
```
> [tiling.cpp L740-794](../op_host/quant_lightning_indexer_tiling.cpp#L740)

### 3.3 MXFP8 kScale Stride 计算

```cpp
// MXFP8: kScale shape [BlockNum, BlockSize, N2, D/32]，stride 需要乘以 D/32
if (inputKeyScaleType_ == ge::DT_FLOAT8_E8M0) {
    QLIInfo.keyDequantScaleStride0 = blockSize_ * (headDim_ / 32);
} else {
    QLIInfo.keyDequantScaleStride0 = blockSize_;
}
```
> [tiling.cpp L853-858](../op_host/quant_lightning_indexer_tiling.cpp#L853)

---

## 四、类型体系 — common.h

> **文件路径**: [`op_kernel/quant_lightning_indexer_common.h`](../op_kernel/quant_lightning_indexer_common.h)

### 4.1 MXFP8 基础类型定义

```cpp
// ===================== MXFP8 类型定义 =====================
using FP8E4M3 = fp8_e4m3fn_t;
using FP8E8M0 = fp8_e8m0_t;

constexpr uint32_t MXFP8_BLOCK_SIZE = 32;  // 每32个D维度元素一个scale
constexpr uint32_t FP8_TWO = 2;            // 2个fp8_e8m0打包成1个bf16
```
> [common.h L19-24](../op_kernel/quant_lightning_indexer_common.h#L19)

### 4.2 QLIType 默认模板（非 MXFP8）

```cpp
template <typename Q_T, typename K_T, typename OUT_T, const bool PAGE_ATTENTION = false,
          LI_LAYOUT Q_LAYOUT_T = LI_LAYOUT::BSND, LI_LAYOUT K_LAYOUT_T = LI_LAYOUT::PA_BSND, 
          typename W_T = half, typename SCALE_T = half, typename QK_T = int32_t, typename... Args>
struct QLIType {
    using queryType = Q_T;
    using keyType = K_T;
    using qkType = QK_T;
    using outputType = OUT_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr LI_LAYOUT layout = Q_LAYOUT_T;
    static constexpr LI_LAYOUT keyLayout = K_LAYOUT_T;
    using weightType = W_T;
    using scaleType = SCALE_T;
    static constexpr bool isMxFp8 = false;    // ← 默认非 MXFP8
};
```
> [common.h L40-54](../op_kernel/quant_lightning_indexer_common.h#L40)

### 4.3 QLIType MXFP8 特化 ★

当模板参数匹配 `<FP8E4M3, FP8E4M3, ..., FP8E8M0, float>` 时，`isMxFp8 = true`。

```cpp
// MXFP8 特化：Q/K为fp8_e4m3，scale为fp8_e8m0，mmad_mx输出float
template <typename OUT_T, const bool PAGE_ATTENTION, LI_LAYOUT Q_LAYOUT_T, LI_LAYOUT K_LAYOUT_T,
          typename W_T, typename... Args>
struct QLIType<FP8E4M3, FP8E4M3, OUT_T, PAGE_ATTENTION, Q_LAYOUT_T, K_LAYOUT_T,
               W_T, FP8E8M0, float, Args...> {
    using queryType = FP8E4M3;
    using keyType = FP8E4M3;
    using qkType = float;          // ← mmad_mx 输出 float
    using outputType = OUT_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr LI_LAYOUT layout = Q_LAYOUT_T;
    static constexpr LI_LAYOUT keyLayout = K_LAYOUT_T;
    using weightType = W_T;
    using scaleType = FP8E8M0;     // ← fp8_e8m0
    static constexpr bool isMxFp8 = true;  // ← MXFP8 标志
};
```
> [common.h L56-71](../op_kernel/quant_lightning_indexer_common.h#L56)

### 4.4 RunInfo 中的 MXFP8 字段

```cpp
struct RunInfo {
    ...
    uint64_t tensorQScaleOffset;   // MXFP8: qScale在Cube核使用的偏移
    ...
};
```
> [common.h L95](../op_kernel/quant_lightning_indexer_common.h#L95)

---

## 五、Kernel 分发 — quant_lightning_indexer.cpp

> **文件路径**: [`op_kernel/quant_lightning_indexer.cpp`](../op_kernel/quant_lightning_indexer.cpp)

### 5.1 MXFP8 运行时分发

通过编译期宏 `ORIG_DTYPE_QUERY_DEQUANT_SCALE` 判断是否 MXFP8，实例化对应模板。

```cpp
if (ORIG_DTYPE_QUERY == DT_FLOAT8_E4M3FN) {
#ifdef ORIG_DTYPE_QUERY_DEQUANT_SCALE
    if (ORIG_DTYPE_QUERY_DEQUANT_SCALE == DT_FLOAT8_E8M0) {
        // MXFP8: q/k=fp8_e4m3, scale=fp8_e8m0, mmad_mx 输出 float
        INVOKE_LI_NO_KFC_OP_IMPL(QLIPreload, fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t,
            PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T),
            bfloat16_t, fp8_e8m0_t, float32_t);
            // ↑W_T    ↑SCALE_T  ↑QK_T
    } else
#endif
    if (ORIG_DTYPE_WEIGHTS == DT_FLOAT16) {
        INVOKE_LI_NO_KFC_OP_IMPL(QLIPreload, fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t,
            PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T),
            half, half, float32_t);
    } else {
        INVOKE_LI_NO_KFC_OP_IMPL(QLIPreload, fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t,
            PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T),
            bfloat16_t, float32_t, float32_t);
    }
}
```
> [quant_lightning_indexer.cpp L51-68](../op_kernel/quant_lightning_indexer.cpp#L51)

**模板参数对应关系**：

| 模板参数位 | MXFP8 值 | 含义 |
|-----------|---------|------|
| Q_T | `fp8_e4m3fn_t` | Query 数据类型 |
| K_T | `fp8_e4m3fn_t` | Key 数据类型 |
| OUT_T | `int32_t` | 输出索引类型 |
| W_T | `bfloat16_t` | Weight 数据类型 |
| SCALE_T | `fp8_e8m0_t` | Scale 数据类型 → 触发 MXFP8 特化 |
| QK_T | `float32_t` | QK 矩阵乘输出类型 → 触发 MXFP8 特化 |

---

## 六、核间调度 — kernel.h

> **文件路径**: [`op_kernel/arch35/quant_lightning_indexer_kernel.h`](../op_kernel/arch35/quant_lightning_indexer_kernel.h)

### 6.1 IS_MXFP8 编译期标志

```cpp
static constexpr bool IS_MXFP8 = QLIT::isMxFp8;
```
> [kernel.h L74](../op_kernel/arch35/quant_lightning_indexer_kernel.h#L74)

### 6.2 AIV/AIC 初始化分支

**AIV (Vector核)**：MXFP8 模式下**不加载** qScale/kScale。

```cpp
if ASCEND_IS_AIV {
    vectorService.InitParams(constInfo, tiling);
    indiceOutGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
    weightsGm.SetGlobalBuffer((__gm__ W_T *)weights);
    if constexpr (!IS_MXFP8) {
        qScaleGm.SetGlobalBuffer((__gm__ SCALE_T *)queryScale);
        kScaleGm.SetGlobalBuffer((__gm__ SCALE_T *)keyScale);
        if (constInfo.setL2DisableFlag) {
            kScaleGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
    }
    ...
}
```
> [kernel.h L426-439](../op_kernel/arch35/quant_lightning_indexer_kernel.h#L426)

**AIC (Cube核)**：MXFP8 模式下**接管** qScale/kScale 并传给 matmulService。

```cpp
} else {
    matmulService.InitParams(constInfo);
    queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    ...
    keyGm.SetGlobalBuffer((__gm__ K_T *)key);
    ...
    if constexpr (IS_MXFP8) {
        qScaleGm.SetGlobalBuffer((__gm__ SCALE_T *)queryScale);
        kScaleGm.SetGlobalBuffer((__gm__ SCALE_T *)keyScale);
        matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm, qScaleGm, kScaleGm);
    } else {
        matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm);
    }
}
```
> [kernel.h L440-457](../op_kernel/arch35/quant_lightning_indexer_kernel.h#L440)

### 6.3 MXFP8 qScale/kScale 偏移计算

qScale shape `[B, S1, N1, D/32]`，kScale shape `[B, S2, N2, D/32]`。

```cpp
// MXFP8: qScale offset, shape [B, S1, N1, D/32]
if constexpr (IS_MXFP8) {
    uint64_t qScalePrefixSum = actualSeqQPrefixSum * constInfo.qHeadNum
                               * (constInfo.headDim / MXFP8_BLOCK_SIZE);
    qScaleCoreOffset_ = qScalePrefixSum
        + runInfo.gS1Idx * constInfo.mBaseSize * (constInfo.headDim / MXFP8_BLOCK_SIZE);
}
```
> [kernel.h L568-572](../op_kernel/arch35/quant_lightning_indexer_kernel.h#L568)

```cpp
if constexpr (IS_MXFP8) {
    // MXFP8: kScale shape [B, S2, N2, D/32]
    keyScaleCoreOffset = (actualSeqKPrefixSum + runInfo.s2Idx * constInfo.s2BaseSize) *
                         constInfo.kHeadNum * (constInfo.headDim / MXFP8_BLOCK_SIZE);
} else {
    keyScaleCoreOffset = (actualSeqKPrefixSum + runInfo.s2Idx * constInfo.s2BaseSize)
                         * constInfo.kHeadNum;
}
```
> [kernel.h L582-588](../op_kernel/arch35/quant_lightning_indexer_kernel.h#L582)

```cpp
runInfo.tensorQScaleOffset = qScaleCoreOffset_;
```
> [kernel.h L592](../op_kernel/arch35/quant_lightning_indexer_kernel.h#L592)

---

## 七、Cube 核矩阵乘 — service_cube.h

> **文件路径**: [`op_kernel/arch35/quant_lightning_indexer_service_cube.h`](../op_kernel/arch35/quant_lightning_indexer_service_cube.h)

### 7.1 类型定义与常量

```cpp
using SCALE_T = typename QLIT::scaleType;
static constexpr bool IS_MXFP8 = QLIT::isMxFp8;

static constexpr uint64_t FP8_BLOCK_CUBE = 32;
```
> [service_cube.h L33-34, L67](../op_kernel/arch35/quant_lightning_indexer_service_cube.h#L33)

### 7.2 L1 Buffer 额外 Scale 空间

```cpp
// MXFP8: 额外预留 qScale 空间（每个 ping-pong buffer 尾部 D_BASIC_BLOCK 个元素）
uint64_t qL1ExtraScale = IS_MXFP8 ? D_BASIC_BLOCK : 0;
pipe->InitBuffer(bufQL1_, QUERY_BUF_NUM * (M_BASIC_BLOCK * D_BASIC_BLOCK + qL1ExtraScale) * sizeof(Q_T));

// MXFP8: 额外预留 kScale 空间（所有 ping-pong buffer 之后）
uint64_t kL1ExtraScale = IS_MXFP8 ? D_BASIC_BLOCK : 0;
pipe->InitBuffer(bufKeyL1_, KEY_BUF_NUM * s2BasicBlock_ * D_BASIC_BLOCK * sizeof(K_T)
                 + kL1ExtraScale * sizeof(K_T));
```
> [service_cube.h L143-150](../op_kernel/arch35/quant_lightning_indexer_service_cube.h#L143)

### 7.3 InitMm1GlobalTensor — 存储 Scale Global Tensor

```cpp
if constexpr (IS_MXFP8) {
    qScaleGm_ = qScaleGm;
    kScaleGm_ = kScaleGm;
}
```
> [service_cube.h L171-174](../op_kernel/arch35/quant_lightning_indexer_service_cube.h#L171)

### 7.4 ComputeMm1 — Scale 加载触发点

```cpp
// s2 循环中加载 kScale
if constexpr (IS_MXFP8) {
    LoadKScaleToL1(s2L1RealSize, s2GmBaseOffset + s2GmOffset, runInfo);
}
...
// s1 循环中加载 qScale（仅首次 s2 迭代）
if constexpr (IS_MXFP8) {
    LoadQScaleToL1(s1gL1RealSize, s1gGmOffset, runInfo);
}
```
> [service_cube.h L197-219](../op_kernel/arch35/quant_lightning_indexer_service_cube.h#L197)

### 7.5 LoadQScaleToL1 — qScale GM → L1 ★

qScale 存储在当前 queryL1_ ping-pong buffer 的数据之后。

```cpp
// MXFP8: 加载 qScale 到 L1（存储在当前 queryL1_ ping-pong buffer 的数据之后）
// qScale shape: [S1*G, D/32]，每个 token 对应 D/32=4 个 fp8_e8m0 scale
template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::LoadQScaleToL1(uint64_t s1gL1RealSize, uint64_t s1gGmOffset,
                                                         const QLICommon::RunInfo &runInfo)
{
    if constexpr (!IS_MXFP8) return;
    // qScale 存储在当前 ping-pong buffer 的数据之后
    // 每个 buffer 的 stride = M_BASIC_BLOCK * D_BASIC_BLOCK + D_BASIC_BLOCK
    uint64_t bufStride = M_BASIC_BLOCK * D_BASIC_BLOCK + D_BASIC_BLOCK;
    uint64_t scaleOffsetInBuf = (queryL1Mte2BufIdx_ % QUERY_BUF_NUM) * bufStride
                                + s1gL1RealSize * D_BASIC_BLOCK;
    LocalTensor<bfloat16_t> scaleL1 = queryL1_[scaleOffsetInBuf].template ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> scaleGmCast;
    scaleGmCast.SetGlobalBuffer((__gm__ bfloat16_t *)(qScaleGm_.GetPhyAddr()));

    uint32_t scalePerToken = constInfo_.headDim / MXFP8_BLOCK_SIZE;  // D/32 = 4
    Dn2NzParams dn2Nzparam;
    dn2Nzparam.dnNum = 1;
    dn2Nzparam.nValue = scalePerToken / FP8_TWO;  // 2个fp8_e8m0打包成1个bf16
    dn2Nzparam.dValue = s1gL1RealSize;
    dn2Nzparam.srcDnMatrixStride = 0;
    dn2Nzparam.srcDValue = scalePerToken / FP8_TWO;
    dn2Nzparam.dstNzC0Stride = scalePerToken / FP8_TWO;
    dn2Nzparam.dstNzNStride = 1;
    dn2Nzparam.dstNzMatrixStride = scalePerToken / FP8_TWO;
    uint64_t gmOffset = runInfo.tensorQScaleOffset + s1gGmOffset * scalePerToken;
    DataCopy(scaleL1, scaleGmCast[gmOffset / FP8_TWO], dn2Nzparam);
}
```
> [service_cube.h L401-429](../op_kernel/arch35/quant_lightning_indexer_service_cube.h#L401)

### 7.6 LoadKScaleToL1 — kScale GM → L1 ★

kScale 存储在 keyL1_ buffer 的尾部区域（所有 ping-pong buffer 之后）。

```cpp
// MXFP8: 加载 kScale 到 L1（存储在 keyL1_ buffer 的尾部区域）
// kScale shape: [S2, D/32]，每个 S2 token 对应 D/32=4 个 fp8_e8m0 scale
template <typename QLIT>
__aicore__ inline void QLIMatmul<QLIT>::LoadKScaleToL1(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                         const QLICommon::RunInfo &runInfo)
{
    if constexpr (!IS_MXFP8) return;
    // kScale 存储在 keyL1_ buffer 尾部
    uint64_t scaleOffsetInBuf = KEY_BUF_NUM * s2BasicBlock_ * D_BASIC_BLOCK;
    LocalTensor<bfloat16_t> scaleL1 = keyL1_[scaleOffsetInBuf].template ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> scaleGmCast;
    scaleGmCast.SetGlobalBuffer((__gm__ bfloat16_t *)(kScaleGm_.GetPhyAddr()));

    uint32_t scalePerToken = constInfo_.headDim / MXFP8_BLOCK_SIZE;  // D/32 = 4
    Dn2NzParams dn2Nzparam;
    dn2Nzparam.dnNum = 1;
    dn2Nzparam.nValue = scalePerToken / FP8_TWO;
    dn2Nzparam.dValue = s2L1RealSize;
    dn2Nzparam.srcDnMatrixStride = 0;
    dn2Nzparam.srcDValue = scalePerToken / FP8_TWO;
    dn2Nzparam.dstNzC0Stride = scalePerToken / FP8_TWO;
    dn2Nzparam.dstNzNStride = 1;
    dn2Nzparam.dstNzMatrixStride = scalePerToken / FP8_TWO;
    uint64_t gmOffset = runInfo.tensorKeyScaleOffset + s2GmOffset * scalePerToken;
    DataCopy(scaleL1, scaleGmCast[gmOffset / FP8_TWO], dn2Nzparam);
}
```
> [service_cube.h L431-456](../op_kernel/arch35/quant_lightning_indexer_service_cube.h#L431)

### 7.7 LoadQueryToL0a — L1 → L0 (Mx 变体) ★

使用 `LoadData` 的 Mx 变体，**同时**加载 fp8_e4m3 数据和 fp8_e8m0 scale 到 L0。

```cpp
if constexpr (IS_MXFP8) {
    // MXFP8: 使用 LoadData Mx 变体，同时加载数据和 scale 到 L0
    LocalTensor<mx_fp8_e4m3_t> dstTensor =
        queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET]
            .template ReinterpretCast<mx_fp8_e4m3_t>();
    // qScale 存储在当前 queryL1_ ping-pong buffer 的数据之后
    uint64_t scaleOffsetInBuf = (queryL1Mte1BufIdx_ % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET
                                + s1gL1RealSize * D_BASIC_BLOCK;
    LocalTensor<fp8_e8m0_t> scaleL1 =
        queryL1_[scaleOffsetInBuf].template ReinterpretCast<fp8_e8m0_t>();

    LoadData2DParamsV2 loadDataA2DParams;
    loadDataA2DParams.mStartPosition = CeilDiv(s1gL1Offset, BLOCK_CUBE);
    loadDataA2DParams.kStartPosition = 0;
    loadDataA2DParams.mStep = CeilDiv(s1gL0RealSize, BLOCK_CUBE);
    loadDataA2DParams.kStep = CeilDiv(constInfo_.headDim, FP8_BLOCK_CUBE);
    loadDataA2DParams.srcStride = CeilDiv(s1gL1RealSize, BLOCK_CUBE);
    loadDataA2DParams.dstStride = CeilDiv(s1gL0RealSize, BLOCK_CUBE);
    loadDataA2DParams.ifTranspose = false;

    uint32_t scalePerToken = constInfo_.headDim / MXFP8_BLOCK_SIZE;  // D/32 = 4
    LoadData2DMxParams loadAScaleParam;
    loadAScaleParam.xStartPosition = 0;
    loadAScaleParam.yStartPosition = 0;
    loadAScaleParam.xStep = CeilDiv(s1gL0RealSize, BLOCK_CUBE);
    loadAScaleParam.yStep = CeilDiv(constInfo_.headDim, FP8_BLOCK_CUBE * FP8_TWO);
    loadAScaleParam.srcStride = CeilDiv(s1gL1RealSize, BLOCK_CUBE);
    loadAScaleParam.dstStride = loadAScaleParam.yStep;
    LoadData(dstTensor,
             queryL1_[(queryL1Mte1BufIdx_ % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET],
             scaleL1, loadDataA2DParams, loadAScaleParam);
}
```
> [service_cube.h L482-510](../op_kernel/arch35/quant_lightning_indexer_service_cube.h#L482)

### 7.8 LoadKeyToL0b — L1 → L0 (Mx 变体) ★

```cpp
if constexpr (IS_MXFP8) {
    // MXFP8: 使用 LoadData Mx 变体，同时加载数据和 scale 到 L0
    LocalTensor<mx_fp8_e4m3_t> dstTensor =
        keyL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET]
            .template ReinterpretCast<mx_fp8_e4m3_t>();
    // kScale 存储在 keyL1_ buffer 的尾部区域（所有 ping-pong buffer 之后）
    uint64_t scaleOffsetInBuf = KEY_BUF_NUM * s2BasicBlock_ * D_BASIC_BLOCK;
    LocalTensor<fp8_e8m0_t> scaleL1 =
        keyL1_[scaleOffsetInBuf].template ReinterpretCast<fp8_e8m0_t>();

    LoadData2DParamsV2 loadData2DV2;
    loadData2DV2.mStartPosition = CeilDiv(s2L1Offset, BLOCK_CUBE);
    loadData2DV2.kStartPosition = 0;
    loadData2DV2.mStep = CeilDiv(s2L0RealSize, BLOCK_CUBE);
    loadData2DV2.kStep = CeilDiv(constInfo_.headDim, FP8_BLOCK_CUBE);
    loadData2DV2.srcStride = CeilDiv(s2L1RealSize, BLOCK_CUBE);
    loadData2DV2.dstStride = CeilDiv(s2L0RealSize, BLOCK_CUBE);
    loadData2DV2.ifTranspose = false;

    uint32_t scalePerToken = constInfo_.headDim / MXFP8_BLOCK_SIZE;  // D/32 = 4
    LoadData2DMxParams loadDataMxParams;
    loadDataMxParams.xStartPosition = 0;
    loadDataMxParams.yStartPosition = 0;
    loadDataMxParams.xStep = CeilDiv(s2L0RealSize, BLOCK_CUBE);
    loadDataMxParams.yStep = CeilDiv(constInfo_.headDim, FP8_BLOCK_CUBE * FP8_TWO);
    loadDataMxParams.srcStride = CeilDiv(s2L1RealSize, BLOCK_CUBE);
    loadDataMxParams.dstStride = loadDataMxParams.yStep;
    LoadData(dstTensor,
             keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * keyBufferOffset_],
             scaleL1, loadData2DV2, loadDataMxParams);
}
```
> [service_cube.h L530-557](../op_kernel/arch35/quant_lightning_indexer_service_cube.h#L530)

### 7.9 ComputeL0c — mmad_mx 矩阵乘 ★

将 L0 tensor 重解释为 `mx_fp8_e4m3_t` 类型，触发硬件 `mmad_mx` 指令。

```cpp
if constexpr (IS_MXFP8) {
    // MXFP8: 使用 mx_fp8_e4m3_t 类型触发 mmad_mx 硬件指令
    LocalTensor<mx_fp8_e4m3_t> qL0Mx =
        queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET]
            .template ReinterpretCast<mx_fp8_e4m3_t>();
    LocalTensor<mx_fp8_e4m3_t> kL0Mx =
        keyL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET]
            .template ReinterpretCast<mx_fp8_e4m3_t>();
    Mmad(cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET], qL0Mx, kL0Mx, mmadParams);
} else {
    Mmad(cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET],
         queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
         keyL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET], mmadParams);
}
```
> [service_cube.h L583-594](../op_kernel/arch35/quant_lightning_indexer_service_cube.h#L583)

---

## 八、Vector 核后处理 — service_vector.h

> **文件路径**: [`op_kernel/arch35/quant_lightning_indexer_service_vector.h`](../op_kernel/arch35/quant_lightning_indexer_service_vector.h)

### 8.1 IS_MXFP8 标志

```cpp
static constexpr bool IS_MXFP8 = QLIT::isMxFp8;
```
> [service_vector.h L37](../op_kernel/arch35/quant_lightning_indexer_service_vector.h#L37)

### 8.2 Buffer 初始化 — 跳过 Scale Buffer

MXFP8 模式下不分配 kScale/qScale UB buffer。

```cpp
pipe->InitBuffer(resMm1Buf_, 2 * CeilDiv(constInfo_.mBaseSize, CVRATIO) * s2BaseSize_ * sizeof(QK_T));
pipe->InitBuffer(weightBuf_, 2 * CeilDiv(s1BaseSize_, CVRATIO) * UB_BANK_DEPTH_STRIDE);
pipe->InitBuffer(weightFloatBuf_, 2 * CeilDiv(s1BaseSize_, CVRATIO) * UB_BANK_DEPTH_STRIDE);
if constexpr (!IS_MXFP8) {
    pipe->InitBuffer(kScaleBuf_, 2 * s2BaseSize_ * 16 * sizeof(SCALE_T));
    kScaleUB_ = kScaleBuf_.Get<SCALE_T>();
    pipe->InitBuffer(qScaleBuf_, 2 * CeilDiv(s1BaseSize_, CVRATIO) * UB_BANK_DEPTH_STRIDE);
    qScaleUB_ = qScaleBuf_.Get<SCALE_T>();
}
pipe->InitBuffer(outBuf_, 2 * CeilDiv(s1BaseSize_, CVRATIO) * s2BaseSize_ * sizeof(uint16_t));
...
// 刷-1（MXFP8 场景不需要 kScale buffer）
if constexpr (!IS_MXFP8) {
    Duplicate(kScaleUB_, static_cast<SCALE_T>(0), 2 * s2BaseSize_ * 16);
}
```
> [service_vector.h L145-180](../op_kernel/arch35/quant_lightning_indexer_service_vector.h#L145)

### 8.3 ProcessVec1 — 跳过 Scale GM → UB 拷贝

```cpp
if constexpr (!IS_MXFP8) {
    //qScaleGm  -->  qScaleUB_
    DataCopyPadExtParams<SCALE_T> padQScaleParams{false, 0, 0, 0};
    DataCopyExtParams qDataCopyExtParams;
    qDataCopyExtParams.blockCount = curAivS1ProcNum;
    qDataCopyExtParams.blockLen = gSize_ * sizeof(SCALE_T);
    qDataCopyExtParams.srcStride = 0;
    qDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - qDataCopyExtParams.blockLen) / 32;
    DataCopyPad(qScaleUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(SCALE_T))],
                qScaleGm[weightGmOffset], qDataCopyExtParams, padQScaleParams);
}
```
> [service_vector.h L358-368](../op_kernel/arch35/quant_lightning_indexer_service_vector.h#L358)

```cpp
if constexpr (!IS_MXFP8) {
    if ((info.s2Idx - info.s2Start) % 16 == 0) {
        WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + kScalepingpong);
        uint32_t getLen = ...;
        //kScaleGm  -->  kScaleUB_
        GetKeyScale(info, kScaleUB_, info.bIdx, curS2Idx, getLen);
        SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_KSCALE + kScalepingpong);
        WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_KSCALE + kScalepingpong);
    }
}
```
> [service_vector.h L373-384](../op_kernel/arch35/quant_lightning_indexer_service_vector.h#L373)

### 8.4 核心计算 — 调用 MxFP8 变体 ★

```cpp
if constexpr (IS_MXFP8) {
    // MXFP8: QK 已含 qScale×kScale，只需 weights 广播乘法 + Relu + ReduceSum
    vector1::BatchMulWeightAndReduceSum_MxFP8(
        outBase, UB_BANK_DEPTH_STRIDE / sizeof(uint16_t),
        qkBase, qkVLstride,
        (uint32_t)(gSize_ * UB_BANK_DEPTH_STRIDE / sizeof(float)),
        weightBase, UB_BANK_DEPTH_STRIDE / sizeof(W_T),
        weightFloatBase,
        gSize_, curAivS1ProcNum);
} else {
    auto qScaleBase = qScaleUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(SCALE_T))];
    auto kScaleBase = kScaleUB_[kScalepingpong * 16 * s2BaseSize_
                      + ((info.s2Idx - info.s2Start) % 16) * s2BaseSize_];
    vector1::BatchMulWeightAndReduceSum(
        outBase, UB_BANK_DEPTH_STRIDE / sizeof(uint16_t),
        qkBase, qkVLstride, (uint32_t)(gSize_ * UB_BANK_DEPTH_STRIDE / sizeof(float)),
        weightBase, UB_BANK_DEPTH_STRIDE / sizeof(W_T), weightFloatBase,
        kScaleBase, (uint32_t)0,
        qScaleBase, UB_BANK_DEPTH_STRIDE / sizeof(SCALE_T),
        gSize_, curAivS1ProcNum);
}
```
> [service_vector.h L397-414](../op_kernel/arch35/quant_lightning_indexer_service_vector.h#L397)

### 8.5 跳过 Scale Buffer 释放标志

```cpp
if constexpr (!IS_MXFP8) {
    if (info.isFirstS2InnerLoop) {
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + qScalepingpong);
    }
    if ((info.s2Idx - info.s2Start) % 16 == 0) {
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + kScalepingpong);
    }
}
```
> [service_vector.h L415-422](../op_kernel/arch35/quant_lightning_indexer_service_vector.h#L415)

---

## 九、VF 底层函数 — vector1.h

> **文件路径**: [`op_kernel/arch35/vf/quant_lightning_indexer_vector1.h`](../op_kernel/arch35/vf/quant_lightning_indexer_vector1.h)

### 9.1 MxFP8 vs 原版对比

| 步骤 | 原版 (含 Scale) | MxFP8 变体 (MXFP8) |
|------|----------------|---------------------|
| 1. 加载 weight | `weights × qScale` | `weights` (不乘 qScale) |
| 2. 加载 QK | QK (int32/float) | QK (float, 已含 scale) |
| 3. 加权累加 | `QK × (weights×qScale)` → `accum × kScale` | `Relu(QK) × weights` → `accum` (不乘 kScale) |
| 4. 输出 | bf16 → sortable key | bf16 → sortable key |

### 9.2 S1=1, float weight — MulWeightAndReduceSum_MxFP8 ★

```cpp
// float in uint16 out, 无 scale 版本（MXFP8）
template <typename QK_T>
__simd_vf__ inline void MulWeightAndReduceSum_MxFP8(__ubuf__ uint16_t* out_,
                                                       __ubuf__ QK_T* qk_,
                                                       const uint32_t qkVLStride,
                                                       __ubuf__ float* weight_,
                                                       const int gSize)
{
    // ... 寄存器声明 ...

    // 加载 weights（不乘 qScale）
    AscendC::MicroAPI::LoadAlign<float>(regW, weight_);

    DuplicateZero(regSum0, maskAllB32);
    DuplicateZero(regSum1, maskAllB32);

    // unroll2: Relu + MulAdd（不乘 kScale）
    for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); i += 2) {
        if constexpr (std::is_same<QK_T, int32_t>::value) {
            MicroAPI::LoadAlign<int32_t>(regQKInt32[0], qk_ + 128 * i);
            MicroAPI::Cast<float, int32_t, castTraitInt32ToFP32>(regQK[0], regQKInt32[0], maskAllB32);
            // ... 同理 regQK[1] ...
        } else {
            MicroAPI::LoadAlign<float>(regQK[0], qk_ + 128 * i);
            MicroAPI::LoadAlign<float>(regQK[1], qk_ + 128 * i + qkVLStride);
        }
        BroadcastLane(regwBrc, regW, i);
        WeightedAccum(regSum0, regQK, regwBrc, maskAllB32);
        // ... 第二个 unroll ...
        BroadcastLane(regwBrc, regW, i + 1);
        WeightedAccum(regSum1, regQK, regwBrc, maskAllB32);
    }

    // 合并累加器（不乘 kScale）
    AscendC::MicroAPI::Add(regSum0[0], regSum0[0], regSum1[0], maskAllB32);
    AscendC::MicroAPI::Add(regSum0[1], regSum0[1], regSum1[1], maskAllB32);

    // float → bf16 → sortable uint16 key
    AscendC::MicroAPI::Cast<bfloat16_t, float, ...>(regSumBF16, regSum0, maskAllB32);
    FloatToSortableKey<bfloat16_t>(regOut, regSumBF16, bf16Ctx, maskAllB16);
    AscendC::MicroAPI::StoreAlign<uint16_t, ...>(out_, regOut, maskAllB16);
}
```
> [vector1.h L877-949](../op_kernel/arch35/vf/quant_lightning_indexer_vector1.h#L877)

### 9.3 S1=1, bfloat16 weight — MulWeightAndReduceSum_MxFP8 ★

```cpp
// bfloat16 weight in, 无 scale 版本（MXFP8）
template <typename QK_T>
__simd_vf__ inline void MulWeightAndReduceSum_MxFP8(__ubuf__ uint16_t* out_,
                                                       __ubuf__ QK_T* qk_,
                                                       const uint32_t qkVLStride,
                                                       __ubuf__ bfloat16_t* weight_,
                                                       const int gSize)
{
    // ... 寄存器声明 ...

    // 加载 bf16 weight 并转换为 float
    AscendC::MicroAPI::LoadAlign<bfloat16_t,
        AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(regWBF16, weight_);
    AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitBF16ToFP32>(regW, regWBF16, maskAllB16);

    // ... 后续逻辑同 float 版本 ...
}
```
> [vector1.h L951-1024](../op_kernel/arch35/vf/quant_lightning_indexer_vector1.h#L951)

### 9.4 S1=2, float weight — MulWeightAndReduceSum2_MxFP8 ★

```cpp
// S1=2 无 scale 版本（MXFP8）, float weight
template <typename QK_T>
__simd_vf__ inline void MulWeightAndReduceSum2_MxFP8(
    __ubuf__ uint16_t* out0_, __ubuf__ uint16_t* out1_, uint32_t outStride,
    __ubuf__ QK_T* qk0_, __ubuf__ QK_T* qk1_, uint32_t qkVLStride, uint32_t qkStride,
    __ubuf__ float* weight0_, __ubuf__ float* weight1_, uint32_t weightStride,
    __ubuf__ float* weightFloat_, const int gSize)
{
    // 加载 2 组 weights
    AscendC::MicroAPI::LoadAlign<float>(regW[0], weight0_);
    AscendC::MicroAPI::LoadAlign<float>(regW[1], weight1_);
    AscendC::MicroAPI::StoreAlign<float, ...>(weightFloat_, regW[1], maskAllB32);

    for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); i++) {
        // 加载 QK0, QK1
        MicroAPI::LoadAlign<float>(regQK0[0], qk0_ + 128 * i);
        MicroAPI::LoadAlign<float>(regQK1[0], qk1_ + 128 * i);
        // ...
        // Relu + MulAddDst（不乘 kScale）
        AscendC::MicroAPI::Relu(regQK0[0], regQK0[0], maskAllB32);
        AscendC::MicroAPI::Relu(regQK1[0], regQK1[0], maskAllB32);
        AscendC::MicroAPI::MulAddDst(regSum0[0], regQK0[0], regwBrc[0], maskAllB32);
        AscendC::MicroAPI::MulAddDst(regSum1[0], regQK1[0], regwBrc[1], maskAllB32);
        // ...
    }
    // float → bf16 → sortable key
    FloatX2ToSortableKey<bfloat16_t>(regOut[0], regOut[1], regSumBF16[0], regSumBF16[1], bf16Ctx, maskAllB16);
}
```
> [vector1.h L1026-1110](../op_kernel/arch35/vf/quant_lightning_indexer_vector1.h#L1026)

### 9.5 S1=2, bfloat16 weight — MulWeightAndReduceSum2_MxFP8 ★

```cpp
// S1=2 无 scale 版本（MXFP8）, bfloat16 weight
template <typename QK_T>
__simd_vf__ inline void MulWeightAndReduceSum2_MxFP8(
    __ubuf__ uint16_t* out0_, __ubuf__ uint16_t* out1_, uint32_t outStride,
    __ubuf__ QK_T* qk0_, __ubuf__ QK_T* qk1_, uint32_t qkVLStride, uint32_t qkStride,
    __ubuf__ bfloat16_t* weight0_, __ubuf__ bfloat16_t* weight1_, uint32_t weightStride,
    __ubuf__ float* weightFloat_, const int gSize)
{
    // 加载 bf16 weight 并转换为 float
    AscendC::MicroAPI::LoadAlign<bfloat16_t,
        AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(regWBF16[0], weight0_);
    AscendC::MicroAPI::LoadAlign<bfloat16_t,
        AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(regWBF16[1], weight1_);
    AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitBF16ToFP32>(regW[0], regWBF16[0], maskAllB16);
    AscendC::MicroAPI::Cast<float, bfloat16_t, castTraitBF16ToFP32>(regW[1], regWBF16[1], maskAllB16);

    // ... 后续逻辑同 float 版本 ...
}
```
> [vector1.h L1112-1203](../op_kernel/arch35/vf/quant_lightning_indexer_vector1.h#L1112)

### 9.6 BatchMulWeightAndReduceSum_MxFP8 — 调度函数 ★

```cpp
// MXFP8 调度函数：无 scale 版本
template<typename QK_T, typename W_T, typename SCORE_T>
__aicore__ inline void BatchMulWeightAndReduceSum_MxFP8(
    const LocalTensor<SCORE_T> &out_, uint32_t outStride,
    const LocalTensor<QK_T> &qk_, uint32_t qkVLStride, uint32_t qkStride,
    const LocalTensor<W_T> &weight_, uint32_t weightStride,
    const LocalTensor<float> &weightFloat_,
    const int gSize, const int batch)
{
    if (batch != 2 && batch != 1) { return; }
    auto weight = (__ubuf__ W_T *)weight_.GetPhyAddr();
    auto weightFloat = (__ubuf__ float *)weightFloat_.GetPhyAddr();
    auto qk = (__ubuf__ QK_T *)qk_.GetPhyAddr();
    auto out = (__ubuf__ uint16_t *)out_.GetPhyAddr();

    if (batch == 2) {
        auto weight1 = weight + weightStride;
        auto qk1 = qk + qkStride;
        auto out1 = out + outStride;
        MulWeightAndReduceSum2_MxFP8(out, out1, outStride,
                                       qk, qk1, qkVLStride, qkStride,
                                       weight, weight1, weightStride, weightFloat,
                                       gSize);
    } else {
        MulWeightAndReduceSum_MxFP8(out, qk, qkVLStride, weight, gSize);
    }
}
```
> [vector1.h L1205-1238](../op_kernel/arch35/vf/quant_lightning_indexer_vector1.h#L1205)

---

## 十、代码位置速查表

| 功能 | 文件 | 行号 | 关键标识 |
|------|------|------|---------|
| MXFP8 类型定义 | [`common.h`](../op_kernel/quant_lightning_indexer_common.h#L19) | L19-24 | `FP8E4M3`, `FP8E8M0`, `MXFP8_BLOCK_SIZE` |
| QLIType MXFP8 特化 | [`common.h`](../op_kernel/quant_lightning_indexer_common.h#L56) | L56-71 | `isMxFp8 = true` |
| 算子类型注册 | [`def.cpp`](../op_host/quant_lightning_indexer_def.cpp#L89) | L84-141 | `DT_FLOAT8_E8M0` 位置4 |
| Tiling 类型校验 | [`tiling.cpp`](../op_host/quant_lightning_indexer_tiling.cpp#L286) | L286-294 | `DT_FLOAT8_E8M0` 组合 |
| Tiling Shape 校验 | [`tiling.cpp`](../op_host/quant_lightning_indexer_tiling.cpp#L740) | L740-783 | `isMxFp8Scale`, `D/32` |
| Tiling kScale Stride | [`tiling.cpp`](../op_host/quant_lightning_indexer_tiling.cpp#L853) | L853-858 | `blockSize_ * (headDim_/32)` |
| Kernel MXFP8 分发 | [`quant_lightning_indexer.cpp`](../op_kernel/quant_lightning_indexer.cpp#L53) | L53-57 | `DT_FLOAT8_E8M0` dispatch |
| AIV/AIC Scale 初始化 | [`kernel.h`](../op_kernel/arch35/quant_lightning_indexer_kernel.h#L430) | L430-456 | `IS_MXFP8` 分支 |
| qScale/kScale 偏移 | [`kernel.h`](../op_kernel/arch35/quant_lightning_indexer_kernel.h#L568) | L568-592 | `headDim / MXFP8_BLOCK_SIZE` |
| L1 Buffer Scale 空间 | [`service_cube.h`](../op_kernel/arch35/quant_lightning_indexer_service_cube.h#L143) | L143-150 | `qL1ExtraScale`, `kL1ExtraScale` |
| LoadQScaleToL1 | [`service_cube.h`](../op_kernel/arch35/quant_lightning_indexer_service_cube.h#L401) | L401-429 | `Dn2Nz`, `scalePerToken/FP8_TWO` |
| LoadKScaleToL1 | [`service_cube.h`](../op_kernel/arch35/quant_lightning_indexer_service_cube.h#L431) | L431-456 | `Dn2Nz`, keyL1_ 尾部 |
| LoadQueryToL0a (Mx) | [`service_cube.h`](../op_kernel/arch35/quant_lightning_indexer_service_cube.h#L482) | L482-510 | `mx_fp8_e4m3_t`, `LoadData` Mx |
| LoadKeyToL0b (Mx) | [`service_cube.h`](../op_kernel/arch35/quant_lightning_indexer_service_cube.h#L530) | L530-557 | `mx_fp8_e4m3_t`, `LoadData` Mx |
| ComputeL0c (mmad_mx) | [`service_cube.h`](../op_kernel/arch35/quant_lightning_indexer_service_cube.h#L583) | L583-594 | `Mmad(qL0Mx, kL0Mx)` |
| Vector Buffer 跳过 | [`service_vector.h`](../op_kernel/arch35/quant_lightning_indexer_service_vector.h#L151) | L151-156 | `!IS_MXFP8` |
| Vector 调用 MxFP8 | [`service_vector.h`](../op_kernel/arch35/quant_lightning_indexer_service_vector.h#L397) | L397-404 | `BatchMulWeightAndReduceSum_MxFP8` |
| MxFP8 S1=1 float | [`vector1.h`](../op_kernel/arch35/vf/quant_lightning_indexer_vector1.h#L877) | L877-949 | `MulWeightAndReduceSum_MxFP8` |
| MxFP8 S1=1 bf16 | [`vector1.h`](../op_kernel/arch35/vf/quant_lightning_indexer_vector1.h#L951) | L951-1024 | `MulWeightAndReduceSum_MxFP8` |
| MxFP8 S1=2 float | [`vector1.h`](../op_kernel/arch35/vf/quant_lightning_indexer_vector1.h#L1026) | L1026-1110 | `MulWeightAndReduceSum2_MxFP8` |
| MxFP8 S1=2 bf16 | [`vector1.h`](../op_kernel/arch35/vf/quant_lightning_indexer_vector1.h#L1112) | L1112-1203 | `MulWeightAndReduceSum2_MxFP8` |
| MxFP8 调度函数 | [`vector1.h`](../op_kernel/arch35/vf/quant_lightning_indexer_vector1.h#L1205) | L1205-1238 | `BatchMulWeightAndReduceSum_MxFP8` |

---

## 附录：相关文档

- [MXFP8 修改计划](mxfp8_modification_plan.md) — 完整的 MXFP8 适配设计方案
- [MXFP8 流程对比](mxfp8_flow_comparison.md) — 新旧流程的可视化对比图
- [算子接口文档](aclnnQuantLightningIndexer.md) — ACL NN 接口说明

---

**文档版本**: 1.0  
**创建日期**: 2026-06-08  
**适用范围**: QLI 算子 MXFP8 量化代码学习
