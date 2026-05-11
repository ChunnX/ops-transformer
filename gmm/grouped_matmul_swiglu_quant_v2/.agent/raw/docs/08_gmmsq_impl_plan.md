# GMMSQ MxA8W4 实现详细规划

> 本文档基于 GMMFR（GroupedMatmulFinalizeRouting）MxA8W4 数据流的完整实现，给出 GMMSQ（GroupedMatmulSwigluQuantV2）MxA8W4 数据流的详细实现规划。

---

## 1. 文件映射关系

### 1.1 GMMFR 源文件 → GMMSQ 目标文件

| GMMFR 源文件 | GMMSQ 目标文件 | 复用程度 | 说明 |
|-------------|---------------|---------|------|
| `op_kernel/arch35/common/basic_block_config.h` | `op_kernel/arch35/common/basic_block_config.h` | 🟢 直接复用 | 结构体定义相同 |
| `op_kernel/arch35/common/basic_block_vf_mx.h` | `op_kernel/arch35/common/basic_block_vf_mx.h` | 🟢 直接复用 | MxA8W4NzParams 等 VF 参数结构体 |
| `op_kernel/arch35/common/tool.h` | `op_kernel/arch35/common/tool.h` | 🟢 直接复用 | 工具函数和常量 |
| `op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_tiling_data.h` | `op_kernel/arch35/weight_quant_basic_block/gmmsq_weight_quant_tiling_data.h` | 🟡 改写 | 去除 sharedInput 字段 |
| `op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_tiling_key.h` | `op_kernel/arch35/weight_quant_basic_block/gmmsq_weight_quant_tiling_key.h` | 🟡 改写 | 减少 HASBIAS 组合 |
| `op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_resplit_controller.h` | `op_kernel/arch35/weight_quant_basic_block/gmmsq_weight_quant_resplit_controller.h` | 🟡 改写 | 去除 Routing 逻辑，适配 N/2 |
| `op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_vcv_basic_block.h` | `op_kernel/arch35/weight_quant_basic_block/gmmsq_weight_quant_vcv_basic_block.h` | 🟡 改写 | PhaseB 改为 SwiGLU+Quant |
| `op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_cube_compute.h` | `op_kernel/arch35/weight_quant_basic_block/gmmsq_weight_quant_cube_compute.h` | 🟢 基本复用 | 仅 GetTensorC 目标地址变化 |
| `op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_cube_compute_tools.h` | `op_kernel/arch35/weight_quant_basic_block/gmmsq_weight_quant_cube_compute_tools.h` | 🟢 直接复用 | 工具函数不变 |
| `op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_vec_compute.h` | `op_kernel/arch35/weight_quant_basic_block/gmmsq_weight_quant_vec_compute.h` | 🔴 大量改写 | 后处理完全重写 |
| `op_kernel/arch35/weight_quant_basic_block/gmm_fr_mx_a8w4_vf.h` | `op_kernel/arch35/weight_quant_basic_block/gmmsq_mx_a8w4_vf.h` | 🟡 改写 | 保留反量化 VF，新增 SwiGLU/Quant VF |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_weight_quant_tiling.cpp` | `op_host/op_tiling/arch35/gmmsq_weight_quant_tiling.cpp` | 🟡 改写 | 场景识别 + TilingData 调整 |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_weight_quant_tiling.h` | `op_host/op_tiling/arch35/gmmsq_weight_quant_tiling.h` | 🟡 改写 | 头文件调整 |
| `op_kernel/grouped_matmul_finalize_routing_apt.cpp`（V310_GMM_FR_ANTI_QUANT 分支） | `op_kernel/grouped_matmul_swiglu_quant_v2_apt.cpp`（新增分支） | 🟡 改写 | 新增 MxA8W4 场景入口 |

### 1.2 文件复用策略

```
GMMFR 目录: op_kernel/arch35/weight_quant_basic_block/
GMMSQ 目录: op_kernel/arch35/weight_quant_basic_block/  (新建)

复用方式:
  - 直接复用: 通过 #include 引用 GMMFR 的 common/ 目录文件
  - 改写: 复制文件到 GMMSQ 目录，修改命名空间和类名
  - 新增: 在 GMMSQ 目录新建文件
```

---

## 2. 实现步骤

### Step 1: Tiling 层改写（场景识别 + TilingData）

**目标文件**：
- `op_host/op_tiling/arch35/gmmsq_weight_quant_tiling.h`
- `op_host/op_tiling/arch35/gmmsq_weight_quant_tiling.cpp`

**工作内容**：

1.1 新增 MxA8W4 场景识别函数：
```cpp
bool IsMxA8W4NzScenario() {
    return (xDtype == DT_FLOAT8_E4M3FN &&
            wDtype == DT_FLOAT4_E2M1 &&
            (wFormat == FORMAT_FRACTAL_NZ || wFormat == FORMAT_FRACTAL_NZ_C0_32) &&
            scaleDtype == DT_FLOAT8_E8M0 &&
            perTokenScaleDtype == DT_FLOAT8_E8M0);
}
```

1.2 定义 GMMSQ 专用 TilingData：
```cpp
struct GMMSQWeightQuantTilingData {
    uint8_t groupListType;
    uint8_t hasBias;          // MxA8W4 固定为 0
    uint16_t reserve1;
    uint32_t groupNum;
    uint32_t coreNum;
    uint32_t outputBs;
    uint32_t groupSize;       // MX_GROUPSIZE = 32
    uint64_t kSize;
    uint64_t nSize;
    // 去除: sharedInputOffset, sharedInputLen, sharedInputWeight, initSize
};
```

1.3 属性校验（MxA8W4 场景）：
- `transpose_x == false`
- `transpose_w == true`
- `groupListType ∈ {0, 1}`
- `dequantMode == 2`（MX 量化）
- `dequantDtype == 0`（DT_FLOAT）
- `quantMode == 2`（MX 量化）
- `bias == nullptr`
- `weightAssistMatrix == nullptr`
- `smoothScale == nullptr`

1.4 Shape 校验：
- `k % 32 == 0`，`k >= 32`
- `n % 128 == 0`，`n >= 128`（注意：GMMSQ 要求 N 128 对齐，而非 32 对齐）
- `x shape = (M, K)`
- `weightScale shape = (E, N, ceil(K/64), 2)`
- `xScale shape = (M, ceil(K/64), 2)`

**验证**：编译通过，Tiling Key 正确生成。

---

### Step 2: Kernel 入口改写（apt.cpp）

**目标文件**：`op_kernel/grouped_matmul_swiglu_quant_v2_apt.cpp`

**工作内容**：

2.1 新增 `V310_GMM_SQ_ANTI_QUANT` 宏：
```cpp
#if (ORIG_DTYPE_X == DT_FLOAT8_E4M3FN && (ORIG_DTYPE_W == DT_FLOAT4_E2M1))
#define V310_GMM_SQ_ANTI_QUANT
#endif
```

2.2 在 `#if ORIG_DTYPE_X_SCALE == DT_FLOAT8_E8M0` 分支中，新增 MxA8W4 场景的 Controller 调用：
```cpp
#if defined(V310_GMM_SQ_ANTI_QUANT)
    if (wFormat == CubeFormat::NZ) {
        GMMSQ::GMMSQWeightQuantResplitController<...> controller;
        controller.Init(x, weight, weightScale, nullptr, groupList, xScale, y, yScale, tiling);
        controller.Process();
    }
#else
    // 原有的 CGMCT 框架路径（GmmSwigluAswt）
#endif
```

2.3 模板参数实例化：
```cpp
GMMSQWeightQuantResplitController<
    DTYPE_X, DTYPE_W, DTYPE_SCALE, DTYPE_SCALE, DTYPE_PERTOKEN_SCALE, DTYPE_BIAS, DTYPE_Y,
    DTYPE_Y_SCALE, HASBIAS,
    WeightQuantBatchMatmulV2::Arch35::MXA8W4_NZNK,
    WeightQuantBatchMatmulV2::Arch35::VEC_ANTIQUANT_CONFIG_DYNAMIC>
    controller;
```

**验证**：编译通过，模板实例化正确。

---

### Step 3: Controller 层改写

**目标文件**：`op_kernel/arch35/weight_quant_basic_block/gmmsq_weight_quant_resplit_controller.h`

**工作内容**：

3.1 类名和命名空间：
```cpp
namespace GMMSQ {
template <typename xType, typename wType, ..., typename yScaleType, bool hasBias, ...>
class GMMSQWeightQuantResplitController { ... };
}
```

3.2 成员变量调整：
```cpp
// 去除:
__gm__ sharedInputDType *shareInputAddr_;
__gm__ logitsType *logitsAddr_;
__gm__ rowIndexType *rowIndexAddr_;

// 新增:
__gm__ yScaleType *yScaleGm_;  // 输出 scale 的 GM 地址
```

3.3 `Init()` 调整：
```cpp
void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR bias,
          GM_ADDR groupList, GM_ADDR perTokenScale,
          GM_ADDR y, GM_ADDR yScale,  // 去除 logitsAddr, rowIndexAddr, shareInput
          const GMMSQWeightQuantTilingData *tiling)
{
    // GM 地址初始化（去除 logits/rowIndex/shareInput）
    // BasicBlock 初始化（传入 yScale 地址）
    basicBlock_.Init(tiling_->groupSize, yGm_, yScaleGm_);
}
```

3.4 `Process()` 调整：
```cpp
void Process() {
    GetBlockIdx();
    InitOffsetParam();
    // 去除: 缓存对齐检查（GMMSQ 不需要 L2 Cache 优化）
    // 去除: InitAtomicGm()（GMMSQ 不需要 GM 清零）
    // 去除: SyncAll()

    for each group:
        GetSplitValueFromGroupList();
        M 轴 L1 分块;
        UpdateGlobalAddr();  // 变化
        for each M 分块:
            SplitNByMultiCore();  // 变化：N 轴切分适配 N/2
}
```

3.5 `UpdateGmAddr()` 调整：
```cpp
void UpdateGmAddr(uint64_t mSize, uint64_t kSize, uint64_t nSize) {
    xGm_ += mSize * kSize;
    weightGm_ += (nSize * kSize) >> 1;
    antiquantScaleGm_ += nSize * CeilDivide(kAlignSize, groupSize);
    perTokenScaleGm_ += mSize * CeilDivide(kAlignSize, groupSize);
    yGm_ += mSize * (nSize / 2);  // 输出 N/2
    yScaleGm_ += mSize * CeilDivide(CeilAlign(nSize / 2, 64), groupSize);  // 新增
    // 去除: logitsAddr_, rowIndexAddr_
}
```

3.6 `SplitNByMultiCore()` 调整：
- N 轴切分时，`basicBlockSize` 保持 256（128 Swish + 128 Gate）
- 输出 N 偏移计算：`outputNOffset = nOffset / 2`（因为输出是 N/2）

**验证**：编译通过，Controller 逻辑正确。

---

### Step 4: BasicBlock 层改写

**目标文件**：`op_kernel/arch35/weight_quant_basic_block/gmmsq_weight_quant_vcv_basic_block.h`

**工作内容**：

4.1 类名和命名空间：
```cpp
namespace GMMSQ {
template <typename xType, ..., typename yScaleType, bool hasBias, ...>
class GMMSQWeightQuantVcvBasicBlock { ... };
}
```

4.2 成员变量调整：
```cpp
// 去除:
LocalTensor<float> ubOutputF32Buffer_;  // 改为 yF32Buffer_

// 新增:
LocalTensor<float> yF32Buffer_;         // MMA Fixpipe 输出（复用 weightHighBit 空间）
LocalTensor<half> maxB16Buffer_;        // ComputeMaxExp 中间结果
LocalTensor<fp8_e8m0_t> scaleE8M0Buffer_;  // MX 量化 scale
LocalTensor<fp8_e4m3fn_t> yFp8Buffer_;     // 量化后 FP8 输出

__gm__ yScaleType *yScaleGm_;           // 输出 scale GM 地址
```

4.3 `Init()` 调整：
```cpp
void Init(uint64_t antiQuantGroupSize, __gm__ yType *y, __gm__ yScaleType *yScale) {
    // L1 Buffer 分配（不变）
    // UB Buffer 分配（重新布局，见 Step 8）
    // AIC 初始化（不变）
    // AIV 初始化（去除 sharedInputWeight）
    vecCompute_.Init(y, yScale);  // 传入 yScale 地址
}
```

4.4 `IterateNzNkWithAiv()` 调整（PhaseB 重写）：

```cpp
// PhaseB: SwiGLU + Quant（替代 MulLogits + RoutingYToGm）
if (curCvLoopIdx > 0) {
    // 1. 通知 AIC 可以 Fixpipe
    SetAivToAic<PIPE_MTE3>(SYNC_AIV_MTE3_AIC_FIX_FLAG);

    // 2. 等待 AIC Fixpipe 完成
    WaitAicToAiv<PIPE_V>(SYNC_AIC_FIX_AIV_VF_FLAG);

    // 3. SwiGLU 计算
    vecCompute_.SwiGLU(yF32Buffer_, lastParam);

    // 4. MX 量化
    vecCompute_.ComputeMaxExp(yF32Buffer_, maxB16Buffer_, lastParam);
    vecCompute_.ComputeScale(maxB16Buffer_, scaleE8M0Buffer_, lastParam);
    vecCompute_.ComputeDataForQuantTargetFp8(yF32Buffer_, scaleE8M0Buffer_, yFp8Buffer_, lastParam);

    // 5. 写出到 GM
    vecCompute_.CopyOutputFromUb2Gm(lastParam.mSize, yFp8Buffer_, lastParam);
    vecCompute_.CopyScaleFromUb2Gm(lastParam.mSize, scaleE8M0Buffer_, lastParam);

    // 6. 同步
    SetFlag<MTE3_V>(EVENT_ID_MTE3_TO_V);
    computeLoopIdx_++;
}
```

4.5 `End()` 调整：
- AIC 侧：不变
- AIV 侧：最后一次 PhaseB 使用 SwiGLU+Quant 替代 MulLogits+Routing

**验证**：编译通过，同步信号正确。

---

### Step 5: VecCompute 层改写

**目标文件**：`op_kernel/arch35/weight_quant_basic_block/gmmsq_weight_quant_vec_compute.h`

**工作内容**：

5.1 类名和命名空间：
```cpp
namespace GMMSQ {
template <typename xType, ..., typename yScaleType, bool hasBias, ...>
class GMMSQWeightQuantVecCompute { ... };
}
```

5.2 保留的函数（从 GMMFR 直接复用）：
- `CopyGmToUb()` — MTE2 搬运 weight + bias
- `WeightAntiQuantComputeNzNk()` — 反量化完整流程
- `AntiQuantProcessNzMxA8W4()` — 反量化参数准备
- `CopyWeightHighBitForAligned()` — UB→L1 搬运
- `SetVToMTE2()` / `WaitVToMTE2()` — 同步辅助
- `End()` — 收尾同步

5.3 去除的函数：
- `MulLogits()`
- `RoutingYToGm()`
- `CopyRowIndexLogitsGmToUb()`
- `CopyShareInputGmToUb()` / `CopyShareInputUbToGm()`
- `CastAndMulWithSharedWeightVf()`
- `InitGmToZero()`

5.4 新增的函数接口（VF 实现见 Step 7）：

```cpp
// SwiGLU: Swish(yF32[:, 0:N/2]) * yF32[:, N/2:N]
void SwiGLU(const LocalTensor<float> &yF32, const BasicBlockOffsetParam &param);

// MX 量化步骤 1: 计算每 32 个元素的 max exponent
void ComputeMaxExp(const LocalTensor<float> &yF32, const LocalTensor<half> &maxB16,
                   const BasicBlockOffsetParam &param);

// MX 量化步骤 2: 计算 shared_exp → E8M0 scale
void ComputeScale(const LocalTensor<half> &maxB16, const LocalTensor<fp8_e8m0_t> &scaleE8M0,
                  const BasicBlockOffsetParam &param);

// MX 量化步骤 3: 量化到 FP8_E4M3FN
void ComputeDataForQuantTargetFp8(const LocalTensor<float> &yF32,
                                   const LocalTensor<fp8_e8m0_t> &scaleE8M0,
                                   const LocalTensor<fp8_e4m3fn_t> &yFp8,
                                   const BasicBlockOffsetParam &param);

// MTE3 写出量化结果
void CopyOutputFromUb2Gm(uint64_t mRealSize, const LocalTensor<fp8_e4m3fn_t> &yFp8,
                          const BasicBlockOffsetParam &param);

// MTE3 写出 scale
void CopyScaleFromUb2Gm(uint64_t mRealSize, const LocalTensor<fp8_e8m0_t> &scaleE8M0,
                         const BasicBlockOffsetParam &param);
```

5.5 `Init()` 调整：
```cpp
void Init(__gm__ yType *y, __gm__ yScaleType *yScale) {
    yFp32Global_.SetGlobalBuffer(y);
    yScaleGlobal_.SetGlobalBuffer(yScale);  // 新增
    // UB Buffer 分配（重新布局）
    // Event ID 初始化（去除 Routing 相关 Event）
}
```

**验证**：编译通过，接口定义正确。

---

### Step 6: CubeCompute 层改写

**目标文件**：`op_kernel/arch35/weight_quant_basic_block/gmmsq_weight_quant_cube_compute.h`

**工作内容**：

6.1 类名和命名空间：
```cpp
namespace GMMSQ {
template <typename xType, ..., typename yType, bool hasBias, ...>
class GMMSQWeightQuantCubeCompute { ... };
}
```

6.2 改动点（极小）：
- `GetTensorC()` 的目标 UB 地址从 `ubOutputF32Buffer_` 改为 `yF32Buffer_`
- 其他所有函数**完全不变**

6.3 工具文件 `gmmsq_weight_quant_cube_compute_tools.h`：
- **直接复用** GMMFR 的工具文件（`LoadAAndScaleL1ToL0`、`LoadBAndScaleL1ToL0`、`MmadCompute`、`FixL0CToDst`、`LoadBiasToBt` 全部不变）

**验证**：编译通过。

---

### Step 7: 新增 SwiGLU 和 Quant VF 实现

**目标文件**：`op_kernel/arch35/weight_quant_basic_block/gmmsq_mx_a8w4_vf.h`

**工作内容**：

7.1 保留 GMMFR 的反量化 VF 函数：
- `FrAntiQuantMxA8W4NzNkVf()` — 4bit→8bit 反量化
- `FrMxA8W4BiasCompute()` — Bias 反量化

7.2 新增 SwiGLU VF 实现：

```cpp
// SwiGLU: Swish(yF32[:, 0:N/2]) * yF32[:, N/2:N]
// 输入: yF32[M, N]，其中前 N/2 为 Swish 分支，后 N/2 为 Gate 分支
// 输出: yF32[M, N/2]（原地覆盖前半部分）
template <bool hasBias>
__simd_vf__ inline void FrSwiGLUVf(uint16_t mSize, uint64_t nHalfSize,
                                    __ubuf__ float *yF32UbAddr)
{
    // Swish(x) = x / (1 + exp(-x)) = x * sigmoid(x)
    // 对每个 M 行:
    //   加载 Swish 分支: ySwish = yF32[m, 0:N/2]
    //   加载 Gate 分支:  yGate   = yF32[m, N/2:N]
    //   计算: ySwish = Swish(ySwish) * yGate
    //   存回: yF32[m, 0:N/2]
    //
    // Swish 的数值实现:
    //   sigmoid(x) = 1 / (1 + exp(-x))
    //   Swish(x) = x * sigmoid(x)
    //
    // 使用 VF 指令: Exp, Add, Div, Mul
}
```

7.3 新增 MX Quant VF 实现：

```cpp
// ComputeMaxExp: 计算每 32 个元素的 max exponent
// 输入: yF32[M, N/2]
// 输出: maxB16[M, ceil(N/2/32)] — 每 32 个元素的最大指数值（B16 格式）
__simd_vf__ inline void FrComputeMaxExpVf(uint16_t mSize, uint64_t nHalfSize,
                                           __ubuf__ float *yF32UbAddr,
                                           __ubuf__ half *maxB16UbAddr)
{
    // 对每个 M 行:
    //   对每 32 个元素:
    //     加载 32 个 float
    //     取绝对值
    //     提取指数位
    //     求最大值
    //     存入 maxB16
}

// ComputeScale: 计算 shared_exp → E8M0 scale
// shared_exp = round(log2(max_val)) - emax
// emax = 8 (for FP8_E4M3FN)
// QScale = 2^shared_exp
__simd_vf__ inline void FrComputeScaleVf(uint16_t mSize, uint64_t nGroupCount,
                                          __ubuf__ half *maxB16UbAddr,
                                          __ubuf__ fp8_e8m0_t *scaleE8M0UbAddr)
{
    // 对每个 maxB16 值:
    //   提取指数 e = (maxB16 >> 10) & 0x1F  // B16 指数位
    //   shared_exp = e - 8  // emax = 8
    //   E8M0 编码: scaleE8M0 = shared_exp  // E8M0 就是纯指数
}

// ComputeDataForQuantTargetFp8: 量化到 FP8_E4M3FN
// Q_i = quantize_to_element_format(S_i / QScale)
__simd_vf__ inline void FrComputeDataForQuantTargetFp8Vf(
    uint16_t mSize, uint64_t nHalfSize,
    __ubuf__ float *yF32UbAddr,
    __ubuf__ fp8_e8m0_t *scaleE8M0UbAddr,
    __ubuf__ fp8_e4m3fn_t *yFp8UbAddr)
{
    // 对每个 M 行:
    //   对每 32 个元素:
    //     加载 scale = scaleE8M0[m, groupIdx]
    //     转换为 float: scale_f = 2^(scale)
    //     对组内每个元素:
    //       yF32[m, n] /= scale_f
    //       量化到 FP8_E4M3FN（硬件指令或手动舍入）
    //     存入 yFp8
}
```

> ⚠️ **注意**：SwiGLU 和 Quant 的 VF 实现由其他团队成员分工完成。GMMSQ 侧需要预留接口，确保函数签名和 UB 地址传递正确。

**验证**：编译通过（接口定义正确，VF 实现可后续填充）。

---

### Step 8: Buffer 分配调整

**目标文件**：`op_kernel/arch35/weight_quant_basic_block/gmmsq_weight_quant_vcv_basic_block.h`（Init 中）

**工作内容**：

8.1 UB Buffer 布局（复用策略）：

```
GMMSQ UB 布局（248KB 内）:
┌────────────────────────────────────────────────────┐  0
│  weightLowBit_  (64KB)                             │
│  [buf0][buf1]  (ubMte2BufferNum=2)                 │
│  ↓ PhaseB 阶段复用为 yFp8Buffer_ (32KB)             │
├────────────────────────────────────────────────────┤  64KB
│  weightHighBit_ (128KB)                            │
│  [buf0][buf1][buf2][buf3]  (四缓冲, 32KB each)      │
│  ↓ PhaseB 阶段复用为 yF32Buffer_ (128KB)            │
├────────────────────────────────────────────────────┤  192KB
│  bias_ (1KB)  |  biasRescale_ (1KB)                │
├────────────────────────────────────────────────────┤  194KB
│  maxB16Buffer_ (2KB)                               │
├────────────────────────────────────────────────────┤  196KB
│  scaleE8M0Buffer_ (16KB)                           │
└────────────────────────────────────────────────────┘  212KB
```

**复用逻辑**：
- `yF32Buffer_` 复用 `weightHighBit_` 的空间：PhaseB 阶段 weightHighBit 已全部搬运到 L1，UB 空间可安全复用
- `yFp8Buffer_` 复用 `weightLowBit_` 的前 32KB：PhaseB 阶段 weightLowBit 已不再使用
- 总使用量：64 + 128 + 1 + 1 + 2 + 16 = **212KB** < 248KB ✅

8.2 L1 Buffer 布局：**与 GMMFR 完全相同**（无 Bias 场景）：
```
L1 (512KB):
  WeightL1_P0(64KB) | ScaleAL1_P0(32KB) | ScaleBL1_P0(32KB) | AL1_P0(128KB) |
  AL1_P1(128KB)     | ScaleBL1_P1(32KB) | ScaleAL1_P1(32KB) | WeightL1_P1(64KB)
```

8.3 L0 Buffer 布局：**与 GMMFR 完全相同**。

**验证**：编译通过，UB 空间不超限。

---

### Step 9: 编译验证

**工作内容**：

9.1 更新 CMakeLists.txt，添加新文件：
```cmake
# op_kernel/arch35/weight_quant_basic_block/ 目录下的新文件
gmmsq_weight_quant_tiling_data.h
gmmsq_weight_quant_tiling_key.h
gmmsq_weight_quant_resplit_controller.h
gmmsq_weight_quant_vcv_basic_block.h
gmmsq_weight_quant_cube_compute.h
gmmsq_weight_quant_cube_compute_tools.h
gmmsq_weight_quant_vec_compute.h
gmmsq_mx_a8w4_vf.h
```

9.2 编译命令：
```bash
bash build.sh --arch=arch35
```

9.3 检查项：
- 所有 `.h` 文件编译通过
- 模板实例化正确
- 无未定义符号
- 无 UB 空间超限警告

**验证**：`build.sh` 返回 exit code 0。

---

### Step 10: 精度测试

**工作内容**：

10.1 编写测试用例：
- 小规模测试：M=4, K=64, N=128, E=2
- 中规模测试：M=256, K=512, N=256, E=4
- 大规模测试：M=1024, K=4096, N=512, E=8

10.2 对比基准：
- 使用 PyTorch 实现等价计算作为 ground truth
- 对比 GMMSQ 输出与 ground truth 的误差

10.3 精度标准：
- FP8_E4M3FN 输出：相对误差 < 1e-3（考虑量化误差）
- E8M0 scale 输出：与 ground truth 一致

**验证**：所有测试用例通过。

---

## 3. 关键技术点

### 3.1 N 轴分半计算策略

```
问题: GMM 输出 Y[M, N]，SwiGLU 需要 Y[:, 0:N/2] 和 Y[:, N/2:N]
      如果 BasicBlock 取 baseN=256，则 Y[M, 256] 占用 256*256*4 = 256KB > UB 空间

方案: 每个 BasicBlock 取 baseN=128，构成 256 的基本块
      - 前 128 个 N 元素 → Swish 分支
      - 后 128 个 N 元素 → Gate 分支
      - 输出 128 个 N/2 元素

N 轴切分:
  nBlkNum = CeilDivide(nSize, 256)  // 256 = 128 Swish + 128 Gate
  nL1Size = 256（每个 BasicBlock 的 N 分块大小）
  输出 N 偏移 = nOffset / 2
```

### 3.2 SwiGLU 的 VF 实现要点

```
输入: yF32[M, 256]，其中 yF32[:, 0:128] 为 Swish 分支，yF32[:, 128:256] 为 Gate 分支
输出: yF32[M, 128]（原地覆盖前 128 个元素）

算法:
  for m in 0..M-1:
    for n in 0..127:
      swish_val = yF32[m, n]
      gate_val  = yF32[m, n + 128]
      // Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
      sigmoid = 1.0 / (1.0 + exp(-swish_val))
      result = swish_val * sigmoid * gate_val
      yF32[m, n] = result

VF 实现注意事项:
  - 使用 Exp 指令计算 exp(-x)
  - 使用 Div 指令计算 1/(1+exp(-x))
  - 注意数值稳定性：x 很大时 sigmoid→1，x 很小时 sigmoid→0
  - 可以使用查表法加速 sigmoid 计算
```

### 3.3 MX Quant 的 VF 实现要点

```
ComputeMaxExp:
  输入: yF32[M, N/2]
  输出: maxB16[M, ceil(N/2/32)]
  算法: 对每 32 个元素，取 abs() 后提取指数位，求最大值

ComputeScale:
  输入: maxB16[M, ceil(N/2/32)]
  输出: scaleE8M0[M, ceil(N/2/32)]
  算法: shared_exp = round(log2(max_val)) - 8
        scaleE8M0 = shared_exp（E8M0 格式就是纯指数）

ComputeDataForQuantTargetFp8:
  输入: yF32[M, N/2], scaleE8M0[M, ceil(N/2/32)]
  输出: yFp8[M, N/2]
  算法: yF32[m, n] /= 2^scaleE8M0[m, n/32]
        量化到 FP8_E4M3FN（舍入到最近偶数）
```

### 3.4 UB 空间优化

```
复用策略:
  1. yF32Buffer_ 复用 weightHighBit_ 空间（128KB）
     - weightHighBit 在 PhaseB 时已全部搬运到 L1，UB 空间可安全复用
     - 注意: 需要确保 MTE3 搬运 weightHighBit→L1 全部完成后再复用

  2. yFp8Buffer_ 复用 weightLowBit_ 空间（前 32KB）
     - weightLowBit 在 PhaseB 时已不再使用
     - yFp8 只需要 32KB（M=256, N/2=128, FP8=1B → 256*128*1 = 32KB）

  3. maxB16Buffer_ 和 scaleE8M0Buffer_ 使用独立空间
     - maxB16: M=256, N/2=128, group=32 → 256*4*2 = 2KB
     - scaleE8M0: M=256, N/2=128, group=32 → 256*4*1 = 1KB（实际分配 16KB 用于 scatter 写出优化）
```

---

## 4. 风险与注意事项

### 4.1 UB 空间风险

| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| yF32Buffer_ 复用 weightHighBit_ 时未完成 MTE3 搬运 | 🔴 高 | 在复用前确保 `WaitFlag<MTE3_V>` 全部完成 |
| yFp8Buffer_ 复用 weightLowBit_ 时 MTE2 仍在搬运 | 🟡 中 | PhaseB 阶段 weightLowBit 已无 MTE2 操作 |
| M 较大时 yF32 超出 128KB | 🟡 中 | 限制 M 的 L1 分块大小 ≤ 256 |

### 4.2 N/2 对齐要求

| 约束 | 说明 |
|------|------|
| N ≥ 128 | N 最小值 128（保证 N/2 ≥ 64） |
| N % 128 == 0 | N 必须 128 对齐（保证 N/2 64 对齐） |
| 尾块处理 | N 轴尾块按 N/2 计算尾块，需要特殊处理 |

### 4.3 性能瓶颈点

| 瓶颈 | 说明 | 优化方向 |
|------|------|---------|
| MX Quant scatter 写出 | N/2 方向每 128 个元素一个 core，搬运粒度小 | 后续可改为 singleCore 全负责 N 方向 |
| SwiGLU VF 计算 | Exp 和 Div 指令延迟较高 | 使用查表法加速 sigmoid |
| 同步开销 | PhaseB 新增多个 VF 计算步骤 | 尽量合并 VF 计算，减少同步点 |

### 4.4 兼容性风险

| 风险 | 说明 |
|------|------|
| 与现有 CGMCT 框架路径共存 | apt.cpp 中通过 `V310_GMM_SQ_ANTI_QUANT` 宏隔离，互不影响 |
| 与 A8W8/A8W4 等其他场景共存 | MxA8W4 场景通过 Tiling 层场景识别独立分发 |
| 与 GMMFR 共享 common 文件 | common/ 目录文件直接复用，注意不要修改 GMMFR 依赖的接口 |

---

## 5. 实现 Checklist

### Phase 1: 基础设施（Step 1-2）

- [ ] Step 1.1: 新增 `gmmsq_weight_quant_tiling.h` — TilingData 结构体
- [ ] Step 1.2: 新增 `gmmsq_weight_quant_tiling.cpp` — 场景识别 + 校验
- [ ] Step 1.3: 新增 `gmmsq_weight_quant_tiling_key.h` — Tiling Key 定义
- [ ] Step 2.1: 修改 `apt.cpp` — 新增 `V310_GMM_SQ_ANTI_QUANT` 宏
- [ ] Step 2.2: 修改 `apt.cpp` — 新增 Controller 调用分支
- [ ] **编译验证 Phase 1**

### Phase 2: Controller + BasicBlock（Step 3-4）

- [ ] Step 3.1: 新增 `gmmsq_weight_quant_resplit_controller.h` — Controller 类
- [ ] Step 3.2: 实现 `Init()` — 去除 Routing 参数
- [ ] Step 3.3: 实现 `Process()` — 去除 InitAtomicGm
- [ ] Step 3.4: 实现 `UpdateGmAddr()` — 输出 N/2 + yScale
- [ ] Step 3.5: 实现 `SplitNByMultiCore()` — N 轴切分适配
- [ ] Step 4.1: 新增 `gmmsq_weight_quant_vcv_basic_block.h` — BasicBlock 类
- [ ] Step 4.2: 实现 `Init()` — UB Buffer 重新分配
- [ ] Step 4.3: 实现 `IterateNzNkWithAiv()` — PhaseB 改为 SwiGLU+Quant
- [ ] Step 4.4: 实现 `End()` — 后处理逻辑调整
- [ ] **编译验证 Phase 2**

### Phase 3: VecCompute + CubeCompute（Step 5-6）

- [ ] Step 5.1: 新增 `gmmsq_weight_quant_vec_compute.h` — VecCompute 类
- [ ] Step 5.2: 保留反量化相关函数
- [ ] Step 5.3: 新增 SwiGLU/Quant 接口函数
- [ ] Step 5.4: 实现 MTE3 写出函数
- [ ] Step 6.1: 新增 `gmmsq_weight_quant_cube_compute.h` — CubeCompute 类
- [ ] Step 6.2: 新增 `gmmsq_weight_quant_cube_compute_tools.h` — 工具函数
- [ ] **编译验证 Phase 3**

### Phase 4: VF 实现（Step 7）

- [ ] Step 7.1: 新增 `gmmsq_mx_a8w4_vf.h` — VF 函数
- [ ] Step 7.2: 保留反量化 VF（`FrAntiQuantMxA8W4NzNkVf`）
- [ ] Step 7.3: 新增 SwiGLU VF（`FrSwiGLUVf`）— 由其他团队实现
- [ ] Step 7.4: 新增 ComputeMaxExp VF（`FrComputeMaxExpVf`）— 由其他团队实现
- [ ] Step 7.5: 新增 ComputeScale VF（`FrComputeScaleVf`）— 由其他团队实现
- [ ] Step 7.6: 新增 QuantizeToFp8 VF（`FrComputeDataForQuantTargetFp8Vf`）— 由其他团队实现
- [ ] **编译验证 Phase 4**

### Phase 5: Buffer 调整 + 编译（Step 8-9）

- [ ] Step 8.1: UB Buffer 布局实现（复用策略）
- [ ] Step 8.2: 更新 CMakeLists.txt
- [ ] Step 9.1: 完整编译
- [ ] Step 9.2: 解决编译错误和警告
- [ ] **编译通过**

### Phase 6: 测试（Step 10）

- [ ] Step 10.1: 编写小规模测试用例
- [ ] Step 10.2: 编写中规模测试用例
- [ ] Step 10.3: 编写大规模测试用例
- [ ] Step 10.4: 精度对比验证
- [ ] Step 10.5: 性能测试
- [ ] **所有测试通过**

---

## 6. 附录：关键接口签名汇总

### Controller
```cpp
template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType,
          typename perTokenScaleType, typename biasType, typename yType, typename yScaleType,
          bool hasBias, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
class GMMSQWeightQuantResplitController {
    void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR bias,
              GM_ADDR groupList, GM_ADDR perTokenScale,
              GM_ADDR y, GM_ADDR yScale,
              const GMMSQWeightQuantTilingData *tiling);
    void Process();
};
```

### BasicBlock
```cpp
template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType,
          typename perTokenScaleType, typename biasType, typename yType, typename yScaleType,
          bool hasBias, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
class GMMSQWeightQuantVcvBasicBlock {
    void Init(uint64_t antiQuantGroupSize, __gm__ yType *y, __gm__ yScaleType *yScale);
    void ComputeBasicBlock(const BasicBlockOffsetParam &cur, const BasicBlockOffsetParam &last);
    void End(const BasicBlockOffsetParam &last);
};
```

### VecCompute
```cpp
template <typename xType, typename wType, typename biasType, typename yType, typename yScaleType,
          bool hasBias, const VecAntiQuantConfig &vecConfig>
class GMMSQWeightQuantVecCompute {
    void Init(__gm__ yType *y, __gm__ yScaleType *yScale);
    // 保留的反量化函数...
    // 新增的后处理函数:
    void SwiGLU(const LocalTensor<float> &yF32, const BasicBlockOffsetParam &param);
    void ComputeMaxExp(const LocalTensor<float> &yF32, const LocalTensor<half> &maxB16, const BasicBlockOffsetParam &param);
    void ComputeScale(const LocalTensor<half> &maxB16, const LocalTensor<fp8_e8m0_t> &scaleE8M0, const BasicBlockOffsetParam &param);
    void ComputeDataForQuantTargetFp8(const LocalTensor<float> &yF32, const LocalTensor<fp8_e8m0_t> &scaleE8M0, const LocalTensor<fp8_e4m3fn_t> &yFp8, const BasicBlockOffsetParam &param);
    void CopyOutputFromUb2Gm(uint64_t mRealSize, const LocalTensor<fp8_e4m3fn_t> &yFp8, const BasicBlockOffsetParam &param);
    void CopyScaleFromUb2Gm(uint64_t mRealSize, const LocalTensor<fp8_e8m0_t> &scaleE8M0, const BasicBlockOffsetParam &param);
};
```

### CubeCompute
```cpp
template <typename xType, typename biasType, typename antiQuantScaleType,
          typename perTokenScaleType, typename yType, bool hasBias, const WqmmConfig &wqmmConfig>
class GMMSQWeightQuantCubeCompute {
    // 与 GMMFR 完全相同，仅 GetTensorC 目标地址变化
    void GetTensorC(LocalTensor<yType> &yUb, const BasicBlockOffsetParam &param);
};
```