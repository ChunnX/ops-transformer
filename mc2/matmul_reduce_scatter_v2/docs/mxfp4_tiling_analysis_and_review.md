# MXFP4 Tiling 切分分析与 PR #7195 检视报告

## 目录

1. [reduce_scatter_fit_balance_tiling FP8 vs FP4 切分差异分析](#一reduce_scatter_fit_balance_tiling-fp8-vs-fp4-切分差异分析)
2. [PR #7195 检视报告](#二pr-7195-检视报告)
3. [检视意见修改记录](#三检视意见修改记录)

---

## 一、reduce_scatter_fit_balance_tiling FP8 vs FP4 切分差异分析

### 1. 概述

分析 `reduce_scatter_fit_balance_tiling.cpp` 中 `MMReduceScatterFitBalanceTiling` 类在输入数据类型为 **FP8**（float8_e5m2/float8_e4m3fn）和 **FP4**（float4_e2m1）时的 tiling 切分差异。假设除输入 dtype 外，所有其它属性（M/N/K shape、rankDim、输出 dtype 等）完全一致。

### 2. 根因：Dtype Size 映射缺失

#### 2.1 GetDataTypeSize 映射表

`mc2_tiling_utils.cpp:104-118` 中的 `DATA_TYPE_SIZE_MAP`：

| ge::DataType | Size (bytes) |
|---|---|
| DT_BF16 | 2 |
| DT_FLOAT16 | 2 |
| DT_FLOAT | 4 |
| DT_HIFLOAT8 | 1 |
| DT_FLOAT8_E4M3FN | 1 |
| DT_FLOAT8_E5M2 | 1 |
| **DT_FLOAT4_E2M1** | **未注册，返回 0** |

**关键发现**：`DT_FLOAT4_E2M1` 未在 `GetDataTypeSize` 映射表中注册，导致 FP4 的 `inputDtypeSize = 0`。

#### 2.2 初始参数差异

| 参数 | FP8 | FP4 | 来源 |
|---|---|---|---|
| `inputDtypeSize` | 1 | **0** | `GetDataTypeSize()` |
| `inMatrixADtypeSize` | 1 | **0** | `matmul_performance.h:100` |
| `inMatrixBDtypeSize` | 1 | **0** | `matmul_performance.h:101` |
| `outMatrixCDtypeSize` | 2 | 2 | 相同（输出为 bf16/fp16） |
| `calcType_` | `QUANT` | **`FP16`（默认值）** | `SetCalcType()` 未匹配 FP4 |
| `isQuantMatmul_` | **true** | **false** | `(A==1 && B==1)` |
| `computesPerCycle` | **8192** | **4096**（默认值） | `GetCalcTypeString()` 查表 |

#### 2.3 CalcType 判定差异

`matmul_performance.h:78-92` 中 `SetCalcType()`：

```cpp
// SOC950 下只匹配了 fp8 类型，fp4 不在条件中
if ((socType == SOC950) &&
    (aType == DT_HIFLOAT8 || aType == DT_FLOAT8_E4M3FN || aType == DT_FLOAT8_E5M2)) {
    calcType_ = QUANT;  // FP8 命中
}
// FP4 不命中任何分支，保持默认 FP16
```

#### 2.4 性能参数查表差异

`GetCalcTypeString()` 生成的 key 格式：`socType_calcType_A_dtypeSize_B_dtypeSize_C_dtypeSize`

| | Key | CUBE_CALC_PER_CYCLE_MAP | L2_PARAMETER_MAP |
|---|---|---|---|
| FP8 | `"4_1_1_1_2"` | HIT → 8192 | HIT → {80, 128, 0.95, 0.85, 0.75} |
| FP4 | `"4_0_0_0_2"` | MISS → 4096（默认） | MISS → 默认 {128, 192, 0.85, 0.75, 0.65} |

### 3. 四阶段切分逻辑差异分析

#### 3.1 阶段一：EstimateMMCommTime()

```
GetTiling() → EstimateMMCommTime() → SetShortTileLen() → SetLongTileLen() → AdjustLongShortTileLen()
```

##### 3.1.1 Cube 利用率（FindCubeUtil）

`MatmulPerformanceArch35::FindCubeUtil()` 仅依赖 M、N、K 和 rankTileNum，**与 dtype 无关**。
- **FP8 和 FP4 的 cubeUtil_ 相同**

##### 3.1.2 Matmul 梯度（GetMatmulGradient）

```cpp
matmulGradient_ = (tmpN * tmpK) / (computesPerCycle * cyclePerMicroSec * coreNum * cubeUtil_)
```

| | computesPerCycle | matmulGradient_ |
|---|---|---|
| FP8 | 8192 | **较小**（基准值） |
| FP4 | 4096 | **约为 FP8 的 2 倍** |

##### 3.1.3 计算耗时 vs 通信耗时

```cpp
totalMatmulTime = MatmulTime(mValue, rankDim) = max(baseM, mValue) * rankTileNum * matmulGradient_
totalTpTime = CommTime(mValue)  // 使用 outMatrixCDtypeSize=2，两者相同
```

| | totalMatmulTime | totalTpTime | ratioCalcComm_ | shortTileAtBack |
|---|---|---|---|---|
| FP8 | 较小 | 相同 | 较小 | 更可能 `false`（计算快于通信） |
| FP4 | **约 2 倍** | 相同 | 较大 | 更可能 `true`（计算慢于通信） |

**影响**：`shortTileAtBack` 决定了后续 `SetLongTileLen()` 中配平方向（以计算时间推通信 or 以通信时间推计算）。

#### 3.2 阶段二：SetShortTileLen()

```cpp
uint64_t l2UseSize = M * K * inMatrixADtypeSize + K * N * inMatrixBDtypeSize;
isLargerThanL2Cache_ = l2UseSize > L2_CACHE_SIZE;  // 128MB
```

| | l2UseSize | isLargerThanL2Cache_ |
|---|---|---|
| FP8 | `K*(M+N)` 字节 | 大 shape 下可能 `true` |
| FP4 | **0**（因为 dtypeSize=0） | **始终 `false`** |

##### L2 Cache 分支

```cpp
if (isLargerThanL2Cache_ && mValue > minLen * 4) {
    tilingM_.SetMinLenByMax(minLen * 2);  // minLen 翻倍
}
```

| | minLen（shortTileLen） |
|---|---|
| FP8（大 shape） | 可能翻倍至 `2 * baseM = 512` |
| FP4 | 始终为 `baseM = 256` |

**影响**：FP4 的 shortTileLen 偏小，可能导致切分更细、通信轮次更多。

#### 3.3 阶段三：SetLongTileLen()

##### 配平方向差异

```cpp
if (shortTileAtBack) {  // FP4 更可能走此分支
    targetTime = MatmulTime(shortTileLen, rankTileNum);
    longTileLen = InverseCommTime(targetTime);  // 以计算时间反推通信块大小
} else {                // FP8 更可能走此分支
    targetTime = CommTime(shortTileLen);
    longTileLen = InverseMatmulTime(targetTime, rankTileNum);  // 以通信时间反推计算块大小
}
```

| | 配平方向 | longTileLen 趋势 |
|---|---|---|
| FP8 | 通信→计算（CommTime 推 InverseMatmulTime） | 取决于通信时间 |
| FP4 | 计算→通信（MatmulTime 推 InverseCommTime） | 因 matmul 耗时大，反推出的 longTileLen 可能更大 |

##### L2 Cache 调整分支

```cpp
if (isLargerThanL2Cache_ && mValue % shortTileLen == 0 && nValue >= 2*mValue*rankDim) {
    longTileLen = shortTileLen;  // 强制长短块一致
}
```
- FP8：可能触发
- FP4：**永不触发**（`isLargerThanL2Cache_ = false`）

##### 量化 Matmul 调整分支

```cpp
else if (isQuantMatmul_ && ratioCalcComm_ > 2 && kValue < 2048 && mValue % shortTileLen == 0) {
    longTileLen = shortTileLen;  // 强制长短块一致
}
```
- FP8（`isQuantMatmul_ = true`）：在 K 较小且计算通信比失衡时触发
- FP4（`isQuantMatmul_ = false`）：**永不触发**

**影响**：FP8 在特定条件下会将 longTileLen 退化为 shortTileLen（均匀切分），FP4 不会。

#### 3.4 阶段四：AdjustLongShortTileLen()

```cpp
bool goodLinearityShape = (kValue * nValue >= 32 * 1MB);
tilingM_.FitTileLengthDiscrete(false, goodLinearityShape);
if (shortTileLen == longTileLen) {
    shortTileLen = 0; numShortTile = 0; numLongTile++;  // 合并为统一块
}
```

- `goodLinearityShape` 与 dtype 无关，两者相同
- 但由于前面阶段的差异，FP8 和 FP4 进入此阶段时的 `shortTileLen` 和 `longTileLen` 已不同
- FP8 更可能触发合并（因为量化调整分支可能使 long==short）

### 4. 差异总结表

| 切分维度 | FP8 (float8_e5m2/e4m3fn) | FP4 (float4_e2m1) | 差异根因 |
|---|---|---|---|
| `inputDtypeSize` | 1 | **0** | `GetDataTypeSize` 未注册 FP4 |
| `calcType_` | QUANT | **FP16（默认）** | `SetCalcType` 未匹配 FP4 |
| `isQuantMatmul_` | true | **false** | `inMatrixADtypeSize != 1` |
| `computesPerCycle` | 8192 | **4096** | CalcType key 查表 MISS |
| `matmulGradient_` | 较小 | **约 2x** | `computesPerCycle` 减半 |
| `totalMatmulTime` | 较小 | **约 2x** | 梯度翻倍 |
| `shortTileAtBack` | 更可能 false | 更可能 **true** | matmul/comm 比值反转 |
| `l2UseSize` | `K*(M+N)` | **0** | `dtypeSize=0` |
| `isLargerThanL2Cache_` | 可能 true | **始终 false** | `l2UseSize=0` |
| `shortTileLen` | 可能 512（大 shape） | **始终 256** | L2 分支不触发 |
| 配平方向 | Comm→Matmul | **Matmul→Comm** | `shortTileAtBack` 不同 |
| 量化调整分支 | 可能触发 | **永不触发** | `isQuantMatmul_=false` |
| longTileLen=shortTileLen | 可能发生 | **不太可能** | 两个调整分支均不触发 |

### 5. `isQuantMatmul_` 差异解释

`isQuantMatmul_` 的判断条件：

```cpp
// reduce_scatter_fit_balance_tiling.h:32
isQuantMatmul_ = (mmInfo_.inMatrixADtypeSize == 1) && (mmInfo_.inMatrixBDtypeSize == 1);
```

而 `inMatrixADtypeSize` 来自 `GetDataTypeSize` 的返回值：

```cpp
// matmul_performance.h:100-101
mmShapeInfo_.inMatrixADtypeSize = args.inputDtypeSize;  // ← GetDataTypeSize 的返回值
mmShapeInfo_.inMatrixBDtypeSize = mmShapeInfo_.inMatrixADtypeSize;
```

```cpp
// mc2_tiling_utils.cpp:105-108
DATA_TYPE_SIZE_MAP = {
    {DT_FLOAT8_E4M3FN, 1},  // FP8 → 1 → isQuantMatmul_ = true
    {DT_FLOAT8_E5M2, 1},    // FP8 → 1 → isQuantMatmul_ = true
    // DT_FLOAT4_E2M1 未注册 → 返回 0 → isQuantMatmul_ = false
};
```

**本质原因**：`isQuantMatmul_` 的语义是"输入为 1 字节量化类型"，FP8 恰好是 1 字节所以命中，FP4 不在映射表中返回 0 所以不命中。这个判断条件本身是一个**间接推断**——用 dtype size 等于 1 来近似判断"是否为量化 matmul"，而不是直接检查 `calcType_ == QUANT`。

### 6. `inputDtypeSize` 在 QuantBmmReduceScatter 路径中的精确影响链

FP4 时 `inputDtypeSize=0`，通过以下链路影响切分结果：

```
matmul_reduce_scatter_tiling_base.cpp:471
  args_.inputDtypeSize = GetDataTypeSize(opName_, DT_FLOAT4_E2M1) = 0
    │
    ▼
matmul_performance.h:100-101  (MatmulPerformanceModel 构造函数)
  mmShapeInfo_.inMatrixADtypeSize = 0
  mmShapeInfo_.inMatrixBDtypeSize = 0
    │
    ├─→ matmul_performance.h:117-121  GetCalcTypeString()
    │     key = "4_0_0_0_2"  (socType=4, calcType=FP16, A=0, B=0, C=2)
    │     CUBE_CALC_PER_CYCLE_MAP 查表 MISS → computesPerCycle = 4096 (默认)
    │     ★ FP8 的 key="4_1_1_1_2" → HIT → computesPerCycle = 8192
    │
    ▼
reduce_scatter_fit_balance_tiling.cpp 四阶段切分：
```

#### 阶段一：EstimateMMCommTime() (line 22-42)

```cpp
matmulPerf_.GetMatmulGradient();
// matmulGradient_ = (tmpN * tmpK) / (computesPerCycle * cyclePerMicroSec * coreNum * cubeUtil_)
//                 FP4: ÷ 4096    FP8: ÷ 8192
// → FP4 的 matmulGradient_ 约为 FP8 的 2 倍

totalMatmulTime = MatmulTime(mValue, rankDim_) = max(baseM, mValue) * rankTileNum * matmulGradient_
// → FP4 的 totalMatmulTime 约为 FP8 的 2 倍
// → FP4 更容易触发 shortTileAtBack = true (认为计算比通信慢)
```

#### 阶段二：SetShortTileLen() (line 44-58)

```cpp
l2UseSize = M * K * inMatrixADtypeSize + K * N * inMatrixBDtypeSize
//         FP4: M*K*0 + K*N*0 = 0      FP8: K*(M+N)
// → FP4: isLargerThanL2Cache_ = false (永远)
// → FP4: shortTileLen 不会翻倍，始终 = baseM = 256
```

#### 阶段三：SetLongTileLen() (line 60-82)

```cpp
// FP4 更可能走 shortTileAtBack=true 分支：
targetTime = MatmulTime(shortTileLen, rankTileNum)  // 因 gradient 大，targetTime 大
longTileLen = InverseCommTime(targetTime)            // 反推出更大的通信块
// → longTileLen 偏大 → 切分更少

// 量化均匀切分分支 (line 78-81):
if (isQuantMatmul_ && ...)  // FP4: isQuantMatmul_=false，永不触发
    longTileLen = shortTileLen  // FP8 可能触发此优化使切分更均匀
```

#### 阶段四：AdjustLongShortTileLen() (line 84-94)

此时 longTileLen 和 shortTileLen 已确定，FP4 的 longTileLen 更大，切分轮次更少。

### 7. 根因总结

| 参数 | FP8 | FP4 | 影响 |
|---|---|---|---|
| `computesPerCycle` | 8192 | **4096** | matmulGradient 翻倍 → matmul 时间高估 2x → longTileLen 偏大 → **切分更少** |
| `inMatrixADtypeSize` | 1 | **0** | L2 估算失效 + isQuantMatmul=false → 均匀切分优化不触发 |

**根因是 `computesPerCycle` 被低估 2 倍**，导致性能模型认为 matmul 很慢，从而选择更大的 tile（更少切分轮次）。

### 8. 修改方案

只需修改 **1 个文件**：`reduce_scatter_fit_balance_tiling.h` 的构造函数，参照 `all_gather_matmul_v2` 的做法：

```cpp
// reduce_scatter_fit_balance_tiling.h
constexpr double RS_FP4_COMPUTE_PER_CYCLE_RATIO = 4;

class MMReduceScatterFitBalanceTiling : public Mc2FitBasedBalanceTiling
{
public:
    explicit MMReduceScatterFitBalanceTiling(const mc2tiling::TilingArgs& args, KernelType kernelType,
        TopoType topoType = TopoType::STANDARD_CARD, SocVersion socVersion = SocVersion::SOC950) :
        Mc2FitBasedBalanceTiling(args, kernelType, topoType, socVersion)
    {
        commPerf_.SetCommShapeLen(args.nValue);
        commPerf_.SetCommDTypeSize(mmInfo_.outMatrixCDtypeSize);
        if (args.geAType == ge::DataType::DT_FLOAT4_E2M1) {
            matmulPerf_.mmShapeInfo_.inMatrixADtypeSize = 1;
            matmulPerf_.mmShapeInfo_.inMatrixBDtypeSize = 1;
            mmInfo_.inMatrixADtypeSize = 1;
            mmInfo_.inMatrixBDtypeSize = 1;
            matmulPerf_.calcType_ = MatmulCalcType::QUANT;
            matmulPerf_.mmShapeInfo_.computesPerCycle =
                RS_FP4_COMPUTE_PER_CYCLE_RATIO * MatmulPerformance::COMPUTES_PER_CYCLE;
            mmInfo_.computesPerCycle = matmulPerf_.mmShapeInfo_.computesPerCycle;
        }
        tilingM_.SetMinLenByMax(matmulPerf_.GetBaseM());
        tilingM_.SetAlignLength(matmulPerf_.GetBaseM());
        isQuantMatmul_ = (mmInfo_.inMatrixADtypeSize == 1) && (mmInfo_.inMatrixBDtypeSize == 1);
    }
    // ...
};
```

修复效果：

| 参数 | 修复前 (FP4) | 修复后 (FP4) | FP8 参考 |
|---|---|---|---|
| `inMatrixADtypeSize` | 0 | **1** | 1 |
| `calcType_` | FP16 | **QUANT** | QUANT |
| `computesPerCycle` | 4096 | **16384** | 8192 |
| `isQuantMatmul_` | false | **true** | true |
| `l2UseSize` | 0 | **K*(M+N)** | K*(M+N) |

FP4 的 `computesPerCycle = 4 * 4096 = 16384`（FP4 理论算力是 FP8 的 2 倍），使 `matmulGradient_` 回归合理值，切分轮次与 FP8 对齐。

### 9. 其他算子 MXFP4 适配情况

| 算子 | inputDtypeSize | inMatrixA/B | calcType_ | computesPerCycle | 结论 |
|---|---|---|---|---|---|
| **all_gather_matmul_v2** | 覆盖为 1 | 覆盖为 1 | 覆盖 QUANT | 覆盖 16384 | **完整适配**（唯一标杆） |
| **allto_all_matmul** | 硬编码 1 | 间接覆盖 | 未覆盖 | 未覆盖 | 用独立拟合公式，不走公共模型 |
| **matmul_allto_all** | **GetDataTypeSize→0** | 0 | 未覆盖 | 未覆盖 | **存在相同问题** |
| **matmul_all_reduce** | `GetSizeByDataType` | 未覆盖 | 未覆盖 | 未覆盖 | 用独立路径，不走公式化 tiling |
| **all_gather_matmul v1** | — | — | — | — | 不支持 FP4 |

---

## 二、PR #7195 检视报告

**PR 地址**：https://gitcode.com/cann/ops-transformer/pull/7195

**PR 概述**：为 `matmul_reduce_scatter_v2` 算子新增 MXFP4（FLOAT4_E2M1）量化场景支持。涉及 9 个文件，75 行新增，36 行删除。

### 检视统计

| 统计项 | 数值 |
|--------|------|
| 检视文件数 | 9 |
| 发现问题总数 | **6 个** |
| CRITICAL | **2 个** |
| MEDIUM | **3 个** |
| LOW | **1 个** |

### CRITICAL 问题（必须修复）

| ID | 问题 | 位置 |
|----|------|------|
| **ISSUE-001** | `GetDataTypeSize` 未注册 `DT_FLOAT4_E2M1`，FP4 返回 `inputDtypeSize=0`，导致 L2 Cache 估算失效、`calcType_` 回退默认值、`computesPerCycle` 用 4096 而非 8192 — **整个 FP4 tiling 策略基于错误参数** | `mc2_tiling_utils.cpp:105` |
| **ISSUE-002** | `ConvertGeTypeToMmType` 未注册 `DT_FLOAT4_E2M1`，FP4 的 `args_.aType` 转换为 `DT_MAX` — **matmul tiling 库匹配失败** | `mc2_tiling_utils.cpp:62` |

> 这两个是已知基础设施缺陷（不在 PR 修改范围内），但 PR 新增的 FP4 功能完全依赖这两个映射，不修复则 FP4 场景无法正确运行。

### MEDIUM 问题（建议修复）

| ID | 问题 | 位置 |
|----|------|------|
| **ISSUE-003** | 文档声明"k必须是偶数"但代码中无对应校验 — FP4 奇数 k 可能导致越界 | `quant_bmm_reduce_scatter_tiling.cpp` MxfpSceneParamCheck |
| **ISSUE-004** | `isMxfp4_` 仅检查 `geAType` 不检查 `geBType` — `geAType=FP8, geBType=FP4` 漏检 | `quant_bmm_reduce_scatter_tiling.cpp:438` |
| **ISSUE-005** | HCCL 限制改为 63 但注释和错误信息仍写 "16" — 误导维护者 | `matmul_reduce_scatter_tiling_base.cpp:109,122` |

### 通过项

- OpDef dtype 列表对齐正确（所有 input/output 均为 32 条目）
- Tiling key `INPUT_TYPE_IS_FP4=2` 注册完整
- HCCL 限制 16→63 数值合理
- API 层 FP4 类型一致性校验完整
- Kernel 条件编译宏正确
- `isMxfp4_` 初始化符合规范

---

## 三、检视意见修改记录

### 修改文件清单

| 检视问题 | 修改文件 | 修改内容 |
|---|---|---|
| **ISSUE-001** (CRITICAL) | `mc2/common/.../mc2_tiling_utils.cpp:110`<br>`mc2/common/.../new_mc2_tiling_utils.cpp:32` | `GetDataTypeSize` / `NewGetDataTypeSize` 注册 `DT_FLOAT4_E2M1 → 1` |
| **ISSUE-002** (CRITICAL) | `mc2/common/.../mc2_tiling_utils.cpp:69,92` | `ConvertGeTypeToMmType` / `ConvertMmTypeToGeType` 双向注册 `DT_FLOAT4_E2M1` |
| **ISSUE-003** (MEDIUM) | `quant_bmm_reduce_scatter_tiling.cpp:296-299` | `MxfpSceneParamCheck` 中 MXFP4 场景新增 `kValue % 2 == 0` 校验 |
| **ISSUE-004** (MEDIUM) | `quant_bmm_reduce_scatter_tiling.cpp:442-443` | `SetScene()` 中 `isMxfp4_` 改为同时检查 `geAType` 和 `geBType` |
| **ISSUE-005** (MEDIUM) | `matmul_reduce_scatter_tiling_base.cpp:107,109,122` | 注释和错误信息中 "16" 统一更新为 "63" |
| **FP4 性能模型** | `reduce_scatter_fit_balance_tiling.h:32-41` | 参照 `all_gather_fit_balance_tiling.h` 模式：FP4 时强制 `inMatrixADtypeSize=1`、`calcType_=QUANT`、`computesPerCycle=16384` |

### 修改详情

#### ISSUE-001：GetDataTypeSize 注册 DT_FLOAT4_E2M1

**mc2_tiling_utils.cpp**:
```cpp
uint64_t GetDataTypeSize(const std::string &opName, ge::DataType type) {
  static const std::map<ge::DataType, int64_t> DATA_TYPE_SIZE_MAP = {
      {ge::DT_BF16, 2},     {ge::DT_FLOAT16, 2},       {ge::DT_FLOAT, 4},
      {ge::DT_HIFLOAT8, 1}, {ge::DT_FLOAT8_E4M3FN, 1}, {ge::DT_FLOAT8_E5M2, 1},
      {ge::DT_FLOAT4_E2M1, 1},  // 新增
  };
```

**new_mc2_tiling_utils.cpp**:
```cpp
uint64_t NewGetDataTypeSize(const std::string &opName, ge::DataType type) {
  static const std::map<ge::DataType, int64_t> DATA_TYPE_SIZE_MAP = {
      {ge::DT_BF16, 2},     {ge::DT_FLOAT16, 2},       {ge::DT_FLOAT, 4},
      {ge::DT_HIFLOAT8, 1}, {ge::DT_FLOAT8_E4M3FN, 1}, {ge::DT_FLOAT8_E5M2, 1},
      {ge::DT_FLOAT4_E2M1, 1},  // 新增
  };
```

#### ISSUE-002：ConvertGeTypeToMmType / ConvertMmTypeToGeType 注册

**mc2_tiling_utils.cpp**:
```cpp
// ConvertGeTypeToMmType
static const std::map<ge::DataType, matmul_tiling::DataType> GE_TO_MM_MAP = {
    // ...
    {ge::DT_FLOAT8_E5M2, matmul_tiling::DataType::DT_FLOAT8_E5M2},
    {ge::DT_FLOAT4_E2M1, matmul_tiling::DataType::DT_FLOAT4_E2M1},  // 新增
};

// ConvertMmTypeToGeType
static const std::map<matmul_tiling::DataType, ge::DataType> MM_TO_GE_MAP = {
    // ...
    {matmul_tiling::DataType::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2},
    {matmul_tiling::DataType::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1},  // 新增
};
```

#### ISSUE-003：FP4 场景 k 必须为偶数校验

**quant_bmm_reduce_scatter_tiling.cpp** `MxfpSceneParamCheck`:
```cpp
if (isMxfp4_) {
    OP_TILING_CHECK(args_.geAType != args_.geBType, ...);
    // 新增：k 必须为偶数
    OP_TILING_CHECK(args_.kValue % EVEN_ALIGN != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "kValue",
                    std::to_string(args_.kValue).c_str(),
                    "In MXFP4 quant mode, kValue must be even"), return false);
}
```

#### ISSUE-004：isMxfp4_ 同时检查 geAType 和 geBType

**quant_bmm_reduce_scatter_tiling.cpp** `SetScene`:
```cpp
// 修改前：仅检查 geAType
if (args_.geAType == ge::DataType::DT_FLOAT4_E2M1) {

// 修改后：同时检查 geAType 和 geBType
if (args_.geAType == ge::DataType::DT_FLOAT4_E2M1 &&
    args_.geBType == ge::DataType::DT_FLOAT4_E2M1) {
```

#### ISSUE-005：HCCL 限制注释和错误信息更新

**matmul_reduce_scatter_tiling_base.cpp**:
```cpp
// 修改前：
// 1. The maximum number of communication attempts is limited to 16
// "The value of x1size must not exceed 256MB after splitting M into 16 parts"

// 修改后：
// 1. The maximum number of communication attempts is limited to 63
// "The value of x1size must not exceed 256MB after splitting M into 63 parts"
```

#### FP4 性能模型修复

**reduce_scatter_fit_balance_tiling.h**:
```cpp
constexpr double RS_FP4_COMPUTE_PER_CYCLE_RATIO = 4;

// 构造函数中新增 FP4 处理块
if (args.geAType == ge::DataType::DT_FLOAT4_E2M1) {
    matmulPerf_.mmShapeInfo_.inMatrixADtypeSize = 1;
    matmulPerf_.mmShapeInfo_.inMatrixBDtypeSize = 1;
    mmInfo_.inMatrixADtypeSize = 1;
    mmInfo_.inMatrixBDtypeSize = 1;
    matmulPerf_.calcType_ = MatmulCalcType::QUANT;
    matmulPerf_.mmShapeInfo_.computesPerCycle =
        RS_FP4_COMPUTE_PER_CYCLE_RATIO * MatmulPerformance::COMPUTES_PER_CYCLE;
    mmInfo_.computesPerCycle = matmulPerf_.mmShapeInfo_.computesPerCycle;
}
```

### 后续回退说明

ISSUE-001、ISSUE-002 和 FP4 性能模型修复在初步提交后被回退，原因是需要进一步分析 `inputDtypeSize` 在 QuantBmmReduceScatter 路径中的实际影响。

**分析结论**：
- `QuantBmmReduceScatterHelper`（滑动窗口切分）完全不依赖 `inputDtypeSize`
- `inputDtypeSize=0` 仅影响 `MMReduceScatterFitBalanceTiling`（M 维切分轮次决策）
- **根因是 `computesPerCycle` 被低估 2 倍**，导致 matmul 时间被高估，切分轮次偏少
- `matmul_allto_all` 算子也存在完全相同的问题

最终仅保留 `reduce_scatter_fit_balance_tiling.h` 的 FP4 性能模型修复（ISSUE-003/004/005 的修改保持不变）。
