# 03 Tiling 拦截校验与切分方案 — GMM A16W4 PerGroup ND

> 本文档描述 A16W4 INT4 weight ND pergroup 数据流的 host tiling 设计。切分策略按现有 A16W8 ND basic-controller 方案扩展，新增内容集中在 pergroup shape 校验、`groupSize` 推导和 tiling key 分发。
> 相关文档：[01 交付件清单](./01_deliverables.md)、[02 与硬件无关方案](./02_platform_independent_design.md)、[04 Kernel 方案](./04_kernel_design.md)

---

## 1. Tiling Data 设计

### 1.1 GMMWeightQuantParam 结构复用

`GMMWeightQuantParam` 已包含 A16W4 pergroup 所需字段，无需新增 tiling data 字段。

```cpp
struct GMMWeightQuantParam {
    uint32_t groupNum = 0;           // matmul 分组数（E）
    uint32_t coreNum = 0;            // AIC 核数
    uint64_t kSize = 0;              // K 轴大小
    uint64_t nSize = 0;              // N 轴大小（原始值）
    uint8_t singleX = 0;             // x 是否单 tensor
    uint8_t singleWeight = 0;        // weight 是否单 tensor
    uint8_t singleY = 0;             // y 是否单 tensor
    int8_t groupType = 0;            // -1=多多多, 0=单单单
    uint8_t groupListType = 0;       // 0=cumsum, 1=count
    uint8_t hasBias = 0;             // 是否有 bias
    uint8_t cubeNumBlocksN = 0;      // N 方向 cube 基本块数
    uint8_t reserved = 0;
    uint32_t groupSize = 0;          // ★ pergroup groupSize（已有字段）
    uint32_t mainBlockSize = 0;      // 尾块重切分用
    uint64_t mainBlockCount = 0;
    uint16_t firstTailBlockSize = 0;
    uint16_t secondTailBlockSize = 0;
    uint16_t firstTailBlockCount = 0;
    uint16_t secondTailBlockCount = 0;
};
```

关键字段：
- `groupSize`：A16W4 pergroup 写入 32/64/128/256。
- `groupNum`：单单单为 expert 数 `E`，多多多为 tensorList 长度。
- `singleX/singleWeight/singleY`、`groupType/groupListType`：沿用 A16W8 controller 的分支逻辑。
- `mList/kList/nList`：单单单仅使用 `kList[0]`、`nList[0]`，多多多逐 tensor 填充。

### 1.2 groupSize 推导

本数据流固定为 pergroup，`groupSize` 必须由 `antiquantScale` 的 K-group 维度推导，不能保留为 0。

单单单（`groupType=0`）：

```text
weight:           [E, *, *]
antiquantScale:   [E, G, N]
groupSize = K / G
```

多多多（`groupType=-1`）：

```text
weight[i]:          [*, *]
antiquantScale[i]:  [G_i, N_i]
groupSize_i = K_i / G_i
```

多多多所有 tensor 的 `groupSize_i` 必须一致，最终写入 `GMMWeightQuantParam.groupSize`。实现上建议先用第一个 scale 推导 `groupSize_`，随后遍历每个 tensor 做一致性校验。

```cpp
// 单单单
G = antiquantScaleShape.GetDim(1);
groupSize_ = kSize_ / G;

// 多多多
G_i = antiquantScaleShape[i].GetDim(0);
groupSize_i = kList_[i] / G_i;
```

```cpp
// antiquantScale shape 维度判断
if (antiquantScaleDimNum == 3) {
    // 3维: (E, G, N) → pergroup 场景（单单单）
    // 或 (G_i, N_i) 扩展为 3 维的情况
    groupNum = transB_ ? shape.GetDim(dim-1) : shape.GetDim(dim-2);
    groupSize_ = kSize_ / groupNum;
} else if (antiquantScaleDimNum == 2) {
    // 2维: (E, N) 或 (G_i, N_i) → perchannel 场景
    groupSize_ = 0;  // perchannel 不使用
}
```

**多多多场景**：antiquantScale shape 为 $(G_i, N_i)$（2 维），需要特殊处理：
- 当 `antiquantScaleDimNum == 2` 且 `IsA16W4ND()` 时，需要判断是否为 pergroup
- 判断依据：$G_i = k_i / gs$，若 $G_i > 1$ 则为 pergroup
- 或者直接检查 $k_i \bmod G_i == 0$ 且 $G_i \neq 1$

**⚠️ 适配点**：多多多场景下 antiquantScale 为 2 维 $(G_i, N_i)$，当前代码可能将其误判为 perchannel。需要新增判断逻辑：

```cpp
if (antiquantScaleDimNum == 2 && IsA16W4ND()) {
    // 多多多场景: scale shape = (G_i, N_i)
    uint64_t scaleDim0 = shape.GetDim(0);  // G_i
    uint64_t scaleDim1 = shape.GetDim(1);  // N_i
    if (scaleDim0 > 1 && kSize_ % scaleDim0 == 0) {
        // pergroup: G_i = k_i / gs
        groupSize_ = kSize_ / scaleDim0;
    } else {
        // perchannel: scale shape = (N_i,)
        groupSize_ = 0;
    }
}
```

### 1.3 GMMWeightQuantTilingData 结构

```cpp
struct GMMWeightQuantTilingData {
    GMMWeightQuantParam gmmWeightQuantParam;  // 自定义参数
    GMMArray gmmArray;                         // MKN 列表
    TCubeTiling mmTilingData;                  // matmul tiling 参数
};
```

`GMMArray` 保持现状：

```cpp
struct GMMArray {
    int32_t mList[128] = {0};
    int32_t kList[128] = {0};
    int32_t nList[128] = {0};
};
```

---

## 2. Tiling 拦截校验

### 2.1 场景识别

A16W4 ND 场景沿用现有 `IsA16W4ND()` 判断：

```cpp
bool IsA16W4ND() const
{
    return ge::GetSizeByDataType(xDType_) == B16_DATA_SIZE &&
           (weightDtype_ == ge::DT_INT4 || weightDtype_ == ge::DT_INT32) &&
           !weightNzFlag_;
}
```

本数据流额外要求 `groupSize_ > 0`，建议增加辅助判断：

```cpp
bool IsA16W4NDPergroup() const
{
    return IsA16W4ND() && groupSize_ > 0;
}
```

**区分 perchannel 和 pergroup 的方式**：
- perchannel：`groupSize_ == 0`（antiquantScale 为 2 维且 $G_i = 1$）
- pergroup：`groupSize_ > 0`（antiquantScale 为 3 维或 2 维但 $G_i > 1$）

### 2.2 基础约束

| 项目 | 约束 |
|------|------|
| x dtype | `FLOAT16` 或 `BFLOAT16` |
| weight dtype | `INT4`，兼容 host 侧 `INT32` 打包描述 |
| x/weight/out format | ND，weight 不走 NZ |
| x transpose | 不支持，`transA=false` |
| weight transpose | 支持 `transB=false/true` |
| 输入模式 | 仅支持单单单 `groupType=0` 和多多多 `groupType=-1` |
| group_size | 仅支持 32/64/128/256 |
| K 轴 | 每个 tensor 的 `K_i % groupSize == 0` |
| INT4 最后一维 | weight 最后一维必须为偶数 |

### 2.3 维度校验

单单单：

| Tensor | 维度 | shape |
|--------|------|-------|
| x | 2D | `[M, K]` |
| weight | 3D | `[E, *, *]`，N/K 由 `transB` 解释 |
| y | 2D | `[M, N]` |
| antiquantScale | 3D | `[E, G, N]` |
| bias | 2D | `[E, N]`，可选 |

多多多：

| Tensor | 维度 | shape |
|--------|------|-------|
| x[i] | 2D | `[M_i, K_i]` |
| weight[i] | 2D | `[* , *]`，N/K 由 `transB` 解释 |
| y[i] | 2D | `[M_i, N_i]` |
| antiquantScale[i] | 2D | `[G_i, N_i]` |
| bias[i] | 1D | `[N_i]`，可选 |

需要在现有 A16W8 校验基础上调整：
- 多多多 A16W4 pergroup 的 `antiquantScale` 从 A16W8 perchannel 的 1D 扩展为 2D。
- 单单单 A16W4 pergroup 的 `antiquantScale` 从 A16W8 perchannel 的 2D 扩展为 3D。
- bias 仍保持 `[E,N]` 或 `[N_i]`，不跟随 pergroup 的 G 维扩展。
- **本场景不支持 antiquantOffset 输入。**

### 2.4 Shape 关系校验

对每个 group/tensor 校验：

```text
G_i = antiquantScale 的 K-group 维度
N_i = weight 的 N 轴
K_i = weight 的 K 轴 = x 的 K 轴

G_i > 0
K_i % G_i == 0
groupSize_i = K_i / G_i
groupSize_i in {32, 64, 128, 256}
antiquantScale 的 N 轴 == N_i
bias 的 N 轴 == N_i
```

多多多还需校验：

```text
groupSize_0 == groupSize_1 == ... == groupSize_{E-1}
```

---

## 3. Tiling 计算方案

### 3.1 SetTiling 主流程

Host 侧入口保持现有 6 步：

```text
SetTiling(context):
  ① AnalyzeAttr(context)       // 解析属性、校验输入、设置 shapeList
  ② CalcResplitTiling(context) // 计算尾块重切分参数（详见 §3.4）
  ③ SetBaseTiling()            // 填充 GMMWeightQuantParam 结构
  ④ SetMatMulTiling()          // 填充 TCubeTiling 结构（baseM/N/K 等）
  ⑤ SetTilingKey(context)      // 生成 tiling key → 选择 kernel 模板
  ⑥ SetCustomParam(context)    // 设置 blockDim=coreNum，拷贝 tiling 数据到 GM
```

各步骤职责独立，①② 依赖 context 中的算子属性，③④ 填充 tiling 数据结构，⑤ 选择编译期模板，⑥ 将数据下发到 device。

### 3.2 baseM/baseN/baseK

Host tiling 复用 A16W8 ND basic-controller 的 `TCubeTiling` 参数：

| 参数 | 值 | 说明 |
|------|:--:|------|
| `baseM` | 256 | 有 bias 时 host 改为 240 |
| `baseN` | 256 | N 方向标准块 |
| `baseK` | 64 | L0 K 基本块 |
| `stepKa` | 4 | A 的 L1 K 步进 |
| `stepKb` | 4 | B 的 L1 K 步进 |
| `depthA1/depthB1` | 8 | 复用 A16W8 |
| `dbL0A/dbL0B/dbL0C` | 2/2/1 | 复用 A16W8 |

`kbL1Size` 和 `kaL1Size` 在 kernel controller 中计算：

```cpp
offsetParam.kbL1Size = mmTiling_->baseK * mmTiling_->stepKb; // 64 * 4 = 256
offsetParam.kaL1Size = offsetParam.kbL1Size;
```

因此：
- L0 基本块为 `baseM/baseN/baseK`。
- L1 K 方向基本块为 `kbL1Size=256`。
- A16W4 pergroup 不改变这些大小。

### 3.3 Kernel 侧 M/N 分块

`GMMWeightQuantBasicController::Process()` 复用 A16W8 的 M/N 二维切分和 round-robin 分核。

动态 `baseM`：

```cpp
if (mSize <= baseM || mSize >= baseM * 4) {
    actualBaseM = baseM;
} else if (mSize <= baseM * 2) {
    actualBaseM = CeilDiv(mSize, 2);
} else {
    actualBaseM = CeilDiv(mSize, 4);
}
```

动态 `baseN`：

```cpp
actualBaseN = (nSize >= baseN * coreNum) ? baseN : (baseN >> 1);
```

分块和分核：

```cpp
mBlockNum = CeilDiv(mSize, actualBaseM);
nBlockNum = CeilDiv(nSize, actualBaseN);

for each block in round-robin by coreNum:
    mOffset = (block / nBlockNum) * actualBaseM;
    nOffset = (block % nBlockNum) * actualBaseN;
    mL1Size = min(actualBaseM, mSize - mOffset);
    nL1Size = min(actualBaseN, nSize - nOffset);
    ComputeBasicBlock(offsetParam);
```

结论：
- M 和 N 都按 A16W8 basic 方案切分。
- 多多多不是“每核一个完整 expert”，而是每个 expert 内继续按 `(mBlock,nBlock)` 形成任务块，各核 round-robin 领取。
- K 不在核间切分，在 BasicBlock 内部按 `kbL1Size=256` 遍历。

### 3.4 kL1=256 与 group_size

`kbL1Size=256` 与支持的 group_size 完全整除：

| group_size | 每个 kL1 tile 包含 group 数 |
|:----------:|:--------------------------:|
| 32 | 8 |
| 64 | 4 |
| 128 | 2 |
| 256 | 1 |

这保证每个 L1 K tile 覆盖整数个 pergroup 段，VF 反量化时不需要处理跨 group 的 scale 边界。

### 3.5 Tail Resplit 策略

本 A16W4 pergroup ND 设计固定选择 `N_FIRST_BASIC_BLOCK`，进入 `GMMWeightQuantBasicController`。也就是说：
- 多多多：basic controller。
- 单单单：basic controller。
- `transB=false/true`：均注册 basic-controller tiling key。

`GMMWeightQuantParam` 中的 tail-resplit 字段保持结构兼容；A16W4 pergroup basic 路径不依赖 resplit controller。

---

## 4. Shape List 切分

### 4.1 单单单

`SetShapeListSplitMSingleXSingleWeightSingleY()` 复用现有 M/K/N 解析逻辑：

```cpp
groupNum_ = wShape.GetDim(0);  // weight 第0维 = E
mSize_ = xShape.GetDim(0);     // x 第0维 = M
kSize_ = xShape.GetDim(xDimNum - 1);  // x 最后一维 = K
nSize_ = wShape.GetDim(wDimNum - 1);  // weight 最后一维 = N（非转置）

// int4 weight 处理
if (weightDtype_ == DT_FLOAT || weightDtype_ == DT_INT32) {
    weightDtype_ = DT_INT4;
    if (!transB_) nSize_ = 8 * nSize_;
}

kList_[0] = kSize_;
nList_[0] = nSizeOri_;
mList_[0] = -1;                    // M 从 groupList 获取
```

`groupList` 决定每个 expert 的 `mSize`：
- `groupListType=0`：前缀和，当前 expert M 为相邻差值。
- `groupListType=1`：count 数组，当前元素即当前 expert M。

### 4.2 多多多

`SetShapeListMultiXMultiWeightMultiY()` 逐 tensor 填充：

```cpp
for (i = 0; i < MAX_TENSOR_CONT; i++) {
    m = xShape.GetDim(xDimNum - 2);
    k = transB_ ? wShape.GetDim(1) : wShape.GetDim(0);
    n = transB_ ? wShape.GetDim(0) : wShape.GetDim(1);
    mList_[i] = m;
    kList_[i] = k;
    nList_[i] = n;
    mSize_ = max(mSize_, m);
    kSize_ = max(kSize_, k);
    nSize_ = max(nSize_, n);
}
```

多多多每个 tensor 可以有不同 `M_i/K_i/N_i`，但 A16W4 pergroup 要求由 scale 推导出的 `groupSize_i` 一致。

总结：

- **单单单**：仅填充 `kList[0]` 和 `nList[0]`，`mList[0] = -1`（M 从 groupList 取值）
- **多多多**：逐 tensor 填充 `mList[i]`、`kList[i]`、`nList[i]`

#### 4.2.1 单单单 vs 多多多 M/K/N 来源差异

| 维度 | 单单单（groupType=0） | 多多多（groupType=-1） |
|------|:---------------------:|:---------------------:|
| tensor 数量 | x/weight/y 各 1 个 | x/weight/y 各多个 |
| M 来源 | groupList 差值（`m_i = groupList[i+1] - groupList[i]`） | `mList[i]`（每个 tensor 独立 M） |
| K 来源 | `kList[0]`（所有 group 共享同一个 K） | `kList[i]`（每个 tensor 独立 K） |
| N 来源 | `nList[0]`（所有 group 共享同一个 N） | `nList[i]`（每个 tensor 独立 N） |
| groupNum | `weight.shape[0]`（weight 第 0 维 = E） | tensor 个数 |
| 尾块重切分 | 可使能（NZ/转置时） | 不使能 |

**Kernel 侧 SetMKN() 取值逻辑**：
```cpp
// 单单单
mSize = groupList[groupIdx + 1] - groupList[groupIdx];  // 从 groupList 差值取 M
kSize = kList[0];
nSize = nList[0];

// 多多多
mSize = mList[groupIdx];
kSize = kList[groupIdx];
nSize = nList[groupIdx];
```

### 4.3 CalcResplitTiling() 尾块重切分适用性

**现有判断逻辑**：
```
EnableTailResplit():
  - 多多多 → 不使能
  - 单单单 + weight ND + B非转置 → 不使能
  - 其余 → 使能
```

**A16W4 pergroup ND 场景**：
- 多多多 + ND → **不使能**（与 A16W8 一致）
- 单单单 + ND + B非转置 → **不使能**（与 A16W8 一致）
- 单单单 + ND + B转置 → **使能**（需要确认 pergroup 场景是否适用）

**⚠️ 适配点**：单单单 + ND + B转置场景的尾块重切分，需要确认 pergroup 的 scale 偏移在 resplit 路径下是否正确。当前方案优先支持 `N_FIRST_BASIC_BLOCK` 路径（不使能 resplit），后续可扩展。

---

## 5. Tiling Key 注册

### 5.1 SetTilingKey

A16W4 pergroup ND 需要将 antiquant type 置为 `PER_GROUP`：

```cpp
if (IsA16W4NDPergroup()) {
    tilingKeyConfig_.antiquantType = static_cast<uint8_t>(QuantType::PER_GROUP);
} else if (antiquantScaleDtype_ == ge::DT_FLOAT8_E8M0) {
    tilingKeyConfig_.antiquantType = static_cast<uint8_t>(QuantType::MX);
} else {
    tilingKeyConfig_.antiquantType = static_cast<uint8_t>(QuantType::PER_CHANNEL);
}
```

其他关键字段：

| 字段 | A16W4 pergroup 取值 |
|------|--------------------|
| `socVersionType` | `SUPPORT_L1_TO_BT_BF16` |
| `algorithm` | `VECTOR_ANTIQUANT + N_FIRST_BASIC_BLOCK` |
| `transposeSituation` | 由 `transA/transB` 生成，`transA=false` |
| `quantType` | `NONE` |
| `optionInputSituation` | bias 是否存在 |
| `weightFormat` | `ND` |
| `templateCustom` | `MTE2_INNER_SIZE_256_BUF_NUM_4` |

### 5.2 新增 ASCENDC_TPL_ARGS_SEL 组合

在 `weight_quant_tiling_key.h` 的 `ASCENDC_TPL_SEL` 中新增以下组合：

#### 5.2.1 非转置场景（WQ_B_TRANS = WQGMM_NO_TRANS）

```cpp
// 组合 1: 无bias 非转置
ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
    ASCENDC_TPL_UINT_SEL(W_TYPE, ASCENDC_TPL_UI_LIST, WQGMM_ND),
    ASCENDC_TPL_UINT_SEL(OFFSET_OR_BIAS_EXIT, ASCENDC_TPL_UI_LIST, WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST),
    ASCENDC_TPL_UINT_SEL(C_QUANT_TYPE, ASCENDC_TPL_UI_LIST, WQGMM_NONE),
    ASCENDC_TPL_UINT_SEL(W_QUANT_TYPE, ASCENDC_TPL_UI_LIST, WQGMM_PER_GROUP),
    ASCENDC_TPL_UINT_SEL(WQ_B_TRANS, ASCENDC_TPL_UI_LIST, WQGMM_NO_TRANS),
    ASCENDC_TPL_UINT_SEL(WQ_A_TRANS, ASCENDC_TPL_UI_LIST, WQGMM_NO_TRANS),
    ASCENDC_TPL_UINT_SEL(TEMPLATE_CUSTOM_SC, ASCENDC_TPL_UI_LIST, WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4),
    ASCENDC_TPL_UINT_SEL(ALGORITHM_SUB_CATEGORY, ASCENDC_TPL_UI_LIST, WQGMM_N_FIRST_BASIC_BLOCK),
    ASCENDC_TPL_UINT_SEL(ALGORITHM_CATEGORY, ASCENDC_TPL_UI_LIST, WQGMM_VECTOR_ANTIQUANT)),

// 组合 2: 有bias 非转置
ASCENDC_TPL_ARGS_SEL(
    ..., WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_EXIST, ..., WQGMM_PER_GROUP, WQGMM_NO_TRANS, ...),
```

#### 5.2.2 转置场景（WQ_B_TRANS = WQGMM_TRANS）

```cpp
// 组合 3~4: 同上，仅 WQ_B_TRANS 改为 WQGMM_TRANS
ASCENDC_TPL_ARGS_SEL(
    ..., WQGMM_PER_GROUP, WQGMM_TRANS, ...),
// ... 共 2 个转置组合
```

### 5.3 新增组合的模板参数值汇总

| 参数 | 值 | 说明 |
|------|-----|------|
| W_TYPE | WQGMM_ND (0) | weight ND 格式 |
| OFFSET_OR_BIAS_EXIT | 0~1 | 2 种 bias 组合（本场景不支持 offset） |
| C_QUANT_TYPE | WQGMM_NONE (0) | 无 C 量化 |
| W_QUANT_TYPE | **WQGMM_PER_GROUP (3)** | ★ weight pergroup 量化 |
| WQ_B_TRANS | WQGMM_NO_TRANS (0) / WQGMM_TRANS (1) | 转置/非转置 |
| WQ_A_TRANS | WQGMM_NO_TRANS (0) | x 固定不转置 |
| TEMPLATE_CUSTOM_SC | WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4 (3) | MTE2 配置 |
| ALGORITHM_SUB_CATEGORY | WQGMM_N_FIRST_BASIC_BLOCK (3) | 基本块模式 |
| ALGORITHM_CATEGORY | WQGMM_VECTOR_ANTIQUANT (0) | 向量反量化 |

### 5.4 转置/非转置场景差异

| 维度 | 非转置（B_TRANS=NO_TRANS） | 转置（B_TRANS=TRANS） |
|------|:-------------------------:|:---------------------:|
| weight shape（多多多） | $(n_i, k_i)$ | $(k_i, n_i)$ |
| weight shape（单单单） | $(E, N, K)$ | $(E, K, N)$ |
| AIV 分发路径 | `ComputeBasicBlockAivNdKnNzNk` | `ComputeBasicBlockAivNdNkNzKn` |
| VF 函数 | `AntiquantComputeKNPerGroup`（K 为内轴） | `AntiquantComputeW4NKPerGroup`（N 为内轴） |
| VfConfig | `{vfN=256, vfK=64}` | `{vfN=64, vfK=256}` |
| tiling key 数量 | 2 个 | 2 个 |

### 5.5 与 kernel 入口 if constexpr 的匹配

每个 tiling key 组合对应 `grouped_matmul_apt.cpp` 中的一个 `if constexpr` 分支。详见 [01 交付件清单 §4](./01_deliverables.md#4-grouped_matmul_aptcpp-kernel-入口适配)。

**匹配规则**：tiling key 的模板参数值与 `if constexpr` 的条件一一对应，编译器根据 tiling key 自动选择匹配的分支。

---

## 6. Buffer 结论

Tiling 不新增 buffer 字段，也不为 A16W4 pergroup 生成新的 buffer 配置。buffer 大小由 kernel 侧复用 A16W8 ND 配置：

| 层级 | 结论 |
|------|------|
| L1 | `WeightL1` 单缓冲 128KB；无 bias 时 `AL1` 单缓冲 128KB；有 bias 时 host 将 `baseM` 调整为 240，`AL1` 单缓冲 124KB 可容纳 A tile |
| UB | 复用 `GetNdBufferInfo()`：weight input 174KB、high-bit output 66KB、scale 4KB |
| VF tile | A16W4 pergroup 的 scale 按 VF tile 消费，最小 `gs=32` 时单次最大 1KB，匹配 A16W8 ND 的 scale 单 buffer |

因此 tiling 侧固定使用 A16W8 ND 的 `baseM/baseN/baseK/stepKb/templateCustom`，不引入额外切分策略。
