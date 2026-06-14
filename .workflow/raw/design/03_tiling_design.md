# 03 Tiling 拦截校验与切分方案 — GMM A16W4 PerGroup ND

> 本文档设计 A16W4 pergroup ND 场景的 tiling 参数校验、拦截逻辑和切分策略。
> 相关文档：[01 交付件清单](./01_deliverables.md)、[02 与硬件无关方案](./02_platform_independent_design.md)、[04 Kernel 方案](./04_kernel_design.md)

---

## 1. Tiling Data 设计

### 1.1 GMMWeightQuantParam 结构 — 无需新增字段

现有 `GMMWeightQuantParam` 结构（定义于 `grouped_matmul_tiling_data_apt.h`）已包含 pergroup 所需的全部字段：

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

**关键字段**：
- `groupSize`：pergroup 场景设为 32/64/128/256，perchannel 场景保持 0
- 其余字段复用现有逻辑

### 1.2 groupSize 字段设置方式

`groupSize` 在 `SetAntiquantGroupSize()` 中根据 `antiquantScale` 的 shape 推导：

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

### 1.3 GMMWeightQuantTilingData 完整结构

```cpp
struct GMMWeightQuantTilingData {
    GMMWeightQuantParam gmmWeightQuantParam;  // 自定义参数
    GMMArray gmmArray;                         // MKN 列表
    TCubeTiling mmTilingData;                  // matmul tiling 参数
};
```

其中 `GMMArray`：
```cpp
struct GMMArray {
    int32_t mList[128] = {0};
    int32_t kList[128] = {0};
    int32_t nList[128] = {0};
};
```

**无需修改**。

---

## 2. Tiling 拦截校验

### 2.1 IsA16W4ND() 判断逻辑

现有判断函数：
```cpp
bool IsA16W4ND() const {
    return ge::GetSizeByDataType(xDType_) == 2 &&  // x 为 16bit
           (weightDtype_ == DT_INT4 || weightDtype_ == DT_INT32) &&
           !weightNzFlag_;  // ND 格式
}
```

**新增 IsA16W4NDPergroup() 判断函数**：

```cpp
bool IsA16W4NDPergroup() const {
    return IsA16W4ND() && groupSize_ > 0;
}
```

**区分 perchannel 和 pergroup 的方式**：
- perchannel：`groupSize_ == 0`（antiquantScale 为 2 维且 $G_i = 1$）
- pergroup：`groupSize_ > 0`（antiquantScale 为 3 维或 2 维但 $G_i > 1$）

### 2.2 CheckGroupSize() 校验逻辑

**现有校验**（仅 S8S4 场景）：
```cpp
// 当前仅校验 S8S4 场景的 groupSize（128/192/256/512）
if (xDType_ == DT_INT8 && weightDtype_ == DT_INT4) {
    // S8S4 groupSize 校验
}
```

**新增 A16W4 pergroup 校验**：

```cpp
if (IsA16W4ND() && groupSize_ > 0) {
    // A16W4 pergroup groupSize 校验
    if (groupSize_ != 32 && groupSize_ != 64 &&
        groupSize_ != 128 && groupSize_ != 256) {
        // 报错：groupSize 不在 {32, 64, 128, 256}
        return ERR_INVALID_GROUP_SIZE;
    }
    // K % groupSize == 0 校验
    if (kSize_ % groupSize_ != 0) {
        // 报错：K 不是 groupSize 的整数倍
        return ERR_K_NOT_ALIGNED;
    }
}
```

### 2.3 CheckTensorDim*() 校验适配

#### 2.3.1 多多多场景：CheckTensorDimMultiXMultiWeightMultiY()

**antiquantScale 维度校验**：
- perchannel：shape 为 $(n_i)$，1 维
- pergroup：shape 为 $(G_i, N_i)$，2 维

**适配方案**：
```cpp
if (IsA16W4ND()) {
    // A16W4 ND: antiquantScale 允许 1 维（perchannel）或 2 维（pergroup）
    if (antiquantScaleDimNum != 1 && antiquantScaleDimNum != 2) {
        return ERR_INVALID_DIM;
    }
}
```

#### 2.3.2 单单单场景：CheckTensorDimSingleXSingleWeightSingleY()

**antiquantScale 维度校验**：
- perchannel：shape 为 $(E, N)$，2 维
- pergroup：shape 为 $(E, G, N)$，3 维

**适配方案**：
```cpp
if (IsA16W4ND()) {
    // A16W4 ND: antiquantScale 允许 2 维（perchannel）或 3 维（pergroup）
    if (antiquantScaleDimNum != 2 && antiquantScaleDimNum != 3) {
        return ERR_INVALID_DIM;
    }
}
```

### 2.4 CheckTensorShape() 校验适配

**新增 pergroup shape 校验**：

```cpp
if (IsA16W4NDPergroup()) {
    // 1. K % groupSize == 0
    if (kSize_ % groupSize_ != 0) {
        return ERR_K_NOT_ALIGNED;
    }

    // 2. antiquantScale N 轴校验
    if (isSingleWeight_) {
        // 单单单: scale shape = (E, G, N)
        // G = K / gs, N 应与 weight 的 N 一致
        uint64_t expectedG = kSize_ / groupSize_;
        if (antiquantScaleShape.GetDim(1) != expectedG) {
            return ERR_SCALE_SHAPE_MISMATCH;
        }
    } else {
        // 多多多: scale shape = (G_i, N_i)
        // G_i = k_i / gs, N_i 应与 weight 的 N_i 一致
        uint64_t expectedG = kSize_ / groupSize_;
        if (antiquantScaleShape.GetDim(0) != expectedG) {
            return ERR_SCALE_SHAPE_MISMATCH;
        }
    }

    // 3. 多多多 pergroup 长度一致性校验
    if (!isSingleWeight_) {
        // 所有 tensor 的 s_i = k_i / G_i 必须相等
        uint64_t s = groupSize_;  // s_i = k_i / G_i = groupSize
        for (each tensor) {
            if (k_i / G_i != s) {
                return ERR_PERGROUP_LENGTH_INCONSISTENT;
            }
        }
    }

    // 4. pergroup 转置偶数校验
    if (transB_ && !isSingleWeight_) {
        if (groupSize_ % 2 != 0) {
            return ERR_PERGROUP_TRANSPOSE_ODD;
        }
    }
}
```

---

## 3. Tiling 计算方案

### 3.0 SetTiling() 主流程（6 步）

Host 侧 tiling 入口 `SetTiling()` 按固定 6 步顺序执行：

```
SetTiling(context):
  ① AnalyzeAttr(context)       // 解析属性、校验输入、设置 shapeList
  ② CalcResplitTiling(context) // 计算尾块重切分参数（详见 §3.4）
  ③ SetBaseTiling()            // 填充 GMMWeightQuantParam 结构
  ④ SetMatMulTiling()          // 填充 TCubeTiling 结构（baseM/N/K 等）
  ⑤ SetTilingKey(context)      // 生成 tiling key → 选择 kernel 模板
  ⑥ SetCustomParam(context)    // 设置 blockDim=coreNum，拷贝 tiling 数据到 GM
```

各步骤职责独立，①② 依赖 context 中的算子属性，③④ 填充 tiling 数据结构，⑤ 选择编译期模板，⑥ 将数据下发到 device。

### 3.1 baseM/baseN/baseK 计算方式

**参照 A16W8 多多多策略**（来自需求文档）：

| 参数 | 值 | 说明 |
|------|-----|------|
| baseM | 256 | 有 bias 时调整为 240 |
| baseN | 256 | — |
| baseK | 64 | — |
| singleCoreM | 256 | 有 bias 时 240 |
| singleCoreN | 256 | — |
| singleCoreK | kSize_ | — |
| stepKa | 4 | — |
| stepKb | 4 | — |
| depthA1 | 8 | — |
| depthB1 | 8 | — |
| dbL0A | 2 | 双缓冲 |
| dbL0B | 2 | 双缓冲 |
| dbL0C | 1 | — |

**A16W4 pergroup 场景**：baseM/baseN/baseK 可完全复用 A16W8 默认值。

**动态调整逻辑**（在 Controller `Process()` 中执行）：
```cpp
uint64_t baseM = (mSize <= baseM || mSize >= baseM*4) ? baseM
               : (mSize <= baseM*2) ? CeilDiv(mSize, 2) : CeilDiv(mSize, 4);
uint64_t baseN = (nSize >= baseN * coreNum) ? baseN : (baseN >> 1);
```

### 3.1.1 Kernel 侧 Basic Controller 分核算法

Controller `Process()` 在每个 AI Core 上执行，采用 **round-robin** 方式分配 (mBlock, nBlock) 任务：

```
Process():
  cubeBlockIdx = GetBlockIdx()
  if IS_AIV: cubeBlockIdx >>= 1    // AIV 核号 = blockIdx / 2

  for each group (groupIdx = 0..groupNum-1):
    SetMKN(groupIdx) → mSize, kSize, nSize

    // ① 动态调整 baseM（根据 M 大小选择二分/四分/不切）
    if mSize <= baseM || mSize >= baseM * 4:
        actualBaseM = baseM           // 256（不调整）
    elif mSize <= baseM * 2:
        actualBaseM = ceil(mSize / 2) // 二分时每块 ≈ M/2
    else:
        actualBaseM = ceil(mSize / 4) // 四分时每块 ≈ M/4

    // ② 动态调整 baseN（根据 N 是否够分满核决定）
    if nSize >= baseN * coreNum:
        actualBaseN = baseN           // 256（N 足够大）
    else:
        actualBaseN = baseN / 2       // 128（N 较小，缩小块以分满核）

    // ③ 计算 M/N 方向块数
    mBlockNum = ceil(mSize / actualBaseM)
    nBlockNum = ceil(nSize / actualBaseN)
    totalBlocks = mBlockNum * nBlockNum

    // ④ Round-robin 分核：每个核按步长 coreNum 领取任务
    curBlock = cubeBlockIdx (adjusted for cross-group remainder)
    while curBlock < count + totalBlocks:
        mOffset = ((curBlock - count) / nBlockNum) * actualBaseM
        nOffset = ((curBlock - count) % nBlockNum) * actualBaseN
        ComputeBasicBlock(offsetParam)   // K 方向在内部按 kbL1Size 步进
        curBlock += coreNum
```

**关键结论**：
- **M 和 N 方向都会切分**，形成二维块网格
- **K 方向不切分**（K 方向在 BasicBlock 内部按 `kbL1Size` 步进遍历）
- 各核以 round-robin 方式领取 `(mBlock, nBlock)` 任务，保证负载均衡
- AIV/AIC 通过 `cubeBlockIdx >>= 1` 实现协同（每对 AIV+AIC 共享一个核号）

### 3.1.2 kL1=256 的选择原因

kL1（即 kbL1Size，L1 层 K 方向 tile 大小）选择 256 的原因是 group_size 需要支持 32/64/128/256，kL1=256 可以保证在这几种 group_size 下，每个 tile 的 K 方向都不会跨越 group 边界，从而简化代码实现。具体来说：

| group_size | 一个 tile 包含的完整 group 数 | 计算 |
|:----------:|:---------------------------:|:----:|
| 32 | 8 | 256 / 32 = 8 |
| 64 | 4 | 256 / 64 = 4 |
| 128 | 2 | 256 / 128 = 2 |
| 256 | 1 | 256 / 256 = 1 |

kL1 = baseK × stepKb × depth = 64 × 4 / depth × depth = 256（其中 baseK=64, stepKb=4）。每个 L1 tile 的 K 方向恰好包含整数个 group，避免了反量化时跨 group 边界的复杂处理逻辑（如部分 group 的 scale 加载、尾 group 的特殊处理等）。

### 3.2 SetAntiquantGroupSize() 中 groupSize 推导逻辑

详见 §1.2。核心公式：

$$
groupSize = \frac{kSize}{G}
$$

其中 $G$ 为 antiquantScale 的 K 方向维度：
- 单单单：$G$ = antiquantScale shape 的第 1 维（3 维时 `(E, G, N)`）
- 多多多：$G$ = antiquantScale shape 的第 0 维（2 维时 `(G_i, N_i)`）

### 3.3 SetTilingKey() 中 antiquantType 设置

**现有代码**：
```cpp
tilingKeyConfig_.antiquantType = PER_CHANNEL;  // A16W8 固定
```

**适配方案**：
```cpp
if (IsA16W4NDPergroup()) {
    tilingKeyConfig_.antiquantType = PER_GROUP;
} else {
    tilingKeyConfig_.antiquantType = PER_CHANNEL;
}
```

**其他 tiling key 参数**：
```cpp
tilingKeyConfig_.socVersionType = SUPPORT_L1_TO_BT_BF16 * 10;  // Ascend 950
tilingKeyConfig_.algorithm = VECTOR_ANTIQUANT * 10 + N_FIRST_BASIC_BLOCK;
tilingKeyConfig_.transposeSituation = (transA << 1) | transB;
tilingKeyConfig_.weightFormat = ND;
tilingKeyConfig_.templateCustom = MTE2_INNER_SIZE_256_BUF_NUM_4;
```

### 3.4 尾块重切分（CalcResplitTiling）

#### 3.4.1 使能条件

```
EnableTailResplit():
  if 多多多（groupType == -1）: return false
  if weight 是 ND 格式 且 B 不转置: return false
  return true
```

**即使能条件**：单单单 + (weight 是 NZ 格式 或 B 转置)。

**A16W4 ND 非转置场景**：不使能尾块重切分，走 Basic Controller 路径。

#### 3.4.2 使能时的三分支逻辑

当尾块重切分使能时，根据 N 与 coreNum 的关系分三种情况：

| 分支 | 条件 | 策略 |
|------|------|------|
| 分支 1 | `N % (coreNum × 256) == 0` | mainBlockSize=256, mainBlockCount = N/(coreNum×256) |
| 分支 2 | `N > coreNum × 128` | CalcFullNumBlocksResplitTiling — 可分满核 |
| 分支 3 | `N ≤ coreNum × 128` | CalcNoFullNumBlocksResplitTiling — 可能无法分满核 |

分支 1 是理想情况（N 可被整除），直接使用固定 256 块大小。分支 2/3 处理 N 不能被整除的场景，通过调整 mainBlockSize 和 tailBlockSize 来减少尾块浪费。

### 3.5 A16W4 PerGroup 适配点

#### 3.5.1 当前代码行为 vs 用户确认策略

| 维度 | 当前 A16W8 代码行为 | 用户确认的 A16W4 pergroup 策略 |
|------|:-------------------:|:-----------------------------:|
| M 方向 | 动态切分（二分/四分/不切） | 切分成多个块 |
| N 方向 | 动态切分（256 或 128） | **不切分**（K 和 N 方向不做切分） |
| K 方向 | BasicBlock 内部步进 | 不切分（一致） |
| 多多多 | 每核领取 (mBlock, nBlock) | **每个核分配一个完整的 expert** |

**⚠️ 差异说明**：用户确认的策略是"K 和 N 方向不做切分，M 方向切分成多个块"，但当前 A16W8 代码中 M 和 N 方向都会切分。A16W4 pergroup 可能需要修改 Controller 的分核逻辑来实现用户描述的策略。

#### 3.5.2 需要评估的适配方案

1. **N 方向不切分**：修改 `actualBaseN` 逻辑，固定为 `nSize`（不切 N）
2. **多多多每核一个 expert**：修改 group 循环逻辑，每个核只处理一个 group
3. **尾块重切分**：A16W4 ND 非转置场景不使能（与 A16W8 一致），无需额外适配

### 3.5.3 ⚠️ 遗留问题：有 bias 场景 L1 空间不足

当 kL1=256, nL1=256, mL1=256 时：

| 区域 | 单缓冲大小 | 双缓冲总计 |
|------|:---------:|:---------:|
| WeightL1 | 128KB（nL1 × stepKb × baseK = 256 × 4 × 64 = 65536 half） | 256KB |
| AL1（无 bias） | 128KB | 256KB |
| **无 bias 总计** | | **512KB ✅** |
| BiasL1（有 bias） | 4KB | 8KB |
| AL1（有 bias） | 124KB | 248KB |
| **有 bias 总计** | | **512KB ⚠️** |

有 bias 时需要额外 4KB × 2 = 8KB 给 BiasL1，导致 AL1 每缓冲缩减到 124KB，但 A 矩阵（NZ 格式）需要 128KB，**差 4KB**。

这是 L1 物理空间 512KB 的硬限制，暂记为遗留问题，后续评估解决方案（如减小有 bias 时的 kL1、调整 BIAS_L1_SIZE、或 A 矩阵采用单缓冲等）。

---

## 4. Shape List 切分

### 4.1 SetShapeList() 适配

#### 4.1.1 单单单路径：SetShapeListSplitMSingleXSingleWeightSingleY()

**现有逻辑完全可复用**：

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
mList_[0] = -1;
```

**pergroup 无需修改**：MKN 取值逻辑与 perchannel 一致。

#### 4.1.2 多多多路径：SetShapeListMultiXMultiWeightMultiY()

**现有逻辑完全可复用**：

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

**pergroup 无需修改**。

### 4.2 mList/kList/nList 填充逻辑

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

### 5.1 新增 ASCENDC_TPL_ARGS_SEL 组合

在 `weight_quant_tiling_key.h` 的 `ASCENDC_TPL_SEL` 中新增以下组合：

#### 5.1.1 非转置场景（WQ_B_TRANS = WQGMM_NO_TRANS）

```cpp
// 组合 1: 无offset无bias 非转置
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

// 组合 2: 无offset有bias 非转置
ASCENDC_TPL_ARGS_SEL(
    ..., WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_EXIST, ..., WQGMM_PER_GROUP, WQGMM_NO_TRANS, ...),

// 组合 3: 有offset无bias 非转置
ASCENDC_TPL_ARGS_SEL(
    ..., WQGMM_ANTIQUANT_OFFSET_EXIST_BIAS_NOT_EXIST, ..., WQGMM_PER_GROUP, WQGMM_NO_TRANS, ...),

// 组合 4: 有offset有bias 非转置
ASCENDC_TPL_ARGS_SEL(
    ..., WQGMM_ANTIQUANT_OFFSET_EXIST_BIAS_EXIST, ..., WQGMM_PER_GROUP, WQGMM_NO_TRANS, ...),
```

#### 5.1.2 转置场景（WQ_B_TRANS = WQGMM_TRANS）

```cpp
// 组合 5~8: 同上，仅 WQ_B_TRANS 改为 WQGMM_TRANS
ASCENDC_TPL_ARGS_SEL(
    ..., WQGMM_PER_GROUP, WQGMM_TRANS, ...),
// ... 共 4 个转置组合
```

### 5.2 新增组合的模板参数值汇总

| 参数 | 值 | 说明 |
|------|-----|------|
| W_TYPE | WQGMM_ND (0) | weight ND 格式 |
| OFFSET_OR_BIAS_EXIT | 0~3 | 4 种 offset/bias 组合 |
| C_QUANT_TYPE | WQGMM_NONE (0) | 无 C 量化 |
| W_QUANT_TYPE | **WQGMM_PER_GROUP (3)** | ★ weight pergroup 量化 |
| WQ_B_TRANS | WQGMM_NO_TRANS (0) / WQGMM_TRANS (1) | 转置/非转置 |
| WQ_A_TRANS | WQGMM_NO_TRANS (0) | x 固定不转置 |
| TEMPLATE_CUSTOM_SC | WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4 (3) | MTE2 配置 |
| ALGORITHM_SUB_CATEGORY | WQGMM_N_FIRST_BASIC_BLOCK (3) | 基本块模式 |
| ALGORITHM_CATEGORY | WQGMM_VECTOR_ANTIQUANT (0) | 向量反量化 |

### 5.3 转置/非转置场景差异

| 维度 | 非转置（B_TRANS=NO_TRANS） | 转置（B_TRANS=TRANS） |
|------|:-------------------------:|:---------------------:|
| weight shape（多多多） | $(n_i, k_i)$ | $(k_i, n_i)$ |
| weight shape（单单单） | $(E, N, K)$ | $(E, K, N)$ |
| AIV 分发路径 | `ComputeBasicBlockAivNdKnNzNk` | `ComputeBasicBlockAivNdNkNzKn` |
| VF 函数 | `AntiQuantInt4NdKn`（K 为内轴） | `AntiQuantInt4NdNk`（N 为内轴） |
| VfConfig | `{vfN=256, vfK=64}` | `{vfN=64, vfK=256}` |
| tiling key 数量 | 4 个 | 4 个 |

### 5.4 与 kernel 入口 if constexpr 的匹配

每个 tiling key 组合对应 `grouped_matmul_apt.cpp` 中的一个 `if constexpr` 分支。详见 [01 交付件清单 §4](./01_deliverables.md#4-grouped_matmul_aptcpp-kernel-入口适配)。

**匹配规则**：tiling key 的模板参数值与 `if constexpr` 的条件一一对应，编译器根据 tiling key 自动选择匹配的分支。
