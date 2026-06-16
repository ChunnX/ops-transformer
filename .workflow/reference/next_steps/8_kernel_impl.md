# 任务 8：Kernel 适配 — GMM A16W4 PerGroup ND

> **任务状态**：待执行
> **前置任务**：任务 7（Tiling 实现 + UT 验证）✅
> **目标**：在 `weight_quant_basic_block` 体系中适配 A16W4 INT4 pergroup ND 计算逻辑，使 GMM 算子支持 A16W4 PerGroup ND 数据流。

---

## 1. 任务概述

### 1.1 目标

基于现有 GMM A16W8 ND perchannel 数据流，新增 A16W4 INT4 pergroup ND 数据流的 kernel 支持。核心修改集中在：
- Controller 层：pergroup scale/offset 地址偏移
- VecCompute 层：pergroup scale/offset UB 地址计算
- VF 层：INT4 pergroup 反量化函数（KN/NK 两条路径）
- 入口注册：8 个新 `if constexpr` 分支

### 1.2 不在范围内

- Aclnn 通路（API checker、infershape）— 由其他人完成
- Tiling 计算 — 已在任务 7 完成
- A2/A3（arch30）路径 — 不触碰

### 1.3 参考文档

| 文档 | 路径 | 用途 |
|------|------|------|
| 交付件清单 | `.workflow/raw/design/01_deliverables.md` | 完整修改文件列表 |
| Kernel 设计方案 | `.workflow/raw/design/04_kernel_design.md` | 详细 kernel 方案（950 行） |
| SRS 实现方案 | `.workflow/raw/development/gmm_a16w4_pergroup_nd_srs.md` | 完整项目实现方案 |
| VF 反量化调研 | `.workflow/raw/development/wqbmmv2_pergroup_dequant_research.md` | wqbmmv2 pergroup 反量化详细调研 |
| 算子规格 | `.workflow/raw/spec/gmm_a16w4nd_op.md` | 输入输出规格约束 |

### 1.4 参考代码（ops-nn 仓库）

| 文件 | 路径 | 参考价值 |
|------|------|----------|
| wqbmmv2 reg_base_common | `~/Code/ops-nn/matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_reg_base_common.h` | KN/NK pergroup 反量化入口和参数构造 |
| wqbmmv2 vf | `~/Code/ops-nn/matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_vf.h` | 所有 VF 函数实现和 Params 结构体 |
| resplit controller | `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/grouped_matmul_weight_quant_resplit_controller.h` | pergroup GM 地址更新参考（L260-262） |

---

## 2. 现有代码结构分析

### 2.1 Kernel 入口（grouped_matmul_apt.cpp）

V310_GMM_ANTI_QUANT 分支的 A16 部分（`#else` 块，L362-419）现有 **6 个分支**，全部为 `WQGMM_PER_CHANNEL`：

| # | W_TYPE | OFFSET_OR_BIAS | W_QUANT_TYPE | B_TRANS | ALGORITHM_SUB | Controller |
|---|--------|----------------|--------------|---------|---------------|------------|
| 1 | ND | NOT_EXIST_BIAS_NOT_EXIST | PER_CHANNEL | TRANS | N_FIRST_TAIL_RESPLIT | ResplitController |
| 2 | ND | EXIST_BIAS_NOT_EXIST | PER_CHANNEL | TRANS | N_FIRST_TAIL_RESPLIT | ResplitController |
| 3 | ND | NOT_EXIST_BIAS_NOT_EXIST | PER_CHANNEL | TRANS | N_FIRST_BASIC_BLOCK | BasicController |
| 4 | ND | EXIST_BIAS_NOT_EXIST | PER_CHANNEL | TRANS | N_FIRST_BASIC_BLOCK | BasicController |
| 5 | ND | NOT_EXIST_BIAS_NOT_EXIST | PER_CHANNEL | NO_TRANS | N_FIRST_BASIC_BLOCK | BasicController |
| 6 | ND | EXIST_BIAS_NOT_EXIST | PER_CHANNEL | NO_TRANS | N_FIRST_BASIC_BLOCK | BasicController |

**关键发现**：`weight_quant_tiling_key.h` 已注册 **8 个 ND PER_GROUP tiling key 组合**（L193-280），但 `grouped_matmul_apt.cpp` 中**没有对应的 `if constexpr` 分支**。

### 2.2 WqmmConfig（basic_block_config.h）

现有 3 个实例（L41-43）：
```cpp
static constexpr WqmmConfig S8S4_NZKN_G = {false, false, QuantType::PER_GROUP, false, QuantType::NONE, CubeFormat::NZ};
static constexpr WqmmConfig A16MXF4_NZKN = {false, false, QuantType::MX, false, QuantType::NONE, CubeFormat::NZ};
static constexpr WqmmConfig MXA8W4_NZNK = {false, true, QuantType::MX, false, QuantType::NONE, CubeFormat::NZ};
```

**无 ND PER_GROUP 实例。**

### 2.3 Buffer 配置（basic_block_config.h）

现有 `GetNdBufferInfo`（L149-163）用于 A16W4 ND PER_CHANNEL。**无 `GetNdPerGroupBufferInfo`。**

`GetBufferConfig` 分发器（L220-242）按 `wqmmConfig` 路由到对应 buffer 函数。

### 2.4 Controller SetGmAddr（basic_controller.h, L195-238）

关键偏移更新（L230-237）：
```cpp
xBaseOffset += offsetParam.mSize * offsetParam.kSize;
if constexpr (IsSameType<wType, int4b_t>::value) {
    weightBaseOffset += (offsetParam.nSize * offsetParam.kSize) >> 1;
} else {
    weightBaseOffset += offsetParam.nSize * offsetParam.kSize;
}
antiquantParamsBaseOffset += offsetParam.nSize;   // ← PER_CHANNEL: 仅 nSize
yBaseOffset += offsetParam.mSize * offsetParam.nSize;
```

**PER_GROUP 需要**：`antiquantParamsBaseOffset += (kSize / groupSize) * nSize`

### 2.5 VecCompute（weight_quant_vec_compute.h）

- **CopyAntiQuantParamsGmToUb**（L494-523）：PER_GROUP 分支已存在（L515-519），使用 `DataCopyPad2D` 搬运 scale，但**未搬运 offset**。
- **CalLocalAddrForVf**（L1136-1176）：scale 地址仅用 `nWeightLowBitUbOffset`（L1141-1142），**缺少 K 方向 group 偏移**。
- **AntiQuantProcessNd**（L1009-1052）：dispatches INT4 到 `AntiQuantInt4NdNk/NdKn`，**无 PER_GROUP 分发**。

### 2.6 VF 函数（basic_block_vf_nd.h）

- **AntiQuantInt4NdNk**（L465-514）：INT4 ND NK 反量化（perchannel）
- **AntiQuantInt4NdKn**（L516-567）：INT4 ND KN 反量化（perchannel）
- **NdNkLoadScaleOffset**（L327-344）：仅 PER_TENSOR/PER_CHANNEL，**无 PER_GROUP**
- **NdKnLoadScaleOffset**（L346-368）：仅 PER_TENSOR/PER_CHANNEL，**无 PER_GROUP**

### 2.7 ComputeBasicBlock 分发（weight_quant_basic_block.h, L348-361）

按 `weightFormat` 和 `bTrans` 分发，**不区分 PER_GROUP/PER_CHANNEL**。PER_GROUP 差异在 AIV vector compute 内部处理。

### 2.8 Cube Compute（weight_quant_cube_compute.h）

**完全通用**，不感知 perchannel/pergroup。无需修改。

---

## 3. 子任务拆分

按依赖关系和代码层次，将任务 8 拆分为 **6 个子任务**，按顺序执行：

```
8.1 基础配置层（WqmmConfig + Buffer + Kernel 入口）
  ↓
8.2 Controller 层（SetGmAddr 地址偏移）
  ↓
8.3 VecCompute 层（scale/offset 搬运 + 地址计算）
  ↓
8.4 VF 层 — KN 路径（bTrans=false 反量化）
  ↓
8.5 VF 层 — NK 路径（bTrans=true 反量化）
  ↓
8.6 编译验证 + 回归测试
```

---

### 3.1 子任务 8.1：基础配置层

**目标**：新增 WqmmConfig 实例、Buffer 配置、Kernel 入口分支注册

**修改文件**：
1. `basic_block_config.h` — 新增 WqmmConfig + Buffer 配置
2. `grouped_matmul_apt.cpp` — 新增 8 个 if constexpr 分支

#### 3.1.1 新增 WqmmConfig 实例

文件：`gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_config.h`

在现有 WqmmConfig 实例（L41-43）后新增 4 个：

```cpp
// A16W4 ND KN pergroup（B 非转置，weight shape = (N, K)）
static constexpr WqmmConfig A16W4_NDKN_PG = {
    false, false, QuantType::PER_GROUP, false, QuantType::NONE, CubeFormat::ND};
// A16W4 ND KN pergroup with offset
static constexpr WqmmConfig A16W4_NDKN_PG_OFF = {
    false, false, QuantType::PER_GROUP, true, QuantType::NONE, CubeFormat::ND};
// A16W4 ND NK pergroup（B 转置，weight shape = (K, N)）
static constexpr WqmmConfig A16W4_NDNK_PG = {
    false, true, QuantType::PER_GROUP, false, QuantType::NONE, CubeFormat::ND};
// A16W4 ND NK pergroup with offset
static constexpr WqmmConfig A16W4_NDNK_PG_OFF = {
    false, true, QuantType::PER_GROUP, true, QuantType::NONE, CubeFormat::ND};
```

#### 3.1.2 Buffer 配置

文件：`basic_block_config.h`

**方案**：复用 `GetNdBufferInfo()`，不新增 `GetNdPerGroupBufferInfo`。

理由：
- `antiQuantScale` 4KB / 4 buffer / 单缓冲 1KB 对 pergroup VF tile 粒度足够（最大需求 1KB）
- `antiQuantOffset` 4KB 保留，pergroup 运行时使用
- UB 空间分配逻辑不变

在 `GetBufferConfig`（L220-242）中，ND PER_GROUP 路由到 `GetNdBufferInfo`（与 PER_CHANNEL 共用）。**需要验证**：现有分发条件是否已覆盖 PER_GROUP + ND 组合。若未覆盖，需在分发逻辑中新增条件。

#### 3.1.3 Kernel 入口分支

文件：`gmm/grouped_matmul/op_kernel/grouped_matmul_apt.cpp`

在 V310_GMM_ANTI_QUANT A16 `#else` 块（L362-419）的末尾，新增 8 个 `if constexpr` 分支：

```text
2 个 transB (NO_TRANS / TRANS)
× 2 个 offset (NOT_EXIST / EXIST)  — 对应 NOT_EXIST_BIAS_xxx / EXIST_BIAS_xxx
× 2 个 bias (NOT_EXIST / EXIST)
= 8 个分支
```

每个分支的模式：

```cpp
else if constexpr (W_TYPE == WQGMM_ND &&
    OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_xxx_BIAS_xxx &&
    C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_GROUP &&
    WQ_B_TRANS == WQGMM_xxx_TRANS && WQ_A_TRANS == WQGMM_NO_TRANS &&
    TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4 &&
    ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_BASIC_BLOCK &&
    ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
    static constexpr WqmmConfig wqmmCfg = {
        false, bTrans, QuantType::PER_GROUP, hasOffset, QuantType::NONE, CubeFormat::ND};
    INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(
        GMMWeightQuantBasicController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);
}
```

**注意**：`VEC_ANTIQUANT_CONFIG_3` 复用现有值 `{ubMte2BufferNum, ubMte2InnerSize}`。需确认其具体数值与 pergroup 兼容。

**8 个分支对应表**：

| # | OFFSET_OR_BIAS_EXIT | B_TRANS | WqmmConfig |
|---|---------------------|---------|------------|
| 1 | NOT_EXIST_BIAS_NOT_EXIST | NO_TRANS | A16W4_NDKN_PG |
| 2 | NOT_EXIST_BIAS_EXIST | NO_TRANS | A16W4_NDKN_PG (hasBias runtime) |
| 3 | EXIST_BIAS_NOT_EXIST | NO_TRANS | A16W4_NDKN_PG_OFF |
| 4 | EXIST_BIAS_EXIST | NO_TRANS | A16W4_NDKN_PG_OFF (hasBias runtime) |
| 5 | NOT_EXIST_BIAS_NOT_EXIST | TRANS | A16W4_NDNK_PG |
| 6 | NOT_EXIST_BIAS_EXIST | TRANS | A16W4_NDNK_PG (hasBias runtime) |
| 7 | EXIST_BIAS_NOT_EXIST | TRANS | A16W4_NDNK_PG_OFF |
| 8 | EXIST_BIAS_EXIST | TRANS | A16W4_NDNK_PG_OFF (hasBias runtime) |

#### 3.1.4 config950 注册

文件：`gmm/grouped_matmul/op_host/grouped_matmul_def.cpp`

**现有 A16W4 ND 条目**（3 个，indices 119-121）：

| # | x | weight | bias | antiquant_scale | antiquant_offset | y |
|---|---|--------|------|-----------------|------------------|---|
| 119 | FP16 | INT4(ND) | FP16 | FP16 | FP16 | FP16 |
| 120 | BF16 | INT4(ND) | BF16 | BF16 | BF16 | BF16 |
| 121 | BF16 | INT4(ND) | FLOAT | BF16 | BF16 | BF16 |

所有现有条目均有 `antiquant_offset` 存在（非 null）。

**需要新增的条目**（4 个，覆盖 `antiquant_offset=null` 的组合）：

根据 950 伪量化规格，A16W4 支持的 dtype 组合为：
- FP16: bias=FP16/null, antiquant_offset=FP16/null
- BF16: bias=BF16/FLOAT32/null, antiquant_offset=BF16/null

现有条目覆盖了 `antiquant_offset=present` 的 3 种 bias 组合。需要补充 `antiquant_offset=null` 的 4 种组合：

| 新增 # | x | weight | bias | antiquant_scale | antiquant_offset | y |
|--------|---|--------|------|-----------------|------------------|---|
| 154 | FP16 | INT4(ND) | FP16 | FP16 | **null** | FP16 |
| 155 | FP16 | INT4(ND) | **null** | FP16 | **null** | FP16 |
| 156 | BF16 | INT4(ND) | BF16 | BF16 | **null** | BF16 |
| 157 | BF16 | INT4(ND) | FLOAT | BF16 | **null** | BF16 |

> **注意**：bias=null 时注册 dtype 使用 `DT_FLOAT` 作为占位符（与现有 scale/offset 等未使用可选参数的占位方式一致）。antiquant_offset=null 时同理使用 `DT_FLOAT` 占位。

**每个新条目需要在 20 个数组中各追加一个元素**（10 个 tensor × DataType + Format）。追加位置为各数组的末尾（当前共 154 个元素，新条目从 index 154 开始）。

**其余 tensor 的固定值**（所有新条目共用）：
- scale: DT_UINT64, FORMAT_ND（占位，本场景不使用）
- offset: DT_FLOAT, FORMAT_ND（占位，本场景不使用）
- group_list: DT_INT64, FORMAT_ND
- per_token_scale: DT_FLOAT, FORMAT_ND（占位，本场景不使用）
- weight format: FORMAT_ND
- 所有非 weight tensor format: FORMAT_ND

#### 3.1.5 验收标准

- [ ] 编译通过
- [ ] 8 个新分支的 tiling key 与 `weight_quant_tiling_key.h` 中已注册的 8 个 ND PER_GROUP 组合一一对应
- [ ] config950 新增 4 个 dtype 组合（antiquant_offset=null），总计 7 个 A16W4 ND 条目

---

### 3.2 子任务 8.2：Controller 层适配

**目标**：修改 `SetGmAddr` 支持 pergroup scale/offset 地址偏移，独立 bias 偏移

**修改文件**：
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/grouped_matmul_weight_quant_basic_controller.h`

#### 3.2.1 SetGmAddr pergroup 偏移

当前代码（L236）：
```cpp
antiquantParamsBaseOffset += offsetParam.nSize;  // PER_CHANNEL
```

修改为：
```cpp
if constexpr (wqmmConfig.antiQuantType == QuantType::PER_GROUP) {
    uint64_t groupNumK = offsetParam.kSize / gmmBaseTiling_->groupSize;
    antiquantParamsBaseOffset += groupNumK * offsetParam.nSize;
} else {
    antiquantParamsBaseOffset += offsetParam.nSize;
}
```

**参考**：resplit controller（L260-262）已有正确实现：
```cpp
antiquantScaleGm_ += nSize * CeilDivide(kSize, static_cast<uint64_t>(gmmBaseTiling_->groupSize));
```

#### 3.2.2 Bias 偏移独立

**问题**：单单单场景下，bias shape 为 `[E, N]`，scale/offset shape 为 `[E, G, N]`。当前 bias 使用 `antiquantParamsBaseOffset` 定位，pergroup 时两者偏移量不同。

**方案**：新增 `biasBaseOffset` 变量，独立于 `antiquantParamsBaseOffset`。

```cpp
uint64_t biasBaseOffset = 0;

// 在 bias 地址设置时使用 biasBaseOffset
// 在每个 group 结束后：
biasBaseOffset += offsetParam.nSize;
```

**注意**：需检查 `SetGmAddr` 中 bias 地址设置的完整逻辑，确认 `singleWeight == 1` 时 bias 地址的当前实现方式。

#### 3.2.3 antiquantOffset 地址

当 `hasAntiQuantOffset=true` 时，offset 的 GM 地址更新与 scale 相同（pergroup 偏移量一致）。需确认当前代码中 `antiquantOffsetGm_` 是否使用独立的偏移变量，还是与 scale 共用 `antiquantParamsBaseOffset`。

**参考**：resplit controller（L260-262）中 scale 和 offset 各自独立更新。

#### 3.2.4 验收标准

- [ ] pergroup 场景下 antiquantScale/offset 地址正确步进 `(kSize/groupSize) * nSize`
- [ ] 单单单场景下 bias 地址正确步进 `nSize`（独立于 scale）
- [ ] perchannel 路径不受影响（`if constexpr` 隔离）

---

### 3.3 子任务 8.3：VecCompute 层适配

**目标**：适配 pergroup scale/offset 的 UB 地址计算和搬运逻辑

**修改文件**：
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_vec_compute.h`

#### 3.3.1 CalLocalAddrForVf 增加 PER_GROUP scale 地址

当前代码（L1141-1142）：
```cpp
uint64_t antiquantParamOffset =
    mte2BufIdx * UB_BUFFER_INFO.antiQuantScaleUbSingleBufferSize + nWeightLowBitUbOffset;
```

修改为：
```cpp
if constexpr (wqmmConfig.antiQuantType == QuantType::PER_GROUP) {
    uint64_t antiquantParamOffset =
        mte2BufIdx * UB_BUFFER_INFO.antiQuantScaleUbSingleBufferSize +
        kWeightLowBitUbOffset / antiQuantGroupSize_ * nSize + nWeightLowBitUbOffset;
} else {
    // perchannel 原逻辑
    uint64_t antiquantParamOffset =
        mte2BufIdx * UB_BUFFER_INFO.antiQuantScaleUbSingleBufferSize + nWeightLowBitUbOffset;
}
```

对 `antiQuantOffset` 地址计算做同样修改（当 `hasAntiQuantOffset=true` 时）。

#### 3.3.2 CopyAntiQuantParamsGmToUb 补充 PER_GROUP offset 搬运

当前 PER_GROUP 分支（L515-519）仅搬运 scale。需新增 offset 搬运：

```cpp
} else if constexpr (wqmmConfig.antiQuantType == QuantType::PER_GROUP) {
    // scale 搬运（已有）
    DataCopyPad2D(ubAntiQuantScaleTotalBuffer_[...],
                  antiQuantScaleGlobal_[ubMte2KOffset / antiQuantGroupSize_ * offsetParam.nSize + ubMte2NOffset],
                  CeilDivide(ubMte2KSize, antiQuantGroupSize_), ubMte2NSize,
                  VEC_MAX_ELEM_B16, offsetParam.nSize);
    // ★ 新增：offset 搬运
    if constexpr (wqmmConfig.hasAntiQuantOffset) {
        DataCopyPad2D(ubAntiQuantOffsetTotalBuffer_[...],
                      antiQuantOffsetGlobal_[ubMte2KOffset / antiQuantGroupSize_ * offsetParam.nSize + ubMte2NOffset],
                      CeilDivide(ubMte2KSize, antiQuantGroupSize_), ubMte2NSize,
                      VEC_MAX_ELEM_B16, offsetParam.nSize);
    }
}
```

**关键约束**：scale/offset 以 **VF tile 粒度**搬运，不是 MTE2 weight tile 粒度。原因：`antiQuantScale` 单缓冲仅 1KB，MTE2 tile（K=256, N=256）的 scale 在 `groupSize=32` 时需要 4KB，超过单缓冲。

#### 3.3.3 AntiQuantProcessNd PER_GROUP 分发

当前 `AntiQuantProcessNd`（L1009-1052）对 INT4 分发到 `AntiQuantInt4NdNk/NdKn`（perchannel VF）。

需新增 PER_GROUP 分发：

```cpp
if constexpr (wqmmConfig.antiQuantType == QuantType::PER_GROUP) {
    if constexpr (bTrans) {
        AntiQuantInt4PerGroupNdNk(...);  // 新增
    } else {
        AntiQuantInt4PerGroupNdKn(...);  // 新增
    }
} else {
    // 原有 perchannel 分发
    AntiQuantInt4NdNk/NdKn(...);
}
```

#### 3.3.4 验收标准

- [ ] PER_GROUP scale UB 地址包含 K 方向 group 偏移
- [ ] PER_GROUP offset 在 `hasAntiQuantOffset=true` 时正确搬运到 UB
- [ ] perchannel 路径不受影响

---

### 3.4 子任务 8.4：VF 层 — KN 路径反量化（bTrans=false）

**目标**：实现 pergroup KN 路径的 INT4 反量化

**修改文件**：
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_vf_nd.h`

#### 3.4.1 方案概述

KN 路径使用**统一的** `AntiquantPerGroupKN` 风格 VF 函数处理**所有 groupSize**（32/64/128/256），不按 groupSize 分派。

原因：scale 按 `(group, N-tile)` 加载，一个 group 的 scale 在 K 内循环中被复用 groupSize 次。groupSize 只影响 K 内循环迭代次数，不影响 scale 加载方式。

**参考**：ops-nn `AntiquantComputeKNPerGroup`（reg_base_common.h L1139-1180）+ `AntiquantPerGroupKN`（vf.h L347-443）

#### 3.4.2 NdKnLoadScaleOffset PER_GROUP 分支

当前 `NdKnLoadScaleOffset`（L346-368）仅处理 PER_TENSOR/PER_CHANNEL。需新增 PER_GROUP 分支：

```cpp
} else if constexpr (wqmmConfig.antiQuantType == QuantType::PER_GROUP) {
    // 从 UB [G_tile, N_tile_align] 布局加载当前 group 的 scale
    // DIST_NORM: 连续加载 128 个 FP16 元素
    MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(
        antiQuantScaleVreg, antiQuantScaleBasePhyAddr);
    MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(
        antiQuantScaleVreg1, antiQuantScaleBasePhyAddr1);
    if constexpr (wqmmConfig.hasAntiQuantOffset) {
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(
            antiQuantOffsetVreg, antiQuantOffsetBasePhyAddr);
        MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_NORM>(
            antiQuantOffsetVreg1, antiQuantOffsetBasePhyAddr1);
    }
}
```

#### 3.4.3 AntiQuantInt4PerGroupNdKn 函数

新增 pergroup KN VF 函数，参照 wqbmmv2 `AntiquantPerGroupKN`：

**参数构造**（参考 `ParamsKN<xType>`）：

| 参数 | 计算方式 | 说明 |
|------|----------|------|
| `kBub` | `bubKLen` | K 方向实际长度 |
| `nBub` | `bubNLen` | N 方向实际长度 |
| `nBubXTypeAlign` | `CeilAlign(nBub, BLOCK_CUBE)` | N 对齐到 16 |
| `nBubWTypeAlign` | `CeilAlign(nBub, ONE_BLK_W_NUM)` | N 对齐到 32（INT4） |
| `nLoop` | `CeilDiv(nBubWTypeAlign, VEC_MAX_ELEM_B16)` | N 方向外层循环次数 |
| `groupSize` | `tiling->groupSize` | pergroup 大小 |
| `groupNum` | `CeilDiv(kBub, groupSize) - 1` | 完整 group 数（减 1 因 tail 单独处理） |
| `groupTail` | `kBub % groupSize > 0 ? kBub % groupSize : groupSize` | 尾部 group 大小 |
| `dataBlockStride` | `CeilAlign(kBub, BLOCK_CUBE) + 1` | NZ 输出 block 间距 |
| `wNStride` | `VEC_MAX_ELEM_B16 >> 1 = 64` | INT4 weight N 方向步长（字节） |
| `wKStride` | `nBubWTypeAlign >> 1` | INT4 weight K 方向步长（字节） |
| `wGroupStride` | `groupSize * nBubWTypeAlign >> 1` | INT4 weight group 间步长（字节） |

**循环结构**（三层）：

```text
for nIdx = 0 to nLoop-1:                          // N 方向外层
  maskWeight = UpdateMask(nBubXTypeAlign)
  for gIdx = 0 to groupNum-1:                      // 完整 group
    scale ← DIST_NORM(scaleBase + gIdx * nBubXTypeAlign + nIdx * vfElemB16)
    if hasOffset: offset ← DIST_NORM(offsetBase + ...)
    for kIdx = 0 to groupSize-1:                   // K 方向
      weightB4 ← DIST_UNPACK4_B8(weightInBase + ...)
      weightB16 ← Cast<xType, int4x2_t>(weightB4, maskWeight)
      if hasOffset: weightB16 ← Add(weightB16, offset, maskWeight)
      weightB16 ← Mul(weightB16, scale, maskWeight)
      DataCopy<DATA_BLOCK_COPY, POST_MODE_UPDATE>(weightOutBase, weightB16, ...)
  // 处理 groupTail
  weightOutBase += weightOutStride
```

**UB 地址映射**：
- `scaleBaseAddr` → `ubAntiQuantScaleTotalBuffer_` 当前 VF tile 偏移
- `offsetBaseAddr` → `ubAntiQuantOffsetTotalBuffer_` 当前 VF tile 偏移
- `weightInBaseAddr` → `ubWeightInputLowBitTotalBuffer_` 当前偏移
- `weightOutBaseAddr` → `ubHighBitTotalBuffer_` 当前偏移

#### 3.4.4 双 AIV 切分影响

KN 路径 AIV 沿 K 切分，每个 AIV 处理 K/2（=128）。group 数减半，但通过 `param.groupNum` 和 `param.groupTail` 自动处理。**无需特殊处理。**

#### 3.4.5 验收标准

- [ ] 所有 groupSize（32/64/128/256）计算正确
- [ ] 有/无 offset 均正确
- [ ] 转置=false 路径正确
- [ ] perchannel KN 路径不受影响

---

### 3.5 子任务 8.5：VF 层 — NK 路径反量化（bTrans=true）

**目标**：实现 pergroup NK 路径的 INT4 反量化，按 groupSize 分派 4 个专用 VF 函数

**修改文件**：
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_vf_nd.h`

#### 3.5.1 核心差异：Scale/Offset 地址公式

**不能原样迁移 ops-nn scale/offset 索引。**

| 维度 | wqbmmv2 NK | GMM NK |
|------|-----------|--------|
| GM scale 布局 | `[N, G]` | `[G, N]`（始终，不随 bTrans 变化） |
| UB scale 布局 | `[N_tile, G_tile_align]` | `[G_tile, N_tile_align]` |
| scale 地址公式 | `base + groupIdx + nBubIdx × outerStrideScale` | `base + groupIdx × nTileAlign + nBubIdx` |

GMM 的 scale/offset 固定为 `[G, N]` 输入布局，按 VF tile 从 GM 搬入 UB 后自然布局为 `[G_tile, N_tile_aligned]`。

#### 3.5.2 NdNkLoadScaleOffset PER_GROUP 分支

当前 `NdNkLoadScaleOffset`（L327-344）仅处理 PER_TENSOR/PER_CHANNEL。需新增 PER_GROUP 分支。

NK 路径的 scale 加载按 groupSize 分派，不适合在 `NdNkLoadScaleOffset` 中统一处理。建议在 pergroup NK VF 函数内部直接处理 scale 加载。

#### 3.5.3 AntiQuantInt4PerGroupNdNk 函数

新增 pergroup NK 入口函数，按 groupSize 分派：

```cpp
if (antiQuantGroupSize_ == 32) {
    // 调用 gs=32 专用 VF
} else if (antiQuantGroupSize_ == 64) {
    // 调用 gs=64 专用 VF
} else if (antiQuantGroupSize_ == 128) {
    // 调用 gs=128 专用 VF
} else if (antiQuantGroupSize_ == 256) {
    // 调用 gs=256 专用 VF
}
```

#### 3.5.4 gs=128：AntiquantW4Pergroup128NK_GMM

最简单的 NK 实现。每个 group 恰好 128 个 K 元素 = 1 个向量寄存器宽度。

**参数**：

| 参数 | 计算方式 |
|------|----------|
| `groupNum` | `CeilDiv(bubKLen, 128)` |
| `bubNLen` | `bubNLen` |
| `innerStrideWeight` | `64`（128 INT4 = 64 字节） |
| `outerStrideWeight` | `CeilAlign(bubKLen, 32) >> 1` |
| `nTileAlign` | `CeilAlign(bubNLen, BLOCK_CUBE)` |
| `dataBlockStride` | `CeilAlign(bubNLen, 128) + 1` |

**循环结构**：

```text
for groupIdx = 0 to groupNum-1:
  for nBubIdx = 0 to bubNLen-1:
    // ★ GMM 专用 scale 地址：[G, N] 布局
    scale ← DIST_BRC_B16(scaleBase + groupIdx * nTileAlign + nBubIdx)
    if hasOffset: offset ← DIST_BRC_B16(offsetBase + groupIdx * nTileAlign + nBubIdx)
    weightB4 ← DIST_UNPACK4_B8(weightIn + groupIdx * innerStrideWeight + nBubIdx * outerStrideWeight)
    weightB16 ← Cast(weightB4)
    if hasOffset: weightB16 ← Add(weightB16, offset)
    weightB16 ← Mul(weightB16, scale)
    DataCopy<DATA_BLOCK_COPY>(weightOut, weightB16, ...)
  weightOutBase += weightOutAddrOffset
```

#### 3.5.5 gs=256：AntiquantW4Pergroup256NK_GMM

与 gs=128 差异：每个 group 256 K 元素 = 2 个向量寄存器，需 2 次 weight 加载、2 次输出。

| 差异点 | gs=128 | gs=256 |
|--------|--------|--------|
| weight 加载次数 | 1 | 2（偏移 64 字节） |
| `innerStrideWeight` | 64 | 128 |
| 输出 | 1 次 DATA_BLOCK_COPY | 2 次 |

#### 3.5.6 gs=64：AntiquantW4Pergroup64NK_GMM

每个 group 64 K 元素 = 半个向量寄存器，需 `Select` 拼接相邻 2 个 group 的 scale。

**循环结构**：

```text
for innerIdx = 0 to innerExtend-1:          // K 方向按 128 元素分块
  for outerIdx = 0 to outerExtend-1:         // N 方向逐行
    // 从 [G, N] 加载 2 个相邻 group 的 scale
    oriScale00 ← DIST_BRC_B16(scaleBase + (innerIdx*2)     * nTileAlign + outerIdx)
    oriScale01 ← DIST_BRC_B16(scaleBase + (innerIdx*2 + 1) * nTileAlign + outerIdx)
    scale0 ← Select(oriScale00, oriScale01, maskH)
    // offset 同理
    weightB4 ← DIST_UNPACK4_B8(weightIn + innerIdx*64 + ...)
    weightB16 ← Cast + Add(offset) + Mul(scale)
    DataCopy(weightOut, weightB16, ...)
  weightOutBase += outerStride
```

#### 3.5.7 gs=32：AntiquantW4Pergroup32NK_GMM

每个 group 32 K 元素 = 1/4 向量寄存器。

**核心差异**：GMM `[G,N]` 布局下，同一 N 列的 8 个 group scale 在 UB 中相隔 `nTileAlign`，不连续。**不能使用 `DIST_E2B_B16`**（要求源地址连续）。

**方案**：8 次 `DIST_BRC_B16` + 4 次 `Interleave` + 2 次 `Select`

```text
for innerIdx = 0 to innerExtend-1:          // K 方向按 256 元素分块
  for outerIdx = 0 to outerExtend-1:         // N 方向逐行
    // 8 次广播加载
    s0 ← DIST_BRC_B16(scaleBase + (innerIdx*8 + 0) * nTileAlign + outerIdx)
    s1 ← DIST_BRC_B16(scaleBase + (innerIdx*8 + 1) * nTileAlign + outerIdx)
    ... s7
    // Interleave 构造 64 元素对
    AB ← Interleave(s0, s1)    // [s0×64, s1×64]
    CD ← Interleave(s2, s3)
    EF ← Interleave(s4, s5)
    GH ← Interleave(s6, s7)
    // Select 拼接 128 元素寄存器
    scale0 ← Select(AB, CD, maskH)   // [s0×32, s1×32, s2×32, s3×32]
    scale1 ← Select(EF, GH, maskH)   // [s4×32, s5×32, s6×32, s7×32]
    // offset 同理
    // 2 次 weight 加载（各 128 INT4）
    weightB4Vl0 ← DIST_UNPACK4_B8(weightIn + innerIdx*128 + ...)
    weightB4Vl1 ← DIST_UNPACK4_B8(weightIn + 64 + innerIdx*128 + ...)
    // Cast + Add + Mul + Store（各 2 次）
```

**指令开销**：scale 部分 ~14 条指令（vs wqbmmv2 的 ~2 条）。gs=32 在实际模型中使用较少，初始实现优先正确性。

#### 3.5.8 NK 路径地址公式汇总

| groupSize | GMM scale 地址 | 加载方式 |
|-----------|---------------|----------|
| 128 | `base + g × nTileAlign + n` | DIST_BRC_B16 |
| 256 | `base + g × nTileAlign + n` | DIST_BRC_B16 |
| 64 | `base + g × nTileAlign + n` (2次) | DIST_BRC_B16 + Select |
| 32 | `base + (g×8+i) × nTileAlign + n` (8次) | DIST_BRC_B16 × 8 + Interleave × 4 + Select × 2 |

#### 3.5.9 双 AIV 切分影响

NK 路径 AIV 沿 N 切分，每个 AIV 处理 N/2（=128），K 完整（=256）。group 数不变。

- gs=32 NK：每个 AIV 处理 K=256（完整），对应 8 个 group。需要构造 2 个 128 元素 scale 寄存器。**已在 §3.5.7 给出方案。**

#### 3.5.10 验收标准

- [ ] gs=32/64/128/256 各计算正确
- [ ] 有/无 offset 均正确
- [ ] 转置=true 路径正确
- [ ] scale 地址公式使用 GMM `[G,N]` 布局，不使用 ops-nn `[N,G]` 布局
- [ ] perchannel NK 路径不受影响

---

### 3.6 子任务 8.6：编译验证 + 回归测试

**目标**：确保编译通过，已有测试不受影响

#### 3.6.1 编译验证

```bash
bash build.sh --opkernel --soc=ascend950 --ops=grouped_matmul -j8
```

> **注意**：当前环境 kernel 代码编写完成后无法直接运行 UT 验证，仅通过编译检查保证代码正确性。

#### 3.6.2 LSP Diagnostics

对所有修改文件运行 `lsp_diagnostics`，确保零错误。

#### 3.6.3 验收标准

- [ ] 编译通过（exit code 0）
- [ ] LSP diagnostics 零错误
- [ ] 无新增编译告警

> **注意**：当前环境 kernel 代码编写完成后无法直接运行 UT 验证。编译通过是主要的代码正确性检查手段。UT 验证需在有 NPU 硬件的环境中执行。

---

## 4. 修改文件汇总

| 序号 | 文件 | 子任务 | 修改量级 | 修改内容 |
|:----:|------|:------:|:--------:|----------|
| 1 | `basic_block_config.h` | 8.1 | 小 | 新增 4 个 WqmmConfig 实例 |
| 2 | `grouped_matmul_apt.cpp` | 8.1 | 中 | 新增 8 个 if constexpr 分支 |
| 3 | `grouped_matmul_def.cpp` | 8.1 | 小 | config950 新增 4 个 dtype 组合（antiquant_offset=null） |
| 4 | `grouped_matmul_weight_quant_basic_controller.h` | 8.2 | 小 | SetGmAddr pergroup 偏移 + bias 独立 |
| 5 | `weight_quant_vec_compute.h` | 8.3 | 中 | CalLocalAddrForVf + CopyAntiQuantParamsGmToUb + AntiQuantProcessNd |
| 6 | `basic_block_vf_nd.h` | 8.4, 8.5 | **大** | KN pergroup VF + NK pergroup VF（4 个 groupSize 专用函数） |

**不修改**：
- `grouped_matmul_tiling_data_apt.h` — `groupSize` 字段已存在
- `weight_quant_basic_block.h` — ComputeBasicBlock 分发复用
- `weight_quant_cube_compute.h` — Cube 计算复用
- `weight_quant_tiling_key.h` — 8 个 tiling key 已在任务 7 注册

---

## 5. 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| `CalLocalAddrForVf` 修改影响 perchannel | perchannel 路径回归 | `if constexpr` 隔离，回归测试验证 |
| gs=32 NK scale 加载指令开销大（~14 条 vs 2 条） | 性能下降 | 初始实现优先正确性，后续可用 AddrReg 优化 |
| `Select`/`Interleave` mask 构造不确定 | VF 函数实现错误 | 参考 ops-nn vf.h 源码和 Ascend C MicroAPI 文档确认 |
| bias 偏移独立可能影响已有逻辑 | 单单单 perchannel bias 地址错误 | 仅在 PER_GROUP 路径使用独立 biasBaseOffset |
| `CopyAntiQuantParamsGmToUb` 的 pergroup offset 搬运时机 | offset 未正确搬入 UB | 确认 PER_GROUP + hasAntiQuantOffset 分支被正确触发 |
| `VEC_ANTIQUANT_CONFIG_3` 参数与 pergroup 兼容性 | buffer 数量/大小不匹配 | 实现前确认 `{ubMte2BufferNum, ubMte2InnerSize}` 值 |

---

## 6. 执行顺序与依赖

```text
8.1 基础配置层 ──────────────────┐
  │                              │
  ↓                              │
8.2 Controller 层                │ 编译验证点：8.1+8.2 完成后可首次编译
  │                              │
  ↓                              │
8.3 VecCompute 层                │
  │                              │
  ↓                              │
8.4 VF — KN 路径 ──────┐         │
  │                    │         │
  ↓                    │         │
8.5 VF — NK 路径 ──────┤         │
  │                    │         │
  ↓                    ↓         ↓
8.6 编译验证 + 回归测试 ←─────────┘
```

**建议编译验证点**（编译命令：`bash build.sh --opkernel --soc=ascend950 --ops=grouped_matmul -j8`）：
1. 8.1 + 8.2 完成后：首次编译验证（确认 WqmmConfig 和入口分支无语法错误）
2. 8.3 完成后：编译验证（确认 VecCompute 修改无语法错误）
3. 8.4 + 8.5 完成后：完整编译验证

---

## 7. 关键代码参考索引

| 需要实现的功能 | 参考代码位置 | 行号 |
|--------------|-------------|------|
| pergroup GM 地址偏移 | resplit_controller.h | L260-262 |
| KN pergroup 参数构造 | wqbmmv2 reg_base_common.h | L1139-1180 |
| KN pergroup VF 函数 | wqbmmv2 vf.h `AntiquantPerGroupKN` | L347-443 |
| NK pergroup 分派逻辑 | wqbmmv2 reg_base_common.h `AntiquantComputeW4NKPerGroup` | L1185-1409 |
| NK gs=128 VF | wqbmmv2 vf.h `AntiquantW4Pergroup128NK` | L640-691 |
| NK gs=256 VF | wqbmmv2 vf.h `AntiquantW4Pergroup256NK` | L694-761 |
| NK gs=64 VF | wqbmmv2 vf.h `AntiquantW4Pergroup64NK` | L567-637 |
| NK gs=32 VF | wqbmmv2 vf.h `AntiquantW4Pergroup32NK` | L446-564 |
| ParamsKN 结构体 | wqbmmv2 vf.h | L76-104 |
| ParamsGroupSize128And256 | wqbmmv2 vf.h | L184-204 |
| ParamsGroupSize64 | wqbmmv2 vf.h | L162-181 |
| ParamsGroupSize32 | wqbmmv2 vf.h | L207-222 |
| PER_GROUP scale 搬运 | weight_quant_vec_compute.h | L515-519 |
| 现有 INT4 NdNk VF | basic_block_vf_nd.h `AntiQuantInt4NdNk` | L465-514 |
| 现有 INT4 NdKn VF | basic_block_vf_nd.h `AntiQuantInt4NdKn` | L516-567 |
| 现有 NdNkLoadScaleOffset | basic_block_vf_nd.h | L327-344 |
| 现有 NdKnLoadScaleOffset | basic_block_vf_nd.h | L346-368 |
| 现有 CalLocalAddrForVf | weight_quant_vec_compute.h | L1136-1176 |
| 现有 AntiQuantProcessNd | weight_quant_vec_compute.h | L1009-1052 |
