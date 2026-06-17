# 01 交付件清单 — GMM A16W4 PerGroup ND

> 本文档明确 A16W4 pergroup ND 项目的完整交付件清单，覆盖从 tiling 注册到 kernel 实现的全部新增/修改文件。
> 相关文档：[02 与硬件无关方案](./02_platform_independent_design.md)、[03 Tiling 方案](./03_tiling_design.md)、[04 Kernel 方案](./04_kernel_design.md)

---

## 1. Tiling 模板

### 1.1 结论：复用现有类，无需新增

**不新增 tiling 类**。复用现有 `GroupedWeightQuantBatchMatmulTiling`（`grouped_weight_quant_batch_matmul_tiling.{h,cpp}`），在其内部扩展 A16W4 pergroup ND 分支。

| 文件 | 路径 | 修改类型 |
|------|------|----------|
| tiling 类头文件 | `gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.h` | 修改（新增辅助函数声明） |
| tiling 类实现 | `gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp` | 修改（新增 pergroup 分支逻辑） |

**修改内容摘要**：
- 新增 `IsA16W4NDPergroup()` 判断函数
- `CheckGroupSize()` 新增 A16W4 pergroup 的 groupSize 校验（32/64/128/256）
- `CheckTensorDim*()` 适配 pergroup antiquantScale 3 维 shape
- `CheckTensorShape()` 适配 pergroup 整除校验
- `SetAntiquantGroupSize()` 确认 3 维分支正确解析 pergroup shape
- `SetTilingKey()` 中 `antiquantType` 设为 `PER_GROUP`

### 1.2 Tiling Data 结构

| 文件 | 路径 | 修改类型 |
|------|------|----------|
| tiling 数据结构 | `gmm/grouped_matmul/op_kernel/arch35/grouped_matmul_tiling_data_apt.h` | **无需修改** |

**说明**：`GMMWeightQuantParam` 已有 `groupSize` 字段（`uint32_t`），`GMMWeightQuantTilingData` 结构完整，无需新增字段。详见 [03 Tiling 方案 §2.1](./03_tiling_design.md#21-tiling-data-设计)。

---

## 2. Tiling 模板注册与分发

### 2.1 Tiling Key 注册

| 文件 | 路径 | 修改类型 |
|------|------|----------|
| tiling key 定义 | `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_tiling_key.h` | **修改（新增 ASCENDC_TPL_ARGS_SEL 组合）** |

**新增组合**（详见 [03 Tiling 方案 §2.5](./03_tiling_design.md#25-tiling-key-注册)）：

| 序号 | W_TYPE | OFFSET_OR_BIAS_EXIT | C_QUANT_TYPE | W_QUANT_TYPE | WQ_B_TRANS | ALGORITHM_SUB_CATEGORY | 说明 |
|:----:|--------|---------------------|--------------|--------------|------------|----------------------|------|
| 1 | WQGMM_ND | NOT_EXIST_BIAS_NOT_EXIST | WQGMM_NONE | **WQGMM_PER_GROUP** | WQGMM_NO_TRANS | N_FIRST_BASIC_BLOCK | pergroup 无offset无bias 非转置 |
| 2 | WQGMM_ND | NOT_EXIST_BIAS_NOT_EXIST | WQGMM_NONE | **WQGMM_PER_GROUP** | WQGMM_TRANS | N_FIRST_BASIC_BLOCK | pergroup 无offset无bias 转置 |
| 3 | WQGMM_ND | NOT_EXIST_BIAS_EXIST | WQGMM_NONE | **WQGMM_PER_GROUP** | WQGMM_NO_TRANS | N_FIRST_BASIC_BLOCK | pergroup 无offset有bias 非转置 |
| 4 | WQGMM_ND | NOT_EXIST_BIAS_EXIST | WQGMM_NONE | **WQGMM_PER_GROUP** | WQGMM_TRANS | N_FIRST_BASIC_BLOCK | pergroup 无offset有bias 转置 |
| 5 | WQGMM_ND | EXIST_BIAS_NOT_EXIST | WQGMM_NONE | **WQGMM_PER_GROUP** | WQGMM_NO_TRANS | N_FIRST_BASIC_BLOCK | pergroup 有offset无bias 非转置 |
| 6 | WQGMM_ND | EXIST_BIAS_NOT_EXIST | WQGMM_NONE | **WQGMM_PER_GROUP** | WQGMM_TRANS | N_FIRST_BASIC_BLOCK | pergroup 有offset无bias 转置 |
| 7 | WQGMM_ND | EXIST_BIAS_EXIST | WQGMM_NONE | **WQGMM_PER_GROUP** | WQGMM_NO_TRANS | N_FIRST_BASIC_BLOCK | pergroup 有offset有bias 非转置 |
| 8 | WQGMM_ND | EXIST_BIAS_EXIST | WQGMM_NONE | **WQGMM_PER_GROUP** | WQGMM_TRANS | N_FIRST_BASIC_BLOCK | pergroup 有offset有bias 转置 |

所有新增组合的公共参数：
- `KERNEL_TYPE_SEL`: `ASCENDC_TPL_MIX_AIC_1_2`
- `WQ_A_TRANS`: `WQGMM_NO_TRANS`（x 固定不转置）
- `TEMPLATE_CUSTOM_SC`: `WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4`
- `ALGORITHM_CATEGORY`: `WQGMM_VECTOR_ANTIQUANT`

### 2.2 Tiling 分发逻辑

`TilingGMM()` 中的分发逻辑**无需修改**。当前伪量化场景直接调用 `GroupedWeightQuantBatchMatmulTiling.SetTiling()`，pergroup 的差异化处理在 `SetTiling()` 内部通过分支实现。

**无需新增 `REGISTER_OPS_TILING_TEMPLATE` 注册**。

---

## 3. grouped_matmul_def.cpp 适配

| 文件 | 路径 | 修改类型 |
|------|------|----------|
| 算子定义 | `gmm/grouped_matmul/op_host/grouped_matmul_def.cpp` | 需要新增A16W4数据流 |

需要新增两组数据流：

第一组 支持float16
- x: float16, ND
- weight: int4, ND
- bias: float16, ND
- antiquant_scale: float16, ND
- antiquant_offset: float16, ND
- group_list: int64, ND

第二组 支持bfloat16
- x: bfloat16, ND
- weight: int4, ND
- bias: float32, ND
- antiquant_scale: bfloat16, ND
- antiquant_offset: bfloat16, ND
- group_list: int64, ND

---

## 4. grouped_matmul_apt.cpp Kernel 入口适配

| 文件 | 路径 | 修改类型 |
|------|------|----------|
| kernel 入口 | `gmm/grouped_matmul/op_kernel/grouped_matmul_apt.cpp` | **修改（新增 if constexpr 分支）** |

**修改内容**：

在 `V310_GMM_ANTI_QUANT` 分支的 `#else`（非 INT8/FP8 场景，即 A16 场景）中新增 `if constexpr` 分支。

**新增分支的模板参数匹配条件**：

```cpp
// A16W4 ND pergroup — 无offset无bias 非转置
else if constexpr (W_TYPE == WQGMM_ND &&
    OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST &&
    C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_GROUP &&
    WQ_B_TRANS == WQGMM_NO_TRANS && WQ_A_TRANS == WQGMM_NO_TRANS &&
    TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4 &&
    ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_BASIC_BLOCK &&
    ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {
    static constexpr WqmmConfig wqmmCfg = {false, false, QuantType::PER_GROUP, false,
                                           QuantType::NONE, CubeFormat::ND};
    INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(GMMWeightQuantBasicController, wqmmCfg,
                                                     VEC_ANTIQUANT_CONFIG_3);
}
```

类似地，需要为上述 8 种 tiling key 组合各新增一个 `if constexpr` 分支（共 8 个分支）。

**宏调用参数**：
- Controller 类：`GMMWeightQuantBasicController`
- `WqmmConfig`：新增实例（见 [04 Kernel 方案 §3.2](./04_kernel_design.md#32-basicblock-层适配)）
- `VecAntiQuantConfig`：复用 `VEC_ANTIQUANT_CONFIG_3`（`{4, 256}`）

**新增 using 声明**：
```cpp
using WeightQuantBatchMatmulV2::Arch35::A16W4_NDKN_PG;  // 新增
using WeightQuantBatchMatmulV2::Arch35::A16W4_NDNK_PG;  // 新增
```

---

## 5. Kernel 代码实现

### 5.1 新增/修改的 Kernel 文件

| 文件 | 路径 | 修改类型 | 修改内容 |
|------|------|----------|----------|
| **basic_block_config.h** | `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_config.h` | 修改 | 新增 `WqmmConfig` 实例、新增 `GetNdPerGroupBufferInfo()` |
| **basic_block_vf_nd.h** | `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_vf_nd.h` | **修改（主要新增代码）** | 新增 pergroup ND VF 函数 |
| **weight_quant_vec_compute.h** | `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_vec_compute.h` | 修改 | `AntiQuantProcessNd()` 新增 pergroup 分发 |
| **grouped_matmul_weight_quant_basic_controller.h** | `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/grouped_matmul_weight_quant_basic_controller.h` | 修改 | `SetGmAddr()` pergroup 地址偏移 |

### 5.2 新增 WqmmConfig 实例

在 `basic_block_config.h` 中新增：

```cpp
// A16W4 ND NK pergroup（B 非转置，weight shape = (N, K)）
static constexpr WqmmConfig A16W4_NDKN_PG = {false, false, QuantType::PER_GROUP, false, QuantType::NONE, CubeFormat::ND};
// A16W4 ND NK pergroup with offset
static constexpr WqmmConfig A16W4_NDKN_PG_OFF = {false, false, QuantType::PER_GROUP, true, QuantType::NONE, CubeFormat::ND};
// A16W4 ND KN pergroup（B 转置，weight shape = (K, N)）
static constexpr WqmmConfig A16W4_NDNK_PG = {false, true, QuantType::PER_GROUP, false, QuantType::NONE, CubeFormat::ND};
// A16W4 ND KN pergroup with offset
static constexpr WqmmConfig A16W4_NDNK_PG_OFF = {false, true, QuantType::PER_GROUP, true, QuantType::NONE, CubeFormat::ND};
```

### 5.3 新增 VF 函数

在 `basic_block_vf_nd.h` 中新增：

| 函数 | 排布 | 说明 |
|------|------|------|
| `AntiQuantInt4NdNkPerGroup` | NK（B 非转置） | pergroup ND NK 反量化，参照 wqbmmv2 `AntiquantComputeW4NKPerGroup` |
| `AntiQuantInt4NdKnPerGroup` | KN（B 转置） | pergroup ND KN 反量化，参照 wqbmmv2 `AntiquantComputeKNPerGroup` |

详见 [04 Kernel 方案 §3.4](./04_kernel_design.md#34-vec-反量化函数实现)。

### 5.4 Controller 修改

`SetGmAddr()` 中 `antiquantParamsBaseOffset` 更新公式修改：

```cpp
// 当前 perchannel:
antiquantParamsBaseOffset += offsetParam.nSize;

// 新增 pergroup 分支:
if constexpr (wqmmConfig.antiQuantType == QuantType::PER_GROUP) {
    antiquantParamsBaseOffset += (offsetParam.kSize / gmmBaseTiling_->groupSize) * offsetParam.nSize;
} else {
    antiquantParamsBaseOffset += offsetParam.nSize;
}
```

详见 [04 Kernel 方案 §3.1](./04_kernel_design.md#31-controller-层适配)。

---

## 6. 测试交付件

### 6.1 Tiling UT

| 文件 | 路径 | 修改类型 |
|------|------|----------|
| tiling UT | `gmm/grouped_matmul/tests/ut/op_host/test_grouped_weight_quant_arch35_tiling.cpp` | 修改（新增用例） |

**新增用例覆盖**：
- groupSize = 32/64/128/256 各场景
- 转置/非转置
- 单单单/多多多
- 有/无 antiquantOffset
- 有/无 bias
- 异常拦截：groupSize 非法、K % groupSize != 0、shape 不匹配等

### 6.2 Kernel UT

| 文件 | 路径 | 修改类型 |
|------|------|----------|
| kernel UT | `gmm/grouped_matmul/tests/ut/op_kernel/` 下新增或修改 | 新增用例 |

**新增用例覆盖**：
- x=FLOAT16/BFLOAT16 + weight=INT4 + 各 groupSize
- 各种 shape 组合（M=1, K=groupSize, N=1 等边界值）
- 精度验证（与 CPU 参考实现对比）

---

## 7. 编译配置

| 文件 | 路径 | 修改类型 |
|------|------|----------|
| CMakeLists.txt | `gmm/grouped_matmul/CMakeLists.txt` | **确认是否需要修改** |
| 测试 CMakeLists.txt | `gmm/grouped_matmul/tests/ut/op_kernel/CMakeLists.txt` | 可能需要新增测试目标 |

**编译宏**：
- 无需新增编译宏。A16W4 pergroup ND 复用现有 `V310_GMM_ANTI_QUANT` 宏路径
- `weight_quant_tiling_key.h` 中新增的 `ASCENDC_TPL_ARGS_SEL` 组合会自动被 tiling key 框架处理

---

## 8. 交付件汇总

### 8.1 需要修改的文件

| 序号 | 文件 | 修改量级 | 修改类型 |
|:----:|------|:--------:|----------|
| 1 | `grouped_weight_quant_batch_matmul_tiling.h` | 小 | 新增辅助函数声明 |
| 2 | `grouped_weight_quant_batch_matmul_tiling.cpp` | 中 | 新增 pergroup 校验/计算分支 |
| 3 | `weight_quant_tiling_key.h` | 小 | 新增 8 个 ASCENDC_TPL_ARGS_SEL 组合 |
| 4 | `grouped_matmul_def.cpp` | 小~中 | config950 新增 INT4 ND dtype 组合 |
| 5 | `grouped_matmul_apt.cpp` | 中 | 新增 8 个 if constexpr 分支 |
| 6 | `basic_block_config.h` | 小 | 新增 WqmmConfig 实例 + buffer 配置 |
| 7 | `basic_block_vf_nd.h` | **大** | 新增 pergroup VF 函数（核心新增代码） |
| 8 | `weight_quant_vec_compute.h` | 中 | AntiQuantProcessNd pergroup 分发 |
| 9 | `grouped_matmul_weight_quant_basic_controller.h` | 小 | SetGmAddr pergroup 偏移 |
| 10 | `test_grouped_weight_quant_arch35_tiling.cpp` | 中 | 新增 tiling UT 用例 |
| 11 | kernel UT 文件 | 中 | 新增 kernel UT 用例 |

### 8.2 无需修改的文件（确认可复用）

| 文件 | 说明 |
|------|------|
| `grouped_matmul_tiling_data_apt.h` | GMMWeightQuantParam 已有 groupSize 字段 |
| `weight_quant_basic_block.h` | BasicBlock 分发逻辑通用 |
| `weight_quant_cube_compute.h` | Cube 计算流程通用 |
| `weight_quant_vcv_basic_block.h` | VCV 路径不涉及 |
| `grouped_matmul_weight_quant_resplit_controller.h` | Resplit 路径不涉及（ND 非转置不使能尾块重切分） |
