# MoeReRouting 算子 FP8 scales 3D shape 支持修改方案

## 一、需求描述

当 `tokens` 输入类型为 `DT_FLOAT8_E5M2` 或 `DT_FLOAT8_E4M3FN` **且芯片平台为 Ascend950** 时，`per_token_scales` 除了现有的 shape 外，还需支持 3D shape：

- 现有 scales shape：
  - `[A]` — 每 token 1 个 scale（1D）
  - `[A, K/64]` — 每 token K/64 个 scale，按 64 通道分 block（2D）
- **新增** scales shape（仅 Ascend950）：
  - `[A, K/64, 2]` — 每 token K/64 个 block，每个 block 2 个 scale（3D）

> 变量约定：`A = tokenSum`（总 token 数），`H = tokenSize`（token 维度），`N = rankNum`（卡数），`E = expertNum`（专家数）

---

## 二、现有代码分析

### 2.1 核心算子逻辑

MoeReRouting 是一个纯数据搬运型算子（MoE 路由重排），对 scales 的操作本质上是 **per-token DataCopy**，不涉及 scales 数值的计算或解释。因此 scales 从 2D 扩展到 3D 时，只需将 scales 的**总元素数**（`K/64 * 2`）作为 `scaleSize_` 传入 kernel 即可。

### 2.2 现有 scales 处理流程

```
op_host/moe_re_routing_tiling_base.cpp :: GetShapeAttrsInfo()
  └─ 判断 scales shape 维度：
     ├─ 1D: scaleSize_ = 1
     └─ 2D: scaleSize_ = dim[1]（要求 dim[0] == tokenSum）

op_host/moe_re_routing_re_tiling.cpp :: PostTiling()
  └─ tilingData_.set_scaleSize(scaleSize_)（FP8+float 时乘以 4）

op_kernel/arch35/moe_re_routing_re_regbase.h :: ProcessTokenScale()
  └─ 按 scaleSize_ 进行 per-token DataCopy（与 token 数据搬运方式相同）
```

### 2.3 需要修改的文件清单

| 文件 | 修改范围 |
|:---|:---|
| `op_host/moe_re_routing_tiling_base.cpp` | `GetShapeTuple` 支持 3D、`GetShapeAttrsInfo` 解析 3D scales |
| `op_host/moe_re_routing_re_tiling.cpp` | `PrintTilingData` 增加 3D 日志信息（可选） |
| `op_host/moe_re_routing_r_tiling.cpp` | `PrintTilingData` 增加 3D 日志信息（可选） |
| `tests/ut/op_host/test_moe_re_routing_infershape.cpp` | 增加 3D scales 测试用例 |

> **kernel 层（op_kernel/）无需修改** — kernel 按 `scaleSize_` 做纯搬运，不关心 scales 的语义维度。

---

## 三、详细修改方案

### 修改 1：`op_host/moe_re_routing_tiling_base.cpp`

#### 1.1 修改 `GetShapeTuple` 函数

**现状**：`GetShapeTuple` 硬编码检查 `GetDimNum() != DIM_SIZE_TWO`（必须是 2D）。

**修改**：新增一个 `GetShapeTupleN` 函数，支持提取 N 维 shape 的总元素数，或直接修改现有函数使其不强制 2D 检查。

```cpp
// 新增：获取多维 shape 的总元素数（仅提取最后两个维度的乘积用于 scaleSize）
static std::tuple<int64_t, int64_t, int64_t> GetShapeTupleN(
    const gert::TilingContext *context, const int64_t index = 0)
{
    const gert::StorageShape *shapePtr = context->GetInputShape(index);
    OP_CHECK_IF(
        shapePtr == nullptr, OP_LOGE(context->GetNodeName(), "Shape is nullptr."),
        return std::make_tuple(0, 0, 0));
    auto dimNum = shapePtr->GetStorageShape().GetDimNum();
    OP_CHECK_IF(dimNum < 1 || dimNum > 3,
        OP_LOGE(context->GetNodeName(), "Shape must be 1D~3D, but get %zu.", dimNum),
        return std::make_tuple(0, 0, 0));
    if (dimNum == 1) {
        return std::make_tuple(shapePtr->GetStorageShape().GetDim(0), 1, 1);
    } else if (dimNum == 2) {
        return std::make_tuple(shapePtr->GetStorageShape().GetDim(0),
                               shapePtr->GetStorageShape().GetDim(1), 1);
    } else {  // dimNum == 3
        return std::make_tuple(shapePtr->GetStorageShape().GetDim(0),
                               shapePtr->GetStorageShape().GetDim(1),
                               shapePtr->GetStorageShape().GetDim(2));
    }
}
```

#### 1.2 修改 `GetShapeAttrsInfo` 函数中 scales 解析部分

**现状**（约第 239-257 行）：
```cpp
auto scaleShapePtr = context_->GetInputShape(IN_TOKEN_SCALE_INDEX);
if (scaleShapePtr != nullptr) {
    auto &scaleShape = context_->GetInputShape(IN_TOKEN_SCALE_INDEX)->GetStorageShape();
    if (scaleShape.GetDimNum() == 1) {
        scaleSize_ = 1;
    } else {
        auto scaleShapeTuple = GetShapeTuple(context_, IN_TOKEN_SCALE_INDEX);
        scaleSize_ = std::get<1>(scaleShapeTuple);
        auto scaleSum = std::get<0>(scaleShapeTuple);
        OP_CHECK_IF(scaleSum != tokenSum_, ...);
    }
    ...
}
```

**修改为**：
```cpp
auto scaleShapePtr = context_->GetInputShape(IN_TOKEN_SCALE_INDEX);
if (scaleShapePtr != nullptr) {
    auto &scaleShape = context_->GetInputShape(IN_TOKEN_SCALE_INDEX)->GetStorageShape();
    auto scaleDimNum = scaleShape.GetDimNum();
    if (scaleDimNum == 1) {
        scaleSize_ = 1;
    } else if (scaleDimNum == 2) {
        auto scaleShapeTuple = GetShapeTuple(context_, IN_TOKEN_SCALE_INDEX);
        scaleSize_ = std::get<1>(scaleShapeTuple);
        auto scaleSum = std::get<0>(scaleShapeTuple);
        OP_CHECK_IF(scaleSum != tokenSum_,
            OP_LOGE(context_->GetNodeName(),
                "scale tokenSum: %ld should equal to tokenSum: %ld.", scaleSum, tokenSum_),
            return ge::GRAPH_FAILED);
    } else if (scaleDimNum == 3) {
        // 新增：3D scales shape [A, K/64, 2]，仅 Ascend950 平台 + FP8 tokens 支持
        if (socVersion_ != platform_ascendc::SocVersion::ASCEND950) {
            OP_LOGE(context_->GetNodeName(),
                "3D scale shape is only supported on Ascend950 platform, actual platform %d.",
                static_cast<int>(socVersion_));
            return ge::GRAPH_FAILED;
        }
        if (tokenDtype_ != ge::DT_FLOAT8_E4M3FN && tokenDtype_ != ge::DT_FLOAT8_E5M2) {
            OP_LOGE(context_->GetNodeName(),
                "3D scale shape requires tokens to be DT_FLOAT8_E4M3FN or DT_FLOAT8_E5M2, actual %s.",
                ge::TypeUtils::DataTypeToSerialString(tokenDtype_).c_str());
            return ge::GRAPH_FAILED;
        }
        auto scaleShapeTuple3D = GetShapeTupleN(context_, IN_TOKEN_SCALE_INDEX);
        auto scaleSum = std::get<0>(scaleShapeTuple3D);
        auto scaleDim1 = std::get<1>(scaleShapeTuple3D);
        auto scaleDim2 = std::get<2>(scaleShapeTuple3D);
        OP_CHECK_IF(scaleSum != tokenSum_,
            OP_LOGE(context_->GetNodeName(),
                "scale tokenSum: %ld should equal to tokenSum: %ld.", scaleSum, tokenSum_),
            return ge::GRAPH_FAILED);
        // FP8 2-block scale：仅当第三维为 2 时支持
        OP_CHECK_IF(scaleDim2 != 2,
            OP_LOGE(context_->GetNodeName(),
                "3D scale shape dim[2] must be 2, actual %ld.", scaleDim2),
            return ge::GRAPH_FAILED);
        scaleSize_ = scaleDim1 * scaleDim2;  // K/64 * 2
    } else {
        OP_CHECK_IF(true,
            OP_LOGE(context_->GetNodeName(),
                "per_token_scales shape only support 1D~3D, actual %zuD.", scaleDimNum),
            return ge::GRAPH_FAILED);
    }
    auto scalePtr = context_->GetOptionalInputDesc(IN_TOKEN_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scalePtr);
    scaleDtype_ = scalePtr->GetDataType();
    hasScale_ = true;
}
```

---

### 修改 2：`op_host/moe_re_routing_re_tiling.cpp`

#### 2.1 修改 `PrintTilingData`（可选，用于调试日志）

在日志输出中增加 `blockNumE`、`blockFactorE` 信息（已有），无需额外修改。3D scales 的信息已经通过 `scaleSize_`（= K/64 * 2）体现。

---

### 修改 3：`op_host/moe_re_routing_r_tiling.cpp`

#### 3.1 修改 `PrintTilingData`（可选）

同上，无需额外修改。

---

### 修改 4：`tests/ut/op_host/test_moe_re_routing_infershape.cpp`

#### 4.1 新增 3D scales 正向测试用例

```cpp
TEST_F(MoeReRouting, MoeReRouting_infershape_3d_scale_fp8)
{
    // tokens: [256, 7168] FP8_E4M3FN
    // scales: [256, 112, 2]  即 K=7168, K/64=112, 每 block 2 个 scale
    gert::InfershapeContextPara infershapeContextPara("MoeReRouting",
                                                      {
                                                        {{{256, 7168}, {256, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                                        {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{256, 112, 2}, {256, 112, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {"expert_token_num_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                                        {"idx_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {256, 7168},   // permute_tokens
        {256, 112, 2}, // permute_per_token_scales（保持 3D）
        {256},         // permute_token_idx
        {16}           // expert_token_num
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
```

> **说明**：InferShape 测试不校验平台信息，仅验证 shape 透传正确性。3D scales 的平台拦截在 tiling 阶段（`GetShapeAttrsInfo`）执行，需通过 tiling UT 或 ST 验证。

#### 4.2 新增 tiling 负向测试用例（非 Ascend950 平台拒绝 3D scales）

在 tiling 测试中新增用例，模拟非 Ascend950 平台传入 3D scales，预期 tiling 返回 `GRAPH_FAILED`。

---

## 四、修改影响面分析

### 4.1 无需修改的部分

| 模块 | 原因 |
|:---|:---|
| `op_kernel/moe_re_routing.h` | 旧版 kernel 按 `tl_->hasScale` 判断是否搬运 scales，按 `scaleSize` 做 DataCopy，不关心维度 |
| `op_kernel/arch35/moe_re_routing_re_regbase.h` | 同上，`ProcessTokenScale` 按 `scaleSize_` 搬运，语义不变 |
| `op_kernel/arch35/moe_re_routing_r_regbase.h` | 同上，`ProcessTokenScale/CopyIn/CopyOut` 按 `scaleSize_` 搬运 |
| `op_kernel/moe_re_routing_apt.cpp` | kernel 入口，模板参数和 tiling key 选择逻辑不变 |
| `moe_re_routing_infershape.cpp` | InferShape 直接透传 scales 形状（`*permutePerTokenScalesShape = *perTokenScaleShape`），天然支持 3D |

### 4.2 需要修改的部分

| 模块 | 修改内容 |
|:---|:---|
| `moe_re_routing_tiling_base.cpp` | 增加 3D scales shape 解析逻辑（核心修改） |
| `test_moe_re_routing_infershape.cpp` | 新增 3D scales UT 用例 |

### 4.3 向后兼容性

- 现有 1D `[A]` 和 2D `[A, K/64]` scales 逻辑**完全不受影响**
- 新增的 3D 分支通过 `scaleDimNum == 3` 隔离
- 对非 FP8 类型的 tokens 传入 3D scales 时，`CheckDtypeAndAttr` 中的现有校验逻辑不变（scales dtype 校验与维度无关）
- **非 Ascend950 平台传入 3D scales 会被 tiling 层拦截**，报错明确提示平台限制

### 4.4 平台限制

| 芯片平台 | 3D scales `[A, K/64, 2]` 支持 |
|:---|:---|
| Ascend950PR/Ascend950DT | 支持 |
| Ascend910B / Ascend910_93 | 不支持（报错拦截） |
| Kirin X90 | 不支持（报错拦截） |
| Kirin 9030 | 不支持（报错拦截） |

**拦截位置**：`GetShapeAttrsInfo()` 函数中，3D shape 分支入口处检查 `socVersion_`。

---

## 五、约束条件

1. **仅 Ascend950 平台支持 3D scales**：非 Ascend950 平台传入 3D shape 会在 tiling 阶段报错拦截。
2. **3D scales 仅在 tokens 为 FP8 类型时有意义**：虽然 tiling 层面不强制，但 `CheckDtypeAndAttr` 中已有 FP8 tokens 必须搭配 float/float8_e8m0 scales 的校验。
3. **第三维必须为 2**：对应每个 64 通道 block 内的两个子 scale，其他值将报错。
4. **第一维必须等于 tokenSum**：与现有 2D scales 的约束一致。
5. **第二维必须整除 tokenSize**：即 `K/64` 必须满足 `K % 64 == 0`，这是 FP8 block scaling 的固有约束。

---

## 六、验证方案

1. **UT 验证**：新增 `MoeReRouting_infershape_3d_scale_fp8` 测试用例，覆盖 InferShape 和 InferDtype。
2. **ST 验证（Ascend950）**：在 Ascend950 平台上执行包含 3D scales 的算子调用，验证：
   - 算子编译通过（tiling 正确生成 `scaleSize_ = K/64 * 2`）
   - 输出 `permute_per_token_scales` shape 与输入一致（3D 透传）
   - 输出数值正确（scales 数据跟随 tokens 正确重排）
3. **负向验证（非 Ascend950）**：在 Ascend910B 等非 Ascend950 平台上执行相同调用，预期 tiling 阶段报错退出。
