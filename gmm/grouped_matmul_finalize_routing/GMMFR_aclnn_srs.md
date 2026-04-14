# aclnnGroupedMatmulFinalizeRoutingWeightNzV2 算子aclnn通路SRS

## 1. 迭代信息

| 项目 | 内容 |
| ---- | --- |
| 迭代名称 | GMMFR MxA8W4 Aclnn通路支持 |
| 迭代负责人 | 孙堆杨 |
| 更新时间 | 2026.04.07 |

## 2. 需求背景

GMMFR融合算子支持MxA8W4 weightNz支持。新增MxA8W4量化数据流：激活为float8_e4m3，权重为float4_e2m1，通过scale（x2scale，float8_e8m0）和pertokenScale（x1scale, float8_e8m0）进行反量化。scale参数复用现有V2接口的`scale`入参，无需新增参数。

计算公式：

- 1.分组矩阵乘法GMM（MxA8W4反量化过程）：

  $$
  y_i = (x_i * perTokenScale_i) \times (w_i * scale_i)  + bias_i
  $$

  其中 $x_i$ 为float8_e4m3类型激活，$weight_i$ 为float4_e2m1类型权重，$scale_i$ 为float8_e8m0类型权重反量化因子，$perTokenScale_i$ 为float8_e8m0类型激活反量化因子。

- 2.路由专家与专家输出分配：

  对于每个token j，执行路由与输出专家分配：

  $$
  y[rowIndex[i], :] = y[rowIndex[i], :] + y_{i(j)}[j - start_{i(j)}] * logit[i]
  $$

  其中 $i(j)$ 是token j被分配到的专家索引，$y_{i(j)}[j - start_{i(j)}]$ 是该token在对应专家下的计算结果，$logit[i]$ 为MoE专家logit权重。

- 3.共享专家输出融合：

  $$
  out[output_{bs}, n] = 0
  $$

  $$
  out[sharedInputOffset : sharedInputOffset + bsdp, :] = sharedInputWeight \times sharedInput[:, :]
  $$

- 4.最终输出：

  $$
  out[rowIndex[i], :] = \sum_{i \in \mathcal{E}[j]} y_i[j - start_i] * logit[i] + sharedInputWeight \times sharedInput[j, :]
  $$

  其中 $\mathcal{E}[j]$ 表示分配给token j的专家集合。

## 3. 规格约束

### 3.1 输入规格 — Tensor 参数

| 参数名 | 必选/可选 | dtype | format | shape | 说明 |
| ------ | --------- | ----- | ------ | ----- | ---- |
| x1 | 必选 | float8_e4m3fn | ND | (m, k) | 左矩阵（激活），固定不转置 |
| x2 | 必选 | float4_e2m1 | [^1] NZ_C0_32 | viewShape=(e, n, k)</br>oriShape=(e, ceil(k, 32), ceil(n, 16), 16, 32) | 右矩阵（权重），固定转置 |
| scale | 必选 | float8_e8m0 | ND | (e, n, ceil(k/64), 2) | MX weight 反量化缩放因子 |
| bias | 可选 | bfloat16 | ND | (e, n) | 偏置 |
| offsetOptional | 不使用 | - | - | - | MxA8W4场景为null |
| antiquantScaleOptional | 不使用 | - | - | - | MxA8W4场景为null |
| antiquantOffsetOptional | 不使用 | - | - | - | MxA8W4场景为null |
| pertokenScaleOptional | 必选 | float8_e8m0 | ND | (m, ceil(k/64), 2) | MX x 反量化缩放因子 |
| groupList | 必选 | int64 | ND | (e,) | 分组列表，表示各专家的token分布 |
| sharedInput | 可选 | bfloat16 | ND | (bsdp, n) | 共享专家输入，bsdp <= outputBS |
| logit | 必选 | float32 | ND | (m,) | MoE专家logit权重 |
| rowIndex | 必选 | int64 | ND | (m,) | scatter-add行索引 |

[^1]: 注意weight的format格式为C0_32而不是C0_64，原因是weight以float4_e2m1类型加载权重，但类型转换为float8_e4m3进行计算，所以实际C0_SIZE以float8_e4m3进行计算，得到 `32B / sizeof(float8_e4m3) = 32` ，所以format格式为NZ_C0_32。

### 3.2 输入规格 — Scalar / 属性参数

| 参数名 | 类型 | 取值 | 说明 |
| ------ | ---- | ---- | ---- |
| dtype | int64 | {0} | 输出数据类型，固定为0（float32） |
| sharedInputWeight | float32 | - | 共享专家combine系数 |
| sharedInputOffset | int64 | >= 0 | 共享专家输出在总输出中的偏移 |
| transposeX1 | bool | {false} | 左矩阵不转置，固定false |
| transposeX2 | bool | {true} | 右矩阵转置，固定true |
| groupListType | int64 | {0, 1} | 0: cumsum模式, 1: count模式 [^1] |
| tuningConfigOptional | int64数组 | - | 调优参数，可选，本数据流不需要 |

[^1]: cumsum模式：即前缀和形式表示的每个group的大小；count模式：即直接给出每个group的大小。

### 3.3 约束条件

| 约束项 | 条件 | 说明 |
| ------ | ---- | ---- |
| k对齐 | k % 32 == 0 | k维度需32对齐 |
| n对齐 | n % 32 == 0 | n维度需32对齐 |
| 专家数上限 | e <= 1024 | 专家数量不超过1024 |
| sharedInput batch | bsdp <= outputBS | 共享专家batch不超过输出batch |
| offset范围 | sharedInputOffset + bsdp <= outputBS | 偏移加共享专家batch不超输出batch |

### 3.4 输出规格

| 参数名 | dtype | format | shape | 说明 |
| ------ | ----- | ------ | ----- | ---- |
| out | float32 | ND | (outputBS, n) | 融合计算结果 |

## 4. Aclnn GMMFR MxA8W4通路支持修改点

MxA8W4场景直接复用现有V2接口的`scale`参数传入float8_e8m0权重缩放因子，`antiquantScaleOptional`保持不使用。因此无需新增参数、无需修改Params结构体/L0接口/V2 API签名。核心修改集中在算子原型dtype组合、Checker校验逻辑和CommonProcess数据流适配。

### 4.0 算子原型新增A8W4 dtype组合

**源代码相对路径**: `gmm/grouped_matmul_finalize_routing/op_host/grouped_matmul_finalize_routing_def.cpp`

**修改点描述**: 当前 `config91095`（ascend950配置）中无A8W4的dtype组合（x=float8_e4m3fn, w=float4_e2m1）。需要在 `config91095` 的各Input/Output中新增一行A8W4对应的dtype，使算子原型支持该数据流。

**源代码/伪代码**:

在 `config91095` 的各 Input/Output DataType 列表末尾追加A8W4 dtype：

``` cpp
// x (activation)
config91095.Input("x")
    .ParamType(REQUIRED)
    .DataType({..., ge::DT_FLOAT8_E4M3FN})  // 新增A8W4: x=FP8_E4M3FN
    ...

// w (weight) - NZ格式
config91095.Input("w")
    .ParamType(REQUIRED)
    .DataType({..., ge::DT_FLOAT4_E2M1})    // 新增A8W4: w=FP4_E2M1
    .Format({..., ge::FORMAT_FRACTAL_NZ})    // A8W4权重为NZ格式
    ...

// scale (weight scale)
config91095.Input("scale")
    .ParamType(REQUIRED)
    .DataType({..., ge::DT_FLOAT8_E8M0})    // 新增A8W4: scale=FP8_E8M0
    ...

// bias
config91095.Input("bias")
    .ParamType(OPTIONAL)
    .DataType({..., ge::DT_BF16})            // 新增A8W4: bias=BF16
    ...

// pertoken_scale (activation scale)
config91095.Input("pertoken_scale")
    .ParamType(OPTIONAL)
    .DataType({..., ge::DT_FLOAT8_E8M0})    // 新增A8W4: pertokenScale=FP8_E8M0
    ...

// 其余Input (group_list, shared_input, logit, row_index, offset) 和 Output (y) 
// 的dtype与现有MX场景一致，均需追加对应行
```

---

### 4.1 新增MxA8W4专用Checker，与原有Checker解耦

**源代码相对路径**: `gmm/grouped_matmul_finalize_routing/op_api/grouped_matmul_finalize_routing_weight_quant_950_checker.h`（新增）及 `.cpp`（新增）

**修改点描述**: 原有 `AclnnGroupedMatmulFinalizeRoutingDAV3510Checker` 的校验逻辑中存在以下与A8W4不兼容的设计：
1. `CheckDtypeValidForMX` 末尾的FP4/FP8交叉校验要求x1和x2必须同时为FP4或同时为FP8，A8W4是FP8+FP4混合组合会被拦截
2. `CheckXAndWeightFormat` 中MX模式要求权重为ND格式，A8W4权重为NZ_C0_32格式会被拦截

为保持代码解耦，不在原有checker中打补丁，而是新建独立的 `grouped_matmul_finalize_routing_weight_quant_950_checker.h/.cpp`，专门处理MxA8W4场景的参数校验。原有checker不做任何修改。

**源代码/伪代码**:

头文件 `grouped_matmul_finalize_routing_weight_quant_950_checker.h`：

``` cpp
#ifndef OP_API_INC_GROUPED_MATMUL_FINALIZE_ROUTING_WEIGHT_QUANT_950_CHECKER_H
#define OP_API_INC_GROUPED_MATMUL_FINALIZE_ROUTING_WEIGHT_QUANT_950_CHECKER_H

#include "opdev/format_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "quant_grouped_matmul_finalize_routing_util.h"
#include "util/math_util.h"

namespace GmmFinalizeRouting {

constexpr size_t ZERO_DIM = 0UL;
constexpr size_t ONE_DIM = 1UL;
constexpr size_t TWO_DIM = 2UL;
constexpr size_t FOUR_DIM = 4UL;
constexpr int64_t GMMFR_SPLIT_SIZE = 64L;
constexpr int64_t GMMFR_SPLIT_FACTOR = 2L;
constexpr int64_t MAX_NUM_EXPERTS = 1024L;

static const std::initializer_list<op::DataType> A8W4_X_TYPE_LIST = {op::DataType::DT_FLOAT8_E4M3FN};
static const std::initializer_list<op::DataType> A8W4_W_TYPE_LIST = {op::DataType::DT_FLOAT4_E2M1};
static const std::initializer_list<op::DataType> A8W4_SCALE_TYPE_LIST = {op::DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<op::DataType> A8W4_PERTOKEN_SCALE_TYPE_LIST = {op::DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<op::DataType> A8W4_BIAS_TYPE_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> A8W4_SHARED_INPUT_TYPE_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> A8W4_LOGIT_TYPE_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> A8W4_GROUP_LIST_TYPE_LIST = {op::DataType::DT_INT64};
static const std::initializer_list<op::DataType> A8W4_ROW_INDEX_TYPE_LIST = {op::DataType::DT_INT64};
static const std::initializer_list<op::DataType> A8W4_OUT_TYPE_LIST = {op::DataType::DT_FLOAT};

class AclnnGroupedMatmulFinalizeRoutingWeightQuantChecker {
public:
    explicit AclnnGroupedMatmulFinalizeRoutingWeightQuantChecker() {};
    ~AclnnGroupedMatmulFinalizeRoutingWeightQuantChecker() {};

    aclnnStatus CheckParams(GroupedMatmulParams &gmmParams)
    {
        gmmParams_ = gmmParams;
        CHECK_RET(CheckNotNull() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
        CHECK_RET(CheckDtypeValid(), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckInputOutDims() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckInputOutShape() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckFormat() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        return ACLNN_SUCCESS;
    }

private:
    GroupedMatmulParams gmmParams_;

    aclnnStatus CheckNotNull()
    {
        CHECK_COND(gmmParams_.x1 != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x should not be nullptr.");
        CHECK_COND(gmmParams_.x2 != nullptr, ACLNN_ERR_PARAM_NULLPTR, "weight should not be nullptr.");
        CHECK_COND(gmmParams_.scale != nullptr, ACLNN_ERR_PARAM_NULLPTR, "scale should not be nullptr.");
        CHECK_COND(gmmParams_.pertokenScaleOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "pertokenScaleOptional should not be nullptr.");
        CHECK_COND(gmmParams_.groupList != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "groupList should not be nullptr.");
        CHECK_COND(gmmParams_.logit != nullptr, ACLNN_ERR_PARAM_NULLPTR, "logit should not be nullptr.");
        CHECK_COND(gmmParams_.rowIndex != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "rowIndex should not be nullptr.");
        CHECK_COND(gmmParams_.out != nullptr, ACLNN_ERR_PARAM_NULLPTR, "out should not be nullptr.");
        return ACLNN_SUCCESS;
    }

    bool CheckDtypeValid()
    {
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.x1, A8W4_X_TYPE_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.x2, A8W4_W_TYPE_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.scale, A8W4_SCALE_TYPE_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.pertokenScaleOptional, A8W4_PERTOKEN_SCALE_TYPE_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.rowIndex, A8W4_ROW_INDEX_TYPE_LIST, return false);
        if (gmmParams_.bias != nullptr) {
            OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.bias, A8W4_BIAS_TYPE_LIST, return false);
        }
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.groupList, A8W4_GROUP_LIST_TYPE_LIST, return false);
        if (gmmParams_.shareInput != nullptr) {
            OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.shareInput, A8W4_SHARED_INPUT_TYPE_LIST, return false);
        }
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.logit, A8W4_LOGIT_TYPE_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmParams_.out, A8W4_OUT_TYPE_LIST, return false);
        return true;
    }

    aclnnStatus CheckInputOutDims()
    {
        auto xDimNumber = gmmParams_.x1->GetViewShape().GetDimNum();
        auto wDimNumber = gmmParams_.x2->GetViewShape().GetDimNum();
        auto scaleDimNumber = gmmParams_.scale->GetViewShape().GetDimNum();
        auto pertokenScaleDimNumber = gmmParams_.pertokenScaleOptional->GetViewShape().GetDimNum();
        auto grouplistDimNumber = gmmParams_.groupList->GetViewShape().GetDimNum();
        auto logitDimNumber = gmmParams_.logit->GetViewShape().GetDimNum();
        auto rowindexDimNumber = gmmParams_.rowIndex->GetViewShape().GetDimNum();
        auto outDimNumber = gmmParams_.out->GetViewShape().GetDimNum();
        CHECK_COND(xDimNumber == TWO_DIM, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of x should be 2, current dim is %lu.", xDimNumber);
        CHECK_COND(wDimNumber == 3, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of w should be 3, current dim is %lu.", wDimNumber);
        CHECK_COND(scaleDimNumber == FOUR_DIM, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of scale should be 4, current dim is %lu.", scaleDimNumber);
        CHECK_COND(pertokenScaleDimNumber == 3, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of pertokenScale should be 3, current dim is %lu.", pertokenScaleDimNumber);
        CHECK_COND(grouplistDimNumber == ONE_DIM, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of grouplist should be 1, current dim is %lu.", grouplistDimNumber);
        CHECK_COND(logitDimNumber == ONE_DIM, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of logit should be 1, current dim is %lu.", logitDimNumber);
        CHECK_COND(rowindexDimNumber == ONE_DIM, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of rowindex should be 1, current dim is %lu.", rowindexDimNumber);
        CHECK_COND(outDimNumber == TWO_DIM, ACLNN_ERR_PARAM_INVALID,
                   "The dim num of out should be 2, current dim is %lu.", outDimNumber);
        if (gmmParams_.bias != nullptr) {
            CHECK_COND(gmmParams_.bias->GetViewShape().GetDimNum() == TWO_DIM, ACLNN_ERR_PARAM_INVALID,
                       "The dim num of bias should be 2.");
        }
        if (gmmParams_.shareInput != nullptr) {
            CHECK_COND(gmmParams_.shareInput->GetViewShape().GetDimNum() == TWO_DIM, ACLNN_ERR_PARAM_INVALID,
                       "The dim num of shareInput should be 2.");
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckInputOutShape()
    {
        int64_t m = gmmParams_.x1->GetViewShape().GetDim(ZERO_DIM);
        int64_t k = gmmParams_.x1->GetViewShape().GetDim(ONE_DIM);
        int64_t n = gmmParams_.transposeX2 ? (gmmParams_.x2)->GetViewShape().GetDim(ONE_DIM) :
                                              (gmmParams_.x2)->GetViewShape().GetDim(TWO_DIM);
        int64_t e = (gmmParams_.x2)->GetViewShape().GetDim(ZERO_DIM);
        int64_t outputBS = gmmParams_.out->GetViewShape().GetDim(ZERO_DIM);

        CHECK_COND(k > 0, ACLNN_ERR_PARAM_INVALID, "k should be positive, but got %ld.", k);

        // x: (m, k)
        op::Shape xExpectShape = {m, k};
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.x1, xExpectShape, return ACLNN_ERR_PARAM_INVALID);

        // w: transposeX2=true -> (e, n, k); false -> (e, k, n)
        if (gmmParams_.transposeX2) {
            op::Shape wExpectShape = {e, n, k};
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.x2, wExpectShape, return ACLNN_ERR_PARAM_INVALID);
        } else {
            op::Shape wExpectShape = {e, k, n};
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.x2, wExpectShape, return ACLNN_ERR_PARAM_INVALID);
        }

        // scale: transposeX2=true -> (e, n, ceil(k/64), 2); false -> (e, ceil(k/64), n, 2)
        if (gmmParams_.transposeX2) {
            op::Shape scaleExpectShape = {e, n, Ops::Base::CeilDiv(k, GMMFR_SPLIT_SIZE), GMMFR_SPLIT_FACTOR};
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.scale, scaleExpectShape, return ACLNN_ERR_PARAM_INVALID);
        } else {
            op::Shape scaleExpectShape = {e, Ops::Base::CeilDiv(k, GMMFR_SPLIT_SIZE), n, GMMFR_SPLIT_FACTOR};
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.scale, scaleExpectShape, return ACLNN_ERR_PARAM_INVALID);
        }

        // pertokenScale: (m, ceil(k/64), 2)
        op::Shape pertokenScaleExpectShape = {m, Ops::Base::CeilDiv(k, GMMFR_SPLIT_SIZE), GMMFR_SPLIT_FACTOR};
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.pertokenScaleOptional, pertokenScaleExpectShape, return ACLNN_ERR_PARAM_INVALID);

        // groupList: (e,)
        op::Shape grouplistExpectShape = {e};
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.groupList, grouplistExpectShape, return ACLNN_ERR_PARAM_INVALID);

        // logit: (m,)
        op::Shape logitExpectShape = {m};
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.logit, logitExpectShape, return ACLNN_ERR_PARAM_INVALID);

        // rowIndex: (m,)
        op::Shape rowindexExpectShape = {m};
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.rowIndex, rowindexExpectShape, return ACLNN_ERR_PARAM_INVALID);

        // out: (outputBS, n)
        op::Shape outputExpectShape = {outputBS, n};
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.out, outputExpectShape, return ACLNN_ERR_PARAM_INVALID);

        // bias: (e, n)
        if (gmmParams_.bias != nullptr) {
            op::Shape biasExpectShape = {e, n};
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.bias, biasExpectShape, return ACLNN_ERR_PARAM_INVALID);
        }

        // shareInput: (bsdp, n), bsdp <= outputBS, sharedInputOffset + bsdp <= outputBS
        if (gmmParams_.shareInput != nullptr) {
            int64_t bsdp = gmmParams_.shareInput->GetViewShape().GetDim(ZERO_DIM);
            CHECK_COND(bsdp <= outputBS, ACLNN_ERR_PARAM_INVALID,
                       "shareInput batch %ld should <= outputBS %ld.", bsdp, outputBS);
            CHECK_COND(gmmParams_.shareInputOffset + bsdp <= outputBS, ACLNN_ERR_PARAM_INVALID,
                       "sharedInputOffset(%ld) + shareInput batch(%ld) should <= outputBS(%ld), but got %ld.",
                       gmmParams_.shareInputOffset, bsdp, outputBS, gmmParams_.shareInputOffset + bsdp);
            op::Shape shareInputExpectShape = {bsdp, n};
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.shareInput, shareInputExpectShape, return ACLNN_ERR_PARAM_INVALID);
        }

        // 专家数上限
        CHECK_COND(e <= MAX_NUM_EXPERTS, ACLNN_ERR_PARAM_INVALID,
                   "e must be <= 1024, but got %ld.", e);

        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckFormat()
    {
        // x1: ND
        if (op::IsPrivateFormat(gmmParams_.x1->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x should be ND format.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // weight: NZ格式（A8W4场景权重为NZ）
        if (gmmParams_.x2->GetStorageFormat() != Format::FORMAT_FRACTAL_NZ) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight should be NZ format, current is %s.",
                    op::ToString(gmmParams_.x2->GetStorageFormat()).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
        // scale, pertokenScale, bias, groupList, logit, rowIndex, shareInput, out: ND
        auto checkND = [](const aclTensor *t, const char *name) -> aclnnStatus {
            if (t != nullptr && op::IsPrivateFormat(t->GetStorageFormat())) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s should be ND format.", name);
                return ACLNN_ERR_PARAM_INVALID;
            }
            return ACLNN_SUCCESS;
        };
        CHECK_RET(checkND(gmmParams_.scale, "scale"), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(checkND(gmmParams_.pertokenScaleOptional, "pertokenScale"), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(checkND(gmmParams_.bias, "bias"), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(checkND(gmmParams_.groupList, "groupList"), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(checkND(gmmParams_.logit, "logit"), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(checkND(gmmParams_.rowIndex, "rowIndex"), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(checkND(gmmParams_.shareInput, "shareInput"), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(checkND(gmmParams_.out, "out"), ACLNN_ERR_PARAM_INVALID);
        return ACLNN_SUCCESS;
    }
};

} // namespace GmmFinalizeRouting
#endif
```

调用侧修改 — 在 `aclnn_grouped_matmul_finalize_routing.cpp` 的 `CheckParams` 函数中（第516行），根据场景分发到新Checker：

``` cpp
static aclnnStatus CheckParams(GroupedMatmulParams &params)
{
    bool isA8W4 = params.x1 != nullptr && params.x2 != nullptr &&
                  params.x1->GetDataType() == DataType::DT_FLOAT8_E4M3FN &&
                  params.x2->GetDataType() == DataType::DT_FLOAT4_E2M1;

    if (isA8W4 && op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        // MxA8W4场景：使用专用Checker，与原有Checker解耦
        GmmFinalizeRouting::AclnnGroupedMatmulFinalizeRoutingWeightQuantChecker checker;
        return checker.CheckParams(params);
    }

    // 原有场景：保持不变
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        GmmFinalizeRouting::AclnnGroupedMatmulFinalizeRoutingDAV3510Checker checker;
        aclnnStatus status = checker.CheckParams(params);
        CHECK_RET(status == ACLNN_SUCCESS, status);
    } else {
        // ... 原有非950逻辑
    }
    return ACLNN_SUCCESS;
}
```

---

### 4.2 PreMatmulCalcProcess中scale的transpose一致性检查适配A8W4

**源代码相对路径**: `gmm/grouped_matmul_finalize_routing/op_api/aclnn_grouped_matmul_finalize_routing.cpp`

**修改点描述**: `PreMatmulCalcProcess`（第798行）中，对MX场景的scale进行transpose处理后，检查 `transposeX2 == transposescale`（第815行）。A8W4场景下scale为float8_e8m0、shape为(e, n, ceil(k/64), 2)，`WeightNZCaseProcessForMXScale`处理后 `transposescale=false`，而 `transposeX2=true`，导致一致性检查不通过。需要为A8W4场景跳过或调整该检查。

**源代码/伪代码**:

当前代码（第812-816行）：

``` cpp
    if (scale != nullptr && CheckType(x1->GetDataType(), X_WEIGHT_TYPE_SUPPORT_LIST_MX) &&
        CheckType(scale->GetDataType(), SCALE_TYPE_SUPPORT_LIST_MX)) {
        bool transposescale = false;
        ret = WeightNZCaseProcessForMXScale(scale, transposescale, executor);
        CHECK_RET(transposeX2 == transposescale, ret);
    }
```

修改为：

``` cpp
    if (scale != nullptr && CheckType(x1->GetDataType(), X_WEIGHT_TYPE_SUPPORT_LIST_MX) &&
        CheckType(scale->GetDataType(), SCALE_TYPE_SUPPORT_LIST_MX)) {
        bool transposescale = false;
        ret = WeightNZCaseProcessForMXScale(scale, transposescale, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        // A8W4场景：scale已按transposeX2对应的shape传入，无需transpose一致性检查
        bool isA8W4 = x1->GetDataType() == DataType::DT_FLOAT8_E4M3FN &&
                      x2->GetDataType() == DataType::DT_FLOAT4_E2M1;
        if (!isA8W4) {
            CHECK_RET(transposeX2 == transposescale, ret);
        }
    }
```
