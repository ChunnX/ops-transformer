# GMM MxA8W4 单多单场景支持 SRS

---

## 1. 文档信息

| 项目              | 内容                          |
| ----------------- | ----------------------------- |
| 文档标题          | GMM MxA8W4 单多单场景支持 SRS |
| 迭代/特性名称     | GMM MxA8W4 单多单支持         |
| 算子类型          | 单算子                        |
| 主领域目录        | gmm/grouped_matmul            |
| 责任人            | 待定                          |
| 更新时间          | 2026-06-11                    |
| 文档版本          | v0.1                          |
| 关联 STC          | 待补充                        |
| 关联里程碑/需求单 | 无                            |

---

## 2. 需求背景

### 2.1 背景

GMM（Grouped Matmul）在 arch35 MxA8W4 伪量化数据流下已有 resplit controller 实现。现有地址更新逻辑默认按单 tensor 内部连续 offset 推进：

```cpp
xGm_ += mSize * kSize;
weightGm_ += (nSize * kSize) >> 1;
antiquantScaleGm_ += nSize * CeilDivide(kSize, groupSize);
biasGm_ += nSize;
yGm_ += mSize * nSize;
```

该逻辑适用于 weight 为单 tensor、group 维度在同一个 tensor 内展开的场景。

新需求要求 MxA8W4 支持“单多单”：

- `x` 为单 tensor；
- `weight` 为多 tensor；
- `weightScale` 为多 tensor，在代码中对应 `antiquantScaleOptional` / `antiquantScaleGm_`；
- `bias` 为多 tensor，当 `biasOptional` 存在时长度与 `weight` 相同；
- `y` 为单 tensor。

在该场景中，`weight`、`weightScale`、`bias` 需要按 group 从 tensor list 中取地址，不能继续按当前 tensor 内部 offset 推进。

### 2.2 目标

本轮交付目标：

1. 为 GMM MxA8W4 resplit controller 增加编译期单多单模板参数；
2. 通过 TilingKey 传递单多单信息，避免 kernel 宏中运行时判断；
3. 在单多单场景下，`weight`、`weightScale/antiquantScale`、`bias` 按 tensor list 下标切换；
4. 保持 `x`、`perTokenScale`、`y` 按单 tensor 内部 offset 推进；
5. 保持非 MxA8W4、非单多单路径行为不变。

本期不做：

- 非 MxA8W4 数据流的单多单扩展；
- 非 resplit controller 路径改造；
- API 语义、数学定义、输出格式变更；
- 大范围 host shape 推导重构。

### 2.3 范围

- **包含**：
  - MxA8W4 TilingKey 增加 `IS_SINGLE_MULTI_SINGLE`；
  - MxA8W4 调用宏透传 `IS_SINGLE_MULTI_SINGLE`；
  - `GMMWeightQuantResplitController` 新增 `isSingleMultiSingle` 模板参数；
  - `UpdateGmAddr` 支持单多单场景下 `weight`、`weightScale/antiquantScale`、`bias` tensor list 取址；
  - host tiling 校验 `antiquantScaleOptional`、`biasOptional` 与 `weight` 等长。
- **不包含**：
  - 非 MxA8W4 数据流；
  - 非 arch35 或非 resplit 路径；
  - groupListType / splitItem / groupType 语义变更。

---

## 3. 方案简述

| 层次      | 要点                                                                                                                                                                       |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Host      | MxA8W4 单多单时识别 `singleX && !singleWeight && singleY`，并要求 `antiquantScaleOptional`、`biasOptional` 与 `weight` tensor list 等长                            |
| Tiling    | TilingKey 增加 `IS_SINGLE_MULTI_SINGLE` 编译期 bool 参数，MxA8W4 单多单场景传入 1，其他路径传入 0                                                                        |
| Kernel    | MxA8W4 调用宏将 `IS_SINGLE_MULTI_SINGLE` 传给 `GMMWeightQuantResplitController`；controller 在 `UpdateGmAddr` 中使用 `if constexpr` 区分单多单和原连续 offset 路径 |
| 测试/交付 | 覆盖 MxA8W4 单多单 path、非单多单回归、weightScale/bias list size 非法路径                                                                                                 |

---

## 4. 数学定义与语义

本需求不改变 GMM 的数学语义，仅改变 MxA8W4 单多单场景下 group 间输入地址推进方式。

对于第 `g` 个 group：

```text
Y_g = Matmul(X_g, Weight_g, WeightScale_g) + Bias_g
```

其中：

- `X_g` 来自单 tensor `x` 中按 groupList 切分出的第 `g` 段；
- `Weight_g` 来自 `weight[g]`；
- `WeightScale_g` 来自 `antiquantScaleOptional[g]`；
- `Bias_g` 来自 `biasOptional[g]`，仅当 bias 存在；
- `Y_g` 写入单 tensor `y` 中按 M 方向连续排列的位置。

---

## 5. 输入输出规格

### 5.A 张量接口型

#### 5.A.1 输入 — 张量参数

| 参数名                                   | 必选/可选 | dtype                      | format     | shape / 存储语义                                                                            | 说明                                            |
| ---------------------------------------- | --------- | -------------------------- | ---------- | ------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `x`                                    | 必选      | FLOAT8_E4M3FN              | ND         | 单 tensor，view/storage shape `[M, K]`                                                    | M 方向按 groupList 切分                         |
| `weight`                               | 必选      | FLOAT4_E2M1                | FRACTAL_NZ | 多 tensor，`weight[g]` view shape `[Ng, K]`；NZ storage shape `[K/32, Ng/16, 16, 32]` | 单多单场景要求 tensor list 长度为 `groupNum`  |
| `antiquantScaleOptional` / weightScale | 必选      | FLOAT8_E8M0                | ND         | 多 tensor，`antiquantScale[g]` shape `[Ng, K/64, 2, 2]`                                 | tensor list 长度必须等于 `weight` 长度        |
| `biasOptional`                         | 可选      | 与现有 MxA8W4 支持保持一致 | ND         | 多 tensor，`bias[g]` shape `[Ng]`                                                       | 存在时 tensor list 长度必须等于 `weight` 长度 |
| `perTokenScaleOptional`                | 必选      | FLOAT8_E8M0                | ND         | 单 tensor，shape `[M, K/64, 2]`                                                           | 仍按单 tensor 内部 offset 推进                  |
| `groupList`                            | 必选      | INT64                      | ND         | `[groupNum]`                                                                              | 表示 M 方向 group 切分                          |

#### 5.A.2 输入 — 标量 / 属性

| 参数名            | 类型  | 取值 / 约束  | 说明                       |
| ----------------- | ----- | ------------ | -------------------------- |
| `groupType`     | int64 | SPLIT_M 场景 | 单多单按 M 方向切分        |
| `splitItem`     | int64 | 2 / 3        | y 为单 tensor              |
| `groupListType` | int64 | 保持既有约束 | 本需求不改变语义           |
| `transA`        | bool  | false        | MxA8W4 单多单不扩展 x 转置 |
| `transB`        | bool  | true         | MxA8W4 当前 NZNK 路径      |

#### 5.A.3 输出

| 输出名 | dtype                  | format | shape      | 说明                             |
| ------ | ---------------------- | ------ | ---------- | -------------------------------- |
| `y`  | 既有 MxA8W4 输出 dtype | ND     | `[M, N]` | 单 tensor，按 group 输出连续写回 |

### 5.C dtype × format × 场景矩阵

| dtype                                                          | src_format                              | dst_format | 场景/备注                                        |
| -------------------------------------------------------------- | --------------------------------------- | ---------- | ------------------------------------------------ |
| x FLOAT8_E4M3FN / weight FLOAT4_E2M1 / weightScale FLOAT8_E8M0 | x ND，weight FRACTAL_NZ，weightScale ND | y ND       | MxA8W4 单多单，weight/weightScale/bias 多 tensor |

---

## 6. 约束条件

| 约束项             | 条件                                                       | 说明 / 校验失败表现                                        |
| ------------------ | ---------------------------------------------------------- | ---------------------------------------------------------- |
| 场景约束           | 仅 MxA8W4 数据流                                           | 非 MxA8W4 不启用 `IS_SINGLE_MULTI_SINGLE`                |
| tensor 组合        | `x` 单、`weight` 多、`y` 单                          | 对应 `singleX == 1 && singleWeight == 0 && singleY == 1` |
| weightScale 长度   | `antiquantScaleOptional->Size() == weight->Size()`       | 不满足时 host tiling / checker 应报 list size 错误         |
| bias 长度          | 若存在 bias，`biasOptional->Size() == weight->Size()`    | 不满足时 host tiling / checker 应报 list size 错误         |
| perTokenScale 长度 | `perTokenScaleOptional` 与 `x` 对应                    | 单多单下仍为单 tensor                                      |
| TilingKey          | 新增 `IS_SINGLE_MULTI_SINGLE`                            | 参数顺序必须与 kernel 模板参数一致                         |
| 地址更新           | 单多单时 `weight/weightScale/bias` 使用 tensor list 取址 | 非单多单时保持连续 offset 推进                             |

---

## 7. 交付标准

| 交付范围         | 当前迭代                                                        |
| ---------------- | --------------------------------------------------------------- |
| 交付通路         | aclnn / kernel arch35 MxA8W4 resplit 路径                       |
| 动静要求         | 动态 groupNum；shape 约束沿用现有 MxA8W4 通路                   |
| 精度标准         | 与原 MxA8W4 单 tensor weight 场景对齐，按既有 golden / 容差策略 |
| 性能标准         | 非单多单路径不劣化；单多单路径使用编译期分支避免额外运行时判断  |
| 客户场景索引     | 通用 MxA8W4 单多单场景                                          |
| 相关设计文档索引 | `gmm/grouped_matmul/docs/GMM_MxA8W4_SMS_SRS.md`               |
| 维测相关索引     | 待补充                                                          |

---

## 8. 具体方案描述

### 8.1 通路 / 调用链

```text
aclnn grouped_matmul
  → host checker / tiling
  → 生成 TilingKey(IS_SINGLE_MULTI_SINGLE)
  → grouped_matmul_apt.cpp MxA8W4 调用宏
  → GMMWeightQuantResplitController<..., isSingleMultiSingle>
  → Process group loop
  → UpdateGmAddr 更新下一 group 地址
```

### 8.2 Tiling

- **输入约束**：MxA8W4 单多单时 `x` 单 tensor、`weight` 多 tensor、`y` 单 tensor；`antiquantScaleOptional` 和 `biasOptional`（若存在）与 `weight` 等长。
- **tiling key**：

  - 新增模板参数：`IS_SINGLE_MULTI_SINGLE`；
  - MxA8W4 单多单时传入 1；
  - 其他合法 MxA8W4 组合传入 0；
  - 本次新增单多单 TilingKey 能力时，不额外新增独立 `ASCENDC_TPL_ARGS_SEL`，而是在原 `W_QUANT_TYPE = WQGMM_MX` 且 `WQ_B_TRANS = WQGMM_TRANS` 的模板组合中增加 `IS_SINGLE_MULTI_SINGLE` 的 0 / 1 取值。

### 8.3 Kernel

- **执行流程**：
  1. `Init()` 取第 0 个 `weight`、`weightScale/antiquantScale`、`bias` tensor 地址；
  2. `Process()` 按 group 遍历；
  3. 当前 group 计算使用当前保存的 GM 指针；
  4. 当前 group 计算完成后调用 `UpdateGmAddr(groupIdx, ...)`；
  5. 单多单场景下，`UpdateGmAddr` 切换到 `groupIdx + 1` 的 tensor list 地址。
- **关键逻辑**：
  - 单多单路径：`weightGm_`、`antiquantScaleGm_`、`biasGm_` 用 `GetTensorAddr(nextGroupIdx, tensorPtr_)` 更新；
  - 非单多单路径：保持 `weightGm_`、`antiquantScaleGm_`、`biasGm_` 连续 offset 更新。
- **Buffer 与同步**：沿用现有 basic block 和 resplit 同步策略，不新增同步。
- **异常与 early-exit**：`nextGroupIdx < groupNum` 时才更新下一 group 地址，最后一个 group 后不越界访问。

### 8.4 代码修改片段

本节直接列出本需求涉及的关键代码修改片段。片段用于说明修改点和参数传递链路，最终实现应以对应源码文件为准。

#### 8.4.1 TilingKey 增加 `IS_SINGLE_MULTI_SINGLE`

文件：`gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_tiling_key.h`

在模板参数声明中增加单多单 bool 参数，并为 0 / 1 增加语义化命名，避免裸值含义不清：

```cpp
// 修改点1：为 IS_SINGLE_MULTI_SINGLE 的 0 / 1 取值增加语义化命名。
// 0 表示非单多单场景，保持原连续 offset 地址推进逻辑；
// 1 表示 MxA8W4 单多单场景，kernel 按 tensor list 下标切换 weight / weightScale / bias 地址。
#define WQGMM_NOT_SINGLE_MULTI_SINGLE 0
#define WQGMM_SINGLE_MULTI_SINGLE 1

ASCENDC_TPL_ARGS_DECL(
    GroupedMatmul, // 算子OpType
    ASCENDC_TPL_UINT_DECL(W_TYPE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, WQGMM_ND, WQGMM_FRACTAL_NZ),
    ASCENDC_TPL_UINT_DECL(OFFSET_OR_BIAS_EXIT, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
        WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST, WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_EXIST,
        WQGMM_ANTIQUANT_OFFSET_EXIST_BIAS_NOT_EXIST, WQGMM_ANTIQUANT_OFFSET_EXIST_BIAS_EXIST,
        WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_FP32_EXIST, WQGMM_ANTIQUANT_OFFSET_EXIST_BIAS_FP32_EXIST),
    ASCENDC_TPL_UINT_DECL(C_QUANT_TYPE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, WQGMM_NONE, WQGMM_PER_TENSOR,
        WQGMM_PER_CHANNEL, WQGMM_PER_GROUP, WQGMM_MX),
    ASCENDC_TPL_UINT_DECL(W_QUANT_TYPE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, WQGMM_NONE, WQGMM_PER_TENSOR,
        WQGMM_PER_CHANNEL, WQGMM_PER_GROUP, WQGMM_MX),
    ASCENDC_TPL_UINT_DECL(WQ_B_TRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, WQGMM_NO_TRANS, WQGMM_TRANS),
    ASCENDC_TPL_UINT_DECL(WQ_A_TRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, WQGMM_NO_TRANS, WQGMM_TRANS),
    ASCENDC_TPL_UINT_DECL(TEMPLATE_CUSTOM_SC, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
        WQGMM_MTE2_INNER_SIZE_512_BUF_NUM_2, WQGMM_MTE2_INNER_SIZE_512_BUF_NUM_4, WQGMM_MTE2_INNER_SIZE_1024_BUF_NUM_2,
        WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4, WQGMM_MTE2_INNER_SIZE_512_BUF_NUM_DEFAULT,
        WQGMM_MTE2_INNER_SIZE_384_BUF_NUM_3, WQGMM_MTE2_INNER_SIZE_DYNAMIC_BUF_NUM_4),
    // 修改点2：新增 TilingKey bool 模板参数，位置在 TEMPLATE_CUSTOM_SC 之后，需与 kernel 模板参数顺序一致。
    ASCENDC_TPL_BOOL_DECL(IS_SINGLE_MULTI_SINGLE, WQGMM_SINGLE_MULTI_SINGLE, WQGMM_NOT_SINGLE_MULTI_SINGLE),
    ASCENDC_TPL_UINT_DECL(ALGORITHM_SUB_CATEGORY, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, WQGMM_VDEFAULT, WQGMM_SPLIT_K,
        WQGMM_N_FIRST_TAIL_RESPLIT, WQGMM_N_FIRST_BASIC_BLOCK),
    ASCENDC_TPL_UINT_DECL(ALGORITHM_CATEGORY, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, WQGMM_VECTOR_ANTIQUANT,
        WQGMM_MULTI_SCALE_DEQUANT, WQGMM_CUBE_ANTIQUANT)
    );
```

在模板参数组合中新增单多单合法取值，但不新增独立 `ASCENDC_TPL_ARGS_SEL`。MxA8W4 单多单与非单多单使用同一个 `W_QUANT_TYPE = WQGMM_MX`、`WQ_B_TRANS = WQGMM_TRANS`、`WQ_A_TRANS = WQGMM_NO_TRANS` 的组合，通过在该组合中增加 `IS_SINGLE_MULTI_SINGLE` 的 0 / 1 取值完成新增能力覆盖：

```cpp
// 修改点3：新增 MxA8W4 单多单合法取值，不新增独立 ASCENDC_TPL_ARGS_SEL。
// 该组合限定 W_QUANT_TYPE 为 WQGMM_MX、B 转置，覆盖 256 / dynamic MTE2 配置；
// IS_SINGLE_MULTI_SINGLE 新增单多单取值 1，同时保留非单多单取值 0。
ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
    ASCENDC_TPL_UINT_SEL(W_TYPE, ASCENDC_TPL_UI_LIST, WQGMM_FRACTAL_NZ),
    ASCENDC_TPL_UINT_SEL(OFFSET_OR_BIAS_EXIT, ASCENDC_TPL_UI_LIST, WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST),
    ASCENDC_TPL_UINT_SEL(C_QUANT_TYPE, ASCENDC_TPL_UI_LIST, WQGMM_NONE),
    ASCENDC_TPL_UINT_SEL(W_QUANT_TYPE, ASCENDC_TPL_UI_LIST, WQGMM_MX),
    ASCENDC_TPL_UINT_SEL(WQ_B_TRANS, ASCENDC_TPL_UI_LIST, WQGMM_TRANS),
    ASCENDC_TPL_UINT_SEL(WQ_A_TRANS, ASCENDC_TPL_UI_LIST, WQGMM_NO_TRANS),
    ASCENDC_TPL_UINT_SEL(TEMPLATE_CUSTOM_SC, ASCENDC_TPL_UI_LIST,
        WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4,
        WQGMM_MTE2_INNER_SIZE_DYNAMIC_BUF_NUM_4),
    ASCENDC_TPL_BOOL_SEL(IS_SINGLE_MULTI_SINGLE,
        WQGMM_SINGLE_MULTI_SINGLE,
        WQGMM_NOT_SINGLE_MULTI_SINGLE),
    ASCENDC_TPL_UINT_SEL(ALGORITHM_SUB_CATEGORY, ASCENDC_TPL_UI_LIST, WQGMM_N_FIRST_TAIL_RESPLIT),
    ASCENDC_TPL_UINT_SEL(ALGORITHM_CATEGORY, ASCENDC_TPL_UI_LIST, WQGMM_VECTOR_ANTIQUANT)),
```

#### 8.4.2 Host tiling 增加单多单字段和 list size 校验

文件：`gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.h`

`TilingKeyConfigure` 增加用于生成 TilingKey 的 bool 字段：

```cpp
class TilingKeyConfigure {
public:
    // ...

    // 修改点4：新增 Host 侧 TilingKey 配置字段，默认 false，避免非单多单路径行为变化。
    // 仅 MxA8W4 且 x 单 tensor、weight 多 tensor、y 单 tensor 时由 SetTilingKey 置 true。
    bool isSingleMultiSingle = false;

    // 对应10-13位 api常量化保留位
    uint16_t apiConstexpr = 0;

public:
    void PrintTilingKeyLog() const
    {
        std::stringstream ss;
        ss << "socVersionType: " << static_cast<uint32_t>(this->socVersionType)
           // ...
           << " templateCustom: " << static_cast<uint32_t>(this->templateCustom)
           << " isSingleMultiSingle: " << static_cast<uint32_t>(this->isSingleMultiSingle)
           << " apiConstexpr: " << static_cast<uint32_t>(this->apiConstexpr);
        OP_LOGI("GMMWeightQuantBatchMatmul", "tilingKeyConfigure: %s", ss.str().c_str());
        return;
    }

    uint64_t GenTilingKey() const;
};
```

文件：`gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp`

`SetShapeList` 增加 MxA8W4 单多单分支：

```cpp
// 修改点5：SetShapeList 增加 MxA8W4 单多单分支。
// 该分支只覆盖 x 单、weight 多、y 单；其他不支持组合仍返回 false。
bool GroupedWeightQuantBatchMatmulTiling:: SetShapeList(const gert::TilingContext *context)
{
    if (isSingleX_ && isSingleWeight_ && isSingleY_) {
        OP_CHECK_IF(!SetShapeListSplitMSingleXSingleWeightSingleY(context),
                    OP_LOGE(context->GetNodeName(), "Unable to get shape list"), return false);
    } else if (!isSingleX_ && !isSingleWeight_ && !isSingleY_) {
        OP_CHECK_IF(!SetShapeListMultiXMultiWeightMultiY(context),
                    OP_LOGE(context->GetNodeName(), "Unable to get MMM shape list"), return false);
    } else if (IsMxA8W4() && isSingleX_ && !isSingleWeight_ && isSingleY_) {
        OP_CHECK_IF(!SetShapeListSplitMSingleXMultiWeightSingleY(context),
                    OP_LOGE(context->GetNodeName(), "Unable to get MxA8W4 SMS shape list"), return false);
    } else {
        // unsupported scenario
        return false;
    }
    return true;
}
```

`CheckTensorListSize` 中单多单场景允许 `x` 单 tensor、`weight` 多 tensor，并要求 `antiquantScaleOptional`、`biasOptional` 与 `weight` 等长：

```cpp
bool GroupedWeightQuantBatchMatmulTiling::CheckTensorListSize(const gert::TilingContext *context) const
{
    // ...
    // 修改点6：Host 侧识别 MxA8W4 单多单场景，仅用于参数校验和 TilingKey 生成。
    // kernel 不再通过 singleX / singleWeight / singleY 做运行时分支判断。
    bool isSingleMultiSingle = IsMxA8W4() && isSingleX_ && !isSingleWeight_ && isSingleY_;
    if (groupType_ == static_cast<int64_t>(GroupType::NO_SPLIT)) {
        OP_CHECK_IF(numX_ != numWeight_,
                    OP_LOGE_FOR_INVALID_LISTSIZE(
                        OP_NAME, "x, weight",
                        std::to_string(numX_) + ", " + std::to_string(numWeight_),
                        "When groupType is -1 (no split), the sizes of x and weight must be all the same"),
                    return false);
    } else if (isSingleMultiSingle) {
        // 修改点7：单多单场景允许 weight 为 tensor list，同时要求 x 仍为单 tensor。
        OP_CHECK_IF(numX_ != 1 || numWeight_ <= 1,
                    OP_LOGE_FOR_INVALID_LISTSIZE(
                        OP_NAME, "x, weight", std::to_string(numX_) + ", " + std::to_string(numWeight_),
                        "In MxA8W4 single-multi-single mode, the size of x must be 1 and weight must be greater than 1"),
                    return false);
    } else {
        OP_CHECK_IF(numX_ != 1 || numWeight_ != 1,
                    OP_LOGE_FOR_INVALID_LISTSIZE(
                        OP_NAME, "x, weight", std::to_string(numX_) + ", " + std::to_string(numWeight_),
                        "When groupType is 0 (split M), the sizes of x and weight should all be 1"),
                    return false);
    }
    // 修改点8：单多单下 weightScale / antiquantScale 也为多 tensor，长度必须与 weight list 对齐。
    uint16_t expectedParamNum = numWeight_;
    OP_CHECK_IF(numAntiquantScale_ != expectedParamNum,
                OP_LOGE_FOR_INVALID_LISTSIZE(OP_NAME, "antiquantScaleOptional",
                std::to_string(numAntiquantScale_),
                    "The size of antiquantScaleOptional does not match the expected value"),
                return false);
    if (hasBias_) {
        // 修改点9：bias 存在时也按多 tensor 处理，长度必须与 weight list 对齐。
        OP_CHECK_IF(numBias_ != expectedParamNum,
                    OP_LOGE_FOR_INVALID_LISTSIZE(OP_NAME, "biasOptional",
                                                 std::to_string(numBias_),
                                                 "biasOptional size does not match the expected value"),
                    return false);
    }
    return true;
}
```

单多单场景逐 tensor 校验：

```cpp
// 修改点10：原 CheckEveryTensor 只按 numX_ 遍历，适用于单单单 / 多多多；
// MxA8W4 单多单下 x 为单 tensor，但 weight / weightScale / bias 为多 tensor，必须走专用校验入口。
const bool isSingleMultiSingle = IsMxA8W4() && isSingleX_ && !isSingleWeight_ && isSingleY_;
if (isSingleMultiSingle) {
    OP_CHECK_IF(!CheckEveryTensorSingleXMultiWeightSingleY(context),
                OP_LOGE(context->GetNodeName(), "CheckEveryTensorSingleXMultiWeightSingleY failed."), return false);
} else {
    OP_CHECK_IF(!CheckEveryTensor(context), OP_LOGE(context->GetNodeName(), "CheckEveryTensor failed."), return false);
}
```

```cpp
// 修改点11：单多单专用 dtype 校验不复用原 CheckTensorDtype 的 IsA16W4ND 早退逻辑，
// 避免 MxA8W4 下 weight / antiquantScale / bias 后续 tensor dtype 校验 no-op。
bool GroupedWeightQuantBatchMatmulTiling::CheckTensorDtypeSingleXMultiWeightSingleY(
    const gert::TilingContext *context, uint32_t attrIdx, size_t idx, const ge::DataType &tensorDtype,
    const std::string &tensorType) const
{
    auto tmpDesc = context->GetDynamicInputDesc(attrIdx, idx);
    auto tmpDType = tmpDesc->GetDataType();
    OP_CHECK_IF(tmpDType != tensorDtype,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    OP_NAME, tensorType, ge::TypeUtils::DataTypeToSerialString(tmpDType),
                    std::string("The dtype of each tensor in ") + tensorType + " tensor list must be consistent. " +
                        tensorType + "[" + std::to_string(idx) +
                        "]'s dtype is different from the first tensor's dtype."),
                return false);
    return true;
}
```

```cpp
// 修改点12：单多单 shape 校验按同下标 weight[i] 对齐 N 轴；
// antiquantScale 额外校验 K/group 维度，K 从 x[0] 局部获取，不依赖 SetShapeList 之后才赋值的 kSize_。
bool GroupedWeightQuantBatchMatmulTiling::CheckTensorShapeSingleXMultiWeightSingleY(
    const gert::TilingContext *context, uint32_t attrIdx, size_t idx, const std::string &tensorType) const
{
    auto tensorShapePtr = context->GetDynamicInputShape(attrIdx, idx);
    auto wShapePtr = context->GetDynamicInputShape(WEIGHT_IDX, idx);
    auto tensorShape = tensorShapePtr->GetStorageShape();
    auto wShape = wShapePtr->GetStorageShape();
    uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());
    uint32_t tensorDimNum = static_cast<uint32_t>(tensorShape.GetDimNum());

    int64_t weightNDimValue = wShape.GetDim(wDimNum - 3) * wShape.GetDim(wDimNum - 2); // MXA8W4 transposed NZ
    int64_t tensorNDimValue = attrIdx == ANTIQUANT_SCALE_IDX ? tensorShape.GetDim(tensorDimNum - 3) :
                                                              tensorShape.GetDim(tensorDimNum - 1);
    OP_CHECK_IF(weightNDimValue != tensorNDimValue,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(OP_NAME,
            tensorType + ", weight",
            Ops::Base::ToString(tensorShape) + ", " + Ops::Base::ToString(wShape),
            "NDim of " + tensorType + " must be equal to NDim of weight"),
        return false);
    if (attrIdx == ANTIQUANT_SCALE_IDX) {
        auto xShapePtr = context->GetDynamicInputShape(X_IDX, 0);
        auto xShape = xShapePtr->GetStorageShape();
        uint32_t xDimNum = static_cast<uint32_t>(xShape.GetDimNum());
        uint64_t kSize = transA_ ? xShape.GetDim(0) : xShape.GetDim(xDimNum - 1);
        int64_t scaleGroupNum = tensorShape.GetDim(tensorDimNum - PENULTIMATE_DIM) * MX_GROUP_FACTOR;
        OP_CHECK_IF(scaleGroupNum <= 0 || kSize % static_cast<uint64_t>(scaleGroupNum) != 0,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, tensorType,
                Ops::Base::ToString(tensorShape),
                "The groupNum[" + std::to_string(scaleGroupNum) + "] of " + tensorType +
                " must be greater than 0 and divisible by kSize[" + std::to_string(kSize) + "]"),
            return false);
        OP_CHECK_IF(kSize % MX_GROUP_SIZE != 0 || scaleGroupNum != static_cast<int64_t>(kSize / MX_GROUP_SIZE),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, tensorType,
                Ops::Base::ToString(tensorShape),
                "The groupSize of " + tensorType + " must be " + std::to_string(MX_GROUP_SIZE)),
            return false);
    }
    return true;
}
```

```cpp
// 修改点13：按输入类型拆分单多单逐 tensor 校验，避免 CheckEveryTensorSingleXMultiWeightSingleY 形成超大函数。
bool GroupedWeightQuantBatchMatmulTiling::CheckWeightSingleXMultiWeightSingleY(
    const gert::TilingContext *context) const
{
    uint32_t targetWeightDim = weightNzFlag_ ? 4 : 2; // NZ (n1, k1, k0, n0) 4维 ND (k, n) 2维
    for (size_t i = 0; i < numWeight_; ++i) {
        OP_CHECK_IF(!CheckNotNullPtr(context, WEIGHT_IDX, i),
                    OP_LOGE(context->GetNodeName(), "weight's tensor or Desc is nullptr."), return false);
        OP_CHECK_IF(!CheckTensorDimEqualTarget(context, WEIGHT_IDX, i, targetWeightDim, "weight"),
                    OP_LOGE(context->GetNodeName(), "Invalid weight dimension."), return false);
        OP_CHECK_IF(!CheckTensorDtypeSingleXMultiWeightSingleY(context, WEIGHT_IDX, i, weightDtype_, "weight"),
                    OP_LOGE(context->GetNodeName(), "Check Tensor weight Dtype failed"), return false);
        OP_CHECK_IF(!CheckWeightInnerAxisEven(context, i),
                    OP_LOGE(context->GetNodeName(), "CheckWeightInnerAxisEven failed"), return false);
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckAntiquantScaleSingleXMultiWeightSingleY(
    const gert::TilingContext *context) const
{
    for (size_t i = 0; i < numAntiquantScale_; ++i) {
        OP_CHECK_IF(!CheckNotNullPtr(context, ANTIQUANT_SCALE_IDX, i),
                    OP_LOGE(context->GetNodeName(), "antiquantScale's tensor or Desc is nullptr."), return false);
        OP_CHECK_IF(!CheckTensorDimEqualTarget(context, ANTIQUANT_SCALE_IDX, i, 4, "antiquantScale"),
                    OP_LOGE(context->GetNodeName(), "Invalid antiquantScale dimension."), return false);
        OP_CHECK_IF(!CheckTensorDtypeSingleXMultiWeightSingleY(context, ANTIQUANT_SCALE_IDX, i, antiquantScaleDtype_,
                                                               "antiquantScale"),
                    OP_LOGE(context->GetNodeName(), "Check Tensor antiquantScale Dtype failed"), return false);
        OP_CHECK_IF(!CheckTensorShapeSingleXMultiWeightSingleY(context, ANTIQUANT_SCALE_IDX, i, "antiquantScale"),
                    OP_LOGE(context->GetNodeName(), "Check antiquantScale tensor shape failed"), return false);
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckAntiquantOffsetSingleXMultiWeightSingleY(
    const gert::TilingContext *context) const
{
    if (!hasAntiquantOffset_) {
        return true;
    }
    for (size_t i = 0; i < numAntiquantOffset_; ++i) {
        OP_CHECK_IF(!CheckNotNullPtr(context, ANTIQUANT_OFFSET_IDX, i),
                    OP_LOGE(context->GetNodeName(), "antiquantOffset's tensor or Desc is nullptr."), return false);
        OP_CHECK_IF(!CheckTensorDimEqualTarget(context, ANTIQUANT_OFFSET_IDX, i, 2, "antiquantOffset"),
                    OP_LOGE(context->GetNodeName(), "Invalid antiquantOffset dimension."), return false);
        OP_CHECK_IF(!CheckTensorDtypeSingleXMultiWeightSingleY(context, ANTIQUANT_OFFSET_IDX, i, antiquantOffsetDtype_,
                                                               "antiquantOffset"),
                    OP_LOGE(context->GetNodeName(), "Check Tensor antiquantOffset Dtype failed"), return false);
        OP_CHECK_IF(!CheckTensorShapeSingleXMultiWeightSingleY(context, ANTIQUANT_OFFSET_IDX, i, "antiquantOffset"),
                    OP_LOGE(context->GetNodeName(), "Check antiquantOffset tensor shape failed"), return false);
    }
    return true;
}

bool GroupedWeightQuantBatchMatmulTiling::CheckBiasSingleXMultiWeightSingleY(
    const gert::TilingContext *context) const
{
    if (!hasBias_) {
        return true;
    }
    for (size_t i = 0; i < numBias_; ++i) {
        OP_CHECK_IF(!CheckNotNullPtr(context, BIAS_IDX, i),
                    OP_LOGE(context->GetNodeName(), "bias's tensor or Desc is nullptr."), return false);
        OP_CHECK_IF(!CheckTensorDimEqualTarget(context, BIAS_IDX, i, 1, "bias"),
                    OP_LOGE(context->GetNodeName(), "Invalid bias dimension."), return false);
        OP_CHECK_IF(!CheckTensorDtypeSingleXMultiWeightSingleY(context, BIAS_IDX, i, biasDtype_, "bias"),
                    OP_LOGE(context->GetNodeName(), "Check Tensor bias Dtype failed"), return false);
        OP_CHECK_IF(!CheckTensorShapeSingleXMultiWeightSingleY(context, BIAS_IDX, i, "bias"),
                    OP_LOGE(context->GetNodeName(), "Check bias tensor shape failed"), return false);
    }
    return true;
}
```

```cpp
// 修改点14：单多单总入口只负责校验 x[0]，并调度各输入类型的专用逐 tensor 校验函数。
bool GroupedWeightQuantBatchMatmulTiling::CheckEveryTensorSingleXMultiWeightSingleY(
    const gert::TilingContext *context) const
{
    OP_CHECK_IF(!CheckNotNullPtr(context, X_IDX, 0),
                OP_LOGE(context->GetNodeName(), "x's tensor or Desc is nullptr."), return false);
    OP_CHECK_IF(!CheckTensorDimEqualTarget(context, X_IDX, 0, 2, "x"),
                OP_LOGE(context->GetNodeName(), "Invalid x dimension."), return false);
    OP_CHECK_IF(!CheckDimValue(context, 0), OP_LOGE(context->GetNodeName(), "CheckDimValue failed"), return false);
    OP_CHECK_IF(!CheckXAndWeightFormat(context, 0),
                OP_LOGE(context->GetNodeName(), "Check X And Weight Format failed"), return false);
    OP_CHECK_IF(!CheckTensorDtypeSingleXMultiWeightSingleY(context, X_IDX, 0, xDType_, "x"),
                OP_LOGE(context->GetNodeName(), "Check Tensor x Dtype failed"), return false);
    OP_CHECK_IF(!CheckWeightSingleXMultiWeightSingleY(context),
                OP_LOGE(context->GetNodeName(), "CheckWeightSingleXMultiWeightSingleY failed"), return false);
    OP_CHECK_IF(!CheckAntiquantScaleSingleXMultiWeightSingleY(context),
                OP_LOGE(context->GetNodeName(), "CheckAntiquantScaleSingleXMultiWeightSingleY failed"), return false);
    OP_CHECK_IF(!CheckAntiquantOffsetSingleXMultiWeightSingleY(context),
                OP_LOGE(context->GetNodeName(), "CheckAntiquantOffsetSingleXMultiWeightSingleY failed"), return false);
    OP_CHECK_IF(!CheckBiasSingleXMultiWeightSingleY(context),
                OP_LOGE(context->GetNodeName(), "CheckBiasSingleXMultiWeightSingleY failed"), return false);
    return true;
}
```

单多单场景 shape list 计算：

```cpp
// 修改点15：新增单多单 shape list 计算。
// M / K 从单 tensor x 获取；N 从 y 或 weight 获取；groupNum 按 weight tensor list 个数累计。
bool GroupedWeightQuantBatchMatmulTiling::SetShapeListSplitMSingleXMultiWeightSingleY(
    const gert::TilingContext *context)
{
    auto xTensor = context->GetDynamicInputTensor(X_IDX, 0);  // 0: get first tensor
    OP_CHECK_IF(xTensor == nullptr, OP_LOGE(context->GetNodeName(), "xTensor is nullptr."), return false);
    gert::Shape xShape = xTensor->GetStorageShape();
    uint32_t xDimNum = static_cast<uint32_t>(xShape.GetDimNum());
    mSize_ = transA_ ? xShape.GetDim(1) : xShape.GetDim(0);
    kSize_ = transA_ ? xShape.GetDim(0) : xShape.GetDim(xDimNum - 1);

    if (weightNzFlag_) {
        const gert::StorageShape *yShapePtr = context->GetOutputShape(0);
        OP_CHECK_IF(yShapePtr == nullptr, OP_LOGE(context->GetNodeName(), "yShapePtr is nullptr."), return false);
        const gert::Shape &yShape = yShapePtr->GetOriginShape();
        nSizeOri_ = yShape.GetDim(static_cast<uint32_t>(yShape.GetDimNum()) - 1);
    }

    bool isFp4PackType = weightDtype_ == ge::DT_FLOAT || weightDtype_ == ge::DT_INT32;
    for (uint16_t i = 0; i < GroupedMatmul::MAX_TENSOR_CONT; i++) {
        auto wTensor = context->GetDynamicInputTensor(WEIGHT_IDX, i);
        if (wTensor == nullptr) {
            break;
        }
        gert::Shape wShape = wTensor->GetStorageShape();
        uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());
        uint64_t nSize = transB_ ? wShape.GetDim(wDimNum - 2) : wShape.GetDim(wDimNum - 1);
        if (weightNzFlag_) {
            nSize = transB_ ? wShape.GetDim(wDimNum - 3) * wShape.GetDim(wDimNum - 2)
                            : wShape.GetDim(wDimNum - 4) * wShape.GetDim(wDimNum - 1);
        }
        if (isFp4PackType && !transB_) {
            nSize = static_cast<uint64_t>(8) * nSize;
        }
        if (!weightNzFlag_) {
            nSizeOri_ = nSize;
        }
        groupNum_ += 1U;
        kList_[i] = static_cast<int32_t>(kSize_);
        nList_[i] = static_cast<int32_t>(nSizeOri_);
        mList_[i] = -1;
        nSize_ = std::max(nSize_, nSize);
    }
    if (isFp4PackType) {
        weightDtype_ = weightDtype_ == ge::DT_FLOAT ? ge::DT_FLOAT4_E2M1 : ge::DT_INT4;
    }
    return true;
}
```

#### 8.4.3 Host 生成 TilingKey 时写入单多单参数

文件：`gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp`

MxA8W4 数据流下通过 TilingKey 传递单多单信息，不在 kernel 宏中运行时判断：

```cpp
// 修改点16：MxA8W4 场景下根据 Host 侧识别结果写入 isSingleMultiSingle。
// 单多单信息通过 TilingKey 进入 kernel 模板参数，避免 kernel 内运行时判断。
void GroupedWeightQuantBatchMatmulTiling::SetTilingKey(gert::TilingContext *context)
{
    // ...
    if (xDType_ == ge::DT_INT8 && weightDtype_ == ge::DT_INT4) {
        tilingKeyConfig_.templateCustom = static_cast<uint8_t>(Mte2Configuration::MTE2_INNER_SIZE_512_BUF_NUM_DEFAULT);
        if (groupSize_ == 192u) {
            tilingKeyConfig_.templateCustom = static_cast<uint8_t>(Mte2Configuration::MTE2_INNER_SIZE_384_BUF_NUM_3);
        }
    } else if (xDType_ == ge::DT_FLOAT8_E4M3FN && weightDtype_ == ge::DT_FLOAT4_E2M1 &&
               antiquantScaleDtype_ == ge::DT_FLOAT8_E8M0) {
        tilingKeyConfig_.templateCustom = static_cast<uint8_t>(Mte2Configuration::MTE2_INNER_SIZE_DYNAMIC_BUF_NUM_4);
        tilingKeyConfig_.isSingleMultiSingle = isSingleX_ && !isSingleWeight_ && isSingleY_;
    }
    tilingKeyConfig_.apiConstexpr = 0U;
    context->SetTilingKey(tilingKeyConfig_.GenTilingKey());
}
```

`GenTilingKey` 将 `isSingleMultiSingle` 放在 `TEMPLATE_CUSTOM_SC` 之后，必须与 kernel 模板参数顺序一致：

```cpp
// 修改点17：GET_TPL_TILING_KEY 参数顺序新增 isSingleMultiSingle。
// 顺序必须与 ASCENDC_TPL_ARGS_DECL 和 grouped_matmul kernel 模板参数完全一致。
uint64_t TilingKeyConfigure::GenTilingKey() const
{
    PrintTilingKeyLog();
    constexpr uint8_t DECIMAL = 10U;
    uint64_t transInfo = this->transposeSituation;
    bool atrans_ = (transInfo == static_cast<uint64_t>(GmmTrans::ATrans)) ||
                   (transInfo == static_cast<uint64_t>(GmmTrans::ABTrans));
    bool btrans_ = (transInfo == static_cast<uint64_t>(GmmTrans::BTrans)) ||
                   (transInfo == static_cast<uint64_t>(GmmTrans::ABTrans));
    return GET_TPL_TILING_KEY(
        static_cast<uint64_t>(this->weightFormat), static_cast<uint64_t>(this->optionInputSituation),
        static_cast<uint64_t>(this->quantType), static_cast<uint64_t>(this->antiquantType),
        static_cast<uint64_t>(btrans_), static_cast<uint64_t>(atrans_), static_cast<uint64_t>(this->templateCustom),
        static_cast<uint64_t>(this->isSingleMultiSingle), static_cast<uint64_t>(this->algorithm % DECIMAL),
        static_cast<uint64_t>(this->algorithm / DECIMAL));
}
```

#### 8.4.4 Kernel 入口透传 `IS_SINGLE_MULTI_SINGLE`

文件：`gmm/grouped_matmul/op_kernel/grouped_matmul_apt.cpp`

kernel 模板参数增加 `IS_SINGLE_MULTI_SINGLE`：

```cpp
// 修改点18：kernel 模板参数列表增加 IS_SINGLE_MULTI_SINGLE。
// 位置与 TilingKey 中 bool 参数保持一致，用于编译期选择地址更新逻辑。
template <int8_t W_TYPE, int8_t OFFSET_OR_BIAS_EXIT, int8_t C_QUANT_TYPE, int8_t W_QUANT_TYPE, int8_t WQ_B_TRANS,
          int8_t WQ_A_TRANS, int8_t TEMPLATE_CUSTOM_SC, bool IS_SINGLE_MULTI_SINGLE, int8_t ALGORITHM_SUB_CATEGORY,
          int8_t ALGORITHM_CATEGORY>
__global__ __aicore__ void grouped_matmul(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset,
                                          GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR groupList,
                                          GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
```

MxA8W4 调用宏将 `IS_SINGLE_MULTI_SINGLE` 作为编译期参数传给 `GMMWeightQuantResplitController`：

```cpp
// 修改点19：仅 MxA8W4 调用宏透传 IS_SINGLE_MULTI_SINGLE。
// 其他 weight quant 路径继续传 false / 非单多单值，避免扩大影响范围。
#define INVOKE_GMM_WEIGHT_QUANT_MXA8W4_CONTROLLER_OP_IMPL(templateClass, ...)                                          \
    do {                                                                                                               \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, gmmWeightQuantParam, gmmBaseParams_, tiling);                 \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, mmTilingData, mmTilingData_, tiling);                         \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DTYPE_ANTIQUANT_SCALE, DTYPE_SCALE, DTYPE_PER_TOKEN_SCALE, DTYPE_BIAS,    \
                      DTYPE_Y, WeightQuantMatmulBasicBlock, __VA_ARGS__, IS_SINGLE_MULTI_SINGLE>                       \
            op;                                                                                                        \
        op.Init(x, weight, scale, antiquantScale, antiquantOffset, bias, groupList, perTokenScale, y, &gmmBaseParams_, \
                &mmTilingData_, tiling, &tPipe);                                                                       \
        op.Process();                                                                                                  \
    } while (0)
```

#### 8.4.5 Resplit controller 增加模板参数并按 tensor list 切换地址

文件：`gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/grouped_matmul_weight_quant_resplit_controller.h`

controller 模板参数和类型别名增加 `bool isSingleMultiSingle`：

```cpp
// 修改点20：resplit controller 增加编译期 bool 模板参数。
// 使用 if constexpr 在编译期消除非当前分支，避免在 group 循环中增加运行时判断开销。
#define GMM_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM                                               \
    template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, \
              typename perTokenScaleType, typename biasType, typename yType,                   \
              GMM_WQ_BASIC_BLOCK_TEMPLATE_CLASS BasicBlock, const WqmmConfig &wqmmConfig,      \
              const VecAntiQuantConfig &vecConfig, bool isSingleMultiSingle>

#define GMM_WQ_RESPLIT_CONTROLLER_CLASS                                                                              \
    GMMWeightQuantResplitController<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType, biasType, yType, \
                                    BasicBlock, wqmmConfig, vecConfig, isSingleMultiSingle>
```

`Init` 保存 `weight`、`antiquantScale`、`bias` tensor list 首地址，用于后续按 group 下标取址：

```cpp
// 修改点21：Init 保存 tensor list 原始 GM_ADDR。
// 单多单场景后续需要通过 GetTensorAddr(groupIdx, xxxTensorPtr_) 切换到下一组 weight / weightScale / bias。
xGm_ = GetTensorAddr<xType>(0, x);
weightTensorPtr_ = weight;
antiquantScaleTensorPtr_ = antiquantScale;
biasTensorPtr_ = bias;
weightGm_ = GetTensorAddr<wType>(0, weightTensorPtr_);
antiquantScaleGm_ = GetTensorAddr<antiQuantScaleType>(0, antiquantScaleTensorPtr_);
antiquantOffsetGm_ = GetTensorAddr<xType>(0, antiquantOffset);
biasGm_ = GetTensorAddr<biasType>(0, biasTensorPtr_);
scaleGm_ = GetTensorAddr<scaleType>(0, scale);
perTokenScaleGm_ = reinterpret_cast<__gm__ perTokenScaleType *>(perTokenScale);
yGm_ = GetTensorAddr<yType>(0, y);
```

`UpdateGmAddr` 中仅在单多单场景切换 `weight`、`weightScale/antiquantScale`、`bias` tensor list 地址；`x`、`perTokenScale`、`y` 保持单 tensor 内部 offset 推进：

```cpp
// 修改点22：UpdateGmAddr 增加单多单地址切换分支。
// 单多单：weight / antiquantScale(weightScale) / bias 使用 tensor list 下标切换；
// 非单多单：保持原连续 offset 推进逻辑不变。
GMM_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void GMM_WQ_RESPLIT_CONTROLLER_CLASS::UpdateGmAddr(uint64_t groupIdx, uint64_t mSize,
                                                                     uint64_t kSize, uint64_t nSize)
{
    // x 为单 tensor，始终按 M 方向 group 大小推进。
    xGm_ += mSize * kSize;
    if constexpr (isSingleMultiSingle) {
        uint64_t nextGroupIdx = groupIdx + 1;
        // 最后一组计算结束后不再访问 nextGroupIdx，避免 tensor list 越界。
        if (nextGroupIdx < gmmBaseTiling_->groupNum) {
            // weight 为多 tensor，下一组直接切换到 weight[nextGroupIdx]。
            weightGm_ = GetTensorAddr<wType>(nextGroupIdx, weightTensorPtr_);
            // weightScale 在代码中对应 antiquantScale，单多单下同样为多 tensor。
            antiquantScaleGm_ = GetTensorAddr<antiQuantScaleType>(nextGroupIdx, antiquantScaleTensorPtr_);
            if (gmmBaseTiling_->hasBias != 0) {
                // bias 可选；存在时为多 tensor，按 group 下标切换。
                biasGm_ = GetTensorAddr<biasType>(nextGroupIdx, biasTensorPtr_);
            }
        }
    } else {
        // 非单多单路径保持原逻辑：weight / scale / bias 均在同一个 tensor 内按 offset 推进。
        if constexpr (IsSameType<wType, int4b_t>::value || IsSameType<wType, fp4x2_e2m1_t>::value ||
                      IsSameType<wType, fp4x2_e1m2_t>::value) {
            weightGm_ += (nSize * kSize) >> 1;
        } else {
            weightGm_ += nSize * kSize;
        }

        if constexpr (wqmmConfig.antiQuantType == QuantType::PER_GROUP || wqmmConfig.antiQuantType == QuantType::MX) {
            antiquantScaleGm_ += nSize * CeilDivide(kSize, static_cast<uint64_t>(gmmBaseTiling_->groupSize));
            antiquantOffsetGm_ += nSize * CeilDivide(kSize, static_cast<uint64_t>(gmmBaseTiling_->groupSize));
        } else {
            antiquantScaleGm_ += nSize;
            antiquantOffsetGm_ += nSize;
        }

        scaleGm_ += nSize;
        biasGm_ += nSize;
    }

    // perTokenScale 为单 tensor，MxA8W4 下按 M 方向和 groupSize 对应的 K 分组数推进。
    if constexpr (IsMxA8W4<xType, wqmmConfig.antiQuantType>()) {
        perTokenScaleGm_ += mSize * CeilDivide(kSize, static_cast<uint64_t>(gmmBaseTiling_->groupSize));
    } else {
        perTokenScaleGm_ += mSize;
    }

    // y 为单 tensor，按当前 group 的 M*N 输出区域推进。
    yGm_ += mSize * nSize;
}
```

---

## 9. 软件设计与接口总览

### 9.1 模块职责

| 模块   | 说明                                                              |
| ------ | ----------------------------------------------------------------- |
| Host   | 校验 MxA8W4 单多单 tensor list 长度，设置 `isSingleMultiSingle` |
| Tiling | 生成包含 `IS_SINGLE_MULTI_SINGLE` 的 TilingKey                  |
| Kernel | 根据编译期参数选择 tensor list 取址或连续 offset 更新             |
| 测试   | 覆盖单多单、多 weightScale、多 bias、非单多单回归                 |

### 9.2 接口列表（与实现对齐）

| 接口项                  | 说明                                                                              |
| ----------------------- | --------------------------------------------------------------------------------- |
| Host API                | aclnn grouped_matmul weight quant MxA8W4 通路，API 名称沿用现有接口               |
| Tiling 数据结构         | `TilingKeyConfigure::isSingleMultiSingle`，`GMMWeightQuantParam` 沿用现有字段 |
| Kernel entry / 模板参数 | `grouped_matmul_apt.cpp` 中新增 `IS_SINGLE_MULTI_SINGLE` 模板参数             |
| Controller 模板参数     | `GMMWeightQuantResplitController<..., bool isSingleMultiSingle>`                |
| 对外泄漏 API            | 无新增对外 API                                                                    |

---

## 11. 外部依赖

| 依赖项                       | 内容                                                                        | 影响级别 |
| ---------------------------- | --------------------------------------------------------------------------- | -------- |
| CANN / Ascend C 模板参数机制 | `ASCENDC_TPL_BOOL_DECL`、`ASCENDC_TPL_BOOL_SEL`、`GET_TPL_TILING_KEY` | 高       |
| 硬件型号 / SOC               | arch35 MxA8W4 支持硬件                                                      | 高       |
| 上下游算子 / Pass            | grouped_matmul aclnn 调用链                                                 | 中       |
| 测试框架与 golden            | 现有 GMM MxA8W4 用例体系                                                    | 中       |

---

## 12. 待确认事项

1. 单多单 MxA8W4 场景是否需要补充专门 STC 用例路径。
2. bias 不存在、bias 多 tensor 存在两种场景是否均纳入测试矩阵。
3. 是否需要在对外 API 文档中同步说明 `antiquantScaleOptional` / `biasOptional` 与 `weight` 等长要求。

---

## 13. 修订记录

| 版本 | 日期       | 作者   | 说明                                 |
| ---- | ---------- | ------ | ------------------------------------ |
| v0.1 | 2026-06-11 | Claude | 按脱敏母版整理 GMM MxA8W4 单多单需求 |

---

## 附录 A：与 STC 的交叉检查清单

- [ ] MxA8W4 单多单 happy path：x 单 tensor、weight 多 tensor、weightScale 多 tensor、y 单 tensor。
- [ ] bias 存在：bias 多 tensor 且长度等于 weight 长度。
- [ ] bias 不存在：不访问 bias tensor list。
- [ ] 非法 list size：weightScale 长度与 weight 不一致。
- [ ] 非法 list size：bias 长度与 weight 不一致。
- [ ] 非单多单 MxA8W4 回归：`IS_SINGLE_MULTI_SINGLE == 0` 路径行为不变。
- [ ] TilingKey 白盒检查：MxA8W4 单多单场景生成 `IS_SINGLE_MULTI_SINGLE == 1`。
