# grouped_matmul_swiglu_quant_v2 MxA8W4 单多单支持 SRS

## 1. 文档目的

本文档描述 `grouped_matmul_swiglu_quant_v2` 算子在 arch35 MxA8W4 数据流下支持“单多单”场景的需求、设计约束、实现方案和验收标准。

“单多单”含义如下：

- `x`：单 tensor；
- `weight`：多 tensor，每个 group 对应一个独立 weight tensor；
- `y` / `output`：单 tensor。

本次修改仅覆盖 `grouped_matmul_swiglu_quant_v2` 的 MxA8W4 weight quant arch35 路径，不扩展到其它数据流。

## 2. 背景

`grouped_matmul_swiglu_quant_v2` 的 arch35 MxA8W4 路径使用 `GMMSQWeightQuantResplitController` 做 group 遍历和多核切分。

原实现默认 weight 为单 tensor，group 之间通过连续地址偏移推进：

```cpp
weightGm_ += (nSize * kSize) >> 1;
weightScaleGm_ += nSize * CeilDivide(kAlignSize, groupSize);
```

该逻辑适用于 weight 的第 0 维包含 group 维度的单 tensor 场景，例如：

```text
weight      : [E, N, K] view / [E, K1, N1, N0, K0] storage
weightScale : [E, N, ceil(K / 64), 2]
```

在单多单场景下，`weight` 和 `weightScale` 是 tensor list：

```text
weight      : [weight_0, weight_1, ..., weight_{E-1}]
weightScale : [weightScale_0, weightScale_1, ..., weightScale_{E-1}]
```

每个 group 应通过 tensor list 下标取地址，而不是在当前 tensor 内按 offset 推进。因此需要引入编译期模板参数区分单多单路径，并调整地址更新逻辑。

## 3. 范围

### 3.1 本次包含范围

本次修改包含以下文件：

| 文件 | 修改内容 |
| --- | --- |
| `op_kernel/grouped_matmul_swiglu_quant_v2_apt.cpp` | kernel 模板参数新增 `IS_SINGLE_MULTI_SINGLE`，并传给 MxA8W4 resplit controller |
| `op_kernel/arch35/weight_quant_basic_block/grouped_matmul_swiglu_quant_v2_weight_quant_tiling_key.h` | TilingKey 新增 `IS_SINGLE_MULTI_SINGLE` 模板参数 |
| `op_kernel/arch35/weight_quant_basic_block/gmmsq_weight_quant_resplit_controller.h` | controller 模板参数新增 `isSingleMultiSingle`，`UpdateGmAddr` 支持单多单 tensor list 取址 |
| `op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_weight_quant_tiling.h` | tiling 输入参数结构新增 `isSingleMultiSingle` |
| `op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_weight_quant_tiling.cpp` | host tiling 识别 weight 多 tensor，生成对应 TilingKey，并检查单多单 weight / weightScale shape |

### 3.2 本次不包含范围

本次明确不修改：

```text
gmm/grouped_matmul_swiglu_quant_v2/op_api/grouped_matmul_swiglu_quant_v2_utils.h
```

本次不包含以下事项：

1. API checker 大范围放开或重构；
2. 非 MxA8W4 数据流支持单多单；
3. 非 arch35 路径改造；
4. 算子语义、数学公式、输出布局变更；
5. `groupListType` 语义变更。

## 4. 术语

| 术语 | 说明 |
| --- | --- |
| GMMSQ | grouped matmul swiglu quant v2 的简称 |
| MxA8W4 | `x=FLOAT8_E4M3FN`，`weight=FLOAT4_E2M1`，`xScale/weightScale=FLOAT8_E8M0` 的 MX 伪量化数据流 |
| 单多单 | x 单 tensor、weight 多 tensor、output 单 tensor |
| TilingKey | host 侧生成并传给 kernel 的模板实例选择 key |
| `IS_SINGLE_MULTI_SINGLE` | TilingKey 中新增的编译期 bool 参数，表示是否为单多单场景 |
| `isSingleMultiSingle` | controller 中对应的编译期 bool 模板参数 |

## 5. 功能需求

### FR-1：通过 TilingKey 表达单多单场景

MxA8W4 weight quant TilingKey 应新增编译期 bool 参数：

```cpp
ASCENDC_TPL_BOOL_DECL(IS_SINGLE_MULTI_SINGLE, 1, 0)
```

要求：

1. `IS_SINGLE_MULTI_SINGLE == 1` 表示 MxA8W4 单多单场景；
2. `IS_SINGLE_MULTI_SINGLE == 0` 表示原有单 tensor weight 场景；
3. 该参数应参与 `ASCENDC_TPL_SEL` 合法组合校验；
4. 该参数应由 host tiling 通过 `GET_TPL_TILING_KEY` 传入。

### FR-2：kernel 模板参数透传

`grouped_matmul_swiglu_quant_v2_apt.cpp` 中，MxA8W4 weight quant 路径的 kernel 模板应新增：

```cpp
template <int8_t QUANT_B_TRANS, int8_t QUANT_A_TRANS, bool IS_SINGLE_MULTI_SINGLE>
```

要求：

1. 仅 `GMM_SWIGLU_QUANT_WEIGHT_QUANT` 路径新增该模板参数；
2. 非 weight quant 路径保持原有模板参数列表；
3. 实例化 `GMMSQWeightQuantResplitController` 时将 `IS_SINGLE_MULTI_SINGLE` 传入。

### FR-3：controller 增加单多单模板参数

`GMMSQWeightQuantResplitController` 应新增模板参数：

```cpp
bool isSingleMultiSingle
```

要求：

1. 该参数为编译期参数；
2. 使用 `if constexpr` 分离单多单和非单多单地址更新逻辑；
3. 非单多单路径保持原行为。

### FR-4：单多单场景 weight / weightScale 使用 tensor list 下标更新

单多单场景下，`UpdateGmAddr` 不应对 `weightGm_` 和 `weightScaleGm_` 做连续 offset 推进。

当前 group 计算结束后，应切换到下一 group 对应的 tensor list 地址：

```cpp
uint64_t nextGroupIdx = groupIdx + 1;
if (nextGroupIdx < tiling_->groupNum) {
    weightGm_ = GetTensorAddr<wType>(nextGroupIdx, weightTensorPtr_);
    weightScaleGm_ = GetTensorAddr<weightScaleType>(nextGroupIdx, weightScaleTensorPtr_);
}
```

说明：

- `Init()` 阶段已初始化为第 0 个 tensor；
- `UpdateGmAddr` 在当前 group 计算结束后调用；
- 因此使用 `groupIdx + 1` 表示下一 group 的 tensor 地址；
- 最后一个 group 后不再更新，避免越界。

### FR-5：单多单场景 x / xScale / y / yScale 仍按单 tensor 连续推进

单多单只改变 weight 和 weightScale 的取址方式。

以下地址仍按单 tensor 内部 offset 推进：

```cpp
xGm_ += mSize * kSize;
xScaleGm_ += mSize * CeilDivide(kAlignSize, groupSize);
yGm_ += mSize * (nSize / 2);
yScaleGm_ += mSize * CeilDivide(CeilAlign(nSize / 2, 64), groupSize);
```

### FR-6：host tiling 识别单多单并生成 TilingKey

host tiling 应识别 weight 是否为多 tensor：

```cpp
params.isSingleMultiSingle = context.GetDynamicInputShape(WEIGHT_INDEX, 1) != nullptr;
```

要求：

1. 当 `WEIGHT_INDEX` 第 1 个动态输入存在时，认为是 weight 多 tensor；
2. `GetTilingKey()` 应将 `params.isSingleMultiSingle` 作为第三个模板实参传入；
3. 非单多单场景传入 0，保持原有 TilingKey 行为。

### FR-7：host tiling 适配单多单 shape

单多单场景下，host tiling 需按多 tensor weight 的 shape 解析 `N/K/groupNum`。

#### 7.1 weight view shape

单多单场景下单个 weight tensor 的 view shape 为：

```text
[N, K]
```

因此：

```cpp
params.nSize = wShape.GetDim(0);
params.kSize = xShape.GetDim(1);
```

#### 7.2 weight storage shape

单多单场景下单个 weight tensor 的 NZ storage shape 期望为：

```text
[ceil(K / 32), ceil(N / 16), 16, 32]
```

代码中对应检查：

```cpp
const std::vector<int64_t> expectedWeightShape = {
    CeilDiv(kSize, C0_SIZE),
    CeilDiv(nSize, CUBE_BLOCK),
    CUBE_BLOCK,
    C0_SIZE
};
```

其中：

- `C0_SIZE = 32`；
- `CUBE_BLOCK = 16`。

#### 7.3 weightScale shape

单多单场景下单个 weightScale tensor 的 shape 期望为：

```text
[N, ceil(K / 64), 2]
```

代码中对应检查：

```cpp
const std::vector<int64_t> expectedWeightScaleShape = {
    nSize,
    scaleKSize,
    MX_INNER_DIM
};
```

#### 7.4 groupNum

单多单场景下：

```cpp
groupNum = weight tensor list length
```

host tiling 通过遍历 `WEIGHT_INDEX` 的动态输入个数得到 `groupNum`。

## 6. 非功能需求

### NFR-1：性能影响

单多单分支使用 `if constexpr`，不应在 kernel 运行时引入额外分支开销。

### NFR-2：兼容性

非单多单路径必须保持原有行为：

```cpp
weightGm_ += (nSize * kSize) >> 1;
weightScaleGm_ += nSize * CeilDivide(kAlignSize, groupSize);
```

### NFR-3：修改边界

不得修改 `op_api/grouped_matmul_swiglu_quant_v2_utils.h`。

## 7. 设计说明

### 7.1 TilingKey 参数顺序

MxA8W4 weight quant 路径原 TilingKey 参数为：

```text
QUANT_B_TRANS, QUANT_A_TRANS
```

修改后为：

```text
QUANT_B_TRANS, QUANT_A_TRANS, IS_SINGLE_MULTI_SINGLE
```

host 侧对应：

```cpp
return GET_TPL_TILING_KEY(
    static_cast<int64_t>(inputParams_.wTrans),
    0L,
    static_cast<uint64_t>(inputParams_.isSingleMultiSingle));
```

kernel 侧对应：

```cpp
template <int8_t QUANT_B_TRANS, int8_t QUANT_A_TRANS, bool IS_SINGLE_MULTI_SINGLE>
```

### 7.2 地址更新时序

`Init()` 阶段：

```cpp
weightGm_ = GetTensorAddr<wType>(0, weightTensorPtr_);
weightScaleGm_ = GetTensorAddr<weightScaleType>(0, weightScaleTensorPtr_);
```

`Process()` 中每轮 group：

1. 当前 group 使用当前保存的 `weightGm_ / weightScaleGm_` 计算；
2. 当前 group 计算结束后调用 `UpdateGmAddr(groupIdx, ...)`；
3. 单多单场景中，`UpdateGmAddr` 切换到 `groupIdx + 1` 的 tensor 地址。

| 当前 groupIdx | 当前使用 tensor | UpdateGmAddr 后 |
| --- | --- | --- |
| 0 | weight[0], weightScale[0] | weight[1], weightScale[1] |
| 1 | weight[1], weightScale[1] | weight[2], weightScale[2] |
| ... | ... | ... |
| groupNum - 1 | weight[groupNum - 1], weightScale[groupNum - 1] | 不更新 |

## 8. 验收标准

### AC-1：TilingKey 可区分单多单

`grouped_matmul_swiglu_quant_v2_weight_quant_tiling_key.h` 中应存在 `IS_SINGLE_MULTI_SINGLE` 的声明和合法组合选择。

### AC-2：kernel 模板可接收单多单参数

`grouped_matmul_swiglu_quant_v2_apt.cpp` 中 MxA8W4 weight quant 路径应将 `IS_SINGLE_MULTI_SINGLE` 传给 `GMMSQWeightQuantResplitController`。

### AC-3：单多单地址更新正确

单多单场景下：

1. group 0 使用 weight[0] / weightScale[0]；
2. group 1 使用 weight[1] / weightScale[1]；
3. 最后一个 group 后不访问越界 tensor。

### AC-4：非单多单行为保持不变

`isSingleMultiSingle == false` 时，weight / weightScale 仍按原连续 offset 推进。

### AC-5：API utils 文件无改动

以下文件不应出现在本次 diff 中：

```text
gmm/grouped_matmul_swiglu_quant_v2/op_api/grouped_matmul_swiglu_quant_v2_utils.h
```

### AC-6：静态检查通过

至少通过：

```bash
git diff --check
```

## 9. 风险与注意事项

1. TilingKey 参数顺序必须与 kernel 模板参数顺序一致；
2. 单多单场景的 `groupNum` 依赖 weight tensor list 长度；
3. `weightScale` tensor list 长度需与 weight tensor list 长度一致，否则 kernel 按相同 groupIdx 取址时存在越界风险；
4. `UpdateGmAddr` 中 `groupIdx + 1` 不是从 1 开始处理，而是在当前 group 结束后预取下一 group 地址；
5. 如果后续 API checker 对 MxA8W4 单多单做更严格校验，应与本文档中的 shape 约束保持一致。

## 10. 变更摘要

本次变更通过 TilingKey 新增 `IS_SINGLE_MULTI_SINGLE`，使 `grouped_matmul_swiglu_quant_v2` 的 arch35 MxA8W4 weight quant 路径能够在编译期区分单多单场景。

核心行为变化为：

- 单多单：`weight` / `weightScale` 通过 tensor list 下标切换；
- 非单多单：保持原有连续地址 offset 推进；
- `x` / `xScale` / `y` / `yScale` 始终按单 tensor 内部 offset 推进。
