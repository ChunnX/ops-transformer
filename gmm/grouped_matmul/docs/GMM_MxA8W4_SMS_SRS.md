# GMM MxA8W4 单多单场景支持 SRS

## 1. 文档概述

本文档描述 GMM（Grouped Matmul）在 MxA8W4 伪量化数据流下支持“单多单”场景的需求与实现约束。

“单多单”表示：

- `x` 为单 tensor：`aclTensorList` 中仅包含 1 个 tensor；
- `weight` 为多 tensor：`aclTensorList` 中包含多个 tensor，每个 group 使用一个独立 weight tensor；
- `y` 为单 tensor：`aclTensorList` 中仅包含 1 个 tensor。

本需求仅面向 MxA8W4 数据流，不扩展到其他 weight quant / anti-quant 数据流。

## 2. 背景与问题

当前 GMM MxA8W4 resplit controller 的地址更新逻辑默认按连续内存偏移推进：

- `xGm_ += mSize * kSize`；
- `weightGm_ += nSize * kSize` 或 int4/fp4 场景下按半字节压缩后推进；
- `antiquantScaleGm_`、`biasGm_`、`yGm_` 等也按连续 tensor 内部 offset 推进。

该逻辑适用于“单单单”这类 weight 为单 tensor、按 group 在同一个 tensor 中连续切分的场景。

在“单多单”场景下，`weight` 是多 tensor，每个 group 的 weight 地址应通过 tensor list 下标获取，而不是在当前 weight tensor 内按连续 offset 推进。因此需要在 kernel 模板中区分单多单场景，并调整 `UpdateGmAddr` 的 weight 相关取址方式。

## 3. 需求范围

### 3.1 包含范围

本需求包含以下修改范围：

1. 在 `GMM_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM` 中新增模板参数，用于表示当前 kernel 是否为 MxA8W4 单多单场景。
2. 在 MxA8W4 调用宏 `INVOKE_GMM_WEIGHT_QUANT_MXA8W4_CONTROLLER_OP_IMPL` 中，将 TilingKey 解析得到的模板参数传入 resplit controller。
3. 修改 `UpdateGmAddr`，使单多单场景下 weight 及其按 weight 分组的辅助 tensor 使用 tensor list 下标更新地址。
4. 通过 TilingKey 增加 `IS_SINGLE_MULTI_SINGLE` 模板参数，避免在 kernel 内使用 `gmmBaseParams_.singleX == 1 && gmmBaseParams_.singleWeight == 0 && gmmBaseParams_.singleY == 1` 进行运行时判断。

### 3.2 不包含范围

本需求不包含：

1. 非 MxA8W4 数据流的单多单支持扩展。
2. host 侧 shape 推导、输入校验规则的大范围重构。
3. 非 resplit controller 路径的地址更新逻辑修改。
4. `groupType`、`splitItem`、`groupListType` 等 API 约束的语义变更。

## 4. 术语定义

| 术语 | 说明 |
| --- | --- |
| GMM | Grouped Matmul 算子 |
| MxA8W4 | x 为 FP8 E4M3，weight 为 FP4 E2M1，antiquantScale 为 FP8 E8M0 的 MX 伪量化数据流 |
| 单多单 | x 单 tensor、weight 多 tensor、y 单 tensor |
| TilingKey | host 侧生成并传入 kernel 的模板实例选择 key |
| resplit controller | `GMMWeightQuantResplitController`，负责 GMM weight quant resplit 场景的 group 循环、地址更新和 basic block 调度 |

## 5. 功能需求

### FR-1：新增单多单模板参数

`GMM_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM` 应新增一个 `bool` 模板参数，用于表示是否为单多单场景。

建议命名：

```cpp
bool isSingleMultiSingle
```

要求：

1. 该参数为编译期常量，供 `if constexpr` 使用。
2. 原有非单多单路径应显式传入 `false`。
3. MxA8W4 单多单路径应由 TilingKey 传入 `true`。

### FR-2：TilingKey 传递单多单模板参数

TilingKey 模板参数列表应新增：

```cpp
ASCENDC_TPL_BOOL_DECL(IS_SINGLE_MULTI_SINGLE, 1, 0)
```

要求：

1. 该模板参数应加入 `GET_TPL_TILING_KEY` 的参数序列。
2. 该模板参数应加入 kernel 模板形参序列。
3. MxA8W4 调用宏 `INVOKE_GMM_WEIGHT_QUANT_MXA8W4_CONTROLLER_OP_IMPL` 应将 `IS_SINGLE_MULTI_SINGLE` 传入 `GMMWeightQuantResplitController`。
4. 不应在 `INVOKE_GMM_WEIGHT_QUANT_MXA8W4_CONTROLLER_OP_IMPL` 中通过运行时条件判断单多单场景。

### FR-3：非 MxA8W4 路径保持兼容

新增模板参数后，所有现有 resplit controller 实例化路径必须补齐模板实参。

要求：

1. 非 MxA8W4 resplit controller 调用传入 `false`。
2. VCV resplit controller 调用传入 `false`。
3. 原有 A16MxFp4、S8S4、PER_CHANNEL 等路径地址更新行为保持不变。

### FR-4：单多单场景 weight 地址更新改为 tensor list 取址

在 `UpdateGmAddr` 中，单多单场景下，`weightGm_` 不应按连续 offset 推进，而应从 tensor list 中取下一个 group 对应的 tensor 地址。

行为要求：

1. 当前 group 计算使用当前已保存的 `weightGm_`。
2. `UpdateGmAddr` 在当前 group 计算结束后调用，因此更新目标为下一 group。
3. 地址更新下标应为 `groupIdx + 1`。
4. 更新前应检查 `groupIdx + 1 < gmmBaseTiling_->groupNum`，避免最后一个 group 后越界取址。

示例逻辑：

```cpp
uint64_t nextGroupIdx = groupIdx + 1;
if (nextGroupIdx < gmmBaseTiling_->groupNum) {
    weightGm_ = GetTensorAddr<wType>(nextGroupIdx, weightTensorPtr_);
}
```

### FR-5：单多单场景辅助 tensor 地址更新

在 MxA8W4 单多单场景下，与 `weight` 按 group 对应的辅助输入也应使用 tensor list 下标更新。

至少包括：

- `antiquantScaleGm_`；
- `biasGm_`，仅在 `hasBias != 0` 时更新；
- `antiquantOffsetGm_`，仅在 `wqmmConfig.hasAntiQuantOffset` 为 true 时更新；
- `scaleGm_`，仅在 `wqmmConfig.quantType != QuantType::NONE` 时更新。

说明：MxA8W4 当前主路径通常不使用 antiquantOffset 和 scale，但保留编译期条件可避免影响其他复用路径。

### FR-6：x、y、perTokenScale 地址更新保持单 tensor 连续推进

单多单场景下：

1. `x` 为单 tensor，因此 `xGm_` 应继续按 `mSize * kSize` 推进。
2. `y` 为单 tensor，因此 `yGm_` 应继续按 `mSize * nSize` 推进。
3. MxA8W4 的 `perTokenScaleGm_` 与 `x` 的 M/K group 切分相关，应继续按当前 group 的 M 方向范围推进：

```cpp
perTokenScaleGm_ += mSize * CeilDivide(kSize, groupSize);
```

## 6. 设计约束

### 6.1 编译期分支

单多单与非单多单路径应使用 `if constexpr` 分离，避免运行时分支影响 kernel 性能。

### 6.2 TilingKey 驱动

是否为单多单场景应由 TilingKey 决定，不应在 kernel 调用宏中通过 `gmmBaseParams_` 运行时字段判断。

禁止方式：

```cpp
if (gmmBaseParams_.singleX == 1 &&
    gmmBaseParams_.singleWeight == 0 &&
    gmmBaseParams_.singleY == 1) {
    ...
}
```

推荐方式：

```cpp
templateClass<..., __VA_ARGS__, IS_SINGLE_MULTI_SINGLE> op;
```

### 6.3 最小影响面

本需求应尽量限制修改范围：

1. kernel 调用宏只重点修改 `INVOKE_GMM_WEIGHT_QUANT_MXA8W4_CONTROLLER_OP_IMPL`；
2. 其它调用宏仅为补齐新增模板参数传 `false`；
3. 不引入额外 host 侧 shape 推导函数，除非已有 tiling 流程无法生成正确 TilingKey。

## 7. 地址更新流程说明

### 7.1 初始化阶段

`Init()` 阶段先取第 0 个 tensor 地址：

```cpp
weightGm_ = GetTensorAddr<wType>(0, weightTensorPtr_);
antiquantScaleGm_ = GetTensorAddr<antiQuantScaleType>(0, antiquantScaleTensorPtr_);
biasGm_ = GetTensorAddr<biasType>(0, biasTensorPtr_);
```

因此 group 0 计算前，controller 中保存的是第 0 个 group 的地址。

### 7.2 group 循环阶段

每个 group 的执行顺序为：

1. 使用当前 `xGm_ / weightGm_ / antiquantScaleGm_ / yGm_` 等地址计算当前 group；
2. 当前 group 计算结束后调用 `UpdateGmAddr(groupIdx, ...)`；
3. `UpdateGmAddr` 将地址更新到下一 group。

因此在 `UpdateGmAddr` 中，单多单路径取 tensor list 下标时使用 `groupIdx + 1`，表示下一 group 的 tensor 地址。

| 当前 `groupIdx` | 当前计算使用的 weight tensor | `UpdateGmAddr` 后切换到 |
| --- | --- | --- |
| 0 | weight[0] | weight[1] |
| 1 | weight[1] | weight[2] |
| ... | ... | ... |
| groupNum - 1 | weight[groupNum - 1] | 不再更新，避免越界 |

## 8. 涉及文件

| 文件 | 修改点 |
| --- | --- |
| `gmm/grouped_matmul/op_kernel/grouped_matmul_apt.cpp` | MxA8W4 调用宏传入 `IS_SINGLE_MULTI_SINGLE`；kernel 模板形参新增该 bool 参数 |
| `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_tiling_key.h` | TilingKey 模板参数新增 `IS_SINGLE_MULTI_SINGLE` |
| `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/grouped_matmul_weight_quant_resplit_controller.h` | controller 模板参数新增 `isSingleMultiSingle`；`UpdateGmAddr` 增加单多单地址更新逻辑 |
| `gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp` | MxA8W4 场景设置 TilingKey 中的 `isSingleMultiSingle` |
| `gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.h` | TilingKey 配置结构新增 `isSingleMultiSingle` 字段 |

## 9. 验收标准

### AC-1：模板参数传递正确

MxA8W4 kernel 编译实例中应能从 TilingKey 获取 `IS_SINGLE_MULTI_SINGLE`，并传递给 `GMMWeightQuantResplitController`。

### AC-2：非单多单路径行为不变

非单多单路径应保持原有连续 offset 更新逻辑，包括：

- `weightGm_` 按 `nSize * kSize` 或压缩类型 `(nSize * kSize) >> 1` 推进；
- `antiquantScaleGm_`、`antiquantOffsetGm_`、`scaleGm_`、`biasGm_` 按原逻辑推进；
- `xGm_`、`yGm_`、`perTokenScaleGm_` 按原逻辑推进。

### AC-3：单多单路径 weight 地址正确

单多单路径下：

1. group 0 使用 tensor list 下标 0 的 weight；
2. group 1 使用 tensor list 下标 1 的 weight；
3. group N 使用 tensor list 下标 N 的 weight；
4. 最后一个 group 后不再访问 `groupNum` 下标。

### AC-4：无运行时单多单判断

`INVOKE_GMM_WEIGHT_QUANT_MXA8W4_CONTROLLER_OP_IMPL` 中不应存在以下运行时判断：

```cpp
gmmBaseParams_.singleX == 1 &&
gmmBaseParams_.singleWeight == 0 &&
gmmBaseParams_.singleY == 1
```

### AC-5：静态检查通过

修改文件应通过基础静态检查：

```bash
git diff --check
```

## 10. 风险与注意事项

1. TilingKey 参数顺序必须与 kernel 模板形参顺序严格一致。
2. 新增 TilingKey bool 参数后，所有合法模板组合均需补齐 `ASCENDC_TPL_BOOL_SEL`，否则可能导致 TilingKey 校验失败。
3. 单多单路径中 `weightTensorPtr_` 等应保存原始 tensor list 地址，不能用当前 `weightGm_` 反推 tensor list 地址。
4. `UpdateGmAddr` 使用 `groupIdx + 1` 是因为该函数在当前 group 计算结束后调用，更新的是下一 group 地址。
5. 若后续 host 侧放开 MxA8W4 单多单输入校验，应确保 `isSingleMultiSingle` 的设置与 API 实际场景一致。

## 11. 变更摘要

本次变更目标是以最小影响面支持 GMM MxA8W4 单多单场景：

- 通过 TilingKey 增加 `IS_SINGLE_MULTI_SINGLE` 编译期参数；
- MxA8W4 调用宏直接传递该参数；
- resplit controller 根据该模板参数在 `UpdateGmAddr` 中选择地址推进方式；
- 单多单时 weight 相关地址通过 tensor list 下标切换，x/y/perTokenScale 仍按单 tensor 内部 offset 推进。
