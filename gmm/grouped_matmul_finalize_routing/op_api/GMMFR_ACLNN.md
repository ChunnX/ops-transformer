# aclnnGroupedMatmulFinalizeRoutingWeightNzV2 算子aclnn通路SRS

## 1. 迭代信息

| 项目 | 内容 |
| ---- | --- |
| 迭代名称 | GMMFR MxA8W4 Aclnn通路支持 |

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

| 参数名 | 必选/可选 | dtype | format | shape | 说明 | 数据范围 |
| ------ | --------- | ----- | ------ | ----- | ---- | ---- |
| x1 | 必选 | float8_e4m3fn | ND | (m, k) | 左矩阵（激活），固定不转置 | 该类型支持范围 |
| x2 | 必选 | float4_e2m1 | NZ_C0_32[^1] | viewShape=(e, n, k)</br>oriShape=(e, ceil(k, 32), ceil(n, 16), 16, 32) | 右矩阵（权重），固定转置 | 该类型支持范围 |
| scale | 必选 | float8_e8m0 | ND | (e, n, ceil(k/64), 2) | MX weight 反量化缩放因子 | 该类型支持范围 |
| bias | 可选 | bfloat16 | ND | (e, n) | 偏置 | 该类型支持范围 |
| offsetOptional | 不使用 | - | - | - | MxA8W4场景为null | - |
| antiquantScaleOptional | 不使用 | - | - | - | MxA8W4场景为null | - |
| antiquantOffsetOptional | 不使用 | - | - | - | MxA8W4场景为null | - |
| pertokenScaleOptional | 必选 | float8_e8m0 | ND | (m, ceil(k/64), 2) | MX x 反量化缩放因子 | 该类型支持范围 |
| groupList | 必选 | int64 | ND | (e,) | 分组列表，表示各专家的token分布 | 随机分组，每个组范围[0, m]，分组总和=m |
| sharedInput | 可选 | bfloat16 | ND | (bsdp, n) | 共享专家输入，bsdp <= outputBS | 该类型支持范围 |
| logit | 必选 | float32 | ND | (m,) | MoE专家logit权重 | 该类型支持范围 |
| rowIndex | 必选 | int64 | ND | (m,) | scatter-add行索引 | $[0, outputBs)$ |

[^1]: 注意weight的format格式为C0_32而不是C0_64，原因是weight以float4_e2m1类型加载权重，但类型转换为float8_e4m3进行计算，所以实际C0_SIZE以float8_e4m3进行计算，得到 `32B / sizeof(float8_e4m3) = 32` ，所以format格式为NZ_C0_32。

### 3.2 输入规格 — Scalar / 属性参数

| 参数名 | 类型 | 取值 | 说明 | 数据范围 |
| ------ | ---- | ---- | ---- | ---- |
| dtype | int64 | {0} | 输出数据类型，固定为0（float32） | 固定取0 |
| sharedInputWeight | float32 | - | 共享专家combine系数 | 该类型支持范围 |
| sharedInputOffset | int64 | >= 0 | 共享专家输出在总输出中的偏移 | 见约束条件 |
| transposeX1 | bool | {false} | 左矩阵不转置，固定false | 固定取false |
| transposeX2 | bool | {true} | 右矩阵转置，固定true | 固定取true |
| groupListType | int64 | {0, 1} | 0: cumsum模式, 1: count模式 [^2] | `取值`要求的范围 |
| tuningConfigOptional | int64数组 | - | 调优参数，可选，本数据流不需要 | - |

[^2]: cumsum模式：即前缀和形式表示的每个group的大小；count模式：即直接给出每个group的大小。

### 3.3 输出规格

| 参数名 | dtype | format | shape | 说明 |
| ------ | ----- | ------ | ----- | ---- |
| out | float32 | ND | (outputBS, n) | 融合计算结果 |

### 3.4 约束条件

| 约束项 | 条件 | 说明 |
| ------ | ---- | ---- |
| k对齐 | k % 32 == 0 | k维度需32对齐 |
| n对齐 | n % 32 == 0 | n维度需32对齐 |
| 专家数上限 | e <= 1024 | 专家数量不超过1024 |
| sharedInput batch | bsdp <= outputBS | 共享专家batch不超过输出batch |
| offset范围 | sharedInputOffset + bsdp <= outputBS | 偏移加共享专家batch不超输出batch |

## 4. 用户用例表格配置

用户输入的excel表格包含了简单的关键参数，具体的用例表格根据用户用例表参数推倒生成，主要包括:

- M: M轴
- N：N轴
- K：K轴
- E：专家数
- groupType: groupList分组的模式
- groupList: 整形数组，表格每个专家分组大小
- hasBias: true/false，是否有bias
- hasSharedInput: true/false，是否有sharedInput
- bsdp：即sharedInput的batch轴大小
- sharedInputWeight: 共享专家部分的权重
- sharedInputOffset: sharedInput共享专家的偏移
- dtype: 表示输出类型索引，固定为0，即为float32
- outputBs：最终输出batch轴大小

其他列包括（在最前面）：
- case_type：用例分组，用于区分不同的用例类型
- case_name：默认为空，自动生成用例名，否则指定用例名
- enable：表示是否生成该用例

## 4. Aclnn用例表说明

### 4.1 输入参数顺序

aclnn框架调用时参数顺序:

- x1
- x2
- scale
- bias
- perTokenScale
- groupList
- shared_input
- logits
- row_index
- offset

### 4.2 其他输入attr：

- dtype: buildins, int64_t
- shared_input_weight: buildins, float
- shared_input_offset: buildins, int64_t
- transpose_x: buildins, bool
- transpose_w: buildins, bool
- output_bs: buildins, int32_t
- group_list_type: buildins, int32_t
- group_list: int_array, int64_t
- has_bias: buildins, int32_t
- transpose_flag: buildins, int32_t

说明：
- 注意到输入参数的tensor部分有groupList，同时attr中也有group_list，实际的groupList数据从attr的group_list中读取，而输入参数部分的groupList是随机值，会被attr的group_list改写

### 4.3 aclnn_name

aclnn算子名为`aclnnGroupedMatmulFinalizeRoutingWeightNzV2`。
