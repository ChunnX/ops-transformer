# aclnnGroupedMatmulFinalizeRoutingWeightNzV2 算子aclnn通路SRS

## 需求背景

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

## 规格约束

### 输入规格 — Tensor 参数

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

### 输入规格 — Scalar / 属性参数

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

### 约束条件

| 约束项 | 条件 | 说明 |
| ------ | ---- | ---- |
| k对齐 | k % 32 == 0 | k维度需32对齐 |
| n对齐 | n % 32 == 0 | n维度需32对齐 |
| 专家数上限 | e <= 1024 | 专家数量不超过1024 |
| sharedInput batch | bsdp <= outputBS | 共享专家batch不超过输出batch |
| offset范围 | sharedInputOffset + bsdp <= outputBS | 偏移加共享专家batch不超输出batch |

### 输出规格

| 参数名 | dtype | format | shape | 说明 |
| ------ | ----- | ------ | ----- | ---- |
| out | float32 | ND | (outputBS, n) | 融合计算结果 |

