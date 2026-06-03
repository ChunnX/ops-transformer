# GMMSQ算子原型设计说明书

## 需求背景

GroupedMatmulSwigluQuantV2（GMMSQ）融合算子支持MxA8W4伪量化数据流。新增MxA8W4数据流：激活为float8_e4m3fn，权重为float4_e2m1，通过xScale（float8_e8m0）和weightScale（float8_e8m0）进行MX反量化。权重在计算过程中从float4_e2m1反量化为float8_e4m3后，再执行矩阵乘、swiglu和量化。

> MX量化的block size为32个元素，即K方向每32个元素一个scale

计算公式：

- 1.分组矩阵乘法GMM（MxA8W4反量化过程）：

  其中 $X_i$ 为float8_e4m3fn类型激活，$W_i$ 为float4_e2m1类型权重（计算时反量化为float8_e4m3），$xScale_i$ 为float8_e8m0类型激活反量化因子，$wScale_i$ 为float8_e8m0类型权重反量化因子。

  $$
  C_i = (X_i \cdot dequant(W_i)) \odot xScale_{i\ Broadcast} \odot wScale_{i\ Broadcast}
  $$

- 2.SwiGLU激活：

  $$
  C_{i,act}, gate_{i} = split(C_i)
  $$

  $$
  S_i = Swish(C_{i,act}) \odot gate_{i}
  $$

  其中 $Swish(x)=\frac{x}{1+e^{-x}}$

- 3.共享指数MX量化输出结果：

  $$
  shared\_exp = \left\lfloor \log_2(max_j(|S_{i,j}|)) \right\rceil - emax
  $$

  $$
  QScale_i = 2^{shared\_exp}
  $$

  $$
  Q_{i,j} = quantize\_to\_element\_format(S_{i,j} / QScale_i)
  $$

  其中 $emax$ 为对应数据类型的最大正则数的指数位：

  |   DataType    | emax |
  | :-----------: | :--: |
  | FLOAT8_E4M3FN |  8   |

## 规格约束

### 输入规格 — Tensor 参数

| 参数名 | 必选/可选 | dtype | format | shape | 说明 |
| ------ | --------- | ----- | ------ | ----- | ---- |
| x | 必选 | float8_e4m3fn | ND | (m, k) | 左矩阵（激活），固定不转置 |
| xScale | 必选 | float8_e8m0 | ND | (m, ceil(k/64), 2) | MX x 反量化缩放因子 |
| groupList | 必选 | int64 | ND | (e,) | 分组列表，表示各专家的token分布 |
| weight | 必选 | float4_e2m1 | [^1] FRACTAL_NZ_C0_32 | viewShape=(e, n, k)</br>oriShape=(e, ceil(k, 32), ceil(n, 16), 16, 32) | 右矩阵（权重），固定转置 |
| weightScale | 必选 | float8_e8m0 | ND | (e, n, ceil(k/64), 2) | MX weight 反量化缩放因子 |
| weightAssistMatrix | 不使用 | - | - | - | MxA8W4场景为null |
| bias | 不使用 | - | - | - | MxA8W4场景为null |
| smoothScale | 不使用 | - | - | - | MxA8W4场景为null |

[^1]: 注意weight的format格式为FRACTAL_NZ_C0_32而不是FRACTAL_NZ_C0_64，原因是weight以float4_e2m1类型加载权重，但类型转换为float8_e4m3进行计算，所以实际C0_SIZE以float8_e4m3进行计算，得到 `32B / sizeof(float8_e4m3) = 32` ，所以format格式为FRACTAL_NZ_C0_32。

### 输入规格 — Scalar / 属性参数

| 参数名 | 类型 | 取值 | 说明 |
| ------ | ---- | ---- | ---- |
| dequantMode | int64 | {2} | 反量化计算类型，固定为2（MX量化） |
| dequantDtype | int64 | {0} | 中间计算数据类型，固定为0（DT_FLOAT） |
| quantMode | int64 | {2} | 量化计算类型，固定为2（MX量化） |
| quantDtype | int64 | {36} | 输出量化类型，固定为36（y量化为DT_FLOAT8_E4M3FN） |
| transposeWeight | bool | {true} | weight是否转置，固定为转置 |
| groupListType | int64 | {0, 1} | 0: cumsum模式, 1: count模式 [^1] |
| tuningConfig | int64数组 | - | 调优参数，可选，MxA8W4场景传入空指针 |

[^1]: cumsum模式：即前缀和形式表示的每个group的大小；count模式：即直接给出每个group的大小。

### 约束条件

| 约束项 | 条件 | 说明 |
| ------ | ---- | ---- |
| k对齐 | k % 32 == 0 | k维度需32对齐 |
| N对齐 | n % 128 == 0 | n维度需128对齐 |
| K最小值 | k >= 32 | k维度最小值为32 |
| N最小值 | n >= 128 | n维度最小值为128 |
| weight格式 | FRACTAL_NZ或FRACTAL_NZ_C0_32 | weight数据格式约束 |
| 转置约束 | transpose_x == false | 左矩阵固定不转置 |
| 转置约束 | transpose_w == true | 右矩阵固定转置 |
| weightScale转置 | 与weight保持一致 | weightScale转置属性需与weight一致 |
| 专家数上限 | e <= 1024 | 专家数量不超过1024 |

### 输出规格

| 参数名 | dtype | format | shape | 说明 |
| ------ | ----- | ------ | ----- | ---- |
| output | float8_e4m3fn | ND | (m, n/2) | 量化后的输出矩阵 |
| outputScale | float8_e8m0 | ND | (m, ceil((n/2)/64), 2) | 输出量化缩放因子 |
