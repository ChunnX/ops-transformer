# aclnnGroupedMatmulV5 A16W4 Weight ND PerGroup 算子 SRS

## 需求背景

本数据流是 GMM 算子 aclnnGroupedMatmulV5 接口的 A16W4(INT4) weight ND pergroup 伪量化场景，面向 Ascend 950PR/Ascend 950DT 平台。该场景是对 A2/A3 已有 A16W4 伪量化数据流的兼容性扩展，在 Ascend 950 上新增 pergroup 量化模式支持，同时保持与 A2/A3 场景的输入兼容。

激活（x）为 FLOAT16 或 BFLOAT16 类型，权重（weight）为 INT4 类型，通过 antiquant_scale 和可选的 antiquant_offset 进行反量化后参与矩阵乘计算。weight 数据格式为 ND（非 NZ），支持转置和非转置。

计算公式：

- 伪量化反量化 + 矩阵乘：

  $$
  y_i = x_i \times (weight_i + antiquant\_offset_i) \times antiquant\_scale_i + bias_i
  $$

  其中 $x_i$ 为 FLOAT16/BFLOAT16 类型激活，$weight_i$ 为 INT4 类型权重，$antiquant\_scale_i$ 为伪量化缩放因子，$antiquant\_offset_i$ 为伪量化偏移量（可选），$bias_i$ 为偏置（可选）。

  当 antiquant_offset 为空时（对称量化），公式退化为：

  $$
  y_i = x_i \times weight_i \times antiquant\_scale_i + bias_i
  $$

支持两种 tensor 模式：

- **多多多**（groupType=-1）：x 多 tensor、weight 多 tensor、out 多 tensor
- **单单单**（groupType=0）：x 单 tensor、weight 单 tensor、out 单 tensor

## 规格约束

### 输入规格 — Tensor 参数

| 参数名 | ParamType | 必选/可选 | dtype | format | shape | 说明 |
| ------ | --------- | --------- | ----- | ------ | ----- | ---- |
| x | DYNAMIC | 必选 | FLOAT16/BFLOAT16 | ND | 多多多(groupType=-1): 2维, $(m_i, k_i)$</br>单单单(groupType=0): 2维, $(M, K)$ | 左矩阵（激活），固定不转置 |
| weight | DYNAMIC | 必选 | INT4 | ND | 多多多(groupType=-1): 2维, $(n_i, k_i)$ 或 $(k_i, n_i)$（转置）</br>单单单(groupType=0): 3维, $(E, N, K)$ 或 $(E, K, N)$（转置） | 右矩阵（权重），支持转置和非转置 |
| biasOptional | DYNAMIC | 可选 | FLOAT16/BFLOAT16/null | ND | 多多多(groupType=-1): 1维, $(n_i)$</br>单单单(groupType=0): 2维, $(E, N)$ | 偏置。bias 类型与 x 保持一致  |
| antiquantScaleOptional | DYNAMIC | 必选 | FLOAT16/BFLOAT16 | ND | 多多多（groupType=-1），shape为$[G_i, N_i]$，其中 $G_i = k_i / gs$<br>单单单（groupType=0），shape为$[E, G, N]$，其中 $G = K / gs$ [^1] | 伪量化缩放因子（pergroup），dtype 与 x 一致 |
| antiquantOffsetOptional | DYNAMIC | 可选 | FLOAT16/BFLOAT16/null | ND | 多多多（groupType=-1），shape为$[G_i, N_i]$，其中 $G_i = k_i / gs$<br>单单单（groupType=0），shape为$[E, G, N]$，其中 $G = K / gs$ [^1] | 伪量化偏移量（pergroup），dtype 与 x 一致。对称量化时为 null |
| groupListOptional | OPTIONAL | 条件必选 | INT64 | ND | 1维, $(E,)$ | 分组列表。groupType=0 时必选，groupType=-1 时必须传空 |
| scaleOptional | DYNAMIC | 不使用 | - | - | - | 本场景为 null |
| offsetOptional | DYNAMIC | 不使用 | - | - | - | 本场景为 null |
| perTokenScaleOptional | OPTIONAL | 不使用 | - | - | - | 本场景为 null |
| activationInputOptional | - | 不使用 | - | - | - | 本场景为 null |
| activationQuantScaleOptional | - | 不使用 | - | - | - | 本场景为 null |
| activationQuantOffsetOptional | - | 不使用 | - | - | - | 本场景为 null |

[^1]: **antiquantScaleOptional / antiquantOffsetOptional shape 约束**（pergroup 场景，其中 $E$ 为 matmul 组数即分组数，$G$ 为 pergroup 数，$G_i$ 为第 $i$ 个 tensor 的 pergroup 数，$gs$ 为 group_size）：

### 输入规格 — Scalar / 属性参数

| 参数名 | 类型 | 取值 | 说明 |
| ------ | ---- | ---- | ---- |
| splitItem | int64 | 多多多: {0, 1}；单单单: {2, 3} | 输出 tensor 切分模式。0/1 代表输出为多 tensor，2/3 代表输出为单 tensor |
| groupType | int64 | {-1, 0} | 分组轴。-1: 不分组（多多多），0: m 轴分组（单单单） |
| groupListType | int64 | {0, 1} | groupList 输入方式。0: cumsum 模式，1: count 模式 [^2] |
| actType | int64 | {0} | 激活函数类型，本场景固定为 0（GMM_ACT_TYPE_NONE） |
| transpose_weight | bool | {false} | 保留属性，未启用。weight 转置由 tensor 的 shape/stride 决定 |
| transpose_x | bool | {false} | 保留属性，未启用。x 固定不转置 |
| tuningConfigOptional | int64 数组 | - | Ascend 950 不支持此参数 |

[^2]: cumsum 模式：即前缀和形式表示的每个 group 的大小；count 模式：即直接给出每个 group 的大小。

### 约束条件

| 约束项 | 条件 | 说明 |
| ------ | ---- | ---- |
| K 整除 group_size | $K \bmod gs = 0$ | K 维度必须为 group_size 的整数倍 |
| group_size 取值 | $gs \in \{32, 64, 128, 256\}$ | Ascend 950 平台特定约束，仅支持这四种 group_size |
| weight 最后一维偶数 | weight 最后一维大小为偶数 | 最后一维指 weight 不转置时的 N 轴或转置时的 K 轴 |
| pergroup 整除 | $G_i$ 整除 $k_i$ | pergroup 数 $G$ 或 $G_i$ 必须能整除对应的 $k_i$ |
| 多多多 pergroup 长度一致 | $s_i = k_i / G_i$ 全部相等 | 若 weight 为多 tensor，所有 pergroup 长度 $s_i$ 必须相等 |
| pergroup 转置偶数 | weight 转置时 $s_i$ 为偶数 | pergroup 场景下，当 weight 转置时，要求 pergroup 长度 $s_i$ 是偶数 |
| x 不转置 | x 固定不转置 | 所有 groupType 下 x 均不支持转置 |
| weight 转置统一 | 多多多时 weight tensorList 中每个 tensor 转置状态需保持统一 | 不允许部分 tensor 转置、部分不转置 |
| ND 进 ND 出 | x、weight、out 均为 ND 格式 | 仅支持 ND 进 ND 出，不支持 NZ 格式 |
| 量化模式 | pergroup | pergroup 量化模式，支持对称量化（antiquantOffset 为 null）和非对称量化（antiquantOffset 非 null） |
| groupList 约束 | groupType=0 时必传，groupType=-1 时传空 | groupType=0 时 groupListOptional 第 1 维最大支持 1024，即 E 最大 1024 |
| groupListType=0 约束 | 非负单调非递减数列 | 最后一个值不大于 x 中 tensor 的第一维 |
| groupListType=1 约束 | 非负数列 | 数值的总和不大于 x 中 tensor 的第一维 |
| 多多多 x 维度 | x 中 tensor 为 2 维 | shape 为 $(m_i, k_i)$，weight 中 tensor 需为 2 维，y 中 tensor 维度和 x 保持一致 |
| 单单单 weight 维度 | weight 中 tensor 需为 3 维 | shape 为 $(E, N, K)$ 或 $(E, K, N)$ |
| 单单单 bias 维度 | bias 中 tensor 需为 2 维 | shape 为 $(E, N)$ |
| 多多多 bias 维度 | bias 中 tensor 需为 1 维 | shape 为 $(N_i)$ |

### 输出规格

| 参数名 | ParamType | dtype | format | shape | 说明 |
| ------ | --------- | ----- | ------ | ----- | ---- |
| out | DYNAMIC | FLOAT16/BFLOAT16 | ND | 多多多(groupType=-1): 2维, $(m_i, n_i)$</br>单单单(groupType=0): 2维, $(M, N)$ | 矩阵乘计算结果，dtype 与 x 一致 |
| activationFeatureOutOptional | - | - | - | - | 本场景为 null |
| dynQuantScaleOutOptional | - | - | - | - | 本场景为 null |
