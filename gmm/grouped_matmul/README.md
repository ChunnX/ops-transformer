# GroupedMatmul

## 目录

- [1. 产品支持情况](#1-产品支持情况)
- [2. 专业术语](#2-专业术语)
  - [2.1 场景分类](#21-场景分类)
  - [2.2 量化模式](#22-量化模式)
  - [2.3 量化粒度](#23-量化粒度)
  - [2.4 分组与配置](#24-分组与配置)
- [3. 功能说明](#3-功能说明)
  - [3.1 分组方式](#31-分组方式)
  - [3.2 计算公式](#32-计算公式)
  - [3.3 版本演进](#33-版本演进)
- [4. 参数说明](#4-参数说明)
  - [4.1 通用参数](#41-通用参数)
  - [4.2 非量化场景参数](#42-非量化场景参数)
  - [4.3 伪量化场景参数](#43-伪量化场景参数)
  - [4.4 全量化场景参数](#44-全量化场景参数)
- [5. 约束说明](#5-约束说明)
  - [5.1 公共约束](#51-公共约束)
  - [5.2 Ascend 950PR/Ascend 950DT](#52-ascend-950prascend-950dt)
  - [5.3 Atlas A3/A2 训练系列产品](#53-atlas-a3a2-训练系列产品)
- [6. 接口调用](#6-接口调用)

---

## 1. 产品支持情况

| 产品 | 是否支持 |
|:----------------------------|:-----------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Kirin X90 处理器系列产品</term> | √ |
| <term>Kirin 9030 处理器系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

---

## 2. 专业术语

> 本章集中解释下文中出现的简称和专业术语，建议在阅读后续章节前先了解。

<a id="21-场景分类"></a>

### 2.1 场景分类

GroupedMatmul 根据计算过程中对数据精度的处理方式，分为三大场景：

| 场景 | 含义 | 简单理解 |
|------|------|----------|
| **非量化** | x 和 weight 均以浮点类型（FLOAT16/BFLOAT16/FLOAT32）直接参与计算 | 常规矩阵乘，不涉及任何缩放/偏移 |
| **伪量化** | weight 以低精度整数（INT4/INT8/FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8）存储，计算时通过 `antiquantScale` / `antiquantOffset` 反量化为浮点再与 x 相乘 | "存储时省空间，计算时还原" |
| **全量化** | x 和 weight 均以整数类型（INT8/INT4/FLOAT8 等）参与计算，结果通过 `scale` / `offset` / `perTokenScale` 还原为浮点 | "全程低精度计算，最后还原" |

<a id="22-量化模式"></a>

### 2.2 量化模式

量化模式描述了 x 和 weight 分别采用什么粒度的量化策略，命名格式为 `X侧量化粒度 - Weight侧量化粒度`：

| 简称 | 全称 | x 侧量化粒度 | weight 侧量化粒度 |
|------|------|:---:|:---:|
| **T-C** | Tensor-Channel（pertensor-perchannel） | 按 Tensor（x 全体共享一套量化参数） | 按 Channel（weight 每列一套量化参数） |
| **T-T** | Tensor-Tensor（pertensor-pertensor） | 按 Tensor | 按 Tensor |
| **K-C** | perToken-perChannel（pertoken-perchannel） | 按 Token（x 每行一套量化参数） | 按 Channel |
| **K-T** | perToken-perTensor（pertoken-pertensor） | 按 Token | 按 Tensor |
| **G-B** | perGroup-perBlock（pergroup-perblock） | 按 Group（x 按 perTokenScale 的 block 分组） | 按 Block（weight 按 K/N 方向 block 分组） |
| **MX** | MX 量化 | 按 Token + block | 按 Token + block（特殊的 pergroup 量化，block size: M=1, N=1, K=32） |

<a id="23-量化粒度"></a>

### 2.3 量化粒度

| 术语 | 含义 | 举例 |
|------|------|------|
| **perchannel** | 对 weight 的每一列（N 方向）使用独立的缩放因子 | scale 的 shape 为 `(E, N)` |
| **pertensor** | 对一个 Tensor 整体使用统一的缩放因子 | scale 的 shape 为 `(E, 1)` |
| **pertoken** | 对 x 的每一行（M 方向）使用独立的缩放因子 | perTokenScale 的 shape 为 `(M,)` |
| **pergroup** | 对 weight 按 K 方向等分为若干组（Group），每组使用独立的缩放因子 | scale 的 shape 为 `(E, G, N)` |
| **perblock** | 对 weight 按 K/N 方向固定大小（如 128）分块，每块独立缩放 | 参见 [G-B 量化](#gb量化) |

<a id="24-分组与配置"></a>

### 2.4 分组与配置

| 参数/术语 | 含义 |
|-----------|------|
| **groupType** | 指定分组轴：`-1` = 不分组，`0` = M 轴分组，`2` = K 轴分组 |
| **groupListOptional** | 描述分组轴上各组大小的数组（具体格式取决于 groupListType） |
| **groupListType** | groupList 的数据格式：`0` = 累积和，`1` = 各组大小，`2` = (组索引, 组大小) 对 |
| **splitItem** | 输出是否为单 Tensor：`0/1` = 多 Tensor 输出，`2/3` = 单 Tensor 输出 |
| **actType** | 激活函数类型（0~5）：NONE / RELU / GELU_TANH / GELU_ERR_FUNC / FAST_GELU / SILU |

---

## 3. 功能说明

### 3.1 分组方式

算子功能：实现分组矩阵乘计算，公式为 $y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$，其中 $g$ 为分组数。

- **M 轴分组（groupType=0）**：$k_i$、$n_i$ 各组相同，$m_i$ 可以不相同。
- **K 轴分组（groupType=2）**：$m_i$、$n_i$ 各组相同，$k_i$ 可以不相同。
- **不分组（groupType=-1）**：各组 shape 完全独立。

<a id="32-计算公式"></a>

### 3.2 计算公式

#### 3.2.1 非量化场景

$$
y_i = x_i \times weight_i + bias_i
$$

#### 3.2.2 伪量化场景

> weight 以低精度存储，计算时反量化为浮点。

- **x 为 FLOAT16/BFLOAT16，weight 为 INT4/INT8**（仅支持 x、weight、y 均为单 Tensor）：

$$
y_i = x_i \times (weight_i + antiquant\\_offset_i) \times antiquant\\_scale_i + bias_i
$$

- **x 为 INT8，weight 为 INT4**（仅支持 x、weight、y 均为单 Tensor）：

  其中 $bias$ 为必选参数，是离线计算的辅助结果，$bias_i = 8 \times weight_i \times scale_i$，并沿 K 轴规约。

$$
y_i = ((x_i - 8) \times weight_i \times scale_i + bias_i) \times per\\_token\\_scale_i
$$

- **x 为 FLOAT16/BFLOAT16，weight 为 FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8**（仅支持 A5 设备）：

$$
y_i = x_i \times (weight_i + antiquant\\_offset_i) \times antiquant\\_scale_i + bias_i
$$

#### 3.2.3 全量化场景

##### 静态量化（T-C / T-T，无 perTokenScaleOptional）

$$
y_i = (x_i \times weight_i) \times scale_i + offset_i
$$

- x 为 INT8，bias 为 INT32 时：

$$
y_i = (x_i \times weight_i + bias_i) \times scale_i + offset_i
$$

- x 为 INT8，bias 为 FLOAT16/BFLOAT16/FLOAT32，无 offset 时：

$$
y_i = (x_i \times weight_i) \times scale_i + bias_i
$$

##### 动态量化（T-T / T-C / K-T / K-C）

$$
y_i = (x_i \times weight_i) \times scale_i \times per\\_token\\_scale_i
$$

- x 为 INT8，bias 为 INT32 时：

$$
y_i = (x_i \times weight_i + bias_i) \times scale_i \times per\\_token\\_scale_i
$$

- x 为 INT8，bias 为 FLOAT16/BFLOAT16/FLOAT32 时：

$$
y_i = (x_i \times weight_i) \times scale_i \times per\\_token\\_scale_i + bias_i
$$

##### MX 量化

<a id="mx量化"></a>

MX 量化是特殊的 pergroup 量化，block size 固定为 gsM=1, gsN=1, gsK=32：

$$
y_i[m,n] = \sum_{j=0}^{kLoops-1} \bigg( \Big( \sum_{k=0}^{gsK-1} xSlice_i \times weightSlice_i \Big) \times \big( per\\_token\\_scale_i[m/gsM, j] \times scale_i[j, n/gsN] \big) \bigg) + bias_i[n]
$$

<a id="gb量化"></a>

##### G-B 量化

G-B 是 pergroup-perblock 量化，block size 固定为 gsM=1, gsN=gsK=128：

$$
y_i[m,n] = \sum_{j=0}^{kLoops-1} \bigg( \Big( \sum_{k=0}^{gsK-1} xSlice_i \times weightSlice_i \Big) \times \big( per\\_token\\_scale_i[m/gsM, j] \times scale_i[j, n/gsN] \big) \bigg) + bias_i[n]
$$

> 其中 $xSlice_i$ 为 $x_i$ 第 $m$ 行长度为 gsK 的向量，$weightSlice_i$ 为 $weight_i$ 第 $n$ 列长度为 gsK 的向量，$kLoops = \lceil K_i / gsK \rceil$，支持末尾切片长度不足 gsK。

### 3.3 版本演进

| 版本变化 | Ascend 950PR/Ascend 950DT | Atlas A3/A2 训练系列产品 |
|---------|----------------|----------|
| V4 → V5 | — | 增加可选参数 `tuningConfigOptional`，数组中第一个值表示各专家处理的 token 数预期值，tiling 时按该值进行最优切分 |
| V1 → V4 | 支持分组轴（groupType）；非量化支持 x/weight 转置；支持静态量化（T-C/T-T）BFLOAT16/FLOAT16/FLOAT32 输出 + bias；支持动态量化（K-C/K-T/T-T/T-C/MX/G-B）BFLOAT16/FLOAT16/FLOAT32 输出 + bias；支持伪量化 weight 为 INT4/FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8（仅 perchannel 模式） | 支持分组轴（groupType）；非量化支持 x/weight 转置；支持 x/weight/y 均为单 Tensor 非量化 FLOAT32 输入；支持伪量化 weight=INT4（perchannel/pergroup 模式） |

---

## 4. 参数说明

### 4.1 通用参数

以下参数在所有场景中有效：

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|:---:|------|----------|:---:|
| x | 输入 | 公式中的输入 $x$ | FLOAT<sup>1</sup>、FLOAT16、INT16<sup>1</sup>、INT8、INT4<sup>1</sup>、BFLOAT16、FLOAT8_E5M2<sup>2</sup>、FLOAT8_E4M3FN<sup>2</sup>、HIFLOAT8<sup>2</sup> | ND |
| weight | 输入 | 公式中的 $weight$ | FLOAT<sup>1</sup>、FLOAT16、INT16<sup>1</sup>、INT8、INT4、BFLOAT16、FLOAT8_E5M2<sup>2</sup>、FLOAT8_E4M3FN<sup>2</sup>、HIFLOAT8<sup>2</sup> | ND/NZ |
| out | 输出 | 公式中的输出 $y$ | FLOAT、FLOAT16、INT32<sup>1</sup>、INT8<sup>1</sup>、BFLOAT16 | — |
| splitItem | 属性 | 输出是否做 Tensor 切分：`0/1`=多 Tensor，`2/3`=单 Tensor | INT64 | — |
| groupType | 属性 | 分组轴：`-1`=不分组，`0`=M 轴分组，`2`=K 轴分组 | INT64 | — |
| groupListType | 属性 | groupList 数据格式：`0`=累积和，`1`=各组大小，`2`=(组索引, 组大小) 对 | INT64 | — |
| actType | 属性 | 激活函数类型：0=NONE / 1=RELU / 2=GELU_TANH / 3=GELU_ERR_FUNC(不支持) / 4=FAST_GELU / 5=SILU | INT64 | — |
| groupListOptional | 可选输入 | 描述分组轴方向各组大小分布的数组 | INT64 | ND |
| tuningConfigOptional | 可选输入 | [0]=预期 token 数(优化 tiling)，[1]=是否使能 weight 亲核格式(A8W4)，[2]=允许额外内存上限 | INT64 | — |

> - <sup>1</sup>：仅 Ascend 950PR/950DT 不支持的数据类型
> - <sup>2</sup>：仅 Atlas A3/A2 不支持的数据类型
> - Kirin X90/Kirin 9030 不支持 BFLOAT16、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT8_E8M0、HIFLOAT8

### 4.2 非量化场景参数

非量化场景下，以下输入参数必须置空：`scaleOptional`、`offsetOptional`、`antiquantScaleOptional`、`antiquantOffsetOptional`、`perTokenScaleOptional`。

| 参数名 | 非量化特有说明 | 数据类型 |
|--------|------|----------|
| biasOptional | 可选，偏置项 | FLOAT、FLOAT16、INT32、BFLOAT16<sup>2</sup> |

### 4.3 伪量化场景参数

伪量化场景下，以下参数必须置空：`scaleOptional`、`offsetOptional`、`perTokenScaleOptional`。

| 参数名 | 伪量化特有说明 | 数据类型 |
|--------|------|----------|
| antiquantScaleOptional | 伪量化缩放因子 | FLOAT16、BFLOAT16 |
| antiquantOffsetOptional | 伪量化偏移量 | FLOAT16、BFLOAT16 |
| biasOptional | 可选偏置项 | FLOAT、FLOAT16、BFLOAT16<sup>2</sup> |

> **注意**：x 为 INT8、weight 为 INT4 的 A8W4 伪量化场景中，`scaleOptional` 和 `offsetOptional` 也可能生效（详见 [约束说明](#53-atlas-a3a2-训练系列产品)）。

### 4.4 全量化场景参数

全量化场景下，以下参数必须置空：`antiquantScaleOptional`、`antiquantOffsetOptional`。

| 参数名 | 全量化特有说明 | 数据类型 |
|--------|------|----------|
| scaleOptional | 量化缩放因子 | FLOAT、UINT64、BFLOAT16、FLOAT8_E8M0<sup>2</sup>、INT64<sup>2</sup> |
| offsetOptional | 量化偏移量（仅静态量化使用） | FLOAT |
| perTokenScaleOptional | x 侧的 per-token 量化缩放因子（仅动态量化/MX/G-B 使用） | FLOAT、FLOAT8_E8M0<sup>2</sup> |
| biasOptional | 可选偏置项，支持 INT32/BFLOAT16/FLOAT16/FLOAT32 | FLOAT、FLOAT16、INT32、BFLOAT16<sup>2</sup> |
| activationInputOptional | 可选输入 | 激活函数反向输入，当前仅支持 nullptr | — | — |
| activationQuantScaleOptional | 可选输入 | 当前仅支持 nullptr | — | — |
| activationQuantOffsetOptional | 可选输入 | 当前仅支持 nullptr | — | — |
| activationFeatureOutOptional | 输出 | 激活函数输入数据，当前仅支持 nullptr | — | — |
| dynQuantScaleOutOptional | 输出 | 当前仅支持 nullptr | — | — |

---

## 5. 约束说明

### 5.1 公共约束

- **确定性计算**：算子默认提供确定性实现。
- 详细接口级约束请参见 [aclnnGroupedMatmulV5 约束说明](docs/aclnnGroupedMatmulV5.md#6-约束说明)。

### 5.2 Ascend 950PR/Ascend 950DT

<a id="52-ascend-950prascend-950dt"></a>

- **groupType**：支持 M 轴分组（0）和不分组（-1）。全量化和非量化额外支持 K 轴分组（2）。
- **groupListType**：支持 0、1、2。
- **tuningConfigOptional**：不支持。
- **actType**（0~5）：
  - 非量化/伪量化场景仅支持 0（NONE）。
  - 全量化场景下，当 x 和 weight 为 INT8、量化模式为静态 T-C 或动态 K-C、scale 为 FLOAT32 或 BFLOAT16 时，支持 0、1、2、4、5；其余仅支持 0。

#### 非量化场景数据类型

| groupType | x | weight | biasOptional | out |
|:---:|:---:|:---:|:---|:---|
| -1/0/2 | BFLOAT16 | BFLOAT16 | BFLOAT16/FLOAT32/null | BFLOAT16 |
| -1/0/2 | FLOAT16 | FLOAT16 | FLOAT16/FLOAT32/null | FLOAT16 |
| -1/0/2 | FLOAT32 | FLOAT32 | FLOAT32/null | FLOAT32 |

#### 静态量化场景数据类型

| groupType | x | weight | biasOptional | scaleOptional | out |
|:---:|:---:|:---:|:---|:---|:---|
| 0 | INT8 | INT8 | INT32/null | UINT64/INT64 | BFLOAT16/FLOAT16/INT8 |
| 0 | INT8 | INT8 | INT32/null | null/UINT64/INT64 | INT32 |
| 0 | INT8 | INT8 | INT32/BFLOAT16/FLOAT32/null | BFLOAT16/FLOAT32 | BFLOAT16 |
| 0 | INT8 | INT8 | INT32/FLOAT16/FLOAT32/null | FLOAT32 | FLOAT16 |
| 0 | HIFLOAT8 | HIFLOAT8 | null | UINT64/INT64 | BFLOAT16/FLOAT16/FLOAT32 |
| 0/2 | HIFLOAT8 | HIFLOAT8 | null | FLOAT32 | BFLOAT16/FLOAT16/FLOAT32 |
| 0 | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E5M2/FLOAT8_E4M3FN | null | UINT64/INT64 | BFLOAT16/FLOAT16/FLOAT32 |
| 0/2 | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E5M2/FLOAT8_E4M3FN | null | FLOAT32 | BFLOAT16/FLOAT16/FLOAT32 |

- **scaleOptional shape**（$g$=分组数）：

| groupType | 子场景 | shape |
|:---:|:---|:---|
| 0/2 | perchannel | `(g, N)` |
| 0/2 | pertensor | `(g, 1)` 或 `(g,)` |

#### 动态量化（T-T/T-C/K-T/K-C）场景数据类型

| groupType | x | weight | biasOptional | scaleOptional | perTokenScaleOptional | out |
|:---:|:---:|:---:|:---|:---|:---|:---|
| 0 | INT8 | INT8 | INT32/BFLOAT16/FLOAT32/null | BFLOAT16/FLOAT32 | FLOAT32 | BFLOAT16 |
| 0 | INT8 | INT8 | INT32/FLOAT16/FLOAT32/null | FLOAT32 | FLOAT32 | FLOAT16 |
| 0/2 | HIFLOAT8 | HIFLOAT8 | null | FLOAT32 | FLOAT32 | BFLOAT16/FLOAT16/FLOAT32 |
| 0/2 | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E5M2/FLOAT8_E4M3FN | null | FLOAT32 | FLOAT32 | BFLOAT16/FLOAT16/FLOAT32 |

- **scaleOptional shape**：同上表。
- **perTokenScaleOptional shape**：

| groupType | 子场景 | shape |
|:---:|:---|:---|
| 0 | pertoken | `(M,)` |
| 0 | pertensor | `(g, 1)` 或 `(g,)` |
| 2 | pertoken | `(g, M)` |
| 2 | pertensor | `(g, 1)` 或 `(g,)` |

#### 动态量化（MX）场景

| groupType | x | weight | biasOptional | scaleOptional | perTokenScaleOptional | out |
|:---:|:---:|:---:|:---:|:---:|:---|:---|
| 0/2 | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E5M2/FLOAT8_E4M3FN | null | FLOAT8_E8M0 | FLOAT8_E8M0 | BFLOAT16/FLOAT16/FLOAT32 |
| 0 | FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT32/null | FLOAT8_E8M0 | FLOAT8_E8M0 | BFLOAT16/FLOAT16/FLOAT32 |

- **scaleOptional shape**：

| groupType | shape |
|:---:|:---|
| 0 | `(g, N, ceil(K/64), 2)`（weight 转置）/ `(g, ceil(K/64), N, 2)`（weight 不转置） |
| 2 | `(K/64 + g, N, 2)`，偏移详见 [aclnn 文档](docs/aclnnGroupedMatmulV5.md) |

- **perTokenScaleOptional shape**：

| groupType | shape |
|:---:|:---|
| 0 | `(M, ceil(K/64), 2)` |
| 2 | `(K/64 + g, M, 2)` |

- x 为 FLOAT4_E2M1 时，K 须为偶数且 K≠2；weight 非转置时 N 须为偶数。

#### 动态量化（G-B）场景

| groupType | x | weight | scaleOptional | perTokenScaleOptional | out |
|:---:|:---:|:---:|:---|:---|:---|
| 0/2 | HIFLOAT8 | HIFLOAT8 | FLOAT32 | FLOAT32 | BFLOAT16/FLOAT16/FLOAT32 |
| 0/2 | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT32 | FLOAT32 | BFLOAT16/FLOAT16/FLOAT32 |

- **scaleOptional shape**（$g$=分组数）：

| groupType | shape |
|:---:|:---|
| 0 | `(g, ceil(N/gsN), ceil(K/gsK))`（转置）/ `(g, ceil(K/gsK), ceil(N/gsN))`（非转置），gsN=gsK=128 |
| 2 | `(K/gsK + g, ceil(N/gsN))`，偏移详见 [aclnn 文档](docs/aclnnGroupedMatmulV5.md) |

- **perTokenScaleOptional shape**：

| groupType | shape |
|:---:|:---|
| 0 | `(M, ceil(K/gsK))` |
| 2 | `(K/gsK + g, M)` |

#### 伪量化场景数据类型

| groupType | x | weight | biasOptional | antiquantScaleOptional | antiquantOffsetOptional | out |
|:---:|:---:|:---:|:---|:---|:---|:---|
| -1/0 | BFLOAT16 | INT8/INT4 | BFLOAT16/FLOAT32/null | BFLOAT16 | BFLOAT16/null | BFLOAT16 |
| -1/0 | FLOAT16 | INT8/INT4 | FLOAT16/null | FLOAT16 | FLOAT16/null | FLOAT16 |
| 0 | BFLOAT16 | FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8 | BFLOAT16/FLOAT32/null | BFLOAT16 | null | BFLOAT16 |
| 0 | FLOAT16 | FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8 | FLOAT16/null | FLOAT16 | null | FLOAT16 |

- weight 为 FLOAT8 系列时，antiquantOffsetOptional 仅支持空指针或空 TensorList，weight 仅支持转置。

#### groupType 支持场景

| groupType | x Tensor 数 | weight Tensor 数 | y Tensor 数 | 场景限制 |
|:---:|:---:|:---:|:---:|:---|
| -1 | 多 | 多 | 多 | splitItem=0/1；非量化 x/out 2维，伪量化 x 2~6维；weight 2维；groupListOptional 必须传空；支持 weight 转置（统一）；x 不支持转置；仅支持非量化和伪量化；仅 ND 进 ND 出 |
| 0 | 单 | 单 | 单 | splitItem=2/3；weight 3维 `(g,N,K)` 或 `(g,K,N)`；x/out 2维 `(M,K)`/`(M,N)`；bias 2维 `(g,N)`；必须传 groupListOptional；最多 1024 组；x 不转置，weight 可转置；支持 NZ 格式（x/weight 为 INT8 时）；仅 ND 出 |
| 0 | 单 | 多 | 单 | splitItem=2/3；必须传 groupListOptional（长度 ≤1024）；weight 2维，N 轴必须相等；x 不转置，weight 可转置（统一）；仅 ND 进 ND 出；仅支持非量化 |
| 0 | 多 | 多 | 单 | splitItem=2；weight 2维，N 轴必须相等；groupListOptional 可选；x 不转置，weight 可转置（统一）；仅 ND 进 ND 出；仅支持非量化 |
| 2 | 单 | 单 | 单 | splitItem=2/3；x 必须转置；weight 不能转置；x/weight 2维 `(K,M)`/`(K,N)`；out 3维 `(g,M,N)`；必须传 groupListOptional；bias 传空；仅 ND 进 ND 出；仅支持非量化和量化 |

### 5.3 Atlas A3/A2 训练系列产品

<a id="53-atlas-a3a2-训练系列产品"></a>

- **x/weight 最后一维** < 65536。
- **x/weight 转置要求**对应的 Tensor 必须非连续。
- **weight 数据格式**为 FRACTAL_NZ 时，shape 需满足 NZ 格式要求。
- **perTokenScaleOptional**：仅支持 x/weight/out 均为单 Tensor。

#### 输入数据类型和格式总览

> 详细场景约束请参见 [aclnnGroupedMatmulV5 约束说明](docs/aclnnGroupedMatmulV5.md#6-约束说明)。

| 类型 | x | weight | bias | scale | offset | antiquantScale | antiquantOffset | perTokenScale | groupList | out |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 非量化 | FLOAT | FLOAT (ND) | FLOAT/null | — | — | — | — | — | INT64 | FLOAT |
| 非量化 | FLOAT16 | FLOAT16 (ND/NZ) | FLOAT16/null | — | — | — | — | — | INT64 | FLOAT16 |
| 非量化 | BFLOAT16 | BFLOAT16 (ND/NZ) | FLOAT/null | — | — | — | — | — | INT64 | BFLOAT16 |
| 伪量化 A16W8 | FLOAT16 | INT8 (ND) | FLOAT16/null | — | — | FLOAT16 | FLOAT16 | — | INT64 | FLOAT16 |
| 伪量化 A16W8 | BFLOAT16 | INT8 (ND) | FLOAT/null | — | — | BFLOAT16 | BFLOAT16 | — | INT64 | BFLOAT16 |
| 伪量化 A16W4 | FLOAT16 | INT4 (ND) | FLOAT16/null | — | — | FLOAT16 | FLOAT16 | — | INT64 | FLOAT16 |
| 伪量化 A16W4 | BFLOAT16 | INT4 (ND) | FLOAT/null | — | — | BFLOAT16 | BFLOAT16 | — | INT64 | BFLOAT16 |
| 伪量化 A8W4 | INT8 | INT4 (ND/NZ) | FLOAT | UINT64 | — | — | — | FLOAT | INT64 | BFLOAT16 |
| 伪量化 A8W4 | INT8 | INT4 (ND/NZ) | FLOAT | UINT64 | FLOAT/null | — | — | FLOAT | INT64 | FLOAT16 |
| 全量化 A8W8 | INT8 | INT8 (ND) | INT32/null | UINT64 | — | — | — | — | INT64 | INT8 |
| 全量化 A8W8 | INT8 | INT8 (ND/NZ) | INT32/null | BFLOAT16 | — | — | — | FLOAT/null | INT64 | BFLOAT16 |
| 全量化 A8W8 | INT8 | INT8 (NZ) | — | BFLOAT16/FLOAT | — | — | — | FLOAT/null | INT64 | BFLOAT16 |
| 全量化 A8W8 | INT8 | INT8 (ND/NZ) | INT32/null | FLOAT | — | — | — | FLOAT/null | INT64 | FLOAT16 |
| 全量化 A8W8 | INT8 | INT8 (ND/NZ) | INT32/null | — | — | — | — | — | INT64 | INT32 |
| 全量化 A4W4 | INT4 | INT4 (ND/NZ) | — | UINT64 | — | — | — | FLOAT/null | INT64 | FLOAT16/BFLOAT16 |

#### groupType 支持场景

| groupType | x Tensor 数 | weight Tensor 数 | y Tensor 数 | splitItem | groupListOptional | 转置 | 其余场景限制 |
|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|
| -1 | 多 | 多 | 多 | 0/1 | 必须传空 | x 不支持转置；weight 可转置（统一） | x 2维，weight 2维，y 维度同 x |
| 0 | 单 | 单 | 单 | 2/3 | 必须传；groupListType=0 时末值 ≤ x 第一维 | x 不转置；weight 可转置（A8W4/A4W4 不可） | weight 3维，x/y 2维 |
| 0 | 单 | 多 | 单 | 2/3 | 必须传；同左 | x 不转置；weight 可转置（统一） | x/weight/y 2维；weight N 轴须相等 |
| 0 | 多 | 多 | 单 | 2/3 | 可选 | x 不转置；weight 可转置（统一） | x/weight/y 2维；weight N 轴须相等 |
| 2 | 单 | 单 | 单 | 2/3 | 必须传；末值 ≤ x 第二维 | x 必须转置；weight 不能转置 | x/weight 2维，y 3维；bias 传空 |
| 2 | 单 | 多 | 多 | 0/1 | 必须传空 | x 必须转置；weight 不能转置 | x/weight/y 2维；bias 传空 |

---

## 6. 接口调用

| 调用方式 | 接口文档 | 调用样例 | 说明 |
|----------|----------|----------|------|
| aclnn 调用 | [aclnnGroupedMatmulV5](docs/aclnnGroupedMatmulV5.md) | 见接口文档 [示例代码索引](docs/aclnnGroupedMatmulV5.md#72-示例代码索引) | 两段式接口：先调用 `aclnnGroupedMatmulV5GetWorkspaceSize` 获取 workspace 大小和 executor，再调用 `aclnnGroupedMatmulV5` 执行计算 |
