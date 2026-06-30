# GroupedMatmul

<style>
abbr {
  border-bottom: 1px dotted #666;
  cursor: help;
}
</style>

## 1. GroupedMatmul 简介

GroupedMatmul 是分组矩阵乘算子，支持按 M 轴或 K 轴对输入张量进行分组，分别对各组执行矩阵乘法。涵盖非量化、伪量化和全量化三种计算场景，广泛应用于 MoE 等大模型推理加速。

## 目录

- [1. GroupedMatmul 简介](#1-groupedmatmul-简介)
- [2. 产品支持情况](#2-产品支持情况)
- [3. 专业术语](#3-专业术语)
- [4. 功能说明](#4-功能说明)
  - [4.1 分组方式](#41-分组方式)
  - [4.2 计算公式](#42-计算公式)
  - [4.3 版本演进](#43-版本演进)
- [5. 参数说明](#5-参数说明)
- [6. 约束说明](#6-约束说明)
  - [6.1 公共约束](#61-公共约束)
  - [6.2 Atlas A3/A2 训练系列产品](#62-atlas-a3a2-训练系列产品)
  - [6.3 Ascend 950PR/Ascend 950DT](#63-ascend-950prascend-950dt)
- [7. 接口调用](#7-接口调用)

---

## 2. 产品支持情况

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

## 3. 专业术语

| 术语 | 简要说明 |
|------|----------|
| 非量化 | x 和 weight 均为浮点类型直接计算 |
| 伪量化 | weight 低精度存储，通过 antiquantScale/Offset 反量化后计算 |
| 全量化 | x 和 weight 均为低精度计算，通过 scale 还原 |
| perchannel | 每列独立缩放因子 |
| pertensor | 一个 Tensor 统一缩放因子 |
| pertoken | 每行独立缩放因子 |
| pergroup | 按 K 方向等分多组 |
| perblock | 按 K/N 方向固定大小分块 |
| T-C | pertensor-perchannel：输入侧统一缩放，权重侧每列独立缩放 |
| T-T | pertensor-pertensor：输入和权重均为统一缩放 |
| K-C | pertoken-perchannel：输入侧每行独立缩放，权重侧每列独立缩放 |
| K-T | pertoken-pertensor：输入侧每行独立缩放，权重侧统一缩放 |
| G-B | pergroup-perblock：输入侧按 K 分组缩放，权重侧按 N×K 分块缩放 |
| MX | 特殊 pergroup 量化（block 粒度 M=1,N=1,K=32） |
| ND | 常规连续排布 |
| NZ | FRACTAL_NZ 格式（亲核排布） |

## 4. 功能说明

### 4.1 分组方式

算子功能：实现分组矩阵乘计算，公式为 $y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$，其中 $g$ 为分组数。

- **M 轴分组（<abbr title="指定分组轴：-1=不分组，0=M轴，2=K轴">groupType</abbr>=0）**：$k_i$、$n_i$ 各组相同，$m_i$ 可以不相同。
- **K 轴分组（<abbr title="指定分组轴：-1=不分组，0=M轴，2=K轴">groupType</abbr>=2）**：$m_i$、$n_i$ 各组相同，$k_i$ 可以不相同。
- **不分组（<abbr title="指定分组轴：-1=不分组，0=M轴，2=K轴">groupType</abbr>=-1）**：各组 shape 完全独立。

<a id="42-计算公式"></a>

### 4.2 计算公式

#### 4.2.1 <abbr title="x 和 weight 均以浮点类型直接参与计算，不涉及任何缩放/偏移">非量化</abbr>场景

| x dtype | weight dtype | bias | 计算公式 | 说明 |
|:---:|:---:|:---:|:---|------|
| FP16, BF16, FLOAT32 | FP16, BF16, FLOAT32 | 可选 | $y_i = x_i \times weight_i + bias_i$ | 常规矩阵乘，不涉及任何缩放/偏移 |

#### 4.2.2 <abbr title="weight 以低精度整数存储，计算时通过 antiquantScale/Offset 反量化为浮点">伪量化</abbr>场景

> weight 以低精度存储，计算时反量化为浮点。

| 子场景 | x dtype | weight dtype | bias | 计算公式 | 说明 |
|--------|:---:|:---:|:---:|:---|------|
| A16W4/A16W8 | FP16, BF16 | INT4, INT8 | 可选 | $y_i = x_i \times (weight_i + antiquant\\_offset_i) \times antiquant\\_scale_i + bias_i$ | 仅支持 x、weight、y 均为单 Tensor |
| A8W4 (S8S4) | INT8 | INT4 | **必选** (FLOAT32) | $y_i = ((x_i - 8) \times weight_i \times scale_i + bias_i) \times per\\_token\\_scale_i$ | 仅支持单 Tensor；$bias_i = 8 \times weight_i \times scale_i$ 沿 K 轴规约 |
| A16F8 | FP16, BF16 | FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8 | 可选 | $y_i = x_i \times (weight_i + antiquant\\_offset_i) \times antiquant\\_scale_i + bias_i$ | 仅支持 Ascend 950PR/950DT |

#### 4.2.3 <abbr title="x 和 weight 均以低精度参与计算，结果通过 scale/offset/perTokenScale 还原">全量化</abbr>场景

##### <abbr title="量化参数离线预计算，推理时不变">静态</abbr>量化

| 量化模式 | x dtype | weight dtype | bias | 计算公式 | 说明 |
|----------|:---:|:---:|:---:|:---|------|
| **T-C / T-T** | INT8 | INT8 | 无 | $y_i = (x_i \times weight_i) \times scale_i + offset_i$ | 基础公式，无 perTokenScaleOptional |
| **T-C / T-T** | INT8 | INT8 | INT32 | $y_i = (x_i \times weight_i + bias_i) \times scale_i + offset_i$ | bias 在 scale 之前累加 |
| **T-C / T-T** | INT8 | INT8 | FP16, BF16, FP32 | $y_i = (x_i \times weight_i) \times scale_i + bias_i$ | 无 offset 时适用 |

##### <abbr title="量化参数在线计算，依赖实际输入数据">动态</abbr>量化

| 量化模式 | x dtype | weight dtype | bias | 计算公式 | 说明 |
|----------|:---:|:---:|:---:|:---|------|
| **T-T / T-C / K-T / K-C** | INT8 | INT8 | 无 | $y_i = (x_i \times weight_i) \times scale_i \times per\\_token\\_scale_i$ | 基础公式 |
| **T-T / T-C / K-T / K-C** | INT8 | INT8 | INT32 | $y_i = (x_i \times weight_i + bias_i) \times scale_i \times per\\_token\\_scale_i$ | bias 在 scale 之前累加 |
| **T-T / T-C / K-T / K-C** | INT8 | INT8 | FP16, BF16, FP32 | $y_i = (x_i \times weight_i) \times scale_i \times per\\_token\\_scale_i + bias_i$ | bias 在 perTokenScale 之后累加 |

##### 特殊 pergroup 量化

| 量化模式 | block size | 计算公式 | 说明 |
|----------|:---:|:---|------|
| **<a id="mx量化"></a>MX** | gsM=1, gsN=1, gsK=32 | $y_i[m,n] = \sum_{j=0}^{kLoops-1} \bigg( \Big( \sum_{k=0}^{gsK-1} xSlice_i \times weightSlice_i \Big) \times \big( per\\_token\\_scale_i[m/gsM, j] \times scale_i[j, n/gsN] \big) \bigg) + bias_i[n]$ | 特殊的 pergroup 量化 |
| **<a id="gb量化"></a>G-B** | gsM=1, gsN=gsK=128 | $y_i[m,n] = \sum_{j=0}^{kLoops-1} \bigg( \Big( \sum_{k=0}^{gsK-1} xSlice_i \times weightSlice_i \Big) \times \big( per\\_token\\_scale_i[m/gsM, j] \times scale_i[j, n/gsN] \big) \bigg) + bias_i[n]$ | pergroup-perblock 量化 |

> 其中 $xSlice_i$ 为 $x_i$ 第 $m$ 行长度为 gsK 的向量，$weightSlice_i$ 为 $weight_i$ 第 $n$ 列长度为 gsK 的向量，$kLoops = \lceil K_i / gsK \rceil$，支持末尾切片长度不足 gsK。

### 4.3 版本演进

| 版本变化 | Ascend 950PR/Ascend 950DT | Atlas A3/A2 训练系列产品 |
|---------|----------------|----------|
| V4 → V5 | — | 增加可选参数 `tuningConfigOptional`，数组中第一个值表示各专家处理的 token 数预期值，tiling 时按该值进行最优切分 |
| V1 → V4 | 支持分组轴（groupType）；非量化支持 x/weight 转置；支持静态量化（T-C/T-T）BFLOAT16/FLOAT16/FLOAT32 输出 + bias；支持动态量化（K-C/K-T/T-T/T-C/MX/G-B）BFLOAT16/FLOAT16/FLOAT32 输出 + bias；支持伪量化 weight 为 INT4/FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8（仅 perchannel 模式） | 支持分组轴（groupType）；非量化支持 x/weight 转置；支持 x/weight/y 均为单 Tensor 非量化 FLOAT32 输入；支持伪量化 weight=INT4（perchannel/pergroup 模式） |

**确定性计算**：GroupedMatmul 默认提供确定性实现。

---

## 5. 参数说明

> 标注规则：
> - <sup>1</sup>：仅 <term>Ascend 950PR/950DT</term> 不支持
> - <sup>2</sup>：仅 <term>Atlas A3/A2</term> 不支持

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|:---:|------|----------|:---:|
| x | 输入 | 公式中的输入 $x$ | FLOAT、FLOAT16、INT16<sup>1</sup>、INT8、INT4<sup>1</sup>、BFLOAT16、FLOAT8_E5M2<sup>2</sup>、FLOAT8_E4M3FN<sup>2</sup>、HIFLOAT8<sup>2</sup>、FLOAT4_E2M1<sup>2</sup> | ND |
| weight | 输入 | 公式中的 $weight$ | FLOAT、FLOAT16、INT16<sup>1</sup>、INT8、INT4、BFLOAT16、FLOAT8_E5M2<sup>2</sup>、FLOAT8_E4M3FN<sup>2</sup>、HIFLOAT8<sup>2</sup>、FLOAT4_E2M1<sup>2</sup> | ND/NZ |
| biasOptional | 可选输入 | 公式中的 $bias$ | FLOAT、FLOAT16、INT32、BFLOAT16<sup>2</sup> | ND |
| scaleOptional | 可选输入 | 公式中的 $scale$，代表量化参数中的缩放因子 | FLOAT、UINT64、BFLOAT16、FLOAT8_E8M0<sup>2</sup>、INT64<sup>2</sup> | ND |
| offsetOptional | 可选输入 | 公式中的 $offset$，代表量化参数中的偏移量 | FLOAT | ND |
| antiquantScaleOptional | 可选输入 | 公式中的 $antiquant\_scale$，代表伪量化参数中的缩放因子 | FLOAT16、BFLOAT16 | ND |
| antiquantOffsetOptional | 可选输入 | 公式中的 $antiquant\_offset$，代表伪量化参数中的偏移量 | FLOAT16、BFLOAT16 | ND |
| perTokenScaleOptional | 可选输入 | 公式中的 $per\_token\_scale$，代表量化参数中由 x 量化引入的缩放因子 | FLOAT、FLOAT8_E8M0<sup>2</sup> | ND |
| groupListOptional | 可选输入 | 代表输入和输出分组轴方向的 matmul 大小分布 | INT64 | ND |
| activationInputOptional | 可选输入 | 代表激活函数的反向输入，当前只支持传入 nullptr | — | — |
| activationQuantScaleOptional | 可选输入 | 激活函数量化缩放因子（预留参数），当前只支持传入 nullptr | — | — |
| activationQuantOffsetOptional | 可选输入 | 激活函数量化偏移量（预留参数），当前只支持传入 nullptr | — | — |
| <abbr title="代表输出是否要做 Tensor 切分：0/1=多 Tensor 输出；2/3=单 Tensor 输出">splitItem</abbr> | 属性 | 代表输出是否要做 Tensor 切分 | INT64 | — |
| <abbr title="指定分组轴：-1=不分组，0=M轴，2=K轴">groupType</abbr> | 属性 | 代表需要分组的轴 | INT64 | — |
| <abbr title="代表 groupList 输入的分组方式：0=累积和；1=各组大小；2=[组索引, 组大小] 对">groupListType</abbr> | 属性 | 代表 groupList 输入的分组方式 | INT64 | — |
| <abbr title="代表激活函数类型：0=NONE/1=ReLU/2=GELU_TANH/4=FAST_GELU/5=SILU">actType</abbr> | 属性 | 代表激活函数类型 | INT64 | — |
| tuningConfigOptional | 可选输入 | 第一个数代表各个专家处理的 token 数的预期值；第二个数代表 A8W4 可选开启 weight 先转置的 NZ 格式；第三个数代表允许额外使用的内存空间 | INT64 | — |
| out | 输出 | 公式中的输出 $y$ | FLOAT、FLOAT16、INT32、INT8、BFLOAT16 | ND |
| activationFeatureOutOptional | 输出 | 激活函数的输入数据，当前只支持传入 nullptr | — | — |
| dynQuantScaleOutOptional | 输出 | 动态量化缩放因子，当前只支持传入 nullptr | — | — |

- **<term>Ascend 950PR/Ascend 950DT</term>**：
  - 上表数据类型列中的角标 <sup>1</sup> 代表该系列不支持的数据类型。
  - 输入参数 x、weight 均不支持 INT16 类型，且 x 不支持 INT4 类型。
  - 输入参数 x、weight，输出参数 out 在非量化场景支持最多 1024 个 Tensor，在伪量化支持最多 128 个 Tensor，在全量化场景最多支持 1 个 Tensor。
- **<term>Atlas A3/A2 训练系列产品</term>**：
  - 上表数据类型列中的角标 <sup>2</sup> 代表该系列不支持的数据类型。
  - 不支持 FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8、FLOAT8_E8M0 类型。
  - 输入参数 biasOptional 不支持 BFLOAT16。
  - 输入参数 scaleOptional 不支持 INT64 类型。
  - 输入参数 x、weight，输出参数 out 支持最多 128 个 Tensor。
- **<term>Kirin X90/Kirin 9030 处理器系列产品</term>**：
  - 不支持 BFLOAT16、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT8_E8M0、HIFLOAT8 类型。

---

## 6. 约束说明

**确定性计算**：GroupedMatmul 默认提供确定性实现。

> 各场景详细约束请参见 [aclnnGroupedMatmulV5 约束说明](docs/aclnnGroupedMatmulV5.md#5-约束说明)。

### 6.1 公共约束

- x/weight 中每组 Tensor 的最后一维 < 65536。
  - $x_i$ 最后一维：x 不转置时为 K 轴，x 转置时为 M 轴。
  - $weight_i$ 最后一维：weight 不转置时为 N 轴，weight 转置时为 K 轴。
- x/weight 转置时对应 Tensor 必须 [非连续](../../../docs/zh/context/非连续的Tensor.md)。
- weight 为 [FRACTAL_NZ](../../../docs/zh/context/数据格式.md) 格式时，shape 须满足 NZ 格式要求。
- perTokenScaleOptional：通常仅 1 维，长度与 x 的 M 一致。仅支持 x/weight/out 均为单 Tensor 场景。
- groupListOptional：out 中 TensorList 长度为 1 时，groupList 约束输出数据有效范围，未指定部分不参与更新。
- groupListType=0：groupListOptional 须为非负单调非递减数列（累积和）。
  groupListType=1：须为非负数列（各组大小）。
  groupListType=2：须为非负数列，shape `[E, 2]`（[组索引, 组大小]）。

<a id="62-atlas-a3a2-训练系列产品"></a>

### 6.2 Atlas A3/A2 训练系列产品

> "—" 表示该参数在当前场景下必须传空（nullptr）

<details>
<summary>数据类型速查表</summary>

| 类型 | x | weight | bias | scale | antiquantScale | antiquantOffset | perTokenScale | groupList | out |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 非量化 | FLOAT | FLOAT (ND) | FLOAT/null | — | — | — | — | INT64 | FLOAT |
| 非量化 | FLOAT16 | FLOAT16 (ND/NZ) | FLOAT16/null | — | — | — | — | INT64 | FLOAT16 |
| 非量化 | BFLOAT16 | BFLOAT16(ND/NZ) | FLOAT/null | — | — | — | — | INT64 | BFLOAT16 |
| 伪量化-A16W8 | FLOAT16 | INT8 (ND) | FLOAT16/null | — | FLOAT16 | FLOAT16 | — | INT64 | FLOAT16 |
| 伪量化-A16W8 | BFLOAT16 | INT8 (ND) | FLOAT/null | — | BFLOAT16 | BFLOAT16 | — | INT64 | BFLOAT16 |
| 伪量化-A16W4 | FLOAT16 | INT4 (ND) | FLOAT16/null | — | FLOAT16 | FLOAT16 | — | INT64 | FLOAT16 |
| 伪量化-A16W4 | BFLOAT16 | INT4 (ND) | FLOAT/null | — | BFLOAT16 | BFLOAT16 | — | INT64 | BFLOAT16 |
| 伪量化-A8W4 | INT8 | INT4 (ND/NZ) | FLOAT | UINT64 | — | FLOAT/null | FLOAT | INT64 | BFLOAT16/FLOAT16 |
| 全量化-A8W8 | INT8 | INT8 (ND/NZ) | INT32/null | UINT64/BFLOAT16/FLOAT | — | — | —/FLOAT | INT64 | INT8/BFLOAT16/FLOAT16/INT32 |
| 全量化-A4W4 | INT4 | INT4 (ND/NZ) | — | UINT64 | — | — | FLOAT/null | INT64 | FLOAT16/BFLOAT16 |

> 其余参数（offset、activationXxx 等）在各场景下均须传空，详见 [aclnnGroupedMatmulV5 约束说明](docs/aclnnGroupedMatmulV5.md#53-atlas-a3a2-训练系列产品)。

</details>

<details>
<summary>groupType 支持场景</summary>

> x、weight、y 输入为 aclTensorList。"单" = 单元素 TensorList，"多" = 多元素 TensorList。

| groupType | x | weight | y | splitItem | groupListOptional | 转置 | 其余场景限制 |
|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|
| -1 | 多 | 多 | 多 | 0/1 | 必须传空 | x 不转置；weight 可转置（统一） | x 2-6维（维度一致），weight 2维，y 同 x |
| 0 | 单 | 单 | 单 | 2/3 | 必须传 | x 不转置；weight 可转置（A8W4/A4W4 不可） | weight 3维，x/y 2维 |
| 0 | 单 | 多 | 单 | 2/3 | 必须传 | x 不转置；weight 可转置（统一） | x/weight/y 2维；weight N 轴相等 |
| 0 | 多 | 多 | 单 | 2/3 | 可选 | x 不转置；weight 可转置（统一） | x/weight/y 2维；weight N 轴相等 |
| 2 | 单 | 单 | 单 | 2/3 | 必须传 | x 必须转置；weight 不转置 | x/weight 2维，y 3维；bias 传空 |
| 2 | 单 | 多 | 多 | 0/1 | 必须传空 | x 必须转置；weight 不转置 | x/weight/y 2维；weight 最大 128 组；bias 传空 |

</details>

<a id="63-ascend-950prascend-950dt"></a>

### 6.3 Ascend 950PR/Ascend 950DT

- groupType：支持 M 轴分组（0）和不分组（-1）。非量化/全量化额外支持 K 轴分组（2）。
- groupListType：支持 0、1、2。仅全量化 groupType=0 时支持 groupListType=2。
- tuningConfigOptional：不支持。
- actType（0~5）：
  - 非量化/伪量化仅支持 0。
  - 全量化下 x/weight=INT8、静态 T-C 或动态 K-C、scale=FLOAT32/BFLOAT16 时支持 0/1/2/4/5；其余仅支持 0。

<details>
<summary>数据类型速查表</summary>

**非量化：**

| groupType | x | weight | biasOptional | out |
|:---:|:---:|:---:|:---|:---|
| -1/0/2 | BFLOAT16 | BFLOAT16 | BFLOAT16/FLOAT32/null | BFLOAT16 |
| -1/0/2 | FLOAT16 | FLOAT16 | FLOAT16/FLOAT32/null | FLOAT16 |
| -1/0/2 | FLOAT32 | FLOAT32 | FLOAT32/null | FLOAT32 |

**静态量化：**

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

**动态量化（T-T/T-C/K-T/K-C）：**

| groupType | x | weight | biasOptional | scaleOptional | perTokenScaleOptional | out |
|:---:|:---:|:---:|:---|:---|:---|:---|
| 0 | INT8 | INT8 | INT32/BFLOAT16/FLOAT32/null | BFLOAT16/FLOAT32 | FLOAT32 | BFLOAT16 |
| 0 | INT8 | INT8 | INT32/FLOAT16/FLOAT32/null | FLOAT32 | FLOAT32 | FLOAT16 |
| 0/2 | HIFLOAT8 | HIFLOAT8 | null | FLOAT32 | FLOAT32 | BFLOAT16/FLOAT16/FLOAT32 |
| 0/2 | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E5M2/FLOAT8_E4M3FN | null | FLOAT32 | FLOAT32 | BFLOAT16/FLOAT16/FLOAT32 |

**动态量化（MX）：**

| groupType | x | weight | biasOptional | scaleOptional | perTokenScaleOptional | out |
|:---:|:---:|:---:|:---:|:---:|:---|:---|
| 0/2 | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E5M2/FLOAT8_E4M3FN | null | FLOAT8_E8M0 | FLOAT8_E8M0 | BFLOAT16/FLOAT16/FLOAT32 |
| 0 | FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT32/null | FLOAT8_E8M0 | FLOAT8_E8M0 | BFLOAT16/FLOAT16/FLOAT32 |

**动态量化（G-B）：**

| groupType | x | weight | scaleOptional | perTokenScaleOptional | out |
|:---:|:---:|:---:|:---|:---|:---|
| 0/2 | HIFLOAT8 | HIFLOAT8 | FLOAT32 | FLOAT32 | BFLOAT16/FLOAT16/FLOAT32 |
| 0/2 | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT32 | FLOAT32 | BFLOAT16/FLOAT16/FLOAT32 |

**伪量化：**

| groupType | x | weight | biasOptional | antiquantScaleOptional | antiquantOffsetOptional | out |
|:---:|:---:|:---:|:---|:---|:---|:---|
| -1/0 | BFLOAT16 | INT8/INT4 | BFLOAT16/FLOAT32/null | BFLOAT16 | BFLOAT16/null | BFLOAT16 |
| -1/0 | FLOAT16 | INT8/INT4 | FLOAT16/null | FLOAT16 | FLOAT16/null | FLOAT16 |
| 0 | BFLOAT16 | FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8 | BFLOAT16/FLOAT32/null | BFLOAT16 | null | BFLOAT16 |
| 0 | FLOAT16 | FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8 | FLOAT16/null | FLOAT16 | null | FLOAT16 |

- weight 为 FLOAT8 系列时，antiquantOffsetOptional 仅支持空指针或空 TensorList，weight 仅支持转置。

</details>

<details>
<summary>groupType 支持场景</summary>

> x、weight、y 输入为 aclTensorList。"单" = 单元素 TensorList，"多" = 多元素 TensorList。

| groupType | x | weight | y | splitItem | groupListOptional | 转置 | 其余场景限制 |
|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|
| -1 | 多 | 多 | 多 | 0/1 | 必须传空 | x 不转置；weight 可转置（统一） | 非量化 x/out 2维，伪量化 x 2~6维；weight 2维；仅非量化/伪量化；仅 ND 进 ND 出 |
| 0 | 单 | 单 | 单 | 2/3 | 必须传 | x 不转置；weight 可转置 | weight 3维 `(g,N,K)` 或 `(g,K,N)`；x/out 2维；bias 2维 `(g,N)`；最多 1024 组；仅 ND 出 |
| 0 | 单 | 多 | 单 | 2/3 | 必须传（长度 ≤1024） | x 不转置；weight 可转置（统一） | x/weight/y 2维；weight N 轴相等；仅 ND 进 ND 出；仅非量化 |
| 0 | 多 | 多 | 单 | 2 | 可选 | x 不转置；weight 可转置（统一） | x/weight/y 2维；weight N 轴相等；仅 ND 进 ND 出；仅非量化 |
| 2 | 单 | 单 | 单 | 2/3 | 必须传 | x 必须转置；weight 不转置 | x/weight 2维 `(K,M)`/`(K,N)`；y 3维 `(g,M,N)`；bias 传空；仅 ND 进 ND 出；仅非量化/量化 |
| 2 | 单 | 多 | 多 | 0/1 | 可选 | x 必须转置；weight 不转置 | x/weight/y 2维 `(M,N)`；weight 最大 128 组；不支持 bias；仅 ND 进 ND 出；仅非量化 |

</details>

---

## 7. 接口调用

| 调用方式 | 接口文档 | 调用样例 | 说明 |
|----------|----------|----------|------|
| aclnn 调用 | [aclnnGroupedMatmulV5](docs/aclnnGroupedMatmulV5.md) | 见接口文档 [示例代码索引](docs/aclnnGroupedMatmulV5.md#63-示例代码索引) | 两段式接口：先调用 `aclnnGroupedMatmulV5GetWorkspaceSize` 获取 workspace 大小和 executor，再调用 `aclnnGroupedMatmulV5` 执行计算 |
