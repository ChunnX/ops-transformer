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
  - [3.1 场景分类](#31-场景分类)
  - [3.2 量化模式](#32-量化模式)
  - [3.3 量化粒度](#33-量化粒度)
- [4. 功能说明](#4-功能说明)
  - [4.1 分组方式](#41-分组方式)
  - [4.2 计算公式](#42-计算公式)
  - [4.3 版本演进](#43-版本演进)
- [5. 接口调用](#5-接口调用)

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

> 本章集中解释下文中出现的简称和专业术语，建议在阅读后续章节前先了解。

<a id="31-场景分类"></a>

### 3.1 场景分类

GroupedMatmul 根据计算过程中对数据精度的处理方式，分为三大场景：

| 场景       | 含义 | 简单理解 |
|------------|------|----------|
| **非量化** | x 和 weight 均以浮点类型（FLOAT16/BFLOAT16/FLOAT32）直接参与计算 | 常规矩阵乘，不涉及任何缩放/偏移 |
| **伪量化** | weight 以低精度整数（INT4/INT8/FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8）存储，计算时通过 `antiquantScale` / `antiquantOffset` 反量化为浮点再与 x 相乘 | "存储时省空间，计算时还原" |
| **全量化** | x 和 weight 均以整数类型（INT8/INT4/FLOAT8 等）参与计算，结果通过 `scale` / `offset` / `perTokenScale` 还原为浮点 | "全程低精度计算，最后还原" |

<a id="32-量化模式"></a>

### 3.2 量化模式

量化模式描述了 x 和 weight 分别采用什么粒度的量化策略，命名格式为 `X侧量化粒度 - Weight侧量化粒度`：

| 简称 | 全称 | x 侧量化粒度 | weight 侧量化粒度 |
|------|------|:---:|:---:|
| **T-C** | Tensor-Channel（pertensor-perchannel） | 按 Tensor（x 全体共享一套量化参数） | 按 Channel（weight 每列一套量化参数） |
| **T-T** | Tensor-Tensor（pertensor-pertensor） | 按 Tensor | 按 Tensor |
| **K-C** | perToken-perChannel（pertoken-perchannel） | 按 Token（x 每行一套量化参数） | 按 Channel |
| **K-T** | perToken-perTensor（pertoken-pertensor） | 按 Token | 按 Tensor |
| **G-B** | perGroup-perBlock（pergroup-perblock） | 按 Group（x 按 perTokenScale 的 block 分组） | 按 Block（weight 按 K/N 方向 block 分组） |
| **MX** | MX 量化 | 按 Token + block | 按 Token + block（特殊的 pergroup 量化，block size: M=1, N=1, K=32） |

<a id="33-量化粒度"></a>

### 3.3 量化粒度

| 术语 | 含义 | 举例 |
|------|------|------|
| **perchannel** | 对 weight 的每一列（N 方向）使用独立的缩放因子 | scale 的 shape 为 `(E, N)` |
| **pertensor** | 对一个 Tensor 整体使用统一的缩放因子 | scale 的 shape 为 `(E, 1)` |
| **pertoken** | 对 x 的每一行（M 方向）使用独立的缩放因子 | perTokenScale 的 shape 为 `(M,)` |
| **pergroup** | 对 weight 按 K 方向等分为若干组（Group），每组使用独立的缩放因子 | scale 的 shape 为 `(E, G, N)` |
| **perblock** | 对 weight 按 K/N 方向固定大小（如 128）分块，每块独立缩放 | 参见 [G-B 量化](#gb量化) |

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

## 5. 接口调用

| 调用方式 | 接口文档 | 调用样例 | 说明 |
|----------|----------|----------|------|
| aclnn 调用 | [aclnnGroupedMatmulV5](docs/aclnnGroupedMatmulV5.md) | 见接口文档 [示例代码索引](docs/aclnnGroupedMatmulV5.md#82-示例代码索引) | 两段式接口：先调用 `aclnnGroupedMatmulV5GetWorkspaceSize` 获取 workspace 大小和 executor，再调用 `aclnnGroupedMatmulV5` 执行计算 |
