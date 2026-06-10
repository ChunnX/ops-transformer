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

## 5. 接口调用

| 调用方式 | 接口文档 | 调用样例 | 说明 |
|----------|----------|----------|------|
| aclnn 调用 | [aclnnGroupedMatmulV5](docs/aclnnGroupedMatmulV5.md) | 见接口文档 [示例代码索引](docs/aclnnGroupedMatmulV5.md#63-示例代码索引) | 两段式接口：先调用 `aclnnGroupedMatmulV5GetWorkspaceSize` 获取 workspace 大小和 executor，再调用 `aclnnGroupedMatmulV5` 执行计算 |
