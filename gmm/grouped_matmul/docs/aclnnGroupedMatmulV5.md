# aclnnGroupedMatmulV5

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/gmm/grouped_matmul)

## 目录

- [1. 产品支持情况](#1-产品支持情况)
- [2. 专业术语](#2-专业术语)
- [3. 功能说明](#3-功能说明)
  - [3.1 接口功能概览](#31-接口功能概览)
  - [3.2 两段式接口说明](#32-两段式接口说明)
- [4. 函数原型](#4-函数原型)
- [5. 场景决策](#5-场景决策)
  - [5.1 决策流程](#51-决策流程)
  - [5.2 场景→必传参数速查](#52-场景必传参数速查)
  - [5.3 典型最小调用流程](#53-典型最小调用流程)
- [6. 参数说明](#6-参数说明)
  - [6.1 通用参数](#61-通用参数)
  - [6.2 非量化场景参数](#62-非量化场景参数)
  - [6.3 伪量化场景参数](#63-伪量化场景参数)
  - [6.4 全量化场景参数](#64-全量化场景参数)
  - [6.5 aclnnGroupedMatmulV5 参数](#65-aclnngroupedmatmulv5-参数)
  - [6.6 返回值与错误码](#66-返回值与错误码)
- [7. 约束说明](#7-约束说明)
  - [7.1 公共约束](#71-公共约束)
  - [7.2 Ascend 950PR/Ascend 950DT](#72-ascend-950prascend-950dt)
  - [7.3 Atlas A3/A2 训练系列产品](#73-atlas-a3a2-训练系列产品)
- [8. 接口调用](#8-接口调用)
  - [8.1 调用流程](#81-调用流程)
  - [8.2 示例代码索引](#82-示例代码索引)

---

## 1. 产品支持情况

| 产品 | 是否支持 |
|:----------------------------|:-----------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

---

## 2. 专业术语

本章简要列出接口文档中涉及的术语，完整的概念性解释请参见 [GroupedMatmul README - 专业术语](../README.md#3-专业术语)。

| 类别 | 术语 | 简要说明 |
|------|------|----------|
| 场景 | 非量化 | x 和 weight 均为浮点类型直接计算 |
| 场景 | 伪量化 | weight 低精度存储，通过 antiquantScale/Offset 反量化后计算 |
| 场景 | 全量化 | x 和 weight 均为低精度计算，通过 scale 还原 |
| 量化模式 | T-C | pertensor-perchannel |
| 量化模式 | T-T | pertensor-pertensor |
| 量化模式 | K-C | pertoken-perchannel |
| 量化模式 | K-T | pertoken-pertensor |
| 量化模式 | G-B | pergroup-perblock |
| 量化模式 | MX | 特殊 pergroup 量化（block: M=1,N=1,K=32） |
| 量化粒度 | perchannel | 每列独立缩放因子 |
| 量化粒度 | pertensor | 一个 Tensor 统一缩放因子 |
| 量化粒度 | pertoken | 每行独立缩放因子 |
| 量化粒度 | pergroup | 按 K 方向等分多组 |
| 配置 | groupType | 分组轴：-1=不分组，0=M轴，2=K轴 |
| 配置 | splitItem | 0/1=多 Tensor 输出，2/3=单 Tensor 输出 |
| 配置 | groupListType | groupList 数据格式：0=累积和，1=各组大小，2=索引+大小 |
| 配置 | actType | 激活函数类型（0~5） |
| 数据格式 | ND | 常规连续排布 |
| 数据格式 | NZ | FRACTAL_NZ 格式（亲核排布） |

---

## 3. 功能说明

### 3.1 接口功能概览

实现分组矩阵乘计算。分组方式和计算公式详见 [GroupedMatmul README - 功能说明](../README.md#4-功能说明)。

> 详细计算公式和量化原理请参见 [GroupedMatmul README - 计算公式](../README.md#42-计算公式)。

> 版本演进详见 [GroupedMatmul README - 版本演进](../README.md#43-版本演进)。

### 3.2 两段式接口说明

算子采用两段式接口，调用顺序为：

1. **第一段** `aclnnGroupedMatmulV5GetWorkspaceSize`：校验入参、计算所需 workspace 大小、生成算子执行器（executor）。
2. **第二段** `aclnnGroupedMatmulV5`：使用第一段返回的 workspace 和 executor 在指定 Stream 上执行计算。

---

## 4. 函数原型

```c++
aclnnStatus aclnnGroupedMatmulV5GetWorkspaceSize(
    const aclTensorList *x,
    const aclTensorList *weight,
    const aclTensorList *biasOptional,
    const aclTensorList *scaleOptional,
    const aclTensorList *offsetOptional,
    const aclTensorList *antiquantScaleOptional,
    const aclTensorList *antiquantOffsetOptional,
    const aclTensorList *perTokenScaleOptional,
    const aclTensor     *groupListOptional,
    const aclTensorList *activationInputOptional,
    const aclTensorList *activationQuantScaleOptional,
    const aclTensorList *activationQuantOffsetOptional,
    int64_t              splitItem,
    int64_t              groupType,
    int64_t              groupListType,
    int64_t              actType,
    aclIntArray         *tuningConfigOptional,
    aclTensorList       *out,
    aclTensorList       *activationFeatureOutOptional,
    aclTensorList       *dynQuantScaleOutOptional,
    uint64_t            *workspaceSize,
    aclOpExecutor      **executor)
```

```c++
aclnnStatus aclnnGroupedMatmulV5(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

---

## 5. 场景决策

> **目的**：您已确定计算场景和产品代际，需要快速判断——哪些参数必传、哪些必须置空、支持哪些 dtype 组合。

### 5.1 决策流程

按以下步骤自检，定位到对应子表：

```
1. 确定场景    → 非量化 / 伪量化 / 全量化
2. 确定产品    → Ascend 950PR/950DT 或 Atlas A3/A2
3. 确定量化模式 → 仅全量化需确定：静态 / 动态(T-C/T-T/K-C/K-T) / MX / G-B
4. 确定分组轴  → groupType: -1(不分组) / 0(M轴) / 2(K轴)
5. 查对应约束  → [7.2 Ascend 950PR/950DT](#72-ascend-950prascend-950dt) 或 [7.3 Atlas A3/A2](#73-atlas-a3a2-训练系列产品) 查看 dtype 与约束详情
```

### 5.2 场景→必传参数速查

| 场景 | **必传参数** | **必须置空（传 nullptr）** |
|------|:---|:---|
| **非量化** | `x`, `weight`, `out` | `scaleOptional`, `offsetOptional`, `antiquantScaleOptional`, `antiquantOffsetOptional`, `perTokenScaleOptional` |
| **伪量化** | `x`, `weight`, `antiquantScaleOptional`, `out` | `scaleOptional`, `offsetOptional`, `perTokenScaleOptional` |
| **全量化-静态** | `x`, `weight`, `scaleOptional`, `out` | `antiquantScaleOptional`, `antiquantOffsetOptional`, `perTokenScaleOptional` |
| **全量化-动态** | `x`, `weight`, `scaleOptional`, `perTokenScaleOptional`, `out` | `antiquantScaleOptional`, `antiquantOffsetOptional`, `offsetOptional` |
| **全量化-MX/G-B** | `x`, `weight`, `scaleOptional`, `perTokenScaleOptional`, `out` | `antiquantScaleOptional`, `antiquantOffsetOptional`, `offsetOptional` |

> **biasOptional** 在所有场景中均可选。**groupListOptional** 根据 groupType 决定是否必传，详见 [groupType 场景矩阵](#72-ascend-950prascend-950dt)。

### 5.3 典型最小调用流程

<details>
<summary>非量化最小调用（M 轴分组，单 Tensor 输出）</summary>

```c++
// 参数清单：x, weight, out 共 3 个，其余全部传 nullptr
aclTensorList *x             = CreateTensorList({(g, M, K)}, FLOAT16);
aclTensorList *weight        = CreateTensorList({(g, K, N)}, FLOAT16);
aclTensorList *out           = CreateTensorList({(g, M, N)}, FLOAT16);
aclTensorList *groupList     = CreateTensorList(Ms, INT64);     // cumsum 格式，见 6.1 通用参数

aclnnGroupedMatmulV5GetWorkspaceSize(
    x, weight, nullptr, nullptr, nullptr,   // bias, scale, offset
    nullptr, nullptr, nullptr,              // antiquantScale, antiquantOffset, perTokenScale
    groupList,
    nullptr, nullptr, nullptr,              // activation 系列
    2, 0, 1, 0,                             // splitItem, groupType, groupListType, actType
    nullptr, out, nullptr, nullptr, &wsSize, &executor);
```

</details>

<details>
<summary>伪量化 A16W8 最小调用（M 轴分组，单 Tensor 输出，perchannel 量化）</summary>

```c++
// 参数清单：x, weight, antiquantScale, out 共 4 个
aclTensorList *x              = CreateTensorList({(g, M, K)}, FLOAT16);
aclTensorList *weight         = CreateTensorList({(g, K, N)}, INT8);
aclTensorList *antiquantScale = CreateTensorList({(g, N)},    FLOAT);  // perchannel
aclTensorList *out            = CreateTensorList({(g, M, N)}, FLOAT16);
aclTensorList *groupList      = CreateTensorList(Ms, INT64);

aclnnGroupedMatmulV5GetWorkspaceSize(
    x, weight, nullptr, nullptr, nullptr,   // bias, scale, offset
    antiquantScale, nullptr, nullptr,       // antiquantOffset, perTokenScale
    groupList,
    nullptr, nullptr, nullptr,
    2, 0, 1, 0, nullptr, out, nullptr, nullptr, &wsSize, &executor);
```

</details>

<details>
<summary>全量化静态 A8W8 最小调用（M 轴分组，单 Tensor 输出，perchannel 量化）</summary>

```c++
// 参数清单：x, weight, scale, out 共 4 个
aclTensorList *x         = CreateTensorList({(g, M, K)}, INT8);
aclTensorList *weight    = CreateTensorList({(g, K, N)}, INT8);
aclTensorList *scale     = CreateTensorList({(g, N)},    UINT64);  // perchannel
aclTensorList *out       = CreateTensorList({(g, M, N)}, BFLOAT16);
aclTensorList *groupList = CreateTensorList(Ms, INT64);

aclnnGroupedMatmulV5GetWorkspaceSize(
    x, weight, nullptr, scale, nullptr,     // bias, scale, offset
    nullptr, nullptr, nullptr,              // antiquant 系列, perTokenScale
    groupList,
    nullptr, nullptr, nullptr,
    2, 0, 1, 0, nullptr, out, nullptr, nullptr, &wsSize, &executor);
```

</details>

---

## 6. 参数说明

> 标注规则：
> - <span title="Ascend 950PR/950DT 不支持"><sup>1</sup></span>：仅 Ascend 950PR/950DT 不支持
> - <span title="Atlas A3/A2 不支持"><sup>2</sup></span>：仅 Atlas A3/A2 不支持

<a id="61-通用参数"></a>

### 6.1 通用参数

以下参数在所有场景中有效（作为 `aclnnGroupedMatmulV5GetWorkspaceSize` 的入参）：

#### Tensor 类参数

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|--------|:---:|------|------|----------|:---:|:---|:---:|
| x | 输入 | 公式中的输入 $x$ | TensorList 长度 [1,128] 或 [1,1024]（A5 非量化） | FLOAT、FLOAT16、INT16<span title="仅 Ascend 950PR/950DT 不支持"><sup>1</sup></span>、INT8、INT4<span title="仅 Ascend 950PR/950DT 不支持"><sup>1</sup></span>、BFLOAT16、FLOAT8_E5M2<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span>、FLOAT8_E4M3FN<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span>、HIFLOAT8<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span>、FLOAT4_E2M1<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span> | ND | 2~6 | √ |
| weight | 输入 | 公式中的 $weight$ | TensorList 长度 [1,128] 或 [1,1024] | FLOAT、FLOAT16、INT16<span title="仅 Ascend 950PR/950DT 不支持"><sup>1</sup></span>、INT8、INT4、BFLOAT16、FLOAT8_E5M2<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span>、FLOAT8_E4M3FN<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span>、HIFLOAT8<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span>、FLOAT4_E2M1<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span> | ND | 2~3 | √ |
| out | 输出 | 公式中的输出 $y$ | TensorList 长度 [1,128] 或 [1,1024] | FLOAT、FLOAT16、INT32、INT8、BFLOAT16 | ND | 2~3 | N/A |
| groupListOptional | 可选输入 | 分组轴方向各组大小分布 | 格式取决于 groupListType | INT64 | ND | 1~2 | √ |
| tuningConfigOptional | 可选输入 | [0]=预期 token 数（优化 tiling）；[1]=A8W4 场景是否使能 weight NZ 格式；[2]=允许额外内存上限 | 兼容历史版本，不用时传 `nullptr` | INT64 | ND | 3 | N/A |

#### 属性类参数

| 参数名 | 输入/输出 | 描述 | 取值说明 |
|--------|:---:|------|------|
| splitItem | 输入 | 输出是否做 Tensor 切分 | `0/1` = 多 Tensor 输出；`2/3` = 单 Tensor 输出。接口不区分 0/1 和 2/3 的内部差异 |
| groupType | 输入 | 分组轴 | `-1` = 不分组；`0` = M 轴分组；`2` = K 轴分组 |
| groupListType | 输入 | groupList 数据格式 | `0` = 累积和（cumsum）；`1` = 各组大小；`2` = [组索引, 组大小] 对。详见 [约束说明](#7-约束说明) |
| actType | 输入 | 激活函数类型 | `0`=NONE / `1`=RELU / `2`=GELU_TANH / `3`=GELU_ERR_FUNC(不支持) / `4`=FAST_GELU / `5`=SILU |
| workspaceSize | 输出 | 需要在 Device 侧申请的 workspace 大小（字节） | N/A |
| executor | 输出 | 算子执行器，包含计算流程 | N/A |

**产品代际差异：**

- **Ascend 950PR/Ascend 950DT**：
  - x/weight 不支持 INT16；x 不支持 INT4
  - x/weight/out 在非量化支持最多 1024 个 Tensor，伪量化最多 128 个，全量化最多 1 个
- **Atlas A3/A2**：
  - 不支持 FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8、FLOAT8_E8M0 类型
  - biasOptional 不支持 BFLOAT16
  - scaleOptional 不支持 INT64
  - x/weight/out 均最多 128 个 Tensor

<a id="62-非量化场景参数"></a>

### 6.2 非量化场景参数

<details>
<summary>展开查看非量化场景参数</summary>

以下参数必须传空：`scaleOptional`、`offsetOptional`、`antiquantScaleOptional`、`antiquantOffsetOptional`、`perTokenScaleOptional`、`activationInputOptional`、`activationQuantScaleOptional`、`activationQuantOffsetOptional`。

| 参数名 | 非量化特有说明 | 数据类型 | 维度(shape) |
|--------|------|----------|:---|
| biasOptional | 可选偏置 | FLOAT、FLOAT16、INT32、BFLOAT16<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span> | 1~2 |

</details>

<a id="63-伪量化场景参数"></a>

### 6.3 伪量化场景参数

<details>
<summary>展开查看伪量化场景参数</summary>

以下参数必须传空：`scaleOptional`、`offsetOptional`、`perTokenScaleOptional`、`activationInputOptional`、`activationQuantScaleOptional`、`activationQuantOffsetOptional`。

> **例外**：x 为 INT8、weight 为 INT4 的 A8W4 伪量化场景中 `scaleOptional`/`offsetOptional` 可能生效，详见 [A8W4 场景约束](#a8w4场景约束)。

<a id="631-perchannel-模式"></a>

#### 6.3.1 perchannel 模式

| 参数名 | 说明 | 数据类型 | 维度(shape) |
|--------|------|----------|:---|
| antiquantScaleOptional | 伪量化缩放因子 | FLOAT16、BFLOAT16 | 1~3 |
| antiquantOffsetOptional | 伪量化偏移量 | FLOAT16、BFLOAT16 | 1~3 |
| biasOptional | 可选偏置 | FLOAT、FLOAT16、BFLOAT16<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span> | 1~2 |

<a id="632-pergroup-模式"></a>

#### 6.3.2 pergroup 模式（仅 A16W4、weight=INT4 时支持）

shape 约束参见 [约束说明](#73-atlas-a3a2-训练系列产品) 中对应的伪量化场景约束。

</details>

<a id="64-全量化场景参数"></a>

### 6.4 全量化场景参数

<details>
<summary>展开查看全量化场景参数</summary>

以下参数必须传空：`antiquantScaleOptional`、`antiquantOffsetOptional`、`activationInputOptional`。

<a id="641-静态量化参数子集"></a>

#### 6.4.1 静态量化参数子集

| 参数名 | 说明 | 数据类型 | 维度(shape) |
|--------|------|----------|:---|
| scaleOptional | 量化缩放因子 | FLOAT、UINT64、BFLOAT16、FLOAT8_E8M0<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span>、INT64<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span> | 1~4 |
| offsetOptional | 量化偏移量（可选） | FLOAT | 3 |
| biasOptional | 可选偏置 | FLOAT、FLOAT16、INT32、BFLOAT16<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span> | 1~2 |

> `perTokenScaleOptional` 必须传空。

<a id="642-动态量化参数子集"></a>

#### 6.4.2 动态量化（T-T/T-C/K-T/K-C）参数子集

| 参数名 | 说明 | 数据类型 | 维度(shape) |
|--------|------|----------|:---|
| scaleOptional | 量化缩放因子 | FLOAT、UINT64、BFLOAT16 | 1~4 |
| perTokenScaleOptional | x 侧 per-token 缩放因子 | FLOAT、FLOAT8_E8M0<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span> | 1~3 |
| biasOptional | 可选偏置 | FLOAT、FLOAT16、INT32、BFLOAT16<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span> | 1~2 |

> `offsetOptional` 必须传空。

<a id="643-mx量化参数子集"></a>

#### 6.4.3 MX 量化参数子集

| 参数名 | 说明 | 数据类型 |
|--------|------|----------|
| scaleOptional | MX 缩放因子 | FLOAT8_E8M0 |
| perTokenScaleOptional | x 侧 MX 缩放因子 | FLOAT8_E8M0 |

> `offsetOptional`、`biasOptional`（x=FLOAT4_E2M1 时 bias 可为 FLOAT32）必须传空或按需设置。

<a id="644-g-b量化参数子集"></a>

#### 6.4.4 G-B 量化参数子集

| 参数名 | 说明 | 数据类型 |
|--------|------|----------|
| scaleOptional | perblock 缩放因子 | FLOAT32 |
| perTokenScaleOptional | x 侧 pergroup 缩放因子 | FLOAT32 |

> `biasOptional`、`offsetOptional` 必须传空。

<a id="645-激活函数辅助参数"></a>

#### 6.4.5 激活函数辅助参数

以下参数在全量化场景下用于激活函数相关计算（当前仅支持传空）：

| 参数名 | 输入/输出 | 描述 |
|--------|:---:|------|
| activationQuantScaleOptional | 输入 | 激活函数量化缩放因子（预留参数） |
| activationQuantOffsetOptional | 输入 | 激活函数量化偏移量（预留参数） |
| activationFeatureOutOptional | 输出 | 激活函数前向输入数据 |
| dynQuantScaleOutOptional | 输出 | 动态量化缩放因子 |

</details>

<a id="65-aclnngroupedmatmulv5-参数"></a>

### 6.5 aclnnGroupedMatmulV5 参数

第二段接口 `aclnnGroupedMatmulV5` 的参数：

| 参数名 | 输入/输出 | 描述 |
|--------|:---:|------|
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | workspace 大小，由第一段接口返回 |
| executor | 输入 | 算子执行器，由第一段接口返回 |
| stream | 输入 | 指定执行任务的 Stream |

<a id="66-返回值与错误码"></a>

### 6.6 返回值与错误码

**返回类型**：`aclnnStatus`，具体参见 [aclnn 返回码](../../../docs/zh/context/aclnn返回码.md)。

第一段接口 `aclnnGroupedMatmulV5GetWorkspaceSize` 完成入参校验，以下场景报错：

| 返回值 | 错误码 | 描述 |
|--------|:---:|------|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 必选参数是空指针 |
| | | weight 元素存在空指针 |
| | | x 元素为空但 out 元素不为空 |
| | | x 元素不为空但 out 元素为空 |
| ACLNN_ERR_PARAM_INVALID | 161002 | x/weight/biasOptional/scaleOptional/offsetOptional/antiquantScaleOptional/antiquantOffsetOptional/groupListOptional/out 的数据类型/数据格式不在支持范围 |
| | | weight 长度不在支持范围 |
| | | bias 不为空时长度不等于 weight 长度 |
| | | groupListOptional 维度不符合要求 |
| | | splitItem=2/3 时 out 长度不等于 1 |
| | | splitItem=0/1 时 out 长度不等于 weight 长度，或 groupListOptional 长度不等于 weight 长度 |
| | | tuningConfigOptional 元素为负数或大于 x 行数 M |

---

## 7. 约束说明

**确定性计算**：aclnnGroupedMatmulV5 默认确定性实现。

> "—" 表示该参数在当前场景下必须传空（nullptr）

<a id="71-公共约束"></a>

### 7.1 公共约束

以下约束在所有场景和产品代际下通用：

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

<a id="72-ascend-950prascend-950dt"></a>

### 7.2 Ascend 950PR/Ascend 950DT

<details>
<summary>展开查看 Ascend 950PR/950DT 约束详情</summary>

- groupType：支持 M 轴分组（0）和不分组（-1）。非量化/全量化额外支持 K 轴分组（2）。
- groupListType：支持 0、1、2。仅全量化 groupType=0 时支持 groupListType=2。
- tuningConfigOptional：不支持。
- actType（0~5）：
  - 非量化/伪量化仅支持 0。
  - 全量化下 x/weight=INT8、静态 T-C 或动态 K-C、scale=FLOAT32/BFLOAT16 时支持 0/1/2/4/5；其余仅支持 0。

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
| 2 | `(K/64 + g, N, 2)` |

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
| 2 | `(K/gsK + g, ceil(N/gsN))` |

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

> x、weight、y 输入为 aclTensorList。"单" = 单元素 TensorList，"多" = 多元素 TensorList。

| groupType | x | weight | y | splitItem | groupListOptional | 转置 | 其余场景限制 |
|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|
| -1 | 多 | 多 | 多 | 0/1 | 必须传空 | x 不转置；weight 可转置（统一） | 非量化 x/out 2维，伪量化 x 2~6维；weight 2维；仅非量化/伪量化；仅 ND 进 ND 出 |
| 0 | 单 | 单 | 单 | 2/3 | 必须传 | x 不转置；weight 可转置 | weight 3维 `(g,N,K)` 或 `(g,K,N)`；x/out 2维；bias 2维 `(g,N)`；最多 1024 组；仅 ND 出；INT8 场景 weight 支持 NZ 格式 |
| 0 | 单 | 多 | 单 | 2/3 | 必须传（长度 ≤1024） | x 不转置；weight 可转置（统一） | x/weight/y 2维；weight N 轴相等；仅 ND 进 ND 出；仅非量化 |
| 0 | 多 | 多 | 单 | 2 | 可选 | x 不转置；weight 可转置（统一） | x/weight/y 2维；weight N 轴相等；仅 ND 进 ND 出；仅非量化 |
| 2 | 单 | 单 | 单 | 2/3 | 必须传 | x 必须转置；weight 不转置 | x/weight 2维 `(K,M)`/`(K,N)`；y 3维 `(g,M,N)`；bias 传空；仅 ND 进 ND 出；仅非量化/量化 |
| 2 | 单 | 多 | 多 | 0/1 | 可选 | x 必须转置；weight 不转置 | x/weight/y 2维 `(M,N)`；weight 最大 128 组；不支持 bias；仅 ND 进 ND 出；仅非量化 |

#### groupListOptional 配置示例

以 shape：$M=789$、$K=4096$、$N=7168$、$E=9$（其中第 0 组 123 token，第 2/5 组各 333 token，其余为空）为例：

| groupListType | groupListOptional 值 |
|:---:|:---|
| 0（累积和） | `[123, 123, 456, 456, 456, 789, 789, 789, 789]` |
| 1（各组大小） | `[123, 0, 333, 0, 0, 333, 0, 0, 0]` |
| 2（索引+大小） | `[[0, 123], [2, 333], [5, 333], [1, 0], [3, 0], [4, 0], [6, 0], [7, 0], [8, 0]]`（非零组前置） |

</details>

<a id="73-atlas-a3a2-训练系列产品"></a>

### 7.3 Atlas A3/A2 训练系列产品

<details>
<summary>展开查看 Atlas A3/A2 约束详情</summary>

#### 7.3.1 场景分类与数据类型约束

| 场景名 | x | weight | out | 约束说明 |
|--------|:---:|:---:|:---:|------|
| 非量化 | FLOAT32 | FLOAT32 | FLOAT32 | [非量化场景约束](#非量化场景约束) |
| 非量化 | BFLOAT16 | BFLOAT16 | BFLOAT16 | [非量化场景约束](#非量化场景约束) |
| 非量化 | FLOAT16 | FLOAT16 | FLOAT16 | [非量化场景约束](#非量化场景约束) |
| 全量化-A8W8 | INT8 | INT8 | BFLOAT16/FLOAT16/INT32/INT8 | [A8W8场景约束](#a8w8场景约束) |
| 全量化-A4W4 | INT4 | INT4 | BFLOAT16/FLOAT16 | [A4W4场景约束](#a4w4场景约束) |
| 伪量化-A8W4 | INT8 | INT4 | BFLOAT16/FLOAT16 | [A8W4场景约束](#a8w4场景约束) |
| 伪量化-A16W8 | BFLOAT16/FLOAT16 | INT8 | BFLOAT16/FLOAT16 | [A16W8场景约束](#a16w8场景约束) |
| 伪量化-A16W4 | BFLOAT16/FLOAT16 | INT4 | BFLOAT16/FLOAT16 | [A16W4场景约束](#a16w4场景约束) |

<a id="非量化场景约束"></a>

##### 非量化场景约束

**数据类型要求：**

| x | weight | bias | scale | offset | antiquantScale | antiquantOffset | perTokenScale | groupList | out |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| FLOAT | FLOAT (ND) | FLOAT/null | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | INT64 | FLOAT |
| FLOAT16 | FLOAT16 (ND/NZ) | FLOAT16/null | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | INT64 | FLOAT16 |
| BFLOAT16 | BFLOAT16(ND/NZ) | FLOAT/null | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | INT64 | BFLOAT16 |

**约束**：
- 支持 groupType=-1、0、2，actType=0，groupListType=0/1。

<a id="a8w8场景约束"></a>

##### A8W8 场景约束

**数据类型要求：**

| x | weight | bias | scale | perTokenScale | out |
|:---|:---|:---|:---|:---|:---|
| INT8 | INT8 (ND) | INT32/null | UINT64 | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | INT8 |
| INT8 | INT8 (ND/NZ) | INT32/null | BFLOAT16 | FLOAT/null | BFLOAT16 |
| INT8 | INT8 (NZ) | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | FLOAT/BFLOAT16 | FLOAT/null | BFLOAT16 |
| INT8 | INT8 (ND/NZ) | INT32/null | FLOAT | FLOAT/null | FLOAT16 |
| INT8 | INT8 (ND/NZ) | INT32/null | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | INT32 |

> 其余参数（offset、antiquantScale、antiquantOffset、activationXxx 等）均须传空。

**约束**：
- 仅 groupType=0（M 轴分组）。
- x、weight、out 仅支持长度为 1 的 TensorList。
- x 不支持转置，2 维 `(M, K)`；weight 3 维 `(E, K, N)` 或 `(E, N, K)`。
- **定轴算法**（优化性能）需同时满足：
  - shape（满足其一）：`(M, 7168) × (7168, 4096)` 或 `(M, 2048) × (2048, 7168)`
  - tuningConfigOptional：[0] ∈ (128, 512)，[1]=0，[3] = -1 或 ≥ M×N×4

<a id="a4w4场景约束"></a>

##### A4W4 场景约束

**数据类型要求：**

| x | weight | bias | scale | perTokenScale | out |
|:---|:---|:---|:---|:---|:---|
| INT4 | INT4 (ND/NZ) | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | UINT64 | FLOAT/null | FLOAT16/BFLOAT16 |

**约束**：
- groupType=0，actType=0，groupListType=0/1。
- x/weight/out 仅支持长度为 1 的 TensorList。
- x 不转置，2 维 `(M, K)`。
- weight 3 维 `(E, K, N)`。ND 格式时 N 为 8 的倍数。
- weight NZ 格式支持转置，ND 仅支持非转置。
- perchannel：scale shape `(E, N)`；pergroup：scale shape `(E, G, N)`，G 能整除 K 且 K/G 为偶数。
- 开启右矩阵 NZ 转置后：K/G 对齐 64，K 对齐 64，N 对齐 16。

<a id="a8w4场景约束"></a>

##### A8W4 场景约束

**数据类型要求：**

| x | weight | bias | scale | offset | perTokenScale | out |
|:---|:---|:---|:---|:---|:---|:---|
| INT8 | INT4 (ND/NZ) | FLOAT | UINT64 | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | FLOAT | BFLOAT16 |
| INT8 | INT4 (ND/NZ) | FLOAT | UINT64 | FLOAT/null | FLOAT | FLOAT16 |

**约束**：
- groupType=0，actType=0。
- x/weight/out 仅支持长度为 1 的 TensorList。
- x/weight 均不转置。x 2 维 `(M, K)`；weight 3 维 `(E, K, N)`。
- bias 为离线辅助结果：$bias_i = 8 \times weight_i \times scale_i$ 沿 K 轴规约，shape `(E, N)`。
- weight=INT32 时，每个 INT32 视为 8 个 INT4。

**offset 为空（对称量化）：**
- 按 count 模式（不检查 groupListType），K 为 quantGroupSize=256 的整数倍，K ≤ 18432。
- N 为 8 的整数倍。
- scale shape `(E, K/256, N)`。
- tuningConfigOptional[0] > N/4 时性能更优（显存额外增加 $g \times K \times N$ 字节）。

**offset 不为空（非对称量化）：**
- scale shape `(E, 1, N)`。
- offset 为离线辅助结果（$antiquantOffset \times scale$），shape `(E, 1, N)`，dtype=FLOAT32。
- tuningConfigOptional[1] 可置 1 使能 weight 亲核格式（需 weight shape `(E, N, K)` 经 ND2NZ 转换），K ≥ 2048 && N ≥ 2048 时有性能优势。

<a id="a16w8场景约束"></a>

##### A16W8 场景约束

**数据类型要求：**

| x | weight | bias | antiquantScale | antiquantOffset | out |
|:---|:---|:---|:---|:---|:---|
| FLOAT16 | INT8 (ND) | FLOAT16/null | FLOAT16 | FLOAT16 | FLOAT16 |
| BFLOAT16 | INT8 (ND) | FLOAT/null | BFLOAT16 | BFLOAT16 | BFLOAT16 |

**约束**：
- x 不转置；groupType=-1/0；actType=0；groupListType=0/1。
- 仅 perchannel 模式。
- weight 多 Tensor 时，pergroup 长度 $s_i = K_i / G_i$ 须各组相等。
- antiquantScale/Offset shape：

| 子场景 | shape |
|:---|:---|
| weight 单，perchannel | `(E, N)` |
| weight 多，perchannel | `(N_i)` |

<a id="a16w4场景约束"></a>

##### A16W4 场景约束

**数据类型要求：**

| x | weight | bias | antiquantScale | antiquantOffset | out |
|:---|:---|:---|:---|:---|:---|
| FLOAT16 | INT4 (ND) | FLOAT16/null | FLOAT16 | FLOAT16 | FLOAT16 |
| BFLOAT16 | INT4 (ND) | FLOAT/null | BFLOAT16 | BFLOAT16 | BFLOAT16 |

**约束**：
- x 不转置；groupType=-1/0；actType=0；groupListType=0/1。
- weight 最后一维为偶数；pergroup 场景下 weight 转置时 pergroup 长度为偶数。
- pergroup 场景：G 须能整除对应 $K_i$。多 Tensor 时所有 $s_i$ 相等。
- 非对称量化仅 perchannel。
- antiquantScale/Offset shape：

| 子场景 | shape |
|:---|:---|
| weight 单，perchannel | `(E, N)` |
| weight 单，pergroup | `(E, G, N)` |
| weight 多，perchannel | `(N_i)` |
| weight 多，pergroup | `(G_i, N_i)` |

#### 7.3.2 groupType 支持场景

> x、weight、y 输入为 aclTensorList。"单" = 单元素 TensorList，"多" = 多元素 TensorList。

| groupType | x | weight | y | splitItem | groupListOptional | 转置 | 其余场景限制 |
|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|
| -1 | 多 | 多 | 多 | 0/1 | 必须传空 | x 不转置；weight 可转置（统一） | x 2维（维度一致），weight 2维，y 同 x |
| 0 | 单 | 单 | 单 | 2/3 | 必须传；groupListType=0 末值 ≤ x 第一维；groupListType=1 总和 ≤ x 第一维；groupListType=2 第二列总和 ≤ x 第一维；最大 1024 组 | x 不转置；weight 可转置（A8W4/A4W4 不可） | weight 3维，x/y 2维 |
| 0 | 单 | 多 | 单 | 2/3 | 必须传；groupListType=0 末值 ≤ x 第一维；groupListType=1 总和 ≤ x 第一维；groupListType=2 第二列总和 ≤ x 第一维；最大 128 组 | x 不转置；weight 可转置（统一） | x/weight/y 2维；weight N 轴相等 |
| 0 | 多 | 多 | 单 | 2/3 | 可选；若传则 groupListType=0 差值 = x 第一维、groupListType=1 数值 = x 第一维、groupListType=2 第二列 = x 第一维；最大 128 组 | x 不转置；weight 可转置（统一） | x/weight/y 2维；weight N 轴相等 |
| 2 | 单 | 单 | 单 | 2/3 | 必须传；groupListType=0 末值 ≤ x 第二维；groupListType=1 总和 ≤ x 第二维；groupListType=2 第二列总和 ≤ x 第二维；最大 1024 组 | x 必须转置；weight 不转置 | x/weight 2维，y 3维；bias 传空 |
| 2 | 单 | 多 | 多 | 0/1 | 必须传空 | x 必须转置；weight 不转置 | x/weight/y 2维；weight 最大 128 组；原始 shape 各 weight 第一维之和 ≤ x 第一维；bias 传空 |

#### 7.3.3 tuningConfigOptional 配置示例

tuningConfigOptional 为 Host 侧 aclIntArray，存储 INT64 元素。兼容历史版本，不使用时不传即可。

| 索引 | 语义 | 适用场景 |
|:---:|------|------|
| [0] | 预期各专家处理的 token 数，tiling 按此优化 | A8W4、A8W8、A4W4（x/weight/out 单 Tensor） |
| [1] | 是否使能 weight 亲核格式（需先转置再 NZ） | A8W4 |
| [2] | 允许额外 workspace 上限（-1=不限），用于定轴算法优化 | A8W8 |

</details>

---

## 8. 接口调用

### 8.1 调用流程

1. 根据场景准备输入（x/weight 及各场景所需的量化/伪量化参数）。
2. 调用 `aclnnGroupedMatmulV5GetWorkspaceSize` 获取 workspace 大小和 executor。
3. 根据返回的 workspaceSize 申请 Device 侧内存。
4. 调用 `aclnnGroupedMatmulV5` 执行计算。
5. 同步等待 Stream 任务完成。
6. 将输出从 Device 拷贝到 Host。

### 8.2 示例代码索引

以下表格提供各场景下的示例代码文件链接，请参考文件中的注释了解对应场景说明。编译与运行请参考 [编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

| 场景 | 设备 | 示例文件 | 说明 |
|------|:---:|------|------|
| 非量化 | A5 | [examples/arch35/test_aclnn_grouped_matmul_v5_noquant.cpp](../examples/arch35/test_aclnn_grouped_matmul_v5_noquant.cpp) | A5 非量化示例 |
| 伪量化（weight NZ） | A5 | [examples/arch35/test_aclnn_grouped_matmul_weight_nz.cpp](../examples/arch35/test_aclnn_grouped_matmul_weight_nz.cpp) | A5 伪量化 NZ 格式示例 |
| 伪量化（V5 NZ） | A5 | [examples/arch35/test_aclnn_grouped_matmul_v5_weight_nz.cpp](../examples/arch35/test_aclnn_grouped_matmul_v5_weight_nz.cpp) | A5 V5 伪量化 NZ 示例 |
| MX 量化 | A5 | [examples/arch35/test_aclnn_grouped_matmul_v4_mx_quant.cpp](../examples/arch35/test_aclnn_grouped_matmul_v4_mx_quant.cpp) | A5 MX 量化示例 |
| 全量化（静态） | A5 | [examples/arch35/test_aclnn_grouped_matmul_v5_quant_static.cpp](../examples/arch35/test_aclnn_grouped_matmul_v5_quant_static.cpp) | x=INT8, weight=INT8, scale=FLOAT32, perTokenScale=nullptr |
| 全量化（动态 K-C） | A5 | [examples/arch35/test_aclnn_grouped_matmul_v5_quant_dynamic.cpp](../examples/arch35/test_aclnn_grouped_matmul_v5_quant_dynamic.cpp) | x=INT8, weight=INT8, scale=FLOAT32, perTokenScale=FLOAT |
| G-B 量化 | A5 | [examples/arch35/test_aclnn_grouped_matmul_v5_quant_gb.cpp](../examples/arch35/test_aclnn_grouped_matmul_v5_quant_gb.cpp) | x/w=FLOAT8_E5M2, scale=FLOAT32 (3维), perTokenScale=FLOAT32 (2维) |
| 非量化 | A2/A3 | [test_aclnn_grouped_matmul.cpp](../examples/test_aclnn_grouped_matmul.cpp) | 非量化基础示例（涵盖通用调用流程） |
| 全量化-A8W8 | A2/A3 | [test_aclnn_grouped_matmul_v5_a8w8.cpp](../examples/test_aclnn_grouped_matmul_v5_a8w8.cpp) | x=INT8, weight=INT8, scale=UINT64, out=BF16 |
| 全量化-A4W4 | A2/A3 | [test_aclnn_grouped_matmul_v5_a4w4.cpp](../examples/test_aclnn_grouped_matmul_v5_a4w4.cpp) | x=INT4, weight=INT4, scale=UINT64, out=FP16 |
| 伪量化-A8W4 | A2/A3 | [test_aclnn_grouped_matmul_v5_a8w4.cpp](../examples/test_aclnn_grouped_matmul_v5_a8w4.cpp) | x=INT8, weight=INT4, scale=UINT64, perTokenScale=FLOAT |
| 伪量化-A16W8 | A2/A3 | [test_aclnn_grouped_matmul_v5_a16w8.cpp](../examples/test_aclnn_grouped_matmul_v5_a16w8.cpp) | x=BF16, weight=INT8, antiquantScale=BF16 |
| 伪量化-A16W4 | A2/A3 | [test_aclnn_grouped_matmul_v5_a16w4.cpp](../examples/test_aclnn_grouped_matmul_v5_a16w4.cpp) | x=BF16, weight=INT4, antiquantScale=BF16 |
