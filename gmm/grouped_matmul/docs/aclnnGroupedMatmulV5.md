# aclnnGroupedMatmulV5

<style>
abbr {
  border-bottom: 1px dotted #666;
  cursor: help;
}
</style>

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/gmm/grouped_matmul)

## 目录

- [1. 产品支持情况](#1-产品支持情况)
- [2. 专业术语](#2-专业术语)
- [3. 功能说明](#3-功能说明)
- [4. 函数原型](#4-函数原型)
- [5. aclnnGroupedMatmulV5GetWorkspaceSize](#5-aclnngroupedmatmulv5getworkspacesize)
- [6. aclnnGroupedMatmulV5](#6-aclnngroupedmatmulv5)
- [7. 约束说明](#7-约束说明)
  - [7.1 公共约束](#71-公共约束)
  - [7.2 Ascend 950PR/Ascend 950DT](#72-ascend-950prascend-950dt)
    - [7.2.1 场景速查表](#721-场景速查表)
  - [7.3 Atlas A3/A2 训练系列产品](#73-atlas-a3a2-训练系列产品)
    - [7.3.1 场景速查表](#731-场景速查表)
- [8. 接口调用](#8-接口调用)
  - [8.1 调用流程](#81-调用流程)
  - [8.2 配置示例](#82-配置示例)
    - [8.2.1 groupListOptional 配置示例](#821-grouplistoptional-配置示例)
    - [8.2.2 tuningConfigOptional 配置示例](#822-tuningconfigoptional-配置示例)
  - [8.3 示例代码索引](#83-示例代码索引)

---

<a id="1-产品支持情况"></a>

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

<a id="2-专业术语"></a>

## 2. 专业术语

| 术语 | 简要说明 |
|------|----------|
| 非量化 | x 和 weight 均为浮点类型直接计算 |
| 伪量化 | weight 低精度存储，通过 antiquantScale/Offset 反量化后计算 |
| 全量化 | x 和 weight 均为低精度计算，通过 scale 还原 |
| A8W4 | A8 表示 Activation 采用 8bit 量化，W4 表示 weight 采用 4bit 量化，同理还有 A8W8、A4W4 |
| A16W8 | A16 表示 Activation 采用 16bit 非量化，W8 表示 weight 采用 8bit 量化，同理还有 A16W4 |
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

---

<a id="3-功能说明"></a>

## 3. 功能说明

接口功能：实现分组矩阵乘计算。公式为 $y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$，其中 $g$ 为分组数。

- **M 轴分组（<abbr title="指定分组轴：-1=不分组，0=M轴，2=K轴">groupType</abbr>=0）**：$k_i$、$n_i$ 各组相同，$m_i$ 可以不相同。
- **K 轴分组（<abbr title="指定分组轴：-1=不分组，0=M轴，2=K轴">groupType</abbr>=2）**：$m_i$、$n_i$ 各组相同，$k_i$ 可以不相同。
- **不分组（<abbr title="指定分组轴：-1=不分组，0=M轴，2=K轴">groupType</abbr>=-1）**：各组 shape 完全独立。

基础计算公式：

$$
y_i=x_i\times weight_i + bias_i
$$

详细计算公式（按场景分类：<abbr title="x 和 weight 均为浮点类型直接计算">非量化</abbr>、<abbr title="weight 低精度存储，通过 antiquantScale/Offset 反量化后计算">伪量化</abbr>、<abbr title="x 和 weight 均为低精度计算，通过 scale 还原">全量化</abbr>）及版本演进请参见 [GroupedMatmul README](https://gitcode.com/cann/ops-transformer/blob/8cde1c9ef2d33811a441a3dc62152841506b0997/gmm/grouped_matmul/README.md#%E5%8A%9F%E8%83%BD%E8%AF%B4%E6%98%8E)。

---
<a id="4-函数原型"></a>

## 4. 函数原型

每个算子分为两段式接口，必须先调用 `aclnnGroupedMatmulV5GetWorkspaceSize` 接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用 `aclnnGroupedMatmulV5` 接口执行计算。

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

> 标注规则：
> - <span title="Ascend 950PR/950DT 不支持"><sup>1</sup></span>：仅 Ascend 950PR/950DT 不支持
> - <span title="Atlas A3/A2 不支持"><sup>2</sup></span>：仅 Atlas A3/A2 不支持

<a id="5-aclnngroupedmatmulv5getworkspacesize"></a>

## 5. aclnnGroupedMatmulV5GetWorkspaceSize

- **参数说明：**

  以下参数为 `aclnnGroupedMatmulV5GetWorkspaceSize` 的入参和出参。

  | 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
  |--------|:---:|------|------|----------|:---:|:---|:---:|
  | x | 输入 | 公式中的输入 $x$ | TensorList 长度 [1,128] 或 [1,1024]（A5 非量化） | FLOAT、FLOAT16、INT16<span title="仅 Ascend 950PR/950DT 不支持"><sup>1</sup></span>、INT8、INT4<span title="仅 Ascend 950PR/950DT 不支持"><sup>1</sup></span>、BFLOAT16、FLOAT8_E5M2<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span>、FLOAT8_E4M3FN<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span>、HIFLOAT8<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span>、FLOAT4_E2M1<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span> | <abbr title="常规连续排布">ND</abbr> | 2~6 | √ |
  | weight | 输入 | 公式中的 $weight$ | TensorList 长度 [1,128] 或 [1,1024] | FLOAT、FLOAT16、INT16<span title="仅 Ascend 950PR/950DT 不支持"><sup>1</sup></span>、INT8、INT4、BFLOAT16、FLOAT8_E5M2<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span>、FLOAT8_E4M3FN<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span>、HIFLOAT8<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span>、FLOAT4_E2M1<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span> | ND/<abbr title="FRACTAL_NZ 格式（亲核排布）">NZ</abbr> | 2~3 | √ |
  | biasOptional | 可选输入 | 公式中的 $bias$ | 长度与 weight 相同 | FLOAT、FLOAT16、INT32、BFLOAT16<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span> | ND | 1~2 | √ |
  | scaleOptional | 可选输入 | 公式中的 $scale$，代表量化参数中的缩放因子 | 一般情况下，长度与 weight 相同。综合约束请参见 [约束说明](#7-约束说明) | FLOAT、UINT64、BFLOAT16、FLOAT8_E8M0<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span>、INT64<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span> | ND | 1~4 | √ |
  | offsetOptional | 可选输入 | 公式中的 $offset$，代表量化参数中的偏移量 | 长度与 weight 相同 | FLOAT | ND | 3 | √ |
  | antiquantScaleOptional | 可选输入 | 公式中的 $antiquant\_scale$，代表伪量化参数中的缩放因子 | 长度与 weight 相同。综合约束请参见 [约束说明](#7-约束说明) | FLOAT16、BFLOAT16 | ND | 1~3 | √ |
  | antiquantOffsetOptional | 可选输入 | 公式中的 $antiquant\_offset$，代表伪量化参数中的偏移量 | 长度与 weight 相同。综合约束请参见 [约束说明](#7-约束说明) | FLOAT16、BFLOAT16 | ND | 1~3 | √ |
  | perTokenScaleOptional | 可选输入 | 公式中的 $per\_token\_scale$，代表量化参数中由 x 量化引入的缩放因子 | 一般情况下，只支持 1 维且长度与 x 的 M 相同。综合约束请参见 [约束说明](#7-约束说明) | FLOAT、FLOAT8_E8M0<span title="仅 Atlas A3/A2 不支持"><sup>2</sup></span> | ND | 1~3 | √ |
  | groupListOptional | 可选输入 | 代表输入和输出分组轴方向的 matmul 大小分布 | 根据 groupListType 输入不同格式数据 | INT64 | ND | 1~2 | √ |
  | activationInputOptional | 可选输入 | 代表激活函数的反向输入，当前只支持传入 nullptr | - | - | - | - | - |
  | activationQuantScaleOptional | 可选输入 | 激活函数量化缩放因子（预留参数），当前只支持传入 nullptr | - | - | - | - | - |
  | activationQuantOffsetOptional | 可选输入 | 激活函数量化偏移量（预留参数），当前只支持传入 nullptr | - | - | - | - | - |
  | splitItem | 输入 | 代表输出是否要做 Tensor 切分 | `0/1` = 多 Tensor 输出；`2/3` = 单 Tensor 输出。接口不区分 0/1 和 2/3 的内部差异 | - | - | - | - |
  | groupType | 输入 | 代表需要分组的轴 | 如矩阵乘为 $C[m,n]=A[m,k]\times B[k,n]$：`-1` = 不分组；`0` = M 轴分组；`2` = K 轴分组 | - | - | - | - |
  | groupListType | 输入 | 代表 groupList 输入的分组方式 | `0` = 累积和（cumsum）；`1` = 各组大小；`2` = [组索引, 组大小] 对。详见 [约束说明](#7-约束说明) | - | - | - | - |
  | actType | 输入 | 代表激活函数类型 | `0`=GMM_ACT_TYPE_NONE / `1`=GMM_ACT_TYPE_RELU / `2`=GMM_ACT_TYPE_GELU_TANH / `3`=GMM_ACT_TYPE_GELU_ERR_FUNC（不支持） / `4`=GMM_ACT_TYPE_FAST_GELU / `5`=GMM_ACT_TYPE_SILU | - | - | - | - |
  | tuningConfigOptional | 可选输入 | 第一个数代表各个专家处理的 token 数的预期值；第二个数代表 <abbr title="A8 表示 Activation 采用 8bit 量化，W4 表示 weight 采用 4bit 量化，同理还有 A8W8、A4W4">A8W4</abbr> 可选开启 weight 先转置的 NZ 格式；第三个数代表允许额外使用的内存空间。详见 [约束说明](#7-约束说明) | 兼容历史版本，用户如不使用该参数，不传（即为 nullptr）即可 | INT64 | - | 3 | - |
  | out | 输出 | 公式中的输出 $y$ | TensorList 长度 [1,128] 或 [1,1024] | FLOAT、FLOAT16、INT32、INT8、BFLOAT16 | ND | 2~3 | - |
  | activationFeatureOutOptional | 输出 | 激活函数的输入数据，当前只支持传入 nullptr | - | - | - | - | - |
  | dynQuantScaleOutOptional | 输出 | 动态量化缩放因子，当前只支持传入 nullptr | - | - | - | - | - |
  | workspaceSize | 输出 | 返回需要在 Device 侧申请的 workspace 大小（字节） | - | - | - | - | - |
  | executor | 输出 | 返回 op 执行器，包含了算子计算流程 | - | - | - | - | - |

  - **<term>Ascend 950PR/Ascend 950DT</term>**：
    - 上表数据类型列中的角标 <span title="Ascend 950PR/950DT 不支持"><sup>1</sup></span> 代表该系列不支持的数据类型
    - 输入参数 x、weight 均不支持 INT16 类型，且 x 不支持 INT4 类型
    - 输入参数 x、weight，输出参数 out 在非量化场景支持最多 1024 个 Tensor，在伪量化支持最多 128 个 Tensor，在全量化场景最多支持 1 个 Tensor
  - **<term>Atlas A3/A2 训练系列产品</term>**：
    - 上表数据类型列中的角标 <span title="Atlas A3/A2 不支持"><sup>2</sup></span> 代表该系列不支持的数据类型
    - 不支持 FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8、FLOAT8_E8M0 类型
    - 输入参数 biasOptional 不支持 BFLOAT16
    - 输入参数 scaleOptional 不支持 INT64 类型
    - 输入参数 x、weight，输出参数 out 支持最多 128 个 Tensor

- **返回值：**

  **返回类型**：`aclnnStatus`，具体参见 [aclnn 返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一阶段接口完成入参校验，出现以下场景时报错。

  <table>
    <thead>
      <tr>
        <th style="width: 250px">返回值</th>
        <th style="width: 130px">错误码</th>
        <th style="width: 850px">描述</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td rowspan="4"> ACLNN_ERR_PARAM_NULLPTR </td>
        <td rowspan="4"> 161001 </td>
        <td>传入参数是必选输入、输出或者必选属性，且是空指针。</td>
      </tr>
      <tr>
        <td>传入参数 weight 的元素存在空指针。</td>
      </tr>
      <tr>
        <td>传入参数 x 的元素为空指针，且传出参数 out 的元素不为空指针。</td>
      </tr>
      <tr>
        <td>传入参数 x 的元素不为空指针，且传出参数 out 的元素为空指针。</td>
      </tr>
      <tr>
        <td rowspan="7"> ACLNN_ERR_PARAM_INVALID </td>
        <td rowspan="7"> 161002 </td>
        <td>x、weight、biasOptional、scaleOptional、offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、groupListOptional、out 的数据类型和数据格式不在支持的范围内。</td>
      </tr>
      <tr>
        <td>weight 的长度不在支持范围。</td>
      </tr>
      <tr>
        <td>若 bias 不为空，bias 的长度不等于 weight 的长度。</td>
      </tr>
      <tr>
        <td>groupListOptional 维度不符合要求。</td>
      </tr>
      <tr>
        <td>splitItem 为 2、3 的场景，out 长度不等于 1。</td>
      </tr>
      <tr>
        <td>splitItem 为 0、1 的场景，out 长度不等于 weight 的长度，groupListOptional 长度不等于 weight 的长度。</td>
      </tr>
      <tr>
        <td>传入参数 tuningConfigOptional 的元素为负数，或者大于 x 的行数 m。</td>
      </tr>
    </tbody>
  </table>

<a id="6-aclnngroupedmatmulv5"></a>

## 6. aclnnGroupedMatmulV5

- **参数说明：**

  以下参数为 `aclnnGroupedMatmulV5` 的入参：

  | 参数名 | 输入/输出 | 描述 |
  |--------|:---:|------|
  | workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
  | workspaceSize | 输入 | 在 Device 侧申请的 workspace 大小，由第一段接口 `aclnnGroupedMatmulV5GetWorkspaceSize` 获取 |
  | executor | 输入 | op 执行器，包含了算子计算流程 |
  | stream | 输入 | 指定执行任务的 Stream |

- **返回值：**

  **返回类型**：`aclnnStatus`，具体参见 [aclnn 返回码](../../../docs/zh/context/aclnn返回码.md)。

---


<a id="7-约束说明"></a>

## 7. 约束说明

**确定性计算**：aclnnGroupedMatmulV5 默认确定性实现。

> "—" 表示该参数在当前场景下必须传空（nullptr）

<a id="71-公共约束"></a>

### 7.1 公共约束

每一个场景约束中，只关注该场景需要用到的参数，其余参数均视为该场景不涉及，可以填 nullptr。

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

- groupType：支持 M 轴分组（0）和不分组（-1）。非量化/全量化额外支持 K 轴分组（2）。
- groupListType：支持 0、1、2。仅全量化 groupType=0 时支持 groupListType=2。
- tuningConfigOptional：不支持。
- actType（0~5）：
  - 非量化/伪量化仅支持 0。
  - 全量化下 x/weight=INT8、静态 <abbr title="pertensor-perchannel：输入侧统一缩放，权重侧每列独立缩放">T-C</abbr> 或动态 <abbr title="pertoken-perchannel：输入侧每行独立缩放，权重侧每列独立缩放">K-C</abbr>、scale=FLOAT32/BFLOAT16 时支持 0/1/2/4/5；其余仅支持 0。

<a id="721-场景速查表"></a>

#### 7.2.1 场景速查表

| 场景名 | x | weight | out | 约束说明 |
|--------|:---:|:---:|:---|------|
| 非量化 | FLOAT32/FLOAT16/BFLOAT16 | FLOAT32/FLOAT16/BFLOAT16 | FLOAT32/FLOAT16/BFLOAT16 | [非量化场景约束](#ascend950-非量化场景约束) |
| 静态量化 | INT8 | INT8 | BFLOAT16/FLOAT16/INT32/INT8 | [静态量化场景约束](#ascend950-静态量化场景约束) |
| 静态量化 | HIFLOAT8 / FLOAT8_E5M2 / FLOAT8_E4M3FN | HIFLOAT8 / FLOAT8_E5M2 / FLOAT8_E4M3FN | BFLOAT16/FLOAT16/FLOAT32 | [静态量化场景约束](#ascend950-静态量化场景约束) |
| 动态量化（<abbr title="pertensor-pertensor：输入和权重均为统一缩放">T-T</abbr>/T-C/<abbr title="pertoken-pertensor：输入侧每行独立缩放，权重侧统一缩放">K-T</abbr>/K-C） | INT8 | INT8 | BFLOAT16/FLOAT16 | [动态量化（T-T/T-C/K-T/K-C）场景约束](#ascend950-动态量化-ttck) |
| 动态量化（T-T/T-C/K-T/K-C） | HIFLOAT8 / FLOAT8_E5M2 / FLOAT8_E4M3FN | HIFLOAT8 / FLOAT8_E5M2 / FLOAT8_E4M3FN | BFLOAT16/FLOAT16/FLOAT32 | [动态量化（T-T/T-C/K-T/K-C）场景约束](#ascend950-动态量化-ttck) |
| 动态量化（<abbr title="特殊 pergroup 量化（block 粒度 M=1,N=1,K=32）">MX</abbr>） | FLOAT8_E5M2/FLOAT8_E4M3FN / FLOAT4_E2M1 | FLOAT8_E5M2/FLOAT8_E4M3FN / FLOAT4_E2M1 | BFLOAT16/FLOAT16/FLOAT32 | [动态量化（MX）场景约束](#ascend950-动态量化-mx) |
| 动态量化（<abbr title="pergroup-perblock：输入侧按 K 分组缩放，权重侧按 N×K 分块缩放">G-B</abbr>） | HIFLOAT8 / FLOAT8_E5M2 / FLOAT8_E4M3FN | HIFLOAT8 / FLOAT8_E5M2 / FLOAT8_E4M3FN | BFLOAT16/FLOAT16/FLOAT32 | [动态量化（G-B）场景约束](#ascend950-动态量化-gb) |
| 伪量化 | FLOAT16 / BFLOAT16 | INT8/INT4 | FLOAT16 / BFLOAT16 | [伪量化场景约束](#ascend950-伪量化场景约束) |
| 伪量化 | FLOAT16 / BFLOAT16 | FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8 | FLOAT16 / BFLOAT16 | [伪量化场景约束](#ascend950-伪量化场景约束) |

<a id="ascend950-非量化场景约束"></a>

<details>
<summary>非量化场景约束</summary>

| groupType | x | weight | biasOptional | out |
|:---:|:---:|:---:|:---|:---|
| -1/0/2 | BFLOAT16 | BFLOAT16 | BFLOAT16/FLOAT32/null | BFLOAT16 |
| -1/0/2 | FLOAT16 | FLOAT16 | FLOAT16/FLOAT32/null | FLOAT16 |
| -1/0/2 | FLOAT32 | FLOAT32 | FLOAT32/null | FLOAT32 |

</details>

<a id="ascend950-静态量化场景约束"></a>

<details>
<summary>静态量化场景约束</summary>

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
| 0/2 | <abbr title="每列独立缩放因子">perchannel</abbr> | `(g, N)` |
| 0/2 | <abbr title="一个 Tensor 统一缩放因子">pertensor</abbr> | `(g, 1)` 或 `(g,)`，输出为 INT8 时不支持 pertensor 场景 |

</details>

<a id="ascend950-动态量化-ttck"></a>

<details>
<summary>动态量化（T-T/T-C/K-T/K-C）场景约束</summary>

| groupType | x | weight | biasOptional | scaleOptional | perTokenScaleOptional | out |
|:---:|:---:|:---:|:---|:---|:---|:---|
| 0 | INT8 | INT8 | INT32/BFLOAT16/FLOAT32/null | BFLOAT16/FLOAT32 | FLOAT32 | BFLOAT16 |
| 0 | INT8 | INT8 | INT32/FLOAT16/FLOAT32/null | FLOAT32 | FLOAT32 | FLOAT16 |
| 0/2 | HIFLOAT8 | HIFLOAT8 | null | FLOAT32 | FLOAT32 | BFLOAT16/FLOAT16/FLOAT32 |
| 0/2 | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E5M2/FLOAT8_E4M3FN | null | FLOAT32 | FLOAT32 | BFLOAT16/FLOAT16/FLOAT32 |

- **scaleOptional shape**：同上表，推荐在 pertensor 场景 shape 使用 `(g,)`，防止与 G-B 量化模式混淆。
- **perTokenScaleOptional shape**：

| groupType | 子场景 | shape |
|:---:|:---|:---|
| 0 | <abbr title="每行独立缩放因子">pertoken</abbr> | `(M,)` |
| 0 | pertensor | `(g, 1)` 或 `(g,)` |
| 2 | pertoken | `(g, M)` |
| 2 | pertensor | `(g, 1)` 或 `(g,)` |

</details>

<a id="ascend950-动态量化-mx"></a>

<details>
<summary>动态量化（MX）场景约束</summary>

| groupType | x | weight | biasOptional | scaleOptional | perTokenScaleOptional | out |
|:---:|:---:|:---:|:---:|:---:|:---|:---|
| 0/2 | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E5M2/FLOAT8_E4M3FN | null | FLOAT8_E8M0 | FLOAT8_E8M0 | BFLOAT16/FLOAT16/FLOAT32 |
| 0 | FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT32/null | FLOAT8_E8M0 | FLOAT8_E8M0 | BFLOAT16/FLOAT16/FLOAT32 |

- **scaleOptional shape**：

| groupType | shape |
|:---:|:---|
| 0 | `(g, N, ceil(K/64), 2)`（weight 转置）/ `(g, ceil(K/64), N, 2)`（weight 不转置） |
| 2 | `(K/64 + g, N, 2)`，scale_i 起始地址偏移为 `((K_0 + K_1 + ... + K_{i-1}) / 64 + g_i) * N * 2` |

- **perTokenScaleOptional shape**：

| groupType | shape |
|:---:|:---|
| 0 | `(M, ceil(K/64), 2)` |
| 2 | `(K/64 + g, M, 2)` |

- x 为 FLOAT4_E2M1 时，K 须为偶数且 K≠2；weight 非转置时 N 须为偶数。

</details>

<a id="ascend950-动态量化-gb"></a>

<details>
<summary>动态量化（G-B）场景约束</summary>

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
| 2 | `(K/gsK + g, M)`，per_token_scale_i 地址偏移为 `((K_0 + K_1 + ... + K_{i-1}) / gsK + g_i) * M` |

- **动态量化特殊场景处理**：
  - 在动态量化场景 M 分组或 K 分组情况下，当 N 等于 1 且 scaleOptional 的 shape 为 `(g, 1)` 时，weight 既可以 pertensor 量化也可以 perchannel 量化时，优先选择 pertensor 量化模式。
  - 在动态量化场景 M 分组情况下，当 g = M 且 perTokenScaleOptional 的 shape 为 `(g,)` 时，x 选择 pertoken 量化模式；当 g = M，K ≤ 128 且 perTokenScaleOptional 的 shape 为 `(g, 1)` 时，根据 weight 的量化模式选择 x 的量化模式（weight 如果是 perchannel 或者 pertensor 量化，x 选择 pertensor 量化；weight 如果是 <abbr title="按 K/N 方向固定大小分块">perblock</abbr> 量化，x 选择 <abbr title="按 K 方向等分多组">pergroup</abbr> 量化）。
  - 在动态量化场景 K 分组情况下，K 小于 128，N 小于等于 128 且 scaleOptional 的 shape 为 `(g, 1)` 时，按照现有量化模式区分规则，既可以为非 pergroup 量化，又可以为 G-B 量化，此种场景现一律按照 G-B 量化处理。
  - 在动态量化场景 K 分组情况下，当 M 等于 1 且 perTokenScaleOptional 的 shape 为 `(g, 1)` 时，x 既可以 pertoken 量化也可以 pertensor 量化时，优先选择 pertensor 量化模式。
  - 在动态量化场景 K 分组情况下，K 小于 128，M 等于 1 且 perTokenScaleOptional 的 shape 为 `(g, 1)` 时，如果 N 小于等于 128，x 则选择 pergroup 量化；如果 N 大于 128，根据 weight 的量化模式选择 x 的量化模式（weight 如果是 perchannel 或者 pertensor 量化，x 选择 pertensor 量化；weight 如果是 perblock 量化，x 选择 pergroup 量化）。
  - 在动态量化场景 K 分组情况下，K 小于 128，M 不等于 1 时，如果 N 小于等于 128，x 则选择 pergroup 量化；如果 N 大于 128，根据 weight 的量化模式选择 x 的量化模式（weight 如果是 perchannel 或者 pertensor 量化，x 选择 pertoken 量化；weight 如果是 perblock 量化，x 选择 pergroup 量化）。

</details>

<a id="ascend950-伪量化场景约束"></a>

<details>
<summary>伪量化场景约束</summary>

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
| 0 | 单 | 单 | 单 | 2/3 | 必须传 | x 不转置；weight 可转置 | weight 3维 `(g,N,K)` 或 `(g,K,N)`；x/out 2维；bias 2维 `(g,N)`；最多 1024 组；仅 ND 出；INT8 场景 weight 支持 NZ 格式 |
| 0 | 单 | 多 | 单 | 2/3 | 必须传（长度 ≤1024） | x 不转置；weight 可转置（统一） | x/weight/y 2维；weight N 轴相等；仅 ND 进 ND 出；仅非量化 |
| 0 | 多 | 多 | 单 | 2 | 可选 | x 不转置；weight 可转置（统一） | x/weight/y 2维；weight N 轴相等；仅 ND 进 ND 出；仅非量化 |
| 2 | 单 | 单 | 单 | 2/3 | 必须传 | x 必须转置；weight 不转置 | x/weight 2维 `(K,M)`/`(K,N)`；y 3维 `(g,M,N)`；bias 传空；仅 ND 进 ND 出；仅非量化/量化 |
| 2 | 单 | 多 | 多 | 0/1 | 可选 | x 必须转置；weight 不转置 | x/weight/y 2维 `(M,N)`；weight 最大 128 组；不支持 bias；仅 ND 进 ND 出；仅非量化 |

</details>

<a id="73-atlas-a3a2-训练系列产品"></a>

### 7.3 Atlas A3/A2 训练系列产品

<a id="731-场景速查表"></a>

#### 7.3.1 场景速查表

| 场景名 | x | weight | out | 约束说明 |
|--------|:---:|:---:|:---:|------|
| 非量化 | FLOAT32 | FLOAT32 | FLOAT32 | [非量化场景约束](#非量化场景约束) |
| 非量化 | BFLOAT16 | BFLOAT16 | BFLOAT16 | [非量化场景约束](#非量化场景约束) |
| 非量化 | FLOAT16 | FLOAT16 | FLOAT16 | [非量化场景约束](#非量化场景约束) |
| 全量化-<abbr title="A8 表示 Activation 采用 8bit 量化，W4 表示 weight 采用 4bit 量化，同理还有 A8W8、A4W4">A8W8</abbr> | INT8 | INT8 | BFLOAT16/FLOAT16/INT32/INT8 | [A8W8场景约束](#a8w8场景约束) |
| 全量化-<abbr title="A8 表示 Activation 采用 8bit 量化，W4 表示 weight 采用 4bit 量化，同理还有 A8W8、A4W4">A4W4</abbr> | INT4 | INT4 | BFLOAT16/FLOAT16 | [A4W4场景约束](#a4w4场景约束) |
| 伪量化-A8W4 | INT8 | INT4 | BFLOAT16/FLOAT16 | [A8W4场景约束](#a8w4场景约束) |
| 伪量化-<abbr title="A16 表示 Activation 采用 16bit 非量化，W8 表示 weight 采用 8bit 量化，同理还有 A16W4">A16W8</abbr> | BFLOAT16/FLOAT16 | INT8 | BFLOAT16/FLOAT16 | [A16W8场景约束](#a16w8场景约束) |
| 伪量化-<abbr title="A16 表示 Activation 采用 16bit 非量化，W8 表示 weight 采用 8bit 量化，同理还有 A16W4">A16W4</abbr> | BFLOAT16/FLOAT16 | INT4 | BFLOAT16/FLOAT16 | [A16W4场景约束](#a16w4场景约束) |

<a id="非量化场景约束"></a>

<details>
<summary>非量化场景约束</summary>

**数据类型要求：**

| x | weight | bias | scale | offset | antiquantScale | antiquantOffset | perTokenScale | groupList | out |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| FLOAT | FLOAT (ND) | FLOAT/null | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | INT64 | FLOAT |
| FLOAT16 | FLOAT16 (ND/NZ) | FLOAT16/null | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | INT64 | FLOAT16 |
| BFLOAT16 | BFLOAT16(ND/NZ) | FLOAT/null | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | INT64 | BFLOAT16 |

**约束**：
- 支持 groupType=-1、0、2，actType=0，groupListType=0/1。

</details>

<a id="a8w8场景约束"></a>

<details>
<summary>A8W8 场景约束</summary>

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
  - tuningConfigOptional：[0] ∈ (128, 512)，[1]=0，[2] = -1 或 ≥ M×N×4

</details>

<a id="a4w4场景约束"></a>

<details>
<summary>A4W4 场景约束</summary>

**数据类型要求：**

| x | weight | bias | scale | perTokenScale | out |
|:---|:---|:---|:---|:---|:---|
| INT4 | INT4 (ND/NZ) | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | UINT64 | FLOAT/null | FLOAT16/BFLOAT16 |

**约束**：
- groupType=0，actType=0，groupListType=0/1/2。
- x/weight/out 仅支持长度为 1 的 TensorList。
- x 不转置，2 维 `(M, K)`。
- weight 3 维 `(E, K, N)`。ND 格式时 N 为 8 的倍数。
- weight NZ 格式支持转置，ND 仅支持非转置。
- perchannel：scale shape `(E, N)`；pergroup：scale shape `(E, G, N)`，G 能整除 K 且 K/G 为偶数。
- 开启右矩阵 NZ 转置后：K/G 对齐 64，K 对齐 64，N 对齐 16。

</details>

<a id="a8w4场景约束"></a>

<details>
<summary>A8W4 场景约束</summary>

**数据类型要求：**

| x | weight | bias | scale | offset | perTokenScale | out |
|:---|:---|:---|:---|:---|:---|:---|
| INT8 | INT4 (ND/NZ) | FLOAT | UINT64 | <span title="该参数在当前场景下必须传空（nullptr）">—</span> | FLOAT | BFLOAT16 |
| INT8 | INT4 (ND/NZ) | FLOAT | UINT64 | FLOAT/null | FLOAT | FLOAT16 |

**约束**：
- groupType=0，actType=0。
- 当前支持x、out均为长度为1的TensorList。
- weight、scaleOptional、biasOptional和offsetOptional支持单Tensor场景（tensorlist长度为1）和多Tensor场景（tensorlist长度大于1）
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
- tuningConfigOptional[1]可置1，以开启 weight 亲核格式（需 weight shape `(E, N, K)` 经 ND2NZ 转换），K ≥ 2048 && N ≥ 2048 时有性能优势。

</details>

<a id="a16w8场景约束"></a>

<details>
<summary>A16W8 场景约束</summary>

**数据类型要求：**

| x | weight | bias | antiquantScale | antiquantOffset | out |
|:---|:---|:---|:---|:---|:---|
| FLOAT16 | INT8 (ND) | FLOAT16/null | FLOAT16 | FLOAT16 | FLOAT16 |
| BFLOAT16 | INT8 (ND) | FLOAT/null | BFLOAT16 | BFLOAT16 | BFLOAT16 |

**约束**：
- x 不转置；groupType=-1/0；actType=0；groupListType=0/1/2。
- 仅 perchannel 模式。
- weight 多 Tensor 时，pergroup 长度 $s_i = K_i / G_i$ 须各组相等。
- antiquantScale/Offset shape：

| 子场景 | shape |
|:---|:---|
| weight 单，perchannel | `(E, N)` |
| weight 多，perchannel | `(N_i)` |

</details>

<a id="a16w4场景约束"></a>

<details>
<summary>A16W4 场景约束</summary>

**数据类型要求：**

| x | weight | bias | antiquantScale | antiquantOffset | out |
|:---|:---|:---|:---|:---|:---|
| FLOAT16 | INT4 (ND) | FLOAT16/null | FLOAT16 | FLOAT16 | FLOAT16 |
| BFLOAT16 | INT4 (ND) | FLOAT/null | BFLOAT16 | BFLOAT16 | BFLOAT16 |

**约束**：
- x 不转置；groupType=-1/0；actType=0；groupListType=0/1/2。
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

</details>

<a id="a3a2-grouptype-支持场景"></a>

<details>
<summary>groupType 支持场景</summary>

> x、weight、y 输入为 aclTensorList。"单" = 单元素 TensorList，"多" = 多元素 TensorList。

| groupType | x | weight | y | splitItem | groupListOptional | 转置 | 其余场景限制 |
|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|
| -1 | 多 | 多 | 多 | 0/1 | 必须传空 | x 不转置；weight 可转置（统一） | x 2-6维（维度一致），weight 2维，y 同 x |
| 0 | 单 | 单 | 单 | 2/3 | 必须传；groupListType=0 末值 ≤ x 第一维；groupListType=1 总和 ≤ x 第一维；groupListType=2 第二列总和 ≤ x 第一维；最大 1024 组 | x 不转置；weight 可转置（A8W4/A4W4 不可） | weight 3维，x/y 2维 |
| 0 | 单 | 多 | 单 | 2/3 | 必须传；groupListType=0 末值 ≤ x 第一维；groupListType=1 总和 ≤ x 第一维；groupListType=2 第二列总和 ≤ x 第一维；最大 128 组 | x 不转置；weight 可转置（统一） | x/weight/y 2维；weight N 轴相等 |
| 0 | 多 | 多 | 单 | 2/3 | 可选；若传则 groupListType=0 差值 = x 第一维、groupListType=1 数值 = x 第一维、groupListType=2 第二列 = x 第一维；最大 128 组 | x 不转置；weight 可转置（统一） | x/weight/y 2维；weight N 轴相等 |
| 2 | 单 | 单 | 单 | 2/3 | 必须传；groupListType=0 末值 ≤ x 第二维；groupListType=1 总和 ≤ x 第二维；groupListType=2 第二列总和 ≤ x 第二维；最大 1024 组 | x 必须转置；weight 不转置 | x/weight 2维，y 3维；bias 传空 |
| 2 | 单 | 多 | 多 | 0/1 | 必须传空 | x 必须转置；weight 不转置 | x/weight/y 2维；weight 最大 128 组；原始 shape 各 weight 第一维之和 ≤ x 第一维；bias 传空 |

</details>

---

<a id="8-接口调用"></a>

## 8. 接口调用

<a id="81-调用流程"></a>

### 8.1 调用流程

1. 根据场景准备输入（x/weight 及各场景所需的量化/伪量化参数）。
2. 调用 `aclnnGroupedMatmulV5GetWorkspaceSize` 获取 workspace 大小和 executor。
3. 根据返回的 workspaceSize 申请 Device 侧内存。
4. 调用 `aclnnGroupedMatmulV5` 执行计算。
5. 同步等待 Stream 任务完成。
6. 将输出从 Device 拷贝到 Host。

<a id="82-配置示例"></a>

### 8.2 配置示例

<a id="821-grouplistoptional-配置示例"></a>

#### 8.2.1 groupListOptional 配置示例

以 shape：$M=789$、$K=4096$、$N=7168$、$E=9$（其中第 0 组 123 token，第 2/5 组各 333 token，其余为空）为例：

| groupListType | groupListOptional 值 |
|:---:|:---|
| 0（累积和） | `[123, 123, 456, 456, 456, 789, 789, 789, 789]` |
| 1（各组大小） | `[123, 0, 333, 0, 0, 333, 0, 0, 0]` |
| 2（索引+大小） | `[[0, 123], [2, 333], [5, 333], [1, 0], [3, 0], [4, 0], [6, 0], [7, 0], [8, 0]]`（非零组前置） |

<a id="822-tuningconfigoptional-配置示例"></a>

#### 8.2.2 tuningConfigOptional 配置示例

仅 <term>Atlas A3/A2 训练系列产品</term> 支持，Ascend 950PR/950DT 不支持此参数。
tuningConfigOptional 为 Host 侧 aclIntArray，存储 INT64 元素。

| 索引 | 语义 | 适用场景 |
|:---:|------|------|
| [0] | 预期各专家处理的 token 数，tiling 按此优化 | A8W4、A8W8、A4W4（x/weight/out 单 Tensor） |
| [1] | 是否开启 weight 亲核格式（需先转置再 NZ） | A8W4 |
| [2] | 允许额外 workspace 上限（-1=不限），用于定轴算法优化 | A8W8 |

<a id="83-示例代码索引"></a>

### 8.3 示例代码索引

以下表格提供各场景下的示例代码文件链接，请参考文件中的注释了解对应场景说明。编译与运行请参考 [编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

| 场景 | 设备 | 示例文件 | 说明 |
|------|:---:|------|------|
| 非量化 | 公共 | [test_aclnn_grouped_matmul.cpp](../examples/test_aclnn_grouped_matmul.cpp) | 非量化基础示例（涵盖通用调用流程） |
| 全量化-A8W8 | 公共 | [test_aclnn_grouped_matmul_v5_a8w8.cpp](../examples/test_aclnn_grouped_matmul_v5_a8w8.cpp) | x=INT8, weight=INT8, scale=UINT64, out=BF16 |
| 伪量化-A16W8 | 公共 | [test_aclnn_grouped_matmul_v5_a16w8.cpp](../examples/test_aclnn_grouped_matmul_v5_a16w8.cpp) | x=BF16, weight=INT8, antiquantScale=BF16 |
| 伪量化-A16W4 | 公共 | [test_aclnn_grouped_matmul_v5_a16w4.cpp](../examples/test_aclnn_grouped_matmul_v5_a16w4.cpp) | x=BF16, weight=INT4, antiquantScale=BF16 |
| 非量化 | A5 | [examples/arch35/test_aclnn_grouped_matmul_v5_noquant.cpp](../examples/arch35/test_aclnn_grouped_matmul_v5_noquant.cpp) | A5 非量化示例 |
| 伪量化（weight NZ） | A5 | [examples/arch35/test_aclnn_grouped_matmul_weight_nz.cpp](../examples/arch35/test_aclnn_grouped_matmul_weight_nz.cpp) | A5 伪量化 NZ 格式示例 |
| 伪量化（V5 NZ） | A5 | [examples/arch35/test_aclnn_grouped_matmul_v5_weight_nz.cpp](../examples/arch35/test_aclnn_grouped_matmul_v5_weight_nz.cpp) | A5 V5 伪量化 NZ 示例 |
| MX 量化 | A5 | [examples/arch35/test_aclnn_grouped_matmul_v4_mx_quant.cpp](../examples/arch35/test_aclnn_grouped_matmul_v4_mx_quant.cpp) | A5 MX 量化示例 |
| 全量化（静态） | A5 | [examples/arch35/test_aclnn_grouped_matmul_v5_quant_static.cpp](../examples/arch35/test_aclnn_grouped_matmul_v5_quant_static.cpp) | x=INT8, weight=INT8, scale=FLOAT32, perTokenScale=nullptr |
| 全量化（动态 K-C） | A5 | [examples/arch35/test_aclnn_grouped_matmul_v5_quant_dynamic.cpp](../examples/arch35/test_aclnn_grouped_matmul_v5_quant_dynamic.cpp) | x=INT8, weight=INT8, scale=FLOAT32, perTokenScale=FLOAT |
| G-B 量化 | A5 | [examples/arch35/test_aclnn_grouped_matmul_v5_quant_gb.cpp](../examples/arch35/test_aclnn_grouped_matmul_v5_quant_gb.cpp) | x/w=FLOAT8_E5M2, scale=FLOAT32 (3维), perTokenScale=FLOAT32 (2维) |
| 全量化-A4W4 | A2/A3 | [test_aclnn_grouped_matmul_v5_a4w4.cpp](../examples/test_aclnn_grouped_matmul_v5_a4w4.cpp) | x=INT4, weight=INT4, scale=UINT64, out=FP16 |
| 伪量化-A8W4 | A2/A3 | [test_aclnn_grouped_matmul_v5_a8w4.cpp](../examples/test_aclnn_grouped_matmul_v5_a8w4.cpp) | x=INT8, weight=INT4, scale=UINT64, perTokenScale=FLOAT |
