# GroupedMatmul

<style>
abbr {
  border-bottom: 1px dotted #666;
  cursor: help;
}
</style>

## GroupedMatmul简介

GroupedMatmul是 MoE网络中的重要算子，核心功能是做分组矩阵乘：将输入 token按组分配给不同专家，每组独立执行矩阵乘法。支持按 M轴或K轴对输入张量进行分组，涵盖非量化、伪量化和全量化三种计算场景。在 MoE推理中，GroupedMatmul通常被调用两次：第一次对输入 token与各专家的上投影权重做分组矩阵乘，得到各专家的中间结果；第二次将中间结果与各专家的下投影权重做分组矩阵乘，得到各专家的输出。

```mermaid
graph LR
    B["moe_gating_top_k"] --> C["moe_init_routing"]
    C --> D["grouped_matmul(W1)"]
    D --> E["swiglu"]
    E --> F["grouped_matmul(W2)"]
    F --> G["moe_finalize_routing"]

    classDef gmm fill:#E8ECFF,stroke:#1c7ed6,stroke-width:2px,color:#333;
    classDef other fill:#DCEFFA,stroke:#1c7ed6,stroke-width:2px,color:#333;

    class D,F gmm;
    class B,C,E,G other;
```

---

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      √     |
|<term>Atlas 训练系列产品</term>|      ×     |
|<term>Kirin X90 处理器系列产品</term> | √ |
|<term>Kirin 9030 处理器系列产品</term> | √ |

---

## 专业术语

| 术语 | 简要说明 |
|------|----------|
| 非量化 | x和 weight 均为浮点类型直接计算 |
| 伪量化 | weight低精度存储，通过 antiquantScale/Offset反量化后计算 |
| 全量化 | x和 weight 均为低精度计算，通过 scale还原 |
| perchannel | 每列独立缩放因子 |
| pertensor | 一个 Tensor统一缩放因子 |
| pertoken | 每行独立缩放因子 |
| pergroup | 在 reduce 轴上对数据分组，每组使用独立的量化参数，group size 为 gs |
| perblock | 在所有轴上（含 M 方向）对数据分块，每块使用独立的量化参数，block size 为 bs |
| T-C | pertensor-perchannel：输入侧统一缩放，权重侧每列独立缩放 |
| T-T | pertensor-pertensor：输入和权重均为统一缩放 |
| K-C | pertoken-perchannel：输入侧每行独立缩放，权重侧每列独立缩放 |
| K-T | pertoken-pertensor：输入侧每行独立缩放，权重侧统一缩放 |
| G-B | pergroup-perblock：输入侧按 K分组缩放，权重侧按 N×K分块缩放 |
| MX | pergroup-pergroup（G-G）量化模式，量化参数类型为 FLOAT8_E8M0，group size 为 32 的特例 |
| ND | 常规连续排布 |
| NZ | FRACTAL_NZ格式（亲核排布） |

## 功能说明

接口功能：实现分组矩阵乘计算。公式为 $y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$，其中 $g$ 为分组数。

- **M轴分组（groupType=0）**：$k_i$、$n_i$ 各组相同，$m_i$ 可以不相同。
- **K轴分组（groupType=2）**：$m_i$、$n_i$ 各组相同，$k_i$ 可以不相同。
- **不分组（groupType=-1）**：各组 shape 完全独立。

基础计算公式：

$$
y_i=x_i\times weight_i + bias_i
$$

详细计算公式（按场景分类：非量化、伪量化、全量化）及版本演进请参见 [aclnnGroupedMatmulV5功能说明](docs/aclnnGroupedMatmulV5.md#3-功能说明)。

---

## 参数说明

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
| perTokenScaleOptional | 可选输入 | 公式中的 $per\_token\_scale$，代表量化参数中由 x量化引入的缩放因子 | FLOAT、FLOAT8_E8M0<sup>2</sup> | ND |
| groupListOptional | 可选输入 | 代表输入和输出分组轴方向的 matmul大小分布 | INT64 | ND |
| activationInputOptional | 可选输入 | 代表激活函数的反向输入，当前只支持传入 nullptr | — | — |
| activationQuantScaleOptional | 可选输入 | 激活函数量化缩放因子（预留参数），当前只支持传入 nullptr | — | — |
| activationQuantOffsetOptional | 可选输入 | 激活函数量化偏移量（预留参数），当前只支持传入 nullptr | — | — |
| <abbr title="代表输出是否要做 Tensor切分：0/1=多Tensor 输出；2/3=单Tensor 输出">splitItem</abbr> | 属性 | 代表输出是否要做 Tensor切分 | INT64 | — |
| <abbr title="指定分组轴：-1=不分组，0=M轴，2=K轴">groupType</abbr> | 属性 | 代表需要分组的轴 | INT64 | — |
| <abbr title="代表 groupList输入的分组方式：0=累积和；1=各组大小；2=[组索引, 组大小] 对">groupListType</abbr> | 属性 | 代表 groupList输入的分组方式 | INT64 | — |
| <abbr title="代表激活函数类型：0=NONE/1=ReLU/2=GELU_TANH/4=FAST_GELU/5=SILU">actType</abbr> | 属性 | 代表激活函数类型 | INT64 | — |
| tuningConfigOptional | 可选输入 | 第一个数代表各个专家处理的 token数的预期值；第二个数代表 A8W4 可选开启 weight 先转置的 NZ格式；第三个数代表允许额外使用的内存空间 | INT64 | — |
| out | 输出 | 公式中的输出 $y$ | FLOAT、FLOAT16、INT32、INT8、BFLOAT16 | ND |
| activationFeatureOutOptional | 输出 | 激活函数的输入数据，当前只支持传入 nullptr | — | — |
| dynQuantScaleOutOptional | 输出 | 动态量化缩放因子，当前只支持传入 nullptr | — | — |

- **<term>Ascend 950PR/Ascend 950DT</term>**：
  - 上表数据类型列中的角标 <sup>1</sup> 代表该系列不支持的数据类型。
  - 输入参数 x、weight均不支持 INT16 类型，且 x 不支持 INT4 类型。
  - 输入参数 x、weight，输出参数 out在非量化场景支持最多 1024个Tensor，在伪量化支持最多 128个Tensor，在全量化场景最多支持 1个Tensor。
- **<term>Atlas A3/A2 训练系列产品</term>**：
  - 上表数据类型列中的角标 <sup>2</sup> 代表该系列不支持的数据类型。
  - 不支持 FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8、FLOAT8_E8M0类型。
  - 输入参数 biasOptional不支持 BFLOAT16。
  - 输入参数 scaleOptional不支持 INT64 类型。
  - 输入参数 x、weight，输出参数 out支持最多 128个Tensor。
- **<term>Kirin X90/Kirin 9030 处理器系列产品</term>**：
  - 不支持 BFLOAT16、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT8_E8M0、HIFLOAT8类型。

---

## 约束说明

### 公共约束

- x/weight 中每组 Tensor的最后一维 < 65536。
  - $x_i$ 最后一维：x不转置时为 K轴，x 转置时为 M轴。
  - $weight_i$ 最后一维：weight不转置时为 N轴，weight转置时为 K轴。
- x/weight转置时对应 Tensor必须 [非连续](../../docs/zh/context/非连续的Tensor.md)。
- weight为 [FRACTAL_NZ](../../docs/zh/context/数据格式.md) 格式时，shape 须满足 NZ格式要求。
- perTokenScaleOptional：通常仅1维，长度与 x的M一致。仅支持 x/weight/out均为单Tensor场景。
- groupListOptional：out中 TensorList长度为1时，groupList约束输出数据有效范围，未指定部分不参与更新。
- groupListType：
  - groupListType=0：须为非负单调非递减数列（累积和）。
  - groupListType=1：须为非负数列（各组大小）。
  - groupListType=2：须为非负数列，shape `[E, 2]`（[组索引, 组大小]），非零组前置。

> 各场景详细约束请参见 [aclnnGroupedMatmulV5约束说明](docs/aclnnGroupedMatmulV5.md#7-约束说明)。

---

## 接口调用

| 调用方式 | 接口文档 | 调用样例 | 说明 |
|----------|----------|----------|------|
| aclnn调用 | [aclnnGroupedMatmulV5](docs/aclnnGroupedMatmulV5.md) | 见接口文档 [示例代码索引](docs/aclnnGroupedMatmulV5.md#83-示例代码索引) | 两段式接口：先调用 `aclnnGroupedMatmulV5GetWorkspaceSize` 获取 workspace大小和 executor，再调用 `aclnnGroupedMatmulV5` 执行计算 |
