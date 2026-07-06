# GroupedMatmul

<style>
abbr {
  border-bottom: 1px dotted #666;
  cursor: help;
}
</style>

## GroupedMatmul 简介

GroupedMatmul 是分组矩阵乘算子，支持按 M 轴或 K 轴对输入张量进行分组，分别对各组执行矩阵乘法。涵盖非量化、伪量化和全量化三种计算场景，广泛应用于 MoE 等大模型推理加速。

---

## 产品支持情况

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

## 专业术语

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

## 功能说明

接口功能：实现分组矩阵乘计算。公式为 $y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$，其中 $g$ 为分组数。

- **M 轴分组（groupType=0）**：$k_i$、$n_i$ 各组相同，$m_i$ 可以不相同。
- **K 轴分组（groupType=2）**：$m_i$、$n_i$ 各组相同，$k_i$ 可以不相同。
- **不分组（groupType=-1）**：各组 shape 完全独立。

基础计算公式：

$$
y_i=x_i\times weight_i + bias_i
$$

详细计算公式（按场景分类：非量化、伪量化、全量化）及版本演进请参见 [aclnnGroupedMatmulV5 功能说明](docs/aclnnGroupedMatmulV5.md#3-功能说明)。

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

## 约束说明

### 确定性计算

GroupedMatmul 默认提供确定性实现。

### 公共约束

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

> 各场景详细约束请参见 [aclnnGroupedMatmulV5 约束说明](docs/aclnnGroupedMatmulV5.md#7-%E7%BA%A6%E6%9D%9F%E8%AF%B4%E6%98%8E)。

---

## 接口调用

| 调用方式 | 接口文档 | 调用样例 | 说明 |
|----------|----------|----------|------|
| aclnn 调用 | [aclnnGroupedMatmulV5](docs/aclnnGroupedMatmulV5.md) | 见接口文档 [示例代码索引](docs/aclnnGroupedMatmulV5.md#83-%E7%A4%BA%E4%BE%8B%E4%BB%A3%E7%A0%81%E7%B4%A2%E5%BC%95) | 两段式接口：先调用 `aclnnGroupedMatmulV5GetWorkspaceSize` 获取 workspace 大小和 executor，再调用 `aclnnGroupedMatmulV5` 执行计算 |
