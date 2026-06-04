# 算子名称：BiasGateGelu

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| Atlas A2 训练系列产品                                         | 是       |

## 功能说明

- 算子功能：在 Transformer 模型的 FFN（Feed-Forward Network）层中，实现 **GeGLU（GELU Gated Linear Unit）** 的融合计算。将输入按列均分为 Gate 和 Value 两部分，分别加偏置后，对 Gate 部分施加 GELU 激活函数，再与 Value 部分逐元素相乘。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| blockDim | 输入 | AI Core 的数量，如 Ascend910B 为 40 | int64_t | - |
| in | 输入 | Gate 和 Value 拼接的输入张量，shape 为 (gbH, gbW)，其中 gbW = 2 × hidden_size | float16 | ND |
| bias | 输入 | 偏置向量，shape 为 (gbW,)，前半部分为 Gate 偏置，后半部分为 Value 偏置 | float16 | ND |
| out | 输出 | GeGLU 计算结果，shape 为 (gbH, gbW/2) | float16 | ND |
| gbH | 输入 | 输入张量的行数（token 数量） | int64_t | - |
| gbW | 输入 | 输入张量的列数（必须为偶数） | int64_t | - |

## 约束说明

- 输入张量的列数 `gbW` 必须为偶数，以便均分为 Gate 和 Value 两部分。
- 输入、偏置、输出的数据类型均为 float16（half）。

## 调用说明

```python
torch.ops.ascend_ops.bias_gate_gelu(
    block_dim, in_tensor, bias, out, gbH, gbW)
```
