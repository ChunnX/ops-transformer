# aclnnRmsNormQuant

## 功能描述

### 算子功能
对输入在最后一维上做 RMSNorm（factor = 1/sqrt(mean(x^2)+epsilon)，out = x * factor * gamma + beta），再按 scale/offset 做量化得到 int8 或 int4 输出。

### 计算公式
$$
\text{rms} = \sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon},\quad
\text{factor} = \frac{1}{\text{rms}}
$$
$$
\text{quant\_in} = x \cdot \text{factor} \cdot \gamma + \beta
$$
$$
y = \text{quantize}(\text{quant\_in}; \text{scale}, \text{offset})
$$
即 y 为 quant_in 经 scale/offset 量化后的整型表示（如 int8）。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.rms_norm_quant()` 函数形式提供：

```python
def rms_norm_quant(x, gamma, beta, scale, offset, epsilon):
    """
    实现 RMSNorm + 量化。

    参数:
        x (Tensor): Device 侧输入，数据类型支持 FLOAT16、BF16，数据格式支持 ND。
        gamma (Tensor): Device 侧 scale，最后一维与 x 最后一维一致，数据类型支持 FLOAT16、BF16。
        beta (Tensor): Device 侧 bias，与 gamma 同 shape，数据类型支持 FLOAT16、BF16。
        scale (Tensor): 量化 scale（标量或与需求一致），数据类型支持 FLOAT16、BF16。
        offset (Tensor): 量化 offset，数据类型支持 INT8。
        epsilon (float): RMSNorm 中的 epsilon。

    返回:
        List[Tensor]: 长度为 1，唯一元素为量化结果 y，与 x 同 shape，数据类型为 INT8 或 INT4（由算子属性 dst_type 决定）。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

x = torch.randn(4, 1024, dtype=torch.float16)
hidden = 1024
gamma = torch.ones(hidden, dtype=torch.float16)
beta = torch.zeros(hidden, dtype=torch.float16)
scale = torch.tensor([0.01], dtype=torch.float16)
offset = torch.tensor([0], dtype=torch.int8)

# 返回为 list，取第一个元素即为量化结果 y
y = kernel_gen_ops.rms_norm_quant(x, gamma, beta, scale, offset, epsilon=0.1)[0]
```

## 约束与限制

- 张量数据格式支持 ND。
- x、gamma、beta 支持 FLOAT16、BF16；scale 支持 FLOAT16、BF16；offset 为 INT8。
- gamma/beta 最后一维与 x 最后一维一致；scale/offset 通常为标量或 [1]。
- 输出 y 支持 INT8、INT4（由 dst_type 属性控制，依平台支持）。
