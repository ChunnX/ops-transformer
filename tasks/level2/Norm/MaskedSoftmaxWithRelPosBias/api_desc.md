# aclnnMaskedSoftmaxWithRelPosBias

## 功能描述

### 算子功能
在最后一维上对 (scale_value * x + atten_mask + relative_pos_bias) 做 softmax，用于带相对位置偏置和可选 attention mask 的注意力得分归一化。

### 计算公式
$$
\text{score} = \text{scale\_value} \cdot x + \text{atten\_mask} + \text{relative\_pos\_bias}
$$
$$
y = \text{softmax}(\text{score}, \text{dim}=-1)
$$
其中 atten_mask 为可选，未提供时该项不参与相加。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.masked_softmax_with_rel_pos_bias()` 函数形式提供：

```python
def masked_softmax_with_rel_pos_bias(x, atten_mask, relative_pos_bias, scale_value=1.0):
    """
    实现带 mask 与相对位置偏置的 softmax。

    参数:
        x (Tensor): Device 侧输入，数据类型支持 FLOAT16、BF16、FLOAT32，数据格式支持 ND。
        atten_mask (Tensor, 可选): 注意力 mask，可与 x 广播，未提供时传 None。
        relative_pos_bias (Tensor): Device 侧相对位置偏置，与 x 可广播，数据类型同 x。
        scale_value (float, 可选): 对 x 的缩放系数，默认 1.0。

    返回:
        Tensor: 与 x 同 shape、同 dtype 的 softmax 结果。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入：例如 [batch, heads, seq, seq]
x = torch.randn(2, 8, 128, 128, dtype=torch.float16)
relative_pos_bias = torch.randn(2, 8, 128, 128, dtype=torch.float16)
# 可选 mask
atten_mask = torch.zeros(2, 8, 128, 128, dtype=torch.float16)

y = kernel_gen_ops.masked_softmax_with_rel_pos_bias(
    x, atten_mask, relative_pos_bias, scale_value=1.0
)
```

## 约束与限制

- 张量数据格式支持 ND。
- x、atten_mask（若提供）、relative_pos_bias 支持 FLOAT16、BF16、FLOAT32（部分平台仅 FLOAT16/FLOAT）。
- 输出与 x 同 shape、同 dtype。
