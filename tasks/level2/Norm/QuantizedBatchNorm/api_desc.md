# aclnnQuantizedBatchNorm

## 功能描述
### 算子功能
Quantized Batch Normalization (QuantizedBatchNorm) 是对量化输入进行批量归一化的算子。它对量化输入进行反量化、归一化、仿射变换，然后重新量化输出。

### 计算公式
$$
x_{fp32} = (x_{quant} - zp_{in}) \times scale_{in}
$$
$$
normalized = \frac{x_{fp32} - mean}{\sqrt{var + \epsilon}}
$$
$$
y_{fp32} = normalized \times weight + bias
$$
$$
y_{quant} = round(\frac{y_{fp32}}{scale_{out}} + zp_{out})
$$

其中：
* $x_{quant}$ 是量化输入
* $zp_{in}$ 和 $scale_{in}$ 是输入的零点和缩放因子
* $mean$ 和 $var$ 是运行均值和方差
* $\epsilon$ 是数值稳定性常数
* $weight$ 和 $bias$ 是可选的仿射变换参数
* $zp_{out}$ 和 $scale_{out}$ 是输出的零点和缩放因子
* $y_{quant}$ 是量化输出

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.quantized_batch_norm()` 函数形式提供：

```python
def quantized_batch_norm(x, mean, var, input_scale, input_zero_point, output_scale, output_zero_point, weight=None, bias=None, epsilon=1e-5):
    """
    对量化输入张量执行 Batch Normalization 操作。

    参数:
        x (Tensor): 量化输入Device侧张量，支持DT_INT8、DT_UINT8、DT_INT32。
        mean (Tensor): 运行均值张量，形状与特征维度相同，DT_FLOAT类型。
        var (Tensor): 运行方差张量，形状与特征维度相同，DT_FLOAT类型。
        input_scale (Tensor): 输入缩放因子，标量，DT_FLOAT类型。
        input_zero_point (Tensor): 输入零点，标量，DT_INT32类型。
        output_scale (Tensor): 输出缩放因子，标量，DT_FLOAT类型。
        output_zero_point (Tensor): 输出零点，标量，DT_INT32类型。
        weight (Optional[Tensor]): 可选的缩放参数张量，形状与特征维度相同，DT_FLOAT类型。
        bias (Optional[Tensor]): 可选的偏移参数张量，形状与特征维度相同，DT_FLOAT类型。
        epsilon (double): 用于数值稳定性的一个小常数。

    返回:
        Tensor: 量化后的输出张量，数据类型与输入x相同。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量和参数
input_shape = [1, 2, 4, 4]
num_features = 2
dtype = torch.int8
eps = 1e-5

x = torch.randint(-128, 128, input_shape, dtype=dtype)
mean = torch.rand(num_features, dtype=torch.float32)
var = torch.rand(num_features, dtype=torch.float32) + 1e-3
input_scale = torch.tensor([0.1], dtype=torch.float32)
input_zero_point = torch.tensor([0], dtype=torch.int32)
output_scale = torch.tensor([0.1], dtype=torch.float32)
output_zero_point = torch.tensor([0], dtype=torch.int32)
weight = torch.rand(num_features, dtype=torch.float32)
bias = torch.rand(num_features, dtype=torch.float32)

# 使用 quantized_batch_norm 执行操作
y = kernel_gen_ops.quantized_batch_norm(
    x, mean, var, input_scale, input_zero_point, output_scale, output_zero_point,
    weight, bias, eps
)
```

## 约束与限制
- **功能维度**
  * 数据格式支持：**NCHW**。
  * `x` 和 `y` 的形状和数据类型必须一致。
  * `mean`、`var`、`weight`、`bias` 必须是 **1D 张量**，形状与特征维度匹配，且数据类型为 **DT_FLOAT**。
  * `input_scale`、`input_zero_point`、`output_scale`、`output_zero_point` 必须是标量。
  * `x` 的数据类型支持 **DT_INT8**、**DT_UINT8**、**DT_INT32**。
  * `epsilon` 必须为**浮点数**，double类型，且通常取值如 1e-5 或 1e-6。
