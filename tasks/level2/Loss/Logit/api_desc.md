# aclnnLogit

## 功能描述

### 算子功能
该算子是概率到对数几率（log-odds）转换的一个数学运算，常用于概率值的反变换。它可以将一个介于0和1之间的概率值转换为一个实数值。


### 计算公式
$$
y_i = \ln \left( \frac{z_i}{ 1 -z_i} \right)
$$

其中：

$$
z_i =
\begin{cases} 
\text{eps}, & \text{if } x_i < \text{eps} \\
x_i, & \text{if } \text{eps} \leq x_i \leq 1 - \text{eps} \\
1 - \text{eps}, & \text{if } x_i > 1 - \text{eps}
\end{cases}
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.logit()` 函数形式提供：

```python
def logit(predict, label, reduction='mean'):
    """
    实现概率到对数几率转换（log-odds)。
    
    参数:
        input_tensor (Tensor): 输入张量，数据类型包括float32/float16/bfloat
        eps (double, 可选): 输入input的epsilon限制边界，建议值-1
        
    返回:
        out（Tensor）: 转换后结果张量，数据类型与输入一致

    """
```

## 使用案例

```python
import torch
import numpy as np
import kernel_gen_ops

# 创建输入张量
np.random.seed(1234)
input_shape = [3, 6]
input_data = np.random.uniform(0.1, 0.9, size=input_shape).astype(np.float32)
input_tensor = torch.from_numpy(input_data).to(torch.float32).to('npu')
eps = -1
# 使用gelu执行计算
result = kernel_gen_ops.logit(input_tensor, eps)
```

## 约束与限制

- input取值范围不在[0, 1]，输出NaN，取值0时为-Inf，取值为1时为Inf。

- eps大于1时，输出NaN，取值为1时为Inf。