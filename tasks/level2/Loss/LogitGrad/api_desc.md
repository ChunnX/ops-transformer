# aclnnLogitGrad

## 功能描述

### 算子功能
该AscendC算子用于完成`aclnnLogit`的反向传播。

### 计算公式
$$
dx_i=
\begin{cases} 
NaN, & \text{if } x < \text0 \text{ or } x > 1 ,eps <0 \\
0, & \text{if } x < \text{eps} \text{ or } x > 1 - \text{eps},eps \geq 0 \\
\frac{dy_i}{x_i \cdot (1 - x_i)}, & \text{if } \text{eps} \leq x_i \leq 1 - \text{eps} \\
\end{cases}
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.logit_grad()` 函数形式提供：

```python
def logit_grad(predict, label, reduction='mean'):
    """
    计算预测值和目标值之间的均方误差损失。
    
    参数:
        x_tensor (Tensor): 输入张量，数值在(0-1)之间
        dy_tensor (Tensor): 正向输出结果的梯度
        eps (double, 可选): 输入x_tensor的epsilon限制边界，建议值-1
        
    返回:
        out (Tensor): 公式中的dx, 输出张量，与输入数据类型、shape与输入x_tensor一致
        
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量

# 使用 PyTorch 生成随机数据（范围 [0.1, 0.9]）
torch.manual_seed(1234)
input_tensor = torch.rand(input_shape, dtype=torch.float32, device='npu') * 0.8 + 0.1  # [0.1, 0.9]
dy_tensor = torch.rand(input_shape, dtype=torch.float32, device='npu')
eps = 1e-3

# 执行计算
out = kernel_gen_ops.logit_grad(input_tensor, dy_tensor, eps)
```

## 约束与限制
- eps的取值对x和输出的影响：
    - eps小于0时，x取值范围不在[0, 1]，输出NaN。
    - eps大于等于0时，x取值范围不在[eps, 1 - eps]，输出0。
    - eps大于1时，输出NaN，取值为1时为Inf。
    - eps取值为Inf时，输出0。
    - eps取值为NaN时，输出NaN。
- x取值对输出的影响：
    - x取值0和1时，输出Inf。
    - x为Inf或NaN时，输出为NaN。
- dy为Inf或NaN时，输出为Inf或NaN。