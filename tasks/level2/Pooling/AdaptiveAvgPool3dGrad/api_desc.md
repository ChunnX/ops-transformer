# AdaptiveAvgPool3dGrad

## 功能描述

### 算子功能
该算子是AdaptiveAvgPool3d的反向计算。

### 计算公式



## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.adaptive_avg_pool3d_grad()` 函数形式提供：


```python
def adaptive_avg_pool3d_grad(y_grad: Tensor, x: Tensor):
    """
    实现 adaptive_avg_pool3d_grad 操作。

    参数:
        y_grad (Tensor): 输入张量，Device侧的张量，数据格式支持ND，NCDHW。
        x (Tensor): 输入张量，Device侧的张量，数据格式支持ND，NCDHW。

    返回:
        Tensor: 输出张量，数据类型与输入一致，数据格式保持不变。
    """

```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建输入张量
y_grad = torch.randn([2,2,1,1,2],dtype=torch.int32)
x = torch.randn([2,2,1,1,4],dtype=torch.int32)

result = kernel_gen_ops.adaptive_avg_pool3d_grad(y_grad, x)
```
## 约束与限制

- 输出 out 的shape需要与x的shape结果一致。
- 输入 y_grad 的前两维size需要和输入 x 的前两维size一致。
- 输入 y_grad 的数据类型和输入 x 的数据类型一致。
- 输出 out 的数据类型和输入 x 的数据类型一致。
