# AdaptiveMaxPool3DGrad

## 功能描述

### 算子功能
该算子是正向自适应最大池化的反向传播，将梯度回填到每个自适应窗口最大值的坐标处，相同坐标处累加。

### 计算公式



## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.adaptive_max_pool3d_grad()` 函数形式提供：


```python
def adaptive_max_pool3d_grad(grad: Tensor, x: Tensor, argmax: Tensor):
    """
    实现 adaptive_max_pool3d_grad 操作。

    参数:
        grad (Tensor): 输入张量，Device侧的张量，数据格式支持NCDHW。
        x (Tensor): 输入张量，Device侧的张量，数据格式支持NCDHW。
        argmax (Tensor): 输入张量，Device侧的张量，数据格式支持NCDHW。

    返回:
        Tensor: 输出张量，数据类型与输入一致，数据格式保持不变。
    """

```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建输入张量
y_grad = torch.randn([2,2,2,1,1],dtype=torch.float32)
x = torch.randn([2,2,3,2,2],dtype=torch.float32)
argmax = torch.randn([2,2,2,1,1],dtype=torch.int32)

result = kernel_gen_ops.adaptive_max_pool3d_grad(y_grad, x, argmax)
```
## 约束与限制

- 输入 x 的D、H、W轴的乘积需要小于等于int32的最大表示范围。
- 输入 y_grad 的前两维size需要和输入 x 的前两维size一致。
- 输入 y_grad 的数据类型和输入 x 的数据类型一致。
- 输入 argmax 的数据类型需要为int32。
- 输出 out 的数据类型和输入 x 的数据类型一致。
- 输出 out 的shape需要与x的shape结果一致。
