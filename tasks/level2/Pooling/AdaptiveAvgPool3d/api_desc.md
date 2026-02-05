# AdaptiveAvgPool3d

## 功能描述

### 算子功能
该算子在指定三维输出shape信息的情况下，完成输入张量的3D自适应平均池化计算。

### 计算公式



## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.adaptive_avg_pool3d()` 函数形式提供：


```python
def adaptive_avg_pool3d(x: Tensor, output_size: List[int]):
    """
    实现 adaptive_avg_pool3d 操作。

    参数:
        x (Tensor): 输入张量，Device侧的张量，数据格式支持ND，NCDHW。
        output_size (List[int]): 指定输出在DHW维度上的shape大小。

    返回:
        Tensor: 输出张量，数据类型与输入一致，数据格式保持不变。
    """

```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建输入张量
x = torch.randn([2,3,4,5,6],dtype=torch.int32)
output_size = [2, 2, 2]

result = kernel_gen_ops.adaptive_avg_pool3d(x, output_size)
```
## 约束与限制

- 输出 out 的shape需要与x和output_size推导出的shape结果一致。
- 输入 output_size 的数组长度为3。
- 输出 out 的shape和输入 x 的数据类型一致。
