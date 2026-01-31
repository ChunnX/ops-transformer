# aclnnHistogramV2

## 功能描述

### 算子功能
该算子计算张量直方图。以min和max作为统计上下限，在min和max之间划出等宽的数量为bins的区间，统计张量self中元素在各个区间的数量。如果min和max相等，则使用张量中所有元素的最小值和最大值作为统计的上下限。小于min和大于max的元素不会被统计。

### 计算公式

无

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.histogram_v2()` 函数形式提供：


```python
def histogram_v2(input: Tensor, min: Tensor, max: Tensor, bins: int):
    """
    实现 histogram_v2 操作。

    参数:
        input (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        min (Tensor): 输入张量，Device侧的张量，数据格式支持ND，直方图的统计下限（包括）。
        max (Tensor): 输入张量，Device侧的张量，数据格式支持ND，直方图的统计上线（包括）。
        bins (int): 直方图bins的数量。

    返回:
        Tensor: 输出张量。
    """

```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建输入张量
input = torch.randn(4, 8, dtype=torch.float32)
min = torch.tensor([-0.7], dtype=torch.float32)
max = torch.tensor([0.7], dtype=torch.float32)

result = kernel_gen_ops.histogram_v2(input, min, max, 4)
```
## 约束与限制

- 输入 bins 需要大于等于0。
- 输入 min 需要大于 输入 max 。
- 张量数据格式支持ND。
- 输出 out 的元素数量需要等于bins。
