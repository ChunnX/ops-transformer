# aclnnCrossV2

## 功能描述

### 算子功能
该AscendC算子用于计算计算两个张量在指定维度上的叉积（Cross Product）

### 计算公式

该算子目前仓上无文档（2026-02-09）

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.cross_v2()` 函数形式提供：

```python
def cross_v2(predict, label, reduction='mean'):
    """
    计算预测值和目标值之间的均方误差损失。
    
    参数:
        x1 (Tensor): 第一个输入张量，形状为 [B, D1, D2, ..., Dk]，其中最后一个维度必须 ≥ 3。支持的数据类型有float32、float16、bfloat16、int32、int8、uint8、int16；
        x2 (Tensor): 第二个输入张量，形状与 x1 完全一致，用于与 x1 进行叉积运算。
        dim (int): 指定进行叉积运算的维度索引。支持负索引（如 -1 表示倒数第一维）。必须为整数。
                   - 若 dim 为 -1，则在倒数第一维进行叉积。
                   - 不应超出输入张量的维度大小。
                   - 若维度大小 < 3，将引发错误。
                
        
    返回:
        y (Tensor): 输出张量，形状与输入张量 x1 相同，表示在指定维度上的叉积结果。
    
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
shape = [8, 8, 3]
dtype = torch.float32
x1 = torch.rand(shape, dtype=dtype, device=device) * 2 - 1
x2 = torch.rand(shape, dtype=dtype, device=device) * 2 - 1
# dim对应的输入x该维的大小必须为3
dim = -1 

# 使用gelu执行计算
result = kernel_gen_ops.cross_v2(x1, x2, dim)
```

## 约束与限制

- 输入x1最后一个维度必须 $\leq 3$
- dim:
    - 注意，标杆torch.cross要求x1中dim指定的维度上的大小必须等于3(本算子中dim指定维度的数据大小可以大于等于3)。
    - 若 dim 为 -1，则在倒数第一维进行叉积。
    - 不应超出输入张量的维度大小。