# aclnnLinspace

## 功能描述

### 算子功能
该算子生成一个等间隔数值序列。创建一个大小为steps的1维向量，其值从start起始到end结束（包含）线性均匀分布。

### 计算公式

$$
out = (start, start + \frac{end - start}{steps - 1},...,start + (steps - 2) * \frac{end - start}{steps -1}, end)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.lin_space()` 函数形式提供：


```python
def lin_space(start: Tensor, end: Tensor, steps: int):
    """
    实现 lin_space 操作。

    参数:
        start (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        end (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        steps (int): 值的步长，需要满足steps大于等于0。

    返回:
        Tensor: 输出张量，数据类型与输入一致，数据格式保持不变，输出shape的元素个数需要与steps一致。
    """

```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建输入张量
start = torch.randint(1, 1000, [1],dtype=torch.int32)
end = torch.randint(1, 1000, [1],dtype=torch.int32)
steps = torch.randint(1, 1000, [1],dtype=torch.int64)[0].item()

result = kernel_gen_ops.lin_space(start, end, steps)
```
## 约束与限制

- 输出 out 的元素个数需要与 steps 一致。
- 输入 start 和输入 end 的元素数量必须为1。
- 张量数据格式支持ND。
- 输入 end 的数据类型和输入 start 的数据类型一致。
- 输出 out 的数据类型和输入 start 的数据类型一致。
