# AdaptiveMaxPool3d

## 功能描述

### 算子功能
该算子根据输入的output_size计算每次kernel的大小，对输入x进行3维最大池化操作，输出池化后的值y和索引indices。

### 计算公式

y tensor中对于DHW轴上每个位置为$(l,m,n)$的元素来说，其计算公式为：
  
$$
D^{l}_{left} = floor((l*D)/D_o)
$$

$$
D^{l}_{right} = ceil(((l+1)*D)/D_o)
$$

$$
H^{m}_{left} = floor((m*H)/H_o)
$$

$$
H^{m}_{right} = ceil(((m+1)*H)/H_o)
$$

$$
W^{n}_{left} = floor((n*W)/W_o)
$$

$$
W^{n}_{right} = ceil(((n+1)*W)/W_o)
$$

$$
y(N,C,l,m,n)=\underset {i \in [D^{l}_{left}, D^{l}_{right}],j\in [H^m_{left},H^m_{right}], k \in [W^n_{left},W^n_{right}] }{max} input(N,C,i,j,k)
$$

$$
indices(N,C,l,m,n)=\underset {i \in [D^{l}_{left}, D^{l}_{right}],j\in [H^m_{left},H^m_{right}], k \in [W^n_{left},W^n_{right}] }{argmax} input(N,C,i,j,k)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.adaptive_max_pool3d()` 函数形式提供：


```python
def adaptive_max_pool3d(x: Tensor, output_size: List[int]):
    """
    实现 adaptive_max_pool3d 操作。

    参数:
        x (Tensor): 输入张量，Device侧的张量，数据格式支持NCDHW。
        output_size (List[int]): 指定输出在DHW维度上的shape大小。

    返回:
        Tensor: 输出张量，数据类型与输入一致，数据格式保持不变。
        Tensor: 输出张量，数据类型为INT32，数据格式保持不变。
    """

```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建输入张量
x = torch.randn([1,2,1,3,4],dtype=torch.int32)
output_size = [1, 2, 2]

outputOut, indicesOut = kernel_gen_ops.adaptive_max_pool3d(x, output_size)
```
## 约束与限制

- 输入 x 的D、H、W轴的乘积不能大于int32的最大表示。
- 输入 output_size 的数组长度为3。
- 输出 outputOut 的shape需要与x和output_size推导出的shape结果一致。
- 输出 indicesOut 的shape需要与x和output_size推导出的shape结果一致。
- 输出 outputOut 的数据类型和输入 x 的数据类型一致。
- 输出 indicesOut 的数据类型需要为int32。
