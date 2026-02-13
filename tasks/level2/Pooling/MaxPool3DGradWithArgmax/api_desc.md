# MaxPool3DGradWithArgmax

## 功能描述

### 算子功能
正向最大池化的反向传播，将梯度回填到每个窗口最大值的坐标处，相同坐标处累加。

### 计算公式

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.max_pool3d_grad_with_argmax()` 函数形式提供：


```python
def max_pool3d_grad_with_argmax(x: Tensor, grad: Tensor, indices: Tensor, ksize: List[int], strides: List[int], pads: List[int], dilation: List[int], ceil_mode: bool):
    """
    实现 max_pool3d_grad_with_argmax 操作。

    参数:
        x (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        grad (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        indices (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        ksize (List[int]): 池化窗口的大小，指定输出在DHW维度上的shape大小，数值必须大于0。
        strides (List[int]): 池化操作的步长，数值必须大于0。
        pads (List[int]): 在输入的D、H、W方向上padding补0的层数，数值在[0, ksize / 2]的范围内。
        dilation (List[int]): 表示控制窗口中元素的步幅。
        ceil_mode (bool): 推导出的输出shape是否向上取整。

    返回:
        Tensor: 输出张量，数据类型与输入 x 一致，数据格式保持不变。
    """

```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建输入张量
x = torch.randn([1,1,2,2,2],dtype=torch.float32)
grad = torch.randn([1,1,1,1,1],dtype=torch.float32)
indices = torch.randn([1,1,1,1,1],dtype=torch.int32)
ksize = [2, 2, 2]
strides = [2, 2, 2]
pads = [0, 0, 0]
dilation = [1, 1, 1]
ceil_mode = False

out, indices = kernel_gen_ops.max_pool3d_grad_with_argmax(x, grad, indices, ksize, strides, pads, dilation, ceil_mode)
```
## 约束与限制

- 输入 grad 和输入 x 的数据类型和数据格式需要保持一致。
- 输入 indices 和输入 grad 的shape和数据格式需要保持一致。
- 输入 indices 的数据类型为int32。
- 输入 ksize、strides 的数组长度为3，每个维度的值都要大于0。
- 输入 dilation 的数组长度为3，每个维度的值都必须为1。
- 输入 pads 的数组长度为3，每个维度的值需要在[0, ksize / 2]的范围内。
- 输出 out 的数据类型和输入 x 的数据类型一致。
- 输出 out 的shape需要与根据最大池化语义计算得到的shape结果一致。
- 输出 indices 的shape需要与输出 out 的shape一致。
- depth * height * weight需要小于等于int32的最大值。
