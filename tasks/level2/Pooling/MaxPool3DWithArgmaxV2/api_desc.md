# MaxPool3DWithArgmaxV2

## 功能描述

### 算子功能
  * 对于输入信号的输入通道，提供3维最大池化（Max pooling）操作，输出池化后的值out和索引indices。
  * 输入dims的描述：N - 批次，C - 通道，D - 深度，W - 宽度，H - 高度。
  * 当D * H * W超过int32时，建议在模型尺寸上分割D轴。

### 计算公式
* y tensor中每个元素的计算公式：
    
$$
out(N_i, C_j, d, h, w) = \max\limits_{{k\in[0,k_{D}-1],m\in[0,k_{H}-1],n\in[0,k_{W}-1]}}input(N_i,C_j,stride[0]\times d + k, stride[1]\times h + m, stride[2]\times w + n)
$$
* out tensor的shape推导公式 (默认ceilMode=false，即向下取整)：

$$
[N, C, D_{out}, H_{out}, W_{out}]=[N,C,\lfloor{\frac{D_{in}+2 \times {padding[0] - dilation[0] \times(kernelSize[0] - 1) - 1}}{stride[0]}}\rfloor + 1,\lfloor{\frac{H_{in}+2 \times {padding[1] - dilation[1] \times(kernelSize[1] - 1) - 1}}{stride[1]}}\rfloor + 1, \lfloor{\frac{W_{in}+2 \times {padding[2] - dilation[2] \times(kernelSize[2] - 1) - 1}}{stride[2]}}\rfloor + 1]
$$
* out tensor的shape推导公式 (默认ceilMode=true，即向上取整)：

$$
[N, C, D_{out}, H_{out}, W_{out}]=[N,C,\lceil{\frac{D_{in}+2 \times {padding[0] - dilation[0] \times(kernelSize[0] - 1) - 1}}{stride[0]}}\rceil + 1,\lceil{\frac{H_{in}+2 \times {padding[1] - dilation[1] \times(kernelSize[1] - 1) - 1}}{stride[1]}}\rceil + 1, \lceil{\frac{W_{in}+2 \times {padding[2] - dilation[2] \times(kernelSize[2] - 1) - 1}}{stride[2]}}\rceil + 1]
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.max_pool3d_with_argmax_v2()` 函数形式提供：


```python
def max_pool3d_with_argmax_v2(x: Tensor, ksize: List[int], strides: List[int], pads: List[int], dilation: List[int], ceil_mode: bool, data_format: str):
    """
    实现 max_pool3d_with_argmax_v2 操作。

    参数:
        x (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        ksize (List[int]): 池化窗口的大小，指定输出在DHW维度上的shape大小，数值必须大于0。
        strides (List[int]): 池化操作的步长，数值必须大于0。
        pads (List[int]): 在输入的D、H、W方向上padding补0的层数，数值在[0, ksize / 2]的范围内。
        dilation (List[int]): 表示控制窗口中元素的步幅。
        ceil_mode (bool): 推导出的输出shape是否向上取整。
        data_format (str): 输入数据格式，支持”NCDHW“。

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
x = torch.randn([1,1,2,2,2],dtype=torch.float32)
ksize = [2, 2, 2]
strides = [2, 2, 2]
pads = [0, 0, 0]
dilation = [1, 1, 1]
ceil_mode = False
data_format = "NCDHW"

out, indices = kernel_gen_ops.max_pool3d_with_argmax_v2(x, ksize, strides, pads, dilation, ceil_mode, data_format)
```
## 约束与限制

- 输入 ksize、strides 的数组长度为3，每个维度的值都要大于0。
- 输入 dilation 的数组长度为3，每个维度的值都必须为1。
- 输入 pads 的数组长度为3，每个维度的值需要在[0, ksize / 2]的范围内。
- 输出 out 的数据类型和输入 x 的数据类型一致。
- 输出 out 的shape需要与根据最大池化语义计算得到的shape结果一致。
- 输出 indices 的shape需要与输出 out 的shape一致。
- depth * height * weight需要小于等于int32的最大值。
