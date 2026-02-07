# AvgPool3D

## 功能描述

### 算子功能
该算子对输入Tensor进行窗口为$kD * kH * kW$、步长为$sD * sH * sW$的三维平均池化操作，其中$k$为kernelSize，表示池化窗口的大小，$s$为stride，表示池化操作的步长。

### 计算公式

输入input($N,C,D_{in},H_{in},W_{in}$)、输出out($N,C,D_{out},H_{out},W_{out}$)和池化步长($stride$)、池化窗口大小kernelSize($kD,kH,kW$)的关系是

$$
D_{out}=\lfloor \frac{D_{in}+2*padding[0]-kernelSize[0]}{stride[0]}+1 \rfloor
$$

$$
H_{out}=\lfloor \frac{H_{in}+2*padding[1]-kernelSize[1]}{stride[1]}+1 \rfloor
$$

$$
W_{out}=\lfloor \frac{W_{in}+2*padding[2]-kernelSize[2]}{stride[2]}+1 \rfloor
$$

$$
out(N_i,C_i,d,h,w)=\frac{1}{kD*kH*kW}\sum_{k=0}^{kD-1}\sum_{m=0}^{kH-1}\sum_{n=0}^{kW-1}input(N_i,C_i,stride[0]*d+k,stride[1]*h+m,stride[2]*w+n)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.avg_pool3_d()` 函数形式提供：


```python
def avg_pool3_d(x: Tensor, ksize: List[int], strides: List[int], pads: List[int], ceil_mode: bool, count_include_pad: bool, divisor_override: int, data_format: str):
    """
    实现 avg_pool3_d 操作。

    参数:
        x (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        ksize (List[int]): 池化窗口的大小，指定输出在DHW维度上的shape大小，数值必须大于0。
        strides (List[int]): 池化操作的步长，数值必须大于0。
        pads (List[int]): 在输入的D、H、W方向上padding补0的层数，数值在[0, ksize / 2]的范围内。
        ceil_mode (bool): 推导出的输出shape是否向上取整。
        count_include_pad (bool): 计算平均池化时是否包括padding填充的0。
        divisor_override (int): 取平均的除数，当值为0时，该属性不生效。
        data_format (str): 输入数据格式，支持”NCDHW“。

    返回:
        Tensor: 输出张量，数据类型与输入一致，数据格式保持不变。
    """

```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建输入张量
x = torch.randn([1,16,4,4,4],dtype=torch.float32)
ksize = [4, 4, 4]
strides = [1, 1, 1]
pads = [0, 0, 0]
ceil_mode = False
count_include_pad = True
divisor_override = 0
data_format = "NCDHW"

result = kernel_gen_ops.avg_pool3_d(x, ksize, strides, pads, ceil_mode, count_include_pad, divisor_override, data_format)
```
## 约束与限制

- 输入 ksize、strides 的数组长度为3，每个维度的值都要大于0。
- 输入 pads 的数组长度为3，每个维度的值需要在[0, ksize / 2]的范围内。
- 输出 out 的数据类型和输入 x 的数据类型一致。
- 输出 out 的shape需要与根据平均池化语义计算得到的shape结果一致。
