# AvgPool3DGrad

## 功能描述

### 算子功能
该算子是三维平均池化的反向传播，计算三维平均池化正向传播的输入梯度。

### 计算公式

$$
D_{out} = \left\lfloor \frac{D_{in} + 2 \times \text{pads}[0] - \text{ksize}[0]}{\text{strides}[0]} + 1 \right\rfloor
$$
$$
H_{out} = \left\lfloor \frac{H_{in} + 2 \times \text{pads}[1] - \text{ksize}[1]}{\text{strides}[1]} + 1 \right\rfloor
$$
$$
W_{out} = \left\lfloor \frac{W_{in} + 2 \times \text{pads}[2] - \text{ksize}[2]}{\text{strides}[2]} + 1 \right\rfloor
$$

若属性ceil_mode为 `True`，且满足条件 **$(D_{out} - 1) \times \text{stride}[0] \geq D_{in} + \text{padding}[0]$**，则会跳过最后一个窗口，这将导致维度 **$D_{out}$** 减少 1。此规则同样适用于维度 **$W_{out}$** 和 **$H_{out}$**。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.avg_pool3_d_grad()` 函数形式提供：


```python
def avg_pool3_d_grad(grad_output: Tensor, x: Tensor, ksize: List[int], strides: List[int], pads: List[int], ceil_mode: bool, count_include_pad: bool, divisor_override: int, data_format: str):
    """
    实现 avg_pool3_d_grad 操作。

    参数:
        grad_output (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
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
grad_output = torch.randn([1,16,1,1,1],dtype=torch.float32)
x = torch.randn([1,16,4,4,4],dtype=torch.float32)
ksize = [4, 4, 4]
strides = [1, 1, 1]
pads = [0, 0, 0]
ceil_mode = False
count_include_pad = False
divisor_override = 0

result = kernel_gen_ops.avg_pool3_d_grad(x, ksize, strides, pads, ceil_mode, count_include_pad, divisor_override)
```

## 约束与限制

- 输入 ksize、strides 的数组长度为3，每个维度的值都要大于0。
- 输入 pads 的数组长度为3，每个维度的值需要在[0, ksize / 2]的范围内。
- 输入 grad_output 的shape需要与根据平均池化语义计算得到的shape结果一致。
- 输出 out 的数据类型和输入 x 的数据类型一致。
- 输出 out 的shape需要与输入 x 的shape结果一致。
