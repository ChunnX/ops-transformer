import torch
import torch_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """PyTorch 原生算子参考实现（golden model）。

    必须使用 PyTorch 标准 API 实现与自定义算子相同的数学语义。
    如果算子有精度提升逻辑（如 int16 -> float32 中间计算），
    Model 必须复制相同的逻辑。
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        # TODO: 使用 PyTorch 原生 API 实现等价计算
        return x


class ModelNew(nn.Module):
    """自定义 kernel 版本。

    forward() 签名必须与 Model.forward() 完全相同。
    推导参数在内部计算后传入 kernel_gen_ops。
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        # TODO: 计算推导参数，然后调用 kernel_gen_ops.<kernel_name>(...)
        return kernel_gen_ops.<kernel_name>(x)
