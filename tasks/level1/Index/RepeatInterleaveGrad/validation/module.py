import torch
import torch.nn as nn
import kernel_gen_ops

class Model(nn.Module):
    """
    实现RepeatInterleaveGrad算子功能的标杆模型。
    对应 torch.repeat_interleave 的反向逻辑。
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, y_grad: torch.Tensor, repeats: torch.Tensor, axis: int) -> torch.Tensor:
        """
        Args:
            y_grad: 输入梯度张量，形状为 repeat_interleave 后的形状
            repeats: 重复次数张量 (1D Tensor) 或 标量张量
            axis: 维度

        Returns:
            x_grad: 输出梯度张量，形状为 repeat_interleave 前的形状
        """
        if axis < 0:
            axis += y_grad.ndim
            
        repeats_cpu = repeats.cpu()
        if repeats_cpu.numel() == 1:
            r = int(repeats_cpu.item())
            if r == 0:
                return torch.zeros_like(y_grad)
            return y_grad.unfold(axis, r, r).sum(dim=-1)
        else:
            split_sizes = repeats_cpu.tolist()
            chunks = torch.split(y_grad, split_sizes, dim=axis)
            sums = [chunk.sum(dim=axis, keepdim=True) for chunk in chunks]
            return torch.cat(sums, dim=axis)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, y_grad: torch.Tensor, repeats: torch.Tensor, axis: int) -> torch.Tensor:
        # 调用算子
        return kernel_gen_ops.repeat_interleave_grad(y_grad, repeats, axis)