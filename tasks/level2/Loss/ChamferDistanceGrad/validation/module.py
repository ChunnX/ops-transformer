import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, xyz1, xyz2, idx1, idx2, gradDist1, gradDist2):
        """
        Args:
            xyz1: [B, N, 2] - 点集1坐标
            xyz2: [B, N, 2] - 点集2坐标
            idx1: [B, N] - 每个xyz1点对应xyz2中的最近点索引（int32）
            idx2: [B, N] - 每个xyz2点对应xyz1中的最近点索引（int32）
            gradDist1: [B, N] - 从xyz1到最近xyz2点的距离梯度
            gradDist2: [B, N] - 从xyz2到最近xyz1点的距离梯度
        Returns:
            gradXyz1: [B, N, 2] - xyz1的梯度
            gradXyz2: [B, N, 2] - xyz2的梯度
        """
        B, N, _ = xyz1.shape
        _, M, _ = xyz2.shape

        # 初始化梯度为0
        gradXyz1 = torch.zeros_like(xyz1)
        gradXyz2 = torch.zeros_like(xyz2)

        # 遍历 batch 和点
        for b in range(B):
            for n in range(N):
                # 获取当前点在xyz1中的坐标
                x1, y1 = xyz1[b, n, 0].item(), xyz1[b, n, 1].item()
                # 获取对应xyz2中的点索引
                idx = idx1[b, n].item()
                # 获取该点在xyz2中的坐标
                x2, y2 = xyz2[b, idx, 0].item(), xyz2[b, idx, 1].item()
                # 获取梯度
                g = gradDist1[b, n].item() * 2.0

                # 累加梯度
                gradXyz1[b, n, 0] += (x1 - x2) * g
                gradXyz1[b, n, 1] += (y1 - y2) * g
                gradXyz2[b, idx, 0] -= (x1 - x2) * g
                gradXyz2[b, idx, 1] -= (y1 - y2) * g

            for m in range(M):
                # 获取当前点在xyz2中的坐标
                x2, y2 = xyz2[b, m, 0].item(), xyz2[b, m, 1].item()
                # 获取对应xyz1中的点索引
                idx = idx2[b, m].item()
                # 获取该点在xyz1中的坐标
                x1, y1 = xyz1[b, idx, 0].item(), xyz1[b, idx, 1].item()
                # 获取梯度
                g = gradDist2[b, m].item() * 2.0

                # 累加梯度
                gradXyz2[b, m, 0] += (x2 - x1) * g
                gradXyz2[b, m, 1] += (y2 - y1) * g
                gradXyz1[b, idx, 0] -= (x2 - x1) * g
                gradXyz1[b, idx, 1] -= (y2 - y1) * g

        return [gradXyz1, gradXyz2]


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor, 
                idx1: torch.Tensor, idx2: torch.Tensor, 
                gradDist1: torch.Tensor, gradDist2: torch.Tensor):
        """
        Args:
            xyz1: 点集1的坐标张量，形状为 [B, N, 2]，表示 B 个批次中每个批次包含 N 个二维点，数据类型支持 float32 或 float16；
            xyz2: 点集2的坐标张量，形状为 [B, M, 2]，表示 B 个批次中每个批次包含 M 个二维点，数据类型与 xyz1 一致；
            idx1: 正向传播中，点集1中每个点到点集2的最近点的索引，形状为 [B, N]，数据类型为 int32；
            idx2: 正向传播中，点集2中每个点到点集1的最近点的索引，形状为 [B, M]，数据类型为 int32；
            gradDist1: 正向输出 dist1 的梯度（即上游反向梯度），形状为 [B, N]，与 xyz1、xyz2 数据类型一致；
            gradDist2: 正向输出 dist2 的梯度（即上游反向梯度），形状为 [B, M]，与 xyz1、xyz2 数据类型一致；

        Returns:
            grad_xyz1: 输入 xyz1 对应的梯度，形状为 [B, N, 2]，表示每个点的坐标梯度，用于反向传播更新；
            grad_xyz2: 输入 xyz2 对应的梯度，形状为 [B, M, 2]，表示每个点的坐标梯度，用于反向传播更新；
        """
        import kernel_gen_ops
        return kernel_gen_ops.chamfer_distance_grad(xyz1, xyz2, idx1, idx2, gradDist1, gradDist2)
