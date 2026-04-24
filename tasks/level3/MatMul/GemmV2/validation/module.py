import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, a, b, alpha, beta, c, transpose_a=False, transpose_b=False):
        a_mat = a.transpose(-2, -1) if transpose_a else a
        b_mat = b.transpose(-2, -1) if transpose_b else b
        matmul_out = torch.matmul(a_mat.float(), b_mat.float())
        alpha_scale = alpha.float().reshape(-1, 1)
        beta_scale = beta.float().reshape(-1, 1)
        return alpha_scale * matmul_out + beta_scale * c.float()


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, a, b, alpha, beta, c, transpose_a=False, transpose_b=False):
        return kernel_gen_ops.gemm_v2(a, b, alpha, beta, c, transpose_a, transpose_b)
