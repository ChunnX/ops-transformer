import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1, x2, bias=None, scale=None,
                perm_x1=None, perm_x2=None, perm_y=None,
                batch_split_factor=1):
        if perm_x1 is None:
            perm_x1 = [0, 1, 2]
        if perm_x2 is None:
            perm_x2 = [0, 1, 2]
        if perm_y is None:
            perm_y = [1, 0, 2]

        # Transpose inputs according to perm
        x1_t = x1.permute(*perm_x1)  # (batch, m, k)
        x2_t = x2.permute(*perm_x2)  # (batch, k, n)

        result = torch.bmm(x1_t, x2_t)  # (batch, m, n)

        if bias is not None:
            result = result + bias

        # Apply scale dequant if present
        if scale is not None:
            # scale shape: (batch * n,), result shape: (batch, m, n)
            result = result.to(torch.float32) * scale.reshape(-1).float().unsqueeze(0).unsqueeze(0)
            result = result.to(x1.dtype)

        # Transpose output according to perm_y
        result = result.permute(*perm_y)  # e.g. (m, batch, n)

        # Apply batch_split_factor
        if batch_split_factor > 1:
            m_dim = result.shape[0]
            batch_n = result.shape[1] * result.shape[2]
            result = result.reshape(batch_split_factor, m_dim, batch_n // batch_split_factor)

        return result


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x1, x2, bias=None, scale=None,
                perm_x1=None, perm_x2=None, perm_y=None,
                batch_split_factor=1):
        if perm_x1 is None:
            perm_x1 = [0, 1, 2]
        if perm_x2 is None:
            perm_x2 = [0, 1, 2]
        if perm_y is None:
            perm_y = [1, 0, 2]

        enable_hf32 = False

        result = kernel_gen_ops.transpose_batch_mat_mul(
            x1, x2, bias, scale,
            perm_x1, perm_x2, perm_y,
            enable_hf32, batch_split_factor
        )
        return result
