import torch
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """PyTorch CPU reference for QuantBatchMatmulV4.

    Per-block quantization (perblock strategy):
      out[i,j] = sum_g(x1_scale[i,g] * x2_scale[j//group_n, g] * dot(x1[i,g*128:(g+1)*128], x2[j,g*128:(g+1)*128])) + bias[j]

    Simple per-tensor/per-channel:
      out = x1 @ x2 * scales + bias
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1, x2, bias, x1_scale, x2_scale, y_scale,
                x1_offset, x2_offset, y_offset, x2_table,
                dtype, compute_type, transpose_x1, transpose_x2, group_size):
        M, K = x1.shape
        N = x2.shape[0] if transpose_x2 else x2.shape[1]

        group_k = group_size & 0xFFFF
        group_n = (group_size >> 16) & 0xFFFF

        if group_k > 0 and group_n > 0 and x1_scale is not None and x1_scale.dim() == 2:
            # Per-block quantization
            n_groups_k = K // group_k

            x1_blocks = x1.reshape(M, n_groups_k, group_k).float()

            if transpose_x2:
                x2_blocks = x2.reshape(N, n_groups_k, group_k).float()
            else:
                # x2=[K,N], transpose for matmul then reshape
                x2_t = x2.transpose(-2, -1)  # [N, K]
                x2_blocks = x2_t.reshape(N, n_groups_k, group_k).float()

            # Block-wise dot products [M, N, n_groups_k]
            partial = torch.einsum('mgk,ngk->mng', x1_blocks, x2_blocks)

            # x1_scale: [M, n_groups_k], x2_scale: [n_groups_n, n_groups_k]
            x1_s = x1_scale.float().unsqueeze(1)  # [M, 1, n_groups_k]
            j_groups = torch.arange(N, device=x1.device) // group_n
            x2_s_expanded = x2_scale.float()[j_groups]  # [N, n_groups_k]
            x2_s_expanded = x2_s_expanded.unsqueeze(0)  # [1, N, n_groups_k]

            out = (partial * x1_s * x2_s_expanded).sum(dim=-1)  # [M, N]
        else:
            # Simple per-tensor/per-channel scaling
            x1_mm = x1.transpose(-2, -1) if transpose_x1 else x1
            x2_mm = x2.transpose(-2, -1) if transpose_x2 else x2

            out = torch.matmul(x1_mm.float(), x2_mm.float())

            ndim = out.dim()
            last_dim_shape = [1] * (ndim - 1) + [-1]

            if x2_scale is not None:
                scale = x2_scale.float()
                if scale.dim() < ndim:
                    scale = scale.reshape(last_dim_shape)
                out = out * scale
            if x1_scale is not None:
                scale = x1_scale.float()
                if scale.dim() < ndim:
                    scale = scale.reshape(last_dim_shape)
                out = out * scale

        if bias is not None:
            out = out + bias.float().reshape(1, -1)

        return out.to(torch.bfloat16)


class ModelNew(nn.Module):
    """Custom kernel version using QuantBatchMatmulV4."""

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x1, x2, bias, x1_scale, x2_scale, y_scale,
                x1_offset, x2_offset, y_offset, x2_table,
                dtype, compute_type, transpose_x1, transpose_x2, group_size):
        result = kernel_gen_ops.quant_batch_matmul_v4(
            x1, x2, bias, x1_scale, x2_scale, y_scale,
            x1_offset, x2_offset, y_offset, x2_table,
            dtype, compute_type, transpose_x1, transpose_x2, group_size)
        return result
