import torch
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """PyTorch CPU reference for WeightQuantBatchMatmulV2.

    Computation:
      dequant_weight = weight * antiquant_scale + antiquant_offset
      y = x @ dequant_weight + bias
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, weight, antiquant_scale, antiquant_offset,
                quant_scale, quant_offset, bias,
                transpose_x, transpose_weight, antiquant_group_size,
                dtype, inner_precise):
        x_f = x.float()
        w_f = weight.float()
        s_f = antiquant_scale.float()

        if antiquant_group_size > 0:
            k_size = weight.shape[1 if not transpose_weight else 0]
            n_groups = k_size // antiquant_group_size

            if not transpose_weight:
                w_grouped = w_f.reshape(n_groups, antiquant_group_size, -1)
                s_grouped = s_f.reshape(n_groups, antiquant_group_size, 1)
            else:
                w_grouped = w_f.reshape(-1, n_groups, antiquant_group_size).transpose(-2, -1)
                s_grouped = s_f.reshape(n_groups, antiquant_group_size, 1)

            if antiquant_offset is not None:
                o_grouped = antiquant_offset.float().reshape(n_groups, antiquant_group_size, 1)
            else:
                o_grouped = torch.zeros_like(s_grouped)

            dequant_w = (w_grouped * s_grouped + o_grouped).reshape(k_size, -1 if not transpose_weight else weight.shape[0])
        else:
            if antiquant_offset is not None:
                o_f = antiquant_offset.float()
            else:
                o_f = torch.zeros_like(s_f)
            dequant_w = w_f * s_f + o_f

        if transpose_x:
            x_mm = x_f.transpose(-2, -1)
        else:
            x_mm = x_f

        if transpose_weight:
            w_mm = dequant_w.transpose(-2, -1)
        else:
            w_mm = dequant_w

        out = torch.matmul(x_mm, w_mm)

        if bias is not None:
            out = out + bias.float()

        return out.to(torch.float16)


class ModelNew(nn.Module):
    """Custom kernel version using WeightQuantBatchMatmulV2."""

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x, weight, antiquant_scale, antiquant_offset,
                quant_scale, quant_offset, bias,
                transpose_x, transpose_weight, antiquant_group_size,
                dtype, inner_precise):
        result = kernel_gen_ops.weight_quant_batch_matmul_v2(
            x, weight, antiquant_scale, antiquant_offset,
            quant_scale, quant_offset, bias,
            transpose_x, transpose_weight, antiquant_group_size,
            dtype, inner_precise)
        return result
