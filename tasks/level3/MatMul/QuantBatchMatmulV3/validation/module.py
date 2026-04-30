import torch
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """PyTorch CPU reference for QuantBatchMatmulV3."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1, x2, scale, offset, bias, pertoken_scale,
                dtype, transpose_x1, transpose_x2, group_size):
        # API formula (api_desc.md):
        #   no bias:        out = (x1 @ x2) * scale + offset
        #   INT32 bias:     out = ((x1 @ x2) + bias) * scale + offset
        #   BF16/FP32 bias: out = (x1 @ x2) * scale + bias

        # Transpose if needed (before matmul)
        x1_mm = x1.transpose(-2, -1) if transpose_x1 else x1
        x2_mm = x2.transpose(-2, -1) if transpose_x2 else x2

        # Integer matmul for exact accumulation (int8 @ int8 -> int32)
        out_int = torch.matmul(x1_mm.to(torch.int32), x2_mm.to(torch.int32))

        # Build broadcast shape for per-channel scale/offset
        ndim = out_int.dim()
        last_dim_shape = [1] * (ndim - 1) + [-1]

        if bias is not None:
            if bias.dtype == torch.int32:
                # INT32 bias added to int32 matmul result, then dequant+offset
                out_int = out_int + bias.to(torch.int32)
                out = out_int.to(torch.float32) * scale.to(torch.float32).reshape(last_dim_shape)
                if offset is not None:
                    out = out + offset.to(torch.float32).reshape(last_dim_shape)
            else:
                # BF16/FP32 bias: replaces offset
                out = out_int.to(torch.float32) * scale.to(torch.float32).reshape(last_dim_shape)
                out = out + bias.to(torch.float32)
        else:
            out = out_int.to(torch.float32) * scale.to(torch.float32).reshape(last_dim_shape)
            if offset is not None:
                out = out + offset.to(torch.float32).reshape(last_dim_shape)

        if pertoken_scale is not None:
            out = out * pertoken_scale.to(torch.float32)

        return out.to(torch.bfloat16)


class ModelNew(nn.Module):
    """Custom kernel version using QuantBatchMatmulV3."""

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x1, x2, scale, offset, bias, pertoken_scale,
                dtype, transpose_x1, transpose_x2, group_size):
        result = kernel_gen_ops.quant_batch_matmul_v3(
            x1, x2, scale, offset, bias, pertoken_scale,
            dtype, transpose_x1, transpose_x2, group_size)
        return result
