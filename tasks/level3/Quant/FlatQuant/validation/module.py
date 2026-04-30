import torch
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """PyTorch reference for FlatQuant.

    Computes y[k] = p1 @ x[k] @ p2^T on NPU to match the kernel's cube engine precision,
    then quantizes on CPU.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, kronecker_p1, kronecker_p2, clip_ratio, dst_dtype):
        # x: [K, M, N], p1: [M, M], p2: [N, N]
        K, M, N = x.shape

        # Compute matmul on NPU to match kernel's cube engine precision
        x_npu = x.npu()
        p1_npu = kronecker_p1.npu()
        p2_npu = kronecker_p2.npu()

        y = torch.matmul(x_npu, p2_npu.t())  # [K, M, N]
        y = torch.matmul(p1_npu.unsqueeze(0).expand(K, -1, -1), y)

        # Move back to CPU for quantization
        y = y.cpu().float()

        # Compute per-batch max absolute value
        max_val = y.reshape(K, -1).abs().max(dim=1).values  # [K]

        # Compute quant_scale (matches kernel: max_val * clip_ratio / 7.0)
        quant_scale = max_val * clip_ratio / 7.0  # [K]

        # Compute multiplier (matches kernel: 7.0 / (clip_ratio * max_val), or 7.0 if max_val == 0)
        multiplier = torch.where(
            max_val != 0,
            7.0 / (clip_ratio * max_val),
            torch.ones_like(max_val) * 7.0
        )  # [K]

        # Quantize: round(y * multiplier), clamped to [-7, 7]
        quantized = (y * multiplier.reshape(K, 1, 1)).round().clamp(-7, 7).to(torch.int8)

        return quantized, quant_scale


class ModelNew(nn.Module):
    """Custom kernel version using FlatQuant."""

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x, kronecker_p1, kronecker_p2, clip_ratio, dst_dtype):
        return kernel_gen_ops.flat_quant(x, kronecker_p1, kronecker_p2, clip_ratio, dst_dtype)
