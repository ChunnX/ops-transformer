from typing import List
import torch
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """使用 PyTorch 原生算子的参考实现（golden model）。"""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, grid: torch.Tensor,
                mode: str = 'bilinear', padding_mode: str = 'zeros',
                align_corners: bool = False) -> torch.Tensor:
        # PyTorch CPU grid_sample does not support float16/bfloat16,
        # so cast to float32 for computation
        input_dtype = x.dtype
        x_float = x.float()
        grid_float = grid.float()
        result = torch.nn.functional.grid_sample(x_float, grid_float, mode=mode,
                                                 padding_mode=padding_mode,
                                                 align_corners=align_corners)
        return result.to(input_dtype)


class ModelNew(nn.Module):
    """自定义 kernel 版本。"""
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor, grid: torch.Tensor,
                mode: str = 'bilinear', padding_mode: str = 'zeros',
                align_corners: bool = False) -> torch.Tensor:
        if x.dim() == 5:
            # Workaround: NPU 3D grid_sample kernel (MTE2ForNCHW) uses
            # inputH*inputW as N/C stride instead of inputD*inputH*inputW,
            # causing incorrect reads when N>1 or C>1 in NCDHW format.
            # Process per-batch per-channel (N=1,C=1 indexing is correct).
            N, C = x.shape[0], x.shape[1]
            if N == 1 and C == 1:
                return kernel_gen_ops.grid_sample(
                    x.contiguous(), grid, mode, padding_mode, align_corners)
            out_parts = []
            for n in range(N):
                ch_parts = []
                for c in range(C):
                    x_nc = x[n:n+1, c:c+1].contiguous()
                    g_n = grid[n:n+1].contiguous()
                    out_nc = kernel_gen_ops.grid_sample(
                        x_nc, g_n, mode, padding_mode, align_corners)
                    ch_parts.append(out_nc)
                out_parts.append(torch.cat(ch_parts, dim=1))
            return torch.cat(out_parts, dim=0)
        else:
            return kernel_gen_ops.grid_sample(x, grid, mode, padding_mode, align_corners)
