# 参考 Rfft1D：Model 用 PyTorch 参考实现，ModelNew 用 kernel_gen_ops（内部 C++ 生成 plan）
from typing import Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        window: torch.Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
        normalized: bool,
        onesided: bool,
        return_complex: bool,
    ) -> torch.Tensor:
        return torch.stft(
            x,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=False,
            normalized=normalized,
            onesided=onesided,
            return_complex=return_complex,
        )


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        window: torch.Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
        normalized: bool,
        onesided: bool,
        return_complex: bool,
    ) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.stft(
            x, window, n_fft, hop_length, win_length, normalized, onesided, return_complex
        )
