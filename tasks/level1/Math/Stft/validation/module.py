# 与标杆对齐：参考 ascend_function_torch_stft_bm / ascend_function_torch_stft
# - center=False（与 aclnn 行为一致）；complex 输入时 onesided=False；float16 上转 float32 避免 CPU 报错
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
        # 与标杆 TorchStftApi 一致：complex 输入时 onesided=False
        if x.dtype == torch.complex64 or x.dtype == torch.complex128:
            onesided = False
        # CPU torch.stft 对 float16 会报 "expected scalar type Double but found Half"，先上转到 float32
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
            window = window.float()
        # 与标杆一致：center=False（BM 与 aclnn 对齐）
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
