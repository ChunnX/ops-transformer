# module.py
from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


def _group_norm_grad_ref(dy, x, mean, rstd, gamma, num_groups):
    """PyTorch reference: GroupNorm backward. Returns dx, dgamma, dbeta as list."""
    dtype_orig = x.dtype
    dy_hp = dy.to(torch.float32)
    mean_hp = mean.to(torch.float32)
    rstd_hp = rstd.to(torch.float32)
    x_hp = x.to(torch.float32)
    gamma_hp = gamma.to(torch.float32)

    batch_num, num_channels = x_hp.size(0), x_hp.size(1)
    remaining = x_hp.size()[2:]
    hw = 1
    for s in remaining:
        hw *= s
    num_per_group_channel = num_channels // num_groups
    num_per_group_total = float(num_per_group_channel * hw)

    x_reshaped = x_hp.reshape((batch_num, num_channels, hw))
    dy_reshaped = dy_hp.reshape((batch_num, num_channels, hw))

    dgamma_sum = torch.zeros_like(gamma_hp)
    dbeta_sum = torch.zeros_like(gamma_hp)
    dx_out = torch.zeros_like(x_reshaped)

    for n_i in range(batch_num):
        for g_i in range(num_groups):
            ch_start = g_i * num_per_group_channel
            ch_end = (g_i + 1) * num_per_group_channel
            x_g = x_reshaped[n_i, ch_start:ch_end, :]
            dy_g = dy_reshaped[n_i, ch_start:ch_end, :]
            mean_x = mean_hp[n_i, g_i]
            rstd_x = rstd_hp[n_i, g_i]
            x_norm = (x_g - mean_x) * rstd_x
            gamma_g = gamma_hp[ch_start:ch_end].view(num_per_group_channel, 1)

            temp_1 = torch.sum(dy_g, dim=1)
            temp_2 = torch.sum(dy_g * x_norm, dim=1)
            dbeta_sum[ch_start:ch_end] += temp_1
            dgamma_sum[ch_start:ch_end] += temp_2

            c1 = torch.sum(temp_1 * gamma_g.squeeze(1)) / num_per_group_total
            c2 = torch.sum(temp_2 * gamma_g.squeeze(1)) / num_per_group_total
            dx_g = rstd_x * (dy_g * gamma_g - x_norm * c2 - c1)
            dx_out[n_i, ch_start:ch_end, :] = dx_g

    dx_out = dx_out.reshape(x_hp.shape)
    return [
        dx_out.to(dtype_orig),
        dgamma_sum.to(dtype_orig),
        dbeta_sum.to(dtype_orig),
    ]


class Model(nn.Module):
    def __init__(self, num_groups: int, data_format: str):
        super(Model, self).__init__()
        self.num_groups = num_groups
        self.data_format = data_format

    def forward(
        self,
        dy: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
        x: torch.Tensor,
        gamma: torch.Tensor,
        dx_is_require: bool,
        dgamma_is_require: bool,
        dbeta_is_require: bool,
    ) -> List[torch.Tensor]:
        dx, dgamma, dbeta = _group_norm_grad_ref(dy, x, mean, rstd, gamma, self.num_groups)
        out = []
        if dx_is_require:
            out.append(dx)
        if dgamma_is_require:
            out.append(dgamma)
        if dbeta_is_require:
            out.append(dbeta)
        return out


class ModelNew(nn.Module):
    def __init__(self, num_groups: int, data_format: str):
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.data_format = data_format

    def forward(
        self,
        dy: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
        x: torch.Tensor,
        gamma: torch.Tensor,
        dx_is_require: bool,
        dgamma_is_require: bool,
        dbeta_is_require: bool,
    ) -> List[torch.Tensor]:
        outputs = kernel_gen_ops.group_norm_grad(
            dy, mean, rstd, x, gamma,
            self.num_groups,
            self.data_format,
            dx_is_require,
            dgamma_is_require,
            dbeta_is_require,
        )
        return list(outputs)
