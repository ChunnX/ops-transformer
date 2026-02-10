from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor, input_lengths: List[int], target_lengths: List[int], blank: int, zero_infinity: bool):
        neg_log_likelihood, log_alpha = torch.ops.aten._ctc_loss(log_probs.npu(), targets.npu(), input_lengths, target_lengths, blank=blank, zero_infinity=zero_infinity)
        res = [neg_log_likelihood.cpu(), log_alpha.cpu()]
        res_list = []
        for i in range(len(res)):
            t = res[i]
            t = torch.where(torch.isinf(t) & (t > 0), torch.tensor(1, dtype=t.dtype, device=t.device), t).to(log_probs.dtype)
            t = torch.where(torch.isinf(t) & (t < 0), torch.tensor(-1, dtype=t.dtype, device=t.device), t).to(log_probs.dtype)
            t = torch.where(torch.isnan(t), torch.tensor(0, dtype=t.dtype, device=t.device), t).to(log_probs.dtype)
            res_list.append(t)
        return res_list

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor, input_lengths: List[int], target_lengths: List[int], blank: int, zero_infinity: bool):
        import kernel_gen_ops
        neg_log_likelihood, log_alpha = kernel_gen_ops.ctc_loss_v3(log_probs, targets, input_lengths, target_lengths, blank, zero_infinity)
        res = [neg_log_likelihood.cpu(), log_alpha.cpu()]
        res_list = []
        for i in range(len(res)):
            t = res[i]
            t = torch.where(torch.isinf(t) & (t > 0), torch.tensor(1, dtype=t.dtype, device=t.device), t).to(log_probs.dtype)
            t = torch.where(torch.isinf(t) & (t < 0), torch.tensor(-1, dtype=t.dtype, device=t.device), t).to(log_probs.dtype)
            t = torch.where(torch.isnan(t), torch.tensor(0, dtype=t.dtype, device=t.device), t).to(log_probs.dtype)
            res_list.append(t)
        return res_list