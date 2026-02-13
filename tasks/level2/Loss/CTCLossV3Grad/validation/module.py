from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad: torch.Tensor, log_probs: torch.Tensor, targets: torch.Tensor, input_lengths: List[int],
                target_lengths: List[int], neg_log_likelihood: torch.Tensor, log_aplha: torch.Tensor, blank: int,
                zero_infinity: bool):
        res = torch.ops.aten._ctc_loss_backward(grad.npu(), log_probs.npu(),
                    targets.type(torch.int64).npu(), input_lengths, target_lengths, neg_log_likelihood.npu(), log_aplha.npu(), blank, zero_infinity)
        return res.cpu()

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, grad: torch.Tensor, log_probs: torch.Tensor, targets: torch.Tensor, input_lengths: List[int],
                target_lengths: List[int], neg_log_likelihood: torch.Tensor, log_aplha: torch.Tensor, blank: int,
                zero_infinity: bool):
        import kernel_gen_ops
        return kernel_gen_ops.ctc_loss_v3_grad(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_aplha, blank, zero_infinity)