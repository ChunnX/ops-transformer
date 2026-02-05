from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    Simple model that performs abs operation.
    """

    def __init__(self):
        """
        Initialize the model for abs operation.
        No parameters needed for basic abs operation.
        """
        super(Model, self).__init__()

    def forward(self, x: List[torch.Tensor], scalars: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Computes the abs of input elements.

        Args:
            inputs: Input tensor list of any shape.

        Returns:
            Output tensor of same shape as input with abs applied elementwise.
        """
        res = torch._foreach_pow(x, scalars)
        res_list = []
        for i in range(len(res)):
            t = res[i]
            t = torch.where(torch.isinf(t) & (t > 0), torch.tensor(1, dtype=t.dtype, device=t.device), t).to(x[0].dtype)
            t = torch.where(torch.isinf(t) & (t < 0), torch.tensor(-1, dtype=t.dtype, device=t.device), t).to(x[0].dtype)
            t = torch.where(torch.isnan(t), torch.tensor(0, dtype=t.dtype, device=t.device), t).to(x[0].dtype)
            res_list.append(t)
        return res_list


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: List[torch.Tensor], scalars: List[torch.Tensor]) -> List[torch.Tensor]:
        res = kernel_gen_ops.foreach_pow_scalar_list(x, scalars)
        res_list = []
        for i in range(len(res)):
            t = res[i]
            t = torch.where(torch.isinf(t) & (t > 0), torch.tensor(1, dtype=t.dtype, device=t.device), t)
            t = torch.where(torch.isinf(t) & (t < 0), torch.tensor(-1, dtype=t.dtype, device=t.device), t)
            t = torch.where(torch.isnan(t), torch.tensor(0, dtype=t.dtype, device=t.device), t)
            res_list.append(t)
        return res_list
