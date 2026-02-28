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

    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Computes the abs of input elements.

        Args:
            inputs: Input tensor list of any shape.

        Returns:
            Output tensor of same shape as input with abs applied elementwise.
        """
        if x1[0].dtype in [torch.float16, torch.bfloat16]:
            x1_new = []
            x2_new = []
            for i in range(len(x1)):
                x1_new.append(x1[i].to(torch.float32))
                x2_new.append(x2[i].to(torch.float32))
            res = torch._foreach_pow(x1_new, x2_new)
            return [r.to(x1[0].dtype) for r in res]
        else:
            res = torch._foreach_pow(x1, x2)
            return [r for r in res]


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor]) -> List[torch.Tensor]:
        return kernel_gen_ops.foreach_pow_list(x1, x2)
