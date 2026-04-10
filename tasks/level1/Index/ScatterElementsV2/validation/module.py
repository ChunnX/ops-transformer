import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, var, indices, updates, axis=-1, reduction="none"):
        result = var.clone()
        if reduction == "none":
            result.scatter_(axis, indices.to(torch.int64), updates)
        elif reduction == "add":
            result.scatter_add_(axis, indices.to(torch.int64), updates)
        return result


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, var, indices, updates, axis=-1, reduction="none"):
        return kernel_gen_ops.scatter_elements_v2(var, indices, updates, axis, reduction)
