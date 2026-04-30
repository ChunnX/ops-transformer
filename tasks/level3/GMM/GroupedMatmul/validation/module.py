import torch
import torch_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, weight, bias, scale, offset, antiquant_scale,
                antiquant_offset, group_list, per_token_scale,
                split_item, group_type, group_list_type, act_type, tuning_config):
        # Basic non-quantized matmul: Y = X @ W^T
        y = torch.matmul(x.to(torch.float32), weight.to(torch.float32).t()).to(x.dtype)
        if bias is not None and bias.numel() > 0:
            y = y + bias.to(y.dtype)
        return y


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x, weight, bias, scale, offset, antiquant_scale,
                antiquant_offset, group_list, per_token_scale,
                split_item, group_type, group_list_type, act_type, tuning_config):
        result = kernel_gen_ops.grouped_matmul(
            x, weight, bias, scale, offset, antiquant_scale,
            antiquant_offset, group_list, per_token_scale,
            split_item, group_type, group_list_type, act_type, tuning_config)
        return result
