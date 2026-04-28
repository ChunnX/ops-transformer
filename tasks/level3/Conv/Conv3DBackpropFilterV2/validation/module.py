import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import torch.nn.functional as F
import kernel_gen_ops


class Model(nn.Module):
    def __init__(self, stride, padding, dilation, groups):
        super(Model, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x, filter_size, out_backprop, filter_tensor):
        filter_tensor = filter_tensor.clone()
        filter_tensor.requires_grad = True
        
        input_tensor = x.permute(0, 4, 1, 2, 3).contiguous()
        input_tensor.requires_grad = True
        
        output = F.conv3d(input_tensor, filter_tensor, None, tuple(self.stride), tuple(self.padding), tuple(self.dilation), self.groups)
        output.backward(out_backprop.permute(0, 4, 1, 2, 3).contiguous())
        
        return filter_tensor.grad


class ModelNew(nn.Module):
    def __init__(self, stride, padding, dilation, groups):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x, filter_size, out_backprop, filter_tensor):
        result = kernel_gen_ops.conv3d_backprop_filter_v2(
            x,
            filter_size,
            out_backprop,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            "NDHWC",
            False,
            filter_tensor
        )
        return result
