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

    def forward(self, input_size, filter_tensor, out_backprop, input_tensor):
        input_tensor = input_tensor.clone()
        input_tensor.requires_grad = True
        
        output = F.conv3d(input_tensor, filter_tensor, None, tuple(self.stride), tuple(self.padding), tuple(self.dilation), self.groups)
        output.backward(out_backprop)
        
        return input_tensor.grad


class ModelNew(nn.Module):
    def __init__(self, stride, padding, dilation, groups):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, input_size, filter_tensor, out_backprop, input_tensor):
        result = kernel_gen_ops.conv3d_backprop_input_v2(
            input_size,
            filter_tensor,
            out_backprop,
            input_tensor,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            "NCDHW",
            False
        )
        return result
