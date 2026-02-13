import torch
import random


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    grad_shape = eval(param.get('grad_shape', '[1,1,1,1,1]'))
    grad = torch.randn(grad_shape, device=device, dtype=dtype)
    shape = eval(param.get('shape', '[1,1,1,1,1]'))
    x = torch.randn(shape, device=device, dtype=dtype)
    ksize = eval(param.get('ksize', '[1,1,1]'))
    strides = eval(param.get('strides', '[1,1,1]'))
    pads = eval(param.get('pads', '[0,0,0]'))
    dilation = eval(param.get('dilation', '[1,1,1]'))
    ceil_mode = bool(param.get('ceil_mode', 'True'))

    m = torch.nn.MaxPool3d(kernel_size=ksize, stride=strides, padding=pads,
                           dilation=dilation, return_indices=True, ceil_mode=ceil_mode)
    out, indices = m(x.to(torch.float32))


    return (x, grad, indices.to(torch.int32), ksize, strides, pads, dilation, ceil_mode)


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for sinh.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed
