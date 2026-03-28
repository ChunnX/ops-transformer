import torch
import random
from framework.utils import check_precision


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    shape = eval(param.get('shape', '[1,1,1,1,1]'))
    x = torch.randn(shape, device=device, dtype=dtype)
    ksize = eval(param.get('ksize', '[1,1,1]'))
    strides = eval(param.get('strides', '[1,1,1]'))
    pads = eval(param.get('pads', '[0,0,0]'))
    ceil_mode = bool(param.get('ceil_mode', 'True'))
    count_include_pad = bool(param.get('count_include_pad', 'True'))
    divisor_override = int(param.get('divisor_override', '0'))

    return (x, ksize, strides, pads, ceil_mode, count_include_pad, divisor_override, "NCDHW")


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


def custom_check_precision(param, outputs, outputs_new):
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    if dtype == torch.float32:
        return check_precision(outputs, outputs_new, max_abs_error=0.0001, max_rel_error=0.0001)
    elif dtype == torch.float16:
        return check_precision(outputs, outputs_new, max_abs_error=0.001, max_rel_error=0.001)
    elif dtype == torch.bfloat16:
        return check_precision(outputs, outputs_new, max_abs_error=0.004, max_rel_error=0.004)
    else:
        return check_precision(outputs, outputs_new, max_abs_error=0.0001, max_rel_error=0.0001)
