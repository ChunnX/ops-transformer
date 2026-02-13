import torch
import random


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    grad_out_shape = eval(param.get('grad_out_shape', '[1]'))
    shape = eval(param.get('shape', '[1,1,1]'))
    targets_shape = eval(param.get('targets_shape', '[1,1]'))
    targets_range = eval(param.get('targets_range', '[1,1]'))
    input_lengths = eval(param.get('input_lengths', '[1]'))
    target_lengths = eval(param.get('target_lengths', '[1]'))
    neg_shape = eval(param.get('neg_shape', '[1,1]'))
    alpha_shape = eval(param.get('alpha_shape', '[1,1]'))
    blank = int(param.get('blank', '0'))
    zero_infinity = bool(param.get('zero_infinity', 'False'))

    grad_out = torch.randn(grad_out_shape, device=device, dtype=dtype)
    log_probs = torch.nn.functional.log_softmax(torch.randn(shape, device=device, dtype=dtype), -1)
    targets = torch.randint(targets_range[0], targets_range[1], targets_shape, device=device, dtype=torch.int32)
    neg = torch.randn(neg_shape, device=device, dtype=dtype)
    alpha = torch.randn(alpha_shape, device=device, dtype=dtype)

    return (grad_out, log_probs, targets, input_lengths, target_lengths, neg, alpha, blank, zero_infinity)


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
