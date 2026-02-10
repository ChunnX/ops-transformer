import torch
import random


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    shape = eval(param.get('shape', '[1,1,1]'))
    targets_shape = eval(param.get('targets_shape', '[1,1]'))
    targets_range = eval(param.get('targets_range', '[1,1]'))
    input_lengths = eval(param.get('input_lengths', '[1]'))
    target_lengths = eval(param.get('target_lengths', '[1]'))
    blank = int(param.get('blank', '0'))
    zero_infinity = bool(param.get('zero_infinity', 'False'))

    log_probs = torch.nn.functional.log_softmax(torch.randn(shape, device=device, dtype=dtype), -1)
    targets = torch.randint(targets_range[0], targets_range[1], targets_shape, device=device, dtype=torch.int32)

    return (log_probs, targets, input_lengths, target_lengths, blank, zero_infinity)


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
