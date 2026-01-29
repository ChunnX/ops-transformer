import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape = eval(param.get('shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    bins = int(param.get('bins', '1'))

    input = torch.randn(shape, device=device, dtype=dtype)
    while True:
        min = torch.randn([1], device=device, dtype=dtype)
        max = torch.randn([1], device=device, dtype=dtype)
        if min[0].item() < max[0].item():
            break

    return (input, min, max, bins)


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
