import torch
import random

# Fix random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    dtype_str = param.get('dtype', 'float32')
    if dtype_str == 'bfloat16':
        dtype = torch.bfloat16
    else:
        dtype = getattr(torch, dtype_str)
    x_shape = eval(param.get('x_shape', '[5, 3]'))
    k = int(param.get('k', '3'))
    has_finished_optional = bool(param.get('has_finished_optional', 'False'))
    renorm = int(param.get('renorm', '0'))
    
    x = torch.randn(x_shape, device=device, dtype=dtype)
    
    if has_finished_optional:
        finished_optional = torch.randint(0, 2, (x_shape[0],), device=device, dtype=torch.bool)
    else:
        finished_optional = None
    
    return (x, finished_optional, k, renorm)


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for MoeGatingTopKSoftmaxV2.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed