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
    
    x_shape = eval(param.get('x_shape', '[3, 4]'))
    indices_shape = eval(param.get('indices_shape', '[3, 2]'))
    probs_shape = eval(param.get('probs_shape', '[3, 2]'))
    range_vals = eval(param.get('range', '[1, 5]'))
    num_token_out = int(param.get('num_token_out', '6'))
    pad_mode = bool(param.get('pad_mode', 'False'))
    
    # Generate input tensors
    x = torch.randn(x_shape, device=device, dtype=dtype)
    indices = torch.randint(0, x_shape[1], indices_shape, device=device, dtype=torch.int32)
    probs = torch.rand(probs_shape, device=device, dtype=dtype)
    
    return (x, indices, probs, range_vals, num_token_out, pad_mode)


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for MoeTokenPermuteWithEp.
    
    Args:
        param (dict): Parameters from a pandas DataFrame row
    
    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed