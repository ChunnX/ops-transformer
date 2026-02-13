import torch
import numpy as np

def get_inputs(param, device=None):
    """
    Generate input tensors for the GroupNormSilu operator.
    Inputs: x, gamma, beta; attrs: num_groups, eps, activate_silu.
    """
    x_shape = eval(param.get('x_shape', '[2, 32, 4, 4]'))
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    num_channels = x_shape[1]
    num_groups = int(param.get('num_groups', 8))

    x = (torch.rand(x_shape, device=device, dtype=dtype) * 2.0 - 1.0) * 0.1
    gamma = torch.rand(num_channels, device=device, dtype=dtype) * 0.1 + 0.5
    beta = torch.rand(num_channels, device=device, dtype=dtype) * 0.1

    return (x, gamma, beta)


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the GroupNormSilu model from DataFrame row.
    """
    num_channels = eval(param.get('x_shape', '[2, 32, 4, 4]'))[1]
    num_groups = int(param.get('num_groups', 8))
    eps = float(param.get('eps', 1e-5))
    v = param.get('activate_silu', True)
    if isinstance(v, bool):
        activate_silu = v
    elif isinstance(v, int):
        activate_silu = bool(v)
    else:
        activate_silu = str(v).strip().lower() in ('1', 'true', 'yes')
    return [num_channels, num_groups, eps, activate_silu]
