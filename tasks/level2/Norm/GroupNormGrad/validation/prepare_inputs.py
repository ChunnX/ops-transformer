import torch
import numpy as np

def get_inputs(param, device=None):
    """
    Generate input tensors for the GroupNormGrad operator.
    Inputs: dy, mean, rstd, x, gamma; attrs: num_groups, data_format, dx_is_require, dgamma_is_require, dbeta_is_require.
    """
    x_shape = eval(param.get('x_shape', '[2, 32, 4, 4]'))
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    num_channels = x_shape[1]
    num_groups = int(param.get('num_groups', 8))

    dy = (torch.rand(x_shape, device=device, dtype=dtype) * 2.0 - 1.0) * 0.1
    mean_rstd_shape = (x_shape[0], num_groups)
    mean = torch.rand(mean_rstd_shape, device=device, dtype=dtype) * 0.1
    rstd = torch.rand(mean_rstd_shape, device=device, dtype=dtype) * 0.1 + 1e-3

    x = (torch.rand(x_shape, device=device, dtype=dtype) * 2.0 - 1.0) * 0.1
    gamma = torch.rand(num_channels, device=device, dtype=dtype) * 0.1 + 0.5

    dx_is_require = bool(int(param.get('dx_is_require', 1)))
    dgamma_is_require = bool(int(param.get('dgamma_is_require', 1)))
    dbeta_is_require = bool(int(param.get('dbeta_is_require', 1)))

    return (dy, mean, rstd, x, gamma, dx_is_require, dgamma_is_require, dbeta_is_require)


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the GroupNormGrad model from DataFrame row.
    """
    num_groups = int(param.get('num_groups', 8))
    data_format = param.get('data_format', 'NCHW')
    return [num_groups, data_format]
