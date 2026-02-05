import math
import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape = eval(param.get('shape', '[64]'))
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    n_val = param.get('n', '')
    # CSV 空单元格可能被 pandas 读成 float('nan')，不能直接 int(nan)
    if n_val is None or (isinstance(n_val, float) and math.isnan(n_val)) or str(n_val).strip() in ('', 'nan'):
        n = None
    else:
        n = int(float(n_val))
    norm = int(param.get('norm', '1'))

    input_tensor = torch.randn(shape, device=device, dtype=dtype)
    return (input_tensor, n, norm)


def get_init_inputs(param, device=None):
    return []
