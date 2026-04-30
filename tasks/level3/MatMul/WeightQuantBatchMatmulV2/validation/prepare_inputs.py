import torch
import numpy as np
from typing import List, Optional


def get_init_inputs(param=None, **kwargs):
    return []


def get_inputs(param: dict, **kwargs):
    m = param['m']
    n = param['n']
    k = param['k']
    transpose_x = param.get('transpose_x', 0) == 1
    transpose_weight = param.get('transpose_weight', 0) == 1
    antiquant_group_size = param.get('antiquant_group_size', 0)
    has_bias = param.get('has_bias', 1) == 1
    has_antiquant_offset = param.get('has_antiquant_offset', 1) == 1

    x_dtype = torch.float16

    if not transpose_x:
        x = torch.randn(m, k, dtype=x_dtype)
    else:
        x = torch.randn(k, m, dtype=x_dtype)

    if not transpose_weight:
        weight = torch.randint(-127, 127, (k, n), dtype=torch.int8)
    else:
        weight = torch.randint(-127, 127, (n, k), dtype=torch.int8)

    if antiquant_group_size > 0:
        n_groups = k // antiquant_group_size
        if not transpose_weight:
            antiquant_scale = torch.randn(n_groups, k, dtype=x_dtype) * 0.1
        else:
            antiquant_scale = torch.randn(n_groups, k, dtype=x_dtype) * 0.1
    else:
        if not transpose_weight:
            antiquant_scale = torch.randn(1, k, dtype=x_dtype) * 0.1
        else:
            antiquant_scale = torch.randn(k, 1, dtype=x_dtype) * 0.1

    antiquant_offset = None
    if has_antiquant_offset:
        antiquant_offset = torch.randn_like(antiquant_scale) * 0.05

    quant_scale = None
    quant_offset = None

    bias = None
    if has_bias:
        bias = torch.randn(n, dtype=x_dtype) * 0.1

    dtype_val = -1
    inner_precise = 0

    return [x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias,
            transpose_x, transpose_weight, antiquant_group_size, dtype_val, inner_precise]
