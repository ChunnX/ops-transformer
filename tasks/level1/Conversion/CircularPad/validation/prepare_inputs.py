import ast

import torch


def get_inputs(param, device=None):
    shape = ast.literal_eval(param.get("input_shape", "[1,1,4,4]"))
    shape = tuple(int(x) for x in shape)
    padding = [int(x) for x in ast.literal_eval(param.get("padding", "[1,1,1,1]"))]
    dtype_str = param.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)

    if dtype in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        if dtype == torch.int8:
            low, high = -128, 127
        elif dtype == torch.uint8:
            low, high = 0, 255
        else:
            low, high = -100, 100
        x = torch.randint(low, high, shape, device=device, dtype=dtype)
    else:
        x = torch.randn(shape, device=device, dtype=dtype)

    return (x, padding)


def get_init_inputs(param, device=None):
    return []
