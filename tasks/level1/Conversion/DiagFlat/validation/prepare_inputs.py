import ast

import torch


def get_inputs(param, device=None):
    shape = ast.literal_eval(param.get("input_shape", "[4]"))
    shape = tuple(int(x) for x in shape)
    dtype_str = param.get("dtype", "float16")
    diagonal = int(param.get("diagonal", 0))
    dtype = getattr(torch, dtype_str)

    if dtype in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        if dtype == torch.int8:
            low, high = -128, 127
        elif dtype == torch.uint8:
            low, high = 0, 255
        else:
            low, high = -100, 100
        x = torch.randint(low, high, shape, device=device, dtype=dtype)
    elif dtype == torch.bool:
        x = torch.randint(0, 2, shape, device=device, dtype=torch.bool)
    elif dtype in (torch.complex64, torch.complex128):
        base_r = torch.float32 if dtype == torch.complex64 else torch.float64
        real = torch.randn(shape, device=device, dtype=base_r)
        imag = torch.randn(shape, device=device, dtype=base_r)
        x = torch.complex(real, imag)
    else:
        x = torch.randn(shape, device=device, dtype=dtype)

    return (x, diagonal)


def get_init_inputs(param, device=None):
    return []


def custom_check_precision(row, ref_output, test_output):
    """半精度与 CPU 参考比对时适当放宽阈值。"""
    from framework.kernel_gen_config import config
    from framework.utils import check_precision

    ma = config.eval.max_abs_error
    mr = config.eval.max_rel_error
    if row.get("dtype") in ("float16", "bfloat16"):
        ma = max(ma, 0.125)
        mr = max(mr, 0.05)
    return check_precision(ref_output, test_output, ma, mr)
