import ast

import torch


def _padded_shape(self_shape, padding):
    """与 CircularPad 正向一致：由 unpadded(self) 形状与 padding 得到 grad_output 形状。"""
    s = list(self_shape)
    if len(padding) == 4:
        s[-2] += padding[2] + padding[3]
        s[-1] += padding[0] + padding[1]
    else:
        s[-3] += padding[4] + padding[5]
        s[-2] += padding[2] + padding[3]
        s[-1] += padding[0] + padding[1]
    return tuple(s)


def get_inputs(param, device=None):
    shape = ast.literal_eval(param.get("input_shape", "[1,1,4,4]"))
    shape = tuple(int(x) for x in shape)
    padding = [int(x) for x in ast.literal_eval(param.get("padding", "[1,1,1,1]"))]
    dtype_str = param.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)

    grad_shape = _padded_shape(shape, padding)
    grad_output = torch.randn(grad_shape, device=device, dtype=dtype)

    return (grad_output, padding)


def get_init_inputs(param, device=None):
    return []


def custom_check_precision(row, ref_output, test_output):
    """fp16/bf16 大 shape 梯度累加与 CPU 参考存在半精度误差，适当放宽阈值。"""
    from framework.kernel_gen_config import config
    from framework.utils import check_precision

    ma = config.eval.max_abs_error
    mr = config.eval.max_rel_error
    if row.get('dtype') in ('float16', 'bfloat16'):
        ma = max(ma, 0.125)
        mr = max(mr, 0.05)
    return check_precision(ref_output, test_output, ma, mr)
