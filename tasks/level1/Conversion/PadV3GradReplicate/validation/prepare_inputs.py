import ast

import torch


def _padded_shape(self_shape, padding_four):
    """由未 pad 的 self 形状与 F.pad 四元组 (left,right,top,bottom) 得到 grad_output 形状。"""
    s = list(self_shape)
    pl, pr, pt, pb = padding_four
    s[-2] += pt + pb
    s[-1] += pl + pr
    return tuple(s)


def _ge_padding_eight(padding_four):
    """GE / tiling 约定 8 元组：[N前,N后,C前,C后, padTop, padBottom, padLeft, padRight]。"""
    pl, pr, pt, pb = padding_four
    return [0, 0, 0, 0, pt, pb, pl, pr]


def get_inputs(param, device=None):
    shape = ast.literal_eval(param.get("input_shape", "[1,1,2,2]"))
    shape = tuple(int(x) for x in shape)
    padding_four = [int(x) for x in ast.literal_eval(param.get("padding", "[1,1,1,1]"))]
    if len(padding_four) != 4:
        raise ValueError("PadV3GradReplicate 仅支持 4D NCHW，padding 须为长度 4 的 (left,right,top,bottom)")

    dtype_str = param.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)

    grad_shape = _padded_shape(shape, padding_four)
    grad_output = torch.randn(grad_shape, device=device, dtype=dtype)
    ge_pad = _ge_padding_eight(padding_four)

    return (grad_output, ge_pad)


def get_init_inputs(param, device=None):
    return []


def custom_check_precision(row, ref_output, test_output):
    """fp16/bf16 与 CPU replicate 反传参考存在累加误差，适当放宽。"""
    from framework.kernel_gen_config import config
    from framework.utils import check_precision

    ma = config.eval.max_abs_error
    mr = config.eval.max_rel_error
    if row.get("dtype") in ("float16", "bfloat16"):
        ma = max(ma, 0.125)
        mr = max(mr, 0.05)
    return check_precision(ref_output, test_output, ma, mr)
