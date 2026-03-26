import ast

import torch


def _padded_shape(self_shape, padding_four):
    s = list(self_shape)
    pl, pr, pt, pb = padding_four
    s[-2] += pt + pb
    s[-1] += pl + pr
    return tuple(s)


def _ge_padding_eight(padding_four):
    pl, pr, pt, pb = padding_four
    return [0, 0, 0, 0, int(pt), int(pb), int(pl), int(pr)]


def get_inputs(param, device=None):
    shape = ast.literal_eval(param.get("input_shape", "[1,1,4,4]"))
    shape = tuple(int(x) for x in shape)
    padding_four = [int(x) for x in ast.literal_eval(param.get("padding", "[1,1,1,1]"))]
    if len(padding_four) != 4:
        raise ValueError("PadV4Grad 仅支持 4D NCHW，padding 为 (left,right,top,bottom)")
    if len(shape) != 4:
        raise ValueError("PadV4Grad validation 仅支持 4D")

    dtype_str = param.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)

    pl, pr, pt, pb = padding_four
    h, w = shape[2], shape[3]
    # torch reflect：各边 pad 须小于对应未 pad 空间维长度
    if not (pt < h and pb < h and pl < w and pr < w):
        raise ValueError(f"reflect 约束要求 pad 各边 < 空间维: shape={shape}, pad={padding_four}")

    grad_shape = _padded_shape(shape, padding_four)
    grad_output = torch.randn(grad_shape, device=device, dtype=dtype)
    ge_pad = _ge_padding_eight(padding_four)
    return (grad_output, ge_pad)


def get_init_inputs(param, device=None):
    return []


def custom_check_precision(row, ref_output, test_output):
    from framework.kernel_gen_config import config
    from framework.utils import check_precision

    ma = config.eval.max_abs_error
    mr = config.eval.max_rel_error
    if row.get("dtype") in ("float16", "bfloat16"):
        ma = max(ma, 0.125)
        mr = max(mr, 0.05)
    return check_precision(ref_output, test_output, ma, mr)
