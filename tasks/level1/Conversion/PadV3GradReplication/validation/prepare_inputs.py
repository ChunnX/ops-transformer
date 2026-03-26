import ast

import torch

# GE / tiling：PADDING_API_LENGTH=10，tiling 使用下标 4..9 三对，对应末三维 (D,H,W) 的 pad；4D 时末维对为 W,H，[4,5] 置 0


def _ge10_from_padding_4(pl, pr, pt, pb):
    return [0, 0, 0, 0, 0, 0, int(pt), int(pb), int(pl), int(pr)]


def _ge10_from_padding_6(wl, wr, hl, hr, dl, dr):
    # 与 tiling 一致：下标 4..9 为 D,H,W 三对；前 4 位为 N/C 等占位 0
    return [0, 0, 0, 0, int(dl), int(dr), int(hl), int(hr), int(wl), int(wr)]


def _padded_shape(self_shape, padding_list):
    s = list(self_shape)
    if len(self_shape) == 4:
        pl, pr, pt, pb = padding_list
        s[-2] += pt + pb
        s[-1] += pl + pr
    elif len(self_shape) == 5:
        wl, wr, hl, hr, dl, dr = padding_list
        s[-3] += dl + dr
        s[-2] += hl + hr
        s[-1] += wl + wr
    else:
        raise ValueError("PadV3GradReplication 仅支持 4D 或 5D input_shape")
    return tuple(s)


def get_inputs(param, device=None):
    shape = ast.literal_eval(param.get("input_shape", "[1,1,2,2]"))
    shape = tuple(int(x) for x in shape)
    padding = [int(x) for x in ast.literal_eval(param.get("padding", "[1,1,1,1]"))]
    dtype_str = param.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)

    if len(shape) == 4:
        if len(padding) != 4:
            raise ValueError("4D 用例 padding 须为 [pl,pr,pt,pb]（与 torch F.pad 一致）")
        ge_pad = _ge10_from_padding_4(*padding)
    elif len(shape) == 5:
        if len(padding) != 6:
            raise ValueError("5D 用例 padding 须为 [wl,wr,hl,hr,dl,dr]（末三维 W,H,D）")
        ge_pad = _ge10_from_padding_6(*padding)
    else:
        raise ValueError("input_shape 须为 4D 或 5D")

    grad_shape = _padded_shape(shape, padding)
    grad_output = torch.randn(grad_shape, device=device, dtype=dtype)
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
