import os
import random

import torch
from framework.utils import check_precision

# 与 DynamicQuantUpdateScatter 等任务一致：按 case_id 固定 RNG，避免多次跑批时同一 case 随机输入不同导致精度时好时坏。
_DEFAULT_PREPARE_RNG_BASE = 34

# (case_id, device_str) -> dict，所有随机张量只构建一次，供 get_init_inputs / get_inputs 共用
_BUNDLE_CACHE = {}


def _case_id_as_int(param):
    cid = param.get("case_id", 0)
    try:
        import pandas as pd

        if pd.isna(cid):
            return 0
    except Exception:
        pass
    return int(cid)


def _seed_prepare_rng(param):
    base = int(os.environ.get("NKB_PREPARE_INPUTS_SEED", str(_DEFAULT_PREPARE_RNG_BASE)))
    cid = _case_id_as_int(param)
    seed = (base + cid * 7919) % (2**31)
    torch.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.npu.manual_seed(seed)
    except Exception:
        pass


def _csv_bool(param, key, default=True):
    v = param.get(key, default)
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("false", "0", "no"):
        return False
    return True


def _bundle_key(param, device):
    return (_case_id_as_int(param), str(device))


def _materialize_case_tensors(param, device=None):
    """
    仅此函数内进行 seed 与所有 torch.rand：生成该 case 的完整张量并缓存。
    batch_precision_eval 先调 get_init_inputs 再调 get_inputs，故首次实际在此构建；
    get_init_inputs 不再单独随机构造，避免两套 RNG / 两套数据不一致。
    """
    key = _bundle_key(param, device)
    if key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[key]

    _seed_prepare_rng(param)

    input_shape = eval(param.get("input_shape", "[1]"))
    norm_shape = eval(param.get("normalized_shape", "[1]"))

    dtype_str = param.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)

    scales_dtype_str = param.get("scales_dtype", "float32")
    scales_dtype = getattr(torch, scales_dtype_str)

    zeros_dtype_str = param.get("zeros_dtype", "int32")
    zeros_dtype = getattr(torch, zeros_dtype_str)

    x1 = (torch.rand(input_shape, device=device, dtype=dtype) * 2 - 1)
    x2 = (torch.rand(input_shape, device=device, dtype=dtype) * 2 - 1)

    gamma = torch.rand(norm_shape, device=device, dtype=dtype)
    scales1 = torch.rand(norm_shape, device=device, dtype=scales_dtype)
    scales2 = None
    has_zp1 = _csv_bool(param, "has_zero_points1", True)
    zero_points1 = (
        torch.rand(norm_shape, device=device, dtype=zeros_dtype) if has_zp1 else None
    )
    zero_points2 = None

    axis = -1
    epsilon = param.get("epsilon", 1e-5)
    div_mode = _csv_bool(param, "div_mode", True)

    bundle = {
        "x1": x1,
        "x2": x2,
        "gamma": gamma,
        "scales1": scales1,
        "scales2": scales2,
        "zero_points1": zero_points1,
        "zero_points2": zero_points2,
        "axis": axis,
        "epsilon": epsilon,
        "div_mode": div_mode,
    }
    _BUNDLE_CACHE[key] = bundle
    return bundle


def get_inputs(param, device=None):
    """
    返回 forward 用 (x1, x2)；与 get_init_inputs 共用 _materialize_case_tensors 的同一份缓存数据。
    """
    b = _materialize_case_tensors(param, device=device)
    return (b["x1"], b["x2"])


def get_init_inputs(param, device=None):
    """
    返回 Model / ModelNew 初始化参数列表；随机张量来自与 get_inputs 相同的 bundle，不在此函数内单独 rand。
    """
    b = _materialize_case_tensors(param, device=device)
    return [
        b["gamma"],
        b["scales1"],
        b["scales2"],
        b["zero_points1"],
        b["zero_points2"],
        b["axis"],
        b["epsilon"],
        b["div_mode"],
    ]


def custom_check_precision(param, outputs, outputs_new):
    is_pass = 1
    LOSS = 3e-3
    all_abs_diff, all_rel_diff = [], []
    for out, out_new in zip(outputs, outputs_new):
        result = torch.abs(out - out_new)

        # Get maximum of absolute values for relative error denominator
        deno = torch.maximum(torch.abs(out), torch.abs(out_new))

        # Calculate absolute error check
        result_atol = result

        # Calculate relative error check
        result_rtol = result / (deno + 1e-7)
        print(torch.sum(result_rtol > LOSS))
        print(torch.sum(result_atol > LOSS))

        # Count failures where error exceeds tolerance
        if torch.sum(result_rtol > LOSS) > out.numel() * LOSS and \
        torch.sum(result_atol > LOSS) > out.numel() * LOSS:
            is_pass = 0
        all_abs_diff.append(result_atol)
        all_rel_diff.append(result_rtol)
    all_abs_diff = torch.cat(all_abs_diff)
    all_rel_diff = torch.cat(all_rel_diff)
    return is_pass, all_abs_diff, all_rel_diff
