import os
import random
from ast import literal_eval
from typing import Any, Dict, List, Tuple

import torch
from framework.utils import check_precision

_DEFAULT_PREPARE_RNG_BASE = 41
_BUNDLE_CACHE: Dict[Tuple[int, str], dict] = {}


def _case_id_as_int(param: Any) -> int:
    cid = param.get("case_id", 0)
    try:
        import pandas as pd

        if pd.isna(cid):
            return 0
    except Exception:
        pass
    return int(cid)


def _seed_prepare_rng(param: Any) -> None:
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


def _bundle_key(param: Any, device: Any) -> Tuple[int, str]:
    return (_case_id_as_int(param), str(device))


def _cos_sin_shape(q_shape: List[int], layout: int) -> List[int]:
    """BSND: [B,S,N,D] -> [B,S,1,D]；TND: [T,N,D] -> [T,1,D]。"""
    if layout == 1:
        b, s, _n, d = q_shape
        return [b, s, 1, d]
    if layout == 4:
        t, _n, d = q_shape
        return [t, 1, d]
    raise ValueError(f"unsupported layout {layout}")


def _materialize_case_tensors(param: Any, device: Any = None) -> dict:
    key = _bundle_key(param, device)
    if key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[key]

    _seed_prepare_rng(param)

    layout = int(param.get("layout", 1))
    q_shape = list(literal_eval(str(param.get("q_shape", "[1,2,2,128]"))))
    dtype_str = str(param.get("dtype", "float16"))
    dtype = getattr(torch, dtype_str)

    cos_shape = _cos_sin_shape(q_shape, layout)

    q = (torch.rand(q_shape, device=device, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    k = (torch.rand(q_shape, device=device, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    cos_t = (torch.rand(cos_shape, device=device, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    sin_t = (torch.rand(cos_shape, device=device, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)

    bundle = {
        "q": q,
        "k": k,
        "cos": cos_t,
        "sin": sin_t,
        "layout": layout,
    }
    _BUNDLE_CACHE[key] = bundle
    return bundle


def get_inputs(param: Any, device: Any = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b = _materialize_case_tensors(param, device=device)
    return (b["q"], b["k"], b["cos"], b["sin"])


def get_init_inputs(param: Any, device: Any = None) -> List[int]:
    b = _materialize_case_tensors(param, device=device)
    return [b["layout"]]


def custom_check_precision(param: Any, outputs: Any, outputs_new: Any) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """
    与 half RoPE 数值范围一致；bf16 路径内核在 fp32 累加，阈值略放宽。
    """
    dtype_str = str(param.get("dtype", "float16"))
    if dtype_str == "bfloat16":
        atol, rtol = 2e-2, 2e-2
    elif dtype_str == "float32":
        atol, rtol = 1e-4, 1e-4
    else:
        atol, rtol = 5e-3, 5e-3

    outs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    outs_n = outputs_new if isinstance(outputs_new, (list, tuple)) else [outputs_new]
    acc = 1
    all_abs: List[torch.Tensor] = []
    all_rel: List[torch.Tensor] = []
    for o, n in zip(outs, outs_n):
        a, ad, rd = check_precision([o], [n], atol, rtol)
        acc = 1 if (acc == 1 and a == 1) else 0
        all_abs.append(ad)
        all_rel.append(rd)
    return acc, torch.cat(all_abs), torch.cat(all_rel)
