import os
import random
from typing import Any, List, Optional, Tuple

import torch

from framework.utils import check_precision

_DEFAULT_PREPARE_RNG_BASE = 41


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
    seed = (base + _case_id_as_int(param) * 7919) % (2**31)
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


def _dtype_from_str(name: str) -> torch.dtype:
    name = str(name).strip().lower()
    if name in ("float16", "fp16"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float16


def _cos_sin_shape(b: int, s: int, d: int, cos_seq: str) -> Tuple[int, int, int, int]:
    """cos/sin 为 [B,1,S',64]，S' 为 1 或与 x 的 S 相同。"""
    cs = str(cos_seq).strip().upper()
    if cs == "S" and s > 1:
        return (b, 1, s, d)
    return (b, 1, 1, d)


def get_init_inputs(param: Any, device: Optional[torch.device] = None) -> Tuple[()]:
    return ()


def get_inputs(param: Any, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _seed_prepare_rng(param)
    dev = device or torch.device("cpu")

    b = int(param["B"])
    n = int(param["N"])
    s = int(param["S"])
    d = int(param["D"])
    dtype = _dtype_from_str(param.get("dtype", "float16"))
    cos_seq = str(param.get("cos_seq", "1"))

    x_shape = (b, n, s, d)
    cshape = _cos_sin_shape(b, s, d, cos_seq)

    x = (torch.rand(x_shape, device=dev, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    cos_t = (torch.rand(cshape, device=dev, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    sin_t = (torch.rand(cshape, device=dev, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    out = torch.empty_like(x)
    return (x, cos_t, sin_t, out)


def custom_check_precision(
    param: Any,
    outputs: Any,
    outputs_new: Any,
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    dtype_str = str(param.get("dtype", "float16"))
    if dtype_str == "bfloat16":
        atol, rtol = 2e-2, 2e-2
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
