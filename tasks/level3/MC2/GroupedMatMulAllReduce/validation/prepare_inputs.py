import os
import random
from typing import Any, List, Optional, Tuple

import torch

from npukernelbench.stages.precision.comparator import check_precision

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


def _parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def get_init_inputs(param: Any, device: Optional[torch.device] = None) -> Tuple[()]:
    return ()


def get_inputs(
    param: Any, device: Optional[torch.device] = None
) -> Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    Optional[torch.Tensor],
    int,
    str,
    str,
    int,
    List[torch.Tensor],
]:
    _seed_prepare_rng(param)
    dev = device or torch.device("cpu")

    dtype = _dtype_from_str(param.get("dtype", "float16"))
    has_bias = _parse_bool(param.get("has_bias", True))
    split_item = int(param.get("split_item", 0))
    group = str(param.get("group", "hccl_world_group"))
    reduce_op = str(param.get("reduce_op", "sum"))
    comm_turn = int(param.get("comm_turn", 0))

    m0 = int(param.get("M0", 256))
    k0 = int(param.get("K0", 256))
    n0 = int(param.get("N0", 256))
    m1 = int(param.get("M1", 1024))
    k1 = int(param.get("K1", 256))
    n1 = int(param.get("N1", 256))

    x0 = (torch.rand((m0, k0), device=dev, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    x1 = (torch.rand((m1, k1), device=dev, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    w0 = (torch.rand((k0, n0), device=dev, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    w1 = (torch.rand((k1, n1), device=dev, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)

    bdt = torch.float32 if dtype == torch.bfloat16 else torch.float16
    if has_bias:
        b0 = (torch.rand((n0,), device=dev, dtype=torch.float32) * 0.2 - 0.1).to(dtype=bdt)
        b1 = (torch.rand((n1,), device=dev, dtype=torch.float32) * 0.2 - 0.1).to(dtype=bdt)
    else:
        b0 = torch.zeros((n0,), device=dev, dtype=bdt)
        b1 = torch.zeros((n1,), device=dev, dtype=bdt)
    bias_list = [b0, b1]

    x_list = [x0, x1]
    w_list = [w0, w1]
    y0 = torch.zeros((m0, n0), device=dev, dtype=dtype)
    y1 = torch.zeros((m1, n1), device=dev, dtype=dtype)
    y_list = [y0, y1]

    group_list: Optional[torch.Tensor] = None

    return (x_list, w_list, bias_list, group_list, split_item, group, reduce_op, comm_turn, y_list)


def custom_check_precision(param: Any, outputs, outputs_new):
    return check_precision(outputs, outputs_new, max_abs_error=0.02, max_rel_error=0.02)
