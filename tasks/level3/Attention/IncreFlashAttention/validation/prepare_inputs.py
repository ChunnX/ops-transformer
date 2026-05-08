import math
import os
import random
from typing import Any, Optional, Tuple

import torch

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
    seed = (base + _case_id_as_int(param)) % (2**31)
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


def get_init_inputs(param: Any, device: Optional[torch.device] = None) -> Tuple[Any, ...]:
    row = param
    d = int(row["D"])
    return (
        int(row["num_heads"]),
        int(row["num_kv_heads"]),
        float(1.0 / math.sqrt(d)),
        str(row["layout"]).strip() or "BNSD",
    )


def get_inputs(param: Any, device: Optional[torch.device] = None) -> Tuple[Any, ...]:
    _seed_prepare_rng(param)
    dev = device or torch.device("cpu")

    B = int(param["B"])
    N = int(param["N"])
    S_kv = int(param["S_kv"])
    D = int(param["D"])
    actual_kv = int(param["actual_kv"])
    layout = str(param["layout"]).strip() or "BNSD"

    g = torch.Generator(device=dev)
    g.manual_seed((_case_id_as_int(param) * 10007 + 13) % (2**31))

    if layout != "BNSD":
        raise ValueError("validation 仅覆盖 BNSD layout")

    q = torch.randn(B, N, 1, D, dtype=torch.float16, device=dev, generator=g)
    k = torch.randn(B, N, S_kv, D, dtype=torch.float16, device=dev, generator=g)
    v = torch.randn(B, N, S_kv, D, dtype=torch.float16, device=dev, generator=g)
    out = torch.zeros(B, N, 1, D, dtype=torch.float16, device=dev)
    actual = torch.tensor([actual_kv] * B, dtype=torch.int64, device=dev)

    return (q, k, v, out, actual)


def custom_check_precision(param: Any, ref_out: torch.Tensor, test_out: torch.Tensor):
    """FP16 增量注意力与 FP32 参考对比，阈值与仓库内 level3 融合算子量级一致。"""
    from npukernelbench.stages.precision.comparator import check_precision

    return check_precision(ref_out, test_out, max_abs_error=0.08, max_rel_error=0.05)
