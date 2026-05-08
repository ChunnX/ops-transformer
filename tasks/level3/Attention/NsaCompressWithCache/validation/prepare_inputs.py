import os
import random
from typing import Any, Optional, Tuple

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
    return (
        int(row["compress_block_size"]),
        int(row["compress_stride"]),
        int(row["act_seq_len_type"]),
        int(row["page_block_size"]),
        str(row.get("layout", "TND")).strip() or "TND",
    )


def get_inputs(param: Any, device: Optional[torch.device] = None) -> Tuple[Any, ...]:
    _seed_prepare_rng(param)
    dev = device or torch.device("cpu")

    batch = int(param["batch"])
    blocks_per_batch = int(param["blocks_per_batch"])
    page_bs = int(param["page_block_size"])
    n = int(param["N"])
    d = int(param["D"])
    l_blk = int(param["compress_block_size"])
    result_len = int(param["result_len"])
    slot_row = int(param["slot_row"])

    blocks_num = batch * blocks_per_batch
    g = torch.Generator(device=dev)
    g.manual_seed((_case_id_as_int(param) * 10007 + 13) % (2**31))

    inp = torch.randn(blocks_num, page_bs, n, d, dtype=torch.float16, device=dev, generator=g)
    w = torch.randn(l_blk, n, dtype=torch.float16, device=dev, generator=g)
    slot = torch.tensor([slot_row], dtype=torch.int32, device=dev)
    act = torch.tensor([32], dtype=torch.int64, device=dev)
    bt = torch.arange(blocks_per_batch, dtype=torch.int32, device=dev).unsqueeze(0).expand(batch, -1).contiguous()
    out_cache = torch.zeros(result_len, n, d, dtype=torch.float16, device=dev)

    return (inp, w, slot, act, bt, out_cache)


def custom_check_precision(param: Any, ref_out: torch.Tensor, test_out: torch.Tensor):
    del param
    return check_precision(ref_out, test_out, max_abs_error=0.08, max_rel_error=0.05)
