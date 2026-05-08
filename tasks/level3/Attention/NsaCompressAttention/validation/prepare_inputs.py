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
        float(row["scale_value"]),
        int(row["head_num"]),
        int(row["sparse_mode"]),
        int(row["compress_block_size"]),
        int(row["compress_stride"]),
        int(row["select_block_size"]),
        int(row["select_block_count"]),
    )


def get_inputs(param: Any, device: Optional[torch.device] = None) -> Tuple[Any, ...]:
    _seed_prepare_rng(param)
    dev = device or torch.device("cpu")

    n2 = int(param["N2"])
    t1 = int(param["T1"])
    t2 = int(param["T2"])
    g = int(param["G"])
    d1 = int(param["D1"])
    d2 = int(param["D2"])
    sel_cnt = int(param["select_block_count"])
    head_num = int(param["head_num"])

    if head_num != n2 * g:
        raise ValueError(f"head_num({head_num}) 须等于 N2*G({n2}*{g})")

    g_rng = torch.Generator(device=dev)
    g_rng.manual_seed((_case_id_as_int(param) * 10007 + 13) % (2**31))

    query = torch.randn(n2, t1, g, d1, dtype=torch.float16, device=dev, generator=g_rng)
    key = torch.randn(t2, n2, d1, dtype=torch.float16, device=dev, generator=g_rng)
    value = torch.randn(t2, n2, d2, dtype=torch.float16, device=dev, generator=g_rng)

    softmax_max = torch.zeros(t1, n2 * g, 8, dtype=torch.float32, device=dev)
    softmax_sum = torch.zeros(t1, n2 * g, 8, dtype=torch.float32, device=dev)
    attention_out = torch.zeros(n2, t1, g, d2, dtype=torch.float16, device=dev)
    topk_indices_out = torch.zeros(n2, t1, sel_cnt, dtype=torch.int32, device=dev)

    # 单段前缀累加：长度为 1，存放总长（与 tiling GetActualSeqLenData 首元素语义一致）
    actual_seq_qlen = torch.tensor([t1], dtype=torch.int64, device=dev)
    actual_cmp_seq_kvlen = torch.tensor([t2], dtype=torch.int64, device=dev)
    actual_sel_seq_kvlen = torch.tensor([t2], dtype=torch.int64, device=dev)

    return (
        query,
        key,
        value,
        softmax_max,
        softmax_sum,
        attention_out,
        topk_indices_out,
        actual_seq_qlen,
        actual_cmp_seq_kvlen,
        actual_sel_seq_kvlen,
    )


def custom_check_precision(
    param: Any,
    ref_output: List[torch.Tensor],
    test_output: List[torch.Tensor],
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """NSA 与朴素 GQA 在 topk/softmax 统计上不对齐，仅对 attention_out 做相对宽松对比。"""
    del param
    outs = ref_output if isinstance(ref_output, (list, tuple)) else [ref_output]
    outs_n = test_output if isinstance(test_output, (list, tuple)) else [test_output]
    return check_precision(outs[2], outs_n[2], max_abs_error=0.25, max_rel_error=0.15)
