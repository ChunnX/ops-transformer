import ast
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from npukernelbench.stages.precision.comparator import check_precision


def _seed(case_id: int):
    torch.manual_seed(case_id)
    random.seed(case_id)
    np.random.seed(case_id)


def _parse_bool(v, default=False):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return default


def _row_to_dict(row: Any) -> Dict[str, Any]:
    if hasattr(row, "to_dict"):
        return row.to_dict()
    return dict(row) if isinstance(row, dict) else row


def _align_num_out_tokens(token_num: int, expert_num: int, num_out_tokens: int, drop_and_pad: bool) -> int:
    """与 aclnn CheckParams / tiling：dropless 按 token 对齐，drop_pad 按 expert 对齐。"""
    num_out_tokens = min(token_num * expert_num, int(num_out_tokens))
    num_out_tokens = max(token_num, int(num_out_tokens))
    align = expert_num if drop_and_pad else token_num
    if align <= 0:
        return num_out_tokens
    return (num_out_tokens // align) * align


def _routing_map_like_executor(
    token_num: int, expert_num: int, num_out_tokens: int, drop_and_pad: bool, device: Optional[torch.device], case_id: int
) -> Tuple[torch.Tensor, int]:
    """对齐 executor `init_by_input_data`：dropless 每行 topk 个 True；drop_pad 随机 bool。"""
    _seed(case_id)
    num_out_tokens = _align_num_out_tokens(token_num, expert_num, num_out_tokens, drop_and_pad)

    if drop_and_pad:
        rmap = torch.rand(token_num, expert_num, device=device, dtype=torch.float32) > 0.5
        return rmap.to(torch.bool), num_out_tokens

    topk = num_out_tokens // token_num
    if topk >= 512:
        # 与 executor：cap topK=512，且 numOutTokens 不超过 token*expert
        num_out_tokens = min(token_num * expert_num, token_num * 512)
        num_out_tokens = _align_num_out_tokens(token_num, expert_num, num_out_tokens, False)
        topk = 512
    rmap = torch.zeros((token_num, expert_num), dtype=torch.bool, device=device)
    g = torch.Generator(device="cpu")
    g.manual_seed(case_id + 911)
    for i in range(token_num):
        ind = torch.randperm(expert_num, generator=g)[:topk]
        if device is not None:
            ind = ind.to(device)
        rmap[i, ind] = True
    return rmap, num_out_tokens


def get_init_inputs(row, device=None):
    return []


def get_inputs(row, device=None):
    d = _row_to_dict(row)
    case_id = int(d.get("case_id", 0))
    _seed(case_id)

    dtype_str = d.get("dtype", "float16")
    tensor_dtype = getattr(torch, dtype_str)

    tokens_shape = ast.literal_eval(str(d.get("tokens_shape", "[8,128]")).replace(" ", ""))
    routing_map_shape = ast.literal_eval(str(d.get("routing_map_shape", "[8,16]")).replace(" ", ""))
    token_num, hidden = int(tokens_shape[0]), int(tokens_shape[1])
    r0, expert_num = int(routing_map_shape[0]), int(routing_map_shape[1])
    if r0 != token_num:
        raise ValueError(f"tokens_shape[0]={token_num} must match routing_map_shape[0]={r0}")

    num_out_tokens = int(d.get("num_out_tokens", 32))
    drop_and_pad = _parse_bool(d.get("drop_and_pad", False), False)
    use_probs = _parse_bool(d.get("use_probs", True), True)

    routing_map, num_out_tokens = _routing_map_like_executor(
        token_num, expert_num, num_out_tokens, drop_and_pad, device, case_id
    )

    tokens = torch.randn(token_num, hidden, dtype=tensor_dtype, device=device)
    probs_optional = None
    if use_probs:
        probs_optional = torch.randn(token_num, expert_num, dtype=tensor_dtype, device=device)

    return (tokens, routing_map, probs_optional, num_out_tokens, drop_and_pad)


def custom_check_precision(row, outputs, outputs_new):
    d = _row_to_dict(row)
    dtype_str = d.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)

    outs = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
    outs_new = list(outputs_new) if isinstance(outputs_new, (list, tuple)) else [outputs_new]

    if dtype == torch.float32:
        abs_e, rel_e = 1.0e-4, 1.0e-4
    elif dtype == torch.float16:
        abs_e, rel_e = 5.0e-3, 5.0e-3
    elif dtype == torch.bfloat16:
        abs_e, rel_e = 1.0e-2, 1.0e-2
    else:
        abs_e, rel_e = 1.0e-4, 1.0e-4

    float_ref, float_new = [], []
    for o, on in zip(outs, outs_new):
        if o is None or on is None:
            continue
        if o.numel() == 0 and on.numel() == 0:
            continue
        if on.dtype != o.dtype:
            on = on.to(o.dtype)
        if o.dtype in (torch.int32, torch.int64):
            if not torch.equal(o.cpu(), on.cpu()):
                z = torch.tensor([1.0])
                return 0, z, z
            continue
        float_ref.append(o.float() if o.dtype in (torch.int8, torch.uint8) else o)
        float_new.append(on.float() if on.dtype in (torch.int8, torch.uint8) else on)

    if not float_ref:
        return 1, torch.tensor([0.0]), torch.tensor([0.0])

    return check_precision(float_ref, float_new, max_abs_error=abs_e, max_rel_error=rel_e)
