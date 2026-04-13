import math
import torch
from framework.utils import check_precision


def _cell_str(param, key, default=""):
    v = param.get(key, default)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    return str(v).strip()


def get_inputs(param, device=None):
    permuted_tokens_shape = eval(param.get("permuted_tokens_shape", "[1,512]"))
    probs_shape = _cell_str(param, "probs_shape", "")
    restore_shape = eval(param.get("restore_shape", "[16,16]"))
    padded_mode = bool(param.get("padded_mode", 0))
    dtype_str = param.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)

    rows, hidden = permuted_tokens_shape[0], permuted_tokens_shape[1]
    use_probs = bool(int(param.get("has_probs", 1)))

    permuted_tokens = torch.randn(permuted_tokens_shape, dtype=dtype)

    if use_probs and probs_shape not in ("", "none", "None"):
        probs_shape = eval(probs_shape)
        tokens_num, topk = probs_shape[0], probs_shape[1]
        if rows != tokens_num * topk:
            raise ValueError(
                f"Shape mismatch: permuted_tokens dim0 {rows} != tokens_num*topk ({tokens_num}*{topk})"
            )
        probs = torch.rand(probs_shape, dtype=dtype)
    else:
        tokens_num, topk = rows, 1
        probs = None

    nidx = tokens_num * topk
    sorted_indices = torch.randperm(nidx, dtype=torch.int32)

    return (
        permuted_tokens.to(device),
        sorted_indices.to(device),
        probs.to(device) if probs is not None else None,
        padded_mode,
    )


def get_init_inputs(param, device=None):
    return []


def custom_check_precision(param, outputs, outputs_new):
    dtype_str = param.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)
    if dtype == torch.float32:
        return check_precision(outputs, outputs_new, max_abs_error=0.0001, max_rel_error=0.0001)
    if dtype == torch.float16:
        return check_precision(outputs, outputs_new, max_abs_error=0.001, max_rel_error=0.001)
    if dtype == torch.bfloat16:
        return check_precision(outputs, outputs_new, max_abs_error=0.004, max_rel_error=0.004)
    return check_precision(outputs, outputs_new, max_abs_error=0.0001, max_rel_error=0.0001)
