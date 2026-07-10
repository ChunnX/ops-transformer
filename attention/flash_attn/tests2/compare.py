import logging
import torch
from typing import Callable

logger = logging.getLogger("compare")

DEFAULT_TOLERANCE = {
    torch.float32: {"atol": 1e-3, "rtol": 1e-3},
    torch.float16: {"atol": 1e-1, "rtol": 1e-1},
    torch.bfloat16: {"atol": 1e-1, "rtol": 1e-1},
}

FAIL_RATIO_LIMIT = 0.005


def _resolve_tolerance(dtype, atol, rtol):
    default = DEFAULT_TOLERANCE.get(dtype, {"atol": 1e-3, "rtol": 1e-3})
    return (
        atol if atol is not None else default["atol"],
        rtol if rtol is not None else default["rtol"],
    )


def _stats(ef, rf, atol, rtol):
    diff = torch.abs(ef - rf)
    fail_mask = ~torch.isclose(ef, rf, atol=atol, rtol=rtol)
    fail_cnt = int(fail_mask.sum().item())
    total = ef.numel()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "fail_cnt": fail_cnt,
        "total": total,
        "fail_ratio": fail_cnt / total if total else 0.0,
    }




def compare_outputs(
    cpu_out: dict,
    npu_out: dict,
    *,
    chunk_size: int = 0,
    dtype_handler: Callable[[torch.Tensor], torch.Tensor] = None,
    atol: float = None,
    rtol: float = None,
) -> dict:
    if atol is None and rtol is None:
        try:
            from core import get_tolerance
            for key in cpu_out:
                cpu_t = cpu_out[key]
                tol = get_tolerance(cpu_t.dtype)
                atol = tol["atol"]
                rtol = tol["rtol"]
                break
        except (ImportError, AttributeError):
            pass
    missing_keys = set(cpu_out.keys()) - set(npu_out.keys())
    if missing_keys:
        logger.warning("[cmp] npu_out missing keys: %s", missing_keys)
    result = {}
    all_passed = True
    for key in cpu_out:
        if key not in npu_out:
            continue
        cpu_t = cpu_out[key]
        npu_t = npu_out[key]
        if dtype_handler is not None:
            cpu_t = dtype_handler(cpu_t)
            npu_t = dtype_handler(npu_t)
        cpu_f = cpu_t.float()
        npu_f = npu_t.float()
        a, r = _resolve_tolerance(cpu_t.dtype, atol, rtol)
        total = cpu_f.numel()
        if chunk_size > 0 and total > chunk_size:
            # 分块顺序处理：降低内存峰值，无需多进程和共享内存
            cpu_flat = cpu_f.reshape(-1)
            npu_flat = npu_f.reshape(-1)
            parts = []
            for off in range(0, total, chunk_size):
                length = min(chunk_size, total - off)
                parts.append(_stats(cpu_flat[off:off + length], npu_flat[off:off + length], a, r))
            tot = sum(p["total"] for p in parts)
            fail_cnt = sum(p["fail_cnt"] for p in parts)
            stats = {
                "max_abs": max(p["max_abs"] for p in parts),
                "mean_abs": sum(p["mean_abs"] * p["total"] for p in parts) / tot if tot else 0.0,
                "fail_cnt": fail_cnt,
                "total": tot,
                "fail_ratio": fail_cnt / tot if tot else 0.0,
            }
        else:
            stats = _stats(cpu_f, npu_f, a, r)
        stats["passed"] = stats["fail_ratio"] <= FAIL_RATIO_LIMIT
        if not stats["passed"]:
            all_passed = False
        result[key] = stats
    result["passed"] = all_passed
    return result
