import torch
import torch.multiprocessing as mp
from typing import Callable

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


def _chunk_worker(args):
    cpu_shm, npu_shm, offset, length, atol, rtol = args
    ef = cpu_shm.view(-1)[offset:offset + length]
    rf = npu_shm.view(-1)[offset:offset + length]
    return _stats(ef, rf, atol, rtol)


def compare_outputs(
    cpu_out: dict,
    npu_out: dict,
    *,
    chunk_size: int = 0,
    num_workers: int = 0,
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
            cpu_shm = cpu_f.contiguous().clone()
            npu_shm = npu_f.contiguous().clone()
            cpu_shm.share_memory_()
            npu_shm.share_memory_()
            tasks = []
            for off in range(0, total, chunk_size):
                length = min(chunk_size, total - off)
                tasks.append((cpu_shm, npu_shm, off, length, a, r))
            with mp.Pool(num_workers or None) as pool:
                parts = pool.map(_chunk_worker, tasks)
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
