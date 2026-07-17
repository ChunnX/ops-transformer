import logging
from abc import ABC, abstractmethod
from typing import Callable

import torch

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
        "atol": atol,
        "rtol": rtol,
        "_diff": diff,
        "_fail_mask": fail_mask,  # 供 precision_analyze 复用, 避免重复计算
    }


def _resolve_global_tolerance(cpu_out, atol, rtol):
    """atol/rtol 均为 None 时, 从 core.get_tolerance 取算子自定义阈值。"""
    if atol is None and rtol is None:
        try:
            from core import get_tolerance
            for key in cpu_out:
                tol = get_tolerance(cpu_out[key].dtype)
                return tol["atol"], tol["rtol"]
        except (ImportError, AttributeError):
            pass
    return atol, rtol


def _warn_missing_keys(cpu_out, npu_out, extra=()):
    missing_keys = set(cpu_out.keys()) - set(npu_out.keys()) - set(extra)
    if missing_keys:
        logger.warning("[cmp] npu_out missing keys: %s", missing_keys)


class Comparator(ABC):
    """精度比对抽象基类。

    算子可通过继承子类 override 定制精度标准(FAIL_RATIO_LIMIT/DEFAULT_TOLERANCE)
    和比对算法(two_way/three_way)。
    """

    @abstractmethod
    def compare_two_way(self, cpu_out: dict, npu_out: dict, *,
                        chunk_size: int = 0,
                        dtype_handler: Callable[[torch.Tensor], torch.Tensor] = None,
                        atol: float = None, rtol: float = None) -> dict:
        """CPU vs NPU 双向比对。"""
        ...

    @abstractmethod
    def compare_three_way(self, cpu_out: dict, npu_out: dict, gpu_out: dict, *,
                          chunk_size: int = 0,
                          dtype_handler: Callable[[torch.Tensor], torch.Tensor] = None,
                          atol: float = None, rtol: float = None) -> dict:
        """CPU vs NPU vs GPU 三方比对。GPU 作 benchmark, NPU 误差 vs GPU 误差比倍数。"""
        ...


class DefaultComparator(Comparator):
    """双千分之五默认实现。

    精度标准: fail_ratio <= FAIL_RATIO_LIMIT (0.5%)
    三方比对: NPU 误差不应显著大于 GPU 误差 (倍数上限)。
    """

    FAIL_RATIO_LIMIT = 0.005
    MAX_RE_RTOL = 10.0   # 三方比对: NPU max误差 / GPU max误差 倍数上限
    AVG_RE_RTOL = 2.0    # 三方比对: NPU avg误差 / GPU avg误差 倍数上限

    def compare_two_way(self, cpu_out, npu_out, *,
                        chunk_size=0, dtype_handler=None, atol=None, rtol=None):
        atol, rtol = _resolve_global_tolerance(cpu_out, atol, rtol)
        _warn_missing_keys(cpu_out, npu_out)
        result = {}
        all_passed = True
        for key in cpu_out:
            if key not in npu_out:
                continue
            cpu_t, npu_t = cpu_out[key], npu_out[key]
            if dtype_handler is not None:
                cpu_t = dtype_handler(cpu_t)
                npu_t = dtype_handler(npu_t)
            cpu_f, npu_f = cpu_t.float(), npu_t.float()
            a, r = _resolve_tolerance(cpu_t.dtype, atol, rtol)
            stats = self._compute_stats(cpu_f, npu_f, a, r, chunk_size)
            stats["passed"] = stats["fail_ratio"] <= self.FAIL_RATIO_LIMIT
            if not stats["passed"]:
                all_passed = False
            result[key] = stats
        result["passed"] = all_passed
        return result

    def compare_three_way(self, cpu_out, npu_out, gpu_out, *,
                          chunk_size=0, dtype_handler=None, atol=None, rtol=None):
        atol, rtol = _resolve_global_tolerance(cpu_out, atol, rtol)
        _warn_missing_keys(cpu_out, npu_out, extra=set(gpu_out.keys()))
        result = {}
        all_passed = True
        for key in cpu_out:
            if key not in npu_out or key not in gpu_out:
                continue
            cpu_t, npu_t, gpu_t = cpu_out[key], npu_out[key], gpu_out[key]
            if dtype_handler is not None:
                cpu_t = dtype_handler(cpu_t)
                npu_t = dtype_handler(npu_t)
                gpu_t = dtype_handler(gpu_t)
            cpu_f, npu_f, gpu_f = cpu_t.float(), npu_t.float(), gpu_t.float()
            a, r = _resolve_tolerance(cpu_t.dtype, atol, rtol)
            npu_stats = self._compute_stats(npu_f, cpu_f, a, r, chunk_size)
            gpu_stats = self._compute_stats(gpu_f, cpu_f, a, r, chunk_size)
            # NPU 误差 vs GPU 误差比倍数 (GPU 误差为 0 时退化为 NPU 自身 fail_ratio 判定)
            gpu_max = gpu_stats["max_abs"] + 1e-12
            gpu_mean = gpu_stats["mean_abs"] + 1e-12
            ratio_max = npu_stats["max_abs"] / gpu_max
            ratio_mean = npu_stats["mean_abs"] / gpu_mean
            passed = (npu_stats["fail_ratio"] <= self.FAIL_RATIO_LIMIT
                      and ratio_max <= self.MAX_RE_RTOL
                      and ratio_mean <= self.AVG_RE_RTOL)
            result[key] = {
                **npu_stats,
                "gpu_max_abs": gpu_stats["max_abs"],
                "gpu_mean_abs": gpu_stats["mean_abs"],
                "gpu_fail_ratio": gpu_stats["fail_ratio"],
                "ratio_max": ratio_max,
                "ratio_mean": ratio_mean,
                "passed": passed,
            }
            if not passed:
                all_passed = False
        result["passed"] = all_passed
        return result

    @staticmethod
    def _compute_stats(cpu_f, npu_f, atol, rtol, chunk_size):
        total = cpu_f.numel()
        if chunk_size > 0 and total > chunk_size:
            cpu_flat = cpu_f.reshape(-1)
            npu_flat = npu_f.reshape(-1)
            parts = []
            for off in range(0, total, chunk_size):
                length = min(chunk_size, total - off)
                parts.append(_stats(cpu_flat[off:off + length],
                                    npu_flat[off:off + length], atol, rtol))
            tot = sum(p["total"] for p in parts)
            fail_cnt = sum(p["fail_cnt"] for p in parts)
            return {
                "max_abs": max(p["max_abs"] for p in parts),
                "mean_abs": sum(p["mean_abs"] * p["total"] for p in parts) / tot if tot else 0.0,
                "fail_cnt": fail_cnt,
                "total": tot,
                "fail_ratio": fail_cnt / tot if tot else 0.0,
                "atol": atol,
                "rtol": rtol,
            }
        return _stats(cpu_f, npu_f, atol, rtol)


def compare_outputs(cpu_out, npu_out, *, chunk_size=0, dtype_handler=None,
                    atol=None, rtol=None) -> dict:
    """已废弃: 使用 DefaultComparator().compare_two_way()。"""
    return DefaultComparator().compare_two_way(
        cpu_out, npu_out, chunk_size=chunk_size,
        dtype_handler=dtype_handler, atol=atol, rtol=rtol)
