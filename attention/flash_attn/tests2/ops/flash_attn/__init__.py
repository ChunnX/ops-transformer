from .gen_data import GenDataBackend
from .cpu import CpuBackend
from .npu import NpuBackend

__all__ = ["GenDataBackend", "CpuBackend", "NpuBackend", "forward"]


def forward(inputs: dict, device: str = "npu", **kwargs) -> dict:
    if device == "cpu":
        return CpuBackend().cpu_compute(inputs)
    elif device == "npu":
        return NpuBackend().npu_compute(inputs, **kwargs)
    elif device == "gpu":
        from .gpu import GpuBackend
        return GpuBackend().gpu_compute(inputs)
    raise ValueError(f"Unsupported device: {device}")
