from abc import ABC, abstractmethod

import torch


class GenDataBase(ABC):
    @abstractmethod
    def gen_inputs(self, params: dict) -> dict: ...

    @abstractmethod
    def normalize_params(self, params: dict) -> dict: ...

    def load_inputs(self, pt_path: str) -> dict:
        return torch.load(pt_path, map_location="cpu", weights_only=False)


class CpuBase(ABC):
    @abstractmethod
    def cpu_compute(self, inputs: dict) -> dict: ...


class NpuBase(ABC):
    @abstractmethod
    def npu_compute(self, inputs: dict, graph_path: int = 0) -> dict: ...


class GpuBase(ABC):
    @abstractmethod
    def gpu_compute(self, inputs: dict) -> dict: ...


def get_tolerance(dtype) -> dict:
    """默认精度阈值, 算子可在core/__init__.py中覆盖此函数"""
    from compare import DEFAULT_TOLERANCE
    return DEFAULT_TOLERANCE.get(dtype, {"atol": 1e-3, "rtol": 1e-3})


def forward(inputs: dict, device: str = "npu", **kwargs) -> dict:
    from core.gen_data import GenDataBackend
    from core.cpu import CpuBackend
    from core.npu import NpuBackend
    if device == "cpu":
        return CpuBackend().cpu_compute(inputs)
    elif device == "npu":
        return NpuBackend().npu_compute(inputs, **kwargs)
    elif device == "gpu":
        from core.gpu import GpuBackend
        return GpuBackend().gpu_compute(inputs)
    raise ValueError(f"Unsupported device: {device}")
