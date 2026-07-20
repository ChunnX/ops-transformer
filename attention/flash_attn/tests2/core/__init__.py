from abc import ABC, abstractmethod


class GenDataBase(ABC):
    @abstractmethod
    def gen_inputs(self, params: dict) -> dict: ...

    @abstractmethod
    def normalize_params(self, params: dict) -> dict: ...


class CpuBase(ABC):
    @abstractmethod
    def cpu_compute(self, inputs: dict) -> dict: ...


class NpuBase(ABC):
    @abstractmethod
    def npu_compute(self, inputs: dict, npu_mode: str = "eager") -> dict: ...


class GpuBase(ABC):
    @abstractmethod
    def gpu_compute(self, inputs: dict) -> dict: ...


def get_tolerance(dtype) -> dict:
    """默认精度阈值, 算子可在core/__init__.py中覆盖此函数"""
    from compare import DEFAULT_TOLERANCE
    return DEFAULT_TOLERANCE.get(dtype, {"atol": 1e-3, "rtol": 1e-3})


