import importlib
from abc import ABC, abstractmethod


class GpuBase(ABC):
    @abstractmethod
    def gpu_compute(self, inputs: dict) -> dict:
        ...


def load(op_name: str) -> GpuBase:
    mod = importlib.import_module(f"ops.{op_name}.gpu")
    return mod.Gpu()
