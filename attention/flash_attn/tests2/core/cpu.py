import importlib
from abc import ABC, abstractmethod


class CpuBase(ABC):
    @abstractmethod
    def cpu_compute(self, inputs: dict) -> dict:
        ...


def load(op_name: str) -> CpuBase:
    mod = importlib.import_module(f"ops.{op_name}.cpu")
    return mod.Cpu()
