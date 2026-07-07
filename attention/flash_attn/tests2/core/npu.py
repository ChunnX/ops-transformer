import importlib
from abc import ABC, abstractmethod


class NpuBase(ABC):
    @abstractmethod
    def npu_compute(self, inputs: dict, graph_path: int = 0) -> dict:
        ...


def load(op_name: str) -> NpuBase:
    mod = importlib.import_module(f"ops.{op_name}.npu")
    return mod.NpuBackend()
