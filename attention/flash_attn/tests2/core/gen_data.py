import importlib
from abc import ABC, abstractmethod

import torch


class GenDataBase(ABC):
    @abstractmethod
    def gen_inputs(self, params: dict) -> dict:
        ...

    @abstractmethod
    def normalize_params(self, params: dict) -> dict:
        ...

    def load_inputs(self, pt_path: str) -> dict:
        return torch.load(pt_path)


def load(op_name: str) -> GenDataBase:
    mod = importlib.import_module(f"ops.{op_name}.gen_data")
    return mod.GenData()
