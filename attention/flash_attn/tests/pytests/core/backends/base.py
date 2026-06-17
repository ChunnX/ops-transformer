"""计算后端抽象基类。"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import torch


class Backend(ABC):
    """算子计算后端统一接口。"""

    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def compute(self, inputs: Dict[str, torch.Tensor],
                params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...

    def clear_cache(self):
        pass
