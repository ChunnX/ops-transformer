"""计算后端抽象基类。"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import torch


class Backend(ABC):
    """算子计算后端统一接口。

    用法:
        backend = NPUBackend(device_id=0)
        outputs = backend.compute(inputs, spec)
    """

    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def compute(self, inputs: Dict[str, torch.Tensor],
                params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """在 self.device 上执行算子计算。

        Args:
            inputs: generate_inputs() 产出的张量，在 self.device 上
            params: case 参数 dict（含 layout, dtype, mask_mode 等）
        Returns:
            {"out": ..., "lse": ...}  仍在 self.device 上
        """
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...
