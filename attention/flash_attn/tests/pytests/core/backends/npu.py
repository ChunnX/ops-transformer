"""NPU 后端 — 按 npu_flash_attn.md 参数组调用。"""

import torch
import torch_npu
from typing import Any, Dict

from .base import Backend

try:
    from npu_ops_transformer.ops import npu_flash_attn, npu_flash_attn_metadata
    _HAS_NPU = True
except ImportError:
    _HAS_NPU = False


class NPUBackend(Backend):
    name = "npu"

    def __init__(self, device_id: int = 0):
        self._device = torch.device(f"npu:{device_id}")
        if _HAS_NPU:
            torch.npu.set_device(device_id)

    @property
    def device(self) -> torch.device:
        return self._device

    def is_available(self) -> bool:
        return _HAS_NPU

    def compute(self, inputs: Dict[str, torch.Tensor],
                params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        from core.data import build_flash_attn_params

        meta_kwargs, kernel_kwargs, _ = build_flash_attn_params(
            params, self._device, inputs)

        metadata = npu_flash_attn_metadata(**meta_kwargs)
        out, lse = npu_flash_attn(inputs["q"], inputs["k"], inputs["v"],
                                  metadata=metadata, **kernel_kwargs)
        torch.npu.synchronize()
        return {"out": out, "lse": lse}
