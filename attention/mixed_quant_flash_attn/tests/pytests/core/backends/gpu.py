"""GPU 后端实现 — 封装现有 flash_attn_gpu 调用。"""

import torch
from typing import Any, Dict

from .base import Backend

try:
    from backends.gpu_impl import flash_attn_gpu
    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False


class GPUBackend(Backend):
    name = "gpu"

    def __init__(self, device_id: int = 0):
        self._device = torch.device(f"cuda:{device_id}")
        if _HAS_GPU:
            torch.cuda.set_device(device_id)

    @property
    def device(self) -> torch.device:
        return self._device

    def is_available(self) -> bool:
        return _HAS_GPU and torch.cuda.is_available()

    def compute(self, inputs: Dict[str, torch.Tensor],
                params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """调用 flash_attn_gpu。"""

        from utils.data import generate_npu_mask

        b = params.get("B", 1)
        s1 = params.get("S1", 1)
        s2 = params.get("S2", s1)
        mask_mode = params.get("mask_mode", 0)
        wl = params.get("win_left", 2147483647)
        wr = params.get("win_right", 2147483647)

        atten_mask = generate_npu_mask(b, s1, s2, mask_mode, wl, wr)
        mask_gpu = atten_mask.to(self._device) if atten_mask is not None else None

        out_bnsd = flash_attn_gpu(
            inputs["q"], inputs["k"], inputs["v"],
            None, None, mask_gpu, **params,
        )

        torch.cuda.synchronize()
        return {"out": out_bnsd}
