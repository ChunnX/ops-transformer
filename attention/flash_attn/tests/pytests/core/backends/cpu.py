"""CPU Golden 后端实现 — 调用 tforward 做参考计算。"""

import torch
from typing import Any, Dict

from .base import Backend
from utils.data import generate_cpu_mask


class CPUBackend(Backend):
    name = "cpu"

    def __init__(self):
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    def is_available(self) -> bool:
        return True

    def compute(self, inputs: Dict[str, torch.Tensor],
                params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        from backends.cpu_impl import tforward

        layout = params.get("input_layout", params.get("layout_q", "BNSD"))

        if layout == "BSND":
            q = inputs["q"].permute(0, 2, 1, 3).contiguous().clone().float()
            k = inputs["k"].permute(0, 2, 1, 3).contiguous().clone().float()
            v = inputs["v"].permute(0, 2, 1, 3).contiguous().clone().float()
        elif layout == "TND":
            q = inputs["q"].clone().float()
            k = inputs["k"].clone().float()
            v = inputs["v"].clone().float()
        else:  # BNSD / BSH
            q = inputs["q"].clone().float()
            k = inputs["k"].clone().float()
            v = inputs["v"].clone().float()

        s1 = q.shape[2] if q.dim() == 4 else q.shape[1]
        s2 = k.shape[2] if k.dim() == 4 else k.shape[1]
        b = q.shape[0]
        mask_mode = params.get("mask_mode", 0)
        wl = params.get("win_left", 2147483647)
        wr = params.get("win_right", 2147483647)
        atten_mask = generate_cpu_mask(b, s1, s2, mask_mode, wl, wr)

        out_cpu, x_max, x_sum = tforward(q, k, v, atten_mask=atten_mask, scale=params.get("scale"))
        lse = torch.log(x_sum) + x_max

        return {"out": out_cpu, "lse": lse}
