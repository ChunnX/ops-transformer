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
            q = inputs["q"].unsqueeze(0).permute(0, 2, 1, 3).clone().float()
            k = inputs["k"].unsqueeze(0).permute(0, 2, 1, 3).clone().float()
            v = inputs["v"].unsqueeze(0).permute(0, 2, 1, 3).clone().float()
        else:  # BNSD / BSH
            q = inputs["q"].clone().float()
            k = inputs["k"].clone().float()
            v = inputs["v"].clone().float()

        tforward_kwargs = dict(params)
        tforward_kwargs["scale"] = params.get("scale", 1 / (params["D"] ** 0.5))

        out_cpu, x_max, x_sum = tforward(q, k, v, **tforward_kwargs)
        lse = torch.log(x_sum) + x_max

        return {"out": out_cpu, "lse": lse}
