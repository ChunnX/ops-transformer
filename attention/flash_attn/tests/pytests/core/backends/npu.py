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
        from utils.data import trans_bnsd_to_layout

        # PA 时 k/v 从 BNSD 转为 PA 格式；q 按 layout_q 转
        layout_q  = params.get("layout_q", params.get("input_layout", "BNSD"))
        layout_kv = params.get("layout_kv", layout_q)
        q_npu = trans_bnsd_to_layout(inputs["q"].float(), layout_q, **params).to(dtype=inputs["q"].dtype).to(self._device)
        k_npu = trans_bnsd_to_layout(inputs["k"].float(), layout_kv, **params).to(dtype=inputs["k"].dtype).to(self._device)
        v_npu = trans_bnsd_to_layout(inputs["v"].float(), layout_kv, **params).to(dtype=inputs["v"].dtype).to(self._device)
        npu_inputs = {"q": q_npu, "k": k_npu, "v": v_npu}
        if "block_table" in inputs:
            npu_inputs["block_table"] = inputs["block_table"].to(self._device)

        meta_kwargs, kernel_kwargs, _ = build_flash_attn_params(
            params, self._device, npu_inputs)

        metadata = npu_flash_attn_metadata(**meta_kwargs)
        out, lse = npu_flash_attn(npu_inputs["q"], npu_inputs["k"], npu_inputs["v"],
                                  metadata=metadata, **kernel_kwargs)
        torch.npu.synchronize()
        return {"out": out, "lse": lse}
