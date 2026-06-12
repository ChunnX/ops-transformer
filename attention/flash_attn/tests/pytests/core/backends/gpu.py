"""GPU 后端实现 — 封装现有 flash_attn_gpu 调用。"""

import torch
from typing import Any, Dict

from .base import Backend

try:
    from core.backends.gpu_impl import flash_attn_gpu
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

    def clear_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        layout_kv = params.get("layout_kv", params.get("input_layout", "BNSD"))
        block_size = params.get("block_size", 256)

        if layout_kv == "PA_NZ" or (layout_kv.startswith("PA_") and block_size % 256 != 0):
            from core.backends.cpu import _pa_to_bnsd
            k_bnsd = _pa_to_bnsd(inputs["k"].cpu(), layout_kv, params)
            v_bnsd = _pa_to_bnsd(inputs["v"].cpu(), layout_kv, params)
            non_pa_params = dict(params)
            non_pa_params.pop("block_table", None)
            layout_q = params.get("layout_q", params.get("input_layout", "BNSD"))

            if layout_q == "TND":
                seqused_kv = params.get("seqused_kv") or params.get("actual_seq_kvlen")
                B_kv = k_bnsd.shape[0]
                k_slices, v_slices = [], []
                for i in range(B_kv):
                    s = int(seqused_kv[i]) if seqused_kv else k_bnsd.shape[2]
                    k_slices.append(k_bnsd[i, :, :s, :])
                    v_slices.append(v_bnsd[i, :, :s, :])
                k_tnd = torch.cat(k_slices, dim=1).unsqueeze(0).to(self._device)
                v_tnd = torch.cat(v_slices, dim=1).unsqueeze(0).to(self._device)
                if non_pa_params.get("cu_seqlens_kv") is None and seqused_kv is not None:
                    cu_kv = [0]
                    for s in seqused_kv:
                        cu_kv.append(cu_kv[-1] + int(s))
                    non_pa_params["cu_seqlens_kv"] = cu_kv
                non_pa_params["layout_kv"] = "TND"
                print(f"  [gpu] {layout_kv} TND+PA，KV→TND 走 varlen 路径")
                out_bnsd = flash_attn_gpu(
                    inputs["q"], k_tnd, v_tnd,
                    None, None, mask_gpu, **non_pa_params,
                )
            else:
                k_bnsd = k_bnsd.to(self._device)
                v_bnsd = v_bnsd.to(self._device)
                non_pa_params["layout_kv"] = "BNSD"
                print(f"  [gpu] {layout_kv} (block_size={block_size}) GPU不支持，PA→BNSD 降级")
                out_bnsd = flash_attn_gpu(
                    inputs["q"], k_bnsd, v_bnsd,
                    None, None, mask_gpu, **non_pa_params,
                )
        else:
            out_bnsd = flash_attn_gpu(
                inputs["q"], inputs["k"], inputs["v"],
                None, None, mask_gpu, **params,
            )

        torch.cuda.synchronize()
        return {"out": out_bnsd}
