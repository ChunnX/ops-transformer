"""NPU 后端实现 — 封装现有 flash_attn_npu 调用链。"""

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
        """调用 npu_flash_attn_metadata + npu_flash_attn。"""
        q = inputs["q"]
        k = inputs["k"]
        v = inputs["v"]
        layout_q = params.get("layout_q", "BNSD")
        layout_kv = params.get("layout_kv", layout_q)
        layout_out = params.get("layout_out", layout_q)

        n1 = params["N1"]
        n2 = params.get("N2", n1)
        d = params["D"]
        b = params.get("B", 1)
        s1 = params.get("S1", 1)
        s2 = params.get("S2", s1)

        mask_mode = params.get("mask_mode", 0)
        wl = params.get("win_left", -1) if mask_mode not in (0, 3) else -1
        wr = params.get("win_right", -1) if mask_mode not in (0, 3) else -1
        scale = params.get("scale", 1 / (d ** 0.5))

        cu_q = params.get("cu_seqlens_q")
        cu_kv = params.get("cu_seqlens_kv")
        seq_q = params.get("seqused_q", params.get("actual_seq_qlen"))
        seq_kv = params.get("seqused_kv", params.get("actual_seq_kvlen"))
        block_table = inputs.get("block_table")

        def _u32(x):
            if x is None: return None
            if torch.is_tensor(x): return x.to(dtype=torch.int32, device=self._device).contiguous()
            return torch.tensor(x, dtype=torch.int32, device=self._device).contiguous()

        cu_q_t = _u32(cu_q); cu_kv_t = _u32(cu_kv)
        sq_t = _u32(seq_q); skv_t = _u32(seq_kv)

        if layout_q == "BNSD":
            batch_size = q.shape[0]; nqh = q.shape[1]; msq = q.shape[2]; hd = q.shape[3]
            nkh = k.shape[1]; msk = k.shape[2]
        elif layout_q == "BSND":
            batch_size = q.shape[0]; nqh = q.shape[2]; msq = q.shape[1]; hd = q.shape[3]
            nkh = k.shape[2]; msk = k.shape[1]
        elif layout_q == "TND":
            batch_size = cu_q_t.shape[0] - 1 if cu_q_t is not None else b
            nqh = q.shape[1]; hd = q.shape[2]; nkh = k.shape[1]
            msq = s1; msk = s2
        else:
            batch_size = q.shape[0]; nqh = n1; nkh = n2; hd = d
            msq = q.shape[1]; msk = k.shape[1]

        bt = block_table.to(self._device) if block_table is not None and layout_kv in ("PA_BBND","PA_BNBD","PA_NZ") else None
        if bt is not None:
            nkh = n2; hd = d; msk = max(skv_t) if skv_t is not None else s2

        metadata = npu_flash_attn_metadata(
            cu_seqlens_q=cu_q_t, cu_seqlens_kv=cu_kv_t,
            seqused_q=sq_t, seqused_kv=skv_t,
            num_heads_q=nqh, num_heads_kv=nkh, head_dim=hd,
            batch_size=batch_size, max_seqlen_q=msq, max_seqlen_kv=msk,
            mask_mode=mask_mode, win_left=wl, win_right=wr,
            layout_q=layout_q, layout_kv=layout_kv, layout_out=layout_out,
        )

        out, lse = npu_flash_attn(
            q, k, v,
            block_table=bt, sinks=None, attn_mask=None,
            metadata=metadata,
            cu_seqlens_q=cu_q_t, cu_seqlens_kv=cu_kv_t,
            seqused_q=sq_t, seqused_kv=skv_t,
            softmax_scale=scale, mask_mode=mask_mode,
            win_left=wl, win_right=wr,
            max_seqlen_q=msq, max_seqlen_kv=msk,
            layout_q=layout_q, layout_kv=layout_kv, layout_out=layout_out,
            return_softmax_lse=0,
        )
        torch.npu.synchronize()
        return {"out": out, "lse": lse}
