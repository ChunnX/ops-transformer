"""NPU 后端 — 按 npu_flash_attn 参数组调用。"""

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

        def _u32(x):
            if x is None: return None
            if torch.is_tensor(x): return x.to(dtype=torch.int32, device=self._device).contiguous()
            return torch.tensor(x, dtype=torch.int32, device=self._device).contiguous()

        # === 公共参数组 ===
        q = inputs["q"]
        k = inputs["k"]
        v = inputs["v"]
        layout_q   = params.get("layout_q", params.get("input_layout", "BNSD"))
        layout_kv  = params.get("layout_kv", layout_q)
        layout_out = params.get("layout_out", layout_q)
        d = params["D"]
        scale = params.get("scale", 1 / (d ** 0.5))

        # === Mask 参数组 ===
        mask_mode = params.get("mask_mode", 0)
        wl = wr = -1
        attn_mask = None
        if mask_mode == 3:
            mask = torch.triu(torch.ones(2048, 2048), diagonal=1)
            attn_mask = mask.to(dtype=torch.int8, device=self._device)
        elif mask_mode == 4:
            wl = params.get("win_left", 2147483647)
            wr = params.get("win_right", 2147483647)

        # === SeqLens 参数组 ===
        cu_q  = params.get("cu_seqlens_q")
        cu_kv = params.get("cu_seqlens_kv")
        seq_q = params.get("seqused_q", params.get("actual_seq_qlen"))
        seq_kv = params.get("seqused_kv", params.get("actual_seq_kvlen"))
        cu_q_t  = _u32(cu_q);  cu_kv_t = _u32(cu_kv)
        sq_t    = _u32(seq_q); skv_t   = _u32(seq_kv)

        # 从 tensor shape 推导 metadata 所需参数
        if layout_q == "BNSD":
            bs = q.shape[0]; nqh = q.shape[1]; msq = q.shape[2]; hd = q.shape[3]
            nkh = k.shape[1]; msk = k.shape[2]
        elif layout_q == "BSND":
            bs = q.shape[0]; nqh = q.shape[2]; msq = q.shape[1]; hd = q.shape[3]
            nkh = k.shape[2]; msk = k.shape[1]
        elif layout_q == "TND":
            bs = cu_q_t.shape[0] - 1 if cu_q_t is not None else params.get("B", 1)
            nqh = q.shape[1]; hd = q.shape[2]; nkh = k.shape[1]
            msq = params.get("S1", 1); msk = params.get("S2", params.get("S1", 1))
        else:
            bs = q.shape[0]; nqh = params["N1"]; nkh = params.get("N2", nqh)
            hd = d; msq = q.shape[1]; msk = k.shape[1]

        # === Paged Attention 参数组 ===
        bt = inputs.get("block_table")
        if bt is not None and layout_kv in ("PA_BBND", "PA_BNBD", "PA_NZ"):
            bt = bt.to(self._device)
            nkh = params.get("N2", nqh)
            hd = d
            msk = max(skv_t) if skv_t is not None else params.get("S2", params.get("S1", 1))
        else:
            bt = None

        metadata = npu_flash_attn_metadata(
            cu_seqlens_q=cu_q_t, cu_seqlens_kv=cu_kv_t,
            seqused_q=sq_t, seqused_kv=skv_t,
            num_heads_q=nqh, num_heads_kv=nkh, head_dim=hd,
            batch_size=bs, max_seqlen_q=msq, max_seqlen_kv=msk,
            mask_mode=mask_mode, win_left=wl, win_right=wr,
            layout_q=layout_q, layout_kv=layout_kv, layout_out=layout_out,
        )

        out, lse = npu_flash_attn(
            q, k, v,
            block_table=bt, sinks=None, attn_mask=attn_mask,
            metadata=metadata,
            cu_seqlens_q=cu_q_t, cu_seqlens_kv=cu_kv_t,
            seqused_q=sq_t, seqused_kv=skv_t,
            softmax_scale=scale, mask_mode=mask_mode,
            win_left=wl, win_right=wr,
            max_seqlen_q=-1, max_seqlen_kv=-1,
            layout_q=layout_q, layout_kv=layout_kv, layout_out=layout_out,
            return_softmax_lse=0,
        )
        torch.npu.synchronize()
        return {"out": out, "lse": lse}
