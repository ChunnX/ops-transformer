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
        from utils.data import trans_bnsd_to_layout

        layout_q  = params.get("layout_q", params.get("input_layout", "BNSD"))
        layout_kv = params.get("layout_kv", layout_q)

        # 转回 BNSD：NPU 格式 → CPU golden 需要的 BNSD
        q_bnsd = _to_bnsd(inputs["q"], layout_q, params)
        k_bnsd = _to_bnsd(inputs["k"], layout_kv, params)
        v_bnsd = _to_bnsd(inputs["v"], layout_kv, params)

        tforward_kwargs = dict(params)
        tforward_kwargs["scale"] = params.get("scale", 1 / (params["D"] ** 0.5))

        out_cpu, x_max, x_sum = tforward(q_bnsd, k_bnsd, v_bnsd, **tforward_kwargs)
        lse = torch.log(x_sum) + x_max
        return {"out": out_cpu, "lse": lse}


def _to_bnsd(t: torch.Tensor, layout: str, params: dict) -> torch.Tensor:
    """将任意 layout 的 tensor 转回 BNSD (B,N,S,D) 格式。"""
    if layout == "BNSD" or t.dim() != 4:
        return t.clone().float()
    if layout == "BSND":
        return t.permute(0, 2, 1, 3).contiguous().clone().float()
    if layout.startswith("PA_"):
        return _pa_to_bnsd(t, layout, params)
    return t.clone().float()


def _pa_to_bnsd(t: torch.Tensor, layout: str, params: dict) -> torch.Tensor:
    """PA 格式 → BNSD 重建。"""
    b   = params["B"]
    n2  = params.get("N2", params["N1"])
    d   = params["D"]
    s2  = params["S2"]  # 用 S2，不是 max(seqused_kv)
    seq_kv = params["seqused_kv"]
    bs  = params.get("block_size", 128)
    bt  = params.get("block_table")
    if bt is None:
        return t.clone().float()

    if isinstance(bt, torch.Tensor):
        bt = bt.cpu().numpy()
    result = torch.zeros(b, n2, s2, d, dtype=torch.float32)

    if layout == "PA_BBND":
        for bi in range(b):
            slen = seq_kv[bi] if bi < len(seq_kv) else seq_kv[0]
            nblk = (slen + bs - 1) // bs
            for bj in range(nblk):
                blk_id = bt[bi][bj] if bi < len(bt) and bj < len(bt[bi]) else bj
                if blk_id < 0 or blk_id >= t.shape[0]:
                    continue
                start = bj * bs
                end = min(start + bs, slen)
                cnt = end - start
                if cnt > 0:
                    result[bi, :, start:end, :] = t[blk_id, :cnt, :, :].permute(1, 0, 2).float()

    elif layout == "PA_BNBD":
        for bi in range(b):
            slen = seq_kv[bi] if bi < len(seq_kv) else seq_kv[0]
            nblk = (slen + bs - 1) // bs
            for bj in range(nblk):
                blk_id = bt[bi][bj] if bi < len(bt) and bj < len(bt[bi]) else bj
                if blk_id < 0 or blk_id >= t.shape[0]:
                    continue
                start = bj * bs
                end = min(start + bs, slen)
                cnt = end - start
                if cnt > 0:
                    result[bi, :, start:end, :] = t[blk_id, :, :cnt, :].float()

    else:
        return t.clone().float()

    return result
