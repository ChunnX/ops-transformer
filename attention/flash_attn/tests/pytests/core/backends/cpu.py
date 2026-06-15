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

        layout_q  = params.get("layout_q", params.get("input_layout", "BNSD"))
        layout_kv = params.get("layout_kv", layout_q)

        q_cpu = _to_tforward(inputs["q"], layout_q, params)
        k_cpu = _to_tforward(inputs["k"], layout_kv, params)
        v_cpu = _to_tforward(inputs["v"], layout_kv, params)

        tforward_kwargs = dict(params)
        tforward_kwargs["scale"] = params.get("scale", 1 / (params["D"] ** 0.5))

        out_cpu, x_max, x_sum = tforward(q_cpu, k_cpu, v_cpu, **tforward_kwargs)
        lse = torch.log(x_sum) + x_max
        return {"out": out_cpu, "lse": lse}


def _to_tforward(t: torch.Tensor, layout: str, params: dict) -> torch.Tensor:
    """转换为 tforward 期望的格式。"""
    if layout == "TND":
        return t.unsqueeze(0).permute(0, 2, 1, 3).contiguous().clone().float()
    if layout == "BSND":
        return t.permute(0, 2, 1, 3).contiguous().clone().float()
    if layout.startswith("PA_"):
        bnsd = _pa_to_bnsd(t, layout, params)
        if params.get("layout_q", "") == "TND":
            # TND+PA: PA→BNSD后按 seqused_kv 紧密拼接，与纯 TND+TND 一致
            # result[0, n, cum_kv[b]+s, :] = bnsd[b, n, s, :]  (s < seqused_kv[b])
            B, N2, S2, D = bnsd.shape
            seq_kv = params["seqused_kv"]
            total_kv = sum(seq_kv[:B])
            result = torch.zeros(1, N2, total_kv, D, dtype=torch.float32)
            offset = 0
            for b in range(B):
                slen = seq_kv[b] if b < len(seq_kv) else seq_kv[0]
                result[0, :, offset:offset + slen, :] = bnsd[b, :, :slen, :]
                offset += slen
            return result
        return bnsd.float()
    return t.clone().float()


def _pa_to_bnsd(t: torch.Tensor, layout: str, params: dict) -> torch.Tensor:
    """PA 格式 → BNSD 重建。"""
    b   = params.get("actual_b", params["B"])  # TND 时 B=1，用 actual_b
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

    elif layout == "PA_NZ":
        dtype_str = params.get("Dtype", "fp16")
        blk_elem = 16 if dtype_str in ("fp16", "bf16") else 32
        d_actual = t.shape[2] * t.shape[4]
        result = torch.zeros(b, n2, s2, d_actual, dtype=torch.float32)
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
                    block_data = t[blk_id, :, :, :cnt, :]
                    block_data = block_data.permute(0, 2, 1, 3).reshape(
                        block_data.shape[0], cnt, -1)
                    result[bi, :, start:end, :] = block_data.float()

    else:
        return t.clone().float()

    return result
