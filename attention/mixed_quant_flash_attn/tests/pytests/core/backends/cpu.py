"""CPU Golden 后端实现 — 调用 tforward 做参考计算。"""

import torch
from typing import Any, Dict

from .base import Backend


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
        from core.backends.cpu_golden import tforward

        layout_q  = params.get("layout_q", params.get("input_layout", "BNSD"))
        layout_kv = params.get("layout_kv", layout_q)

        q_cpu = _to_tforward(inputs["q"], layout_q, params)

        if params.get("quant_mode", 0) >= 1:
            q_dtype = inputs["q"].dtype
            k_cpu = _dequant_kv_to_tforward(inputs["k"], inputs["scale_k"], layout_kv, params, target_dtype=q_dtype)
            v_cpu = _dequant_kv_to_tforward(inputs["v"], inputs["scale_v"], layout_kv, params, target_dtype=q_dtype)
        else:
            k_cpu = _to_tforward(inputs["k"], layout_kv, params)
            v_cpu = _to_tforward(inputs["v"], layout_kv, params)

        tforward_kwargs = dict(params)
        tforward_kwargs["scale"] = params.get("scale", 1 / (params["D"] ** 0.5))

        out_cpu, x_max, x_sum = tforward(q_cpu, k_cpu, v_cpu, **tforward_kwargs)
        lse = torch.log(x_sum) + x_max
        lse[x_max <= torch.finfo(torch.float32).min + 1.0] = float('inf')
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


def _dequant_kv_to_tforward(kv_uint8, scale_uint8, layout_kv, params, target_dtype):
    """unpacked fp4 KV + unpacked e8m0 scale → float BNSD (tforward format)"""
    from core.backends.quant_utils import fp4_to_float, e8m0_to_float

    group_size = params.get("group_size", 32)

    if layout_kv.startswith("PA_"):
        kv_bnsd = _pa_to_bnsd(kv_uint8, layout_kv, params)
        scale_bnsd = _pa_to_bnsd(scale_uint8, "PA_BBND", params, d_override=scale_uint8.shape[-1])
    else:
        if layout_kv == "BSND":
            kv_bnsd = kv_uint8.permute(0, 2, 1, 3).contiguous().clone()
            scale_bnsd = scale_uint8.permute(0, 2, 1, 3).contiguous().clone()
        elif layout_kv == "TND":
            kv_bnsd = kv_uint8.unsqueeze(0).permute(0, 2, 1, 3).contiguous().clone()
            scale_bnsd = scale_uint8.unsqueeze(0).permute(0, 2, 1, 3).contiguous().clone()
        else:
            kv_bnsd = kv_uint8
            scale_bnsd = scale_uint8

    kv_float = fp4_to_float(kv_bnsd)
    scale_float = e8m0_to_float(scale_bnsd)
    scale_broadcast = scale_float.repeat_interleave(group_size, dim=-1)
    kv_raw = kv_float * scale_broadcast
    finfo = torch.finfo(target_dtype)
    kv_deq = kv_raw.clamp(-finfo.max, finfo.max).to(target_dtype)

    if params.get("layout_q", "") == "TND":
        B = kv_deq.shape[0]
        N2 = kv_deq.shape[1]
        total = sum(params["seqused_kv"][:B])
        result = torch.zeros(1, N2, total, kv_deq.shape[-1], dtype=torch.float32)
        offset = 0
        for b in range(B):
            slen = params["seqused_kv"][b] if b < len(params["seqused_kv"]) else params["seqused_kv"][0]
            result[0, :, offset:offset + slen, :] = kv_deq[b, :, :slen, :]
            offset += slen
        return result

    return kv_deq.float()


def _pa_to_bnsd(t: torch.Tensor, layout: str, params: dict, d_override=None) -> torch.Tensor:
    """PA 格式 → BNSD 重建。"""
    b   = params.get("actual_b", params["B"])  # TND 时 B=1，用 actual_b
    n2  = params.get("N2", params["N1"])
    d   = d_override if d_override is not None else params["D"]
    s2  = params["S2"]  # 用 S2，不是 max(seqused_kv)
    seq_kv = params["seqused_kv"]
    bs  = params.get("block_size", 128)
    bt  = params.get("block_table")
    if bt is None:
        return t.clone()

    if isinstance(bt, torch.Tensor):
        bt = bt.cpu().numpy()
    result = torch.zeros(b, n2, s2, d, dtype=t.dtype)

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
                    result[bi, :, start:end, :] = t[blk_id, :cnt, :, :].permute(1, 0, 2)

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
                    result[bi, :, start:end, :] = t[blk_id, :, :cnt, :]

    elif layout == "PA_NZ":
        dtype_str = params.get("Dtype", "fp16")
        blk_elem = 16 if dtype_str in ("fp16", "bf16") else 32
        d_actual = t.shape[2] * t.shape[4]
        result = torch.zeros(b, n2, s2, d_actual, dtype=t.dtype)
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
                    result[bi, :, start:end, :] = block_data

    else:
        return t.clone()

    return result
