import torch
from typing import Any, Dict

from core import CpuBase
from core.utils.cpu_golden import tforward
from core.utils.data_utils import pa_to_bnsd


class CpuBackend(CpuBase):
    name = "cpu"

    def __init__(self):
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    def is_available(self) -> bool:
        return True

    def cpu_compute(self, inputs: dict) -> dict:
        params = _build_params(inputs)

        layout_q  = params.get("layout_q", params.get("input_layout", "BNSD"))
        layout_kv = params.get("layout_kv", layout_q)

        q_cpu = _to_tforward(inputs["q"], layout_q, params)
        k_cpu = _to_tforward(inputs["k"], layout_kv, params)
        v_cpu = _to_tforward(inputs["v"], layout_kv, params)

        tforward_kwargs = {k: v for k, v in params.items() if k not in ("q", "k", "v")}
        tforward_kwargs["scale"] = params.get("scale", 1 / (params["D"] ** 0.5))

        out_cpu, x_max, x_sum = tforward(q_cpu, k_cpu, v_cpu, **tforward_kwargs)
        lse = torch.log(x_sum) + x_max
        lse[x_max <= torch.finfo(torch.float32).min + 1.0] = float('inf')
        return {"out": out_cpu, "lse": lse}


def _build_params(inputs: dict) -> dict:
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]

    layout_q  = inputs.get("layout_q", inputs.get("input_layout", "BNSD"))
    layout_kv = inputs.get("layout_kv", layout_q)

    params = dict(inputs)
    params["input_layout"] = layout_q
    params["layout_q"] = layout_q
    params["layout_kv"] = layout_kv

    if layout_q == "BNSD":
        B, N1, S1, D = q.shape
        params.setdefault("B", B); params.setdefault("N1", N1)
        params.setdefault("S1", S1); params.setdefault("D", D)
        if not layout_kv.startswith("PA_"):
            params.setdefault("N2", k.shape[1]); params.setdefault("S2", k.shape[2])
    elif layout_q == "BSND":
        B, S1, N1, D = q.shape
        params.setdefault("B", B); params.setdefault("N1", N1)
        params.setdefault("S1", S1); params.setdefault("D", D)
        if not layout_kv.startswith("PA_"):
            params.setdefault("N2", k.shape[2]); params.setdefault("S2", k.shape[1])
    elif layout_q == "TND":
        params.setdefault("N1", q.shape[1]); params.setdefault("D", q.shape[2])
        params.setdefault("N2", k.shape[1])
        cu_q = inputs.get("cu_seqlens_q")
        params.setdefault("B", (len(cu_q) - 1) if cu_q is not None else 1)
        params.setdefault("S1", q.shape[0])
        params.setdefault("S2", k.shape[0])
    elif layout_q.startswith("PA_"):
        params.setdefault("B", inputs.get("actual_b", 1))
        params.setdefault("N1", q.shape[1] if q.dim() >= 2 else 1)
        params.setdefault("D", q.shape[-1])
        params.setdefault("S1", q.shape[2] if q.dim() >= 3 else 1)

    if layout_kv.startswith("PA_"):
        skv = inputs.get("seqused_kv")
        if skv is not None:
            if isinstance(skv, torch.Tensor):
                params.setdefault("S2", int(skv.max()))
            elif isinstance(skv, list):
                params.setdefault("S2", max(skv))
        params.setdefault("N2", params.get("N1", 1))

    params.setdefault("DV", v.shape[-1])

    if "scale" not in params and "softmax_scale" in params:
        params["scale"] = params["softmax_scale"]

    return params


def _to_tforward(t: torch.Tensor, layout: str, params: dict) -> torch.Tensor:
    """转换为 tforward 期望的格式。"""
    if layout == "TND":
        return t.unsqueeze(0).permute(0, 2, 1, 3).contiguous().clone().float()
    if layout == "BSND":
        return t.permute(0, 2, 1, 3).contiguous().clone().float()
    if layout.startswith("PA_"):
        bnsd = pa_to_bnsd(t, layout, params)
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
