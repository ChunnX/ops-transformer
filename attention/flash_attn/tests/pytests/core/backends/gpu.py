"""GPU 后端实现 — 封装 flash_attn 库调用，支持 PA/TND/BNSD 输入格式。"""

import torch
from typing import Any, Dict

from .base import Backend

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
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

    def compute(self, inputs: Dict[str, torch.Tensor],
                params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        layout_q = params.get("layout_q", params.get("input_layout", "BNSD"))
        layout_kv = params.get("layout_kv", layout_q)

        if layout_kv.startswith("PA_"):
            return self._compute_pa(inputs, params, layout_q, layout_kv)
        elif layout_q == "TND":
            return self._compute_tnd(inputs, params)
        else:
            return self._compute_bnsd(inputs, params)

    def _compute_bnsd(self, inputs, params):
        q, k, v = inputs["q"], inputs["k"], inputs["v"]
        scale = params.get("scale", q.shape[-1] ** (-0.5))
        mask_mode = params.get("mask_mode", 0)
        causal = int(mask_mode) == 3

        q_bshd = q.permute(0, 2, 1, 3).contiguous()
        k_bshd = k.permute(0, 2, 1, 3).contiguous()
        v_bshd = v.permute(0, 2, 1, 3).contiguous()

        out_bshd = flash_attn_func(
            q_bshd, k_bshd, v_bshd,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=causal,
            window_size=(-1, -1),
        )
        out_bnsd = out_bshd.permute(0, 2, 1, 3).contiguous()
        return {"out": out_bnsd}

    def _compute_tnd(self, inputs, params):
        q, k, v = inputs["q"], inputs["k"], inputs["v"]
        scale = params.get("scale", q.shape[-1] ** (-0.5))
        mask_mode = params.get("mask_mode", 0)
        causal = int(mask_mode) == 3

        cu_seqlens_q = params.get("cu_seqlens_q")
        cu_seqlens_kv = params.get("cu_seqlens_kv", cu_seqlens_q)

        cu_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=q.device)
        cu_kv = torch.tensor(cu_seqlens_kv, dtype=torch.int32, device=q.device)

        seqlens_q = [cu_seqlens_q[i + 1] - cu_seqlens_q[i] for i in range(len(cu_seqlens_q) - 1)]
        seqlens_kv = [cu_seqlens_kv[i + 1] - cu_seqlens_kv[i] for i in range(len(cu_seqlens_kv) - 1)]
        max_seqlen_q = max(seqlens_q) if seqlens_q else 0
        max_seqlen_kv = max(seqlens_kv) if seqlens_kv else 0

        out_tnd = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_kv,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=causal,
            window_size=(-1, -1),
        )
        return {"out": out_tnd}

    def _compute_pa(self, inputs, params, layout_q, layout_kv):
        """PA 路径：将 PA KV 还原为 dense，再走 flash_attn_varlen_func。"""
        q, k, v = inputs["q"], inputs["k"], inputs["v"]
        block_table = inputs["block_table"]
        scale = params.get("scale", q.shape[-1] ** (-0.5))
        mask_mode = params.get("mask_mode", 0)
        causal = int(mask_mode) == 3
        seqused_kv = params["seqused_kv"]
        seqused_q = params["seqused_q"]
        cu_seqlens_q = params["cu_seqlens_q"]
        actual_b = len(cu_seqlens_q) - 1
        block_size = params.get("block_size", 128)
        n2 = params.get("N2", params["N1"])

        k_dense = self._pa_to_dense_tnd(k, layout_kv, block_table, seqused_kv, actual_b, block_size, n2)
        v_dense = self._pa_to_dense_tnd(v, layout_kv, block_table, seqused_kv, actual_b, block_size, n2)

        cu_kv = [0]
        for i in range(actual_b):
            cu_kv.append(cu_kv[-1] + int(seqused_kv[i]))

        cu_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=q.device)
        cu_kv_t = torch.tensor(cu_kv, dtype=torch.int32, device=q.device)

        max_seqlen_q = max(seqused_q)
        max_seqlen_kv = max(seqused_kv)

        if layout_q == "TND":
            q_tnd = q
        else:
            raise ValueError("GPU PA 路径当前仅支持 layout_q=TND")

        out_tnd = flash_attn_varlen_func(
            q_tnd, k_dense, v_dense,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_kv_t,
            max_seqlen_q=int(max_seqlen_q),
            max_seqlen_k=int(max_seqlen_kv),
            dropout_p=0.0,
            softmax_scale=scale,
            causal=causal,
            window_size=(-1, -1),
        )
        return {"out": out_tnd}

    def _pa_to_dense_tnd(self, t, layout, block_table, seqused_kv, actual_b, block_size, n2):
        """PA 格式 → dense TND (total_kv, N2, D)，按 seqused_kv 紧密排列。"""
        if isinstance(block_table, torch.Tensor):
            bt = block_table.cpu().numpy()
        else:
            bt = block_table

        total_kv = sum(int(seqused_kv[i]) for i in range(actual_b))
        d = t.shape[-1]
        result = torch.zeros(total_kv, n2, d, dtype=t.dtype, device=t.device)

        offset = 0
        for bi in range(actual_b):
            slen = int(seqused_kv[bi])
            nblk = (slen + block_size - 1) // block_size
            for bj in range(nblk):
                blk_id = int(bt[bi][bj])
                if blk_id < 0 or blk_id >= t.shape[0]:
                    continue
                start = bj * block_size
                end = min(start + block_size, slen)
                cnt = end - start
                if cnt <= 0:
                    continue
                if layout == "PA_BNBD":
                    # (num_blocks, N2, block_size, D) → 取 [blk_id, :, :cnt, :]
                    result[offset + start:offset + end, :, :] = t[blk_id, :, :cnt, :].permute(1, 0, 2)
                elif layout == "PA_BBND":
                    # (num_blocks, block_size, N2, D) → 取 [blk_id, :cnt, :, :]
                    result[offset + start:offset + end, :, :] = t[blk_id, :cnt, :, :]
            offset += slen

        return result
