# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""声明式张量生成 + MLA(m0070) 参数组构造。

flash_mla_with_kvcache 接口（m0070，无 v / 无独立 q_rope/k_rope）：
    flash_mla_with_kvcache(q, k_cache, block_table, cache_seqlens, cu_seqlens_q, seqused_q, attn_mask, metadata,
              head_dim_v=512, softmax_scale, mask_mode, max_seqlen_q, max_seqlen_kv,
              layout_q="TND", layout_kv="PA_NZ", layout_out="NTD", return_softmax_lse) -> (attn_out, softmax_lse)
布局仅支持 TND/NZ/NTD（R1）：q = [total_tokens, N1, 576]，k_cache = PA_NZ [num_blocks, 1, 36, block_size, 16]
（576 = nope 512 + rope 64，合并进最后维），out = NTD = [N1, total_tokens, 512]。
"""

import torch
from typing import Any, Callable, Dict, List, Optional, Tuple

# MLA 常量（R1 / m0070）
MLA_D_TOTAL = 576          # q/k_cache 最后维 = nope 512 + rope 64
MLA_HEAD_DIM_V = 512       # head_dim_v（nope/value 宽）
MLA_NZ_BLK_ELEM = 16       # fp16/bf16 的 NZ 块内元素数（32B / 2B）
MLA_ROPE_DIM = 64


# ============================================================
# 参数组构造 — MLA (m0070)
# ============================================================

def build_flash_mla_with_kvcache_params(p: dict, device: torch.device, inputs: dict
                           ) -> Tuple[dict, dict, str]:
    """从 normalized params 构造 (meta_kwargs, kernel_kwargs, out_layout)。

    meta_kwargs  → flash_mla_with_kvcache_metadata（AICPU 分核预算）
    kernel_kwargs → flash_mla_with_kvcache（主算子）
    """
    layout_q = p.get("layout_q", "TND")
    layout_kv = p.get("layout_kv", "PA_NZ")
    layout_out = p.get("layout_out", "NTD")
    d = p["D"]  # 512（qk nope / head_dim_v）

    scale = p.get("scale", 1 / (d ** 0.5))
    mask_mode = p.get("mask_mode", 0)

    # mask_mode=3 (CAUSAL/RIGHT_DOWN)：attn_mask 必传 INT8 [2048,2048]（checker 语义）
    attn_mask = None
    if mask_mode == 3:
        mask = torch.triu(torch.ones(2048, 2048), diagonal=1)
        attn_mask = mask.to(dtype=torch.int8, device=device)

    def _u32(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.to(torch.int32, device=device).contiguous()
        return torch.tensor(x, dtype=torch.int32, device=device).contiguous()

    cu_q = _u32(p.get("cu_seqlens_q"))
    cache_lens = _u32(p.get("cache_seqlens"))
    sq = _u32(p.get("seqused_q"))

    def _resolve_max_seqlen(explicit, used_seq, cu_seqlens, fallback):
        if explicit is not None:
            return int(explicit)
        if used_seq is not None:
            return int(used_seq.max())
        if cu_seqlens is not None:
            return int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
        return fallback

    q = inputs["q"]
    nqh = q.shape[1]
    bs = p.get("B", 1)

    msq = _resolve_max_seqlen(p.get("max_seqlen_q"), sq, cu_q, p.get("S1", 1))
    msk = _resolve_max_seqlen(p.get("max_seqlen_kv"), cache_lens, None, p.get("S2", p.get("S1", 1)))

    # === metadata kwargs ===
    meta_kwargs = dict(
        num_heads_q=nqh, num_heads_kv=1, head_dim=MLA_HEAD_DIM_V,
        cu_seqlens_q=cu_q, cache_seqlens=cache_lens, seqused_q=sq,
        batch_size=bs, max_seqlen_q=msq, max_seqlen_kv=msk,
        mask_mode=mask_mode,
        layout_q=layout_q, layout_kv=layout_kv, layout_out=layout_out,
    )

    # === kernel kwargs ===
    kernel_kwargs = dict(
        softmax_scale=scale, mask_mode=mask_mode,
        block_table=None if inputs.get("block_table") is None else inputs["block_table"].to(device),
        cache_seqlens=cache_lens, cu_seqlens_q=cu_q, seqused_q=sq,
        attn_mask=attn_mask,
        max_seqlen_q=msq, max_seqlen_kv=msk,
        layout_q=layout_q, layout_kv=layout_kv, layout_out=layout_out,
        return_softmax_lse=p.get("return_softmax_lse", False),
    )
    return meta_kwargs, kernel_kwargs, layout_out


class InputSpec:
    """声明式张量数据生成器（用法同 flash_attn 的 harness）。"""

    def __init__(self, name: str = ""):
        self.name = name
        self._items: List[dict] = []

    def tensor(self, name: str, *,
               shape: str = "",
               layout: dict = None,
               dtype=None,
               method: str = "randn",
               when: Callable = None,
               range_key: str = None,
               **kwargs) -> "InputSpec":
        self._items.append(dict(
            name=name, shape=shape, layout=layout or {},
            dtype=dtype, method=method, when=when,
            range_key=range_key, kwargs=kwargs,
        ))
        return self

    def generate(self, params: dict, device,
                 layout: str = "TND", seed: int = 42) -> Dict[str, torch.Tensor]:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        tensors = {}
        for it in self._items:
            if it["when"] and not it["when"](params):
                continue
            # k_cache 的布局键 = layout_kv（PA_NZ/NZ），其余按 input layout
            eff_layout = params.get("layout_kv", layout) if it["name"] == "k_cache" else layout
            shape = self._resolve(it, params, eff_layout)
            if not shape:
                continue
            dt = it["dtype"] or _infer_dtype(params.get("Dtype", "fp16"))
            kw = dict(it["kwargs"])
            rk = it.get("range_key")
            if rk and rk in params:
                lo, hi = params[rk]
                kw["low"], kw["high"] = float(lo), float(hi)
            t = _generate(it["method"], shape, dt, device, rng, **kw)
            tensors[it["name"]] = t
        return tensors

    def _resolve(self, item, params, layout):
        shape_str = item["layout"].get(layout, item["shape"])
        if not shape_str:
            return None
        dims = []
        for token in shape_str.split(","):
            token = token.strip()
            if token in params:
                dims.append(params[token])
            elif token == "total_s1":
                cu = params.get("cu_seqlens_q")
                dims.append(cu[-1] if cu else params.get("B", 1) * params.get("S1", 1))
            elif token == "max_blocks":
                sv = params.get("cache_seqlens", [params.get("S2", 1)])
                bs = params.get("block_size", 128)
                sm = max(sv) if isinstance(sv, list) else sv
                dims.append((sm + bs - 1) // bs if sm > 0 else 1)
            elif token == "num_blocks":
                dims.append(params.get("num_blocks", 1))
            else:
                dims.append(int(token))
        return tuple(dims)


def _infer_dtype(ds: str) -> torch.dtype:
    return torch.float16 if ds == "fp16" else torch.bfloat16


def _generate(method, shape, dtype, device, rng, **kw):
    if method == "randn":
        return torch.randn(shape, dtype=dtype, device=device, generator=rng)
    if method == "uniform":
        lo, hi = kw.get("low", 0.0), kw.get("high", 1.0)
        t = torch.rand(shape, dtype=dtype, device=device, generator=rng)
        return t * (hi - lo) + lo
    if method == "zeros":
        return torch.zeros(shape, dtype=dtype, device=device)
    if method == "custom":
        return kw["fn"](shape, dtype, device, rng)
    raise ValueError(f"Unknown method: {method}")


# ── MLA 输入 recipe（R1：q=TND 576 宽，k_cache=PA_NZ 576 宽，无 v）──
flash_mla_with_kvcache_inputs = (
    InputSpec("flash_mla_with_kvcache")
    .tensor("q", layout={"TND": "total_s1,N1,576"},
            method="uniform", low=-5.0, high=5.0, range_key="q_range",
            when=lambda p: "q_range" in p)
    .tensor("q", layout={"TND": "total_s1,N1,576"},
            when=lambda p: "q_range" not in p)
    # k_cache: PA_NZ [num_blocks, n2=1, D_nz_sub, block_size, nz_blk_elem]
    # D_nz_sub = 576 / 16 = 36（fp16/bf16 的 NZ 块）
    .tensor("k_cache", layout={"PA_NZ": "num_blocks,1,D_nz_sub,block_size,nz_blk_elem"},
            method="uniform", low=-5.0, high=5.0, range_key="k_range",
            when=lambda p: "k_range" in p)
    .tensor("k_cache", layout={"PA_NZ": "num_blocks,1,D_nz_sub,block_size,nz_blk_elem"},
            when=lambda p: "k_range" not in p)
)