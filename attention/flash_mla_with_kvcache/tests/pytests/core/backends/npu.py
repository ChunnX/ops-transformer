# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""NPU 后端 — m0070 参数组调用 flash_mla_with_kvcache_metadata + flash_mla_with_kvcache，支持 eager / 静态图(export) 模式。"""

import numpy as np
import torch
from typing import Any, Dict

from .base import Backend

try:
    from cann_ops_transformer.ops import (
        flash_mla_with_kvcache,
        flash_mla_with_kvcache_metadata,
    )

    _HAS_NPU = True
except ImportError:
    _HAS_NPU = False

_AIC_CORE_NUM = 36
_AIV_CORE_NUM = 72
_FA_META_SIZE = 16
_FD_META_SIZE = 16
_HEADER_SIZE = 16

_FA_FIELDS = [
    "BN2_START",
    "M_START",
    "S2_START",
    "BN2_END",
    "M_END",
    "S2_END",
    "FIRST_FD_WS_IDX",
]
_FD_FIELDS = ["BN2_IDX", "M_IDX", "WS_IDX", "WS_NUM", "M_START", "M_NUM"]


def print_metadata(metadata):
    arr = metadata.cpu().contiguous().to(torch.int32).numpy().astype(np.uint32)

    section_num = int(arr[0])
    is_fd = int(arr[1])
    m_base_size = int(arr[2])
    s2_base_size = int(arr[3])

    SEP = "=" * 78
    DASH = "-" * 78

    print(f"\n{SEP}")
    print("  [Metadata Header]")
    print(f"  sectionNum  = {section_num}")
    print(f"  isFD        = {is_fd}")
    print(f"  mBaseSize   = {m_base_size}")
    print(f"  s2BaseSize  = {s2_base_size}")

    fa_base = _HEADER_SIZE
    fd_base = _HEADER_SIZE + section_num * _AIC_CORE_NUM * _FA_META_SIZE

    for sec in range(section_num):
        print(f"\n{DASH}")
        print(
            f"  [Section {sec}] FA Metadata — AIC cores  "
            f"(36 cores × 16 slots, 7 fields used)"
        )
        print(DASH)
        cw = 15
        hdr = f"  {'Core':<7}" + "".join(f"{n:>{cw}}" for n in _FA_FIELDS)
        print(hdr)
        print(
            f"  {'':<7}"
            + "".join(f"{'[' + str(i) + ']':>{cw}}" for i in range(len(_FA_FIELDS)))
        )
        print(f"  {'─' * 7}" + "─" * (cw * len(_FA_FIELDS)))
        for core in range(_AIC_CORE_NUM):
            off = fa_base + sec * _AIC_CORE_NUM * _FA_META_SIZE + core * _FA_META_SIZE
            vals = [int(arr[off + i]) for i in range(len(_FA_FIELDS))]
            note = "  (inactive)" if all(v == 0 for v in vals) else ""
            print(f"  AIC{core:02d}  " + "".join(f"{v:>{cw}}" for v in vals) + note)

    for sec in range(section_num):
        print(f"\n{DASH}")
        print(
            f"  [Section {sec}] FD Metadata — active AIV cores only  "
            f"(M_NUM > 0 shown, 72 cores × 16 slots, 6 fields used)"
        )
        print(DASH)
        cw = 12
        hdr = f"  {'Core':<7}" + "".join(f"{n:>{cw}}" for n in _FD_FIELDS)
        print(hdr)
        print(
            f"  {'':<7}"
            + "".join(f"{'[' + str(i) + ']':>{cw}}" for i in range(len(_FD_FIELDS)))
        )
        print(f"  {'─' * 7}" + "─" * (cw * len(_FD_FIELDS)))
        active = 0
        for core in range(_AIV_CORE_NUM):
            off = fd_base + sec * _AIV_CORE_NUM * _FD_META_SIZE + core * _FD_META_SIZE
            vals = [int(arr[off + i]) for i in range(len(_FD_FIELDS))]
            if vals[5] == 0:
                continue
            print(f"  AIV{core:02d} " + "".join(f"{v:>{cw}}" for v in vals))
            active += 1
        if active == 0:
            print("  (no active FD cores)")

    print(f"{SEP}\n")


class _FlashMlaWithKvcacheGraphNetwork(torch.nn.Module):
    """静态图（torch.export）包装器：metadata + flash_mla_with_kvcache 全链路灯。"""

    def forward(
        self,
        q,
        k_cache,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        seqused_q,
        attn_mask,
        num_heads_q,
        num_heads_kv,
        head_dim,
        batch_size,
        max_seqlen_q,
        max_seqlen_kv,
        mask_mode,
        layout_q,
        layout_kv,
        layout_out,
        softmax_scale,
        head_dim_v,
        return_softmax_lse,
    ):
        metadata = torch.ops.cann_ops_transformer.flash_mla_with_kvcache_metadata(
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            seqused_q=seqused_q,
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            mask_mode=mask_mode,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_out=layout_out,
        )
        out, lse = torch.ops.cann_ops_transformer.flash_mla_with_kvcache(
            q,
            k_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
            attn_mask=attn_mask,
            metadata=metadata,
            head_dim_v=head_dim_v,
            softmax_scale=softmax_scale,
            mask_mode=mask_mode,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_out=layout_out,
            return_softmax_lse=return_softmax_lse,
        )
        return out, lse


class NPUBackend(Backend):
    name = "npu"

    def __init__(self, device_id: int = 0, meta_only: bool = False):
        self._device = torch.device(f"npu:{device_id}")
        self._meta_only = meta_only
        if _HAS_NPU:
            try:
                torch.npu.set_device(device_id)
            except Exception:
                pass

    @property
    def device(self) -> torch.device:
        return self._device

    def is_available(self) -> bool:
        # 设备存在 + 算子绑定已注册（torch.ops.cann_ops_transformer.flash_mla_with_kvcache）
        return (
            _HAS_NPU
            and torch.npu.is_available()
            and hasattr(torch.ops.cann_ops_transformer, "flash_mla_with_kvcache")
        )

    def clear_cache(self):
        if _HAS_NPU:
            torch.npu.empty_cache()

    def _run_meta_and_kernel(self, inputs, params):
        from core.data import build_flash_mla_with_kvcache_params

        meta_kwargs, kernel_kwargs, _ = build_flash_mla_with_kvcache_params(
            params, self._device, inputs
        )
        metadata = flash_mla_with_kvcache_metadata(**meta_kwargs)
        if self._meta_only:
            print_metadata(metadata)
            t_total = inputs["q"].shape[0]
            n1 = inputs["q"].shape[1]
            dummy_out = torch.zeros(
                n1, t_total, params["D"], dtype=inputs["q"].dtype, device="cpu"
            )
            return dummy_out, None, None
        out, lse = flash_mla_with_kvcache(
            inputs["q"], inputs["k_cache"], metadata=metadata, **kernel_kwargs
        )
        torch.npu.synchronize()
        return out, lse, metadata

    def compute(
        self, inputs: Dict[str, torch.Tensor], params: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        out, lse, metadata = self._run_meta_and_kernel(inputs, params)
        return {"out": out, "lse": lse, "metadata": metadata}

    def compute_static(
        self, inputs: Dict[str, torch.Tensor], params: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """静态图路径（M9）：同一输入经 torch.export 导出为静态子图后执行。

        torch.export 对自定义 op（含 register_fake meta）支持 shape-固定 导出；
        执行时仍经 dispatcher 走 PrivateUse1 → 同一 aclnn kernel。若导出失败
        （如 torchair/ge 后端缺失等），抛异常由 orchestrator 记录为 gated。
        """
        from core.data import build_flash_mla_with_kvcache_params

        meta_kwargs, kernel_kwargs, _ = build_flash_mla_with_kvcache_params(
            params, self._device, inputs
        )

        net = _FlashMlaWithKvcacheGraphNetwork().npu()
        q, k_cache = inputs["q"], inputs["k_cache"]
        kw = kernel_kwargs
        args = (
            q,
            k_cache,
            kw.get("block_table"),
            kw.get("cache_seqlens"),
            kw.get("cu_seqlens_q"),
            kw.get("seqused_q"),
            kw.get("attn_mask"),
            meta_kwargs["num_heads_q"],
            meta_kwargs["num_heads_kv"],
            meta_kwargs["head_dim"],
            meta_kwargs["batch_size"],
            kw.get("max_seqlen_q", -1),
            kw.get("max_seqlen_kv", -1),
            meta_kwargs["mask_mode"],
            kw.get("layout_q", "TND"),
            kw.get("layout_kv", "NZ"),
            kw.get("layout_out", "NTD"),
            kw.get("softmax_scale", 1.0),
            params.get("D", 512),
            kw.get("return_softmax_lse", False),
        )
        # 预跑一次以分配实际张量（export 内部对 fake tensors 使用 meta impl）
        out, lse = net(*args)
        torch.npu.synchronize()

        exported = torch.export.export(net, args)
        out_s, lse_s = exported.module()(*args)
        torch.npu.synchronize()
        return {"out": out_s, "lse": lse_s, "metadata": None, "static": True}
