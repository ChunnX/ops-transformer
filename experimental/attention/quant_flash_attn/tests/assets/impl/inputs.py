#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR
# PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""输入生成插件：原位填充 quant_flash_attn (MXFP8 SF16) 全部 15 个 tensor slot。"""

import logging
import os
import sys

import numpy
import torch

_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_ROOT = os.path.dirname(_ASSETS_DIR)
if _ASSETS_ROOT not in sys.path:
    sys.path.insert(0, _ASSETS_ROOT)
import quant_flash_attn_mxfp8_sf16_golden as golden_mod

logger = logging.getLogger(__name__)


def _apply_golden_globals(attrs):
    golden_mod._apply_golden_globals(attrs)
    golden_mod._apply_kwargs_globals(attrs)


def _is_numpy_arr(x):
    return isinstance(x, numpy.ndarray)


def _inplace_write(dst, src, slot_name):
    """原位写入，兼容 numpy（ttk CPU backend 分配）/ torch 双形态 slot。

    fp4/fp8 dtype：torch→numpy 或跨 fp8 变体均走位视图（bit 兼容，不走数值转换）。
    """
    if dst is None:
        if src is not None and (not torch.is_tensor(src) or src.numel() > 0):
            raise ValueError(f"[INPUTS] {slot_name}: dst is None but src is not empty")
        return
    if src is None:
        if _is_numpy_arr(dst):
            dst[...] = 0
        else:
            dst.zero_()
        return
    src_t = src if torch.is_tensor(src) else torch.as_tensor(src)
    if tuple(dst.shape) != tuple(src_t.shape):
        raise ValueError(
            f"[INPUTS] {slot_name} shape mismatch: slot {tuple(dst.shape)} "
            f"!= computed {tuple(src_t.shape)}"
        )
    if _is_numpy_arr(dst):
        dst_str = str(dst.dtype)
        if "float4" in dst_str or "float8" in dst_str:
            # torch fp8 → numpy 走 uint8 位桥（本环境 torch .numpy() 不支持 fp8 ScalarType，
            # 与参考仓实现的差异点；位级等价：fp8 与 uint8 同 1 字节）
            src_np = (
                src_t.detach()
                .cpu()
                .contiguous()
                .view(torch.uint8)
                .numpy()
                .view(dst.dtype)
            )
        else:
            src_np = src_t.detach().cpu().contiguous().numpy().astype(dst.dtype)
        dst[...] = src_np
    else:
        dst_str = str(dst.dtype)
        if "float4" in dst_str or "float8" in dst_str:
            dst.copy_(src_t.view(dst.dtype))
        else:
            dst.copy_(src_t.to(dst.dtype))


def _write_int32_list(slot, values, slot_name):
    if slot is None:
        return
    if _is_numpy_arr(slot):
        if values:
            arr = numpy.array(list(values), dtype=slot.dtype)
            if tuple(slot.shape) != tuple(arr.shape):
                raise ValueError(
                    f"[INPUTS] {slot_name} shape mismatch: slot {tuple(slot.shape)} "
                    f"!= computed {tuple(arr.shape)}"
                )
            slot[...] = arr
        else:
            slot[...] = 0
        return
    if values:
        src = torch.tensor(list(values), dtype=torch.int32)
        if tuple(slot.shape) != tuple(src.shape):
            raise ValueError(
                f"[INPUTS] {slot_name} shape mismatch: slot {tuple(slot.shape)} "
                f"!= computed {tuple(src.shape)}"
            )
        slot.copy_(src.to(slot.dtype))
    else:
        slot.zero_()


def _resolve_metadata_op():
    """惰性解析 metadata 算子入口，优先 python 包路径，回退 torch.ops。"""
    try:
        from cann_ops_transformer.ops import quant_flash_attn_metadata

        return quant_flash_attn_metadata
    except ImportError:
        import torch_npu  # noqa: F401

        return torch.ops.cann_ops_transformer.quant_flash_attn_metadata


def _build_metadata(data_dict):
    """调用 quant_flash_attn_metadata 构建 metadata 槽内容（int32 (2, X) CPU 张量）。

    本版 ttk 无 npu_preprocess 钩子（参考仓的该插件是死代码）——metadata 只能作为
    普通 input slot 在输入生成阶段构建：e2e 直接模式随 H2D 上板；prepare 模式落盘
    的即为有效数据（原实现落盘未初始化内存垃圾，replay 后 kernel 读垃圾 sectionNum
    越界调度 → 设备错误 507015）。
    """
    import torch_npu  # noqa: F401

    seqused_q = data_dict.get("act_seq_lens_q")
    seqused_kv = data_dict.get("act_seq_lens_kv")
    sq_t = torch.tensor(list(seqused_q), dtype=torch.int32).npu() if seqused_q else None
    skv_t = (
        torch.tensor(list(seqused_kv), dtype=torch.int32).npu() if seqused_kv else None
    )

    md = _resolve_metadata_op()(
        num_heads_q=golden_mod.N_q,
        num_heads_kv=golden_mod.N_kv,
        head_dim=golden_mod.D,
        quant_mode=golden_mod.QUANT_MODE,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        seqused_q=sq_t,
        seqused_kv=skv_t,
        mask_mode=0,
        layout_q="BNSD",
        layout_q_descale="BNSD",
        layout_kv="BNSD",
        layout_out="BNSD",
        max_seqlen_q=golden_mod.MAX_SEQLEN_Q,
        max_seqlen_kv=golden_mod.MAX_SEQLEN_KV,
    )
    logger.info("[INPUTS] metadata built: shape=%s", tuple(md.shape))
    return md.cpu()


def generate_qfa_mxfp8_sf16_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    quant_mode: int,
    block_table: torch.Tensor = None,
    p_scale: torch.Tensor = None,
    cu_seqlens_q: torch.Tensor = None,
    cu_seqlens_kv: torch.Tensor = None,
    seqused_q: torch.Tensor = None,
    seqused_kv: torch.Tensor = None,
    sinks: torch.Tensor = None,
    attn_mask: torch.Tensor = None,
    metadata: torch.Tensor = None,
    softmax_scale: float = 1.0,
    mask_mode: int = 0,
    win_left: int = -1,
    win_right: int = -1,
    max_seqlen_q: int = -1,
    max_seqlen_kv: int = -1,
    layout_q: str = "BNSD",
    layout_q_descale: str = "BNSD",
    layout_kv: str = "BNSD",
    layout_out: str = "BNSD",
    return_softmax_lse: bool = False,
    **kwargs,
):
    """原位生成 quant_flash_attn (MXFP8 Softmax FP16) 全部 15 个 tensor slot 的真实数据。

    Signature 镜像 torch.ops.cann_ops_transformer.quant_flash_attn。
    """
    attrs = dict(kwargs)
    attrs.setdefault("quant_mode", quant_mode)
    attrs.setdefault("softmax_scale", softmax_scale)
    attrs.setdefault("mask_mode", mask_mode)
    attrs.setdefault("win_left", win_left)
    attrs.setdefault("win_right", win_right)
    attrs.setdefault("max_seqlen_q", max_seqlen_q)
    attrs.setdefault("max_seqlen_kv", max_seqlen_kv)
    attrs.setdefault("layout_q", layout_q)
    attrs.setdefault("layout_q_descale", layout_q_descale)
    attrs.setdefault("layout_kv", layout_kv)
    attrs.setdefault("layout_out", layout_out)
    attrs.setdefault("return_softmax_lse", return_softmax_lse)
    _apply_golden_globals(attrs)

    golden_mod._inject_physical_s_override(q, v, layout_q, layout_kv)
    try:
        data_dict = golden_mod.generate_data()
    finally:
        golden_mod._clear_physical_s_override()

    _inplace_write(q, data_dict["q"], "q (slot 0)")
    _inplace_write(k, data_dict["k"], "k (slot 1)")
    _inplace_write(v, data_dict["v"], "v (slot 2)")
    _inplace_write(q_descale, data_dict["q_descale"], "q_descale (slot 3)")
    _inplace_write(k_descale, data_dict["k_descale"], "k_descale (slot 4)")
    _inplace_write(v_descale, data_dict["v_descale"], "v_descale (slot 5)")
    _inplace_write(block_table, data_dict.get("block_table"), "block_table (slot 6)")
    _inplace_write(p_scale, data_dict.get("p_scale"), "p_scale (slot 7)")
    _write_int32_list(
        cu_seqlens_q, data_dict.get("cu_seqlens_q"), "cu_seqlens_q (slot 8)"
    )
    _write_int32_list(
        cu_seqlens_kv, data_dict.get("cu_seqlens_kv"), "cu_seqlens_kv (slot 9)"
    )
    _write_int32_list(seqused_q, data_dict.get("act_seq_lens_q"), "seqused_q (slot 10)")
    _write_int32_list(
        seqused_kv, data_dict.get("act_seq_lens_kv"), "seqused_kv (slot 11)"
    )
    _inplace_write(sinks, data_dict.get("sinks"), "sinks (slot 12)")
    _inplace_write(attn_mask, data_dict.get("attn_mask"), "attn_mask (slot 13)")
    # metadata 槽（slot 14）：输入生成阶段构建（见 _build_metadata 注释）
    _inplace_write(metadata, _build_metadata(data_dict), "metadata (slot 14)")

    logger.info(
        "[INPUTS] wrote MXFP8 SF16 q/k/v (q=%s), descale (dq=%s, dk=%s, dv=%s)",
        tuple(q.shape),
        tuple(q_descale.shape),
        tuple(k_descale.shape),
        tuple(v_descale.shape),
    )
