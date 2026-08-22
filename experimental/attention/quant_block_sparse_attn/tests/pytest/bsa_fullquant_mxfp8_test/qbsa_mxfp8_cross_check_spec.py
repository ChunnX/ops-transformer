#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Dedicated TTK cross-check TestSpec for QBSA MXFP8 (quant_mode=2).

The local process generates the FP64 CPU reference and runs the QBSA NPU graph.
TTK's native remote-provider path executes ``LowPrecisionTorchProvider`` to
generate the low-precision CPU benchmark.  This file intentionally defines no
custom ``compare`` hook, so ``--compare cross_check`` reaches TTK's native
CrossCheckComparison unchanged.
"""

import importlib.util
import sys
from pathlib import Path

import torch


_CURRENT_DIR = Path(__file__).resolve().parent
_ASSETS_DIR = _CURRENT_DIR.parents[1] / "assets"
_ASSET_IMPL_CACHE = {}
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

# Keep this import at module scope.  When the spec runs remotely, TTK's native
# 424 flow sees the missing module during spec import and synchronizes the CPU
# reference before retrying the request.
import qbsa_mxfp8_golden as _REFERENCE


def _load_asset_impl(stem):
    """Load a normal QBSA asset implementation in the local TTK process."""
    if stem not in _ASSET_IMPL_CACHE:
        path = _ASSETS_DIR / "impl" / f"{stem}.py"
        if not path.is_file():
            raise RuntimeError(
                f"QBSA cross-check asset is unavailable in this process: {path}"
            )
        spec = importlib.util.spec_from_file_location(
            f"qbsa_cross_check_asset_{stem}_{abs(hash(path))}", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ASSET_IMPL_CACHE[stem] = module
    return _ASSET_IMPL_CACHE[stem]


def _load_local_graph():
    """Load the NPU graph only in the source-tree process, never remotely."""
    module_path = _ASSETS_DIR / "bsa_ttk_ops.py"
    if not module_path.is_file():
        return None
    assets_path = str(_ASSETS_DIR)
    if assets_path not in sys.path:
        sys.path.insert(0, assets_path)
    from bsa_ttk_ops import QuantBlockSparseAttnGraph

    return QuantBlockSparseAttnGraph


def _to_list(value):
    if value is None:
        return []
    if torch.is_tensor(value):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def _to_cpu(value):
    return value.detach().cpu() if torch.is_tensor(value) else value


def _numel(value):
    if value is None:
        return 0
    return int(value.numel()) if torch.is_tensor(value) else int(value.size)


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off", "")
    return bool(value)


class LowPrecisionTorchProvider:
    """Low-precision CPU benchmark executed by TTK's remote Torch provider."""

    def __call__(
        self,
        query,
        key,
        value,
        q_descale,
        k_descale,
        v_descale,
        sparse_indices,
        sparse_seq_len,
        p_scale,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        block_table,
        atten_mask,
        metadata,
        *,
        quant_mode=2,
        softmax_scale=1.0,
        mask_mode=3,
        sparse_block_size_q=128,
        sparse_block_size_kv=128,
        layout_q="TND",
        layout_kv="PA_BNBD",
        layout_out="TND",
        layout_sparse_indices="B_N_Qb_Kb",
        return_softmax_lse=False,
        quant_matmul=False,
        sparse_mode="dense",
        seed=0,
        s2_base_size=512,
        quant_group_size=32,
        p_scale_value=None,
        data_range_q=1.0,
        data_range_k=1.0,
        data_range_v=1.0,
        **kwargs,
    ):
        del atten_mask, metadata, kwargs
        if int(quant_mode) != 2:
            raise ValueError("QBSA MXFP8 cross-check provider only supports quant_mode=2")
        if _as_bool(quant_matmul):
            raise ValueError(
                "remote Torch provider only supports the CPU torch.matmul benchmark; "
                "quant_matmul must be false"
            )
        if _numel(cu_seqlens_kv) != 0 or _numel(seqused_q) != 0:
            raise ValueError("cu_seqlens_kv and seqused_q must be empty")

        q_prefix = _to_list(cu_seqlens_q)
        q_lengths = [
            q_prefix[index + 1] - q_prefix[index]
            for index in range(max(len(q_prefix) - 1, 0))
        ]
        kv_lengths = _to_list(seqused_kv)
        optional_p_scale = None if _numel(p_scale) == 0 else _to_cpu(p_scale)
        case = {
            "B": int(sparse_indices.shape[0]),
            "N1": int(query.shape[1]),
            "N2": int(key.shape[1]),
            "D": int(query.shape[-1]),
            "cu_seqlens_q": q_prefix,
            "cu_seqlens_kv": _to_list(cu_seqlens_kv),
            "seqused_q": _to_list(seqused_q),
            "seqused_kv": kv_lengths,
            "s2_base_size": int(s2_base_size),
            "blocknum": int(key.shape[0]),
            "max_block_per_batch": int(block_table.shape[1]),
            "block_size": int(key.shape[2]),
            "mask_mode": int(mask_mode),
            "fp8_dtype": query.dtype,
            "scale_dtype": q_descale.dtype,
            "quant_group_size": int(quant_group_size),
            "layout_q": layout_q,
            "layout_kv": layout_kv,
            "layout_sparse_indices": layout_sparse_indices,
            "layout_out": layout_out,
            "kv_cache_layout": "BnNBsD",
            "p_scale_value": (
                None if p_scale_value is None else float(p_scale_value)
            ),
            "softmax_scale": float(softmax_scale),
            "return_softmax_lse": _as_bool(return_softmax_lse),
            "seed": int(seed),
            "data_range_q": data_range_q,
            "data_range_k": data_range_k,
            "data_range_v": data_range_v,
            "device_id": 0,
            "sparse_q_block_size": int(sparse_block_size_q),
            "sparse_kv_block_size": int(sparse_block_size_kv),
            "sparse_mode": sparse_mode,
            "quant_mode": 2,
        }
        _REFERENCE.validate_mxfp8_case(case)
        _REFERENCE.set_active_case(case)
        attention_out, softmax_lse = _REFERENCE.cpu_mxfp8_golden(
            _to_cpu(query),
            _to_cpu(key),
            _to_cpu(value),
            _to_cpu(q_descale),
            _to_cpu(k_descale),
            _to_cpu(v_descale),
            optional_p_scale,
            q_lengths,
            kv_lengths,
            torch.tensor(q_prefix, dtype=torch.int32),
            _to_cpu(sparse_indices),
            _to_cpu(sparse_seq_len),
            _to_cpu(block_table),
            use_quant_matmul=False,
        )
        return [
            attention_out.to(torch.bfloat16),
            softmax_lse if _as_bool(return_softmax_lse) else None,
        ]


class QuantBlockSparseAttnMxfp8CrossCheckSpec:
    """MXFP8-only native cross-check specification."""

    @staticmethod
    def golden(*args, **kwargs):
        return _load_asset_impl("golden").cpu_mxfp8_cross_check_golden(
            *args, **kwargs
        )

    @staticmethod
    def customize_inputs(*args, **kwargs):
        return _load_asset_impl("inputs").customize_inputs(*args, **kwargs)

    torch_graph = _load_local_graph()
    third_party = {"torch": LowPrecisionTorchProvider}
    tolerance = {
        "bfloat16": {"standard": "cross_check", "level": "L0"},
        "float32": {"standard": "cross_check", "level": "L0"},
    }


__spec__ = {
    "bsa_ttk_ops.quant_block_sparse_attn": "QuantBlockSparseAttnMxfp8CrossCheckSpec",
}
