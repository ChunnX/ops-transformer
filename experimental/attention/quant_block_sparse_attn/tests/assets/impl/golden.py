#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU golden adapter for quant_block_sparse_attn TTK cases."""

import importlib.util
import math
import random
import sys
from pathlib import Path

import torch

from bsa_ttk_ops import _fp8_make_block_table

PYTEST_GOLDEN_MODULE = None
PYTEST_MXFP8_GOLDEN_MODULE = None
_MXFP8_MODULE_NAME = "_qbsa_mxfp8_pytest_golden"


# ==================================================================================================
# Common helpers and TTK entry point
# ==================================================================================================


def to_list(value):
    if value is None:
        return []
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
    if hasattr(value, "reshape") and hasattr(value, "tolist"):
        return [int(x) for x in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(value)]


def to_cpu(value):
    return value.detach().cpu() if torch.is_tensor(value) else value


def _numel(value):
    if value is None:
        return 0
    return int(value.numel()) if torch.is_tensor(value) else int(value.size)


def lengths_from_prefix(value):
    vals = to_list(value)
    return (
        [vals[i + 1] - vals[i] for i in range(len(vals) - 1)] if len(vals) > 1 else []
    )


def _format_golden_outputs(attention_out, softmax_lse, return_softmax_lse):
    """Keep the operator's two output slots and invalidate disabled LSE."""
    return [attention_out, softmax_lse if return_softmax_lse else None]


def cpu_quant_block_sparse_attn(
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
    quant_mode=1,
    softmax_scale=1.0,
    mask_mode=3,
    blocksize=0,
    sparse_block_size_q=128,
    sparse_block_size_kv=128,
    layout_q="TND",
    layout_kv="PA_BNSD",
    layout_out="TND",
    layout_sparse_indices="B_N_Qb_Kb",
    return_softmax_lse=False,
    **kwargs,
):
    """Map the shared 16-input CSV signature to the existing pytest golden."""
    del metadata
    if _numel(cu_seqlens_kv) != 0:
        raise ValueError("cu_seqlens_kv must be empty")
    if _numel(seqused_q) != 0:
        raise ValueError("seqused_q must be empty")
    if p_scale is not None and _numel(p_scale) == 0:
        p_scale = None
    if int(quant_mode) != 2 and blocksize and int(key.shape[2]) != int(blocksize):
        raise ValueError("blocksize does not match key")

    common_kwargs = {
        "layout_kv": layout_kv,
        "layout_q": layout_q,
        "layout_sparse_indices": layout_sparse_indices,
        "layout_out": layout_out,
        "mask_mode": mask_mode,
        "return_softmax_lse": return_softmax_lse,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_kv": cu_seqlens_kv,
        "seqused_q": seqused_q,
        "seqused_kv": seqused_kv,
        "block_table": block_table,
        **kwargs,
    }
    if int(quant_mode) == 2:
        return _mxfp8_cpu_golden(
            query,
            key,
            value,
            q_descale,
            k_descale,
            v_descale,
            p_scale,
            sparse_indices,
            sparse_seq_len,
            atten_mask,
            softmax_scale,
            sparse_block_size_q,
            sparse_block_size_kv,
            **common_kwargs,
        )

    return _fp8_cpu_golden(
        query,
        key,
        value,
        q_descale,
        k_descale,
        v_descale,
        sparse_indices,
        sparse_seq_len,
        atten_mask,
        p_scale,
        softmax_scale=softmax_scale,
        sparse_q_block_size=sparse_block_size_q,
        sparse_kv_block_size=sparse_block_size_kv,
        quant_mode=quant_mode,
        **common_kwargs,
    )


# ==================================================================================================
# MXFP8-only helpers
# ==================================================================================================


def _mxfp8_load_pytest_golden_module():
    """Load the legacy MXFP8 reference under the same name as input.py."""
    global PYTEST_MXFP8_GOLDEN_MODULE
    if PYTEST_MXFP8_GOLDEN_MODULE is not None:
        return PYTEST_MXFP8_GOLDEN_MODULE
    if _MXFP8_MODULE_NAME in sys.modules:
        PYTEST_MXFP8_GOLDEN_MODULE = sys.modules[_MXFP8_MODULE_NAME]
        return PYTEST_MXFP8_GOLDEN_MODULE

    pytest_dir = (
        Path(__file__).resolve().parents[2] / "pytest" / "bsa_fullquant_mxfp8_test"
    )
    module_path = pytest_dir / "qbsa_mxfp8_golden.py"
    spec = importlib.util.spec_from_file_location(_MXFP8_MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load MXFP8 golden module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MXFP8_MODULE_NAME] = module
    sys.path.insert(0, str(pytest_dir))
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_MXFP8_MODULE_NAME, None)
        raise
    finally:
        try:
            sys.path.remove(str(pytest_dir))
        except ValueError:
            pass
    PYTEST_MXFP8_GOLDEN_MODULE = module
    return module


def _mxfp8_normalize_data_range(data_range):
    """Preserve scalar radius or normalize an explicit [min, max] range."""
    if isinstance(data_range, (list, tuple)):
        if len(data_range) != 2:
            raise ValueError(
                f"data_range must be a scalar or [min, max], got: {data_range}"
            )
        low, high = float(data_range[0]), float(data_range[1])
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError(
                f"data_range bounds must be finite, got: [{low}, {high}]"
            )
        if low > high:
            raise ValueError(
                f"data_range min must not exceed max, got: [{low}, {high}]"
            )
        return [low, high]

    radius = float(data_range)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError(
            f"scalar data_range must be finite and non-negative, got: {radius}"
        )
    return radius


def _mxfp8_cpu_golden(
    query,
    key,
    value,
    q_descale,
    k_descale,
    v_descale,
    p_scale,
    sparse_indices,
    sparse_seq_len,
    atten_mask,
    softmax_scale,
    sparse_q_block_size,
    sparse_kv_block_size,
    *,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    block_table=None,
    layout_kv="PA_BNSD",
    layout_q="TND",
    layout_sparse_indices="B_N_Qb_Kb",
    layout_out="TND",
    mask_mode=3,
    return_softmax_lse=False,
    sparse_mode="dense",
    seed=0,
    **kwargs,
):
    """Run the original pytest CPU reference on the exact customized inputs."""
    module = _mxfp8_load_pytest_golden_module()
    testcase_name = kwargs.get("testcase_name") or "__default__"
    cache = getattr(module, "_TTK_MXFP8_CACHE", {})
    cached = cache.pop(testcase_name, None)

    if cached is None:
        batch = int(sparse_indices.shape[0])
        q_prefix = to_list(cu_seqlens_q)
        q_lengths = lengths_from_prefix(cu_seqlens_q)
        kv_lengths = to_list(seqused_kv)
        case = {
            "B": batch,
            "N1": int(query.shape[1]),
            "N2": int(key.shape[1]),
            "D": int(query.shape[-1]),
            "cu_seqlens_q": q_prefix,
            "cu_seqlens_kv": to_list(cu_seqlens_kv),
            "seqused_q": to_list(seqused_q),
            "seqused_kv": kv_lengths,
            "s2_base_size": int(kwargs.get("s2_base_size", 512)),
            "blocknum": int(key.shape[0]),
            "max_block_per_batch": int(block_table.shape[1]),
            "block_size": int(key.shape[2]),
            "mask_mode": int(mask_mode),
            "fp8_dtype": torch.float8_e4m3fn,
            "scale_dtype": torch.float8_e8m0fnu,
            "quant_group_size": int(kwargs.get("quant_group_size", 32)),
            "layout_q": layout_q,
            "layout_kv": layout_kv,
            "layout_sparse_indices": layout_sparse_indices,
            "layout_out": layout_out,
            "kv_cache_layout": "BnNBsD",
            "is_contiguous": True,
            "p_scale_value": (
                None
                if kwargs.get("p_scale_value") is None
                else float(kwargs["p_scale_value"])
            ),
            "softmax_scale": float(softmax_scale),
            "return_softmax_lse": bool(return_softmax_lse),
            "seed": int(seed),
            "data_range_q": _mxfp8_normalize_data_range(
                kwargs.get("data_range_q", 1.0)
            ),
            "data_range_k": _mxfp8_normalize_data_range(
                kwargs.get("data_range_k", 1.0)
            ),
            "data_range_v": _mxfp8_normalize_data_range(
                kwargs.get("data_range_v", 1.0)
            ),
            "device_id": 0,
            "sparse_q_block_size": int(sparse_q_block_size),
            "sparse_kv_block_size": int(sparse_kv_block_size),
            "sparse_mode": sparse_mode,
            "quant_mode": 2,
        }
        module.validate_mxfp8_case(case)
        module.set_active_case(case)
        data = {
            "query": to_cpu(query),
            "key": to_cpu(key),
            "value": to_cpu(value),
            "q_descale": to_cpu(q_descale),
            "k_descale": to_cpu(k_descale),
            "v_descale": to_cpu(v_descale),
            "p_scale": None if p_scale is None else to_cpu(p_scale),
            "sparse_indices": to_cpu(sparse_indices),
            "sparse_seq_len": to_cpu(sparse_seq_len),
            "block_table": to_cpu(block_table),
            "q_lengths": q_lengths,
            "kv_lengths": kv_lengths,
            "cu_seqlens_q": torch.tensor(q_prefix, dtype=torch.int32),
        }
    else:
        case = cached["case"]
        data = cached["data"]
        module.set_active_case(case)

    attention_out, softmax_lse = module.cpu_mxfp8_golden(
        data["query"],
        data["key"],
        data["value"],
        data["q_descale"],
        data["k_descale"],
        data["v_descale"],
        data["p_scale"],
        data["q_lengths"],
        data["kv_lengths"],
        data["cu_seqlens_q"],
        data["sparse_indices"],
        data["sparse_seq_len"],
        data["block_table"],
    )
    attention_out = attention_out.to(torch.bfloat16)
    return _format_golden_outputs(
        attention_out, softmax_lse, return_softmax_lse
    )


# ==================================================================================================
# FP8-only helpers
# ==================================================================================================


def _fp8_load_pytest_golden_module():
    global PYTEST_GOLDEN_MODULE
    if PYTEST_GOLDEN_MODULE is not None:
        return PYTEST_GOLDEN_MODULE
    pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
    module_path = pytest_dir / "quant_block_sparse_attn_golden.py"
    sys.path.insert(0, str(pytest_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            f"bsa_pytest_golden_{abs(hash(module_path))}", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(pytest_dir))
        except ValueError:
            pass
    PYTEST_GOLDEN_MODULE = module
    return module


def _fp8_infer_geometry(
    query,
    key,
    sparse_indices,
    layout_q,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    max_seqlen_q=0,
    max_seqlen_kv=0,
):
    D = int(query.shape[-1])
    N2 = int(key.shape[1]) if key.dim() >= 2 else 1

    if layout_q == "BSND":
        B, S1, N1 = int(query.shape[0]), int(query.shape[1]), int(query.shape[2])
        q_lengths = to_list(seqused_q) or [S1] * B
    else:
        N1 = int(query.shape[1]) if layout_q == "TND" else int(query.shape[0])
        T = int(query.shape[0]) if layout_q == "TND" else int(query.shape[1])
        B = int(sparse_indices.shape[0]) if sparse_indices is not None else 1
        S1 = max_seqlen_q if max_seqlen_q > 0 else (T // B if B > 0 else T)
        q_lengths = to_list(seqused_q) or lengths_from_prefix(cu_seqlens_q) or [S1] * B

    kv_lengths = to_list(seqused_kv) or lengths_from_prefix(cu_seqlens_kv)
    if not kv_lengths:
        if max_seqlen_kv > 0:
            S2 = max_seqlen_kv
        elif key.dim() == 4:
            num_blocks = int(key.shape[0])
            block_size = int(key.shape[2])
            S2 = num_blocks * block_size
        else:
            S2 = int(key.shape[0])
        kv_lengths = [S2] * B if B > 0 else [S2]

    return B, S1, N1, N2, D, q_lengths, kv_lengths


def _fp8_lengths_to_prefix(lengths):
    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return torch.tensor(values, dtype=torch.int32)


def _fp8_auto_block_table(
    B, S2, block_size, pattern="sequential", seed=0, blocknum=None
):
    width = math.ceil(S2 / block_size)
    needed_block_num = B * width
    blocknum = needed_block_num if blocknum is None else int(blocknum)
    return _fp8_make_block_table(
        B, S2, block_size, blocknum, pattern, seed, torch.device("cpu")
    )


def _fp8_pack_pa_to_combined(
    key_pa, value_pa, k_scale_pa, block_size, layout_kv
):
    """Directly pack separate PA key/value/k_descale into a combined KV storage+meta.

    Same layout as bsa_ttk_ops._fp8_pack_combined_kv, so the reference reads exactly
    what the kernel reads (no PA->dense->PA round-trip).
    """
    golden_module = _fp8_load_pytest_golden_module()
    block_num = int(key_pa.shape[0])
    n2 = int(key_pa.shape[1])
    d_size = int(key_pa.shape[3])
    dv_size = int(value_pa.shape[3])
    meta = golden_module.combined_kv_cache.make_combined_kv_meta(
        block_num, block_size, n2, d_size, dv_size, 0, layout_kv
    )
    storage = torch.zeros((block_num * meta["pa_block_stride"],), dtype=torch.uint8)
    key_view, value_view, k_scale_view = golden_module.combined_kv_cache.make_combined_kv_views(
        storage, meta
    )
    key_view.copy_(key_pa)
    value_view.copy_(value_pa)
    k_scale_view.copy_(k_scale_pa if k_scale_pa.dim() == 4 else k_scale_pa.unsqueeze(-1))
    return storage, meta


def _fp8_cpu_golden(
    query,
    key,
    value,
    q_descale,
    k_descale,
    v_descale,
    sparse_indices,
    sparse_seq_len,
    atten_mask,
    p_scale=None,
    *,
    softmax_scale,
    sparse_q_block_size,
    sparse_kv_block_size,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    block_table=None,
    metadata=None,
    max_seqlen_q=0,
    max_seqlen_kv=0,
    layout_kv="PA_BNSD",
    layout_q="BSND",
    layout_sparse_indices="B_N_Qb_Kb",
    layout_out="TND",
    quant_mode=1,
    mask_mode=3,
    return_softmax_lse=False,
    **kwargs,
):
    """CPU reference implementation wrapping the existing pytest golden."""
    golden_module = _fp8_load_pytest_golden_module()

    q_cpu = to_cpu(query)
    key_cpu = to_cpu(key)
    value_cpu = to_cpu(value)
    q_descale_cpu = to_cpu(q_descale)
    k_descale_cpu = to_cpu(k_descale)
    v_descale_cpu = to_cpu(v_descale)
    p_scale_cpu = to_cpu(p_scale)
    sparse_indices_cpu = to_cpu(sparse_indices)
    sparse_seq_len_cpu = to_cpu(sparse_seq_len)
    cu_seqlens_q_cpu = to_cpu(cu_seqlens_q) if cu_seqlens_q is not None else None
    cu_seqlens_kv_cpu = to_cpu(cu_seqlens_kv) if cu_seqlens_kv is not None else None
    seqused_q_cpu = to_cpu(seqused_q) if seqused_q is not None else None
    seqused_kv_cpu = to_cpu(seqused_kv) if seqused_kv is not None else None
    block_table_cpu = to_cpu(block_table) if block_table is not None else None

    B, S1, N1, N2, D, q_lengths, kv_lengths = _fp8_infer_geometry(
        q_cpu,
        key_cpu,
        sparse_indices_cpu,
        layout_q,
        cu_seqlens_q_cpu,
        cu_seqlens_kv_cpu,
        seqused_q_cpu,
        seqused_kv_cpu,
        max_seqlen_q,
        max_seqlen_kv,
    )

    sparse_q_block_size = int(sparse_q_block_size)
    sparse_kv_block_size = int(sparse_kv_block_size)
    max_seqlen_q = int(max_seqlen_q) if max_seqlen_q > 0 else S1
    max_seqlen_kv = (
        int(max_seqlen_kv)
        if max_seqlen_kv > 0
        else max(kv_lengths)
        if kv_lengths
        else S1
    )
    S2 = max(kv_lengths) if kv_lengths else max_seqlen_kv

    if key_cpu.dim() != 4:
        raise ValueError(
            f"kv_cache-based reference requires 4D PA key, got {tuple(key_cpu.shape)}"
        )
    if block_table_cpu is None:
        block_table_cpu = _fp8_auto_block_table(
            B,
            S2,
            sparse_kv_block_size,
            str(kwargs.get("block_table_pattern", "sequential")),
            int(kwargs.get("seed", 0)),
            blocknum=int(key_cpu.shape[0]),
        )

    if cu_seqlens_q_cpu is None and layout_q != "BSND":
        cu_seqlens_q_cpu = _fp8_lengths_to_prefix(q_lengths)
    if cu_seqlens_kv_cpu is None and layout_q != "BSND":
        cu_seqlens_kv_cpu = _fp8_lengths_to_prefix(kv_lengths)
    cu_seqlens_q_inp = cu_seqlens_q_cpu

    v_scale = v_descale_cpu if v_descale_cpu.dim() == 1 else v_descale_cpu.view(-1)

    case = {
        "B": B,
        "S1": S1,
        "S2": S2,
        "N1": N1,
        "N2": N2,
        "D": D,
        "layout_q": layout_q,
        "softmax_scale": float(softmax_scale),
        "sparse_q_block_size": sparse_q_block_size,
        "sparse_kv_block_size": sparse_kv_block_size,
        "mask_mode": int(mask_mode),
        "output_dtype": torch.bfloat16,
        "actlen_mode": "full",
        "S1EQS2": False,
        "seed": int(kwargs.get("seed", 0)),
        "sparse_pattern": str(kwargs.get("sparse_pattern", "sequential")),
        "block_table_pattern": "sequential",
        "sparse_count": int(kwargs.get("sparse_count", 0)),
        "p_scale_value": float(p_scale_cpu[0].item()),
        "pa_block_padding_bytes": 0,
        "layout_kv": layout_kv,
        "layout_sparse_indices": layout_sparse_indices,
        "layout_out": layout_out,
        "quant_mode": int(quant_mode),
    }

    # 直接从 PA 张量打包组合 kv_cache（布局与 bsa_ttk_ops._fp8_pack_combined_kv 一致），
    # 保证参考读到的是 kernel 实际读取的同一份缓存，不再做 PA->dense->PA 往返。
    kv_cache_storage, kv_cache_meta = _fp8_pack_pa_to_combined(
        key_cpu, value_cpu, k_descale_cpu, int(key_cpu.shape[2]), layout_kv
    )
    attention_out, softmax_lse = golden_module._reference_attention(
        case,
        q_cpu,
        kv_cache_storage,
        kv_cache_meta,
        block_table_cpu,
        q_descale_cpu,
        v_scale,
        p_scale_cpu,
        sparse_indices_cpu,
        sparse_seq_len_cpu,
        cu_seqlens_q_inp,
        q_lengths,
        kv_lengths,
    )

    return _format_golden_outputs(
        attention_out, softmax_lse, return_softmax_lse
    )
