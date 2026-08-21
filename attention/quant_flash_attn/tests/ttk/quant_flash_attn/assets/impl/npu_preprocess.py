#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Populate the QuantFlashAttn metadata slot before the main API call.

This module is intentionally independent from TTK.  The framework passes the
main API arguments after H2D; this hook invokes the companion torch operator
and updates ``metadata`` in place.  Profiling and result handling remain in
TTK, and the hook always returns ``None``.
"""

import importlib.util
import logging
import sys
from pathlib import Path

import torch


OPERATOR = "quant_flash_attn"
METADATA_INDEX = 8


def load_metadata_protocol():
    name = "qfa_ttk_metadata_protocol"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name("metadata_protocol.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def get_attribute(kwargs, name, default=None):
    value = kwargs.get(name)
    if value is None:
        value = kwargs.get(f"pytest_{name}")
    return default if value is None else value


def move_to_device(value, target):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device=target.device)
    return torch.as_tensor(value, device=target.device)


def build_metadata_arguments(kwargs, v_descale):
    """Build the metadata API arguments from main-API kwargs and v_descale.

    v_descale is the NPU descale tensor (slot 5), already materialized by
    inputs.py and moved to NPU by TTK H2D.  The metadata operator consumes
    it directly (e8m0 for mxfp8, FP32 for gqa_fp8).
    """
    layout_q = get_attribute(kwargs, "layout_q", "TND")
    layout_q_descale = get_attribute(kwargs, "layout_q_descale", "TND")
    layout_kv = get_attribute(kwargs, "layout_kv", "TND")
    layout_out = get_attribute(kwargs, "layout_out", "TND")
    quant_mode = int(get_attribute(kwargs, "quant_mode", 1))

    cu_seqlens_q = kwargs.get("cu_seqlens_q")
    cu_seqlens_kv = kwargs.get("cu_seqlens_kv")
    seqused_q = kwargs.get("seqused_q")
    seqused_kv = kwargs.get("seqused_kv")

    return {
        "num_heads_q": int(get_attribute(kwargs, "N_q")),
        "num_heads_kv": int(get_attribute(kwargs, "N_kv")),
        "head_dim": int(get_attribute(kwargs, "D")),
        "quant_mode": quant_mode,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_kv": cu_seqlens_kv,
        "seqused_q": seqused_q,
        "seqused_kv": seqused_kv,
        "v_descale": v_descale,
        "batch_size": get_attribute(kwargs, "batch_size"),
        "mask_mode": int(get_attribute(kwargs, "mask_mode", 0)),
        "win_left": int(get_attribute(kwargs, "win_left", -1)),
        "win_right": int(get_attribute(kwargs, "win_right", -1)),
        "layout_q": layout_q,
        "layout_q_descale": layout_q_descale,
        "layout_kv": layout_kv,
        "layout_out": layout_out,
        "max_seqlen_q": int(get_attribute(kwargs, "max_seqlen_q", -1)),
        "max_seqlen_kv": int(get_attribute(kwargs, "max_seqlen_kv", -1)),
    }


def run_metadata(arguments, metadata):
    return torch.ops.cann_ops_transformer.quant_flash_attn_metadata(
        int(arguments["num_heads_q"]),
        int(arguments["num_heads_kv"]),
        int(arguments["head_dim"]),
        int(arguments["quant_mode"]),
        cu_seqlens_q=move_to_device(arguments.get("cu_seqlens_q"), metadata),
        cu_seqlens_kv=move_to_device(arguments.get("cu_seqlens_kv"), metadata),
        seqused_q=move_to_device(arguments.get("seqused_q"), metadata),
        seqused_kv=move_to_device(arguments.get("seqused_kv"), metadata),
        v_descale=arguments.get("v_descale"),
        batch_size=arguments.get("batch_size"),
        max_seqlen_q=int(arguments["max_seqlen_q"]),
        max_seqlen_kv=int(arguments["max_seqlen_kv"]),
        mask_mode=int(arguments["mask_mode"]),
        win_left=int(arguments["win_left"]),
        win_right=int(arguments["win_right"]),
        layout_q=str(arguments["layout_q"]),
        layout_q_descale=str(arguments["layout_q_descale"]),
        layout_kv=str(arguments["layout_kv"]),
        layout_out=str(arguments["layout_out"]),
    )


def run(q, *, metadata=None, **kwargs):
    """Generate and copy metadata for a main QuantFlashAttn invocation."""
    del q
    if metadata is None:
        raise ValueError("QuantFlashAttn npu_preprocess requires metadata")
    protocol = load_metadata_protocol()
    testcase_name = kwargs.get("testcase_name")
    if protocol.metadata_is_materialized(metadata):
        logging.info("[%s] reuse nonzero QFA metadata input", testcase_name)
        return None

    v_descale = kwargs.get("dequant_scale_v")
    arguments = protocol.load_metadata_inputs(OPERATOR, testcase_name)
    if arguments is not None:
        source = "manual-data sidecar"
        # sidecar stores v_descale reference is not available; rebuild v_descale
        # from the main-API NPU tensor (slot 5) regardless of sidecar source.
        arguments["v_descale"] = v_descale
    else:
        arguments = build_metadata_arguments(kwargs, v_descale)
        source = "main API fallback (sidecar unavailable)"
    logging.info("[%s] build QFA metadata from %s", testcase_name, source)
    generated = run_metadata(arguments, metadata)
    if tuple(metadata.shape) != tuple(generated.shape):
        raise ValueError(
            "QFA metadata shape mismatch: "
            f"placeholder={tuple(metadata.shape)}, generated={tuple(generated.shape)}"
        )
    metadata.copy_(generated.to(dtype=metadata.dtype, device=metadata.device))
    rewritten = protocol.rewrite_metadata_input(
        OPERATOR,
        testcase_name,
        METADATA_INDEX,
        metadata,
    )
    if rewritten is not None:
        logging.info("[%s] rewrote QFA metadata input: %s", testcase_name, rewritten)
    return None
