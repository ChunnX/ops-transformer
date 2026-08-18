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

"""TTK adapter for stem_indexer: generates metadata, then calls the main operator."""

import sys
from pathlib import Path

import torch

_CUSTOM_OPS_LOADED = False


def _check_exact_type(value, expected_type, name):
    if type(value) is not expected_type:
        raise TypeError(
            f"{name} must be {expected_type.__name__}, "
            f"but got {type(value).__name__}."
        )


def _ensure_custom_ops():
    global _CUSTOM_OPS_LOADED
    if not _CUSTOM_OPS_LOADED:
        pytest_dir = str(Path(__file__).resolve().parents[2] / "pytest")
        if pytest_dir not in sys.path:
            sys.path.insert(0, pytest_dir)
        import custom_ops  # noqa: F401

        _CUSTOM_OPS_LOADED = True


def stem_indexer(
    qflat: torch.Tensor,
    kflat: torch.Tensor,
    vbias: torch.Tensor,
    q_seq_lens: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    num_prompt_tokens: torch.Tensor,
    metadata: torch.Tensor,
    *,
    causal=True,
    stem_block_size=128,
    stem_stride=16,
    alpha=1.0,
    initial_blocks=4,
    window_size=4,
    k_block_num_rate_medium=0.2,
    k_block_num_bias_medium=30,
    k_block_num_rate_large=0.1,
    k_block_num_bias_large=30,
    topk_score_precision=1,
    metadata_mode: str = "auto",
    num_prompt_tokens_mode: str = "provided",
):
    """Generate metadata if needed, then call npu_stem_indexer."""
    _check_exact_type(causal, bool, "causal")
    _check_exact_type(stem_block_size, int, "stem_block_size")
    _check_exact_type(stem_stride, int, "stem_stride")
    _check_exact_type(alpha, float, "alpha")
    _check_exact_type(initial_blocks, int, "initial_blocks")
    _check_exact_type(window_size, int, "window_size")
    _check_exact_type(k_block_num_rate_medium, float, "k_block_num_rate_medium")
    _check_exact_type(k_block_num_bias_medium, int, "k_block_num_bias_medium")
    _check_exact_type(k_block_num_rate_large, float, "k_block_num_rate_large")
    _check_exact_type(k_block_num_bias_large, int, "k_block_num_bias_large")
    _check_exact_type(topk_score_precision, int, "topk_score_precision")

    _ensure_custom_ops()
    q_heads = int(qflat.shape[1])
    kv_heads = int(kflat.shape[1])

    if num_prompt_tokens_mode == "none":
        num_prompt_tokens = None

    if metadata_mode == "auto":
        metadata = torch.ops.custom.npu_stem_indexer_metadata(
            q_seq_lens,
            kv_seq_lens,
            q_heads,
            kv_heads,
            causal=causal,
            stem_block_size=stem_block_size,
            dim_qkflat=128,
            window_size=window_size,
        )
    elif metadata_mode == "none":
        metadata = None
    # "provided": 保留传入的 metadata 原样，不再重建

    return torch.ops.custom.npu_stem_indexer(
        qflat,
        kflat,
        vbias,
        q_seq_lens,
        kv_seq_lens,
        num_prompt_tokens=num_prompt_tokens,
        metadata=metadata,
        causal=causal,
        stem_block_size=stem_block_size,
        stem_stride=stem_stride,
        alpha=alpha,
        initial_blocks=initial_blocks,
        window_size=window_size,
        k_block_num_rate_medium=k_block_num_rate_medium,
        k_block_num_bias_medium=k_block_num_bias_medium,
        k_block_num_rate_large=k_block_num_rate_large,
        k_block_num_bias_large=k_block_num_bias_large,
        topk_score_precision=topk_score_precision,
    )
