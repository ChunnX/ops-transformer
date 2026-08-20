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
"""Input customization for standalone QuantLightningIndexerMetadata cases."""

import numpy as np


_VECTOR_NAMES = (
    "cu_seqlens_q",
    "cu_seqlens_k",
    "seqused_q",
    "seqused_k",
    "cmp_residual_k",
)


def copy_values(target, values, name):
    if target is None:
        return
    if values is None:
        raise ValueError(f"QuantLightningIndexerMetadata requires {name}_values")
    source = np.asarray(values, dtype=np.int32)
    if tuple(source.shape) != tuple(target.shape):
        raise ValueError(
            f"QuantLightningIndexerMetadata {name} shape mismatch: "
            f"CSV={tuple(target.shape)}, values={tuple(source.shape)}"
        )
    if hasattr(target, "copy_"):
        import torch

        target.copy_(torch.as_tensor(source, dtype=target.dtype, device=target.device))
    else:
        np.copyto(target, source.astype(target.dtype, copy=False))


def generate_quant_lightning_indexer_metadata_inputs(
    num_heads_q,
    num_heads_k,
    head_dim,
    topk,
    quant_mode,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    **kwargs,
):
    """Copy explicit descriptor vectors into metadata API input tensors."""
    del num_heads_q, num_heads_k, head_dim, topk, quant_mode
    tensors = (
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
    )
    for name, tensor in zip(_VECTOR_NAMES, tensors):
        copy_values(tensor, kwargs.get(f"{name}_values"), name)
