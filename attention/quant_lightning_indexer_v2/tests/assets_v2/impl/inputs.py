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

"""Input customization for QuantLightningIndexer V2 metadata-input TTK cases."""

import importlib.util
import sys
from pathlib import Path

import torch

import cann_ops_transformer  # noqa: F401 - registers torch.ops namespace


class PrimaryInputAdapterLoader:
    """Load the shared adapter from the primary assets implementation."""

    @staticmethod
    def load():
        name = "qli_v2_ttk_primary_inputs"
        if name in sys.modules:
            return sys.modules[name]
        path = Path(__file__).resolve().parents[2] / "assets" / "impl" / "inputs.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
        return module


PrimaryInputAdapter = PrimaryInputAdapterLoader.load().QuantLightningIndexerV2InputAdapter


class QuantLightningIndexerV2MetadataInputAdapter(PrimaryInputAdapter):
    """Extend pytest case generation with the explicit metadata input."""

    @staticmethod
    def load_golden_store():
        name = "qli_v2_ttk_v2_golden"
        if name in sys.modules:
            return sys.modules[name]
        path = Path(__file__).with_name("golden.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
        return module

    @staticmethod
    def fill_metadata(metadata, generated):
        if metadata is None:
            raise ValueError("metadata input must be declared in the CSV")
        if tuple(metadata.shape) != tuple(generated.shape):
            raise ValueError(
                f"metadata shape mismatch: CSV={tuple(metadata.shape)} "
                f"generated={tuple(generated.shape)}"
            )
        metadata.copy_(generated.to(dtype=metadata.dtype, device=metadata.device))

    def generate_metadata(self, query, key, cu_seqlens_q, cu_seqlens_k,
                          seqused_q, seqused_k, cmp_residual_k,
                          metadata, layout_query, layout_key, data, kwargs):
        q_heads = int(query.shape[2]) if layout_query == "BSND" else int(query.shape[1])
        k_heads = int(key.shape[1]) if layout_key == "TND" else int(key.shape[2])
        generated = torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(
            num_heads_q=q_heads,
            num_heads_k=k_heads,
            head_dim=int(query.shape[-1]),
            topk=int(kwargs.get("sparse_count", 0)),
            quant_mode=int(kwargs.get("quant_mode", 1)),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            cmp_residual_k=cmp_residual_k,
            batch_size=int(data["params"][0]),
            max_seqlen_q=int(data["max_seqlen_q_meta"]),
            max_seqlen_k=int(data["max_seqlen_k_meta"]),
            layout_q=layout_query,
            layout_k=layout_key,
            mask_mode=int(kwargs.get("sparse_mode", 0)),
            cmp_ratio=int(kwargs.get("cmp_ratio", 1)),
        )
        self.fill_metadata(metadata, generated)


INPUT_ADAPTER = QuantLightningIndexerV2MetadataInputAdapter()


def generate_qli_v2_inputs(query, key, weights, query_dequant_scale,
                           key_dequant_scale, *, cu_seqlens_q=None,
                           cu_seqlens_k=None, seqused_q=None, seqused_k=None,
                           cmp_residual_k=None, block_table=None,
                           output_idx_offset=None, metadata=None,
                           layout_query="BSND", layout_key="BSND", **kwargs):
    """Generate pytest inputs/golden, then populate the explicit metadata tensor."""
    data = INPUT_ADAPTER.customize(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        output_idx_offset,
        layout_query,
        layout_key,
        kwargs,
    )
    INPUT_ADAPTER.generate_metadata(
        query,
        key,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        metadata,
        layout_query,
        layout_key,
        data,
        kwargs,
    )
