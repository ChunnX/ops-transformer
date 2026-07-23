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

"""Raw Kernel input customization for MixedQuantSparseFlashMla."""

import importlib.util
import sys
from pathlib import Path

import torch

import cann_ops_transformer  # noqa: F401 - registers torch.ops namespace


class PrimaryInputAdapterLoader:
    """Load the shared input behavior from the primary assets implementation."""

    @staticmethod
    def load():
        name = "mqsmla_ttk_primary_inputs"
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


PrimaryInputAdapter = PrimaryInputAdapterLoader.load().MixedQuantSparseFlashMlaInputAdapter


class MixedQuantSparseFlashMlaMetadataInputAdapter(PrimaryInputAdapter):
    """Extend pytest input generation with explicit metadata generation."""

    @staticmethod
    def load_golden_store():
        name = "mqsmla_ttk_v3_golden"
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
    def max_seq(cu_seqlens, seqused, fallback=0):
        if seqused is not None and seqused.numel() > 0:
            return int(seqused.max().item())
        if cu_seqlens is not None and cu_seqlens.numel() > 1:
            return int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
        return int(fallback or 0)

    @staticmethod
    def infer_batch(cu_seqlens, seqused):
        if seqused is not None:
            return int(seqused.numel())
        if cu_seqlens is not None:
            return max(int(cu_seqlens.numel()) - 1, 0)
        return 0

    @staticmethod
    def prefix_max(cu_seqlens, fallback=0):
        if cu_seqlens is not None and cu_seqlens.numel() > 0:
            return int(cu_seqlens.max().item())
        return int(fallback or 0)

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

    @staticmethod
    def metadata_tensor(tensor):
        if tensor is None or tensor.device.type == "npu":
            return tensor
        return tensor.npu()

    def generate_metadata(self, q, ori_kv, cmp_kv, ori_sparse_indices, cmp_sparse_indices,
                          cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv,
                          seqused_q, seqused_ori_kv, seqused_cmp_kv,
                          cmp_residual_kv, ori_topk_length, cmp_topk_length,
                          metadata, kwargs):
        layout_q = kwargs.get("layout_q", "BSND")
        layout_kv = kwargs.get("layout_kv", "BSND")
        kv_ref = ori_kv if ori_kv is not None else cmp_kv
        q_fallback = int(q.shape[1] if layout_q == "BSND" else q.shape[0])
        ori_fallback = 0 if ori_kv is None else int(
            ori_kv.shape[0] if layout_kv == "TND" else ori_kv.shape[1]
        )
        cmp_fallback = 0 if cmp_kv is None else int(
            cmp_kv.shape[0] if layout_kv == "TND" else cmp_kv.shape[1]
        )
        has_ori_kv = kwargs.get("has_ori_kv")
        has_cmp_kv = kwargs.get("has_cmp_kv")
        if has_ori_kv is None:
            has_ori_kv = ori_kv is not None
        if has_cmp_kv is None:
            has_cmp_kv = cmp_kv is not None
        generated = torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla_metadata(
            int(q.shape[1] if layout_q == "TND" else q.shape[2]),
            int(kv_ref.shape[1] if layout_kv == "TND" else kv_ref.shape[2]),
            int(q.shape[-1]),
            int(kwargs.get("quant_mode", 1)),
            cu_seqlens_q=self.metadata_tensor(cu_seqlens_q),
            cu_seqlens_ori_kv=self.metadata_tensor(cu_seqlens_ori_kv),
            cu_seqlens_cmp_kv=self.metadata_tensor(cu_seqlens_cmp_kv),
            seqused_q=self.metadata_tensor(seqused_q),
            seqused_ori_kv=self.metadata_tensor(seqused_ori_kv),
            seqused_cmp_kv=self.metadata_tensor(seqused_cmp_kv),
            cmp_residual_kv=self.metadata_tensor(cmp_residual_kv),
            ori_topk_length=self.metadata_tensor(ori_topk_length),
            cmp_topk_length=self.metadata_tensor(cmp_topk_length),
            batch_size=(
                int(q.shape[0])
                if layout_q == "BSND"
                else self.infer_batch(cu_seqlens_q, seqused_q)
            ),
            max_seqlen_q=(
                q_fallback
                if layout_q == "BSND"
                else self.prefix_max(cu_seqlens_q, q_fallback)
            ),
            max_seqlen_ori_kv=self.max_seq(
                cu_seqlens_ori_kv, seqused_ori_kv, ori_fallback
            ),
            max_seqlen_cmp_kv=self.max_seq(
                cu_seqlens_cmp_kv, seqused_cmp_kv, cmp_fallback
            ),
            ori_topk=int(ori_sparse_indices.shape[-1]) if ori_sparse_indices is not None else 0,
            cmp_topk=int(kwargs.get("metadata_cmp_topk") or (
                cmp_sparse_indices.shape[-1] if cmp_sparse_indices is not None else 0
            )),
            rope_head_dim=int(kwargs.get("rope_head_dim", 64)),
            cmp_ratio=int(kwargs.get("cmp_ratio", 1)),
            ori_mask_mode=int(kwargs.get("ori_mask_mode", 4)),
            cmp_mask_mode=int(kwargs.get("cmp_mask_mode", 3)),
            ori_win_left=int(kwargs.get("ori_win_left", 127)),
            ori_win_right=int(kwargs.get("ori_win_right", 0)),
            layout_q=layout_q,
            layout_kv=layout_kv,
            has_ori_kv=bool(has_ori_kv),
            has_cmp_kv=bool(has_cmp_kv),
        )
        self.fill_metadata(metadata, generated)
        torch.npu.synchronize()


INPUT_ADAPTER = MixedQuantSparseFlashMlaMetadataInputAdapter()


def generate_mixed_quant_sparse_flash_mla_inputs(q, *, ori_kv=None, cmp_kv=None,
                                                 ori_sparse_indices=None, cmp_sparse_indices=None,
                                                 ori_block_table=None, cmp_block_table=None,
                                                 cu_seqlens_q=None, cu_seqlens_ori_kv=None,
                                                 cu_seqlens_cmp_kv=None, seqused_q=None,
                                                 seqused_ori_kv=None, seqused_cmp_kv=None,
                                                 cmp_residual_kv=None, ori_topk_length=None,
                                                 cmp_topk_length=None, sinks=None, metadata=None,
                                                 **kwargs):
    """Reuse pytest input processing and generate the explicit metadata tensor."""
    params = INPUT_ADAPTER.build_case_params(
        q,
        ori_kv,
        cmp_kv,
        cmp_sparse_indices,
        kwargs.get("layout_q", "BSND"),
        kwargs.get("layout_kv", "BSND"),
        kwargs,
    )
    data = INPUT_ADAPTER.generate_case(params)
    op_input = data["op_input"]
    for name, tensor in (
        ("q", q),
        ("ori_kv", ori_kv),
        ("cmp_kv", cmp_kv),
        ("ori_sparse_indices", ori_sparse_indices),
        ("cmp_sparse_indices", cmp_sparse_indices),
        ("ori_block_table", ori_block_table),
        ("cmp_block_table", cmp_block_table),
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_ori_kv", cu_seqlens_ori_kv),
        ("cu_seqlens_cmp_kv", cu_seqlens_cmp_kv),
        ("seqused_q", seqused_q),
        ("seqused_ori_kv", seqused_ori_kv),
        ("seqused_cmp_kv", seqused_cmp_kv),
        ("cmp_residual_kv", cmp_residual_kv),
        ("ori_topk_length", ori_topk_length),
        ("cmp_topk_length", cmp_topk_length),
        ("sinks", sinks),
    ):
        INPUT_ADAPTER.copy_tensor(tensor, op_input.get(name), name)
    INPUT_ADAPTER.generate_metadata(
        q,
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        ori_topk_length,
        cmp_topk_length,
        metadata,
        kwargs,
    )


def generate_mixed_quant_sparse_flash_mla_kernel_inputs(
        q, ori_kv, cmp_kv, ori_sparse_indices, cmp_sparse_indices,
        ori_block_table, cmp_block_table, cu_seqlens_q,
        cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q,
        seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv,
        ori_topk_length, cmp_topk_length, sinks, metadata, **kwargs):
    """Adapt Kernel-mode NumPy buffers to the existing Torch input generator."""
    from ttk.utilities import numpy_to_torch_tensor

    arrays = (
        q, ori_kv, cmp_kv, ori_sparse_indices, cmp_sparse_indices,
        ori_block_table, cmp_block_table, cu_seqlens_q,
        cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q,
        seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv,
        ori_topk_length, cmp_topk_length, sinks, metadata,
    )
    tensors = tuple(numpy_to_torch_tensor(array) for array in arrays)
    generate_mixed_quant_sparse_flash_mla_inputs(
        tensors[0],
        ori_kv=tensors[1],
        cmp_kv=tensors[2],
        ori_sparse_indices=tensors[3],
        cmp_sparse_indices=tensors[4],
        ori_block_table=tensors[5],
        cmp_block_table=tensors[6],
        cu_seqlens_q=tensors[7],
        cu_seqlens_ori_kv=tensors[8],
        cu_seqlens_cmp_kv=tensors[9],
        seqused_q=tensors[10],
        seqused_ori_kv=tensors[11],
        seqused_cmp_kv=tensors[12],
        cmp_residual_kv=tensors[13],
        ori_topk_length=tensors[14],
        cmp_topk_length=tensors[15],
        sinks=tensors[16],
        metadata=tensors[17],
        **kwargs,
    )
    return arrays
