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

"""Raw Kernel input customization for SparseFlashMla."""

import importlib.util
import sys
from pathlib import Path

import torch

import cann_ops_transformer  # noqa: F401 - registers torch.ops namespace


class SparseFlashMlaInputAdapter:
    """Reuse pytest generation and populate the explicit metadata input."""

    def __init__(self):
        self.pytest_modules = {}

    @staticmethod
    def load_golden_store():
        name = "smla_ttk_v3_golden"
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

    def load_pytest_module(self, stem, filename):
        if stem in self.pytest_modules:
            return self.pytest_modules[stem]

        pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
        module_path = pytest_dir / filename
        name = f"smla_pytest_{stem}"
        inserted = str(pytest_dir) not in sys.path
        if inserted:
            sys.path.insert(0, str(pytest_dir))
        try:
            if name in sys.modules:
                module = sys.modules[name]
            else:
                spec = importlib.util.spec_from_file_location(name, module_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                try:
                    spec.loader.exec_module(module)
                except Exception:
                    sys.modules.pop(name, None)
                    raise
            self.pytest_modules[stem] = module
            return module
        finally:
            if inserted:
                sys.path.remove(str(pytest_dir))

    @staticmethod
    def list_value(kwargs, name):
        value = kwargs.get(f"{name}_values")
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach().cpu().reshape(-1).tolist()
        return [int(item) for item in value]

    @staticmethod
    def prefix_lengths(value):
        if not value:
            return []
        return [int(value[i + 1]) - int(value[i]) for i in range(len(value) - 1)]

    @staticmethod
    def data_range(input_ranges, index, default=(-10, 10)):
        if input_ranges and index < len(input_ranges) and input_ranges[index] is not None:
            return repr(list(input_ranges[index]))
        return repr(list(default))

    def build_case_params(self, q, ori_kv, cmp_kv, cmp_sparse_indices, layout_q, layout_kv, kwargs):
        cu_q = self.list_value(kwargs, "cu_seqlens_q")
        cu_ori = self.list_value(kwargs, "cu_seqlens_ori_kv")
        cu_cmp = self.list_value(kwargs, "cu_seqlens_cmp_kv")
        seq_q = self.list_value(kwargs, "seqused_q")
        seq_ori = self.list_value(kwargs, "seqused_ori_kv")
        seq_cmp = self.list_value(kwargs, "seqused_cmp_kv")
        residual = self.list_value(kwargs, "cmp_residual_kv")

        if layout_q == "BSND":
            batch_size, q_seq, q_heads, head_dim = [int(x) for x in q.shape]
            q_total = batch_size * q_seq
        else:
            q_total, q_heads, head_dim = [int(x) for x in q.shape]
            batch_size = len(seq_q or self.prefix_lengths(cu_q))
            q_seq = max(self.prefix_lengths(cu_q) or seq_q or [q_total])

        if layout_kv == "BSND":
            _, kv_seq, kv_heads, _ = [int(x) for x in ori_kv.shape]
            kv_total = batch_size * kv_seq
            block_num1, block_size1 = 0, 128
        elif layout_kv == "TND":
            kv_total, kv_heads, _ = [int(x) for x in ori_kv.shape]
            kv_seq = max(self.prefix_lengths(cu_ori) or seq_ori or [kv_total])
            block_num1, block_size1 = 0, 128
        else:
            block_num1, block_size1, kv_heads, _ = [int(x) for x in ori_kv.shape]
            kv_seq = max(seq_ori or [block_size1])
            kv_total = sum(seq_ori or [])

        if cmp_kv is None:
            mode = "SWA"
            cmp_total = 0
            block_num2, block_size2 = 0, block_size1
        elif layout_kv == "BSND":
            mode = "CSA" if cmp_sparse_indices is not None else "HCA"
            cmp_total = batch_size * int(cmp_kv.shape[1])
            block_num2, block_size2 = 0, 128
        elif layout_kv == "TND":
            mode = "CSA" if cmp_sparse_indices is not None else "HCA"
            cmp_total = int(cmp_kv.shape[0])
            block_num2, block_size2 = 0, 128
        else:
            mode = "CSA" if cmp_sparse_indices is not None else "HCA"
            block_num2, block_size2 = int(cmp_kv.shape[0]), int(cmp_kv.shape[1])
            cmp_total = sum(seq_cmp or [])

        input_ranges = kwargs.get("input_ranges") or ()
        params = dict(kwargs)
        params.update({
            "testcase_name": kwargs.get("testcase_name"),
            "layout_q": layout_q,
            "layout_kv": layout_kv,
            "q_type": q.dtype,
            "ori_kv_type": ori_kv.dtype,
            "cmp_kv_type": cmp_kv.dtype if cmp_kv is not None else ori_kv.dtype,
            "B": batch_size,
            "S1": q_seq,
            "S2": kv_seq,
            "T1": q_total,
            "T2": kv_total,
            "T3": cmp_total,
            "N1": q_heads,
            "N2": kv_heads,
            "D": head_dim,
            "K": int(cmp_sparse_indices.shape[-1]) if cmp_sparse_indices is not None else None,
            "block_num1": block_num1,
            "block_num2": block_num2,
            "block_size1": block_size1,
            "block_size2": block_size2,
            "cu_seqlens_q": cu_q,
            "cu_seqlens_ori_kv": cu_ori,
            "cu_seqlens_cmp_kv": cu_cmp,
            "seqused_q": seq_q,
            "seqused_ori_kv": seq_ori,
            "seqused_cmp_kv": seq_cmp,
            "cmp_residual_kv": residual,
            "template_mode": mode,
            "cmp_ratio": None if mode == "SWA" else int(kwargs.get("cmp_ratio", 1)),
            "cmp_mask_mode": 0 if mode == "SWA" else int(kwargs.get("cmp_mask_mode", 3)),
            "q_datarange": self.data_range(input_ranges, 0),
            "ori_kv_datarange": self.data_range(input_ranges, 1),
            "cmp_kv_datarange": self.data_range(input_ranges, 2),
            "qk_equal_len": False,
        })
        params.setdefault("return_softmax_lse", False)
        params.setdefault("ori_kv_topk_mode", "full")
        params.setdefault("cmp_kv_topk_mode", "full")
        params.setdefault("ori_sparse_indices_mode", "full")
        params.setdefault("cmp_sparse_indices_mode", "full")
        params.setdefault("actlen_mode", "full")
        return params

    @staticmethod
    def copy_tensor(dst, src, name):
        if dst is None:
            return
        if src is None:
            raise ValueError(f"{name} is present in CSV but pytest generator returned None")
        src_cpu = src.detach().cpu() if torch.is_tensor(src) else torch.as_tensor(src)
        if tuple(dst.shape) != tuple(src_cpu.shape):
            raise ValueError(
                f"{name} shape mismatch: TTK={tuple(dst.shape)} pytest={tuple(src_cpu.shape)}"
            )
        dst.copy_(src_cpu.to(dtype=dst.dtype, device=dst.device))

    def generate_case(self, params):
        pytest_utils = self.load_pytest_module("utils", "utils.py")
        pytest_golden = self.load_pytest_module("golden", "sparse_flash_mla_golden.py")
        case_params = pytest_utils.generate_case_with_default_param(params)
        data = pytest_golden.gen_data(case_params, case_params.get("template_mode"))
        testcase_name = params.get("testcase_name") or case_params.get("testcase_name")
        self.load_golden_store().CASE_DATA.put(testcase_name, data)
        return data

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
        generated = torch.ops.cann_ops_transformer.sparse_flash_mla_metadata(
            int(q.shape[1] if layout_q == "TND" else q.shape[2]),
            int(kv_ref.shape[1] if layout_kv == "TND" else kv_ref.shape[2]),
            int(q.shape[-1]),
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
                else self.max_seq(cu_seqlens_q, None, q_fallback)
            ),
            max_seqlen_ori_kv=self.max_seq(
                cu_seqlens_ori_kv, seqused_ori_kv, ori_fallback
            ),
            max_seqlen_cmp_kv=self.max_seq(
                cu_seqlens_cmp_kv, seqused_cmp_kv, cmp_fallback
            ),
            ori_topk=int(ori_sparse_indices.shape[-1]) if ori_sparse_indices is not None else 0,
            cmp_topk=int(cmp_sparse_indices.shape[-1]) if cmp_sparse_indices is not None else 0,
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


INPUT_ADAPTER = SparseFlashMlaInputAdapter()


def generate_sparse_flash_mla_inputs(q, *, ori_kv=None, cmp_kv=None, ori_sparse_indices=None,
                                     cmp_sparse_indices=None, ori_block_table=None, cmp_block_table=None,
                                     cu_seqlens_q=None, cu_seqlens_ori_kv=None,
                                     cu_seqlens_cmp_kv=None, seqused_q=None,
                                     seqused_ori_kv=None, seqused_cmp_kv=None,
                                     cmp_residual_kv=None, ori_topk_length=None,
                                     cmp_topk_length=None, sinks=None, metadata=None, **kwargs):
    """Reuse pytest input processing and generate the explicit metadata tensor."""
    layout_q = kwargs.get("layout_q", "BSND")
    layout_kv = kwargs.get("layout_kv", "BSND")
    params = INPUT_ADAPTER.build_case_params(
        q, ori_kv, cmp_kv, cmp_sparse_indices, layout_q, layout_kv, kwargs
    )
    data = INPUT_ADAPTER.generate_case(params)
    op_input = data["input"]
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


def generate_sparse_flash_mla_kernel_inputs(
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
    generate_sparse_flash_mla_inputs(
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
