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

"""Input customization for SparseAttnSharedkv TTK cases."""

import importlib.util
import sys
from pathlib import Path

import torch


class SparseAttnSharedkvInputAdapter:
    """Translate a TTK case to sas pytest golden parameters."""

    TEMPLATE_MODES = frozenset(("SWA", "CFA", "SCFA"))
    DEFAULT_DATA_RANGE = [-10, 10]

    @staticmethod
    def module_load_error(stage, path, exc):
        return RuntimeError(
            "Failed to load sparseAttnSharedkv module; "
            f"stage={stage}; module={path.resolve()}; "
            f"original error: {type(exc).__name__}: {exc}"
        )

    def __init__(self):
        self.pytest_modules = {}

    @staticmethod
    def load_golden_store():
        name = "sas_ttk_golden"
        if name in sys.modules:
            return sys.modules[name]
        path = Path(__file__).with_name("golden.py")
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot create import spec for {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(name, None)
            raise SparseAttnSharedkvInputAdapter.module_load_error(
                "assets Golden store", path, exc
            ) from exc
        return module

    def load_pytest_module(self, stem, filename):
        if stem in self.pytest_modules:
            return self.pytest_modules[stem]

        pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
        module_path = pytest_dir / filename
        name = f"sas_pytest_{stem}"
        inserted = str(pytest_dir) not in sys.path
        if inserted:
            sys.path.insert(0, str(pytest_dir))
        try:
            if name in sys.modules:
                module = sys.modules[name]
            else:
                spec = importlib.util.spec_from_file_location(name, module_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot create import spec for {module_path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
            self.pytest_modules[stem] = module
            return module
        except Exception as exc:
            sys.modules.pop(name, None)
            raise self.module_load_error(f"pytest {stem}", module_path, exc) from exc
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

    @classmethod
    def data_range(cls, input_ranges, index):
        if not input_ranges or index >= len(input_ranges):
            return list(cls.DEFAULT_DATA_RANGE)
        value = input_ranges[index]
        if value is None or any(item is None for item in value):
            return list(cls.DEFAULT_DATA_RANGE)
        return [int(item) for item in value]

    @classmethod
    def select_template_mode(cls, kwargs):
        mode = kwargs.get("template_run_mode") or kwargs.get("template_mode")
        if mode is None:
            return None
        mode = str(mode).strip()
        if mode not in cls.TEMPLATE_MODES:
            raise ValueError(f"unsupported explicit template_run_mode: {mode!r}")
        return mode

    @staticmethod
    def verify_template_inputs(mode, cmp_sparse_indices, cmp_kv):
        if mode == "SWA" and cmp_kv is not None:
            raise ValueError(f"{mode} must not provide cmp_kv in the CSV")
        if mode in ("CFA", "SCFA") and cmp_kv is None:
            raise ValueError(f"{mode} requires cmp_kv in the CSV")
        if mode == "SCFA" and cmp_sparse_indices is None:
            raise ValueError(f"{mode} requires cmp_sparse_indices in the CSV")
        if mode in ("SWA", "CFA") and cmp_sparse_indices is not None:
            raise ValueError(f"{mode} must not provide cmp_sparse_indices in the CSV")

    def build_case_params(self, q, ori_kv, cmp_kv, cmp_sparse_indices,
                          layout_q, layout_kv, kwargs):
        if layout_q not in ("BSND", "TND"):
            raise ValueError(f"unsupported layout_q: {layout_q!r}")
        if layout_kv not in ("PA_ND", "TND"):
            raise ValueError(
                f"unsupported layout_kv: {layout_kv!r}; "
                "sas pytest golden supports only PA_ND/TND"
            )
        cu_q = self.list_value(kwargs, "cu_seqlens_q")
        cu_ori = self.list_value(kwargs, "cu_seqlens_ori_kv")
        cu_cmp = self.list_value(kwargs, "cu_seqlens_cmp_kv")
        seq_q = self.list_value(kwargs, "seqused_q")
        seq_kv = self.list_value(kwargs, "seqused_kv")

        if layout_q == "BSND":
            batch_size, q_seq, q_heads, head_dim = [int(x) for x in q.shape]
            q_total = batch_size * q_seq
        else:
            q_total, q_heads, head_dim = [int(x) for x in q.shape]
            if cu_q is None:
                raise ValueError("TND layout_q requires cu_seqlens_q in the CSV")
            batch_size = len(seq_q or self.prefix_lengths(cu_q))
            q_seq = max(self.prefix_lengths(cu_q) or seq_q or [q_total])
        if kwargs.get("S1") is not None:
            q_seq = int(kwargs["S1"])

        if layout_kv == "TND":
            kv_total, kv_heads, _ = [int(x) for x in ori_kv.shape]
            if cu_ori is None:
                raise ValueError("TND layout_kv requires cu_seqlens_ori_kv in the CSV")
            kv_seq = max(self.prefix_lengths(cu_ori) or seq_kv or [kv_total])
            block_num1, block_size1 = None, None
        else:
            block_num1, block_size1, kv_heads, _ = [int(x) for x in ori_kv.shape]
            if seq_kv is None:
                raise ValueError("PA_ND layout_kv requires seqused_kv in the CSV")
            kv_seq = max(seq_kv)
            kv_total = sum(seq_kv)
        if kwargs.get("S2") is not None:
            kv_seq = int(kwargs["S2"])

        if cmp_kv is None:
            block_num2, block_size2 = None, None
        elif layout_kv == "TND":
            block_num2, block_size2 = None, None
        else:
            block_num2, block_size2 = int(cmp_kv.shape[0]), int(cmp_kv.shape[1])

        mode = self.select_template_mode(kwargs)
        if mode is None:
            if cmp_sparse_indices is not None:
                mode = "SCFA"
            elif cmp_kv is not None:
                mode = "CFA"
            else:
                mode = "SWA"
        self.verify_template_inputs(mode, cmp_sparse_indices, cmp_kv)
        if mode in ("CFA", "SCFA") and kwargs.get("cmp_ratio") is None:
            raise ValueError(f"{mode} requires cmp_ratio in the CSV")

        input_ranges = kwargs.get("input_ranges") or ()
        K = int(cmp_sparse_indices.shape[-1]) if cmp_sparse_indices is not None else None
        return (
            layout_q,
            layout_kv,
            q.dtype,
            ori_kv.dtype,
            cmp_kv.dtype if cmp_kv is not None else None,
            batch_size,
            q_seq,
            q_total,
            q_heads,
            kv_heads,
            head_dim,
            K,
            block_num1,
            block_num2,
            block_size1,
            block_size2,
            cu_q,
            seq_kv,
            kwargs.get("softmax_scale", 1.0),
            kwargs.get("cmp_ratio"),
            kwargs.get("ori_mask_mode", 4),
            kwargs.get("cmp_mask_mode", 3),
            kwargs.get("ori_win_left", 127),
            kwargs.get("ori_win_right", 0),
            kwargs.get("testcase_name"),
            kv_seq,
            self.data_range(input_ranges, 0),
            self.data_range(input_ranges, 1),
            self.data_range(input_ranges, 2),
            seq_q,
            cu_ori,
            cu_cmp,
            kv_total,
        )

    @staticmethod
    def copy_tensor(dst, src, name):
        if dst is None:
            if src is not None:
                raise ValueError(
                    f"{name} is absent from CSV but pytest generator produced a tensor"
                )
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
        pytest_golden = self.load_pytest_module("golden", "sparse_attn_sharedkv_golden.py")
        data = pytest_golden.gen_data(params)
        self.load_golden_store().CASE_DATA.put(params[24], data)
        return data

    def customize(self, q, ori_kv, cmp_kv, cmp_sparse_indices,
                  layout_q, layout_kv, kwargs):
        params = self.build_case_params(
            q, ori_kv, cmp_kv, cmp_sparse_indices, layout_q, layout_kv, kwargs
        )
        return self.generate_case(params)


INPUT_ADAPTER = SparseAttnSharedkvInputAdapter()


def generate_sparse_attn_sharedkv_inputs(q, *, ori_kv=None, cmp_kv=None,
                                         cmp_sparse_indices=None, ori_block_table=None,
                                         cmp_block_table=None, cu_seqlens_q=None,
                                         cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None,
                                         seqused_q=None, seqused_kv=None, sinks=None,
                                         **kwargs):
    """Reuse the pytest input processing for a TTK case."""
    data = INPUT_ADAPTER.customize(
        q,
        ori_kv,
        cmp_kv,
        cmp_sparse_indices,
        kwargs.get("layout_q"),
        kwargs.get("layout_kv"),
        kwargs,
    )
    op_input = data["input"]
    for name, tensor in (
        ("q", q),
        ("ori_kv", ori_kv),
        ("cmp_kv", cmp_kv),
        ("cmp_sparse_indices", cmp_sparse_indices),
        ("ori_block_table", ori_block_table),
        ("cmp_block_table", cmp_block_table),
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_ori_kv", cu_seqlens_ori_kv),
        ("cu_seqlens_cmp_kv", cu_seqlens_cmp_kv),
        ("seqused_q", seqused_q),
        ("seqused_kv", seqused_kv),
        ("sinks", sinks),
    ):
        source = op_input.get(name)
        if name == "seqused_q" and tensor is None and source is not None:
            op_input[name] = None
            continue
        INPUT_ADAPTER.copy_tensor(tensor, source, name)
