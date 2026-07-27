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

"""Input customization for MixedQuantSparseFlashMla TTK cases."""

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import torch


class NumpyBatchRandomContext:
    """Make declared batch slices deterministic before quantized KV packing."""

    SEED_MODULUS = (1 << 63) - 1

    def __init__(self, q, kwargs):
        self.batch_size = int(q.shape[0])
        self.relations = self._parse_relations(q, kwargs)
        self.base_seed = self.relations[0][1]
        self.call_index = 0
        self.original_numpy_uniform = None
        self.original_python_uniform = None

    @classmethod
    def from_case(cls, q, kwargs):
        fields = tuple(kwargs.get(name) for name in (
            "batch_axis", "batch_slice_info", "batch_seed"
        ))
        if all(field is None for field in fields):
            return None
        if any(field is None for field in fields):
            raise ValueError(
                "batch_axis, batch_slice_info and batch_seed must be set together"
            )
        if kwargs.get("layout_q", "BSND") != "BSND" or kwargs.get(
                "layout_kv", "BSND") != "BSND":
            raise ValueError("MQSMLA batch consistency currently supports BSND/BSND only")
        return cls(q, kwargs)

    @staticmethod
    def _parse_relations(q, kwargs):
        batch_axis = kwargs["batch_axis"]
        batch_slices = kwargs["batch_slice_info"]
        batch_seed = kwargs["batch_seed"]
        if not batch_axis or batch_axis[0] != (0,):
            raise ValueError("MQSMLA batch consistency requires q batch_axis=((0,), ...)")
        if not batch_slices or batch_slices[0] is None or batch_seed[0] is None:
            raise ValueError("MQSMLA batch consistency requires slices and seeds on q")
        if any(slices is not None for slices in batch_slices[1:]):
            raise ValueError("MQSMLA batch consistency relations must be declared on q only")

        relations = list(zip(batch_slices[0][0], batch_seed[0][0]))
        if not relations:
            raise ValueError("MQSMLA batch consistency requires at least one q slice")
        if len({seed for _, seed in relations}) != 1:
            raise ValueError("MQSMLA batch consistency currently supports one relation seed per case")
        for slice_value, _ in relations:
            start, stop, step = slice_value
            if step != 1 or start < 0 or start >= stop or stop > int(q.shape[0]):
                raise ValueError(
                    "MQSMLA batch consistency requires in-range, contiguous q batch slices"
                )
        return relations

    def validate_params(self, params):
        if params.get("template_run_mode") not in ("SWA", "HCA"):
            raise ValueError("MQSMLA batch consistency currently excludes sparse modes")
        if int(params.get("quant_mode")) != 1:
            raise ValueError("MQSMLA batch consistency currently supports quant_mode=1 only")

    @classmethod
    def _derive_seed(cls, seed, call_index):
        return (int(seed) + (call_index + 1) * 1000003) % cls.SEED_MODULUS

    def _numpy_uniform(self, low=0.0, high=1.0, size=None):
        call_index = self.call_index
        self.call_index += 1
        rng = np.random.default_rng(self._derive_seed(self.base_seed, call_index))
        value = rng.uniform(low, high, size)
        if size is None:
            return value
        value = np.asarray(value)
        if value.ndim < 2 or int(value.shape[0]) != self.batch_size:
            return value

        for slice_value, seed in self.relations:
            selector = (slice(*slice_value),) + (slice(None),) * (value.ndim - 1)
            piece = value[selector]
            piece_rng = np.random.default_rng(self._derive_seed(seed, call_index))
            value[selector] = piece_rng.uniform(low, high, piece.shape)
        return value

    def _python_uniform(self, low, high):
        call_index = self.call_index
        self.call_index += 1
        rng = random.Random(self._derive_seed(self.base_seed, call_index))
        return rng.uniform(low, high)

    def __enter__(self):
        self.original_numpy_uniform = np.random.uniform
        self.original_python_uniform = random.uniform
        np.random.uniform = self._numpy_uniform
        random.uniform = self._python_uniform
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        np.random.uniform = self.original_numpy_uniform
        random.uniform = self.original_python_uniform
        return False


class MixedQuantSparseFlashMlaInputAdapter:
    """Translate a TTK case to pytest parameters and reuse pytest generation."""

    def __init__(self):
        self.pytest_modules = {}

    @staticmethod
    def load_golden_store():
        name = "mqsmla_batch_ttk_golden"
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
        name = f"mqsmla_pytest_{stem}"
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

    def build_case_params(self, q, ori_kv, cmp_kv, cmp_sparse_indices,
                          layout_q, layout_kv, kwargs):
        cu_q = self.list_value(kwargs, "cu_seqlens_q")
        cu_ori = self.list_value(kwargs, "cu_seqlens_ori_kv")
        cu_cmp = self.list_value(kwargs, "cu_seqlens_cmp_kv")
        seq_q = self.list_value(kwargs, "seqused_q")
        seq_ori = self.list_value(kwargs, "seqused_ori_kv")
        seq_cmp = self.list_value(kwargs, "seqused_cmp_kv")
        residual = self.list_value(kwargs, "cmp_residual_kv")

        if layout_q == "BSND":
            batch_size, q_seq, q_heads, head_dim = [int(x) for x in q.shape]
        else:
            _, q_heads, head_dim = [int(x) for x in q.shape]
            batch_size = len(seq_q or self.prefix_lengths(cu_q))
            q_seq = max(self.prefix_lengths(cu_q) or seq_q or [int(q.shape[0])])

        if layout_kv == "BSND":
            _, kv_seq, kv_heads, _ = [int(x) for x in ori_kv.shape]
            block_num1, block_size1 = None, 64
        elif layout_kv == "TND":
            _, kv_heads, _ = [int(x) for x in ori_kv.shape]
            kv_seq = max(self.prefix_lengths(cu_ori) or seq_ori or [int(ori_kv.shape[0])])
            block_num1, block_size1 = None, 64
        else:
            block_num1, block_size1, kv_heads, _ = [int(x) for x in ori_kv.shape]
            kv_seq = max(self.prefix_lengths(cu_ori) or seq_ori or [block_size1])

        metadata_cmp_topk = int(kwargs.get("metadata_cmp_topk") or 0)
        if metadata_cmp_topk > 0:
            mode = "CSA"
            if layout_kv == "PA_BBND" and cmp_kv is not None:
                block_num2, block_size2 = int(cmp_kv.shape[0]), int(cmp_kv.shape[1])
            else:
                block_num2, block_size2 = None, block_size1
        elif cmp_kv is None:
            mode = "SWA"
            block_num2, block_size2 = 0, block_size1
        elif layout_kv == "PA_BBND":
            mode = "CSA" if cmp_sparse_indices is not None else "HCA"
            block_num2, block_size2 = int(cmp_kv.shape[0]), int(cmp_kv.shape[1])
        else:
            mode = "CSA" if cmp_sparse_indices is not None else "HCA"
            block_num2, block_size2 = None, block_size1

        params = dict(kwargs)
        params.update({
            "Testcase_Name": kwargs.get("testcase_name"),
            "layout_q": layout_q,
            "layout_kv": layout_kv,
            "q_type": q.dtype,
            "ori_kv_type": ori_kv.dtype,
            "cmp_kv_type": cmp_kv.dtype if cmp_kv is not None else ori_kv.dtype,
            "B": batch_size,
            "S1": q_seq,
            "S2": kv_seq,
            "N1": q_heads,
            "N2": kv_heads,
            "D": head_dim,
            "K": metadata_cmp_topk or (
                int(cmp_sparse_indices.shape[-1]) if cmp_sparse_indices is not None else 0
            ),
            "block_num1": block_num1,
            "block_num2": block_num2,
            "block_size1": block_size1,
            "block_size2": block_size2,
            "seqused_q": seq_q,
            "cu_seqlens_q": cu_q,
            "seqused_ori_kv": seq_ori,
            "seqused_cmp_kv": seq_cmp,
            "cu_seqlens_ori_kv": cu_ori,
            "cu_seqlens_cmp_kv": cu_cmp,
            "cmp_residual_kv": residual,
            "template_run_mode": mode,
            "cmp_ratio": 1 if mode == "SWA" else int(kwargs.get("cmp_ratio", 1)),
            "cmp_mask_mode": 0 if mode == "SWA" else int(kwargs.get("cmp_mask_mode", 3)),
            "tile_size": 64,
            "actlen_mode": kwargs.get("actlen_mode", "full"),
            "S1EQS2": bool(kwargs.get("S1EQS2", False)),
            "topk_value_mode": int(kwargs.get("topk_value_mode", 1)),
            "return_softmax_lse": bool(kwargs.get("return_softmax_lse", False)),
        })
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

    def generate_case(self, params, batch_random=None):
        pytest_utils = self.load_pytest_module("utils", "utils.py")
        pytest_check = self.load_pytest_module("check", "check_valid_param.py")
        pytest_golden = self.load_pytest_module(
            "golden", "mixed_quant_sparse_flash_mla_golden.py"
        )
        filled = pytest_utils.fill_none_params(params)
        if filled.get("Testcase_Name") is None:
            filled["Testcase_Name"] = params.get("testcase_name")
        pytest_check.check_valid_param(filled)
        if batch_random is None:
            data = pytest_golden.generate_and_save_testdata(filled, save_pt=False)
        else:
            batch_random.validate_params(filled)
            with batch_random:
                data = pytest_golden.generate_and_save_testdata(filled, save_pt=False)
        testcase_name = params.get("testcase_name") or filled.get("Testcase_Name")
        self.load_golden_store().CASE_DATA.put(testcase_name, data)
        return data


INPUT_ADAPTER = MixedQuantSparseFlashMlaInputAdapter()


def generate_mixed_quant_sparse_flash_mla_inputs(q, *, ori_kv=None, cmp_kv=None,
                                                 ori_sparse_indices=None, cmp_sparse_indices=None,
                                                 ori_block_table=None, cmp_block_table=None,
                                                 cu_seqlens_q=None, cu_seqlens_ori_kv=None,
                                                 cu_seqlens_cmp_kv=None, seqused_q=None,
                                                 seqused_ori_kv=None, seqused_cmp_kv=None,
                                                 cmp_residual_kv=None, ori_topk_length=None,
                                                 cmp_topk_length=None, sinks=None, **kwargs):
    """Reuse the pytest parameter validation and input processing for a TTK case."""
    batch_random = NumpyBatchRandomContext.from_case(q, kwargs)
    params = INPUT_ADAPTER.build_case_params(
        q,
        ori_kv,
        cmp_kv,
        cmp_sparse_indices,
        kwargs.get("layout_q", "BSND"),
        kwargs.get("layout_kv", "BSND"),
        kwargs,
    )
    data = INPUT_ADAPTER.generate_case(params, batch_random)
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
