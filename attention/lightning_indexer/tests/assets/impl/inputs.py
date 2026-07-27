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

__input__ = {
    "e2e": {
        "torch_npu.npu_lightning_indexer": "generate_li_inputs"
    }
}

import importlib.util
import sys
from pathlib import Path

import torch


class LightningIndexerInputAdapter:
    """Translate a TTK case and reuse the canonical pytest case generator."""

    def __init__(self):
        self.pytest_golden = None

    @staticmethod
    def load_golden_store():
        name = "li_ttk_golden"
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

    def load_pytest_golden(self):
        if self.pytest_golden is not None:
            return self.pytest_golden
        pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
        path = pytest_dir / "lightning_indexer_golden.py"
        name = "li_pytest_golden"
        inserted = str(pytest_dir) not in sys.path
        if inserted:
            sys.path.insert(0, str(pytest_dir))
        try:
            if name in sys.modules:
                module = sys.modules[name]
            else:
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                try:
                    spec.loader.exec_module(module)
                except Exception:
                    sys.modules.pop(name, None)
                    raise
            self.pytest_golden = module
            return module
        finally:
            if inserted:
                sys.path.remove(str(pytest_dir))

    @staticmethod
    def list_value(value):
        if value is None:
            return []
        if torch.is_tensor(value):
            value = value.detach().cpu().reshape(-1).tolist()
        elif not isinstance(value, (list, tuple)):
            value = [value]
        return [int(item) for item in value]

    @staticmethod
    def prefix_lengths(value):
        if not value:
            return []
        return [int(value[0])] + [
            int(value[index]) - int(value[index - 1])
            for index in range(1, len(value))
        ]

    @staticmethod
    def data_range(input_ranges, index, default=(-1, 1)):
        if input_ranges and index < len(input_ranges) and input_ranges[index] is not None:
            return list(input_ranges[index])
        return list(default)

    def build_case_params(self, query, key, weights,
                          actual_seq_lengths_query,
                          actual_seq_lengths_key,
                          layout_query, layout_key, kwargs):
        act_q = self.list_value(
            kwargs.get("actual_seq_lengths_query_values", actual_seq_lengths_query)
        )
        act_k = self.list_value(
            kwargs.get("actual_seq_lengths_key_values", actual_seq_lengths_key)
        )

        if layout_query == "BSND":
            batch_size, q_seq, q_head_num, head_dim = map(int, query.shape)
            q_t_size = sum(act_q) if act_q else batch_size * q_seq
            act_q = act_q or [q_seq] * batch_size
        elif layout_query == "TND":
            q_t_size, q_head_num, head_dim = map(int, query.shape)
            batch_size = len(act_q)
            if not batch_size:
                raise ValueError("TND LightningIndexer requires actual_seq_lengths_query_values")
            q_seq = max(self.prefix_lengths(act_q), default=q_t_size)
        else:
            raise ValueError(f"unsupported LightningIndexer query layout: {layout_query}")

        if layout_key == "BSND":
            _, k_seq, k_head_num, _ = map(int, key.shape)
            k_t_size = sum(act_k) if act_k else batch_size * k_seq
            act_k = act_k or [k_seq] * batch_size
            block_size = int(kwargs.get("block_size", 0))
            block_num = int(kwargs.get("block_num", 0))
        elif layout_key == "TND":
            k_t_size, k_head_num, _ = map(int, key.shape)
            if not act_k:
                raise ValueError("TND LightningIndexer requires actual_seq_lengths_key_values")
            k_seq = max(self.prefix_lengths(act_k), default=k_t_size)
            block_size = int(kwargs.get("block_size", 0))
            block_num = int(kwargs.get("block_num", 0))
        elif layout_key == "PA_BSND":
            block_num, block_size, k_head_num, _ = map(int, key.shape)
            if not act_k:
                raise ValueError("PA_BSND LightningIndexer requires actual_seq_lengths_key_values")
            k_seq = max(act_k)
            k_t_size = sum(act_k)
        else:
            raise ValueError(f"unsupported LightningIndexer key layout: {layout_key}")

        input_ranges = kwargs.get("input_ranges") or ()
        actual_seq_dtype = (
            actual_seq_lengths_query.dtype
            if torch.is_tensor(actual_seq_lengths_query) else torch.int32
        )
        return (
            batch_size, q_seq, k_seq, q_t_size, k_t_size,
            q_head_num, k_head_num, head_dim, block_size, block_num,
            query.dtype, weights.dtype, actual_seq_dtype, act_q, act_k,
            layout_query, layout_key, int(kwargs.get("sparse_count", 2048)),
            int(kwargs.get("sparse_mode", 3)),
            self.data_range(input_ranges, 0),
            self.data_range(input_ranges, 1),
            self.data_range(input_ranges, 2),
            bool(kwargs.get("return_value", False)),
        )

    @staticmethod
    def copy_tensor(dst, src, name):
        if dst is None:
            return
        if src is None:
            raise ValueError(f"{name} is present in CSV but pytest generator returned None")
        src_cpu = src.detach().cpu() if torch.is_tensor(src) else torch.as_tensor(src)
        if tuple(dst.shape) != tuple(src_cpu.shape):
            raise ValueError(
                f"{name} shape mismatch: TTK={tuple(dst.shape)} "
                f"pytest={tuple(src_cpu.shape)}"
            )
        dst.copy_(src_cpu.to(dtype=dst.dtype, device=dst.device))

    @staticmethod
    def golden_data(data):
        return {
            "cpu_result": data["cpu_result"].detach().cpu(),
            "cpu_result_raw": data["cpu_result_raw"].detach().cpu(),
            "score_values": data["score_values"].detach().cpu(),
            "value_dtype": data["query"].dtype,
        }

    def customize(self, query, key, weights, actual_seq_lengths_query,
                  actual_seq_lengths_key, block_table,
                  layout_query, layout_key, kwargs):
        params = self.build_case_params(
            query, key, weights, actual_seq_lengths_query,
            actual_seq_lengths_key, layout_query, layout_key, kwargs
        )
        golden_store = self.load_golden_store().CASE_DATA
        golden_store.clear()
        data = self.load_pytest_golden().generate_li_test_data(params)
        for name, dst in (
            ("query", query),
            ("key", key),
            ("weights", weights),
            ("actual_seq_lengths_query", actual_seq_lengths_query),
            ("actual_seq_lengths_key", actual_seq_lengths_key),
            ("block_table", block_table),
        ):
            self.copy_tensor(dst, data.get(name), name)
        golden_store.put(kwargs.get("testcase_name"), self.golden_data(data))


INPUT_ADAPTER = LightningIndexerInputAdapter()


def generate_li_inputs(query, key, weights, *,
                       actual_seq_lengths_query=None,
                       actual_seq_lengths_key=None,
                       block_table=None,
                       layout_query="BSND", layout_key="BSND", **kwargs):
    """Generate the exact canonical pytest inputs and CPU golden for a TTK case."""
    INPUT_ADAPTER.customize(
        query, key, weights, actual_seq_lengths_query,
        actual_seq_lengths_key, block_table, layout_query, layout_key, kwargs
    )
