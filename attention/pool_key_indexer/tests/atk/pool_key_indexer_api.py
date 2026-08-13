#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PoolKeyIndexer ATK 自定义执行方式 (BaseApi)。

注册名: "pool_key_indexer" (对应 YAML api_type)

双后端双标杆路由:
  - name == "golden" → CPU golden 标杆
  - name != "golden" → NPU operator (torch_npu)
"""

import math
import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from pool_key_indexer_golden import (
    PKIGoldenRef,
    _safe_to_cpu,
    _safe_to_npu,
    _safe_to_dtype,
    _get_compute_dtype,
    _normalize_prefix_seq,
    _parse_dtype,
    _construct_block_table,
)

LAYOUT_MAP = {0: "BSND", 1: "TND", 2: "PA_BBND"}


def _parse_layout(raw):
    if isinstance(raw, (int, float)):
        return LAYOUT_MAP.get(int(raw), "BSND")
    return str(raw)


def _parse_bool(raw, default=False):
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(int(raw))
    if isinstance(raw, str):
        return raw.strip().lower() in ("true", "1")
    return default


@register("pool_key_indexer")
class PKIApi(BaseApi):
    def __init__(self, task_result):
        super().__init__(task_result)
        self.run_modes = []

    @staticmethod
    def _compute_output_shape(B, S1, topk, pool_size, layout_query, return_value, query):
        output_indices_len = topk + pool_size - 1
        output_values_len = topk // pool_size
        if layout_query == "BSND":
            idx_shape = [B, S1, output_indices_len]
            val_shape = [B, S1, output_values_len]
        else:
            idx_shape = [S1, output_indices_len]
            val_shape = [S1, output_values_len]
        indices = torch.empty(idx_shape, dtype=torch.int32)
        values = torch.empty(val_shape, dtype=torch.float32)
        return (indices, values)

    @staticmethod
    def _sanitize_values(values, compute_dtype):
        if values is None or (hasattr(values, "numel") and values.numel() == 0):
            return values if values is not None else torch.empty(0, dtype=compute_dtype)
        values = _safe_to_dtype(values, compute_dtype)
        values = torch.where(torch.isinf(values) | torch.isnan(values),
                             torch.zeros_like(values), values)
        return values

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if not with_output:
            return None

        args = list(input_data.args) if input_data.args else []
        kw = dict(input_data.kwargs) if input_data.kwargs else {}

        query = args[0] if len(args) > 0 else kw.get("query")
        key = args[1] if len(args) > 1 else kw.get("poolKey")
        weights = args[2] if len(args) > 2 else kw.get("weights")
        pool_tail_k = args[3] if len(args) > 3 else kw.get("poolTailK")
        act_seq_q_raw = args[4] if len(args) > 4 else kw.get("actualSeqQ")
        act_seq_k_raw = args[5] if len(args) > 5 else kw.get("actualSeqK")
        block_table_raw = args[6] if len(args) > 6 else kw.get("blockTable")

        layout_query = _parse_layout(kw.get("layout_q", "BSND"))
        layout_key = _parse_layout(kw.get("layout_k", "BSND"))
        topk = int(kw.get("topk", 2048))
        pool_size = int(kw.get("pool_size", 1))
        mask_mode = int(kw.get("mask_mode", 3))
        quant_mode = int(kw.get("quant_mode", -1))
        return_value = _parse_bool(kw.get("return_value", False))

        act_seq_q = act_seq_q_raw
        act_seq_k = act_seq_k_raw
        block_table = block_table_raw

        if act_seq_q is not None and hasattr(act_seq_q, "shape") and act_seq_q.shape[0] == 0:
            act_seq_q = None
        if act_seq_k is not None and hasattr(act_seq_k, "shape") and act_seq_k.shape[0] == 0:
            act_seq_k = None
        if block_table is not None and hasattr(block_table, "shape") and block_table.shape[0] == 0:
            block_table = None

        if layout_query == "TND" and act_seq_q is not None:
            act_seq_q = _normalize_prefix_seq(act_seq_q, query.shape[0])
        if layout_key == "TND" and act_seq_k is not None:
            act_seq_k = _normalize_prefix_seq(act_seq_k, key.shape[0])

        if layout_query == "BSND":
            B = query.shape[0]; S1 = query.shape[1]; N1 = query.shape[2]; D = query.shape[3]
        else:
            S1 = query.shape[0]; N1 = query.shape[1]; D = query.shape[2]
            B = len(act_seq_q) if act_seq_q is not None else 1

        if layout_key == "BSND":
            S2 = key.shape[1]; N2 = key.shape[2]
        elif layout_key == "TND":
            S2 = key.shape[0]; N2 = key.shape[1]
        else:
            N2 = key.shape[2]
            S2 = int(act_seq_k.max().item()) if act_seq_k is not None else key.shape[1]

        act_seq_q_t = act_seq_q if act_seq_q is not None else torch.tensor([S1] * B, dtype=torch.int32)
        act_seq_k_t = act_seq_k if act_seq_k is not None else torch.tensor([S2] * B, dtype=torch.int32)

        pool_tail_k_t = pool_tail_k
        if pool_tail_k_t is None:
            pool_tail_k_t = torch.tensor([0] * B, dtype=torch.int32)

        if layout_key == "PA_BBND" and block_table is not None:
            block_num = key.shape[0]
            block_size = key.shape[1]
            block_table = _construct_block_table(B, act_seq_k_t, block_num, block_size)

        is_get_output_task = getattr(getattr(self, "task_result", None), "is_get_output_task", False)
        if is_get_output_task:
            return self._compute_output_shape(B, S1, topk, pool_size, layout_query, return_value, query)

        node_name = getattr(self, "name", None)
        is_golden = (node_name == "golden")

        if is_golden:
            return self._run_golden(query, key, weights, act_seq_q_t, act_seq_k_t,
                                    block_table, pool_tail_k_t, B, S1, S2, N1, N2, D,
                                    layout_query, layout_key, topk, pool_size,
                                    mask_mode, return_value)
        else:
            return self._run_operator(query, key, weights, act_seq_q_t, act_seq_k_t,
                                      block_table, pool_tail_k_t, B, S1, S2, N1, N2, D,
                                      layout_query, layout_key, topk, pool_size,
                                      mask_mode, return_value)

    def _run_golden(self, query, key, weights, act_seq_q_t, act_seq_k_t,
                    block_table, pool_tail_k_t, B, S1, S2, N1, N2, D,
                    layout_query, layout_key, topk, pool_size,
                    mask_mode, return_value):
        q_dtype = query.dtype if hasattr(query, "dtype") else torch.float16
        compute_dtype = _get_compute_dtype(q_dtype)

        query_cpu = _safe_to_cpu(query)
        key_cpu = _safe_to_cpu(key)
        weights_cpu = _safe_to_cpu(weights)
        act_q_cpu = _safe_to_cpu(act_seq_q_t)
        act_k_cpu = _safe_to_cpu(act_seq_k_t)
        ptk_cpu = _safe_to_cpu(pool_tail_k_t)

        golden = PKIGoldenRef(
            batch_size=B, q_seq=S1, k_seq=S2,
            q_head_num=N1, k_head_num=N2, head_dim=D,
            layout_query=layout_query, layout_key=layout_key,
            topk=topk, pool_size=pool_size,
            mask_mode=mask_mode, compute_dtype=compute_dtype,
        )
        indices, g_values = golden.forward(
            query_cpu, key_cpu, weights_cpu, act_q_cpu, act_k_cpu,
            block_table, pool_tail_k=ptk_cpu
        )
        values = self._sanitize_values(g_values, compute_dtype)
        return (indices, values)

    def _run_operator(self, query, key, weights, act_seq_q_t, act_seq_k_t,
                      block_table, pool_tail_k_t, B, S1, S2, N1, N2, D,
                      layout_query, layout_key, topk, pool_size,
                      mask_mode, return_value):
        import torch_npu

        q_dtype = query.dtype if hasattr(query, "dtype") else torch.float16
        compute_dtype = _get_compute_dtype(q_dtype)

        device_id = getattr(self, "device_id", 0)
        if device_id is None:
            device_id = 0
        device_str = f"npu:{device_id}"
        torch.npu.set_device(device_str)

        query_npu = _safe_to_npu(query, device_id)
        key_npu = _safe_to_npu(key, device_id)
        weights_npu = _safe_to_npu(weights, device_id)
        act_q_npu = _safe_to_npu(act_seq_q_t, device_id)
        act_k_npu = _safe_to_npu(act_seq_k_t, device_id)
        ptk_npu = _safe_to_npu(pool_tail_k_t, device_id)
        bt_npu = _safe_to_npu(block_table, device_id) if block_table is not None else None

        result = torch_npu.npu_pool_key_indexer(
            query_npu, key_npu, weights_npu, ptk_npu,
            actual_seq_lengths_query=act_q_npu,
            actual_seq_lengths_key=act_k_npu,
            block_table=bt_npu,
            layout_query=layout_query,
            layout_key=layout_key,
            topk=topk,
            pool_size=pool_size,
            mask_mode=mask_mode,
            quant_mode=-1,
            return_value=return_value,
        )
        torch.npu.synchronize()

        if isinstance(result, tuple):
            npu_indices = _safe_to_cpu(result[0])
            npu_values = _safe_to_cpu(result[1]) if (return_value and len(result) > 1) else None
        else:
            npu_indices = _safe_to_cpu(result)
            npu_values = None

        # Golden 归一化
        query_cpu = _safe_to_cpu(query)
        key_cpu = _safe_to_cpu(key)
        weights_cpu = _safe_to_cpu(weights)
        act_q_cpu = _safe_to_cpu(act_seq_q_t)
        act_k_cpu = _safe_to_cpu(act_seq_k_t)
        ptk_cpu = _safe_to_cpu(pool_tail_k_t)
        block_table_cpu = _safe_to_cpu(block_table) if block_table is not None else None

        ext_values = None
        golden = PKIGoldenRef(
            batch_size=B, q_seq=S1, k_seq=S2,
            q_head_num=N1, k_head_num=N2, head_dim=D,
            layout_query=layout_query, layout_key=layout_key,
            topk=topk, pool_size=pool_size,
            mask_mode=mask_mode, compute_dtype=compute_dtype,
        )
        norm_indices, norm_values = golden.forward(
            query_cpu, key_cpu, weights_cpu, act_q_cpu, act_k_cpu,
            block_table_cpu, pool_tail_k=ptk_cpu,
            ext_indices=npu_indices, ext_values=ext_values,
        )
        values = self._sanitize_values(norm_values, compute_dtype)
        return (norm_indices, values)
