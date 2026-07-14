#!/usr/bin/python
# -*- coding: utf-8 -*-
"""CPU golden adapter for QuantLightningIndexer V2 TTK cases."""

import importlib.util
import sys
from pathlib import Path

import torch

PYTEST_GOLDEN_MODULE = None
_TOPK_SCORES = None
_INDEX_OFFSETS = None


def load_pytest_golden_module():
    global PYTEST_GOLDEN_MODULE
    if PYTEST_GOLDEN_MODULE is not None:
        return PYTEST_GOLDEN_MODULE
    pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
    module_path = pytest_dir / "quant_lightning_indexer_v2_golden.py"
    sys.path.insert(0, str(pytest_dir))
    try:
        spec = importlib.util.spec_from_file_location(f"qli_v2_pytest_golden_{abs(hash(module_path))}", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(pytest_dir))
        except ValueError:
            pass
    PYTEST_GOLDEN_MODULE = module
    return module


def set_topk_scores(scores):
    global _TOPK_SCORES
    _TOPK_SCORES = scores


def get_topk_scores():
    return _TOPK_SCORES


def set_index_offsets(offsets):
    global _INDEX_OFFSETS
    _INDEX_OFFSETS = offsets


def get_index_offsets():
    return _INDEX_OFFSETS


def to_list(value):
    if value is None:
        return []
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(value)]


def to_cpu(value):
    return value.detach().cpu() if torch.is_tensor(value) else value


def restore_pa_bbnd(tensor, block_table, seqused_k, block_size, has_head_dim=True):
    if tensor is None or block_table is None:
        return tensor
    src = to_cpu(tensor)
    table = to_cpu(block_table).to(torch.int64)
    actual = to_list(seqused_k)
    if not actual:
        return tensor
    batch = len(actual)
    n_head = src.shape[2]
    max_s = max(actual)
    if has_head_dim:
        tail = src.shape[3:]
        restored = torch.zeros((batch, n_head, max_s) + tail, dtype=src.dtype)
    else:
        restored = torch.zeros((batch, n_head, max_s), dtype=src.dtype)
    for b_idx, cur_len in enumerate(actual):
        copied = 0
        pages = min(table.shape[1], (cur_len + block_size - 1) // block_size)
        for page_idx in range(pages):
            block_id = int(table[b_idx, page_idx].item())
            if block_id < 0:
                continue
            take = min(block_size, cur_len - copied)
            if has_head_dim:
                restored[b_idx, :, copied:copied + take, ...] = src[block_id, :take, :, ...].permute(1, 0, 2)
            else:
                restored[b_idx, :, copied:copied + take] = src[block_id, :take, :].permute(1, 0)
            copied += take
            if copied >= cur_len:
                break
    return restored


def lengths_from_cu(cu):
    vals = to_list(cu)
    return [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]


def scores_to_layout(golden, scores_bnsd, layout_q, q_lengths):
    shape = list(scores_bnsd.shape)
    return golden.trans_bnsd_to_layout(scores_bnsd, shape, layout_q, q_lengths)


def infer_geometry(query, key, layout_query, layout_key, cu_q, cu_k, seq_q, seq_k):
    q_capacity = lengths_from_cu(cu_q)
    k_capacity = lengths_from_cu(cu_k)
    if layout_query == "BSND":
        batch_size, q_seq, q_head_num, head_dim = [int(x) for x in query.shape]
        q_t_size = batch_size * q_seq
        q_lengths = to_list(seq_q) or [q_seq] * batch_size
    else:
        q_t_size, q_head_num, head_dim = [int(x) for x in query.shape]
        q_lengths = to_list(seq_q) or q_capacity
        batch_size = len(q_lengths)
        q_seq = max(q_capacity or q_lengths, default=q_t_size)
    if layout_key == "BSND":
        _, k_seq, k_head_num, _ = [int(x) for x in key.shape]
        k_t_size = batch_size * k_seq
        block_num, block_size = 0, 1
        k_lengths = to_list(seq_k) or [k_seq] * batch_size
    elif layout_key == "TND":
        k_t_size, k_head_num, _ = [int(x) for x in key.shape]
        k_lengths = to_list(seq_k) or k_capacity
        k_seq = max(k_capacity or k_lengths, default=k_t_size)
        block_num, block_size = 0, 1
    else:
        block_num, block_size, k_head_num, _ = [int(x) for x in key.shape]
        k_lengths = to_list(seq_k)
        k_seq = max(k_lengths, default=block_size)
        k_t_size = sum(k_lengths) if k_lengths else block_num * block_size
    return (
        batch_size, q_seq, k_seq, q_t_size, k_t_size,
        q_head_num, k_head_num, head_dim, block_size, block_num,
        q_lengths, k_lengths, q_capacity,
    )


def cpu_quant_lightning_indexer_v2(query, key, weights, query_dequant_scale, key_dequant_scale, *,
                                   cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None,
                                   seqused_k=None, cmp_residual_k=None, block_table=None,
                                   output_idx_offset=None, sparse_count=0, quant_mode=1,
                                   max_seqlen_q=-1,
                                   layout_query="BSND", layout_key="BSND", sparse_mode=0,
                                   cmp_ratio=1, return_value=0, **kwargs):
    del kwargs
    set_topk_scores(None)
    set_index_offsets(to_cpu(output_idx_offset) if output_idx_offset is not None else None)

    query_cpu = to_cpu(query)
    key_cpu = to_cpu(key)
    weights_cpu = to_cpu(weights)
    q_scale_cpu = to_cpu(query_dequant_scale)
    k_scale_cpu = to_cpu(key_dequant_scale)
    cu_q_cpu = to_cpu(cu_seqlens_q)
    cu_k_cpu = to_cpu(cu_seqlens_k)
    seq_q_cpu = to_cpu(seqused_q)
    seq_k_cpu = to_cpu(seqused_k)
    residual_cpu = to_cpu(cmp_residual_k)
    block_table_cpu = to_cpu(block_table)
    offsets_cpu = to_cpu(output_idx_offset)

    geometry = infer_geometry(
        query_cpu, key_cpu, layout_query, layout_key,
        cu_q_cpu, cu_k_cpu, seq_q_cpu, seq_k_cpu,
    )
    (batch_size, q_seq, k_seq, q_t_size, k_t_size,
     q_head_num, k_head_num, head_dim, block_size, block_num,
     lengths_q, lengths_k, q_capacity) = geometry

    if int(cmp_ratio) == 1 or int(sparse_mode) == 0:
        residual_for_cpu = [0] * int(batch_size)
    else:
        residual_for_cpu = to_list(residual_cpu)

    if layout_key == "PA_BBND":
        key_for_golden = restore_pa_bbnd(key_cpu, block_table_cpu, seq_k_cpu, int(block_size), has_head_dim=True)
        k_scale_for_golden = restore_pa_bbnd(k_scale_cpu, block_table_cpu, seq_k_cpu, int(block_size), has_head_dim=False)
    else:
        key_for_golden = key_cpu
        k_scale_for_golden = k_scale_cpu

    golden_cls = load_pytest_golden_module().GeneralizedQLIV2
    actual_seq_dtype = next(
        (value.dtype for value in (cu_q_cpu, cu_k_cpu, seq_q_cpu, seq_k_cpu) if value is not None),
        torch.int32,
    )
    golden = golden_cls(
        int(batch_size), int(q_seq), int(k_seq), int(q_t_size or 0), int(k_t_size or 0),
        int(q_head_num), int(k_head_num), int(head_dim), int(block_size), int(block_num),
        query_cpu.dtype, q_scale_cpu.dtype,
        actual_seq_dtype, to_list(cu_q_cpu) or None, to_list(cu_k_cpu) or None,
        lengths_q, lengths_k, residual_for_cpu, int(max_seqlen_q), int(quant_mode),
        layout_query, layout_key, int(sparse_count), int(sparse_mode),
        None, None, None, None, None, int(cmp_ratio), int(return_value),
    )
    indices, full_scores_bnsd, sparse_value = golden.forward(
        query_cpu, key_for_golden, weights_cpu, q_scale_cpu, k_scale_for_golden,
        cu_q_cpu, cu_k_cpu, seq_q_cpu, seq_k_cpu, block_table_cpu, offsets_cpu)
    score_lengths = q_capacity if layout_query == "TND" else lengths_q
    set_topk_scores(scores_to_layout(golden, full_scores_bnsd, layout_query, score_lengths))
    if int(return_value):
        return indices, sparse_value
    return indices, torch.zeros(0, dtype=full_scores_bnsd.dtype)
