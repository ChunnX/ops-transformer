#!/usr/bin/python
# -*- coding: utf-8 -*-
"""CPU golden adapter for LightningIndexer V2 TTK cases."""

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
    module_path = pytest_dir / "lightning_indexer_v2_golden.py"
    sys.path.insert(0, str(pytest_dir))
    try:
        spec = importlib.util.spec_from_file_location(f"li_v2_pytest_golden_{abs(hash(module_path))}", module_path)
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


def restore_pa_bbnd(key, block_table, seqused_k, block_size):
    if key is None or block_table is None:
        return key
    src = to_cpu(key)
    table = to_cpu(block_table).to(torch.int64)
    actual = to_list(seqused_k)
    if not actual:
        return key
    batch = len(actual)
    n_head = src.shape[2]
    head_dim = src.shape[3]
    max_s = max(actual)
    restored = torch.zeros((batch, n_head, max_s, head_dim), dtype=src.dtype)
    for b_idx, cur_len in enumerate(actual):
        copied = 0
        pages = min(table.shape[1], (cur_len + block_size - 1) // block_size)
        for page_idx in range(pages):
            block_id = int(table[b_idx, page_idx].item())
            if block_id < 0:
                continue
            take = min(block_size, cur_len - copied)
            restored[b_idx, :, copied:copied + take, :] = src[block_id, :take, :, :].permute(1, 0, 2)
            copied += take
            if copied >= cur_len:
                break
    return restored


def scores_to_layout(golden, scores_bnsd, layout_q, q_lengths):
    shape = list(scores_bnsd.shape)
    return golden.trans_bnsd_to_layout(scores_bnsd, shape, layout_q, q_lengths)


def infer_geometry(q, k, layout_q, layout_k, cu_q, cu_k, seq_q, seq_k):
    q_lengths = to_list(seq_q) or lengths_from_prefix(cu_q)
    k_lengths = to_list(seq_k) or lengths_from_prefix(cu_k)
    if layout_q == "BSND":
        batch_size, q_seq, q_head_num, head_dim = [int(x) for x in q.shape]
        q_t_size = batch_size * q_seq
        q_lengths = q_lengths or [q_seq] * batch_size
    else:
        q_t_size, q_head_num, head_dim = [int(x) for x in q.shape]
        batch_size = len(q_lengths)
        q_seq = max(q_lengths, default=q_t_size)
    if layout_k == "BSND":
        _, k_seq, k_head_num, _ = [int(x) for x in k.shape]
        k_t_size = batch_size * k_seq
        block_num, block_size = 0, 1
        k_lengths = k_lengths or [k_seq] * batch_size
    elif layout_k == "TND":
        k_t_size, k_head_num, _ = [int(x) for x in k.shape]
        k_seq = max(k_lengths, default=k_t_size)
        block_num, block_size = 0, 1
    else:
        block_num, block_size, k_head_num, _ = [int(x) for x in k.shape]
        k_seq = max(k_lengths, default=block_size)
        k_t_size = sum(k_lengths) if k_lengths else block_num * block_size
    return (
        batch_size, q_seq, k_seq, q_t_size, k_t_size,
        q_head_num, k_head_num, head_dim, block_size, block_num,
        q_lengths, k_lengths,
    )


def lengths_from_prefix(value):
    vals = to_list(value)
    return [vals[i + 1] - vals[i] for i in range(len(vals) - 1)] if len(vals) > 1 else []


def cpu_lightning_indexer_v2(q, k, w, *, cu_seqlens_q=None, cu_seqlens_k=None,
                             seqused_q=None, seqused_k=None, cmp_residual_k=None,
                             block_table=None, output_idx_offset=None, topk=0,
                             max_seqlen_q=-1, layout_q="BSND",
                             layout_k="BSND", mask_mode=0, cmp_ratio=1,
                             return_value=0, **kwargs):
    del kwargs
    set_topk_scores(None)
    set_index_offsets(to_cpu(output_idx_offset) if output_idx_offset is not None else None)
    q_cpu = to_cpu(q)
    k_cpu = to_cpu(k)
    w_cpu = to_cpu(w)
    cu_q_cpu = to_cpu(cu_seqlens_q)
    cu_k_cpu = to_cpu(cu_seqlens_k)
    seq_q_cpu = to_cpu(seqused_q)
    seq_k_cpu = to_cpu(seqused_k)
    residual_cpu = to_cpu(cmp_residual_k)
    offsets_cpu = to_cpu(output_idx_offset)
    block_table_cpu = to_cpu(block_table)

    geometry = infer_geometry(
        q_cpu, k_cpu, layout_q, layout_k,
        cu_q_cpu, cu_k_cpu, seq_q_cpu, seq_k_cpu,
    )
    (batch_size, q_seq, k_seq, q_t_size, k_t_size,
     q_head_num, k_head_num, head_dim, block_size, block_num,
     q_lengths, _) = geometry

    if layout_k == "PA_BBND":
        k_for_golden = restore_pa_bbnd(k_cpu, block_table_cpu, seq_k_cpu, int(block_size))
    else:
        k_for_golden = k_cpu

    golden_cls = load_pytest_golden_module().GeneralizedLIV2
    golden = golden_cls(
        int(batch_size), int(q_seq), int(k_seq), int(q_t_size), int(k_t_size),
        int(q_head_num), int(k_head_num), int(head_dim), int(block_size), int(block_num),
        q_cpu.dtype, to_list(cu_q_cpu) or None, to_list(cu_k_cpu) or None,
        to_list(seq_q_cpu) or None, to_list(seq_k_cpu) or None,
        to_list(residual_cpu) or None, layout_q, layout_k, int(topk), int(max_seqlen_q),
        int(mask_mode), int(cmp_ratio), int(return_value),
    )
    indices, full_scores_bnsd, sparse_value = golden.forward(
        q_cpu, k_for_golden, w_cpu, cu_q_cpu, cu_k_cpu, seq_q_cpu, seq_k_cpu,
        residual_cpu, block_table_cpu, offsets_cpu)
    score_lengths = golden.cu_seqlens_q if layout_q == "TND" else q_lengths
    set_topk_scores(scores_to_layout(golden, full_scores_bnsd, layout_q, score_lengths))
    if int(return_value):
        return indices, sparse_value
    return indices, torch.zeros(0, dtype=full_scores_bnsd.dtype)
