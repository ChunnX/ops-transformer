#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Input customization for SparseFlashMla TTK cases."""

import importlib.util
import sys
from pathlib import Path

import torch


def _load_golden():
    name = "smla_ttk_golden"
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


golden = _load_golden()


def _list_value(kwargs, name):
    value = kwargs.get(f"{name}_values")
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1).tolist()
    return [int(item) for item in value]


def _prefix_lengths(value):
    if not value:
        return []
    return [int(value[i + 1]) - int(value[i]) for i in range(len(value) - 1)]


def _data_range(input_ranges, index, default=(-10, 10)):
    if input_ranges and index < len(input_ranges) and input_ranges[index] is not None:
        return repr(list(input_ranges[index]))
    return repr(list(default))


def _case_params(q, ori_kv, cmp_kv, cmp_sparse_indices, layout_q, layout_kv, kwargs):
    cu_q = _list_value(kwargs, "cu_seqlens_q")
    cu_ori = _list_value(kwargs, "cu_seqlens_ori_kv")
    cu_cmp = _list_value(kwargs, "cu_seqlens_cmp_kv")
    seq_q = _list_value(kwargs, "seqused_q")
    seq_ori = _list_value(kwargs, "seqused_ori_kv")
    seq_cmp = _list_value(kwargs, "seqused_cmp_kv")
    residual = _list_value(kwargs, "cmp_residual_kv")

    if layout_q == "BSND":
        batch_size, q_seq, q_heads, head_dim = [int(x) for x in q.shape]
        q_total = batch_size * q_seq
    else:
        q_total, q_heads, head_dim = [int(x) for x in q.shape]
        batch_size = len(seq_q or _prefix_lengths(cu_q))
        q_seq = max(_prefix_lengths(cu_q) or seq_q or [q_total])

    if layout_kv == "BSND":
        _, kv_seq, kv_heads, _ = [int(x) for x in ori_kv.shape]
        kv_total = batch_size * kv_seq
        block_num1, block_size1 = 0, 128
    elif layout_kv == "TND":
        kv_total, kv_heads, _ = [int(x) for x in ori_kv.shape]
        kv_seq = max(_prefix_lengths(cu_ori) or seq_ori or [kv_total])
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
        "q_datarange": _data_range(input_ranges, 0),
        "ori_kv_datarange": _data_range(input_ranges, 1),
        "cmp_kv_datarange": _data_range(input_ranges, 2),
        "qk_equal_len": False,
    })
    return params


def _copy_tensor(dst, src, name):
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


def generate_sparse_flash_mla_inputs(q, *, ori_kv=None, cmp_kv=None, ori_sparse_indices=None,
                                     cmp_sparse_indices=None, ori_block_table=None, cmp_block_table=None,
                                     cu_seqlens_q=None, cu_seqlens_ori_kv=None,
                                     cu_seqlens_cmp_kv=None, seqused_q=None,
                                     seqused_ori_kv=None, seqused_cmp_kv=None,
                                     cmp_residual_kv=None, ori_topk_length=None,
                                     cmp_topk_length=None, sinks=None, **kwargs):
    """Reuse pytest generation to make TTK inputs byte-for-byte valid."""
    del ori_topk_length, cmp_topk_length
    params = _case_params(
        q,
        ori_kv,
        cmp_kv,
        cmp_sparse_indices,
        kwargs.get("layout_q", "BSND"),
        kwargs.get("layout_kv", "BSND"),
        kwargs,
    )
    data = golden.prepare_case(params)
    op_input = data["input"]
    _copy_tensor(q, op_input.get("q"), "q")
    _copy_tensor(ori_kv, op_input.get("ori_kv"), "ori_kv")
    _copy_tensor(cmp_kv, op_input.get("cmp_kv"), "cmp_kv")
    _copy_tensor(ori_sparse_indices, op_input.get("ori_sparse_indices"), "ori_sparse_indices")
    _copy_tensor(cmp_sparse_indices, op_input.get("cmp_sparse_indices"), "cmp_sparse_indices")
    _copy_tensor(ori_block_table, op_input.get("ori_block_table"), "ori_block_table")
    _copy_tensor(cmp_block_table, op_input.get("cmp_block_table"), "cmp_block_table")
    _copy_tensor(cu_seqlens_q, op_input.get("cu_seqlens_q"), "cu_seqlens_q")
    _copy_tensor(cu_seqlens_ori_kv, op_input.get("cu_seqlens_ori_kv"), "cu_seqlens_ori_kv")
    _copy_tensor(cu_seqlens_cmp_kv, op_input.get("cu_seqlens_cmp_kv"), "cu_seqlens_cmp_kv")
    _copy_tensor(seqused_q, op_input.get("seqused_q"), "seqused_q")
    _copy_tensor(seqused_ori_kv, op_input.get("seqused_ori_kv"), "seqused_ori_kv")
    _copy_tensor(seqused_cmp_kv, op_input.get("seqused_cmp_kv"), "seqused_cmp_kv")
    _copy_tensor(cmp_residual_kv, op_input.get("cmp_residual_kv"), "cmp_residual_kv")
    _copy_tensor(sinks, op_input.get("sinks"), "sinks")
