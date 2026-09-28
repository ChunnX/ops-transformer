# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# TTK plugin for the *formal* Compressor and QuantCompressor ABI.
#
# ABI differences from the historical experimental plugin:
#   * compressor has no norm_weight / rope_sin / rope_cos inputs and no
#     rope_head_dim / norm_eps / rotary_mode attributes (the model applies RMSNorm
#     and RoPE after the op), and it declares four outputs.
#   * quant_compressor inserts x_descale / wkv_descale / wgate_descale after ape,
#     carries hifloat8 x/wkv/wgate/x_cache, and declares two outputs.
#
# Fixtures are read from $PRO0921_CASE_DIR; expected outputs are listed in
# <case>/golden.json so one plugin serves every generated case.  Goldens are
# returned in op-def output order and as float32; TTK's comparison normalises
# both sides (bfloat16 -> float32) before isclose.

import json
import os
from pathlib import Path

import numpy as np
import importlib.util


def _install_native_hooks():
    path = Path(__file__).with_name("native_hooks.py")
    spec = importlib.util.spec_from_file_location("pro0921_native_hooks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install()



__input__ = {"kernel": {"compressor": "inputs", "quant_compressor": "inputs_quant", "sparse_flash_mla": "inputs_mla"}}
__golden__ = {"kernel": {"compressor": "golden", "quant_compressor": "golden", "sparse_flash_mla": "golden"}}

COMPRESSOR_INPUTS = (
    "x", "wkv", "wgate", "state_cache", "ape", "state_block_table",
    "cu_seqlens", "seqused", "start_pos", "x_cache", "x_cache_slot_mapping",
)

QUANT_INPUTS = (
    "x", "wkv", "wgate", "state_cache", "ape",
    "x_descale", "wkv_descale", "wgate_descale",
    "state_block_table", "cu_seqlens", "seqused", "start_pos",
    "x_cache", "x_cache_slot_mapping",
)


def _root():
    return Path(os.environ["PRO0921_CASE_DIR"])


def _load(names, args):
    _install_native_hooks()
    root = _root()
    resolved = []
    for name, arg in zip(names, args):
        if arg is None:
            resolved.append(None)
            continue
        path = root / (name + ".npy")
        if not path.exists():
            resolved.append(None)
            continue
        data = np.load(path)
        target = np.asarray(arg).dtype
        if "hifloat8" in str(target):
            # uint8 bit carrier; never let numpy re-encode the codes.
            resolved.append(data.view(target))
        else:
            resolved.append(data.astype(target))
    return resolved


def inputs(x, wkv, wgate, state_cache, ape, state_block_table, cu_seqlens, seqused,
           start_pos, x_cache, x_cache_slot_mapping, **kwargs):
    return _load(COMPRESSOR_INPUTS, (
        x, wkv, wgate, state_cache, ape, state_block_table, cu_seqlens, seqused,
        start_pos, x_cache, x_cache_slot_mapping))


def inputs_quant(x, wkv, wgate, state_cache, ape, x_descale, wkv_descale,
                 wgate_descale, state_block_table, cu_seqlens, seqused, start_pos,
                 x_cache, x_cache_slot_mapping, **kwargs):
    return _load(QUANT_INPUTS, (
        x, wkv, wgate, state_cache, ape, x_descale, wkv_descale, wgate_descale,
        state_block_table, cu_seqlens, seqused, start_pos, x_cache,
        x_cache_slot_mapping))


def golden(*args, **kwargs):
    root = _root()
    manifest = json.loads((root / "golden.json").read_text(encoding="utf-8"))
    out = []
    for name in manifest["files"]:
        if name is None:
            out.append(None)
            continue
        out.append(np.load(root / name).astype(np.float32))
    return out


def inputs_mla(q, ori_kv, cmp_kv, ori_sparse_indices, cmp_sparse_indices,
               ori_block_table, cmp_block_table, cu_seqlens_q, cu_seqlens_ori_kv,
               cu_seqlens_cmp_kv, seqused_q, seqused_ori_kv, seqused_cmp_kv,
               cmp_residual_kv, ori_topk_length, cmp_topk_length, sinks, metadata, **kwargs):
    names = ("q", "ori_kv", "cmp_kv", "ori_sparse_indices", "cmp_sparse_indices",
             "ori_block_table", "cmp_block_table", "cu_seqlens_q", "cu_seqlens_ori_kv",
             "cu_seqlens_cmp_kv", "seqused_q", "seqused_ori_kv", "seqused_cmp_kv",
             "cmp_residual_kv", "ori_topk_length", "cmp_topk_length", "sinks", "metadata")
    values = locals()
    return _load(names, [values[name] for name in names])
