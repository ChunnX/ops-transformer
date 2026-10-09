
#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
import sys
from pathlib import Path
import pandas as pd 
import numpy as np
import torch
import torch_npu
import pytest
import random
import math
import ast
from cann_ops_transformer.ops import compressor

# The Pro0921 grouped contract helper lives in tests/pytest/; resolve it
# relative to this file so the batch loader is import-order independent.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pro0921_grouped_cases import pt_grouped_args  # noqa: E402


def _readback(tensor):
    """CPU read-back for an exact compare: bf16 -> fp32, integer carriers as-is."""
    if tensor is None:
        return None
    tensor = tensor.detach().cpu()
    if tensor.dtype.is_floating_point:
        return tensor.to(torch.float32)
    return tensor


def _assert_grouped_readback(test_data, x_cache, state_cache):
    """Accuracy guard for the grouped batch entry.

    The old entrance only passed ``x_cache`` to the operator; this checks the
    in-place side effects the operator is responsible for:

    * the full ``x_cache`` ring read-back equals ``expected_x_cache`` (bf16
      widened to fp32, quant carriers bit-for-bit);
    * the full ``state_cache`` equals the independent golden state, so a
      grouped call that also corrupts a region it must not touch is caught.

    A mismatch raises ``AssertionError`` (never a soft print/return).
    """
    expected_ring = test_data.get("expected_x_cache")
    if expected_ring is None:
        raise AssertionError(
            "grouped .pt is missing expected_x_cache; regenerate it with the current "
            "batch/compressor_pt_save.py (the read-back must be checked, not skipped)")
    have_ring = _readback(x_cache)
    want_ring = _readback(expected_ring)
    if have_ring is None or have_ring.shape != want_ring.shape:
        raise AssertionError(
            "grouped x_cache read-back shape %r != expected %r"
            % (None if have_ring is None else tuple(have_ring.shape), tuple(want_ring.shape)))
    if not torch.equal(have_ring, want_ring):
        diff = int((have_ring != want_ring).sum())
        raise AssertionError(
            "grouped x_cache read-back mismatch: %d/%d elements differ"
            % (diff, have_ring.numel()))

    expected_state = torch.cat(
        [test_data["cpu_kv_state"], test_data["cpu_score_state"]], dim=2).to(torch.float32)
    have_state = _readback(state_cache)
    if have_state.shape != expected_state.shape:
        raise AssertionError(
            "grouped state_cache read-back shape %r != expected %r"
            % (tuple(have_state.shape), tuple(expected_state.shape)))
    update = torch.cat([test_data["update_kv"], test_data["update_score"]], dim=2).to(torch.bool)
    kept = ~update
    if bool(kept.any()) and not torch.equal(have_state[kept], expected_state[kept]):
        diff = int((have_state[kept] != expected_state[kept]).sum())
        raise AssertionError(
            "grouped state_cache non-updated region changed: %d elements differ" % diff)
    if bool(update.any()):
        close = torch.isclose(have_state[update], expected_state[update],
                              rtol=0.0078125, atol=1e-4)
        if not bool(close.all()):
            raise AssertionError(
                "grouped state_cache updated region mismatch: %d/%d elements off"
                % (int((~close).sum()), int(update.sum())))


def test_compressor_process(filepath, device_id=0):
    # 加载测试数据
    # test_data = torch.load(filepath, map_location="cpu")
    test_data = torch.load(filepath, map_location="cpu",weights_only=False)

    params = test_data['params']
    cpu_result = test_data['cpu_result']
    kv_mask_result = test_data['kv_mask_result']
    update_kv = test_data['update_kv']
    update_score = test_data['update_score']
    cpu_kv_state = test_data['cpu_kv_state']
    cpu_score_state = test_data['cpu_score_state']

    print("执行用例：", filepath)
    torch_npu.npu.set_device(device_id)

    x =test_data['x']
    wkv = test_data['wkv']
    wgate = test_data['wgate']
    kv_state = test_data['kv_state']
    score_state = test_data['score_state']
    ape = test_data['ape']
    block_table = test_data['block_table']
    cu_seqlens = test_data['cu_seqlens']
    seqused = test_data['seqused']
    start_pos = test_data['start_pos']
    cmp_ratio = test_data['cmp_ratio']
    coff = test_data['coff']
    cache_mode = test_data['cache_mode']

    is_contiguous_flag = True
    if cache_mode == 1:  # 连续buffer(支持在第二维非连续)
        if is_contiguous_flag:
            state_cache = torch.zeros((kv_state.shape[0], kv_state.shape[1], 2*kv_state.shape[2]))
            state_cache[:, :, :state_cache.shape[2]//2] = kv_state.clone()
            state_cache[:, :, state_cache.shape[2]//2:] = score_state.clone()
        else :
            layer_stride = random.randint(1, 10)
            print(f"layer_stride: {layer_stride}")
            state_cache_pad = torch.zeros((kv_state.shape[0], (kv_state.shape[1] + layer_stride) * kv_state.shape[2] * 2))
            print(f"state_cache_pad: shape {state_cache_pad.shape}")
            # 使用 as_strided 创建非连续视图
            state_cache = torch.as_strided(
                state_cache_pad,
                size=(kv_state.shape[0], kv_state.shape[1], kv_state.shape[2] * 2),
                stride=((kv_state.shape[1] + layer_stride) * kv_state.shape[2] * 2, kv_state.shape[2] * 2, 1)
            )
            # 填充数据
            state_cache[:, :, :kv_state.shape[2]] = kv_state.clone()
            state_cache[:, :, kv_state.shape[2]:] = score_state.clone()
            print(f"state_cache: shape {state_cache.shape}, dtype: {state_cache.dtype}, is_contiguous: {state_cache.is_contiguous()}, stride: {state_cache.stride()}")  
    else:
        layer_num = random.randint(1, 50)
        layer_start_idx = random.randint(0, layer_num-1)
        print(f"layer_num: {layer_num}")
        print(f"layer_start_idx: {layer_start_idx}")
        state_cache_pad = torch.zeros((kv_state.shape[0], layer_num*kv_state.shape[1]*kv_state.shape[2]*2))
        print(f"state_cache_pad: shape {state_cache_pad.shape}")
        # state_cache_pad = state_cache_pad.to("npu:%s" % DEVICE_ID)
        state_cache = state_cache_pad[:, layer_start_idx*kv_state.shape[1]*kv_state.shape[2]*2: (layer_start_idx+1)*kv_state.shape[1]*kv_state.shape[2]*2].view(-1, kv_state.shape[1], kv_state.shape[2]*2)
        # state_cache = state_cache.to("npu:%s" % DEVICE_ID)
        state_cache[:, :, :state_cache.shape[2]//2] = kv_state.clone()
        state_cache[:, :, state_cache.shape[2]//2:] = score_state.clone()
        print(f"state_cache: shape {state_cache.shape}, dtype: {state_cache.dtype}, is_contiguous: {state_cache.is_contiguous()}, stride0: {state_cache.stride(0)}")


    state_cache = state_cache.npu()
    if cu_seqlens is not None: 
        cu_seqlens = cu_seqlens.npu()
    if seqused is not None: 
        seqused = seqused.npu()
    if start_pos is not None: 
        start_pos = torch.tensor(start_pos).to(torch.int32).npu()
    if block_table is not None: 
        kv_block_table = block_table.npu()
        score_block_table = block_table.npu()

    # 调用compressor算子：普通路径与grouped路径按显式 path 字段选择
    grouped = pt_grouped_args(test_data)
    op_kwargs = dict(
        cmp_ratio=cmp_ratio,
        state_block_table=kv_block_table,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        start_pos=start_pos,
        coff=coff,
        cache_mode=cache_mode,
    )
    x_cache = None
    if grouped["path"] == "grouped":
        x_cache = grouped["x_cache"].npu()
        op_kwargs.update(
            x_cache=x_cache,
            x_cache_slot_mapping=grouped["x_cache_slot_mapping"].npu(),
            x_cache_update=grouped["x_cache_update"],
        )
    npu_result = compressor(
                x.npu(),
                wkv.npu(),
                wgate.npu(),
                state_cache,
                ape.npu(),
                **op_kwargs
            )
    # grouped 路径的副作用在这里内部校验（x_cache 精确读回 + 全量 state），
    # 失败直接 raise AssertionError，不能只 print/return false。
    if grouped["path"] == "grouped":
        torch_npu.npu.synchronize()
        _assert_grouped_readback(test_data, x_cache, state_cache)
    return cpu_result, kv_mask_result, npu_result ,cpu_kv_state, update_kv, cpu_score_state, update_score, state_cache, params
