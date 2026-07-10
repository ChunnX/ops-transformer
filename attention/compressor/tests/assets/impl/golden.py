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

import importlib.util
import sys
import types
from pathlib import Path

import torch
import numpy as np


PYTEST_GOLDEN_MODULE = None


def load_pytest_golden_module():
    """Load tests/pytest/compressor_golden.py as the canonical CPU golden.

    The pytest golden module imports torch_npu, torchair and cann_ops_transformer
    at top-level.  In a TTK e2e CPU-golden context these may be unavailable, so we
    install lightweight stubs before executing the module and clean up afterwards.
    The cpu_compressor function itself only depends on torch + numpy.
    """
    global PYTEST_GOLDEN_MODULE
    if PYTEST_GOLDEN_MODULE is not None:
        return PYTEST_GOLDEN_MODULE

    pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
    module_path = pytest_dir / "compressor_golden.py"

    stubs_installed = {}
    for mod_name in ("torch_npu", "torchair"):
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            if mod_name == "torch_npu":
                stub.npu = types.SimpleNamespace(
                    set_device=lambda *a, **kw: None,
                    config=types.SimpleNamespace(allow_internal_format=True),
                )
                testing_mod = types.ModuleType("torch_npu.testing")
                testcase_mod = types.ModuleType("torch_npu.testing.testcase")
                testcase_mod.TestCase = object
                testcase_mod.run_tests = lambda *a, **kw: None
                testing_mod.testcase = testcase_mod
                stub.testing = testing_mod
            sys.modules[mod_name] = stub
            stubs_installed[mod_name] = stub

    cann_stub = None
    if "cann_ops_transformer" not in sys.modules:
        cann_stub = types.ModuleType("cann_ops_transformer")
        ops_stub = types.ModuleType("cann_ops_transformer.ops")
        ops_stub.compressor = lambda *a, **kw: None
        cann_stub.ops = ops_stub
        sys.modules["cann_ops_transformer"] = cann_stub
        sys.modules["cann_ops_transformer.ops"] = ops_stub
        stubs_installed["cann_ops_transformer"] = cann_stub
        stubs_installed["cann_ops_transformer.ops"] = ops_stub

    sys.path.insert(0, str(pytest_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            f"compressor_pytest_golden_{abs(hash(module_path))}", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(pytest_dir))
        except ValueError:
            pass

    PYTEST_GOLDEN_MODULE = module
    return PYTEST_GOLDEN_MODULE


# ========================= TTK e2e adapter =========================
# The CPU reference is loaded from tests/pytest; assets keeps only the TTK adapter layer.
# TTK execution consumes the API tensors supplied by the caller instead of regenerating inputs.

__golden__ = {
    "e2e": {
        "cann_ops_transformer.ops.compressor": "cpu_compressor",
    }
}


def ttk_to_cpu(tensor):
    if tensor is None:
        return None
    if torch.is_tensor(tensor):
        return tensor.detach().cpu()
    return tensor


def ttk_tensor_to_list(val):
    if val is None:
        return None
    if torch.is_tensor(val):
        val = val.detach().cpu().reshape(-1).tolist()
    elif hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, (list, tuple)):
        return [int(v) for v in val]
    return [int(val)]


def cpu_compressor(x, wkv, wgate, state_cache, ape, cmp_ratio=4, *,
                   state_block_table=None, cu_seqlens=None, seqused=None,
                   start_pos=None, coff=1, cache_mode=1, **kwargs):
    """Golden reference implementation for compressor.

    Adapts the TTK-provided API tensors to the pytest CPU golden (cpu_compressor in
    compressor_golden.py).  The NPU API uses a combined state_cache tensor whose last
    dimension concatenates kv_state and score_state; the CPU golden expects them as
    separate float32 tensors plus boolean update-position masks.

    TTK e2e mode compares only the return value (cmp_kv).  The state_cache is also
    modified in-place so that callers comparing inplace outputs get correct results.
    """
    pytest_golden = load_pytest_golden_module()

    x_cpu = ttk_to_cpu(x)
    wkv_cpu = ttk_to_cpu(wkv)
    wgate_cpu = ttk_to_cpu(wgate)
    ape_cpu = ttk_to_cpu(ape)
    state_cache_cpu = ttk_to_cpu(state_cache)
    block_table = ttk_to_cpu(state_block_table)

    state_cache_f32 = state_cache_cpu.to(torch.float32)
    half_dim = state_cache_f32.shape[-1] // 2

    kv_state_tensor = state_cache_f32[:, :, :half_dim].contiguous().clone()
    score_state_tensor = state_cache_f32[:, :, half_dim:].contiguous().clone()

    update_kv = torch.zeros(kv_state_tensor.shape, dtype=torch.bool)
    update_score = torch.zeros(score_state_tensor.shape, dtype=torch.bool)

    if ape_cpu is not None and torch.is_tensor(ape_cpu):
        ape_cpu = ape_cpu.to(torch.float32)

    start_pos_list = ttk_tensor_to_list(start_pos) if start_pos is not None else None
    cu_seqlens_list = ttk_tensor_to_list(cu_seqlens) if cu_seqlens is not None else None
    seqused_list = ttk_tensor_to_list(seqused) if seqused is not None else None

    if start_pos_list is None:
        if cu_seqlens_list is not None:
            B = len(cu_seqlens_list) - 1
        elif x_cpu.dim() == 3:
            B = x_cpu.shape[0]
        else:
            B = 1
        start_pos_list = [0] * B

    cmp_kv, cmp_kv_mask = pytest_golden.cpu_compressor(
        x_cpu, wkv_cpu, wgate_cpu, kv_state_tensor, score_state_tensor,
        update_kv, update_score, ape_cpu,
        block_table=block_table, cu_seqlens=cu_seqlens_list, seqused=seqused_list,
        start_pos=start_pos_list, cmp_ratio=cmp_ratio, coff=coff, cache_mode=cache_mode,
    )

    state_cache_f32[:, :, :half_dim] = kv_state_tensor
    state_cache_f32[:, :, half_dim:] = score_state_tensor
    if torch.is_tensor(state_cache):
        state_cache.copy_(state_cache_f32.to(state_cache.dtype))

    return cmp_kv
