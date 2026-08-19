#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TestSpec adapter for Compressor assets."""

import importlib.util
from pathlib import Path

import torch

try:
    from ttk.utilities.container_utils import get_global_storage
except Exception:
    get_global_storage = None

try:
    from ttk.core_modules.comparison.cross_check import CrossCheckComparison
    from ttk.core_modules.comparison.resolve import resolve_tolerance
    _TTK_CROSS_CHECK_AVAILABLE = True
except Exception:
    _TTK_CROSS_CHECK_AVAILABLE = False


ASSET_IMPL_DIR = Path(__file__).with_name("impl")


def load_impl_module(stem):
    path = ASSET_IMPL_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(
        f"compressor_assets_impl_{stem}_{abs(hash(path))}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


golden_module = load_impl_module("golden")
inputs_module = load_impl_module("inputs")
compare_module = load_impl_module("compare")


_OUTPUT_NAMES = [
    "cmp_kv",
    "kv_state_update",
    "score_state_update",
    "kv_state_origin",
    "score_state_origin",
    "softmax_score",
    "kv",
]


def _register_compressor_api_params():
    try:
        from ttk.utilities.simple_param_extractor import (
            ParamInfo,
            register_api_params,
        )
    except Exception:
        return
    params = [
        ParamInfo(name="x", type="Tensor"),
        ParamInfo(name="wkv", type="Tensor"),
        ParamInfo(name="wgate", type="Tensor"),
        ParamInfo(name="state_cache", type="Tensor"),
        ParamInfo(name="ape", type="Tensor"),
        ParamInfo(name="cmp_ratio", type="int", default=4, is_optional=True),
        ParamInfo(name="state_block_table", type="Tensor", default=None,
                  is_optional=True, is_keyword_only=True),
        ParamInfo(name="cu_seqlens", type="Tensor", default=None,
                  is_optional=True, is_keyword_only=True),
        ParamInfo(name="seqused", type="Tensor", default=None,
                  is_optional=True, is_keyword_only=True),
        ParamInfo(name="start_pos", type="Tensor", default=None,
                  is_optional=True, is_keyword_only=True),
        ParamInfo(name="coff", type="int", default=1, is_optional=True,
                  is_keyword_only=True),
        ParamInfo(name="cache_mode", type="int", default=1, is_optional=True,
                  is_keyword_only=True),
    ]
    register_api_params(
        "cann_ops_transformer.ops.compressor", params, source="manual_compress"
    )


_register_compressor_api_params()


def _get_compare_method():
    if get_global_storage is not None:
        try:
            return getattr(get_global_storage(), "compare_method", None)
        except Exception:
            return None
    return None


def _resolve_cross_check_params(dtype_str="float32"):
    tolerance = {dtype_str: {"standard": "cross_check", "level": "L1"}}
    standards = resolve_tolerance(tolerance, None, None, [dtype_str], "cross_check")
    return standards[0].params


def _ttk_cross_check_single(npu_out, golden_out, bench_out, idx, dtype_str, params):
    c = CrossCheckComparison(
        npu_out, golden_out, idx, dtype_str, params, third_party=bench_out
    )
    precision_str, log, is_pass, metrics = c.compare()
    return {
        "pass": is_pass,
        "precision": precision_str,
        "metrics": metrics,
        "error_info": None if is_pass else log,
    }


def _split_aclnn_outputs(outputs, kwargs, bench_outputs):
    GOLDEN_OUTPUT_COUNT = 4
    golden_outputs = [compare_module._to_torch(g) for g in outputs[-GOLDEN_OUTPUT_COUNT:]]
    npu_outputs = [compare_module._to_torch(o) for o in outputs[:-GOLDEN_OUTPUT_COUNT]]
    cmp_kv_mask = kwargs.get("cmp_kv_mask", None)
    mid_result_mask = kwargs.get("mid_result_mask", None)
    gradEnabled = kwargs.get("gradEnabled", None)
    update_kv = kwargs.get("update_kv", None)
    update_score = kwargs.get("update_score", None)

    npu_sub = []
    golden_sub = []
    bench_sub = []

    if len(npu_outputs) == 1:
        npu_sub.append(npu_outputs[0][cmp_kv_mask].to(torch.float32))
        golden_sub.append(golden_outputs[0][cmp_kv_mask].to(torch.float32))
        bench = bench_outputs[0] if bench_outputs else None
        bench_sub.append(bench[cmp_kv_mask].to(torch.float32) if bench is not None else None)
        for i in range(1, len(golden_outputs)):
            npu_sub.append(None)
            golden_sub.append(None)
            bench_sub.append(None)
        return npu_sub, golden_sub, bench_sub, _OUTPUT_NAMES

    npu_cmp_kv = npu_outputs[0].to(torch.float32)
    npu_state_cache = npu_outputs[3].to(torch.float32)
    cpu_cmp_kv = golden_outputs[0].to(torch.float32)
    cpu_state_cache = golden_outputs[3].to(torch.float32)

    npu_sub.append(npu_cmp_kv[cmp_kv_mask])
    golden_sub.append(cpu_cmp_kv[cmp_kv_mask])
    bench_sub.append(
        bench_outputs[0][cmp_kv_mask] if bench_outputs else None
    )

    if update_kv is not None and update_score is not None:
        npu_sub.append(npu_state_cache[:, :, : npu_state_cache.shape[2] // 2][update_kv])
        npu_sub.append(npu_state_cache[:, :, npu_state_cache.shape[2] // 2 :][update_score])
        npu_sub.append(npu_state_cache[:, :, : npu_state_cache.shape[2] // 2][~update_kv])
        npu_sub.append(npu_state_cache[:, :, npu_state_cache.shape[2] // 2 :][~update_score])

        golden_sub.append(cpu_state_cache[:, :, : cpu_state_cache.shape[2] // 2][update_kv])
        golden_sub.append(cpu_state_cache[:, :, cpu_state_cache.shape[2] // 2 :][update_score])
        golden_sub.append(cpu_state_cache[:, :, : cpu_state_cache.shape[2] // 2][~update_kv])
        golden_sub.append(cpu_state_cache[:, :, cpu_state_cache.shape[2] // 2 :][~update_score])

        if bench_outputs and len(bench_outputs) > 1:
            bench_state = compare_module._to_torch(bench_outputs[1]).to(torch.float32)
            bench_sub.append(bench_state[:, :, : bench_state.shape[2] // 2][update_kv])
            bench_sub.append(bench_state[:, :, bench_state.shape[2] // 2 :][update_score])
            bench_sub.append(bench_state[:, :, : bench_state.shape[2] // 2][~update_kv])
            bench_sub.append(bench_state[:, :, bench_state.shape[2] // 2 :][~update_score])
        else:
            bench_sub.extend([None, None, None, None])
    else:
        npu_sub.append(npu_state_cache[:, :, : npu_state_cache.shape[2] // 2])
        npu_sub.append(npu_state_cache[:, :, npu_state_cache.shape[2] // 2 :])
        npu_sub.append(None)
        npu_sub.append(None)

        golden_sub.append(cpu_state_cache[:, :, : cpu_state_cache.shape[2] // 2])
        golden_sub.append(cpu_state_cache[:, :, cpu_state_cache.shape[2] // 2 :])
        golden_sub.append(None)
        golden_sub.append(None)

        if bench_outputs and len(bench_outputs) > 1:
            bench_state = compare_module._to_torch(bench_outputs[1]).to(torch.float32)
            bench_sub.append(bench_state[:, :, : bench_state.shape[2] // 2])
            bench_sub.append(bench_state[:, :, bench_state.shape[2] // 2 :])
            bench_sub.append(None)
            bench_sub.append(None)
        else:
            bench_sub.extend([None, None, None, None])

    if gradEnabled:
        npu_sub.append(npu_outputs[1][mid_result_mask].to(torch.float32))
        npu_sub.append(npu_outputs[2][mid_result_mask].to(torch.float32))
        golden_sub.append(golden_outputs[1][mid_result_mask].to(torch.float32))
        golden_sub.append(golden_outputs[2][mid_result_mask].to(torch.float32))
        bench_sub.append(None)
        bench_sub.append(None)

    return npu_sub, golden_sub, bench_sub, _OUTPUT_NAMES


def _run_ttk_cross_check_aclnn(outputs, kwargs, bench_outputs):
    npu_sub, golden_sub, bench_sub, names = _split_aclnn_outputs(
        outputs, kwargs, bench_outputs
    )
    params = _resolve_cross_check_params("float32")
    results = []
    for idx in range(len(golden_sub)):
        name = names[idx] if idx < len(names) else f"output_{idx}"
        npu_out = npu_sub[idx]
        golden_out = golden_sub[idx]
        bench_out = bench_sub[idx] if idx < len(bench_sub) else None
        if npu_out is None and golden_out is None:
            results.append({
                "pass": True,
                "precision": "N/A",
                "error_info": f"{name} is None",
            })
        else:
            results.append(
                _ttk_cross_check_single(npu_out, golden_out, bench_out, idx, "float32", params)
            )

    result_consistency = compare_module._batch_consistency_check(
        compare_module._to_torch(outputs[0]).to(torch.float32), kwargs
    )
    if result_consistency is not None:
        results.append(result_consistency)
    return results


def _split_e2e_outputs(outputs, kwargs, bench_outputs):
    GOLDEN_OUTPUT_COUNT = 2
    golden_outputs = [compare_module._to_torch(g) for g in outputs[-GOLDEN_OUTPUT_COUNT:]]
    npu_outputs = [compare_module._to_torch(o) for o in outputs[:-GOLDEN_OUTPUT_COUNT]]
    cmp_kv_mask = kwargs.get("cmp_kv_mask", None)
    update_kv = kwargs.get("update_kv", None)
    update_score = kwargs.get("update_score", None)

    npu_sub = []
    golden_sub = []
    bench_sub = []

    if len(npu_outputs) == 1:
        npu_sub.append(npu_outputs[0][cmp_kv_mask].to(torch.float32))
        golden_sub.append(golden_outputs[0][cmp_kv_mask].to(torch.float32))
        bench = bench_outputs[0] if bench_outputs else None
        bench_sub.append(bench[cmp_kv_mask].to(torch.float32) if bench is not None else None)
        for i in range(1, len(golden_outputs)):
            npu_sub.append(None)
            golden_sub.append(None)
            bench_sub.append(None)
        return npu_sub, golden_sub, bench_sub, _OUTPUT_NAMES

    npu_cmp_kv = npu_outputs[0]
    npu_state_cache = npu_outputs[1]
    cpu_cmp_kv = golden_outputs[0]
    cpu_state_cache = golden_outputs[1]

    npu_sub.append(npu_cmp_kv[cmp_kv_mask])
    golden_sub.append(cpu_cmp_kv[cmp_kv_mask])
    bench_sub.append(bench_outputs[0][cmp_kv_mask] if bench_outputs else None)

    if update_kv is not None and update_score is not None:
        npu_sub.append(npu_state_cache[:, :, : npu_state_cache.shape[2] // 2][update_kv])
        npu_sub.append(npu_state_cache[:, :, npu_state_cache.shape[2] // 2 :][update_score])
        npu_sub.append(npu_state_cache[:, :, : npu_state_cache.shape[2] // 2][~update_kv])
        npu_sub.append(npu_state_cache[:, :, npu_state_cache.shape[2] // 2 :][~update_score])

        golden_sub.append(cpu_state_cache[:, :, : cpu_state_cache.shape[2] // 2][update_kv])
        golden_sub.append(cpu_state_cache[:, :, cpu_state_cache.shape[2] // 2 :][update_score])
        golden_sub.append(cpu_state_cache[:, :, : cpu_state_cache.shape[2] // 2][~update_kv])
        golden_sub.append(cpu_state_cache[:, :, cpu_state_cache.shape[2] // 2 :][~update_score])

        if bench_outputs and len(bench_outputs) > 1:
            bench_state = compare_module._to_torch(bench_outputs[1]).to(torch.float32)
            bench_sub.append(bench_state[:, :, : bench_state.shape[2] // 2][update_kv])
            bench_sub.append(bench_state[:, :, bench_state.shape[2] // 2 :][update_score])
            bench_sub.append(bench_state[:, :, : bench_state.shape[2] // 2][~update_kv])
            bench_sub.append(bench_state[:, :, bench_state.shape[2] // 2 :][~update_score])
        else:
            bench_sub.extend([None, None, None, None])
    else:
        npu_sub.append(npu_state_cache[:, :, : npu_state_cache.shape[2] // 2])
        npu_sub.append(npu_state_cache[:, :, npu_state_cache.shape[2] // 2 :])
        npu_sub.append(None)
        npu_sub.append(None)

        golden_sub.append(cpu_state_cache[:, :, : cpu_state_cache.shape[2] // 2])
        golden_sub.append(cpu_state_cache[:, :, cpu_state_cache.shape[2] // 2 :])
        golden_sub.append(None)
        golden_sub.append(None)

        if bench_outputs and len(bench_outputs) > 1:
            bench_state = compare_module._to_torch(bench_outputs[1]).to(torch.float32)
            bench_sub.append(bench_state[:, :, : bench_state.shape[2] // 2])
            bench_sub.append(bench_state[:, :, bench_state.shape[2] // 2 :])
            bench_sub.append(None)
            bench_sub.append(None)
        else:
            bench_sub.extend([None, None, None, None])

    return npu_sub, golden_sub, bench_sub, _OUTPUT_NAMES


def _run_ttk_cross_check_e2e(outputs, kwargs, bench_outputs):
    npu_sub, golden_sub, bench_sub, names = _split_e2e_outputs(
        outputs, kwargs, bench_outputs
    )
    params = _resolve_cross_check_params("float32")
    results = []
    for idx in range(len(golden_sub)):
        name = names[idx] if idx < len(names) else f"output_{idx}"
        npu_out = npu_sub[idx]
        golden_out = golden_sub[idx]
        bench_out = bench_sub[idx] if idx < len(bench_sub) else None
        if npu_out is None and golden_out is None:
            results.append({
                "pass": True,
                "precision": "N/A",
                "error_info": f"{name} is None",
            })
        else:
            results.append(
                _ttk_cross_check_single(npu_out, golden_out, bench_out, idx, "float32", params)
            )

    result_consistency = compare_module._batch_consistency_check(
        compare_module._to_torch(outputs[0]), kwargs
    )
    if result_consistency is not None:
        results.append(result_consistency)
    return results


class CompressorSpec:
    golden = golden_module.cpu_compressor
    customize_inputs = inputs_module.generate_compressor_inputs
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L0",
                     "mare_ratio": 10.0, "mere_ratio": 2.0, "rmse_ratio": 2.0,
                     "small_value": 2**-10, "small_value_atol": 2**-16},
        "bfloat16": {"standard": "cross_check", "level": "L0",
                      "mare_ratio": 10.0, "mere_ratio": 2.0, "rmse_ratio": 2.0,
                      "small_value": 2**-10, "small_value_atol": 2**-16},
    }

    def compare(*outputs, compare_context=None, **kwargs):
        compare_method = _get_compare_method()
        golden_module.rebuild_golden_context_from_compare_context(
            compare_context, api_kind="e2e"
        )
        ctx = golden_module.get_golden_context()
        kwargs["cmp_kv_mask"] = ctx.get("cmp_kv_mask")
        kwargs["update_kv"] = ctx.get("update_kv")
        kwargs["update_score"] = ctx.get("update_score")
        kwargs["start_pos_list"] = ctx.get("start_pos_list")
        kwargs["seqused_list"] = ctx.get("seqused_list")
        kwargs["cu_seqlens_list"] = ctx.get("cu_seqlens_list")
        kwargs["cmp_ratio"] = ctx.get("cmp_ratio")
        kwargs["is_th"] = ctx.get("is_th")
        bench_outputs = ctx.get("bench_outputs")
        if (
            compare_method == "cross_check"
            and bench_outputs is not None
            and _TTK_CROSS_CHECK_AVAILABLE
        ):
            return _run_ttk_cross_check_e2e(outputs, kwargs, bench_outputs)
        return compare_module.compare(*outputs, **kwargs)


class AclnnCompressorSpec:
    golden = golden_module.aclnn_compressor_golden
    customize_inputs = inputs_module.aclnn_compressor_input
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1",
                     "mare_ratio": 10.0, "mere_ratio": 2.0, "rmse_ratio": 2.0,
                     "small_value": 2**-10, "small_value_atol": 2**-16},
        "bfloat16": {"standard": "cross_check", "level": "L1",
                      "mare_ratio": 10.0, "mere_ratio": 2.0, "rmse_ratio": 2.0,
                      "small_value": 2**-10, "small_value_atol": 2**-16},
    }

    def compare(*outputs, compare_context=None, **kwargs):
        compare_method = _get_compare_method()
        golden_module.rebuild_golden_context_from_compare_context(
            compare_context, api_kind="aclnn"
        )
        ctx = golden_module.get_golden_context()
        kwargs["cmp_kv_mask"] = ctx.get("cmp_kv_mask")
        kwargs["update_kv"] = ctx.get("update_kv")
        kwargs["update_score"] = ctx.get("update_score")
        kwargs["start_pos_list"] = ctx.get("start_pos_list")
        kwargs["seqused_list"] = ctx.get("seqused_list")
        kwargs["cu_seqlens_list"] = ctx.get("cu_seqlens_list")
        kwargs["cmp_ratio"] = ctx.get("cmp_ratio")
        kwargs["is_th"] = ctx.get("is_th")
        kwargs["gradEnabled"] = ctx.get("gradEnabled")
        kwargs["mid_result_mask"] = ctx.get("mid_result_mask")
        bench_outputs = ctx.get("bench_outputs")
        if (
            compare_method == "cross_check"
            and bench_outputs is not None
            and _TTK_CROSS_CHECK_AVAILABLE
        ):
            return _run_ttk_cross_check_aclnn(outputs, kwargs, bench_outputs)
        return compare_module.compare_aclnn(*outputs, **kwargs)


__spec__ = {
    "cann_ops_transformer.ops.compressor": "CompressorSpec",
    "aclnnCompressor": "AclnnCompressorSpec",
}
