#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TestSpec adapter for Compressor assets.

Cross-check (three-way comparison) functions have been moved to
impl/compare.py. spec.py delegates to compare_module for both
two-way and three-way comparison.
"""

import importlib.util
from pathlib import Path


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

    aclnn_params = [
        ParamInfo(name="x", type="Tensor"),
        ParamInfo(name="wkv", type="Tensor"),
        ParamInfo(name="wgate", type="Tensor"),
        ParamInfo(name="stateCacheRef", type="Tensor"),
        ParamInfo(name="ape", type="Tensor"),
        ParamInfo(name="stateBlockTable", type="Tensor"),
        ParamInfo(name="cuSeqlens", type="Tensor"),
        ParamInfo(name="seqused", type="Tensor"),
        ParamInfo(name="startPos", type="Tensor"),
        ParamInfo(name="cmpRatio", type="int"),
        ParamInfo(name="coff", type="int"),
        ParamInfo(name="cacheMode", type="int"),
        ParamInfo(name="stateCacheStrideDim0", type="int"),
        ParamInfo(name="gradEnabled", type="int"),
        ParamInfo(name="cmpKv", type="Tensor"),
        ParamInfo(name="softmaxScoreOut", type="Tensor"),
        ParamInfo(name="kvOut", type="Tensor"),
    ]
    register_api_params(
        "aclnnCompressor", aclnn_params, source="manual_compress"
    )


_register_compressor_api_params()


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
        compare_method = compare_module._get_compare_method()
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
        kwargs["data_type"] = ctx.get("data_type")
        bench_outputs = ctx.get("bench_outputs")
        if (
            compare_method == "cross_check"
            and bench_outputs is not None
            and compare_module.is_cross_check_available()
        ):
            return compare_module._run_ttk_cross_check_e2e(
                outputs, kwargs, bench_outputs, CompressorSpec.tolerance
            )
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
        compare_method = compare_module._get_compare_method()
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
        kwargs["data_type"] = ctx.get("data_type")
        bench_outputs = ctx.get("bench_outputs")
        if (
            compare_method == "cross_check"
            and bench_outputs is not None
            and compare_module.is_cross_check_available()
        ):
            return compare_module._run_ttk_cross_check_aclnn(
                outputs, kwargs, bench_outputs, AclnnCompressorSpec.tolerance
            )
        return compare_module.compare_aclnn(*outputs, **kwargs)


__spec__ = {
    "cann_ops_transformer.ops.compressor": "CompressorSpec",
    "aclnnCompressor": "AclnnCompressorSpec",
}
