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

"""TestSpec adapter for QuantCompressor assets."""

import importlib.util
from pathlib import Path


ASSET_IMPL_DIR = Path(__file__).with_name("impl")


def load_impl_module(stem):
    path = ASSET_IMPL_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(
        f"quant_compressor_assets_impl_{stem}_{abs(hash(path))}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


golden_module = load_impl_module("golden")
inputs_module = load_impl_module("inputs")
compare_module = load_impl_module("compare")


def _register_quant_compressor_api_params():
    try:
        from ttk.utilities.simple_param_extractor import (
            ParamInfo,
            register_api_params,
        )
    except Exception:
        return

    e2e_params = [
        ParamInfo(name="x", type="Tensor"),
        ParamInfo(name="wkv", type="Tensor"),
        ParamInfo(name="wgate", type="Tensor"),
        ParamInfo(name="state_cache", type="Tensor"),
        ParamInfo(name="ape", type="Tensor"),
        ParamInfo(name="quant_mode", type="int"),
        ParamInfo(name="cmp_ratio", type="int"),
        ParamInfo(
            name="x_descale",
            type="Tensor",
            default=None,
            is_optional=True,
            is_keyword_only=True,
        ),
        ParamInfo(
            name="wkv_descale",
            type="Tensor",
            default=None,
            is_optional=True,
            is_keyword_only=True,
        ),
        ParamInfo(
            name="wgate_descale",
            type="Tensor",
            default=None,
            is_optional=True,
            is_keyword_only=True,
        ),
        ParamInfo(
            name="state_block_table",
            type="Tensor",
            default=None,
            is_optional=True,
            is_keyword_only=True,
        ),
        ParamInfo(
            name="cu_seqlens",
            type="Tensor",
            default=None,
            is_optional=True,
            is_keyword_only=True,
        ),
        ParamInfo(
            name="seqused",
            type="Tensor",
            default=None,
            is_optional=True,
            is_keyword_only=True,
        ),
        ParamInfo(
            name="start_pos",
            type="Tensor",
            default=None,
            is_optional=True,
            is_keyword_only=True,
        ),
        ParamInfo(
            name="coff", type="int", default=1, is_optional=True, is_keyword_only=True
        ),
        ParamInfo(
            name="cache_mode",
            type="int",
            default=1,
            is_optional=True,
            is_keyword_only=True,
        ),
    ]
    register_api_params(
        "cann_ops_transformer.ops.quant_compressor",
        e2e_params,
        source="manual_quant_compress",
    )

    kernel_params = [
        ParamInfo(name="x", type="Tensor"),
        ParamInfo(name="wkv", type="Tensor"),
        ParamInfo(name="wgate", type="Tensor"),
        ParamInfo(name="state_cache", type="Tensor"),
        ParamInfo(name="ape", type="Tensor"),
        ParamInfo(name="x_descale", type="Tensor", default=None, is_optional=True),
        ParamInfo(name="wkv_descale", type="Tensor", default=None, is_optional=True),
        ParamInfo(name="wgate_descale", type="Tensor", default=None, is_optional=True),
        ParamInfo(
            name="state_block_table", type="Tensor", default=None, is_optional=True
        ),
        ParamInfo(name="cu_seqlens", type="Tensor", default=None, is_optional=True),
        ParamInfo(name="seqused", type="Tensor", default=None, is_optional=True),
        ParamInfo(name="start_pos", type="Tensor", default=None, is_optional=True),
    ]
    register_api_params(
        "quant_compressor", kernel_params, source="manual_quant_compress"
    )

    aclnn_params = [
        ParamInfo(name="x", type="Tensor"),
        ParamInfo(name="wkv", type="Tensor"),
        ParamInfo(name="wgate", type="Tensor"),
        ParamInfo(name="stateCacheRef", type="Tensor"),
        ParamInfo(name="ape", type="Tensor"),
        ParamInfo(name="xDescale", type="Tensor"),
        ParamInfo(name="wkvDescale", type="Tensor"),
        ParamInfo(name="wgateDescale", type="Tensor"),
        ParamInfo(name="stateBlockTable", type="Tensor"),
        ParamInfo(name="cuSeqlens", type="Tensor"),
        ParamInfo(name="seqused", type="Tensor"),
        ParamInfo(name="startPos", type="Tensor"),
        ParamInfo(name="quantMode", type="int"),
        ParamInfo(name="cmpRatio", type="int"),
        ParamInfo(name="coff", type="int"),
        ParamInfo(name="cacheMode", type="int"),
        ParamInfo(name="stateCacheStrideDim0", type="int"),
        ParamInfo(name="cmpKv", type="Tensor"),
    ]
    register_api_params(
        "aclnnQuantCompressor", aclnn_params, source="manual_quant_compress"
    )


_register_quant_compressor_api_params()


class QuantCompressorSpec:
    golden = golden_module.cpu_quant_compressor
    customize_inputs = inputs_module.generate_quant_compressor_inputs
    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }

    def compare(*outputs, compare_context=None, **kwargs):
        ctx = golden_module.get_golden_context()
        # Replay (manual_data) 模式下 golden plugin 不重新执行，_GOLDEN_CONTEXT 为
        # 空，compare 拿到的 cmp_kv_mask / update_kv / update_score 均为 None，
        # kv_state_origin / score_state_origin 会退化为 N/A。这里用 TTK 注入的
        # compare_context（携带 testcase.tensors / testcase.attributes）重跑一次
        # CPU golden 重建 context。prepare 模式下 ctx 已由 golden plugin 填好，会
        # 直接走下面的 ctx.get 路径，不会重复计算。
        #
        # 批跑时 _GOLDEN_CONTEXT 是模块级全局变量，前一个 case 的 mask 会残留。
        # 即使 ctx 非空，也必须用当前 case 的 compare_context 重建，否则 mask shape
        # 可能不匹配（不同 case 的 state_cache shape 可能不同）。
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
        return compare_module.compare(*outputs, **kwargs)


class KernelQuantCompressorSpec:
    golden = golden_module.kernel_quant_compressor_golden
    customize_inputs = inputs_module.kernel_quant_compressor_input
    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }

    def compare(*outputs, compare_context=None, **kwargs):
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
        return compare_module.compare(*outputs, **kwargs)


class AclnnQuantCompressorSpec:
    golden = golden_module.aclnn_quant_compressor_golden
    customize_inputs = inputs_module.aclnn_quant_compressor_input
    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }

    def compare(*outputs, compare_context=None, **kwargs):
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
        return compare_module.compare_aclnn(*outputs, **kwargs)


__spec__ = {
    "quant_compressor": "KernelQuantCompressorSpec",
    "cann_ops_transformer.ops.quant_compressor": "QuantCompressorSpec",
    "aclnnQuantCompressor": "AclnnQuantCompressorSpec",
}
