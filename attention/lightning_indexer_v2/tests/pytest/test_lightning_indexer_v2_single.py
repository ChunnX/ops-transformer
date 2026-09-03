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

import importlib
import itertools
import os

import pytest
import torch_npu

import lightning_indexer_v2_golden
import result_compare_method


PARAMSET_MODULE_ALIASES = {
    "default": "test_lightning_indexer_v2_paramset",
    "stc": "test_lightning_indexer_v2_stc",
}
PARAM_NAMES = (
    "batch_size",
    "q_seq",
    "k_seq",
    "q_t_size",
    "k_t_size",
    "q_head_num",
    "k_head_num",
    "head_dim",
    "block_size",
    "block_num",
    "qk_dtype",
    "cu_seqlens_q",
    "cu_seqlens_k",
    "seqused_q",
    "seqused_k",
    "cmp_residual_k",
    "output_idx_offset",
    "layout_q",
    "layout_k",
    "topk",
    "mask_mode",
    "query_datarange",
    "key_datarange",
    "weights_datarange",
    "cmp_ratio",
    "return_value",
    "max_seqlen_q",
)


def _load_enabled_paramsets():
    module_name = os.environ.get(
        "LIV2_PARAMSET_MODULE", "test_lightning_indexer_v2_paramset"
    ).strip()
    module_name = PARAMSET_MODULE_ALIASES.get(module_name, module_name)
    module = importlib.import_module(module_name)
    if hasattr(module, "ENABLED_PARAMSETS"):
        return module_name, module.ENABLED_PARAMSETS
    return module_name, [
        (f"default_{index:03d}", params)
        for index, params in enumerate(module.ENABLED_PARAMS, start=1)
    ]


def _expand_paramsets(module_name, enabled_paramsets):
    requested_names = {
        name.strip()
        for name in os.environ.get("LIV2_CASE_NAMES", "").split(",")
        if name.strip()
    }
    matched_names = set()
    expanded_cases = []

    for paramset_name, params in enabled_paramsets:
        combinations = list(
            itertools.product(*(params[name] for name in PARAM_NAMES))
        )
        for combo_index, combo in enumerate(combinations, start=1):
            case_name = (
                paramset_name
                if len(combinations) == 1
                else f"{paramset_name}_{combo_index:03d}"
            )
            if requested_names and not (
                paramset_name in requested_names or case_name in requested_names
            ):
                continue
            matched_names.update((paramset_name, case_name))
            param_dict = dict(zip(PARAM_NAMES, combo))
            param_dict["case_name"] = case_name
            expanded_cases.append(pytest.param(param_dict, id=case_name))

    unknown_names = sorted(requested_names - matched_names)
    if unknown_names:
        raise ValueError(f"unknown case(s) in {module_name}: {unknown_names}")
    return expanded_cases


PARAMSET_MODULE_NAME, ENABLED_PARAMSETS = _load_enabled_paramsets()
PARAM_COMBINATIONS = _expand_paramsets(PARAMSET_MODULE_NAME, ENABLED_PARAMSETS)


@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", PARAM_COMBINATIONS)
def test_liv2(param_combinations):
    case_name = param_combinations["case_name"]
    test_data = tuple(param_combinations[name] for name in PARAM_NAMES)
    output_idx_offset = param_combinations["output_idx_offset"]
    return_value = param_combinations["return_value"]
    run_mode = os.environ.get("LIV2_RUN_MODE", "eager").strip().lower()

    torch_npu.npu.set_device(0)
    if run_mode == "eager":
        results = lightning_indexer_v2_golden.liv2_output_single(test_data)
    elif run_mode == "graph":
        import lightning_indexer_v2_acl_graph

        results = lightning_indexer_v2_acl_graph.liv2_output_acl_graph(test_data)
    else:
        raise ValueError(f"unsupported run_mode: {run_mode}")

    cpu_result, npu_result, topk_value, cpu_topk_value, npu_topk_value = results
    result, fulfill_percent = result_compare_method.check_result(
        cpu_result,
        npu_result,
        topk_value,
        output_idx_offset,
        test_data,
        cpu_topk_value,
        npu_topk_value,
    )
    print(f"{case_name}: result={result}, fulfill_percent={fulfill_percent}")
    assert result, f"{case_name}: sparse_indices comparison failed ({fulfill_percent}%)"

    if return_value:
        value_result, value_percent = result_compare_method.check_result_return_value(
            cpu_topk_value,
            npu_topk_value,
            test_data,
        )
        print(
            f"{case_name}: return_value_result={value_result}, "
            f"fulfill_percent={value_percent}"
        )
        assert value_result, (
            f"{case_name}: sparse_values comparison failed ({value_percent}%)"
        )
