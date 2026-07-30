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

import os

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

import result_compare_method  # noqa: E402
import stem_indexer_golden  # noqa: E402
import custom_ops  # noqa: E402, F401
from test_stem_indexer_paramset import ENABLED_PARAMS  # noqa: E402

try:
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    _HAS_TORCHAIR = True
except ImportError:
    _HAS_TORCHAIR = False

# 通过环境变量 STEM_INDEXER_MODE 切换执行模式：eager（默认）或 graph
# 例: STEM_INDEXER_MODE=graph python -m pytest test_stem_indexer_single.py -m graph
EXEC_MODE = os.environ.get("STEM_INDEXER_MODE", "eager").strip().lower()
_IS_GRAPH_MODE = EXEC_MODE == "graph"


# 支持通过环境变量 STEM_INDEXER_CASE_ID 指定只跑特定 case_id（逗号分隔多个）
# 例: STEM_INDEXER_CASE_ID=SI_WB_001,SI_WB_002 python -m pytest test_stem_indexer_single.py
_FILTER_IDS_RAW = os.environ.get("STEM_INDEXER_CASE_ID", "").strip()
_FILTER_IDS = {item.strip() for item in _FILTER_IDS_RAW.split(",") if item.strip()}
TEST_CASES = [
    case
    for case in ENABLED_PARAMS
    if not _FILTER_IDS or case.get("case_id") in _FILTER_IDS
]
if _FILTER_IDS:
    print(
        f"Filter by STEM_INDEXER_CASE_ID={_FILTER_IDS_RAW}: "
        f"{len(TEST_CASES)}/{len(ENABLED_PARAMS)} matched."
    )


class StemIndexerNetwork(torch.nn.Module):
    def __init__(self):
        super(StemIndexerNetwork, self).__init__()

    def forward(
        self,
        qflat,
        kflat,
        vbias,
        q_seq_lens,
        kv_seq_lens,
        num_prompt_tokens,
        metadata,
        causal,
        stem_block_size,
        stem_stride,
        alpha,
        initial_blocks,
        window_size,
        k_block_num_rate_medium,
        k_block_num_bias_medium,
        k_block_num_rate_large,
        k_block_num_bias_large,
        topk_score_precision,
    ):
        sparse_indices, sparse_seq_len = torch.ops.custom.npu_stem_indexer(
            qflat,
            kflat,
            vbias,
            q_seq_lens,
            kv_seq_lens,
            num_prompt_tokens=num_prompt_tokens,
            metadata=metadata,
            causal=causal,
            stem_block_size=stem_block_size,
            stem_stride=stem_stride,
            alpha=alpha,
            initial_blocks=initial_blocks,
            window_size=window_size,
            k_block_num_rate_medium=k_block_num_rate_medium,
            k_block_num_bias_medium=k_block_num_bias_medium,
            k_block_num_rate_large=k_block_num_rate_large,
            k_block_num_bias_large=k_block_num_bias_large,
            topk_score_precision=topk_score_precision,
        )
        return sparse_indices, sparse_seq_len


def _build_graph_network():
    if not _HAS_TORCHAIR:
        raise ImportError("graph 模式需要 torchair，请先安装")

    torch._dynamo.reset()
    net = StemIndexerNetwork().npu()
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
    config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = "./"
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    config.experimental_config.topology_sorting_strategy = "StableRDFS"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    compiled_net = torch.compile(
        net, fullgraph=True, backend=npu_backend, dynamic=False
    )
    return compiled_net


def call_stem_indexer_graph(case, inputs):
    npu_inputs = move_inputs_to_npu(inputs)
    attrs = stem_indexer_golden.get_call_attrs(case)
    metadata = build_metadata(case, npu_inputs)
    compiled_net = _build_graph_network()

    npu_result = compiled_net(
        npu_inputs["qflat"],
        npu_inputs["kflat"],
        npu_inputs["vbias"],
        npu_inputs["q_seq_lens"],
        npu_inputs["kv_seq_lens"],
        npu_inputs["num_prompt_tokens"],
        metadata,
        attrs["causal"],
        attrs["stem_block_size"],
        attrs["stem_stride"],
        attrs["alpha"],
        attrs["initial_blocks"],
        attrs["window_size"],
        attrs["k_block_num_rate_medium"],
        attrs["k_block_num_bias_medium"],
        attrs["k_block_num_rate_large"],
        attrs["k_block_num_bias_large"],
        attrs["topk_score_precision"],
    )
    return npu_result


def case_id(case):
    return f"{case['case_id']}:{case['testcase_name']}"


def move_inputs_to_npu(inputs):
    return {name: tensor.npu() for name, tensor in inputs.items()}


def build_metadata(case, npu_inputs):
    if case["expected_result"] == "FAIL":
        return npu_inputs["metadata"]
    metadata_attrs = stem_indexer_golden.get_metadata_attrs(case)
    return torch.ops.custom.npu_stem_indexer_metadata(
        npu_inputs["q_seq_lens"],
        npu_inputs["kv_seq_lens"],
        case["q_heads"],
        case["kv_heads"],
        **metadata_attrs,
    )


def call_stem_indexer(case, inputs):
    npu_inputs = move_inputs_to_npu(inputs)
    attrs = stem_indexer_golden.get_call_attrs(case)
    metadata = build_metadata(case, npu_inputs)
    return torch.ops.custom.npu_stem_indexer(
        npu_inputs["qflat"],
        npu_inputs["kflat"],
        npu_inputs["vbias"],
        npu_inputs["q_seq_lens"],
        npu_inputs["kv_seq_lens"],
        num_prompt_tokens=npu_inputs["num_prompt_tokens"],
        metadata=metadata,
        **attrs,
    )


def run_stem_indexer_case(case):
    torch_npu.npu.set_device(0)
    inputs = stem_indexer_golden.build_case_inputs(case)

    if case["expected_result"] == "FAIL":
        if _IS_GRAPH_MODE:
            pytest.skip("Graph mode does not support FAIL test cases.")
        if case["testcase_name"] == "invalid_sparse_indices_shape":
            pytest.skip(
                "Torch custom op API does not expose output tensor shape injection."
            )
        with pytest.raises(Exception):
            call_stem_indexer(case, inputs)
        return

    expected_indices, expected_seq_len = stem_indexer_golden.stem_indexer_golden(
        case, inputs
    )

    if _IS_GRAPH_MODE:
        npu_result = call_stem_indexer_graph(case, inputs)
    else:
        npu_result = call_stem_indexer(case, inputs)

    torch_npu.npu.synchronize()
    result_compare_method.assert_stem_indexer_result(
        expected_indices, expected_seq_len, npu_result, case, inputs
    )


@pytest.mark.ci
@pytest.mark.graph
@pytest.mark.parametrize("case", TEST_CASES, ids=case_id)
def test_stem_indexer(case):
    run_stem_indexer_case(case)
