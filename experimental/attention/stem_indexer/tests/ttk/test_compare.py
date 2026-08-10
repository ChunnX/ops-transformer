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
from pathlib import Path
from types import SimpleNamespace

import torch


COMPARE_PATH = Path(__file__).parent / "assets" / "impl" / "compare.py"
SPEC = importlib.util.spec_from_file_location("stem_indexer_ttk_compare_test", COMPARE_PATH)
COMPARE_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE_MODULE)


def build_compare_context():
    topk_count = 200
    kv_block_count = topk_count + 2
    qflat = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    kflat = torch.zeros((1, 1, kv_block_count, 1), dtype=torch.float32)
    vbias = torch.ones((1, 1, kv_block_count), dtype=torch.float32)
    vbias[..., topk_count:] = 0.998
    q_seq_lens = torch.tensor([128], dtype=torch.int32)
    kv_seq_lens = torch.tensor([kv_block_count * 128], dtype=torch.int32)
    num_prompt_tokens = kv_seq_lens.clone()
    metadata = torch.zeros((1,), dtype=torch.int32)
    attributes = {
        "causal": False,
        "stem_block_size": 128,
        "stem_stride": 16,
        "alpha": 1.0,
        "initial_blocks": 0,
        "window_size": 0,
        "q_heads": 1,
        "kv_heads": 1,
        "batch_size": 1,
        "q_seq_lens_list": [128],
        "kv_seq_lens_list": [kv_block_count * 128],
        "num_prompt_tokens_list": [kv_block_count * 128],
    }
    return SimpleNamespace(
        testcase_name="boundary_ratio_limit",
        attributes=attributes,
        input_tensors=(
            qflat,
            kflat,
            vbias,
            q_seq_lens,
            kv_seq_lens,
            num_prompt_tokens,
            metadata,
        ),
    )


def test_compare_allows_one_exceeded_index_out_of_two_hundred():
    context = build_compare_context()
    expected_indices = torch.arange(200, dtype=torch.int32).reshape(1, 1, 1, 200)
    actual_indices = expected_indices.clone()
    actual_indices[..., -1] = 200
    seq_len = torch.tensor([[[200]]], dtype=torch.int32)

    result = COMPARE_MODULE.compare(
        actual_indices,
        seq_len,
        expected_indices,
        seq_len,
        compare_context=context,
    )

    assert result["pass"] is True
