# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import List

import torch
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair.ge import attr

from . import check_exact_type


@register_fx_node_ge_converter(torch.ops.custom.npu_stem_indexer_metadata.default)
def convert_npu_stem_indexer_metadata(
    q_seq_lens: Tensor,
    kv_seq_lens: Tensor,
    q_heads: int,
    kv_heads: int,
    *,
    causal: bool = True,
    stem_block_size: int = 128,
    dim_qkflat: int = 128,
    window_size: int = 4,
    meta_outputs: List[TensorSpec] = None,
):
    attr_types = (
        ("q_heads", q_heads, int),
        ("kv_heads", kv_heads, int),
        ("causal", causal, bool),
        ("stem_block_size", stem_block_size, int),
        ("dim_qkflat", dim_qkflat, int),
        ("window_size", window_size, int),
    )
    for name, value, expected_type in attr_types:
        check_exact_type(name, value, expected_type)

    return torchair.ge.custom_op(
        "StemIndexerMetadata",
        inputs={
            "q_seq_lens": q_seq_lens,
            "kv_seq_lens": kv_seq_lens,
        },
        attrs={
            "q_heads": attr.Int(q_heads),
            "kv_heads": attr.Int(kv_heads),
            "causal": attr.Bool(causal),
            "stem_block_size": attr.Int(stem_block_size),
            "dim_qkflat": attr.Int(dim_qkflat),
            "window_size": attr.Int(window_size),
        },
        outputs=["metadata"],
    )
