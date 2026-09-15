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


@register_fx_node_ge_converter(torch.ops.custom.stem_indexer_metadata.default)
def convert_stem_indexer_metadata(
    seq_used_q: Tensor,
    seq_used_kv: Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    *,
    causal: bool = True,
    stem_block_size: int = 128,
    dim_qk_flat: int = 2048,
    window_size: int = 4,
    meta_outputs: List[TensorSpec] = None,
):
    return torchair.ge.custom_op(
        "StemIndexerMetadata",
        inputs={
            "q_seq_lens": seq_used_q,
            "kv_seq_lens": seq_used_kv,
        },
        attrs={
            "q_heads": attr.Int(num_heads_q),
            "kv_heads": attr.Int(num_heads_kv),
            "causal": attr.Bool(causal),
            "stem_block_size": attr.Int(stem_block_size),
            "dim_qkflat": attr.Int(dim_qk_flat),
            "window_size": attr.Int(window_size),
        },
        outputs=["metadata"],
    )
