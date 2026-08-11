# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FlashAttnGrad kernel.

Flash Attention backward gradient computation.
Skeleton stage: entry point with empty implementation.
"""

from dataclasses import dataclass

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField


# ================================================================
#  Tiling data
# ================================================================
@dataclass
class FlashAttnGradTilingData:
    b: int
    s1: int
    s2: int
    n1: int
    d: int
    dv: int
    scaleValue: float


# ================================================================
#  TilingKey
# ================================================================
class FlashAttnGradTilingKey:
    layout = TilingKeyField(bits=1, values=[0, 1])


# ================================================================
#  Kernel entry point
# ================================================================
@pl.jit(
    auto_mutex=True,
    tiling_key=FlashAttnGradTilingKey,
    datatype={
        "q": "input_dtype",
    },
)
def flash_attn_grad(
    q: pl.Ptr[pl.DT_UINT8],
    k: pl.Ptr[pl.DT_UINT8],
    v: pl.Ptr[pl.DT_UINT8],
    dout: pl.Ptr[pl.DT_UINT8],
    attn_out: pl.Ptr[pl.DT_UINT8],
    softmax_lse: pl.Ptr[pl.DT_UINT8],
    cu_seqlens_q: pl.Ptr[pl.DT_UINT8],
    cu_seqlens_kv: pl.Ptr[pl.DT_UINT8],
    seqused_q: pl.Ptr[pl.DT_UINT8],
    seqused_kv: pl.Ptr[pl.DT_UINT8],
    sinks: pl.Ptr[pl.DT_UINT8],
    attn_mask: pl.Ptr[pl.DT_UINT8],
    metadata: pl.Ptr[pl.DT_UINT8],
    dq: pl.Ptr[pl.DT_UINT8],
    dk: pl.Ptr[pl.DT_UINT8],
    dv: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: FlashAttnGradTilingData,
):
    # TODO: read FAG metadata from GM (head[4] offset + 116 int32)
    # TODO: implement flash attention backward gradient computation
    return
