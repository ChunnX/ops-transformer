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

"""TTK graph adapter 占位（结构占位，供未来填充）。

对应 flash_attn 的 tests/assets/impl/graph.py（FlashAttnAclGraph：以
torch.nn.Module 包裹 metadata + kernel 全链路灯，供 ttk e2e 静态图路径调用）。

MLA 侧实现参考：tests/pytests/core/backends/npu.py 的
`_FlashMlaWithKvcacheGraphNetwork`（torch.ops.cann_ops_transformer.flash_mla_metadata
+ flash_mla_with_kvcache 全链路灯，含静态图 export 分支）。本文件仅提供同名类占位，
forward 抛 NotImplementedError；填充时请以 npu.py 的网络为蓝本并适配
TTK graph 调用约定（FA 侧签名见 spec.py 的 torch_graph 用法）。
"""


class FlashMlaWithKvcacheAclGraph:
    """TTK graph adapter 占位：MLA 的 aclnn 静态图适配待填充。

    规划接口（对齐 flash_attn 的 FlashAttnAclGraph）：
        __init__(self, *, batch_size=None, softmax_scale=1.0, mask_mode=0,
                 win_left=-1, win_right=-1, max_seqlen_q=None, max_seqlen_kv=None,
                 layout_q="TND", layout_kv="PA_NZ", layout_out="NTD",
                 return_softmax_lse=False)
        forward(self, q, k_cache, *, block_table=None, cu_seqlens_q=None,
                cache_seqlens=None, seqused_q=None, attn_mask=None, metadata=None)
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "flash_mla_with_kvcache 的 TTK graph adapter 未实现（结构占位）。"
            "填充时参照 pytests/core/backends/npu.py 的 _FlashMlaWithKvcacheGraphNetwork，"
            "及 flash_attn/tests/assets/impl/graph.py 的 FlashAttnAclGraph 调用约定。"
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "flash_mla_with_kvcache 的 TTK graph adapter 未实现（结构占位）。"
        )
