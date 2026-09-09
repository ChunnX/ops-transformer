# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""兼容旧 custom_ops 调用；实际计算统一转发到正式 Torch 扩展。"""

import torch
import torch_npu
from cann_ops_transformer import turbo_quant_sparse_flash_attention

_OP_NAME = "npu_turbo_quant_sparse_flash_attention"
_legacy_library = torch.library.Library("custom", "FRAGMENT")
_schema = str(
    torch.ops.cann_ops_transformer.turbo_quant_sparse_flash_attention.default._schema
)
_legacy_library.define(
    _schema.replace(
        "cann_ops_transformer::turbo_quant_sparse_flash_attention", _OP_NAME, 1
    )
)


def _forward(*args, **kwargs):
    return turbo_quant_sparse_flash_attention(*args, **kwargs)


_legacy_library.impl(_OP_NAME, _forward, "PrivateUse1")
_legacy_library.impl(_OP_NAME, _forward, "Meta")
npu_turbo_quant_sparse_flash_attention = getattr(torch.ops.custom, _OP_NAME)
setattr(torch_npu, _OP_NAME, npu_turbo_quant_sparse_flash_attention)

__all__ = [_OP_NAME]
