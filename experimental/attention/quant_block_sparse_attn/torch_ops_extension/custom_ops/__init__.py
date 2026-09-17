# copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu

# 导入so 和 python
from . import custom_ops_lib  # noqa: F401 - registers the schemas and eager/Meta implementations
from .converter import npu_quant_block_sparse_attn as _attn_converter  # noqa: F401
from .converter import npu_quant_block_sparse_attn_metadata as _metadata_converter  # noqa: F401

# Export only this extension's operators. Other packages may share torch.ops.custom.
npu_quant_block_sparse_attn = torch.ops.custom.npu_quant_block_sparse_attn
npu_quant_block_sparse_attn_metadata = (
    torch.ops.custom.npu_quant_block_sparse_attn_metadata
)
torch_npu.npu_quant_block_sparse_attn = npu_quant_block_sparse_attn
torch_npu.npu_quant_block_sparse_attn_metadata = npu_quant_block_sparse_attn_metadata

__all__ = ["npu_quant_block_sparse_attn", "npu_quant_block_sparse_attn_metadata"]
