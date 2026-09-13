# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Load the installed QBSA frontend for pytest runtime checks."""

import torch

_LOAD_ERRORS = []
try:
    import cann_ops_transformer
except (ImportError, RuntimeError) as error:
    cann_ops_transformer = None
    _LOAD_ERRORS.append(str(error))


def has_quant_block_sparse_attn_op():
    return cann_ops_transformer is not None and callable(
        getattr(cann_ops_transformer, "quant_block_sparse_attn", None)
    )


def has_quant_block_sparse_attn_metadata_op():
    return cann_ops_transformer is not None and callable(
        getattr(cann_ops_transformer, "quant_block_sparse_attn_metadata", None)
    )


def load_errors():
    return list(_LOAD_ERRORS)
