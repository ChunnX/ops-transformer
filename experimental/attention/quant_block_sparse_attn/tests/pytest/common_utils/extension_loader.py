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

"""Import the installed QBSA extension for pytest and TTK."""

import importlib

_LOADED_EXTENSION = None


def ensure_qbsa_extension():
    """Use the normal package import, including its binary and GE converters."""
    global _LOADED_EXTENSION
    if _LOADED_EXTENSION is not None:
        return _LOADED_EXTENSION

    try:
        extension = importlib.import_module("custom_ops")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Cannot import the installed QBSA custom_ops extension. Build and install it with "
            "the Python environment used for this test: "
            "`cd experimental/attention/quant_block_sparse_attn/torch_ops_extension && "
            "bash build_and_install.sh`. "
            f"Original error: {error}"
        ) from error

    import torch

    required_ops = (
        "npu_quant_block_sparse_attn",
        "npu_quant_block_sparse_attn_metadata",
    )
    missing_ops = [name for name in required_ops if not hasattr(torch.ops.custom, name)]
    if missing_ops:
        origin = getattr(extension, "__file__", "<unknown>")
        raise RuntimeError(
            f"custom_ops loaded from {origin} does not register {', '.join(missing_ops)}. "
            "Other experimental operators also use the custom_ops package name. "
            "Install the QBSA extension into this Python environment and restart the process."
        )
    _LOADED_EXTENSION = extension
    return extension
