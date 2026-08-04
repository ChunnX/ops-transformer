# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""mhc_post_backward 兼容转发模块。

mhc_post_backward 的 torch_extension 代码已独立到
mhc/mhc_post_backward/torch_extension/ 目录，本模块保留以兼容
旧导入路径 `cann_ops_transformer.ops.mhc_post_backward`（如 mhc_post 前向的 backward）。
"""

import importlib.util
import os

_MHC_POST_BACKWARD_TE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir,
    os.pardir,
    os.pardir,
    "mhc",
    "mhc_post_backward",
    "torch_extension",
)
_MHC_POST_BACKWARD_FILE = os.path.join(
    _MHC_POST_BACKWARD_TE_DIR, "mhc_post_backward.py"
)

_spec = importlib.util.spec_from_file_location(
    "mhc_post_backward_impl", _MHC_POST_BACKWARD_FILE
)
_mhc_post_backward_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mhc_post_backward_impl)

mhc_post_backward = _mhc_post_backward_impl.mhc_post_backward
_mhc_post_backward_dispatch = _mhc_post_backward_impl._mhc_post_backward_dispatch
mhc_post_backward_op_builder = _mhc_post_backward_impl.mhc_post_backward_op_builder
