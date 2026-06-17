# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
import torch
import torch_npu
from cann_ops_transformer.op_builder.builder import OpBuilder


class EngramEmbeddingUtilsOpBuilder(OpBuilder):
    def __init__(self):
        super(EngramEmbeddingUtilsOpBuilder, self).__init__("engram_embedding_utils")

    def sources(self):
        return ['ops/csrc/engram_embedding_utils.cpp']

    def schema(self):
        return None

    def register_meta(self):
        pass

    def extra_ldflags(self):
        flags = super().extra_ldflags()
        flags.append('-L' + os.path.join(self._cann_path, 'lib64'))
        flags.append('-lhcomm')
        flags.append('-lascendcl')
        return flags


engram_embedding_utils_op_builder = EngramEmbeddingUtilsOpBuilder()
op_module = engram_embedding_utils_op_builder.load()

EngramEmbeddingUtils = op_module.EngramEmbeddingUtils