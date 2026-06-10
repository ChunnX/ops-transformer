# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
from npu_ops_transformer.op_builder.builder import OpBuilder


class EngramEmbedingUitlsOpBuilder(OpBuilder):
    def __init__(self):
        super(EngramEmbedingUitlsOpBuilder, self).__init__("engram_embedding_utils")

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


engram_embedding_utils_op_builder = EngramEmbedingUitlsOpBuilder()
op_module = engram_embedding_utils_op_builder.load()

EngramEmbedingUitls = op_module.EngramEmbeddingUtils