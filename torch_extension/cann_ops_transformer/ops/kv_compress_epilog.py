# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
import torch_npu
from torch.library import impl
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY


class KvCompressEpilogOpBuilder(OpBuilder):
    def __init__(self):
        super(KvCompressEpilogOpBuilder, self).__init__("npu_kv_compress_epilog")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/kv_compress_epilog.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "npu_kv_compress_epilog(Tensor cache, Tensor x, Tensor slot_mapping, *, " \
               "int quant_group_size=64, int quant_mode=1, bool round_scale=True, " \
               "float x_scale=1.0) -> Tensor"

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for FakeTensor / graph mode support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_kv_compress_epilog_meta(cache, x, slot_mapping, *, quant_group_size=64,
                                        quant_mode=1, round_scale=True, x_scale=1.0):
            return torch.empty_like(cache)


# Instantiate the builder
npu_kv_compress_epilog_op_builder = KvCompressEpilogOpBuilder()
op_module = npu_kv_compress_epilog_op_builder.load()  # Compiles/loads the .so file


@impl(AS_LIBRARY, npu_kv_compress_epilog_op_builder.name, "PrivateUse1")
def npu_kv_compress_epilog(cache, x, slot_mapping, *, quant_group_size=64,
                           quant_mode=1, round_scale=True, x_scale=1.0):
    return op_module.npu_kv_compress_epilog(cache, x, slot_mapping, quant_group_size,
                                            quant_mode, round_scale, x_scale)