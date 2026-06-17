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
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


class StemOamPrepVarlenQOpBuilder(OpBuilder):
    def __init__(self):
        super(StemOamPrepVarlenQOpBuilder, self).__init__("npu_stem_oam_prep_varlen_q")

    def sources(self):
        return ['ops/csrc/stem_oam_prep_varlen_q.cpp']

    def schema(self) -> str:
        return "npu_stem_oam_prep_varlen_q(Tensor q, Tensor qSeqLens, Tensor cuSeqLensQ, *, " \
               "Tensor? qScale=None, Tensor? kScale=None, " \
               "int stemBlockSize=128, int stemStride=16) -> Tensor"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_stem_oam_prep_varlen_q_meta(q, qSeqLens, cuSeqLensQ, *,
                                             qScale=None, kScale=None,
                                             stemBlockSize=128, stemStride=16):
            S = stemStride
            qflat_dim = S * 128
            batch = qSeqLens.size(0)
            H_q = q.size(1)
            maxQLen = qSeqLens.max().item()
            B = stemBlockSize
            maxQPadded = ((maxQLen + B - 1) // B) * B
            maxQb = maxQPadded // B
            return torch.empty(batch, H_q, maxQb, qflat_dim, dtype=torch.bfloat16, device='meta')


npu_stem_oam_prep_varlen_q_op_builder = StemOamPrepVarlenQOpBuilder()
op_module = npu_stem_oam_prep_varlen_q_op_builder.load()


@impl(AS_LIBRARY, npu_stem_oam_prep_varlen_q_op_builder.name, "PrivateUse1")
def npu_stem_oam_prep_varlen_q(q, qSeqLens, cuSeqLensQ, *,
                               qScale=None, kScale=None,
                               stemBlockSize=128, stemStride=16):
    return op_module.npu_stem_oam_prep_varlen_q(q, qSeqLens, cuSeqLensQ,
                                                qScale, kScale,
                                                stemBlockSize, stemStride)
