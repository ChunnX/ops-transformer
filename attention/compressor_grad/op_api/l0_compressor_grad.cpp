/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file l0_compressor_grad.cpp
 * \brief L0 operator implementation for CompressorGrad
 */

#include "l0_compressor_grad.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(CompressorGrad);

const bool CompressorGrad(
    const aclTensor *x, const aclTensor *wkv, const aclTensor *wgate,
    const aclTensor *dCmpKv, const aclTensor *softmaxScore, const aclTensor *kv,
    const aclTensor *cuSeqLensOptional, const aclTensor *sequsedOptional,
    const aclTensor *startPosOptional, const char *layoutX, const char *layoutDCmpKv,
    const char *layoutSoftmaxScore, const char *layoutKv,
    const char *layoutDX, int64_t coff, int64_t cmpRatio,
    const aclTensor *dX, const aclTensor *dWkv, const aclTensor *dWgate,
    const aclTensor *dApe, aclOpExecutor *executor)
{
    L0_DFX(CompressorGrad,
           x, wkv, wgate, dCmpKv, softmaxScore, kv,
           cuSeqLensOptional, sequsedOptional, startPosOptional,
           layoutX, layoutSoftmaxScore, layoutKv, layoutDX,
           coff, cmpRatio, dX, dWkv, dWgate, dApe);

    ADD_TO_LAUNCHER_LIST_AICORE(
        CompressorGrad,
        OP_INPUT(x, wkv, wgate, dCmpKv, softmaxScore, kv,
                 cuSeqLensOptional, sequsedOptional, startPosOptional),
        OP_OUTPUT(dX, dWkv, dWgate, dApe),
        OP_ATTR(layoutX, layoutDCmpKv, layoutSoftmaxScore, layoutKv, layoutDX,
                coff, cmpRatio));

    OP_LOGI("CompressorGrad", "l0 registered to launcher list, coff=%ld, cmpRatio=%ld.", coff, cmpRatio);

    return true;
}

}  // namespace l0op
