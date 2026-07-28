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
 * \file l0_compressor_grad.h
 * \brief L0 operator declaration for CompressorGrad
 */

#ifndef OP_API_INC_LEVEL0_L0_COMPRESSOR_GRAD_H_
#define OP_API_INC_LEVEL0_L0_COMPRESSOR_GRAD_H_

#include "opdev/op_executor.h"

namespace l0op {

const bool CompressorGrad(
    const aclTensor *x, const aclTensor *wkv, const aclTensor *wgate,
    const aclTensor *dCmpKv, const aclTensor *softmaxScore, const aclTensor *kv,
    const aclTensor *cuSeqLensOptional, const aclTensor *sequsedOptional,
    const aclTensor *startPosOptional, const char *layoutX, const char *layoutDCmpKv,
    const char *layoutSoftmaxScore, const char *layoutKv,
    const char *layoutDX, int64_t coff, int64_t cmpRatio,
    const aclTensor *dX, const aclTensor *dWkv, const aclTensor *dWgate,
    const aclTensor *dApe, aclOpExecutor *executor);

}  // namespace l0op

#endif  // OP_API_INC_LEVEL0_L0_COMPRESSOR_GRAD_H_
