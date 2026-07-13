/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stem_oam_prep_varlen_q.cpp
 * \brief L0 operator implementation for StemOamPrepVarlenQ
 */

#include "stem_oam_prep_varlen_q.h"

#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(StemOamPrepVarlenQ);

const aclTensor *StemOamPrepVarlenQ(
    const aclTensor *q,
    const aclIntArray *qSeqLens,
    const aclIntArray *cuSeqLensQ,
    const aclTensor *qScale,
    int64_t stemBlockSize,
    int64_t stemStride,
    aclOpExecutor *executor)
{
    L0_DFX(StemOamPrepVarlenQ, q, qSeqLens, cuSeqLensQ, qScale, stemBlockSize, stemStride);

    auto qContiguous = l0op::Contiguous(q, executor);
    CHECK_RET(qContiguous != nullptr, nullptr);

    auto qSeqLensTensor = executor->ConvertToTensor(qSeqLens, DataType::DT_INT64);
    auto mutableQSeqLens = const_cast<aclTensor *>(qSeqLensTensor);
    mutableQSeqLens->SetStorageFormat(Format::FORMAT_ND);
    mutableQSeqLens->SetViewFormat(Format::FORMAT_ND);
    mutableQSeqLens->SetOriginalFormat(Format::FORMAT_ND);

    auto cuSeqLensQTensor = executor->ConvertToTensor(cuSeqLensQ, DataType::DT_INT64);
    auto mutableCuSeqLensQ = const_cast<aclTensor *>(cuSeqLensQTensor);
    mutableCuSeqLensQ->SetStorageFormat(Format::FORMAT_ND);
    mutableCuSeqLensQ->SetViewFormat(Format::FORMAT_ND);
    mutableCuSeqLensQ->SetOriginalFormat(Format::FORMAT_ND);

    const aclTensor *qScaleContiguous = qScale;
    if (qScaleContiguous != nullptr) {
        qScaleContiguous = l0op::Contiguous(qScale, executor);
        CHECK_RET(qScaleContiguous != nullptr, nullptr);
    } else {
        qScaleContiguous = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    auto qFlatTensor = executor->AllocTensor(DataType::DT_BF16, Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(StemOamPrepVarlenQ,
                           OP_INPUT(qContiguous, qSeqLensTensor, cuSeqLensQTensor, qScaleContiguous),
                           OP_OUTPUT(qFlatTensor),
                           OP_ATTR(stemBlockSize, stemStride));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "StemOamPrepVarlenQ infer shape failed.");
        return nullptr;
    }

    ADD_TO_LAUNCHER_LIST_AICORE(
        StemOamPrepVarlenQ,
        OP_INPUT(qContiguous, qSeqLensTensor, cuSeqLensQTensor, qScaleContiguous),
        OP_OUTPUT(qFlatTensor),
        OP_ATTR(stemBlockSize, stemStride));

    return qFlatTensor;
}

}  // namespace l0op
