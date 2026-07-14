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
 * \file stem_oam_prep_paged_kv.cpp
 * \brief
 */

#include "stem_oam_prep_paged_kv.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(StemOamPrepPagedKv);

const aclTensor *StemOamPrepPagedKv(
    const aclTensor *kCache, const aclTensor *vCache, const aclTensor *kvIndices,
    const aclTensor *kvSeqLens, const aclTensor *kScaleCache, const aclTensor *vScale, float lambdaMag,
    int64_t *cacheLayout, int64_t *kvBlockSize, int64_t *stemBlockSize, int64_t *stemStride,
    const aclTensor *kFlat, const aclTensor *vBias, aclOpExecutor *executor)
{
    L0_DFX(StemOamPrepPagedKv, kCache, vCache, kvIndices, kvSeqLens, kScaleCache, vScale, lambdaMag,
           cacheLayout, kvBlockSize, stemBlockSize, stemStride, kFlat, vBias);

    Shape outShapeShape{9};
    auto kFlatShape = executor->AllocTensor(outShapeShape, DataType::DT_INT64, Format::FORMAT_ND);
    auto vBiasShape = executor->AllocTensor(outShapeShape, DataType::DT_INT64, Format::FORMAT_ND);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(StemOamPrepPagedKv,
        OP_INTPUT(kCache, vCache, kvIndices, kvSeqLens, kScaleCache, vScale),
        OP_ATTR({lambdaMag, cacheLayout, kvBlockSize, stemBlockSize, stemStride}),
        OP_OUTPUT(kFlat, vBias),
        OP_OUTSHAPE({kFlatShape, 0}),
        OP_OUTSHAPE({vBiasShape, 1}));

    OP_CHECK(ret == ACL_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SparseLightningIndexerKLLossGradMetadata ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return {kFlat, vBias};
}
}  // namespace l0op
