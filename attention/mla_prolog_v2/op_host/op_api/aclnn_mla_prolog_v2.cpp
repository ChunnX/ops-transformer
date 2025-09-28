/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstring>
#include "graph/types.h"
#include "aclnn_mla_prolog_v2.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

extern aclnnStatus aclnnInnerMlaPrologV2GetWorkspaceSize(
    const aclTensor *tokenX, const aclTensor *weightDq, const aclTensor *weightUqQr, const aclTensor *weightUk,
    const aclTensor *weightDkvKr, const aclTensor *rmsnormGammaCq, const aclTensor *rmsnormGammaCkv,
    const aclTensor *ropeSin, const aclTensor *ropeCos, const aclTensor *cacheIndex,
    aclTensor *kvCacheRef, aclTensor *krCacheRef, const aclTensor *dequantScaleXOptional,
    const aclTensor *dequantScaleWDqOptional, const aclTensor *dequantScaleWUqQrOptional,
    const aclTensor *dequantScaleWDkvKrOptional, const aclTensor *quantScaleCkvOptional,
    const aclTensor *quantScaleCkrOptional, const aclTensor *smoothScalesCqOptional,
    double rmsnormEpsilonCq, double rmsnormEpsilonCkv, char *cacheModeOptional,
    const aclTensor *queryOut, const aclTensor *queryRopeOut, const aclTensor *dequantScaleQNopeOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerMlaPrologV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         const aclrtStream stream);


aclnnStatus aclnnMlaPrologV2GetWorkspaceSize(
    const aclTensor *tokenX,
    const aclTensor *weightDq,
    const aclTensor *weightUqQr,
    const aclTensor *weightUk,
    const aclTensor *weightDkvKr,
    const aclTensor *rmsnormGammaCq,
    const aclTensor *rmsnormGammaCkv,
    const aclTensor *ropeSin,
    const aclTensor *ropeCos,
    const aclTensor *cacheIndex,
    aclTensor *kvCacheRef,
    aclTensor *krCacheRef,
    const aclTensor *dequantScaleXOptional,
    const aclTensor *dequantScaleWDqOptional,
    const aclTensor *dequantScaleWUqQrOptional,
    const aclTensor *dequantScaleWDkvKrOptional,
    const aclTensor *quantScaleCkvOptional,
    const aclTensor *quantScaleCkrOptional,
    const aclTensor *smoothScalesCqOptional,
    double rmsnormEpsilonCq,
    double rmsnormEpsilonCkv,
    char *cacheModeOptional,
    const aclTensor *queryOut,
    const aclTensor *queryRopeOut,
    const aclTensor *dequantScaleQNopeOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    OP_LOGD("tokenX: %p, weightDq: %p, weightUqQr: %p, weightUk: %p, weightDkvKr: %p, rmsnormGammaCq: %p, rmsnormGammaCkv: %p,"
            "ropeSin: %p, ropeCos: %p, cacheIndex: %p, kvCacheRef: %p, krCacheRef: %p, dequantScaleXOptional: %p, dequantScaleWDqOptional: %p,"
            "dequantScaleWUqQrOptional: %p, dequantScaleWDkvKrOptional: %p, quantScaleCkvOptional: %p, quantScaleCkrOptional: %p, smoothScalesCqOptional: %p,"
            "rmsnormEpsilonCq: %f, rmsnormEpsilonCkv: %f, cacheModeOptional: %p, queryOut: %p, queryRopeOut: %p, dequantScaleQNopeOutOptional: %p,"
            "workspaceSize: %p, executor: %p",
             tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, cacheIndex, kvCacheRef, krCacheRef,
             dequantScaleXOptional, dequantScaleWDqOptional, dequantScaleWUqQrOptional, dequantScaleWDkvKrOptional, quantScaleCkvOptional, quantScaleCkrOptional, smoothScalesCqOptional,
             rmsnormEpsilonCq, rmsnormEpsilonCkv, cacheModeOptional, queryOut, queryRopeOut, dequantScaleQNopeOutOptional, workspaceSize, executor);

    OP_LOGE(ACLNN_ERR_INNER, "aclnnMlaPrologV2GetWorkspaceSize is not supported!");
    return ACLNN_ERR_INNER;
}

aclnnStatus aclnnMlaPrologV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream)
{
    OP_LOGD("workspace: %p, workspaceSize: %lu, executor: %p, stream: %p", workspace, workspaceSize, executor, stream);
    OP_LOGE(ACLNN_ERR_INNER, "aclnnMlaPrologV2 is not supported!");
    return ACLNN_ERR_INNER;
}

} // namespace

#ifdef __cplusplus
}
#endif