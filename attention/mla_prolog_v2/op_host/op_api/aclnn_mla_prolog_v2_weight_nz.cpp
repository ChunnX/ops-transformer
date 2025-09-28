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
#include "aclnn_mla_prolog_v2_weight_nz.h"

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


aclnnStatus aclnnMlaPrologV2WeightNzGetWorkspaceSize(
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
    const aclTensor *dequantScaleQNopeOutHolder = nullptr;
    bool isDequantScaleQNope= (dequantScaleQNopeOutOptional != nullptr);
    if (isDequantScaleQNope) {
        dequantScaleQNopeOutHolder = dequantScaleQNopeOutOptional;
    } else {
        std::vector<int64_t> shape = {0};
        int64_t addr = 0xff;
        dequantScaleQNopeOutHolder = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape.data(), 0, ACL_FORMAT_ND,
                                     shape.data(), shape.size(), static_cast<void *>(&addr));
    }
    if (tokenX ->GetDataType() == ge::DT_INT8 && kvCacheRef ->GetDataType() == ge::DT_INT8 && !isDequantScaleQNope) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Check dequantScaleQNopeOut != nullptr failed!");
    }

    aclnnStatus ret = aclnnInnerMlaPrologV2GetWorkspaceSize(
        tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos,
        cacheIndex, kvCacheRef, krCacheRef, dequantScaleXOptional, dequantScaleWDqOptional, dequantScaleWUqQrOptional,
        dequantScaleWDkvKrOptional, quantScaleCkvOptional, quantScaleCkrOptional, smoothScalesCqOptional,
        rmsnormEpsilonCq, rmsnormEpsilonCkv, cacheModeOptional, queryOut, queryRopeOut, dequantScaleQNopeOutHolder,
        workspaceSize, executor);

    if (!isDequantScaleQNope) {
        aclDestroyTensor(dequantScaleQNopeOutHolder);
    }
    return ret;
}

aclnnStatus aclnnMlaPrologV2WeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream)
{
    return aclnnInnerMlaPrologV2(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif