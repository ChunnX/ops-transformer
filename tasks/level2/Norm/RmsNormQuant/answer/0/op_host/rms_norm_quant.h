/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_RMS_NORM_QUANT_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_RMS_NORM_QUANT_H_

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor* RmsNormQuant(
    const aclTensor* x, const aclTensor* gamma, const aclTensor* beta, const aclTensor* scale, const aclTensor* offset,
    double epsilon, int32_t dstType, aclOpExecutor* executor);

const std::array<aclTensor*, 2> RmsNormQuantV2(
    const aclTensor* x, const aclTensor* gamma, const aclTensor* scales1, const aclTensor* scales2Optional,
    const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional, const aclTensor* betaOptional,
    double epsilon, bool divMode, int32_t dstType, aclOpExecutor* executor);
} // namespace l0op

#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_RMS_NORM_QUANT_H_
