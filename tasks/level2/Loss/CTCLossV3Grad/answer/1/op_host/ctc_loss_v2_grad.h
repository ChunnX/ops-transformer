#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_CTCLOSSV2GRAD_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_CTCLOSSV2GRAD_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* CtcLossV2Grad(
    const aclTensor* gradOut, const aclTensor* logProbs, const aclTensor* targets, const aclTensor* inputLengthsTensor,
    const aclTensor* targetLengthsTensor, const aclTensor* negLogLikelihood, const aclTensor* logAlpha, int64_t blank,
    bool zeroInfinity, aclOpExecutor* executor);

const aclTensor* CtcLossV3Grad(
    const aclTensor* gradOut, const aclTensor* logProbs, const aclTensor* targets, const aclTensor* inputLengthsTensor,
    const aclTensor* targetLengthsTensor, const aclTensor* negLogLikelihood, const aclTensor* logAlpha, int64_t blank,
    bool zeroInfinity, aclOpExecutor* executor);
} // namespace l0op

#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_CTCLOSSV2GRAD_OP_H_