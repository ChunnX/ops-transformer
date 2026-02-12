#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_CTCLOSSV2_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_CTCLOSSV2_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const std::tuple<aclTensor*, aclTensor*> CtcLossV2(
    const aclTensor* logProbs, const aclTensor* targets, const aclTensor* inputLengthsTensor,
    const aclTensor* targetLengthsTensor, int64_t blank, bool zeroInfinity, int64_t maxTargetLengths,
    aclOpExecutor* executor);
const std::tuple<op::Shape, op::Shape> CtcLossNpuOutputShape(const aclTensor* logProbs, int64_t maxLength);
const std::tuple<aclTensor*, aclTensor*> CtcLossV3(
    const aclTensor* logProbs, const aclTensor* targets, const aclTensor* inputLengthsTensor,
    const aclTensor* targetLengthsTensor, int64_t blank, bool zeroInfinity, int64_t maxTargetLengths,
    aclOpExecutor* executor);
} // namespace l0op

#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_CTCLOSSV2_OP_H_