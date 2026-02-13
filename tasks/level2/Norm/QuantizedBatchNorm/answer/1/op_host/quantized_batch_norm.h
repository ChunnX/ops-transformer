#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_QUANTIZED_BATCH_NORM_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_QUANTIZED_BATCH_NORM_OP_H_

#include "opdev/op_executor.h"

namespace l0op {

struct QuantizedBatchNormParams {
    const aclTensor* x;
    const aclTensor* mean;
    const aclTensor* var;
    const aclTensor* inputScale;
    const aclTensor* inputZeroPoint;
    const aclTensor* outputScale;
    const aclTensor* outputZeroPoint;
    const aclTensor* weight;
    const aclTensor* bias;
};
const aclTensor* QuantizedBatchNorm(const QuantizedBatchNormParams& params, float epsilon, aclOpExecutor* executor);
} // namespace l0op

#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_QUANTIZED_BATCH_NORM_OP_H_