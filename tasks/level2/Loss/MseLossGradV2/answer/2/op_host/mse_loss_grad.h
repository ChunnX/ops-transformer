#ifndef OP_API_INC_LEVEL0_OP_MSE_LOSS_GRAD_OP_H_
#define OP_API_INC_LEVEL0_OP_MSE_LOSS_GRAD_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *MseLossGradV1(const aclTensor* gradOutput, const aclTensor *self, const aclTensor *target,
                             const std::string& reduction, aclOpExecutor *executor, const aclTensor *out);
const aclTensor *MseLossGradV2(const aclTensor* gradOutput, const aclTensor *self, const aclTensor *target,
                             const std::string& reduction, aclOpExecutor *executor, const aclTensor *out);
const aclTensor *MseLossGrad(const aclTensor* gradOutput, const aclTensor *self, const aclTensor *target,
                             const std::string& reduction, aclOpExecutor *executor);
}

#endif  // OP_API_INC_LEVEL0_OP_MSE_LOSS_GRAD_OP_H_