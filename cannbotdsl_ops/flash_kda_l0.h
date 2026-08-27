#ifndef OPS_TRANSFORMER_CANNBOTDSL_FLASH_KDA_L0_H_
#define OPS_TRANSFORMER_CANNBOTDSL_FLASH_KDA_L0_H_

#include "opdev/op_executor.h"

namespace l0op {

aclnnStatus FlashKda(
    const aclTensor* k, const aclTensor* v, const aclTensor* q,
    const aclTensor* beta, const aclTensor* g, const aclTensor* aLog,
    const aclTensor* dtBias, const aclTensor* initialState, float scale,
    float lowerBound, const aclTensor* out, const aclTensor* finalState,
    aclOpExecutor* executor);

}  // namespace l0op

#endif  // OPS_TRANSFORMER_CANNBOTDSL_FLASH_KDA_L0_H_
