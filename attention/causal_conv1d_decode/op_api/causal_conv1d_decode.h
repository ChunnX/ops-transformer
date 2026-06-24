#ifndef OP_API_INC_LEVEL0_CAUSAL_CONV1D_DECODE_L0OP_H_
#define OP_API_INC_LEVEL0_CAUSAL_CONV1D_DECODE_L0OP_H_

#include "opdev/op_executor.h"

namespace l0op {
bool CausalConv1dDecode(const aclTensor *x, const aclTensor *weight, aclTensor *convStates,
                        const aclTensor *queryStartLoc, const aclTensor *cacheIndices, const aclTensor *bias,
                        const aclTensor *numAcceptedTokens, int64_t activationMode, int64_t padSlotId,
                        int64_t convMode, aclTensor *y, aclOpExecutor *executor);
} // namespace l0op

#endif
