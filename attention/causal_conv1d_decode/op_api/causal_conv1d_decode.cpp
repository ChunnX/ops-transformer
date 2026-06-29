#include "causal_conv1d_decode.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(CausalConv1dDecode);

bool CausalConv1dDecode(const aclTensor *x, const aclTensor *weight, aclTensor *convStates,
                        const aclTensor *queryStartLoc, const aclTensor *cacheIndices, const aclTensor *bias,
                        const aclTensor *numAcceptedTokens, int64_t activationMode, int64_t padSlotId,
                        int64_t convMode, aclTensor *y, aclOpExecutor *executor)
{
    L0_DFX(CausalConv1dDecode, x, weight, convStates, queryStartLoc, cacheIndices, bias, numAcceptedTokens,
           activationMode, padSlotId, convMode, y);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        CausalConv1dDecode,
        OP_INPUT(x, weight, convStates, queryStartLoc, cacheIndices, bias, numAcceptedTokens),
        OP_OUTPUT(y),
        OP_ATTR(activationMode, padSlotId, convMode));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CausalConv1dDecode ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return false;
    }
    return true;
}

} // namespace l0op
