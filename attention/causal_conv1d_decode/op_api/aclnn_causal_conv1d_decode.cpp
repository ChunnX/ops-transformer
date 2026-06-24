#include "aclnn_causal_conv1d_decode.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "causal_conv1d_decode.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {

const aclTensor *EnsureContiguous(const aclTensor *tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return nullptr;
    }
    return l0op::Contiguous(tensor, executor);
}

aclnnStatus CommonProcess(const aclTensor *x, const aclTensor *weight, aclTensor *convStates,
                          const aclTensor *queryStartLoc, const aclTensor *cacheIndices, const aclTensor *bias,
                          const aclTensor *numAcceptedTokens, int64_t activationMode, int64_t padSlotId,
                          int64_t convMode, aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();

    const aclTensor *xFinal =
        uniqueExecutor->CreateView(x, x->GetViewShape(), x->GetStorageShape(), x->GetViewStrides(), x->GetViewOffset());
    CHECK_COND(xFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR, "CreateView for x failed.");

    aclTensor *convStatesFinal =
        uniqueExecutor->CreateView(convStates, convStates->GetViewShape(), convStates->GetStorageShape(),
                                   convStates->GetViewStrides(), convStates->GetViewOffset());
    CHECK_COND(convStatesFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR, "CreateView for convStates failed.");

    const aclTensor *weightFinal = l0op::Contiguous(weight, uniqueExecutor.get());
    CHECK_COND(weightFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous weight failed.");
    const aclTensor *queryStartLocFinal = EnsureContiguous(queryStartLoc, uniqueExecutor.get());
    const aclTensor *cacheIndicesFinal = EnsureContiguous(cacheIndices, uniqueExecutor.get());
    const aclTensor *biasFinal = EnsureContiguous(bias, uniqueExecutor.get());
    const aclTensor *numAcceptedTokensFinal = EnsureContiguous(numAcceptedTokens, uniqueExecutor.get());

    bool ok = l0op::CausalConv1dDecode(xFinal, weightFinal, convStatesFinal, queryStartLocFinal, cacheIndicesFinal,
                                       biasFinal, numAcceptedTokensFinal, activationMode, padSlotId, convMode, y,
                                       uniqueExecutor.get());
    CHECK_RET(ok, ACLNN_ERR_INNER_TILING_ERROR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnCausalConv1dDecodeGetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight, aclTensor *convStates, const aclTensor *queryStartLoc,
    const aclTensor *cacheIndices, const aclTensor *bias, const aclTensor *numAcceptedTokens,
    int64_t activationMode, int64_t padSlotId, int64_t convMode, aclTensor *y, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnCausalConv1dDecode,
                   DFX_IN(x, weight, convStates, queryStartLoc, cacheIndices, bias, numAcceptedTokens), DFX_OUT(y));
    return CommonProcess(x, weight, convStates, queryStartLoc, cacheIndices, bias, numAcceptedTokens, activationMode,
                         padSlotId, convMode, y, workspaceSize, executor);
}

ACLNN_API aclnnStatus aclnnCausalConv1dDecode(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                              aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCausalConv1dDecode);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
