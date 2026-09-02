#include "aclnn_generic_block_sparse_attention_grad_metadata.h"
#include "generic_block_sparse_attention_grad_metadata.h"

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

#include "../generic_block_sparse_attention_grad_metadata_check.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize(
    const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount, const aclTensor *cuSeqLengthsQOptional,
    const aclTensor *cuSeqLengthsKvOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
    int64_t maxQSeqlen, int64_t maxKvSeqlen, int64_t numQHeads, int64_t numKvHeads, int64_t headDim,
    const aclIntArray *blockShape, int64_t isPackedGQA, char *layoutQ, char *layoutKv, int64_t maskType,
    int64_t softmaxPrecision, int64_t winLeft, int64_t winRight, aclTensor *metadata, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    if (workspaceSize == nullptr || executor == nullptr || sparseBlockIdx == nullptr || sparseBlockCount == nullptr ||
        metadata == nullptr || blockShape == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto checkRet = GenericBlockSparseAttentionGradMetadataCheck::ParamsCheck(
        sparseBlockIdx, sparseBlockCount, cuSeqLengthsQOptional, cuSeqLengthsKvOptional, sequsedQOptional,
        sequsedKvOptional, maxQSeqlen, maxKvSeqlen, numQHeads, numKvHeads, headDim, blockShape, isPackedGQA, layoutQ,
        layoutKv, maskType, softmaxPrecision, winLeft, winRight, metadata);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    const auto &platform = op::GetCurrentPlatformInfo();
    const int64_t aicCoreNum = static_cast<int64_t>(platform.GetCubeCoreNum());
    const int64_t aivCoreNum = static_cast<int64_t>(platform.GetVectorCoreNum());
    const std::string socVersion = platform.GetSocLongVersion();

    auto contiguous = [&](const aclTensor *tensor) -> const aclTensor * {
        return tensor == nullptr ? nullptr : l0op::Contiguous(tensor, uniqueExecutor.get());
    };
    const aclTensor *idx = contiguous(sparseBlockIdx);
    const aclTensor *count = contiguous(sparseBlockCount);
    const aclTensor *cuQ = contiguous(cuSeqLengthsQOptional);
    const aclTensor *cuKv = contiguous(cuSeqLengthsKvOptional);
    const aclTensor *usedQ = contiguous(sequsedQOptional);
    const aclTensor *usedKv = contiguous(sequsedKvOptional);
    const aclTensor *blockShapeTensor = uniqueExecutor->ConvertToTensor(blockShape, op::DataType::DT_INT64);
    CHECK_RET(idx != nullptr && count != nullptr && blockShapeTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto output = l0op::GenericBlockSparseAttentionGradMetadata(
        idx, count, cuQ, cuKv, usedQ, usedKv, maxQSeqlen, maxKvSeqlen, numQHeads, numKvHeads, headDim, blockShapeTensor,
        isPackedGQA, layoutQ, layoutKv, maskType, softmaxPrecision, winLeft, winRight, metadata, aicCoreNum, aivCoreNum,
        socVersion.c_str(), uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGenericBlockSparseAttentionGradMetadata(void *workspace, uint64_t workspaceSize,
                                                         aclOpExecutor *executor, aclrtStream stream)
{
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
