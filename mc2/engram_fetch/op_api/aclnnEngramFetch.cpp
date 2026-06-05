#include "aclnnEngramFetch.h"
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"

using namespace Ops::Transformer;
using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static bool CheckNotNull(const aclTensor *bufferhandle, const aclTensor *indices, aclTensor *fetched)
{
    OP_CHECK_NULL(bufferhandle, return false);
    OP_CHECK_NULL(indices, return false);
    OP_CHECK_NULL(fetched, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor *bufferhandle, const aclTensor *indices, aclTensor *fetched,
                               int32_t hiddenSize, int64_t numEntriesPerRank, int64_t outputDtype)
{
    CHECK_RET(CheckNotNull(bufferhandle, indices, fetched), ACLNN_ERR_PARAM_NULLPTR);
    if (hiddenSize <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hiddenSize must be > 0, got %d", hiddenSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (numEntriesPerRank <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "numEntriesPerRank must be > 0, got %ld", numEntriesPerRank);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (outputDtype != 1 && outputDtype != 2 && outputDtype != 3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outputDtype must be 1(BF16)/2(FP16)/3(FP32), got %ld", outputDtype);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnEngramFetchGetWorkspaceSize(const aclTensor *bufferhandle, const aclTensor *indices,
                                             aclTensor *fetched, int32_t hiddenSize, int64_t numEntriesPerRank,
                                             int64_t outputDtype, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnnEngramFetchGetWorkspaceSize start");
    auto retParam = CheckParams(bufferhandle, indices, fetched, hiddenSize, numEntriesPerRank, outputDtype);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    aclnnStatus ret = aclnnInnerEngramFetchGetWorkspaceSize(bufferhandle, indices, fetched, hiddenSize,
                                                            numEntriesPerRank, outputDtype, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnEngramFetch(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    OP_LOGD("aclnnEngramFetch execute start");
    return aclnnInnerEngramFetch(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif