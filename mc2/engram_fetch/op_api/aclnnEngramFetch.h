#ifndef OP_API_INC_ENGRAM_FETCH_
#define OP_API_INC_ENGRAM_FETCH_

#include <string>
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnEngramFetchGetWorkspaceSize(const aclTensor *bufferhandle, const aclTensor *indices,
                                                       aclTensor *fetched, int32_t hiddenSize,
                                                       int64_t numEntriesPerRank, int64_t outputDtype,
                                                       uint64_t *workspaceSize, aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnEngramFetch(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ENGRAM_FETCH_