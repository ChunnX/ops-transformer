#ifndef OP_API_INC_ENGRAM_FETCH_WAIT_
#define OP_API_INC_ENGRAM_FETCH_WAIT_

#include <string>
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnEngramFetchWaitGetWorkspaceSize(const aclTensor *bufferHandle, uint64_t *workspaceSize,
                                                           aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnEngramFetchWait(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ENGRAM_FETCH_WAIT_
