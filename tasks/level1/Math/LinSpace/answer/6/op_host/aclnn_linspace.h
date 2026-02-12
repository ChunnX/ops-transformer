#ifndef OP_API_INC_LINSPACE_H_
#define OP_API_INC_LINSPACE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnLinspaceGetWorkspaceSize(const aclScalar* start, const aclScalar* end, int64_t steps,
                                                    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);
ACLNN_API aclnnStatus aclnnLinspace(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif