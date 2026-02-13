#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnLinearIndexGetWorkspaceSize(const aclTensor *indices, const aclTensor *shape, int64_t axits, bool combine,
                                                 const aclTensor * out,uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnLinearIndex(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif