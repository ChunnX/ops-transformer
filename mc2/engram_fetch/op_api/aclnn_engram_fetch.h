#ifndef OP_API_INC_ENGRAM_FETCH_
#define OP_API_INC_ENGRAM_FETCH_

#include <string>
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现 engram_fetch 算子
 * @brief aclnnEngramFetch的第一段接口，根据具体的计算流程，计算workspace大小。
 * @param [in] commContext: 
 * @param [in] indices: 
 * @param [out] fetched: 
 * @param [out] workspaceSize: 返回需要在Device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnEngramFetchGetWorkspaceSize(const aclTensor *commContext, const aclTensor *indices,
                                                       aclTensor *fetched, int32_t hiddenSize,
                                                       int64_t numEntriesPerRank, int64_t outputDtype,
                                                       uint64_t *workspaceSize, aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnEngramFetch(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ENGRAM_FETCH_