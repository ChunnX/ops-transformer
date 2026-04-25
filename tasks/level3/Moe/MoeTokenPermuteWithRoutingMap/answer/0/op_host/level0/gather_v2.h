/**
 * Bench 侧桩头：CANN 9 公开 include 树中不再提供 level0/gather_v2.h，而 aclnn 链仍调用 l0op::GatherV2。
 * 须与 libopapi.so 中导出符号一致（CANN 9 起为 6 参，含 batch_dims / negative_index_support）。
 */
#ifndef MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_LEVEL0_GATHER_V2_H_
#define MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_LEVEL0_GATHER_V2_H_

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor *GatherV2(const aclTensor *self, int64_t dim, const aclTensor *index, aclOpExecutor *executor,
                          int batch_dims, bool negative_index_support);

} // namespace l0op

#endif
