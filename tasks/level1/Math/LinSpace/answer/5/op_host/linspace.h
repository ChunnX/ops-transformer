#ifndef OP_API_INC_LEVEL0_OP_LINSPACE_OP_H_
#define OP_API_INC_LEVEL0_OP_LINSPACE_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* Linspace(const aclTensor* start, const aclTensor* end, int64_t steps,
                          aclOpExecutor* executor);
}

#endif // OP_API_INC_LEVEL0_OP_LINSPACE_OP_H_