#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* LinearIndex(
    const aclTensor *indices, const aclTensor *shape, int64_t axits, bool combine, aclOpExecutor* executor);
} // namespace l0op