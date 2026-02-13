#ifndef OP_API_INC_LEVEL0_SQUEEZE_ND_H
#define OP_API_INC_LEVEL0_SQUEEZE_ND_H

# include "opdev/op_def.h"

namespace l0op {

const aclTensor *SqueezeNd(const aclTensor *x, const aclIntArray* dim, aclOpExecutor *executor);

const aclTensor *SqueezeNd(const aclTensor *x, int64_t dim, aclOpExecutor *executor);

} // l0op

#endif  // OP_API_INC_LEVEL0_SQUEEZE_ND_H