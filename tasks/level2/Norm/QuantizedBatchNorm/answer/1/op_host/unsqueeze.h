#ifndef OP_API_INC_LEVEL0_UNSQUEEZE_ND_H
#define OP_API_INC_LEVEL0_UNSQUEEZE_ND_H

# include "opdev/op_def.h"

namespace l0op {

const aclTensor *UnsqueezeNd(const aclTensor *x, const aclIntArray* dim, aclOpExecutor *executor);

const aclTensor *UnsqueezeNd(const aclTensor *x, int64_t dim, aclOpExecutor *executor);

} // l0op

#endif  // OP_API_INC_LEVEL0_UNSQUEEZE_ND_H