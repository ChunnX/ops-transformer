#ifndef OP_API_INC_LEVEL0_OP_FILL_OP_H_
#define OP_API_INC_LEVEL0_OP_FILL_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *Fill(const aclTensor *dims, const aclTensor *value, const aclIntArray *outShape,
    aclOpExecutor *executor);
}

#endif