#include "linear_index.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "op_api/aclnn_util.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(LinearIndex);


inline static const aclTensor* LinearIndexAiCore(const aclTensor *indices, const aclTensor *shape, const aclTensor *out, int64_t axits, bool combine, aclOpExecutor* executor) {
  L0_DFX(LinearIndexAiCore, indices, shape, out, axits, combine);
  auto retAicore =
    ADD_TO_LAUNCHER_LIST_AICORE(LinearIndex,
                                OP_INPUT(indices, shape), OP_OUTPUT(out), OP_ATTR(axits, combine));
  CHECK_RET(retAicore == ACLNN_SUCCESS, nullptr);
  return out;
}

const aclTensor* LinearIndex(const aclTensor *indices, const aclTensor *shape, int64_t axits, bool combine, aclOpExecutor* executor) {
    auto out = executor->AllocTensor(indices->GetViewShape(), op::DataType::DT_INT32);
    CHECK_RET(out != nullptr, nullptr);
    return LinearIndexAiCore(indices, shape, out, axits, combine, executor);
}
}  // namespace l0op