#include "linspace.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "common/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(LinSpace);

static const std::initializer_list<op::DataType> ASCEND910_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_INT8, op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_UINT8,
    op::DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor *start) {
  auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
  if (npuArch == NpuArch::DAV_2201 || IsRegBase(npuArch)) {
    return CheckType(start->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
  }
  return CheckType(start->GetDataType(), ASCEND910_AICORE_DTYPE_SUPPORT_LIST);
}

static const aclTensor* LinspaceAiCpu(const aclTensor* start, const aclTensor* end, const aclTensor* steps,
                                      aclTensor* out, aclOpExecutor* executor) {
  L0_DFX(LinspaceAiCpu, start, end, steps, out);
  static internal::AicpuTaskSpace space("LinSpace");
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(LinSpace, OP_ATTR_NAMES({"Tidx"}), OP_INPUT(start, end, steps), OP_OUTPUT(out),
                                        OP_ATTR(start->GetDataType()));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LinspaceAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
  return out;
}

static const aclTensor* LinspaceAiCore(const aclTensor* start, const aclTensor* end, const aclTensor* steps,
                                       aclTensor* out, aclOpExecutor* executor) {
  L0_DFX(LinspaceAiCore, start, end, steps, out);
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(LinSpace, OP_INPUT(start, end, steps), OP_OUTPUT(out));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LinspaceAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return out;
}

const aclTensor* Linspace(const aclTensor* start, const aclTensor* end, int64_t steps,
                          aclOpExecutor* executor) {
  auto out = executor->AllocTensor({op::Shape({steps})}, start->GetDataType());
  auto stepsTensor = executor->ConvertToTensor(executor->AllocScalar(steps), op::DataType::DT_INT32);
  if (IsAiCoreSupport(start)) {
    return LinspaceAiCore(start, end, stepsTensor, out, executor);
  } else {
    return LinspaceAiCpu(start, end, stepsTensor, out, executor);
  }
}
}  // namespace l0op