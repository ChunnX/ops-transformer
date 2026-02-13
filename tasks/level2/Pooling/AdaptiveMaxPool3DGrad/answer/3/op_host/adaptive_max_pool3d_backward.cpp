#include "adaptive_max_pool3d_backward.h"
#include "opdev/data_type_utils.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(AdaptiveMaxPool3DGrad);

const inline aclTensor* AdaptiveMaxPool3DGradAiCore(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, aclTensor* gradInput,
    aclOpExecutor* executor)
{
    L0_DFX(AdaptiveMaxPool3DGradAiCore, self, gradOutput, indices, gradInput);
    auto ret =
        ADD_TO_LAUNCHER_LIST_AICORE(AdaptiveMaxPool3DGrad, OP_INPUT(self, gradOutput, indices), OP_OUTPUT(gradInput));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AdaptiveMaxPool3DGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return gradInput;
}

const aclTensor* AdaptiveMaxPool3DGrad(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, aclOpExecutor* executor)
{
    auto gradInput = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), op::Format::FORMAT_NCDHW);
    if (gradInput == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "gradInput is nullptr.");
        return nullptr;
    }
    return AdaptiveMaxPool3DGradAiCore(gradOutput, self, indices, gradInput, executor);
}
} // namespace l0op