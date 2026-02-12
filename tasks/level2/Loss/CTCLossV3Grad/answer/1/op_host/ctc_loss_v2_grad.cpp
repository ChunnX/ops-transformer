#include "ctc_loss_v2_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(CTCLossV3Grad);

static const std::string REDUCTION_MEAN = "mean";

// CTCLossV3Grad AICORE算子kernel
const aclTensor* CtcLossV3GradAiCore(
    const aclTensor* gradOut, const aclTensor* logProbs, const aclTensor* targets, const aclTensor* inputLengthsTensor,
    const aclTensor* targetLengthsTensor, const aclTensor* negLogLikelihood, const aclTensor* logAlpha, int64_t blank,
    bool zeroInfinity, aclTensor* result, aclOpExecutor* executor)
{
    L0_DFX(
        CtcLossV3GradAiCore, gradOut, logProbs, targets, inputLengthsTensor, targetLengthsTensor, negLogLikelihood,
        logAlpha, blank, zeroInfinity, result);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        CTCLossV3Grad,
        OP_INPUT(gradOut, logProbs, targets, inputLengthsTensor, targetLengthsTensor, negLogLikelihood, logAlpha),
        OP_OUTPUT(result), OP_ATTR(blank, REDUCTION_MEAN, zeroInfinity));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CtcLossV3GradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return result;
}

const op::Shape CtcLossV2GradNpuOutputShape(const aclTensor* logProbs)
{
    return logProbs->GetViewShape();
}

const aclTensor* CtcLossV2Grad(
    const aclTensor* gradOut, const aclTensor* logProbs, const aclTensor* targets, const aclTensor* inputLengthsTensor,
    const aclTensor* targetLengthsTensor, const aclTensor* negLogLikelihood, const aclTensor* logAlpha, int64_t blank,
    bool zeroInfinity, aclOpExecutor* executor)
{
    auto outputShape = CtcLossV2GradNpuOutputShape(logProbs);
    auto result = executor->AllocTensor(outputShape, logProbs->GetDataType(), op::Format::FORMAT_ND);
    return CtcLossV3GradAiCore(
        gradOut, logProbs, targets, inputLengthsTensor, targetLengthsTensor, negLogLikelihood, logAlpha, blank,
        zeroInfinity, result, executor);
}
} // namespace l0op