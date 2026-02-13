#include "avgpool3d_backward.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

namespace l0op {
constexpr int64_t D_DIM = 1;
constexpr int64_t H_DIM = 2;
constexpr int64_t W_DIM = 3;
constexpr int64_t D_DIM_OFFSET = 3;
constexpr int64_t H_DIM_OFFSET = 2;
constexpr int64_t W_DIM_OFFSET = 1;
constexpr int64_t OUTD_DIM = 2;
constexpr int64_t OUTH_DIM = 3;
constexpr int64_t OUTW_DIM = 4;
constexpr int64_t EXPECT_GRAD_SHAPE = 5;

OP_TYPE_REGISTER(AvgPool3DGrad);

const aclTensor* AvgPool3DGrad(
    const aclTensor* self, const aclTensor* shapeOrigInput, const aclTensor* gradOutput, const aclIntArray* ksize,
    const aclIntArray* strides, const aclIntArray* pads, bool ceilMode, bool countIncludePad, int divisorOverride,
    const std::string& dataFormat, aclOpExecutor* executor)
{
    L0_DFX(
        AvgPool3DGrad, gradOutput, shapeOrigInput, ksize, strides, pads, ceilMode, countIncludePad, divisorOverride,
        dataFormat);

    auto gradShape = gradOutput->GetViewShape();
    auto gradSize = gradShape.GetDimNum();
    auto selfShape = self->GetViewShape();
    auto selfSize = selfShape.GetDimNum();

    if (gradSize != EXPECT_GRAD_SHAPE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "grad size [%lu] is not equal to 5", gradSize);
        return nullptr;
    }

    op::Shape outShape;
    outShape.SetDimNum(EXPECT_GRAD_SHAPE);
    if (dataFormat == "NDHWC") {
        auto nSize = gradShape.GetDim(0);
        auto ncSize = gradShape.GetDim(gradSize - 1);
        auto depth = selfShape.GetDim(selfSize - D_DIM_OFFSET);
        auto height = selfShape.GetDim(selfSize - H_DIM_OFFSET);
        auto weight = selfShape.GetDim(selfSize - W_DIM_OFFSET);
        outShape.SetDim(0, nSize);
        outShape.SetDim(1, depth);
        outShape.SetDim(OUTD_DIM, height);
        outShape.SetDim(OUTH_DIM, weight);
        outShape.SetDim(OUTW_DIM, ncSize);
    } else if (dataFormat == "NCDHW") {
        outShape = selfShape;
    }

    auto out = executor->AllocTensor(outShape, gradOutput->GetDataType(), gradOutput->GetStorageFormat());

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        AvgPool3DGrad, OP_INPUT(shapeOrigInput, gradOutput), OP_OUTPUT(out),
        OP_ATTR(ksize, strides, pads, ceilMode, countIncludePad, divisorOverride, dataFormat));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AvgPool3DGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return out;
}
} // namespace l0op