#include "adaptive_avg_pool3d.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/reshape.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/reduce_mean.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_adaptive_avg_pool3d.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const int64_t INDEX_DIM0 = 0;
static const int64_t INDEX_DIM1 = 1;
static const int64_t INDEX_DIM2 = 2;
static const int64_t INDEX_DIM3 = 3;
static const int64_t INDEX_DIM4 = 4;
static const int64_t cdhwShapeSize = 4;
static const int64_t ncdhwShapeSize = 5;
static const int64_t outputSizeLimit = 3;
static const std::initializer_list<op::DataType> dtypeSupportList310P = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> dtypeSupportListOrigin = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(outputSize, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckInputOutputDims(const aclTensor* self, const aclTensor* out)
{
    auto inputShape = self->GetViewShape();
    auto outputShape = out->GetViewShape();
    size_t inputDimNum = inputShape.GetDimNum();
    size_t outputDimNum = outputShape.GetDimNum();
    if (inputDimNum != static_cast<size_t>(cdhwShapeSize) && inputDimNum != static_cast<size_t>(ncdhwShapeSize)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dims %zu should be in 4 or 5.", inputDimNum);
        return false;
    }
    if (inputDimNum != outputDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Out dims %zu should equal to self dims %zu.", outputDimNum, inputDimNum);
        return false;
    }
    return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return dtypeSupportListOrigin;
    } else {
        return dtypeSupportList310P;
    }
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    const auto& dtypeSupportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    if (self->GetStorageFormat() != out->GetStorageFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of input and out should be equal, input [%s], out [%s].",
            op::ToString(self->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }

    if (op::IsPrivateFormat(self->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format don't support private format.");
        return false;
    }
    return true;
}

static bool CheckOutShape(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* out)
{
    uint64_t size = outputSize->Size();
    if (size != static_cast<size_t>(outputSizeLimit)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outputSize length should be 3.");
        return false;
    }
    auto selfShape = self->GetViewShape();
    auto outputShape = out->GetViewShape();
    size_t outputDimNum = outputShape.GetDimNum();
    size_t offset = outputDimNum == static_cast<size_t>(cdhwShapeSize) ? 1UL : 2UL;
    for (size_t i = 0; i < offset; ++i) {
        if (selfShape.GetDim(i) != outputShape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The [%lu] dim of outShape value should match the selfShape's.", i);
            return false;
        }
    }
    for (size_t i = 0; i < size; ++i) {
        if ((*outputSize)[i] <= 0 || (*outputSize)[i] != outputShape.GetDim(i + offset)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "outShape value should match the outputSize value and greater than 0.");
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* outputSize, const aclTensor* out)
{
    CHECK_RET(CheckNotNull(self, outputSize, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckInputOutputDims(self, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOutShape(self, outputSize, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus DoAdaptiveAvgPool3D(
    const aclTensor* self, const aclIntArray* outputSize, aclTensor* out, aclOpExecutor* executor)
{
    auto inputReshape = self;
    if (self->GetViewShape().GetDimNum() == cdhwShapeSize) {
        op::Shape inputShape = self->GetViewShape();
        std::vector<int64_t> valueShape(ncdhwShapeSize);
        valueShape[0] = 1;
        for (int64_t i = 1; i < ncdhwShapeSize; i++) {
            valueShape[i] = inputShape.GetDim(i - 1);
        }
        auto reshapeShape = executor->AllocIntArray(valueShape.data(), ncdhwShapeSize);
        CHECK_RET(reshapeShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        inputReshape = l0op::Reshape(self, reshapeShape, executor);
    }
    CHECK_RET(inputReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    std::vector<int64_t> valuePerm{INDEX_DIM0, INDEX_DIM2, INDEX_DIM3, INDEX_DIM4, INDEX_DIM1};
    auto tranPerm = executor->AllocIntArray(valuePerm.data(), ncdhwShapeSize);
    CHECK_RET(tranPerm != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto inputReshapeTran = l0op::Transpose(inputReshape, tranPerm, executor);
    CHECK_RET(inputReshapeTran != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto pool3dResult = l0op::AdaptiveAvgPool3d(inputReshapeTran, outputSize, executor);
    CHECK_RET(pool3dResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    std::vector<int64_t> valuePermRes{INDEX_DIM0, INDEX_DIM4, INDEX_DIM1, INDEX_DIM2, INDEX_DIM3};
    auto resPerm = executor->AllocIntArray(valuePermRes.data(), ncdhwShapeSize);
    CHECK_RET(resPerm != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto resTran = l0op::Transpose(pool3dResult, resPerm, executor);
    CHECK_RET(resTran != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto resTranReshape = resTran;
    if (self->GetViewShape().GetDimNum() == cdhwShapeSize) {
        auto outShapeVector = op::ToShapeVector(out->GetViewShape());
        aclIntArray* outShapeArray = executor->AllocIntArray(outShapeVector.data(), outShapeVector.size());
        CHECK_RET(outShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
        resTranReshape = l0op::Reshape(resTran, outShapeArray, executor);
    }
    CHECK_RET(resTranReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(resTranReshape, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus SelectAdaptiveAvgPool3D(
    const aclTensor* self, const aclIntArray* outputSize, aclTensor* out, aclOpExecutor* executor)
{
    return DoAdaptiveAvgPool3D(self, outputSize, out, executor);
}

aclnnStatus aclnnAdaptiveAvgPool3dGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* outputSize, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnAdaptiveAvgPool3d, DFX_IN(self, outputSize), DFX_OUT(out));
    auto ret = CheckParams(self, outputSize, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (self->IsEmpty()) {
        *workspaceSize = 0UL;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    ret = SelectAdaptiveAvgPool3D(selfContiguous, outputSize, out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAdaptiveAvgPool3d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAdaptiveAvgPool3d);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif