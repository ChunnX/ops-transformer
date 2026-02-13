#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "aclnn_util.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"

#include "add_rms_norm_dynamic_quant.h"
#include "aclnn_add_rms_norm_dynamic_quant.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr int IDX_INT8 = 2;
constexpr int IDX_HIFLOAT8 = 34;
constexpr int IDX_FLOAT8_E5M2 = 35;
constexpr int IDX_FLOAT8_E4M3FN = 36;
static std::map<op::DataType, int> dstTypeMap = {{op::DataType::DT_INT8, IDX_INT8}, 
       {op::DataType::DT_HIFLOAT8, IDX_HIFLOAT8}, {op::DataType::DT_FLOAT8_E5M2, IDX_FLOAT8_E5M2}, 
       {op::DataType::DT_FLOAT8_E4M3FN, IDX_FLOAT8_E4M3FN}};

namespace AddRmsNormDynamicQuantACLNN {
constexpr int IDX_0 = 0;
constexpr int IDX_1 = 1;
constexpr int IDX_2 = 2;
constexpr int IDX_3 = 3;
constexpr int IDX_4 = 4;

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST_X_SCALE = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> REGBASE_DTYPE_SUPPORT_LIST_Y = {
    op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_HIFLOAT8, op::DataType::DT_INT8};
static bool CheckNotNull(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, aclTensor* y1Out, aclTensor* y2Out,
    const aclTensor* xOut, const aclTensor* scale1Out, const aclTensor* scale2Out)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(gamma, return false);
    OP_CHECK_NULL(y1Out, return false);
    OP_CHECK_NULL(y2Out, return false);
    OP_CHECK_NULL(xOut, return false);
    OP_CHECK_NULL(scale1Out, return false);
    OP_CHECK_NULL(scale2Out, return false);
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* smoothScale1Optional,
    const aclTensor* smoothScale2Optional, const aclTensor* y1Out, const aclTensor* y2Out, const aclTensor* xOut,
    const aclTensor* scale1Out, const aclTensor* scale2Out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, ASCEND910B_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, ASCEND910B_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gamma, ASCEND910B_DTYPE_SUPPORT_LIST_X_SCALE, return false);

    if (nullptr != smoothScale1Optional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(smoothScale1Optional, ASCEND910B_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    }
    if (nullptr != smoothScale2Optional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(smoothScale2Optional, ASCEND910B_DTYPE_SUPPORT_LIST_X_SCALE, return false);
    }

    OP_CHECK_DTYPE_NOT_SUPPORT(y1Out, REGBASE_DTYPE_SUPPORT_LIST_Y, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y2Out, REGBASE_DTYPE_SUPPORT_LIST_Y, return false);
    OP_CHECK_DTYPE_NOT_SAME(y1Out, y2Out, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(xOut, ASCEND910B_DTYPE_SUPPORT_LIST_X_SCALE, return false);

    OP_CHECK_DTYPE_NOT_MATCH(scale1Out, op::DataType::DT_FLOAT, return false);
    OP_CHECK_DTYPE_NOT_MATCH(scale2Out, op::DataType::DT_FLOAT, return false);

    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* smoothScale1Optional,
    const aclTensor* smoothScale2Optional, aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut, aclTensor* scale1Out,
    aclTensor* scale2Out)
{
    CHECK_RET(CheckNotNull(x1, x2, gamma, y1Out, y2Out, xOut, scale1Out, scale2Out), ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(
        CheckDtypeValid(
            x1, x2, gamma, smoothScale1Optional, smoothScale2Optional, y1Out, y2Out, xOut, scale1Out, scale2Out),
        ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}
} // namespace AddRmsNormDynamicQuantACLNN

aclnnStatus ComputeAddRmsNormDynamicQuant(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* smoothScale1Optional,
    const aclTensor* smoothScale2Optional, double epsilon, aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut,
    aclTensor* scale1Out, aclTensor* scale2Out, aclOpExecutor* executor)
{
    aclTensor* y1ComputeOut = nullptr;
    aclTensor* y2ComputeOut = nullptr;
    aclTensor* xComputeOut = nullptr;

    int dstType = dstTypeMap[y1Out->GetDataType()];

    auto addRmsNormQuantOuts = l0op::AddRmsNormDynamicQuant(
        x1, x2, gamma, smoothScale1Optional, smoothScale2Optional, nullptr, epsilon, nullptr, dstType, scale1Out,  
        scale2Out, executor);
    y1ComputeOut = std::get<AddRmsNormDynamicQuantACLNN::IDX_0>(addRmsNormQuantOuts);
    y2ComputeOut = std::get<AddRmsNormDynamicQuantACLNN::IDX_1>(addRmsNormQuantOuts);
    xComputeOut = std::get<AddRmsNormDynamicQuantACLNN::IDX_2>(addRmsNormQuantOuts);

    CHECK_RET(y1ComputeOut != nullptr && y2ComputeOut != nullptr && xComputeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyY1Result = l0op::ViewCopy(y1ComputeOut, y1Out, executor);
    CHECK_RET(viewCopyY1Result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyXResult = l0op::ViewCopy(xComputeOut, xOut, executor);
    CHECK_RET(viewCopyXResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyY2Result = l0op::ViewCopy(y2ComputeOut, y2Out, executor);
    CHECK_RET(viewCopyY2Result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

const aclTensor* ContiguousX(const aclTensor* opt, aclOpExecutor* executor)
{
    if (nullptr == opt) {
        return nullptr;
    }
    return l0op::Contiguous(opt, executor);
}

aclnnStatus aclnnAddRmsNormDynamicQuantGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* smoothScale1Optional,
    const aclTensor* smoothScale2Optional, double epsilon, aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut,
    aclTensor* scale1Out, aclTensor* scale2Out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("Enter aclnnAddRmsNormDynamicQuantGetWorkspaceSize.");
    L2_DFX_PHASE_1(
        aclnnAddRmsNormDynamicQuant, DFX_IN(x1, x2, gamma, smoothScale1Optional, smoothScale2Optional, epsilon),
        DFX_OUT(y1Out, y2Out, xOut, scale1Out, scale2Out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = AddRmsNormDynamicQuantACLNN::CheckParams(
        x1, x2, gamma, smoothScale1Optional, smoothScale2Optional, y1Out, y2Out, xOut, scale1Out, scale2Out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    bool hasEmptyTensor = x1->IsEmpty() || gamma->IsEmpty() || y2Out->IsEmpty();
    bool hasReduceEmptyTensor = gamma->IsEmpty();
    if (hasEmptyTensor) {
        OP_LOGW("Got empty tensor in aclnnAddRmsNormQuant!");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto x1Cont = l0op::Contiguous(x1, uniqueExecutor.get());
    auto x2Cont = l0op::Contiguous(x2, uniqueExecutor.get());
    auto gammaCont = l0op::Contiguous(gamma, uniqueExecutor.get());

    CHECK_RET(gammaCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(x2Cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(x1Cont != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto s1Cont = ContiguousX(smoothScale1Optional, uniqueExecutor.get());
    auto s2Cont = ContiguousX(smoothScale2Optional, uniqueExecutor.get());

    ret = ComputeAddRmsNormDynamicQuant(
        x1Cont, x2Cont, gammaCont, s1Cont, s2Cont, epsilon, y1Out, y2Out, xOut, scale1Out, scale2Out,
        uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    OP_LOGD("Finish aclnnAddRmsNormQuantGetWorkspaceSize.");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAddRmsNormDynamicQuant(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAddRmsNormDynamicQuant);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif