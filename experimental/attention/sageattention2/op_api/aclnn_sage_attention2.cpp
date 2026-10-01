#include "aclnn_sage_attention2.h"
#include <cmath>
#include <limits>
#include <string_view>
#include <tuple>
#include <utility>
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "../op_kernel/platform_traits.h"
#include "../op_kernel/sage_attention2_tiling_data.h"

namespace sage::interfaces {
namespace {

using namespace op;
using sage::core::IsKnownPvAccumName;
using sage::core::IsKnownQuantGranName;
using sage::core::MAX_HEAD_DIM;
using sage::core::ParsePvAccum;
using sage::core::ParseQuantGran;
using sage::core::SagePvAccum;

#define VALIDATE_TRUE(expr, code) do { if (!(expr)) return (code); } while (0)

struct TensorShape4D {
    int64_t b = 0, h = 0, s = 0, d = 0;
    explicit TensorShape4D(const aclTensor *t) noexcept {
        if (t) {
            const auto &view = t->GetViewShape();
            if (view.GetDimNum() == 4) {
                b = view[0]; h = view[1]; s = view[2]; d = view[3];
            }
        }
    }
};

template <typename... Tensors>
inline bool ValidateTensorsMeta(const aclTensor *first, const Tensors *...rest) noexcept {
    auto validateSingle = [](const aclTensor *tensor) noexcept -> bool {
        if (!tensor) return false;
        const auto &view = tensor->GetViewShape();
        if (view.GetDimNum() != 4) return false;
        const auto elemType = tensor->GetDataType();
        if (elemType != DataType::DT_FLOAT16 && elemType != DataType::DT_BF16) return false;
        for (int idx = 0; idx < 4; ++idx) {
            if (view[idx] <= 0 || view[idx] >= static_cast<int64_t>(UINT32_MAX - 64)) return false;
        }
        // 末维连续性校验：必须满足 stride(-1) == 1
        const auto &strides = tensor->GetViewStrides();
        if (strides.size() != 4 || strides[3] != 1) return false;
        return true;
    };
    if (!validateSingle(first)) return false;
    const auto leadDtype = first->GetDataType();
    auto validateAndMatch = [leadDtype, &validateSingle](const aclTensor *t) noexcept -> bool {
        return validateSingle(t) && (t->GetDataType() == leadDtype);
    };
    return (validateAndMatch(rest) && ...);
}

inline bool ValidateStorageContiguity(const aclTensor *tensor) noexcept {
    if (!tensor || tensor->GetViewOffset() != 0) return false;
    const auto &shapeView = tensor->GetViewShape();
    const auto &strideView = tensor->GetViewStrides();
    int64_t cumStride = 1;
    for (int axis = static_cast<int>(shapeView.GetDimNum()) - 1; axis >= 0; --axis) {
        if (shapeView[axis] != 1 && strideView[axis] != cumStride) return false;
        cumStride *= shapeView[axis];
    }
    return true;
}

template <typename... Tensors>
inline auto TransposeToHndAll(bool isNhd, aclOpExecutor *exec, const Tensors *... tensors) noexcept {
    auto toHnd = [isNhd, exec](const aclTensor *t) noexcept -> const aclTensor * {
        if (!isNhd || !t) return t;
        const auto &sv = t->GetViewShape();
        const auto &stv = t->GetViewStrides();
        return exec->CreateView(t, op::Shape{sv[0], sv[2], sv[1], sv[3]},
                                t->GetStorageShape(),
                                op::Strides{stv[0], stv[2], stv[1], stv[3]},
                                t->GetViewOffset());
    };
    return std::make_tuple(toHnd(tensors)...);
}

template <typename... Tensors>
inline auto MakeContiguousAll(aclOpExecutor *exec, const Tensors *... tensors) noexcept {
    return std::make_tuple(l0op::Contiguous(tensors, exec)...);
}

} // namespace
} // namespace sage::interfaces

namespace l0op {
OP_TYPE_REGISTER(SageAttention2);

inline aclnnStatus BuildSageLauncher(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *out, const aclTensor *lse,
    float smScale, bool smoothK, bool isCausal, bool returnLse,
    int64_t quantCode, int64_t accumCode, bool smoothV,
    aclOpExecutor *executor)
{
    return ADD_TO_LAUNCHER_LIST_AICORE(
        SageAttention2,
        OP_INPUT(q, k, v),
        OP_OUTPUT(out, lse),
        OP_ATTR(smScale, smoothK, isCausal, returnLse, quantCode, accumCode, smoothV));
}
} // namespace l0op

extern "C" {

aclnnStatus aclnnSageAttention2GetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const char *tensorLayout,
    bool isCausal, const char *qkQuantGran, double smScale, const char *pvAccumDtype,
    bool smoothK, bool smoothV, bool returnLse, const aclTensor *out, const aclTensor *lse,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    using namespace sage::interfaces;
    using namespace l0op;

    VALIDATE_TRUE(workspaceSize && executor && q && k && v && out && tensorLayout && qkQuantGran && pvAccumDtype,
                  ACLNN_ERR_PARAM_NULLPTR);
    VALIDATE_TRUE(!returnLse || lse != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    *workspaceSize = 0; *executor = nullptr;

    const int64_t quantCode = static_cast<int64_t>(ParseQuantGran(qkQuantGran));
    const int64_t accumCode = static_cast<int64_t>(ParsePvAccum(pvAccumDtype));
    const bool validScale = std::isfinite(smScale) && (std::abs(smScale) <= static_cast<double>(std::numeric_limits<float>::max()));
    VALIDATE_TRUE(IsKnownQuantGranName(qkQuantGran) && IsKnownPvAccumName(pvAccumDtype) && validScale,
                  ACLNN_ERR_PARAM_INVALID);

    const std::string_view layoutToken(tensorLayout);
    const bool isNhd = (layoutToken == "NHD");
    VALIDATE_TRUE(isNhd || layoutToken == "HND", ACLNN_ERR_PARAM_INVALID);

    if (smoothV && accumCode != static_cast<int64_t>(sage::core::SagePvAccum::FP32)) {
        OP_LOGW("smooth_v is ignored unless pv_accum_dtype is \"fp32\".");
    }

    VALIDATE_TRUE(ValidateTensorsMeta(q, k, v, out) && (q->GetViewShape() == out->GetViewShape()),
                  ACLNN_ERR_PARAM_INVALID);

    const bool platformValid = (op::GetCurrentPlatformInfo().GetSocLongVersion().rfind("Ascend950", 0) == 0);
    VALIDATE_TRUE(platformValid, ACLNN_ERR_PARAM_INVALID);

    L2_DFX_PHASE_1(aclnnSageAttention2,
                   DFX_IN(q, k, v, tensorLayout, isCausal, qkQuantGran, smScale, pvAccumDtype,
                          smoothK, smoothV, returnLse),
                   DFX_OUT(out, lse));

    auto execContext = CREATE_EXECUTOR();
    VALIDATE_TRUE(execContext.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto [hndQ, hndK, hndV, hndOut] = TransposeToHndAll(isNhd, execContext.get(), q, k, v, out);
    VALIDATE_TRUE(hndQ && hndK && hndV && hndOut, ACLNN_ERR_INNER_NULLPTR);

    TensorShape4D qDims(hndQ), kDims(hndK), vDims(hndV);

    const bool geometryValid = (hndK->GetViewShape() == hndV->GetViewShape())
                            && (qDims.b == kDims.b) && (qDims.d == kDims.d)
                            && (qDims.d <= MAX_HEAD_DIM) && (qDims.h % kDims.h == 0)
                            && (uint64_t(kDims.b) * kDims.h < UINT32_MAX)
                            && (!isCausal || qDims.s == kDims.s);
    VALIDATE_TRUE(geometryValid, ACLNN_ERR_PARAM_INVALID);

    const op::Shape lseShape{qDims.b, qDims.h, qDims.s};
    if (returnLse) {
        VALIDATE_TRUE(lse->GetDataType() == DataType::DT_FLOAT && lse->GetViewShape() == lseShape, ACLNN_ERR_PARAM_INVALID);
    }

    auto [cQ, cK, cV] = MakeContiguousAll(execContext.get(), hndQ, hndK, hndV);
    VALIDATE_TRUE(cQ && cK && cV, ACLNN_ERR_INNER_NULLPTR);

    const bool directOut = !isNhd && ValidateStorageContiguity(hndOut);
    const aclTensor *targetOut = directOut ? hndOut : execContext->AllocTensor(hndQ->GetViewShape(), q->GetDataType());

    const bool directLse = (!returnLse) || (!isNhd && ValidateStorageContiguity(lse));
    const aclTensor *targetLse = (returnLse && directLse) ? lse : execContext->AllocTensor(lseShape, DataType::DT_FLOAT);
    VALIDATE_TRUE(targetOut && targetLse, ACLNN_ERR_INNER_NULLPTR);

    const auto launchResult = l0op::BuildSageLauncher(
        cQ, cK, cV, targetOut, targetLse,
        static_cast<float>(smScale), smoothK, isCausal, returnLse,
        quantCode, accumCode, smoothV,
        execContext.get());
    if (launchResult != ACLNN_SUCCESS) return launchResult;

    if (!directOut) {
        VALIDATE_TRUE(l0op::ViewCopy(targetOut, hndOut, execContext.get()), ACLNN_ERR_INNER_NULLPTR);
    }
    if (returnLse && !directLse) {
        VALIDATE_TRUE(l0op::ViewCopy(targetLse, lse, execContext.get()), ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = execContext->GetWorkspaceSize(); execContext.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSageAttention2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    VALIDATE_TRUE(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    L2_DFX_PHASE_2(aclnnSageAttention2);
    return ::CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

} // extern "C"
