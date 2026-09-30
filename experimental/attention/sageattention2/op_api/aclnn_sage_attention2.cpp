#include "aclnn_sage_attention2.h"
#include <cmath>
#include <limits>
#include <string_view>
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
using sage::core::MAX_HEAD_DIM;
using sage::core::ParsePvAccum;
using sage::core::ParseQuantGran;
using sage::core::SagePvAccum;

struct TensorInspector {
    [[nodiscard]] static bool IsValidRank4Input(const aclTensor *tensor) noexcept
    {
        if (!tensor) {
            return false;
        }
        const auto &viewShape = tensor->GetViewShape();
        if (viewShape.GetDimNum() != 4) {
            return false;
        }
        for (int i = 0; i < 4; ++i) {
            const int64_t dim = viewShape[i];
            if (dim <= 0 || dim >= static_cast<int64_t>(UINT32_MAX - 64)) {
                return false;
            }
        }
        const auto dt = tensor->GetDataType();
        return (dt == DataType::DT_FLOAT16 || dt == DataType::DT_BF16);
    }

    [[nodiscard]] static bool IsContiguouslyPacked(const aclTensor *tensor) noexcept
    {
        if (!tensor || tensor->GetViewOffset() != 0) {
            return false;
        }
        const auto &shape = tensor->GetViewShape();
        const auto strides = tensor->GetViewStrides();
        int64_t accumulatedStride = 1;
        for (int64_t idx = static_cast<int64_t>(shape.GetDimNum()) - 1; idx >= 0; --idx) {
            const int d = static_cast<int>(idx);
            if (shape[d] != 1 && strides[d] != accumulatedStride) {
                return false;
            }
            accumulatedStride *= shape[d];
        }
        return true;
    }
};

struct LayoutAdapter {
    [[nodiscard]] static const aclTensor *ToCanonicalHndView(
        const aclTensor *tensor, bool isNhd, aclOpExecutor *executor)
    {
        if (!isNhd) {
            return tensor;
        }
        auto shape = tensor->GetViewShape();
        auto strides = tensor->GetViewStrides();
        std::swap(shape[1], shape[2]);
        std::swap(strides[1], strides[2]);
        return executor->CreateView(tensor, shape, tensor->GetStorageShape(), strides, tensor->GetViewOffset());
    }
};

} // namespace
} // namespace sage::interfaces

namespace l0op {
OP_TYPE_REGISTER(SageAttention2);

inline aclnnStatus DispatchSageAttention2Kernel(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    float scale, bool smoothK, bool isCausal, bool returnLse,
    int64_t gran, int64_t accum, bool smoothV,
    const aclTensor *out, const aclTensor *lse,
    aclOpExecutor *executor)
{
    return ADD_TO_LAUNCHER_LIST_AICORE(
        SageAttention2,
        OP_INPUT(q, k, v),
        OP_OUTPUT(out, lse),
        OP_ATTR(scale, smoothK, isCausal, returnLse, gran, accum, smoothV));
}

} // namespace l0op

extern "C" {

aclnnStatus aclnnSageAttention2GetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const char *tensorLayout, bool isCausal,
    const char *qkQuantGran, double smScale,
    const char *pvAccumDtype, bool smoothK, bool smoothV,
    bool returnLse, const aclTensor *out, const aclTensor *lse,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    using namespace sage::interfaces;

    if (!workspaceSize || !executor || !q || !k || !v || !out || !tensorLayout || !qkQuantGran || (returnLse && !lse)) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    *workspaceSize = 0;
    *executor = nullptr;

    const int64_t quantGranularCode = static_cast<int64_t>(ParseQuantGran(qkQuantGran));
    if (quantGranularCode < 0 || !std::isfinite(smScale) ||
        std::abs(smScale) > static_cast<double>(std::numeric_limits<float>::max())) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    const int64_t accumCode = static_cast<int64_t>(ParsePvAccum(pvAccumDtype));
    if (accumCode < 0) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    const std::string_view layoutView(tensorLayout);
    const bool isNhd = (layoutView == "NHD");
    if (!isNhd && layoutView != "HND") {
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (smoothV && accumCode != static_cast<int64_t>(sage::core::SagePvAccum::FP32)) {
        OP_LOGW("smooth_v is ignored unless pv_accum_dtype is \"fp32\".");
    }

    if (!TensorInspector::IsValidRank4Input(q) ||
        !TensorInspector::IsValidRank4Input(k) ||
        !TensorInspector::IsValidRank4Input(v) ||
        !TensorInspector::IsValidRank4Input(out) ||
        q->GetDataType() != k->GetDataType() ||
        q->GetDataType() != v->GetDataType() ||
        q->GetDataType() != out->GetDataType() ||
        q->GetViewShape() != out->GetViewShape()) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    static const bool isAscend950Platform =
        (op::GetCurrentPlatformInfo().GetSocLongVersion().rfind("Ascend950", 0) == 0);
    if (!isAscend950Platform) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    L2_DFX_PHASE_1(aclnnSageAttention2,
                   DFX_IN(q, k, v, tensorLayout, isCausal, qkQuantGran, smScale, pvAccumDtype,
                          smoothK, smoothV, returnLse),
                   DFX_OUT(out, lse));

    auto executionOwner = CREATE_EXECUTOR();
    if (!executionOwner.get()) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }

    const aclTensor *canonQ = LayoutAdapter::ToCanonicalHndView(q, isNhd, executionOwner.get());
    const aclTensor *canonK = LayoutAdapter::ToCanonicalHndView(k, isNhd, executionOwner.get());
    const aclTensor *canonV = LayoutAdapter::ToCanonicalHndView(v, isNhd, executionOwner.get());
    const aclTensor *canonOut = LayoutAdapter::ToCanonicalHndView(out, isNhd, executionOwner.get());

    if (!canonQ || !canonK || !canonV || !canonOut) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    const auto &qShape = canonQ->GetViewShape();
    const auto &kShape = canonK->GetViewShape();
    const auto &vShape = canonV->GetViewShape();

    if (kShape != vShape ||
        qShape[0] != kShape[0] ||
        qShape[3] != kShape[3] ||
        qShape[3] > MAX_HEAD_DIM ||
        (qShape[1] % kShape[1] != 0) ||
        (uint64_t(kShape[0]) * kShape[1] >= UINT32_MAX) ||
        (isCausal && qShape[2] != kShape[2])) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    op::Shape expectedLseShape{qShape[0], qShape[1], qShape[2]};
    if (returnLse && (lse->GetDataType() != DataType::DT_FLOAT || lse->GetViewShape() != expectedLseShape)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    const aclTensor *contigQ = l0op::Contiguous(canonQ, executionOwner.get());
    const aclTensor *contigK = l0op::Contiguous(canonK, executionOwner.get());
    const aclTensor *contigV = l0op::Contiguous(canonV, executionOwner.get());
    if (!contigQ || !contigK || !contigV) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    const bool directWriteOut = !isNhd && TensorInspector::IsContiguouslyPacked(canonOut);
    const aclTensor *targetOut = directWriteOut ? canonOut : executionOwner->AllocTensor(qShape, q->GetDataType());

    const bool directWriteLse = (!returnLse) || (!isNhd && TensorInspector::IsContiguouslyPacked(lse));
    const aclTensor *targetLse = (returnLse && directWriteLse)
        ? lse
        : executionOwner->AllocTensor(expectedLseShape, DataType::DT_FLOAT);

    if (!targetOut || !targetLse) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    const auto dispatchStatus = l0op::DispatchSageAttention2Kernel(
        contigQ, contigK, contigV,
        static_cast<float>(smScale), smoothK, isCausal, returnLse,
        quantGranularCode, accumCode, smoothV,
        targetOut, targetLse, executionOwner.get());

    if (dispatchStatus != ACLNN_SUCCESS) {
        return dispatchStatus;
    }

    if (!directWriteOut && !l0op::ViewCopy(targetOut, canonOut, executionOwner.get())) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    if (returnLse && !directWriteLse && !l0op::ViewCopy(targetLse, lse, executionOwner.get())) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    *workspaceSize = executionOwner->GetWorkspaceSize();
    executionOwner.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSageAttention2(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    if (!executor) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    L2_DFX_PHASE_2(aclnnSageAttention2);
    return ::CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

} // extern "C"
