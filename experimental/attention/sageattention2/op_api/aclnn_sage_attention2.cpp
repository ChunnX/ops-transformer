/**
 * canonical aclnn 两段式（对齐仓内规范）：
 * 一段式 L2_DFX_PHASE_1 → CREATE_EXECUTOR → l0op::Contiguous → AllocTensor →
 * ADD_TO_LAUNCHER_LIST_AICORE → ViewCopy 回写 → GetWorkspaceSize/ReleaseTo；
 * 二段式 L2_DFX_PHASE_2 → CommonOpExecutorRun。
 * 支持 HND 与 NHD 原生布局直通，无需转置。
 * 两段式 C 接口唯一交付实现（aclnn/ 目录仅承载闭式 Host Tiling 计算核心与
 * 字符串属性编码协议，不含第二套两段式实现）。
 */

#include "aclnn_sage_attention2.h"

#include <cmath>
#include <cstring>

#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(SageAttention2);

aclnnStatus BuildSage(const aclTensor *q, const aclTensor *k, const aclTensor *v,
                      float scale, const char *layout, bool causal, const char *gran,
                      const char *accum, bool smoothK, bool smoothV,
                      const aclTensor *out, const aclTensor *lse, aclOpExecutor *executor)
{
    return ADD_TO_LAUNCHER_LIST_AICORE(
        SageAttention2, OP_INPUT(q, k, v), OP_OUTPUT(out, lse),
        OP_ATTR(layout, causal, gran, scale, accum, smoothK, smoothV));
}
} // namespace l0op

namespace {
bool TensorValid(const aclTensor *t)
{
    if (t == nullptr || t->GetViewShape().GetDimNum() != 4) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        if (t->GetViewShape()[i] <= 0 || t->GetViewShape()[i] >= UINT32_MAX - 64) {
            return false;
        }
    }
    return t->GetDataType() == DataType::DT_FLOAT16 || t->GetDataType() == DataType::DT_BF16;
}
} // namespace

extern "C" aclnnStatus aclnnSageAttention2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const char *tensorLayout, bool isCausal, const char *qkQuantGran, double smScale,
    const char *pvAccumDtype, bool smoothK, bool smoothV,
    const aclTensor *attentionOut, const aclTensor *softmaxLse,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (workspaceSize == nullptr || executor == nullptr || query == nullptr || key == nullptr ||
        value == nullptr || attentionOut == nullptr || tensorLayout == nullptr ||
        qkQuantGran == nullptr || pvAccumDtype == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    *workspaceSize = 0;
    *executor = nullptr;

    const bool returnLse = (softmaxLse != nullptr);

    // 任务书口径：per_thread/per_warp 为合法算法粒度，非法取值报错（不静默降级）
    if (std::strcmp(qkQuantGran, "per_thread") != 0 && std::strcmp(qkQuantGran, "per_warp") != 0) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!std::isfinite(smScale)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 任务书口径：布局仅 HND/NHD，非法取值报错（不静默降级）
    const bool nhd = (std::strcmp(tensorLayout, "NHD") == 0);
    if (!nhd && std::strcmp(tensorLayout, "HND") != 0) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (std::strcmp(pvAccumDtype, "fp32") != 0 && std::strcmp(pvAccumDtype, "fp32+fp32") != 0 &&
        std::strcmp(pvAccumDtype, "fp32+fp16") != 0) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!TensorValid(query) || !TensorValid(key) || !TensorValid(value) || !TensorValid(attentionOut) ||
        query->GetDataType() != key->GetDataType() || query->GetDataType() != value->GetDataType() ||
        query->GetDataType() != attentionOut->GetDataType()) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (GetCurrentPlatformInfo().GetSocLongVersion().find("Ascend950") != 0) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    const int hq_idx = nhd ? 2 : 1;
    const int sq_idx = nhd ? 1 : 2;
    const auto &qShape = query->GetViewShape();
    const auto &kShape = key->GetViewShape();
    const auto &vShape = value->GetViewShape();
    const auto &outShape = attentionOut->GetViewShape();

    if (outShape != qShape || kShape != vShape || qShape[0] != kShape[0] || qShape[3] != kShape[3] ||
        qShape[3] > 128 || qShape[hq_idx] % kShape[hq_idx] != 0 ||
        static_cast<uint64_t>(kShape[0]) * kShape[hq_idx] >= UINT32_MAX ||
        (isCausal && qShape[sq_idx] != kShape[sq_idx])) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    Shape lseShape{qShape[0], qShape[hq_idx], qShape[sq_idx]};
    if (returnLse && (softmaxLse->GetDataType() != DataType::DT_FLOAT ||
                      softmaxLse->GetViewShape() != lseShape)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    L2_DFX_PHASE_1(aclnnSageAttention2,
                   DFX_IN(query, key, value,
                          tensorLayout, isCausal, qkQuantGran, smScale, pvAccumDtype,
                          smoothK, smoothV),
                   DFX_OUT(attentionOut, softmaxLse));
    auto owner = CREATE_EXECUTOR();
    if (owner.get() == nullptr) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }

    auto qc = l0op::Contiguous(query, owner.get());
    auto kc = l0op::Contiguous(key, owner.get());
    auto vc = l0op::Contiguous(value, owner.get());
    if (qc == nullptr || kc == nullptr || vc == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto outTensor = owner->AllocTensor(qShape, query->GetDataType());
    if (outTensor == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto lseTensor = owner->AllocTensor(lseShape, DataType::DT_FLOAT);
    if (lseTensor == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    const aclnnStatus ret = l0op::BuildSage(
        qc, kc, vc, static_cast<float>(smScale),
        tensorLayout, isCausal, qkQuantGran, pvAccumDtype, smoothK, smoothV,
        outTensor, lseTensor, owner.get());
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    if (!l0op::ViewCopy(outTensor, attentionOut, owner.get()) ||
        (returnLse && !l0op::ViewCopy(lseTensor, softmaxLse, owner.get()))) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    *workspaceSize = owner->GetWorkspaceSize();
    owner.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnSageAttention2(void *workspace, uint64_t workspaceSize,
                                           aclOpExecutor *executor, aclrtStream stream)
{
    if (executor == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    L2_DFX_PHASE_2(aclnnSageAttention2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
