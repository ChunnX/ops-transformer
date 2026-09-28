/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_sage_attention2.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
using namespace op;
namespace l0op {
OP_TYPE_REGISTER(SageAttention2);

aclnnStatus BuildSage(const aclTensor *q, const aclTensor *k, const aclTensor *v, float scale, bool smooth, bool causal,
                      bool returnLse, const aclTensor *out, const aclTensor *lse, aclOpExecutor *executor)
{
    return ADD_TO_LAUNCHER_LIST_AICORE(SageAttention2, OP_INPUT(q, k, v), OP_OUTPUT(out, lse),
                                       OP_ATTR(scale, smooth, causal, returnLse));
}
} // namespace l0op
namespace {
constexpr int64_t MAX_HEAD_DIM = 128;

const aclTensor *Canonical(const aclTensor *t, bool nhd, aclOpExecutor *executor)
{
    if (!nhd) {
        return t;
    }
    auto shape = t->GetViewShape();
    auto strides = t->GetViewStrides();
    std::swap(shape[1], shape[2]);
    std::swap(strides[1], strides[2]);
    return executor->CreateView(t, shape, t->GetStorageShape(), strides, t->GetViewOffset());
}
bool TensorValid(const aclTensor *t)
{
    if (!t || t->GetViewShape().GetDimNum() != 4) {
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
extern "C" aclnnStatus aclnnSageAttention2GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v,
                                                           const char *tensorLayout, bool isCausal,
                                                           const char *qkQuantGran, double smScale,
                                                           const char *pvAccumDtype, bool smoothK, bool smoothV,
                                                           bool returnLse, const aclTensor *out, const aclTensor *lse,
                                                           uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (!workspaceSize || !executor || !q || !k || !v || !out || !tensorLayout || !qkQuantGran || (returnLse && !lse)) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    *workspaceSize = 0;
    *executor = nullptr;
    const bool perToken = std::strcmp(qkQuantGran, "per_token") == 0;
    if ((!perToken && std::strcmp(qkQuantGran, "per_thread") != 0 && std::strcmp(qkQuantGran, "per_warp") != 0) ||
        !std::isfinite(smScale) ||
        std::abs(smScale) > static_cast<double>(std::numeric_limits<float>::max())) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!pvAccumDtype || (std::strcmp(pvAccumDtype, "fp32+fp16") != 0 &&
                          std::strcmp(pvAccumDtype, "fp32+fp32") != 0 &&
                          std::strcmp(pvAccumDtype, "fp32") != 0)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    bool nhd = std::strcmp(tensorLayout, "NHD") == 0;
    if (!nhd && std::strcmp(tensorLayout, "HND") != 0) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (smoothV && (std::strcmp(pvAccumDtype, "fp32+fp16") == 0 || std::strcmp(pvAccumDtype, "fp32+fp32") == 0))
        OP_LOGW("smooth_v is ignored: SageAttention2 uses MXFP8 PV without value centering.");
    if (!TensorValid(q) || !TensorValid(k) || !TensorValid(v) || !TensorValid(out) ||
        q->GetDataType() != k->GetDataType() || q->GetDataType() != v->GetDataType() ||
        q->GetDataType() != out->GetDataType() || q->GetViewShape() != out->GetViewShape()) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (GetCurrentPlatformInfo().GetSocLongVersion().find("Ascend950") != 0) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    L2_DFX_PHASE_1(aclnnSageAttention2,
                   DFX_IN(q, k, v, tensorLayout, isCausal, qkQuantGran, smScale, pvAccumDtype,
                          smoothK, smoothV, returnLse),
                   DFX_OUT(out, lse));
    auto owner = CREATE_EXECUTOR();
    if (!owner.get()) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    auto qc = Canonical(q, nhd, owner.get());
    auto kc = Canonical(k, nhd, owner.get());
    auto vc = Canonical(v, nhd, owner.get());
    auto oc = Canonical(out, nhd, owner.get());
    if (!qc || !kc || !vc || !oc) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    const auto &a = qc->GetViewShape();
    const auto &b = kc->GetViewShape();
    if (b != vc->GetViewShape() || a[0] != b[0] || a[3] != b[3] || a[3] > MAX_HEAD_DIM || a[1] % b[1] ||
        uint64_t(b[0]) * b[1] >= UINT32_MAX || (isCausal && a[2] != b[2])) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    Shape lseShape{a[0], a[1], a[2]};
    if (returnLse && (lse->GetDataType() != DataType::DT_FLOAT || lse->GetViewShape() != lseShape)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    qc = l0op::Contiguous(qc, owner.get());
    kc = l0op::Contiguous(kc, owner.get());
    vc = l0op::Contiguous(vc, owner.get());
    if (!qc || !kc || !vc) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto result = owner->AllocTensor(a, q->GetDataType());
    auto resultLse = owner->AllocTensor(lseShape, DataType::DT_FLOAT);
    if (!result || !resultLse) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto ret = l0op::BuildSage(qc, kc, vc, static_cast<float>(smScale), smoothK, isCausal, returnLse, result, resultLse,
                               owner.get());
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    if (!l0op::ViewCopy(result, oc, owner.get()) || (returnLse && !l0op::ViewCopy(resultLse, lse, owner.get()))) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    *workspaceSize = owner->GetWorkspaceSize();
    owner.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
extern "C" aclnnStatus aclnnSageAttention2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           aclrtStream stream)
{
    if (!executor) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    L2_DFX_PHASE_2(aclnnSageAttention2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
