/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * canonical aclnn 两段式（对齐仓内规范）：
 * 一段式 L2_DFX_PHASE_1 → CREATE_EXECUTOR → Canonical 视图（HND/NHD 互换）→
 * l0op::Contiguous → AllocTensor → ADD_TO_LAUNCHER_LIST_AICORE → ViewCopy 回写 →
 * GetWorkspaceSize/ReleaseTo；二段式 L2_DFX_PHASE_2 → CommonOpExecutorRun。
 * 与 aclnn/ 目录的移植版接口并存（aclnn/ 面向无 opdev 运行时的移植态与单测，
 * 本目录为仓内交付态规范路径）。
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
                      const aclTensor *keyMean, const aclTensor *lseCorr,
                      float scale, const char *layout, bool causal, const char *gran,
                      const char *accum, bool smoothK, bool smoothV,
                      const aclTensor *out, const aclTensor *lse, aclOpExecutor *executor)
{
    // 可选输入按非空实况逐一绑定（DEFECT-A5-2：原"两者同时非空才绑定"会静默
    // 丢弃单独传入的 key_mean）；OP_ATTR 与 def.cpp 的 7 属性注册
    // （tensor_layout/is_causal/qk_quant_gran/sm_scale/pv_accum_dtype/smooth_k/smooth_v）
    // 逐项一致——return_lse 经 softmaxLse 张量是否非空表达，不是注册属性，
    // return_lse 不入 OP_ATTR（由 lse 输出可选性承载），保持 7 项与 def 注册一致。
    if (keyMean != nullptr && lseCorr != nullptr) {
        return ADD_TO_LAUNCHER_LIST_AICORE(
            SageAttention2, OP_INPUT(q, k, v, keyMean, lseCorr), OP_OUTPUT(out, lse),
            OP_ATTR(layout, causal, gran, scale, accum, smoothK, smoothV));
    }
    if (keyMean != nullptr) {
        return ADD_TO_LAUNCHER_LIST_AICORE(
            SageAttention2, OP_INPUT(q, k, v, keyMean), OP_OUTPUT(out, lse),
            OP_ATTR(layout, causal, gran, scale, accum, smoothK, smoothV));
    }
    // lseCorr 单独非空时不经 OP_INPUT 下发（DEFECT-A5-8）：OP_INPUT 按算子原型
    // 输入序位置绑定，中间可选项不可跳位——4 参形态会把 lse_correction 落入
    // key_mean 槽（smooth_k=1 时核内按 [b,hkv,d_pad] fp32 误读为 km，数值/越界
    // 双重风险）。基线核对该通道仅指针承接、消费留给向量化 pass（op_kernel
    // lse_correction_gm_ 契约注释），此处与基线一致保持 3 参直落。
    return ADD_TO_LAUNCHER_LIST_AICORE(
        SageAttention2, OP_INPUT(q, k, v), OP_OUTPUT(out, lse),
        OP_ATTR(layout, causal, gran, scale, accum, smoothK, smoothV));
}
} // namespace l0op

namespace {
// HND/NHD 的 1、2 维互换视图（BNSD<->BSND 同形互换），storage 不变
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
    const aclTensor *keyMeanOptional, const aclTensor *lseCorrectionOptional,
    const char *tensorLayout, bool isCausal, const char *qkQuantGran, double smScale,
    const char *pvAccumDtype, bool smoothK, bool smoothV, bool returnLse,
    const aclTensor *attentionOut, const aclTensor *softmaxLse,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (workspaceSize == nullptr || executor == nullptr || query == nullptr || key == nullptr ||
        value == nullptr || attentionOut == nullptr || tensorLayout == nullptr ||
        qkQuantGran == nullptr || pvAccumDtype == nullptr || (returnLse && softmaxLse == nullptr)) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    *workspaceSize = 0;
    *executor = nullptr;

    // 任务书口径：per_thread/per_warp 为合法算法粒度，非法取值报错（不静默降级）
    if (std::strcmp(qkQuantGran, "per_thread") != 0 && std::strcmp(qkQuantGran, "per_warp") != 0) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!std::isfinite(smScale)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 布局同义词域与 def.cpp/proto 对齐：BNSD~HND、BSND~NHD（DEFECT-A5-4，
    // def 默认值即 "BNSD"，仅认 HND/NHD 会把缺省调用拒之门外）
    const char *layout_canon = tensorLayout;
    if (std::strcmp(layout_canon, "BNSD") == 0) {
        layout_canon = "HND";
    } else if (std::strcmp(layout_canon, "BSND") == 0) {
        layout_canon = "NHD";
    }
    const bool nhd = (std::strcmp(layout_canon, "NHD") == 0);
    if (!nhd && std::strcmp(layout_canon, "HND") != 0) {
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

    L2_DFX_PHASE_1(aclnnSageAttention2,
                   DFX_IN(query, key, value, keyMeanOptional, lseCorrectionOptional,
                          tensorLayout, isCausal, qkQuantGran, smScale, pvAccumDtype,
                          smoothK, smoothV, returnLse),
                   DFX_OUT(attentionOut, softmaxLse));
    auto owner = CREATE_EXECUTOR();
    if (owner.get() == nullptr) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    auto qc = Canonical(query, nhd, owner.get());
    auto kc = Canonical(key, nhd, owner.get());
    auto vc = Canonical(value, nhd, owner.get());
    auto oc = Canonical(attentionOut, nhd, owner.get());
    if (qc == nullptr || kc == nullptr || vc == nullptr || oc == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    const auto &qShape = qc->GetViewShape();
    const auto &kShape = kc->GetViewShape();
    if (kShape != vc->GetViewShape() || qShape[0] != kShape[0] || qShape[3] != kShape[3] ||
        qShape[3] > 128 || qShape[1] % kShape[1] != 0 ||
        static_cast<uint64_t>(kShape[0]) * kShape[1] >= UINT32_MAX ||
        (isCausal && qShape[2] != kShape[2])) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    Shape lseShape{qShape[0], qShape[1], qShape[2]};
    if (returnLse && (softmaxLse->GetDataType() != DataType::DT_FLOAT ||
                      softmaxLse->GetViewShape() != lseShape)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    qc = l0op::Contiguous(qc, owner.get());
    kc = l0op::Contiguous(kc, owner.get());
    vc = l0op::Contiguous(vc, owner.get());
    if (qc == nullptr || kc == nullptr || vc == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto outTensor = owner->AllocTensor(qShape, query->GetDataType());
    auto lseTensor = owner->AllocTensor(lseShape, DataType::DT_FLOAT);
    if (outTensor == nullptr || lseTensor == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    const aclnnStatus ret = l0op::BuildSage(
        qc, kc, vc, keyMeanOptional, lseCorrectionOptional, static_cast<float>(smScale),
        layout_canon, isCausal, qkQuantGran, pvAccumDtype, smoothK, smoothV,
        outTensor, lseTensor, owner.get());
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    if (!l0op::ViewCopy(outTensor, oc, owner.get()) ||
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
