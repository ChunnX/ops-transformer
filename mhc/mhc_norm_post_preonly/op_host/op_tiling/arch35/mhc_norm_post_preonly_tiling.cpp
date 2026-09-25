/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mhc_norm_post_preonly_tiling.h"

#include <algorithm>
#include <cstdint>
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "log/log.h"

namespace optiling {

using namespace Ops::Transformer::OpTiling;

namespace {
constexpr uint32_t HOST_N_MAX = 4U;
constexpr uint32_t HOST_D_TILE_MAX = 2560U;
constexpr uint32_t HOST_MAX_TOKENS = 1024U * 1024U;
constexpr uint32_t NUM_TWO = 2U;
constexpr uint32_t PARTIAL_SLOT_BYTES = 32U;
constexpr uint32_t PARTIAL_SLOTS_PER_TOKEN = 2U;
constexpr uint32_t DEFAULT_TILING_PRIORITY = 1000U;
constexpr int32_t DUAL_SIBLING_SCHEDULE_MODE = 1;
constexpr size_t PRE_MIX_INPUT_INDEX = 7U;
constexpr size_t GAMMA_0_INPUT_INDEX = 8U;
constexpr size_t GAMMA_1_INPUT_INDEX = 9U;
constexpr size_t GAMMA_2_INPUT_INDEX = 10U;
constexpr size_t H_IN_PRIME_OUTPUT_INDEX = 0U;
constexpr size_t X_2_OUT_OUTPUT_INDEX = 1U;
constexpr size_t H_IN_F32_OUTPUT_INDEX = 2U;

constexpr uint64_t TILING_KEY_BIT_USE_RMS_MID = 1ULL << 0;
constexpr uint64_t TILING_KEY_BIT_RETURN_H_IN_F32 = 1ULL << 1;
constexpr uint64_t TILING_KEY_BIT_DUAL_SIBLING_AIV = 1ULL << 2;

// TilingKey三位编码：[dualSibling, returnHInF32, useRmsMid]。
// 000=单AIV/4op/不返回FP32，001=单AIV/5op/不返回FP32；
// 010=单AIV/4op/返回FP32，  011=单AIV/5op/返回FP32；
// 100=双sibling/4op/不返回FP32，101=双sibling/5op/不返回FP32；
// 110=双sibling/4op/返回FP32，  111=双sibling/5op/返回FP32。

bool IsSupportedD(uint32_t embedDim)
{
    return embedDim == 2560U || embedDim == 5120U || embedDim == 6144U || embedDim == 7168U;
}
} // namespace

REGISTER_OPS_TILING_TEMPLATE(MhcNormPostPreonly, MhcNormPostPreonlyTiling, DEFAULT_TILING_PRIORITY);

ge::graphStatus MhcNormPostPreonlyTiling::GetPlatformInfo()
{
    const auto *compileInfo = context_->GetCompileInfo<MhcNormPostPreonlyCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context_->GetNodeName(), "get compile info failed"),
                return ge::GRAPH_FAILED);

    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_->GetNodeName(), "get platform info failed"),
                return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);

    uint64_t aivNum = compileInfo->aivNum != 0U ? compileInfo->aivNum : ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aivNum == 0U || aivNum > UINT32_MAX,
                OP_LOGE(context_->GetNodeName(), "invalid AIV core count: %lu", aivNum), return ge::GRAPH_FAILED);
    coreNumAiv_ = static_cast<uint32_t>(aivNum);
    systemWorkspaceSize_ = static_cast<size_t>(compileInfo->systemWorkspaceSize);
    if (systemWorkspaceSize_ == 0U) {
        systemWorkspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcNormPostPreonlyTiling::CheckInputDtype() const
{
    auto hOutDesc = context_->GetInputDesc(0);
    auto residualDesc = context_->GetInputDesc(1);
    OP_CHECK_IF(hOutDesc == nullptr || residualDesc == nullptr,
                OP_LOGE(context_->GetNodeName(), "h_out or residual descriptor is null"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(hOutDesc->GetDataType() != residualDesc->GetDataType(),
                OP_LOGE(context_->GetNodeName(), "h_out and residual must have the same dtype"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(hOutDesc->GetDataType() != ge::DT_BF16 && hOutDesc->GetDataType() != ge::DT_FLOAT16,
                OP_LOGE(context_->GetNodeName(), "h_out and residual dtype must be BFLOAT16 or FLOAT16"),
                return ge::GRAPH_FAILED);

    constexpr size_t fp32InputIndices[] = {
        2U, 3U, 4U, 5U, 6U, GAMMA_0_INPUT_INDEX, GAMMA_1_INPUT_INDEX, GAMMA_2_INPUT_INDEX};
    constexpr const char *fp32InputNames[] = {"h_post", "h_res",   "phi",     "alpha",
                                              "bias",   "gamma_0", "gamma_1", "gamma_2"};
    for (size_t i = 0; i < sizeof(fp32InputIndices) / sizeof(fp32InputIndices[0]); ++i) {
        auto desc = context_->GetOptionalInputDesc(fp32InputIndices[i]);
        OP_CHECK_IF(desc != nullptr && desc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE(context_->GetNodeName(), "%s dtype must be FLOAT32", fp32InputNames[i]),
                    return ge::GRAPH_FAILED);
    }

    auto hInPrimeDesc = context_->GetOutputDesc(H_IN_PRIME_OUTPUT_INDEX);
    auto x2OutDesc = context_->GetOutputDesc(X_2_OUT_OUTPUT_INDEX);
    OP_CHECK_IF(hInPrimeDesc == nullptr || x2OutDesc == nullptr,
                OP_LOGE(context_->GetNodeName(), "required output descriptor is null"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        hInPrimeDesc->GetDataType() != hOutDesc->GetDataType() || x2OutDesc->GetDataType() != hOutDesc->GetDataType(),
        OP_LOGE(context_->GetNodeName(), "h_in_prime and x_2_out dtype must match h_out"), return ge::GRAPH_FAILED);
    auto hInF32Desc = context_->GetOutputDesc(H_IN_F32_OUTPUT_INDEX);
    OP_CHECK_IF(hInF32Desc != nullptr && hInF32Desc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "h_in_f32 dtype must be FLOAT32"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcNormPostPreonlyTiling::CheckInputShape()
{
    auto hOutShape = context_->GetInputShape(0);
    auto residualShape = context_->GetInputShape(1);
    auto hPostShape = context_->GetInputShape(2);
    auto hResShape = context_->GetOptionalInputShape(3);
    auto phiShape = context_->GetInputShape(4);
    auto alphaShape = context_->GetInputShape(5);
    auto biasShape = context_->GetInputShape(6);
    auto gamma0Shape = context_->GetOptionalInputShape(GAMMA_0_INPUT_INDEX);
    // gamma_1 is required by the interface, but it is located after two optional inputs.
    // Runtime TilingContext exposes such trailing inputs through the optional-input accessor.
    auto gamma1Shape = context_->GetOptionalInputShape(GAMMA_1_INPUT_INDEX);
    auto gamma2Shape = context_->GetOptionalInputShape(GAMMA_2_INPUT_INDEX);
    OP_CHECK_IF(hOutShape == nullptr || residualShape == nullptr || hPostShape == nullptr || phiShape == nullptr ||
                    alphaShape == nullptr || biasShape == nullptr || gamma1Shape == nullptr,
                OP_LOGE(context_->GetNodeName(), "required input shape is null"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        context_->GetOptionalInputShape(PRE_MIX_INPUT_INDEX) != nullptr,
        OP_LOGE(context_->GetNodeName(), "pre_mix is an interface-reserved optional input and must be empty currently"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(gamma0Shape == nullptr,
                OP_LOGE(context_->GetNodeName(),
                        "gamma_0 is an interface-reserved optional input and must not be empty currently"),
                return ge::GRAPH_FAILED);

    const auto &hOut = hOutShape->GetStorageShape();
    const auto &residual = residualShape->GetStorageShape();
    const auto &hPost = hPostShape->GetStorageShape();
    const auto &phi = phiShape->GetStorageShape();
    const auto &alpha = alphaShape->GetStorageShape();
    const auto &bias = biasShape->GetStorageShape();
    const auto &gamma0 = gamma0Shape->GetStorageShape();
    const auto &gamma1 = gamma1Shape->GetStorageShape();
    const size_t hOutDimNum = hOut.GetDimNum();
    const size_t residualDimNum = residual.GetDimNum();

    OP_CHECK_IF(hOutDimNum != 2U && hOutDimNum != 3U,
                OP_LOGE(context_->GetNodeName(), "h_out must be 2D [T,D] or 3D [B,S,D], got %zuD", hOutDimNum),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(residualDimNum != hOutDimNum + 1U,
                OP_LOGE(context_->GetNodeName(), "residual.ndim (%zu) must be h_out.ndim (%zu) + 1", residualDimNum,
                        hOutDimNum),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(hPost.GetDimNum() != 2U && hPost.GetDimNum() != 3U,
                OP_LOGE(context_->GetNodeName(), "h_post must be 2D [T,N] or 3D [B,S,N], got %zuD", hPost.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(hResShape == nullptr,
                OP_LOGE(context_->GetNodeName(),
                        "h_res is an interface-reserved optional input and must not be empty currently"),
                return ge::GRAPH_FAILED);
    const auto &hRes = hResShape->GetStorageShape();
    OP_CHECK_IF(
        hRes.GetDimNum() != 3U && hRes.GetDimNum() != 4U,
        OP_LOGE(context_->GetNodeName(), "h_res must be 3D [T,N,N] or 4D [B,S,N,N], got %zuD", hRes.GetDimNum()),
        return ge::GRAPH_FAILED);

    if (hOutDimNum == 2U) {
        OP_CHECK_IF(hOut.GetDim(0) <= 0 || hOut.GetDim(1) <= 0,
                    OP_LOGE(context_->GetNodeName(), "h_out dimensions must be positive"), return ge::GRAPH_FAILED);
        totalTokens_ = static_cast<uint32_t>(hOut.GetDim(0));
        embedDim_ = static_cast<uint32_t>(hOut.GetDim(1));
        headCount_ = static_cast<uint32_t>(residual.GetDim(1));
    } else {
        OP_CHECK_IF(hOut.GetDim(0) <= 0 || hOut.GetDim(1) <= 0 || hOut.GetDim(NUM_TWO) <= 0,
                    OP_LOGE(context_->GetNodeName(), "h_out dimensions must be positive"), return ge::GRAPH_FAILED);
        const uint64_t flattenedTokens = static_cast<uint64_t>(hOut.GetDim(0)) * hOut.GetDim(1);
        OP_CHECK_IF(flattenedTokens > HOST_MAX_TOKENS,
                    OP_LOGE(context_->GetNodeName(), "B*S must not exceed %u", HOST_MAX_TOKENS),
                    return ge::GRAPH_FAILED);
        totalTokens_ = static_cast<uint32_t>(flattenedTokens);
        embedDim_ = static_cast<uint32_t>(hOut.GetDim(NUM_TWO));
        headCount_ = static_cast<uint32_t>(residual.GetDim(NUM_TWO));
    }

    OP_CHECK_IF(phi.GetDimNum() != 2U,
                OP_LOGE(context_->GetNodeName(), "phi must be 2D [N, N*D], got %zuD", phi.GetDimNum()),
                return ge::GRAPH_FAILED);
    const uint32_t phiRows = static_cast<uint32_t>(phi.GetDim(0));
    phiCols_ = static_cast<uint32_t>(phi.GetDim(1));

    OP_CHECK_IF(
        totalTokens_ == 0U || totalTokens_ > HOST_MAX_TOKENS,
        OP_LOGE(context_->GetNodeName(), "totalTokens (B*S) must be in [1, %u], got %u", HOST_MAX_TOKENS, totalTokens_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(headCount_ != HOST_N_MAX,
                OP_LOGE(context_->GetNodeName(), "headCount (N) must be %u, got %u", HOST_N_MAX, headCount_),
                return ge::GRAPH_FAILED);

    const int64_t tokenDim0 = hOut.GetDim(0);
    const int64_t tokenDim1 = hOutDimNum == 3U ? hOut.GetDim(1) : 0;
    if (hOutDimNum == 2U) {
        OP_CHECK_IF(residual.GetDimNum() != 3U || residual.GetDim(0) != tokenDim0 || residual.GetDim(1) != HOST_N_MAX ||
                        residual.GetDim(2) != embedDim_,
                    OP_LOGE(context_->GetNodeName(), "residual must be [T,4,D] matching h_out"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(hPost.GetDimNum() != 2U || hPost.GetDim(0) != tokenDim0 || hPost.GetDim(1) != HOST_N_MAX,
                    OP_LOGE(context_->GetNodeName(), "h_post must be [T,4] matching h_out"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(hRes.GetDimNum() != 3U || hRes.GetDim(0) != tokenDim0 || hRes.GetDim(1) != HOST_N_MAX ||
                        hRes.GetDim(2) != HOST_N_MAX,
                    OP_LOGE(context_->GetNodeName(), "h_res must be [T,4,4] matching h_out"), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(residual.GetDimNum() != 4U || residual.GetDim(0) != tokenDim0 || residual.GetDim(1) != tokenDim1 ||
                        residual.GetDim(2) != HOST_N_MAX || residual.GetDim(3) != embedDim_,
                    OP_LOGE(context_->GetNodeName(), "residual must be [B,S,4,D] matching h_out"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(hPost.GetDimNum() != 3U || hPost.GetDim(0) != tokenDim0 || hPost.GetDim(1) != tokenDim1 ||
                        hPost.GetDim(2) != HOST_N_MAX,
                    OP_LOGE(context_->GetNodeName(), "h_post must be [B,S,4] matching h_out"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(hRes.GetDimNum() != 4U || hRes.GetDim(0) != tokenDim0 || hRes.GetDim(1) != tokenDim1 ||
                        hRes.GetDim(2) != HOST_N_MAX || hRes.GetDim(3) != HOST_N_MAX,
                    OP_LOGE(context_->GetNodeName(), "h_res must be [B,S,4,4] matching h_out"),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(alpha.GetDimNum() != 1U || alpha.GetDim(0) != 1, OP_LOGE(context_->GetNodeName(), "alpha must be [1]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(bias.GetDimNum() != 1U || bias.GetDim(0) != HOST_N_MAX,
                OP_LOGE(context_->GetNodeName(), "bias must be [4]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(gamma0.GetDimNum() != 1U || gamma0.GetDim(0) != embedDim_ || gamma1.GetDimNum() != 1U ||
                    gamma1.GetDim(0) != embedDim_,
                OP_LOGE(context_->GetNodeName(),
                        "gamma_0 and gamma_1 must both be [D=%u], got gamma_0 rank=%zu dim0=%ld, "
                        "gamma_1 rank=%zu dim0=%ld",
                        embedDim_, gamma0.GetDimNum(), gamma0.GetDimNum() == 0U ? -1L : gamma0.GetDim(0),
                        gamma1.GetDimNum(), gamma1.GetDimNum() == 0U ? -1L : gamma1.GetDim(0)),
                return ge::GRAPH_FAILED);
    if (gamma2Shape != nullptr) {
        const auto &gamma2 = gamma2Shape->GetStorageShape();
        OP_CHECK_IF(gamma2.GetDimNum() != 1U || gamma2.GetDim(0) != static_cast<int64_t>(HOST_N_MAX * embedDim_),
                    OP_LOGE(context_->GetNodeName(), "gamma_2 must be [4*D] when provided"), return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(
        !IsSupportedD(embedDim_),
        OP_LOGE(context_->GetNodeName(), "embedDim (D) must be one of 2560, 5120, 6144, or 7168, got %u", embedDim_),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(phiRows != headCount_ || phiCols_ != headCount_ * embedDim_,
                OP_LOGE(context_->GetNodeName(), "phi shape invalid: expected [%u, %u], got [%u, %u]", headCount_,
                        headCount_ * embedDim_, phiRows, phiCols_),
                return ge::GRAPH_FAILED);

    auto hInPrimeShape = context_->GetOutputShape(H_IN_PRIME_OUTPUT_INDEX);
    auto x2OutShape = context_->GetOutputShape(X_2_OUT_OUTPUT_INDEX);
    OP_CHECK_IF(hInPrimeShape == nullptr || x2OutShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "required output shape is null"), return ge::GRAPH_FAILED);
    const auto &hInPrime = hInPrimeShape->GetOriginShape();
    const auto &x2Out = x2OutShape->GetOriginShape();
    // Dynamic launch flattens preallocated output shapes before tiling. Validate their element counts here;
    // ACLNN validates the original ranks/dimensions and rejects the interface-reserved h_pre before launch.
    OP_CHECK_IF(hInPrime.GetShapeSize() != hOut.GetShapeSize(),
                OP_LOGE(context_->GetNodeName(), "h_in_prime element count must match h_out"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(x2Out.GetShapeSize() != residual.GetShapeSize(),
                OP_LOGE(context_->GetNodeName(), "x_2_out element count must match residual"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcNormPostPreonlyTiling::CheckAttrs()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        const float *normEps = attrs->GetAttrPointer<float>(0);
        const float *mhcPreNormEps = attrs->GetAttrPointer<float>(1);
        const float *hcEps = attrs->GetAttrPointer<float>(2);
        const bool *returnHInF32 = attrs->GetBool(3);
        normEps_ = normEps != nullptr ? *normEps : normEps_;
        mhcPreNormEps_ = mhcPreNormEps != nullptr ? *mhcPreNormEps : mhcPreNormEps_;
        hcEps_ = hcEps != nullptr ? *hcEps : hcEps_;
        returnHInF32_ = returnHInF32 != nullptr ? *returnHInF32 : returnHInF32_;
    }

    OP_CHECK_IF(normEps_ < 0.0F, OP_LOGE(context_->GetNodeName(), "norm_eps must be >= 0, got %e", normEps_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(mhcPreNormEps_ < 0.0F,
                OP_LOGE(context_->GetNodeName(), "mhc_pre_norm_eps must be >= 0, got %e", mhcPreNormEps_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(hcEps_ < 0.0F, OP_LOGE(context_->GetNodeName(), "hc_eps must be >= 0, got %e", hcEps_),
                return ge::GRAPH_FAILED);

    useRmsMid_ = context_->GetOptionalInputShape(GAMMA_2_INPUT_INDEX) != nullptr;
    auto hInF32Shape = context_->GetOutputShape(H_IN_F32_OUTPUT_INDEX);
    if (returnHInF32_) {
        OP_CHECK_IF(hInF32Shape == nullptr,
                    OP_LOGE(context_->GetNodeName(), "h_in_f32 must be present when return_h_in_f32 is true"),
                    return ge::GRAPH_FAILED);
        const auto &hOut = context_->GetInputShape(0)->GetStorageShape();
        const auto &hInF32 = hInF32Shape->GetOriginShape();
        OP_CHECK_IF(hInF32.GetShapeSize() != hOut.GetShapeSize(),
                    OP_LOGE(context_->GetNodeName(), "h_in_f32 element count must match h_out"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcNormPostPreonlyTiling::GetShapeAttrsInfo()
{
    if (CheckInputDtype() != ge::GRAPH_SUCCESS || CheckInputShape() != ge::GRAPH_SUCCESS ||
        CheckAttrs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void MhcNormPostPreonlyTiling::FillTilingData()
{
    tilingData_.set_totalTokens(totalTokens_);
    tilingData_.set_headCount(headCount_);
    tilingData_.set_embedDim(embedDim_);
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_rowsPerCore(rowsPerCore_);
    tilingData_.set_rowsPerCoreSp(rowsPerCoreSp_);
    tilingData_.set_blockPivot(blockPivot_);
    tilingData_.set_dimTile(dimTile_);
    tilingData_.set_dimLoop(dimLoop_);
    tilingData_.set_lastDimTile(lastDimTile_);
    tilingData_.set_phiCols(phiCols_);
    tilingData_.set_normEps(normEps_);
    tilingData_.set_mhcPreNormEps(mhcPreNormEps_);
    tilingData_.set_hcEps(hcEps_);
    tilingData_.set_invD(1.0F / static_cast<float>(embedDim_));
    tilingData_.set_invK(1.0F / static_cast<float>(headCount_ * embedDim_));
}

ge::graphStatus MhcNormPostPreonlyTiling::DoOpTiling()
{
    dimTile_ = std::min(embedDim_, HOST_D_TILE_MAX);
    lastDimTile_ = embedDim_ % dimTile_;
    dimLoop_ = embedDim_ / dimTile_ + (lastDimTile_ != 0U ? 1U : 0U);

    enableDualSibling_ = totalTokens_ <= coreNumAiv_ / 2U;
    usedCoreNum_ = std::min(totalTokens_, enableDualSibling_ ? coreNumAiv_ / 2U : coreNumAiv_);
    rowsPerCore_ = totalTokens_ / usedCoreNum_;
    rowsPerCoreSp_ = rowsPerCore_ + 1U;
    blockPivot_ = totalTokens_ % usedCoreNum_;

    FillTilingData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t MhcNormPostPreonlyTiling::GetTilingKey() const
{
    return (useRmsMid_ ? TILING_KEY_BIT_USE_RMS_MID : 0ULL) | (returnHInF32_ ? TILING_KEY_BIT_RETURN_H_IN_F32 : 0ULL) |
           (enableDualSibling_ ? TILING_KEY_BIT_DUAL_SIBLING_AIV : 0ULL);
}

ge::graphStatus MhcNormPostPreonlyTiling::GetWorkspaceSize()
{
    workspaceSize_ = systemWorkspaceSize_;
    if (enableDualSibling_) {
        workspaceSize_ += static_cast<size_t>(totalTokens_) * PARTIAL_SLOTS_PER_TOKEN * PARTIAL_SLOT_BYTES;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcNormPostPreonlyTiling::PostTiling()
{
    OP_CHECK_IF(
        tilingData_.GetDataSize() % sizeof(uint64_t) != 0U,
        OP_LOGE(context_->GetNodeName(), "tiling data size[%zu] is not aligned to 8", tilingData_.GetDataSize()),
        return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(usedCoreNum_);
    if (enableDualSibling_) {
        context_->SetScheduleMode(DUAL_SIBLING_SCHEDULE_MODE);
    }

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null"),
                return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

void MhcNormPostPreonlyTiling::PrintTilingData() const
{
    OP_LOGD(context_->GetNodeName(),
            "tokens=%u, N=%u, D=%u, blockDim=%u, rows=[%u,%u], pivot=%u, dimTile=%u, dimLoop=%u, "
            "dualSibling=%s, useRmsMid=%s, returnHInF32=%s",
            totalTokens_, headCount_, embedDim_, usedCoreNum_, rowsPerCore_, rowsPerCoreSp_, blockPivot_, dimTile_,
            dimLoop_, enableDualSibling_ ? "true" : "false", useRmsMid_ ? "true" : "false",
            returnHInF32_ ? "true" : "false");
}

static ge::graphStatus TilingMhcNormPostPreonly(gert::TilingContext *context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForMhcNormPostPreonly(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("MhcNormPostPreonly", "context is null"),
                return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "platformInfo is null"),
                return ge::GRAPH_FAILED);
    auto compileInfo = context->GetCompiledInfo<MhcNormPostPreonlyCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "compileInfo is null"),
                return ge::GRAPH_FAILED);

    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    compileInfo->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->systemWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MhcNormPostPreonly)
    .Tiling(TilingMhcNormPostPreonly)
    .TilingParse<MhcNormPostPreonlyCompileInfo>(TilingPrepareForMhcNormPostPreonly);
} // namespace optiling
