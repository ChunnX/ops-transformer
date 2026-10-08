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
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"

namespace optiling {

// Host侧常量，与Kernel公共头文件保持一致
constexpr uint32_t HOST_N_MAX = 4;
constexpr uint32_t HOST_D_TILE_MAX = 2560;
constexpr uint32_t HOST_D_ALIGN_ELEM = 16;
constexpr uint32_t HOST_MAX_TOKENS = 1024 * 1024; // B*S 上限
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t PARTIAL_SLOT_BYTES = 32;
constexpr uint32_t PARTIAL_SLOTS_PER_TOKEN = 2;

constexpr uint64_t TILING_KEY_USE_RMS_MID = 1ULL << 0;
constexpr uint64_t TILING_KEY_RETURN_H_IN_F32 = 1ULL << 1;
constexpr uint64_t TILING_KEY_DUAL_SIBLING_AIV = 1ULL << 2;

static ge::graphStatus TilingMhcNormPostPreonly(gert::TilingContext *context)
{
    // 平台信息
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t coreNumAiv = ascendcPlatform.GetCoreNumAiv();
    if (coreNumAiv == 0)
        coreNumAiv = 1;

    // 获取形状，同时支持[B,S,D]/[T,D]和[B,S,N,D]/[T,N,D]
    auto h_out_shape = context->GetInputShape(0);         // [B,S,D] or [T,D]
    auto res_shape = context->GetInputShape(1);           // [B,S,N,D] or [T,N,D]
    auto h_post_shape = context->GetInputShape(2);        // [B,S,N] or [T,N]
    auto h_res_shape = context->GetOptionalInputShape(3); // [B,S,N,N] or [T,N,N]
    auto phi_shape = context->GetInputShape(4);           // [N, N*D]
    if (h_out_shape == nullptr || res_shape == nullptr || h_post_shape == nullptr || phi_shape == nullptr) {
        OP_LOGE("MhcNormPostPreonly", "GetInputShape is return nullptr");
        return ge::GRAPH_FAILED;
    }

    // h_out与residual必须使用相同低精度类型，避免Kernel按错误元素类型解释输入。
    auto hOutDesc = context->GetInputDesc(0);
    auto residualDesc = context->GetInputDesc(1);
    if (hOutDesc == nullptr || residualDesc == nullptr || hOutDesc->GetDataType() != residualDesc->GetDataType()) {
        OP_LOGE("MhcNormPostPreonly", "h_out and residual must have the same dtype");
        return ge::GRAPH_FAILED;
    }

    const auto &hs = h_out_shape->GetStorageShape();
    const auto &rs = res_shape->GetStorageShape();
    const auto &hps = h_post_shape->GetStorageShape();
    const auto &ps = phi_shape->GetStorageShape();

    // 格式一致性：residual维数必须比h_out多1
    size_t hDimNum = hs.GetDimNum();
    size_t rDimNum = rs.GetDimNum();
    if (hDimNum != 2U && hDimNum != 3U) {
        OP_LOGE("MhcNormPostPreonly", "h_out must be 2D [T,D] or 3D [B,S,D], got %zuD", hDimNum);
        return ge::GRAPH_FAILED;
    }
    if (rDimNum != hDimNum + 1U) {
        OP_LOGE("MhcNormPostPreonly", "residual.ndim (%zu) must be h_out.ndim (%zu) + 1", rDimNum, hDimNum);
        return ge::GRAPH_FAILED;
    }
    // h_post为二维[T,N]或三维[B,S,N]；h_res非空时为[T,N,N]或[B,S,N,N]
    size_t hpDimNum = hps.GetDimNum();
    if (hpDimNum != 2U && hpDimNum != 3U) {
        OP_LOGE("MhcNormPostPreonly", "h_post must be 2D [T,N] or 3D [B,S,N], got %zuD", hpDimNum);
        return ge::GRAPH_FAILED;
    }
    if (h_res_shape == nullptr) {
        OP_LOGE("MhcNormPostPreonly",
                "h_res is an interface-reserved optional input and must not be empty currently.");
        return ge::GRAPH_FAILED;
    }
    const auto &hrs = h_res_shape->GetStorageShape();
    size_t hrDimNum = hrs.GetDimNum();
    if (hrDimNum != 3U && hrDimNum != 4U) {
        OP_LOGE("MhcNormPostPreonly", "h_res must be 3D [T,N,N] or 4D [B,S,N,N], got %zuD", hrDimNum);
        return ge::GRAPH_FAILED;
    }

    uint32_t totalTokens;
    uint32_t D;
    uint32_t N;
    if (hDimNum == 2U) {
        // [T, D] format
        totalTokens = static_cast<uint32_t>(hs.GetDim(0));
        D = static_cast<uint32_t>(hs.GetDim(1));
        N = static_cast<uint32_t>(rs.GetDim(1)); // [T, N, D]
    } else {
        // [B, S, D] format
        uint32_t B = static_cast<uint32_t>(hs.GetDim(0));
        uint32_t S = static_cast<uint32_t>(hs.GetDim(1));
        D = static_cast<uint32_t>(hs.GetDim(NUM_TWO));
        N = static_cast<uint32_t>(rs.GetDim(NUM_TWO)); // [B, S, N, D]
        totalTokens = B * S;
    }

    if (ps.GetDimNum() != 2U) {
        OP_LOGE("MhcNormPostPreonly", "phi must be 2D [N, N*D], got %zuD", ps.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    uint32_t phiRows = static_cast<uint32_t>(ps.GetDim(0)); // N (pre-branch only)
    uint32_t phiCols = static_cast<uint32_t>(ps.GetDim(1)); // N*D

    // 参数校验
    if (totalTokens == 0 || totalTokens > HOST_MAX_TOKENS) {
        OP_LOGE("MhcNormPostPreonly", "totalTokens (B*S) must be in [1, %u], got %u", HOST_MAX_TOKENS,
                totalTokens);
        return ge::GRAPH_FAILED;
    }
    if (N != HOST_N_MAX) {
        OP_LOGE("MhcNormPostPreonly", "headCount (N) must be %u, got %u", HOST_N_MAX, N);
        return ge::GRAPH_FAILED;
    }
    if (D == 0 || D % HOST_D_ALIGN_ELEM != 0 || D > UINT32_MAX / HOST_N_MAX) {
        OP_LOGE("MhcNormPostPreonly", "embedDim (D) must be positive and %u-element aligned, got %u",
                HOST_D_ALIGN_ELEM, D);
        return ge::GRAPH_FAILED;
    }
    if (phiRows != N || phiCols != N * D) {
        OP_LOGE("MhcNormPostPreonly", "phi shape invalid: expected [%u, %u], got [%u, %u]", N, N * D, phiRows,
                phiCols);
        return ge::GRAPH_FAILED;
    }

    // 属性均为可选，先设置默认值
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    float normEps = 1e-6F;
    float mhcPreNormEps = 1e-6F;
    float hcEps = 1e-6F;
    bool returnHInF32 = false;
    if (attrs != nullptr) {
        // op_host_aclnn生成的ACLNN封装按_def.cpp中的Float声明保存属性
        const float *eps0 = attrs->GetAttrPointer<float>(0);
        if (eps0 != nullptr) {
            normEps = *eps0;
        }
        const float *preEps = attrs->GetAttrPointer<float>(1);
        if (preEps != nullptr) {
            mhcPreNormEps = *preEps;
        }
        const float *eps1 = attrs->GetAttrPointer<float>(2);
        if (eps1 != nullptr) {
            hcEps = *eps1;
        }

        const bool *returnHInF32Ptr = attrs->GetBool(3);
        if (returnHInF32Ptr != nullptr) {
            returnHInF32 = *returnHInF32Ptr;
        }
    }

    // 校验数值稳定项
    if (normEps < 0.0F) {
        OP_LOGE("MhcNormPostPreonly", "norm_eps must be >= 0, got %e", normEps);
        return ge::GRAPH_FAILED;
    }
    if (hcEps < 0.0F) {
        OP_LOGE("MhcNormPostPreonly", "hc_eps must be >= 0, got %e", hcEps);
        return ge::GRAPH_FAILED;
    }
    if (mhcPreNormEps < 0.0F) {
        OP_LOGE("MhcNormPostPreonly", "mhc_pre_norm_eps must be >= 0, got %e", mhcPreNormEps);
        return ge::GRAPH_FAILED;
    }

    // 检查索引10处的可选gamma_2；非空时执行RMSNorm_mid
    auto gamma2_shape = context->GetOptionalInputShape(10);
    bool useRmsMid = (gamma2_shape != nullptr);

    // D轴分块必须在核映射前完成
    uint32_t dimTile = std::min(D, HOST_D_TILE_MAX);

    uint32_t lastDimTile = D % dimTile;
    uint32_t dimLoop = D / dimTile + (lastDimTile ? 1 : 0);

    // A5的MIX_AIC_1_2以一个物理AI Core为blockDim单位，并展开两个sibling AIV。
    // 对精度集覆盖的四档D分别实测双AIV收益，由tiling按已验证维度稳定路由。
    // 4op（无RMS_mid）与5op均走head切分双sibling；BF16/FP16均支持。
    const bool dualSiblingDim = D == 2560U || D == 5120U || D == 6144U || D == 7168U;
    const bool enableDualSibling =
        totalTokens == 4U && dualSiblingDim && !returnHInF32 &&
        (hOutDesc->GetDataType() == ge::DT_BF16 || hOutDesc->GetDataType() == ge::DT_FLOAT16);
    const uint32_t coresPerToken = enableDualSibling ? 2U : 1U;
    const uint32_t usedCoreNum = std::min(totalTokens, enableDualSibling ? (coreNumAiv / 2U) : coreNumAiv);
    uint32_t rowsPerCore = totalTokens / usedCoreNum;
    uint32_t remainTokens = totalTokens % usedCoreNum;
    uint32_t rowsPerCoreSp = rowsPerCore + 1;
    uint32_t blockPivot = remainTokens;

    // 填充Tiling数据
    MhcNormPostPreonlyTilingData tiling;
    tiling.set_totalTokens(totalTokens);
    tiling.set_headCount(N);
    tiling.set_embedDim(D);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_rowsPerCore(rowsPerCore);
    tiling.set_rowsPerCoreSp(rowsPerCoreSp);
    tiling.set_blockPivot(blockPivot);
    tiling.set_dimTile(dimTile);
    tiling.set_dimLoop(dimLoop);
    tiling.set_lastDimTile(lastDimTile);
    tiling.set_phiRows(phiRows);
    tiling.set_phiCols(phiCols);
    tiling.set_normEps(normEps);
    tiling.set_mhcPreNormEps(mhcPreNormEps);
    tiling.set_hcEps(hcEps);
    tiling.set_invD(1.0F / static_cast<float>(D));
    tiling.set_invK(1.0F / static_cast<float>(N * D));
    tiling.set_useRmsMid(useRmsMid ? 1U : 0U);
    tiling.set_coresPerToken(coresPerToken);
    tiling.set_returnHInF32(static_cast<int8_t>(returnHInF32));

    // CrossCore同步不占GM；双sibling路径每个token只分配一个32B partial1槽。
    size_t sysWsBytes = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        OP_LOGE("MhcNormPostPreonly", "GetWorkspaceSizes is nullptr");
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] =
        sysWsBytes + (enableDualSibling ? (size_t)totalTokens * PARTIAL_SLOTS_PER_TOKEN * PARTIAL_SLOT_BYTES : 0U);

    // 输出Tiling数据
    context->SetBlockDim(usedCoreNum);
    if (enableDualSibling) {
        context->SetScheduleMode(1);
    }
    uint64_t tilingKey = (useRmsMid ? TILING_KEY_USE_RMS_MID : 0ULL) |
                         (returnHInF32 ? TILING_KEY_RETURN_H_IN_F32 : 0ULL) |
                         (coresPerToken == 2 ? TILING_KEY_DUAL_SIBLING_AIV : 0ULL);
    context->SetTilingKey(tilingKey);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    OP_LOGD(context->GetNodeName(), "returnHInF32:  %s", (returnHInF32 ? "true" : "false"));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForMhcNormPostPreonly(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MhcNormPostPreonly)
    .Tiling(TilingMhcNormPostPreonly)
    .TilingParse<MhcNormPostPreonlyCompileInfo>(TilingPrepareForMhcNormPostPreonly);
} // namespace optiling
