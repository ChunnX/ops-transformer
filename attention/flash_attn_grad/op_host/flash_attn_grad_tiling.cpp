/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attn_grad_tiling.h"
#include <set>

#include "flash_attn_grad_tiling_check.h"
#include "log/log.h"
#include "op_host/tiling_util.h"

namespace optiling {

static bool HasAttenMask(gert::TilingContext *context, int64_t maskMode)
{
    if (maskMode == 0 || (maskMode != 3 && maskMode != 4)) {
        return false;
    }
    auto attnMaskShape = context->GetOptionalInputShape(ATTN_MASK_INDEX);
    return (attnMaskShape != nullptr) && (attnMaskShape->GetStorageShape().GetDimNum() > 0);
}

static ge::graphStatus FlashAttnGradTilingFunc(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context->GetNodeName(), "GetPlatformInfo is nullptr."),
                return ge::GRAPH_PARAM_INVALID);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0,
                OP_LOGE(context->GetNodeName(), "num of core obtained is 0, aicNum=%u, aivNum=%u.", aicNum, aivNum),
                return ge::GRAPH_PARAM_INVALID);

    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context)) {
        OP_LOGE(context->GetNodeName(), "SOC Version is not support.");
        return ge::GRAPH_PARAM_INVALID;
    }

    OP_CHECK_IF(context->GetWorkspaceSizes(1) == nullptr,
                OP_LOGE(context->GetNodeName(), "workSpaceSize got from ge is nullptr."),
                return ge::GRAPH_PARAM_INVALID);

    auto checkRet = FlashAttnGradCheck::CheckParams(context);
    OP_CHECK_IF(checkRet != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckParams failed."),
                return ge::GRAPH_PARAM_INVALID);

    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    auto attrs = context->GetAttrs();
    std::string layoutQStr = "BSND";
    int64_t maxSeqlenQ = -1;
    int64_t maxSeqlenKv = -1;
    int64_t maskMode = 0;
    float scaleValue = 0.0f;
    if (attrs != nullptr) {
        auto scalePtr = attrs->GetAttrPointer<float>(0);
        if (scalePtr != nullptr) { scaleValue = *scalePtr; }
        auto maskModePtr = attrs->GetAttrPointer<int64_t>(1);
        if (maskModePtr != nullptr) { maskMode = *maskModePtr; }
        auto maxSeqlenQPtr = attrs->GetAttrPointer<int64_t>(4);
        if (maxSeqlenQPtr != nullptr) { maxSeqlenQ = *maxSeqlenQPtr; }
        auto maxSeqlenKvPtr = attrs->GetAttrPointer<int64_t>(5);
        if (maxSeqlenKvPtr != nullptr) { maxSeqlenKv = *maxSeqlenKvPtr; }
        auto layoutQPtr = attrs->GetAttrPointer<char>(6);
        if (layoutQPtr != nullptr) { layoutQStr = std::string(layoutQPtr); }
    }

    int64_t layout = (layoutQStr == "TND") ? 1 : 0;

    auto qShape = context->GetInputShape(Q_INDEX);
    auto kShape = context->GetInputShape(K_INDEX);
    auto vShape = context->GetInputShape(V_INDEX);

    int64_t bSize = 0;
    int64_t s1Size = 0;
    int64_t s2Size = 0;
    int64_t n1Size = 0;
    int64_t n2Size = 0;
    int64_t dSize = 0;
    int64_t dvSize = 0;

    if (layout == 1) {
        s1Size = maxSeqlenQ > 0 ? maxSeqlenQ : qShape->GetStorageShape().GetDim(0);
        s2Size = maxSeqlenKv > 0 ? maxSeqlenKv : kShape->GetStorageShape().GetDim(0);
        n1Size = qShape->GetStorageShape().GetDim(1);
        n2Size = kShape->GetStorageShape().GetDim(1);
        dSize = qShape->GetStorageShape().GetDim(2);
        dvSize = vShape->GetStorageShape().GetDim(2);
        auto cuSeqShape = context->GetOptionalInputShape(CU_SEQLENS_Q_INDEX);
        if (cuSeqShape != nullptr && cuSeqShape->GetStorageShape().GetDimNum() > 0) {
            bSize = cuSeqShape->GetStorageShape().GetDim(0) - 1;
        } else {
            bSize = 1;
        }
    } else {
        bSize = qShape->GetStorageShape().GetDim(0);
        s1Size = qShape->GetStorageShape().GetDim(1);
        n1Size = qShape->GetStorageShape().GetDim(2);
        dSize = qShape->GetStorageShape().GetDim(3);
        dvSize = vShape->GetStorageShape().GetDim(3);
        s2Size = kShape->GetStorageShape().GetDim(1);
        n2Size = kShape->GetStorageShape().GetDim(2);
    }

        // Decide the block-to-core mapping. Linear splitting spreads the live cores
    // across many heads: at S=8192 D=128 with 36 cores, core 0 works head 0
    // while core 35 works head 31, so 36 cores stream K/V for 32 heads at once
    // and the working set blows past L2 (128 MB). msprof shows mte2_ratio 0.72
    // on both cube and vector while mac *drops* to 0.733 -- HBM-bound, not
    // compute-bound. Swizzle hands out whole s2 columns strided across cores
    // instead, keeping the live cores inside one head.
    //
    // Mirrors the AscendC reference's isExceedL2Cache / isSplitByBlockIdx gate.
    // Count every tensor a head touches (~22 MB/head at S=8192): q+dout+attn_out
    // read fp16, dq written fp32, k+v read fp16, dk+dv written fp32. Counting
    // only K+V lands exactly on 128 MB at N=32 and wrongly reports "fits",
    // where swizzle actually wins by 34%.
    //
    // Swizzle's granularity is a whole column, so it is skipped when there are
    // too few columns to balance (N=1 has 64 columns for 36 cores: core 0 takes
    // 2 while core 35 takes 1, a 2x skew that costs 14%).
    int64_t swizzle = 0;
    if (layout != 1) {
        constexpr int64_t CUBE_BASE_M = 128;
        constexpr int64_t CUBE_BASE_N = 128;
        constexpr int64_t L2_CACHE_BYTES = 128L * 1024 * 1024;
        constexpr int64_t FP16_BYTES = 2;
        constexpr int64_t FP32_BYTES = 4;
        int64_t s1Outer = (s1Size + CUBE_BASE_M - 1) / CUBE_BASE_M;
        int64_t s2Outer = (s2Size + CUBE_BASE_N - 1) / CUBE_BASE_N;
        int64_t perHeadBlocks = s1Outer * s2Outer;
        int64_t totalBlocks = bSize * n2Size * perHeadBlocks;
        // Must use aicNum, not blockDim: blockDim is the mixed scheduling value
        // from CalcTschBlockDim, while the kernel's coreNum comes from
        // pl.get_block_num() which equals the physical cube core count (36).
        // Using blockDim here would miscount the live heads.
        if (perHeadBlocks > 0 && aicNum > 0) {
            int64_t cores = static_cast<int64_t>(aicNum);
            int64_t perCore = (totalBlocks + cores - 1) / cores;
            // How many distinct heads are live at once under linear splitting
            std::set<int64_t> liveHeads;
            for (int64_t c = 0; c < cores; ++c) {
                liveHeads.insert((c * perCore) / perHeadBlocks);
            }
            int64_t perHeadBytes = 3 * s1Size * dSize * FP16_BYTES + s1Size * dSize * FP32_BYTES +
                                   2 * s2Size * dSize * FP16_BYTES + 2 * s2Size * dSize * FP32_BYTES;
            int64_t liveBytes = static_cast<int64_t>(liveHeads.size()) * perHeadBytes;
            // 2x L2, not 1x: the measured crossover is a 264 MB working set
            // (N=12), not the N=6 that merely exceeds 128 MB. Just past L2 there
            // is still substantial partial hitting, and there swizzle's coarser
            // granularity (a whole s1Outer column) loses on load balance.
            // Measured (S=8192, 36 cores, us, linear/swizzle): N=1 193/222,
            // N=8 1556/1761, N=10 1904/2007, N=12 2833/2677, N=16 4121/3550,
            // N=24 6483/4982. Linear wins through N=10, swizzle from N=12 on.
            if (liveBytes > 2 * L2_CACHE_BYTES && bSize * n2Size * s2Outer >= 2 * cores) {
                swizzle = 1;
            }
        }
    }

    FlashAttnGradTilingData *tilingData = context->GetTilingData<FlashAttnGradTilingData>();
    tilingData->b = bSize;
    tilingData->s1 = s1Size;
    tilingData->s2 = s2Size;
    tilingData->n1 = n1Size;
    tilingData->n2 = n2Size;
    tilingData->d = dSize;
    tilingData->dv = dvSize;
    tilingData->scaleValue = scaleValue == 0.0f ? 1.0f / sqrt(static_cast<float>(dSize)) : scaleValue;

    bool hasAttenMask = HasAttenMask(context, maskMode);
    // TilingKey: bit[1:0]=template(0=bn2gs1s2,1=bn2,2=bn2s2,3=bn2multiblk),
    //            bit2=layout(0=non-TND,1=TND), bit3=hasAttenMask,
    //            bit4=swizzle(0=linear split, 1=s2-column strided)
    uint64_t tilingKey = static_cast<uint64_t>(0) |
                         (static_cast<uint64_t>(layout) << 2) |
                         (static_cast<uint64_t>(hasAttenMask) << 3) |
                         (static_cast<uint64_t>(swizzle) << 4);
    context->SetTilingKey(tilingKey);

    // Workspace layout must match the kernel's make_ptr partitioning exactly:
    //   [0]          libapi reserve (system)
    //   [+libapi]    dq accumulator  B*S1*N2*D fp32
    //   [+...]       dk accumulator  B*S2*N2*D fp32
    //   [+...]       dv accumulator  B*S2*N2*D fp32
    // The kernel accumulates dq/dk/dv in fp32 via atomicAdd, then its post
    // stage applies scale and casts down to the fp16/bf16 outputs. dv gets no
    // scale (matches qkvIdx < 2 in the AscendC reference).
    constexpr size_t FP32_SIZE = 4;
    size_t libapiSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t dqElems = static_cast<size_t>(bSize) * s1Size * n2Size * dSize;
    size_t dkvElems = static_cast<size_t>(bSize) * s2Size * n2Size * dSize;
    size_t workspaceSize = libapiSize + (dqElems + dkvElems * 2) * FP32_SIZE;
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    OP_LOGI(context->GetNodeName(),
            "FlashAttnGrad tiling: blockDim=%u, layout=%ld, maskMode=%ld, hasAttenMask=%d, swizzle=%ld, B=%ld, S1=%ld, S2=%ld, N1=%ld, D=%ld, tilingKey=%llu, workspaceSize=%zu.",
            blockDim, layout, maskMode, hasAttenMask, swizzle, bSize, s1Size, s2Size, n1Size, dSize, tilingKey, workspaceSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForFlashAttnGrad([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FlashAttnGrad)
    .Tiling(FlashAttnGradTilingFunc)
    .TilingParse<FlashAttnGradCompileInfo>(TilingParseForFlashAttnGrad);

} // namespace optiling
