/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mega_mhc_tiling.h"
#include "mega_mhc_memory_plan.h"
#include "../../../op_kernel/arch35/execution_policy.h"
#include "../../../op_kernel/arch35/mega_mhc_tiling_key.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include "err/ops_err.h"
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "platform/platform_infos_def.h"
#include "register/op_def_registry.h"

namespace optiling {
namespace {
constexpr uint32_t kHeadCount = 4;
constexpr uint32_t kProjectionQ = 24;
constexpr uint32_t kUbLimitBytes = 248 * 1024;
constexpr uint32_t kWorkspaceAlign = 512;
constexpr uint64_t kRuntimeWorkspaceBytes = 20ULL * 1024ULL * 1024ULL;
constexpr uint32_t kSlotCount = 2;

/** Compute an overflow-safe ceiling division for host-side tile planning. */
uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

/** Align a workspace region while rejecting uint64 overflow. */
bool AlignWorkspace(uint64_t value, uint64_t alignment, uint64_t &aligned)
{
    if (alignment == 0) {
        return false;
    }
    const uint64_t blocks = CeilDiv(value, alignment);
    if (blocks > std::numeric_limits<uint64_t>::max() / alignment) {
        return false;
    }
    aligned = blocks * alignment;
    return true;
}

/** Multiply workspace extents while rejecting uint64 overflow. */
bool CheckedBytes(uint64_t first, uint64_t second, uint64_t third, uint64_t elementBytes, uint64_t &bytes)
{
    if (first != 0 && second > std::numeric_limits<uint64_t>::max() / first) {
        return false;
    }
    const uint64_t firstTwo = first * second;
    if (firstTwo != 0 && third > std::numeric_limits<uint64_t>::max() / firstTwo) {
        return false;
    }
    const uint64_t elements = firstTwo * third;
    if (elements != 0 && elementBytes > std::numeric_limits<uint64_t>::max() / elements) {
        return false;
    }
    bytes = elements * elementBytes;
    return true;
}

} // namespace

REGISTER_OPS_TILING_TEMPLATE(MegaMhc, MegaMhcTiling, 1000);

/** Select this host tiler only when the graph is targeting an Ascend 950 platform. */
bool MegaMhcTiling::IsCapable()
{
    return context_ != nullptr && context_->GetPlatformInfo() != nullptr;
}

/** Read per-core memory and execution-unit capacities from the 950 platform description. */
ge::graphStatus MegaMhcTiling::GetPlatformInfo()
{
    auto *platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_->GetNodeName(), "Ascend 950 platform info is required for mega_mhc tiling");
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(platformInfo);
    aicCount_ = platform.GetCoreNumAic();
    aivCount_ = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes_);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Bytes_);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0ABytes_);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0BBytes_);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CBytes_);
    if (aicCount_ == 0 || aivCount_ < 2 || ubBytes_ == 0 || l1Bytes_ == 0 || l0ABytes_ == 0 || l0BBytes_ == 0 ||
        l0CBytes_ == 0) {
        OP_LOGE(context_->GetNodeName(), "Ascend 950 returned incomplete per-core resources for mega_mhc");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

/** Validate supported shapes/types and read optional-stage and precision attributes. */
ge::graphStatus MegaMhcTiling::GetShapeAttrsInfo()
{
    return ParseInputsAndAttributes();
}

/** Parse and validate the N=4, aligned D in [2560,7168] mega_mhc input contract. */
ge::graphStatus MegaMhcTiling::ParseInputsAndAttributes()
{
    if (context_->GetDynamicInputTensor(8, 0) != nullptr) {
        OP_LOGE(context_->GetNodeName(), "pre_mix is reserved and must be absent in this version");
        return ge::GRAPH_FAILED;
    }
    // The ACLNN launcher uses a [0] FP32 output to preserve the reserved kernel argument slot.
    const auto *hPreDesc = context_->GetOutputDesc(4);
    const auto *hPreShape = context_->GetOutputShape(4);
    if (hPreDesc != nullptr &&
        (hPreShape == nullptr || hPreDesc->GetDataType() != ge::DT_FLOAT ||
         hPreShape->GetStorageShape().GetDimNum() != 1 || hPreShape->GetStorageShape().GetDim(0) != 0)) {
        OP_LOGE(context_->GetNodeName(), "h_pre is reserved; only an empty internal placeholder is supported");
        return ge::GRAPH_FAILED;
    }
    constexpr size_t kInputCount = 11;
    for (size_t index = 0; index < kInputCount; ++index) {
        const bool optional = index == 8 || index == 10;
        // Optional inputs are indexed by their prototype position even when earlier options are absent.
        const auto *desc = index >= 8 ? context_->GetOptionalInputDesc(index) : context_->GetInputDesc(index);
        const auto *shapeContext = index >= 8 ? context_->GetOptionalInputShape(index) : context_->GetInputShape(index);
        const auto *tensor = context_->GetDynamicInputTensor(index, 0);
        if ((!optional && (desc == nullptr || shapeContext == nullptr || tensor == nullptr)) ||
            (optional && tensor != nullptr && (desc == nullptr || shapeContext == nullptr))) {
            OP_LOGE(context_->GetNodeName(), "mega_mhc input %zu is missing", index);
            return ge::GRAPH_FAILED;
        }
    }

    const auto *xShapeContext = context_->GetInputShape(0);
    const auto &xShape = xShapeContext->GetStorageShape();
    if ((xShape.GetDimNum() != 2 && xShape.GetDimNum() != 3) || xShape.GetDim(xShape.GetDimNum() - 1) <= 0) {
        OP_LOGE(context_->GetNodeName(), "h_out must have shape [T,D] or [B,S,D]");
        return ge::GRAPH_FAILED;
    }
    const int64_t d = xShape.GetDim(xShape.GetDimNum() - 1);
    if (d < 2560 || d > 7168 || d % 256 != 0) {
        OP_LOGE(context_->GetNodeName(), "mega_mhc requires D in [2560,7168], aligned to 256; got %ld", d);
        return ge::GRAPH_FAILED;
    }
    d_ = static_cast<uint32_t>(d);
    totalRows_ = 1;
    for (int64_t axis = 0; axis < xShape.GetDimNum() - 1; ++axis) {
        const int64_t dimension = xShape.GetDim(axis);
        if (dimension <= 0 || totalRows_ > std::numeric_limits<uint32_t>::max() / static_cast<uint64_t>(dimension)) {
            OP_LOGE(context_->GetNodeName(), "h_out leading dimensions must be positive and fit in uint32");
            return ge::GRAPH_FAILED;
        }
        totalRows_ *= static_cast<uint64_t>(dimension);
    }
    if (totalRows_ == 0 || totalRows_ > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE(context_->GetNodeName(), "total token count must be in [1, UINT32_MAX]");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &residual = context_->GetInputShape(1)->GetStorageShape();
    if (residual.GetDimNum() != xShape.GetDimNum() + 1 || residual.GetDim(residual.GetDimNum() - 2) != kHeadCount ||
        residual.GetDim(residual.GetDimNum() - 1) != d_) {
        OP_LOGE(context_->GetNodeName(), "residual must be [T,4,D] or [B,S,4,D]");
        return ge::GRAPH_FAILED;
    }
    for (int64_t axis = 0; axis < xShape.GetDimNum() - 1; ++axis) {
        if (residual.GetDim(axis) != xShape.GetDim(axis)) {
            OP_LOGE(context_->GetNodeName(), "residual leading shape must match h_out");
            return ge::GRAPH_FAILED;
        }
    }

    const auto &hPost = context_->GetInputShape(2)->GetStorageShape();
    const auto &hRes = context_->GetInputShape(3)->GetStorageShape();
    const auto &phi = context_->GetInputShape(4)->GetStorageShape();
    const auto &alpha = context_->GetInputShape(5)->GetStorageShape();
    const auto &bias = context_->GetInputShape(6)->GetStorageShape();
    const auto &gamma1 = context_->GetInputShape(7)->GetStorageShape();
    const uint64_t q = kProjectionQ;
    k_ = kHeadCount * d_;
    if (hPost.GetDimNum() != xShape.GetDimNum() || hPost.GetDim(hPost.GetDimNum() - 1) != kHeadCount ||
        hRes.GetDimNum() != xShape.GetDimNum() + 1 || hRes.GetDim(hRes.GetDimNum() - 2) != kHeadCount ||
        hRes.GetDim(hRes.GetDimNum() - 1) != kHeadCount || phi.GetDimNum() != 2 ||
        phi.GetDim(0) != static_cast<int64_t>(q) || phi.GetDim(1) != k_ || alpha.GetDimNum() != 1 ||
        alpha.GetDim(0) != 3 || bias.GetDimNum() != 1 || bias.GetDim(0) != static_cast<int64_t>(q) ||
        gamma1.GetDimNum() != 1 || gamma1.GetDim(0) != d_) {
        OP_LOGE(context_->GetNodeName(), "mega_mhc parameter shapes do not match N=4, D=%u", d_);
        return ge::GRAPH_FAILED;
    }
    for (int64_t axis = 0; axis < xShape.GetDimNum() - 1; ++axis) {
        if (hPost.GetDim(axis) != xShape.GetDim(axis) || hRes.GetDim(axis) != xShape.GetDim(axis)) {
            OP_LOGE(context_->GetNodeName(), "h_post and h_res leading shapes must match h_out");
            return ge::GRAPH_FAILED;
        }
    }

    inputType_ = context_->GetInputDesc(0)->GetDataType();
    if ((inputType_ != ge::DT_FLOAT16 && inputType_ != ge::DT_BF16) ||
        context_->GetInputDesc(1)->GetDataType() != inputType_) {
        OP_LOGE(context_->GetNodeName(), "h_out and residual must share FP16 or BF16 dtype");
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 2; index < kInputCount; ++index) {
        if (index == 8 || index == 10) {
            if (context_->GetDynamicInputTensor(index, 0) == nullptr) {
                continue;
            }
        }
        const auto *desc = index >= 8 ? context_->GetOptionalInputDesc(index) : context_->GetInputDesc(index);
        if (desc->GetDataType() != ge::DT_FLOAT) {
            OP_LOGE(context_->GetNodeName(), "mega_mhc coefficient and gamma inputs must be FP32");
            return ge::GRAPH_FAILED;
        }
    }

    hasMid_ = context_->GetDynamicInputTensor(10, 0) != nullptr;
    {
        const auto *gamma0Context = context_->GetOptionalInputShape(9);
        if (gamma0Context == nullptr || gamma0Context->GetStorageShape().GetDimNum() != 1 ||
            gamma0Context->GetStorageShape().GetDim(0) != d_) {
            OP_LOGE(context_->GetNodeName(), "gamma0 must have shape [D]");
            return ge::GRAPH_FAILED;
        }
    }
    if (hasMid_) {
        const auto &gamma2 = context_->GetOptionalInputShape(10)->GetStorageShape();
        if (gamma2.GetDimNum() != 1 || gamma2.GetDim(0) != k_) {
            OP_LOGE(context_->GetNodeName(), "gamma2 must have shape [4D] when RMSNorm_mid is enabled");
            return ge::GRAPH_FAILED;
        }
    }

    const auto *attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context_->GetNodeName(), "mega_mhc attributes are missing");
        return ge::GRAPH_FAILED;
    }
    const float *normEps = attrs->GetAttrPointer<float>(0);
    const float *mhcPreNormEps = attrs->GetAttrPointer<float>(1);
    const float *hcEps = attrs->GetAttrPointer<float>(2);
    const int64_t *computeMode = attrs->GetAttrPointer<int64_t>(3);
    normEps_ = normEps == nullptr ? 1.0e-6f : *normEps;
    mhcPreNormEps_ = mhcPreNormEps == nullptr ? 1.0e-6f : *mhcPreNormEps;
    hcEps_ = hcEps == nullptr ? 1.0e-6f : *hcEps;
    if (!std::isfinite(normEps_) || normEps_ <= 0.0f || !std::isfinite(mhcPreNormEps_) || mhcPreNormEps_ <= 0.0f ||
        !std::isfinite(hcEps_) || hcEps_ < 0.0f) {
        OP_LOGE(context_->GetNodeName(),
                "norm_eps and mhc_pre_norm_eps must be positive and hc_eps must be nonnegative");
        return ge::GRAPH_FAILED;
    }
    const int64_t mode = computeMode == nullptr ? 0 : *computeMode;
    if (mode != 0 && mode != 1) {
        OP_LOGE(context_->GetNodeName(), "compute_mode supports only 0 (FP32) and 1 (HF32)");
        return ge::GRAPH_FAILED;
    }
    // Scheduling is an internal tiling decision, not a public attribute.
    scheduleMode_ = mega_mhc::ScheduleMode::MK;
    projectionMode_ = mode == 0 ? mega_mhc::ProjectionMode::FP32 : mega_mhc::ProjectionMode::HF32;
    return ge::GRAPH_SUCCESS;
}

/** Serialize the host-selected M/K grid and all Cube loop/tail decisions into TilingData. */
void MegaMhcTiling::CopyCubePlanToTilingData()
{
    tilingData_.set_activeAic(cubePlan_.activeAic);
    tilingData_.set_activeAiv(cubePlan_.activeAiv);
    tilingData_.set_mGrid(cubePlan_.mGrid);
    tilingData_.set_kGrid(cubePlan_.kGrid);
    tilingData_.set_mWaveCount(cubePlan_.mWaveCount);
    tilingData_.set_rowsPerWave(cubePlan_.rowsPerWave);
    tilingData_.set_lastWaveMGrid(cubePlan_.lastWaveMGrid);
    tilingData_.set_lastWaveRows(cubePlan_.lastWaveRows);
    tilingData_.set_mTile(cubePlan_.mTile);
    tilingData_.set_splitK(cubePlan_.splitK);
    tilingData_.set_largeKCoreCount(cubePlan_.largeKCoreCount);
    tilingData_.set_shortSplitK(cubePlan_.shortSplitK);
    tilingData_.set_shortKCoreL1FullLoops(cubePlan_.shortKCoreL1FullLoops);
    tilingData_.set_kL1(cubePlan_.l1K);
    tilingData_.set_l1KFullLoops(cubePlan_.l1KFullLoops);
    tilingData_.set_l1KTail(cubePlan_.l1KTail);
    tilingData_.set_lastKCoreL1FullLoops(cubePlan_.lastKCoreL1FullLoops);
    tilingData_.set_lastKCoreL1Tail(cubePlan_.lastKCoreL1Tail);
    tilingData_.set_lastMCoreRows(cubePlan_.lastMCoreRows);
    tilingData_.set_lastKCoreElements(cubePlan_.lastKCoreElements);
    tilingData_.set_mL0FullLoops(cubePlan_.mL0FullLoops);
    tilingData_.set_mL0TailRows(cubePlan_.mL0TailRows);
    tilingData_.set_lastMCoreL0FullLoops(cubePlan_.lastMCoreL0FullLoops);
    tilingData_.set_lastMCoreL0TailRows(cubePlan_.lastMCoreL0TailRows);
    tilingData_.set_loadMStride(cubePlan_.loadMStride);
    tilingData_.set_loadNStride(cubePlan_.loadNStride);
    tilingData_.set_fullL0LoadKStep(cubePlan_.fullL0LoadKStep);
    tilingData_.set_l0TailLoadKStep(cubePlan_.l0TailLoadKStep);
    tilingData_.set_tailL1TailLoadKStep(cubePlan_.tailL1TailLoadKStep);
    tilingData_.set_lastCoreTailLoadKStep(cubePlan_.lastCoreTailLoadKStep);
    tilingData_.set_baseM(cubePlan_.baseM);
    tilingData_.set_baseN(cubePlan_.baseN);
    tilingData_.set_baseK(cubePlan_.baseK);
    // Mmad API requires PIPE_M when (m/16)*(n/16)<10; decide once on Host.
    tilingData_.set_mmadNeedsBarrier((cubePlan_.nzM / 16U) * (cubePlan_.nzN / 16U) < 10U ? 1U : 0U);
    tilingData_.set_nzM(cubePlan_.nzM);
    tilingData_.set_nzN(cubePlan_.nzN);
    tilingData_.set_l1ASlotElements(cubePlan_.l1ASlotElements);
    tilingData_.set_l1BOffsetElements(cubePlan_.l1BOffsetElements);
    tilingData_.set_l1BSlotElements(cubePlan_.l1BSlotElements);
    tilingData_.set_l0ASlotElements(cubePlan_.l0ASlotElements);
    tilingData_.set_l0BSlotElements(cubePlan_.l0BSlotElements);
    tilingData_.set_l0CSlotElements(cubePlan_.l0CSlotElements);
    tilingData_.set_fixpipeRowStride(cubePlan_.fixpipeRowStride);
    tilingData_.set_fullMmadKStep(cubePlan_.fullMmadKStep);
    tilingData_.set_fullMmadKLoopCount(cubePlan_.fullMmadKLoopCount);
    tilingData_.set_l0TailMmadKStep(cubePlan_.l0TailMmadKStep);
    tilingData_.set_l0TailMmadKLoopCount(cubePlan_.l0TailMmadKLoopCount);
    tilingData_.set_l0TailMmadFinalKStep(cubePlan_.l0TailMmadFinalKStep);
    tilingData_.set_tailL1TailMmadKStep(cubePlan_.tailL1TailMmadKStep);
    tilingData_.set_tailL1TailMmadKLoopCount(cubePlan_.tailL1TailMmadKLoopCount);
    tilingData_.set_tailL1TailMmadFinalKStep(cubePlan_.tailL1TailMmadFinalKStep);
    tilingData_.set_lastCoreTailMmadKStep(cubePlan_.lastCoreTailMmadKStep);
    tilingData_.set_lastCoreTailMmadKLoopCount(cubePlan_.lastCoreTailMmadKLoopCount);
    tilingData_.set_lastCoreTailMmadFinalKStep(cubePlan_.lastCoreTailMmadFinalKStep);
    tilingData_.set_l0KFullLoops(cubePlan_.l0KLoops);
    tilingData_.set_l0KTail(cubePlan_.l0KTail);
    tilingData_.set_tailL1L0FullLoops(cubePlan_.tailL1L0FullLoops);
    tilingData_.set_tailL1L0Tail(cubePlan_.tailL1L0Tail);
    tilingData_.set_lastCoreTailL1L0FullLoops(cubePlan_.lastCoreTailL1L0FullLoops);
    tilingData_.set_lastCoreTailL1L0Tail(cubePlan_.lastCoreTailL1L0Tail);
    tilingData_.set_partialGroupK(cubePlan_.partialGroupK);
    tilingData_.set_partialGroupCount(cubePlan_.partialGroupCount);
}

/** Build CubePlan, choose stage tiles from the real UB size, and lay out checked workspace regions. */
ge::graphStatus MegaMhcTiling::DoOpTiling()
{
    const bool packedNz = (hasMid_ ? MegaMhc::kFiveProjectionTransport : MegaMhc::kFourProjectionTransport) ==
                          MegaMhc::ProjectionTransport::UbToL1;
    mega_mhc::CubePlanRequest request;
    request.rows = totalRows_;
    request.schedule = scheduleMode_;
    request.wavePipeline = false;
    request.preferWideTile = true; // Both variants project materialized rows; wave tiling overrides this preference.
    request.k = k_;
    request.q = kProjectionQ;
    request.aicCount = aicCount_;
    request.aivCount = aivCount_;
    request.l1ABytes = l1Bytes_ / 2;
    request.l1BBytes = l1Bytes_ - request.l1ABytes;
    request.l0ABytes = l0ABytes_;
    request.l0BBytes = l0BBytes_;
    request.l0CBytes = l0CBytes_;
    request.mode = projectionMode_;
    if (!mega_mhc::BuildCubePlan(request, cubePlan_)) {
        OP_LOGE(context_->GetNodeName(), "no legal M/K CubePlan fits the reported 950 L1/L0 capacities");
        return ge::GRAPH_FAILED;
    }

    const uint64_t availableUb = std::min<uint64_t>(ubBytes_, kUbLimitBytes);
    // Try waves only when every producer owns complete rows and all live buffers fit.
    if (scheduleMode_ == mega_mhc::ScheduleMode::MK && !hasMid_ && std::min(aicCount_, aivCount_ / 2U) >= 4U) {
        // Prefer a complete-wave plan; if alignment invalidates it, retain the larger batch and use its tail.
        for (uint32_t candidate = 0U; candidate < 2U; ++candidate) {
            auto waveRequest = request;
            waveRequest.wavePipeline = true;
            waveRequest.preferCompleteWaves = candidate == 0U;
            mega_mhc::CubePlan wavePlan;
            if (mega_mhc::BuildCubePlan(waveRequest, wavePlan) && wavePlan.mTile % (2U * wavePlan.kGrid) == 0U &&
                wavePlan.mWaveCount > 1U && wavePlan.lastKCoreL1FullLoops >= 2U &&
                mega_mhc::ProducerUbBytes(d_, false, true) +
                        mega_mhc::ProjectionUbBytes(k_, wavePlan.l1K, wavePlan.mTile / 2U) <=
                    availableUb) {
                request = waveRequest;
                cubePlan_ = wavePlan;
                break;
            }
        }
    }
    tilingData_.set_totalRows(static_cast<uint32_t>(totalRows_));
    tilingData_.set_n(kHeadCount);
    tilingData_.set_d(d_);
    tilingData_.set_k(k_);
    tilingData_.set_q(kProjectionQ);
    tilingData_.set_normEps(normEps_);
    tilingData_.set_mhcPreNormEps(mhcPreNormEps_);
    tilingData_.set_hcEps(hcEps_);
    CopyCubePlanToTilingData();
    const uint32_t waveProducerRows = request.wavePipeline ? cubePlan_.mTile / (cubePlan_.kGrid * 2U) : 0U;
    const uint32_t waveTileCount = cubePlan_.lastKCoreL1FullLoops;
    const uint32_t rowsPerTick =
        request.wavePipeline ? static_cast<uint32_t>(CeilDiv(waveProducerRows, waveTileCount - 1U)) : 0U;
    const uint32_t tickCount =
        request.wavePipeline ? static_cast<uint32_t>(CeilDiv(waveProducerRows, rowsPerTick)) : 0U;
    tilingData_.set_waveProducerRowsPerTick(rowsPerTick);
    tilingData_.set_wavePipeline(request.wavePipeline ? 1U : 0U);
    tilingData_.set_waveProducerRows(waveProducerRows);
    tilingData_.set_lastWaveFullProducers(request.wavePipeline ? cubePlan_.lastWaveRows / waveProducerRows : 0U);
    tilingData_.set_lastWaveProducerTailRows(request.wavePipeline ? cubePlan_.lastWaveRows % waveProducerRows : 0U);
    tilingData_.set_waveProducerTileStride(request.wavePipeline ? (waveTileCount - 1U) / tickCount : 0U);
    tilingData_.set_waveGroupSize(request.wavePipeline ? cubePlan_.kGrid * 2U : 0U);

    // Exact simultaneous allocations, each padded like Kernel::InitUbBuffer.
    // The baseline does not reuse live buffers within a phase. Destroy() separates
    // RMS0, producer/projection and epilogue lifetimes, so their peaks do not add.
    const uint32_t projectionRows = static_cast<uint32_t>(CeilDiv(cubePlan_.mTile, 2));
    // Producer rows are complete tokens; double input slots are always enabled in the scheduler.
    tilingData_.set_producerPrefetch(0U); // Reserved legacy field.
    tilingData_.set_postRows(1U);
    tilingData_.set_projectionRows(projectionRows);
    // Layout padding is optional: never disable an otherwise legal wave to obtain it.
    const uint64_t liveProducer = request.wavePipeline ? mega_mhc::ProducerUbBytes(d_, false, true) : 0U;
    const bool padStageRows = packedNz && liveProducer + mega_mhc::ProjectionUbBytes(k_, cubePlan_.l1K, projectionRows,
                                                                                     1U, projectionRows + 1U) <=
                                              availableUb;
    const uint32_t stageRowStride = projectionRows + static_cast<uint32_t>(padStageRows);
    tilingData_.set_stageRowStride(stageRowStride);
    // Direct NZ stores require padding to avoid their strided-store penalty; preserve waves via GM otherwise.
    tilingData_.set_projectionTransport(static_cast<uint32_t>(padStageRows ? MegaMhc::ProjectionTransport::UbToL1 :
                                                                             MegaMhc::ProjectionTransport::GmStage));
    // Keep the epilogue's reduction batch explicit, independently of its one-row compute tile.
    const uint32_t gammaTile = hasMid_ ? mega_mhc::MidGammaTile(d_, availableUb) : k_;
    if (gammaTile == 0U) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t gammaCount = static_cast<uint32_t>(CeilDiv(k_, gammaTile));
    tilingData_.set_midGammaTile(gammaTile);
    tilingData_.set_midGammaTileCount(gammaCount);
    tilingData_.set_midGammaLastTile(k_ - (gammaCount - 1U) * gammaTile);
    // Wave projection rows do not constrain the independent epilogue lifetime or token ownership.
    const uint32_t reductionRowLimit =
        request.wavePipeline ?
            static_cast<uint32_t>(CeilDiv(totalRows_, cubePlan_.activeAiv)) :
            std::min(projectionRows, static_cast<uint32_t>(CeilDiv(totalRows_, cubePlan_.activeAiv)));
    const uint32_t capacityRows = mega_mhc::FitReductionRows(reductionRowLimit, d_, cubePlan_.partialGroupCount,
                                                             cubePlan_.fixpipeRowStride, cubePlan_.kGrid, availableUb);
    const uint32_t reduceRows = mega_mhc::BalanceReductionRows(reductionRowLimit, capacityRows);
    if (reduceRows == 0U) {
        return ge::GRAPH_FAILED;
    }
    tilingData_.set_reduceRows(reduceRows);
    tilingData_.set_reduceRowsAligned((reduceRows + 7U) / 8U * 8U);
    // Once gates are complete, the partial batch can hold a second x2 row.
    tilingData_.set_epiloguePrefetch(static_cast<uint64_t>(reduceRows) * cubePlan_.partialGroupCount *
                                                 cubePlan_.fixpipeRowStride * sizeof(float) >=
                                             k_ * 2ULL ?
                                         1U :
                                         0U);
    tilingData_.set_slotCount(kSlotCount);
    tilingData_.set_ubReservedBytes(0);
    const uint64_t producerPeak = mega_mhc::ProducerUbBytes(d_, hasMid_, false, gammaTile);
    // Add a second FP32 UB stage only when the simultaneous live buffers fit.
    // Cube geometry and GM staging remain unchanged.
    const uint64_t doubleProjection =
        mega_mhc::ProjectionUbBytes(k_, cubePlan_.l1K, projectionRows, 2U, stageRowStride);
    const uint64_t concurrentProducer = request.wavePipeline ? mega_mhc::ProducerUbBytes(d_, false, true) : 0U;
    const uint32_t stageUbSlots = doubleProjection + concurrentProducer <= availableUb ? 2U : 1U;
    tilingData_.set_stageUbSlots(stageUbSlots);
    const uint64_t projectionPeak =
        mega_mhc::ProjectionUbBytes(k_, cubePlan_.l1K, projectionRows, stageUbSlots, stageRowStride);
    const uint64_t concurrentUb = projectionPeak + concurrentProducer;
    // FP32 projection leaves Cube compute time for the previous wave's vector epilogue.
    // Keep HF32's lighter projection on the separate epilogue schedule.
    const uint32_t waveEpilogueRows =
        request.mode == mega_mhc::ProjectionMode::FP32 && request.wavePipeline && cubePlan_.mWaveCount > 1U &&
                concurrentUb < availableUb ?
            mega_mhc::FitReductionRows(waveProducerRows, d_, tilingData_.get_partialGroupCount(),
                                       cubePlan_.fixpipeRowStride, cubePlan_.kGrid, availableUb - concurrentUb) :
            0U;
    tilingData_.set_waveEpilogueRows(waveEpilogueRows);
    if (waveEpilogueRows != 0U) {
        // This schedule owns the epilogue; expose its actual row plan directly to the kernel.
        tilingData_.set_reduceRows(waveEpilogueRows);
        tilingData_.set_reduceRowsAligned((waveEpilogueRows + 7U) / 8U * 8U);
        tilingData_.set_epiloguePrefetch(0U);
    }
    const uint64_t epiloguePeak = mega_mhc::EpilogueUbBytes(d_, cubePlan_.partialGroupCount, cubePlan_.fixpipeRowStride,
                                                            reduceRows, cubePlan_.kGrid);
    const uint64_t rms0Peak = mega_mhc::UbAllocation(d_ * 2U) + mega_mhc::UbAllocation(8U);
    const uint64_t waveEpiloguePeak =
        waveEpilogueRows != 0U ? mega_mhc::EpilogueUbBytes(d_, cubePlan_.partialGroupCount, cubePlan_.fixpipeRowStride,
                                                           waveEpilogueRows, cubePlan_.kGrid) :
                                 0U;
    const uint64_t pipelinePeak =
        request.wavePipeline ? mega_mhc::ProducerUbBytes(d_, false, true) + projectionPeak + waveEpiloguePeak : 0U;
    const uint64_t ubPeak = std::max({producerPeak, projectionPeak, epiloguePeak, rms0Peak, pipelinePeak});
    if (ubPeak > availableUb) {
        OP_LOGE(context_->GetNodeName(), "planned stage peak UB %llu exceeds available UB %llu",
                static_cast<unsigned long long>(ubPeak), static_cast<unsigned long long>(availableUb));
        return ge::GRAPH_FAILED;
    }
    tilingData_.set_ubPeakBytes(ubPeak);
    const uint64_t l1Elements = cubePlan_.l1BOffsetElements + 2ULL * cubePlan_.l1BSlotElements;
    tilingData_.set_l1PeakBytes(l1Elements * sizeof(float));
    return BuildWorkspacePlan();
}

/** Compute disjoint FP32 partial and optional GM-staging workspace regions with 64-bit offsets. */
ge::graphStatus MegaMhcTiling::BuildWorkspacePlan()
{
    uint64_t invRms0Bytes = 0;
    if (!CheckedBytes(totalRows_, 1, 1, sizeof(float), invRms0Bytes)) {
        OP_LOGE(context_->GetNodeName(), "mega_mhc RMS0 workspace size overflows uint64");
        return ge::GRAPH_FAILED;
    }
    if (hasMid_) {
        invRms0Bytes = 0; // RMS0 stays local to the five-op producer.
    } else if (tilingData_.get_wavePipeline() != 0U) {
        // Wave epochs reuse RMS0 storage: every AIV writes one 128-byte slot, even for small T.
        invRms0Bytes = std::max<uint64_t>(invRms0Bytes, static_cast<uint64_t>(cubePlan_.activeAiv) * 128U);
    }
    uint64_t partialMmBytes = 0;
    uint64_t partialRmsBytes = 0;
    if (!CheckedBytes(cubePlan_.partialGroupCount, totalRows_, cubePlan_.fixpipeRowStride, sizeof(float),
                      partialMmBytes) ||
        !CheckedBytes(cubePlan_.kGrid, totalRows_, 1, sizeof(float), partialRmsBytes)) {
        OP_LOGE(context_->GetNodeName(), "mega_mhc workspace size overflows uint64");
        return ge::GRAPH_FAILED;
    }
    uint64_t partialMmOffset = 0;
    if (!AlignWorkspace(invRms0Bytes, kWorkspaceAlign, partialMmOffset)) {
        OP_LOGE(context_->GetNodeName(), "mega_mhc RMS0 workspace alignment overflows uint64");
        return ge::GRAPH_FAILED;
    }
    if (partialMmBytes > std::numeric_limits<uint64_t>::max() - partialMmOffset) {
        OP_LOGE(context_->GetNodeName(), "mega_mhc projection workspace end overflows uint64");
        return ge::GRAPH_FAILED;
    }
    uint64_t partialRmsOffset = 0;
    if (!AlignWorkspace(partialMmOffset + partialMmBytes, kWorkspaceAlign, partialRmsOffset)) {
        OP_LOGE(context_->GetNodeName(), "mega_mhc partial workspace alignment overflows uint64");
        return ge::GRAPH_FAILED;
    }
    if (partialRmsBytes > std::numeric_limits<uint64_t>::max() - partialRmsOffset) {
        OP_LOGE(context_->GetNodeName(), "mega_mhc RMS workspace end overflows uint64");
        return ge::GRAPH_FAILED;
    }
    uint64_t stageOffset = 0;
    if (!AlignWorkspace(partialRmsOffset + partialRmsBytes, kWorkspaceAlign, stageOffset)) {
        OP_LOGE(context_->GetNodeName(), "mega_mhc stage workspace alignment overflows uint64");
        return ge::GRAPH_FAILED;
    }
    uint64_t stageElementsBase = 0;
    if (!CheckedBytes(cubePlan_.activeAic, kSlotCount, cubePlan_.mTile, 1, stageElementsBase)) {
        OP_LOGE(context_->GetNodeName(), "mega_mhc staging workspace size overflows uint64");
        return ge::GRAPH_FAILED;
    }
    // `CheckedBytes` receives three extents; multiply by kL1 and FP32 separately
    // so every size conversion remains checked before TilingData serialization.
    if (cubePlan_.l1K != 0 && stageElementsBase > std::numeric_limits<uint64_t>::max() / cubePlan_.l1K) {
        OP_LOGE(context_->GetNodeName(), "mega_mhc staging element count overflows uint64");
        return ge::GRAPH_FAILED;
    }
    const uint64_t stageElements = stageElementsBase * cubePlan_.l1K;
    if (stageElements > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        OP_LOGE(context_->GetNodeName(), "mega_mhc staging bytes overflow uint64");
        return ge::GRAPH_FAILED;
    }
    const uint64_t stageBytes = stageElements * sizeof(float);
    if (stageBytes > std::numeric_limits<uint64_t>::max() - stageOffset) {
        OP_LOGE(context_->GetNodeName(), "mega_mhc workspace end overflows uint64");
        return ge::GRAPH_FAILED;
    }
    const uint64_t userBytes = stageOffset + stageBytes;
    if (userBytes > std::numeric_limits<uint64_t>::max() - kRuntimeWorkspaceBytes) {
        OP_LOGE(context_->GetNodeName(), "mega_mhc total workspace overflows uint64");
        return ge::GRAPH_FAILED;
    }
    workspaceBytes_ = userBytes + kRuntimeWorkspaceBytes;
    tilingData_.set_invRms0Offset(0);
    tilingData_.set_invRms0Bytes(invRms0Bytes);
    tilingData_.set_partialMmOffset(partialMmOffset);
    tilingData_.set_partialRmsOffset(partialRmsOffset);
    tilingData_.set_stageOffset(stageOffset);
    tilingData_.set_partialMmBytes(partialMmBytes);
    tilingData_.set_partialRmsBytes(partialRmsBytes);
    tilingData_.set_stageBytes(stageBytes);
    tilingData_.set_userWorkspaceBytes(userBytes);
    return ge::GRAPH_SUCCESS;
}

/** This kernel uses only explicit host tiling and Basic/RegBase APIs, so no library tiling is needed. */
ge::graphStatus MegaMhcTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

/** Select the schedule, optional Mid and Cube precision at compile time through the shared key declaration. */
uint64_t MegaMhcTiling::GetTilingKey() const
{
    static_assert(static_cast<uint32_t>(mega_mhc::ScheduleMode::MK) == MEGA_MHC_SPLIT_M_K);
    static_assert(static_cast<uint32_t>(mega_mhc::ScheduleMode::ND) == MEGA_MHC_SPLIT_ND);
    static_assert(static_cast<uint32_t>(mega_mhc::ScheduleMode::BS) == MEGA_MHC_SPLIT_BS);
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(scheduleMode_), hasMid_,
                              projectionMode_ == mega_mhc::ProjectionMode::HF32);
}

/** Workspace size is finalized while building the checked user-workspace layout. */
ge::graphStatus MegaMhcTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

/** Serialize TilingData, launch the M/K block count, and publish its workspace requirement. */
ge::graphStatus MegaMhcTiling::PostTiling()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    if (tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity() ||
        tilingData_.GetDataSize() % sizeof(uint64_t) != 0) {
        OP_LOGE(context_->GetNodeName(), "mega_mhc TilingData size must be uint64 aligned");
        return ge::GRAPH_FAILED;
    }
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(cubePlan_.activeAic);
    context_->SetScheduleMode(1);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context_->GetNodeName(), "workspace size array is null"),
                return ge::GRAPH_FAILED);
    if (workspaceBytes_ > std::numeric_limits<size_t>::max()) {
        OP_LOGE(context_->GetNodeName(), "mega_mhc workspace does not fit size_t");
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = static_cast<size_t>(workspaceBytes_);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
