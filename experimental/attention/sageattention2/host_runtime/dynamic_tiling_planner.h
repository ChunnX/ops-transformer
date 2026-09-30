#pragma once

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../core/platform_traits.h"
#include "../core/numerical_types.h"
#include "../op_kernel/sage_attention2_template_tiling_key.h"
#include "hardware_topology.h"
#include "workspace_calculator.h"

namespace sage::host_runtime {

using namespace sage::core;

inline ge::graphStatus LogTilingFailure(int sourceLine)
{
    std::fprintf(stderr, "[SageAttention2 Tiling] Validation gate failed at line %d\n", sourceLine);
    return ge::GRAPH_FAILED;
}

inline ge::graphStatus ComputeSageAttention2Tiling(gert::TilingContext *context)
{
    if (!context || !context->GetAttrs()) {
        return LogTilingFailure(__LINE__);
    }

    for (int tensorIdx = 0; tensorIdx < 3; ++tensorIdx) {
        if (!context->GetInputShape(tensorIdx) || !context->GetInputDesc(tensorIdx)) {
            return LogTilingFailure(__LINE__);
        }
        const auto &storageShape = context->GetInputShape(tensorIdx)->GetStorageShape();
        if (storageShape.GetDimNum() != 4) {
            return LogTilingFailure(__LINE__);
        }
        for (int dimIdx = 0; dimIdx < 4; ++dimIdx) {
            if (storageShape[dimIdx] <= 0 || storageShape[dimIdx] >= UINT32_MAX - 64) {
                return LogTilingFailure(__LINE__);
            }
        }
    }

    const auto &qShape = context->GetInputShape(0)->GetStorageShape();
    const auto &kShape = context->GetInputShape(1)->GetStorageShape();
    const auto &vShape = context->GetInputShape(2)->GetStorageShape();

    auto inputDataType = context->GetInputDesc(0)->GetDataType();
    if ((inputDataType != ge::DT_FLOAT16 && inputDataType != ge::DT_BF16) ||
        context->GetInputDesc(1)->GetDataType() != inputDataType ||
        context->GetInputDesc(2)->GetDataType() != inputDataType ||
        kShape != vShape || qShape[0] != kShape[0] || qShape[3] != kShape[3] ||
        qShape[3] > MAX_HEAD_DIM || (qShape[1] % kShape[1] != 0) ||
        uint64_t(kShape[0]) * kShape[1] >= UINT32_MAX) {
        return LogTilingFailure(__LINE__);
    }

    if (qShape[2] > UINT32_MAX - 127 || kShape[2] > UINT32_MAX - 511) {
        return LogTilingFailure(__LINE__);
    }

    SageAttention2TilingData tilingPayload{};
    auto &params = tilingPayload.params;

    params.b = qShape[0];
    params.hq = qShape[1];
    params.hk = kShape[1];
    params.sq = qShape[2];
    params.sk = kShape[2];
    params.d = qShape[3];
    params.dp = PaddedHeadDim(params.d);

    const auto *opAttrs = context->GetAttrs();
    auto attrSmScale = opAttrs->GetAttrPointer<float>(0);
    auto attrSmoothK = opAttrs->GetAttrPointer<bool>(1);
    auto attrCausal = opAttrs->GetAttrPointer<bool>(2);
    auto attrReturnLse = opAttrs->GetAttrPointer<bool>(3);
    auto attrQuantGran = opAttrs->GetAttrPointer<int64_t>(4);
    auto attrPvAccum = opAttrs->GetAttrPointer<int64_t>(5);
    auto attrSmoothV = opAttrs->GetAttrPointer<bool>(6);

    if (!attrSmScale || !attrSmoothK || !attrCausal || !attrReturnLse ||
        !attrQuantGran || !attrPvAccum || !attrSmoothV ||
        !std::isfinite(*attrSmScale) || (*attrCausal && params.sq != params.sk)) {
        return LogTilingFailure(__LINE__);
    }

    if (!IsValidQuantGran(*attrQuantGran) || !IsValidPvAccum(*attrPvAccum)) {
        return LogTilingFailure(__LINE__);
    }

    const float rawScale = *attrSmScale;
    const float absScale = std::abs(rawScale);
    const float balancedFactor = std::sqrt(absScale * 2048.0f);
    if (!std::isfinite(balancedFactor) || balancedFactor > SAGE_FP16_MAX) {
        return LogTilingFailure(__LINE__);
    }

    params.kFactor = balancedFactor;
    params.qFactor = (rawScale < 0.0f) ? -balancedFactor : balancedFactor;
    params.smoothK = *attrSmoothK ? 1U : 0U;
    params.causal = *attrCausal ? 1U : 0U;
    params.returnLse = *attrReturnLse ? 1U : 0U;

    const bool isPerWarp = (*attrQuantGran == static_cast<int64_t>(SageQuantGran::PER_WARP));
    params.qg = QTokensPerScale(isPerWarp);
    params.kg = KTokensPerScale(isPerWarp);
    params.accum = static_cast<SagePvAccum>(*attrPvAccum);
    params.smoothV = (*attrSmoothV && params.accum == SagePvAccum::FP32) ? 1U : 0U;

    params.qs[2] = params.d;
    params.qs[1] = uint64_t(params.sq) * params.d;
    params.qs[0] = uint64_t(params.hq) * params.qs[1];
    params.ks[2] = params.vs[2] = params.d;
    params.ks[1] = params.vs[1] = uint64_t(params.sk) * params.d;
    params.ks[0] = params.vs[0] = uint64_t(params.hk) * params.ks[1];

    uint32_t vectorCount = 0;
    if (!DiscoverHardwareTopology(context, params.cores, vectorCount)) {
        return LogTilingFailure(__LINE__);
    }
    context->SetBlockDim(params.cores);

    params.qp = RoundUpTo(params.sq, 128U);
    params.kp = RoundUpTo(params.sk, 512U);

    const bool isWideStream = (params.dp == 128 && params.d == params.dp &&
                               params.sq % 128 == 0 && params.sk % 512 == 0 && params.sk >= 4096);
    context->SetTilingKey(GET_TPL_TILING_KEY(static_cast<uint32_t>(inputDataType), isWideStream));
    context->SetScheduleMode(1);

    WorkspacePlanner planner(0);
    const uint64_t qStagingBatch = isWideStream ? 0ULL : params.b;

    if (!planner.Suballocate(tilingPayload.mean, {params.b, params.hk, params.dp, sizeof(float)}) ||
        !planner.Suballocate(tilingPayload.vMean, {params.b, params.hk, params.dp, sizeof(float)}) ||
        !planner.Suballocate(tilingPayload.vMxScale, {params.b, params.hk, params.kp / 32, params.dp}) ||
        !planner.Suballocate(tilingPayload.q8, {qStagingBatch, params.hq, params.qp, params.dp}) ||
        !planner.Suballocate(tilingPayload.k8, {params.b, params.hk, params.kp, params.dp}) ||
        !planner.Suballocate(tilingPayload.v8, {params.b, params.hk, params.kp, params.dp}) ||
        !planner.Suballocate(tilingPayload.qScale, {qStagingBatch, params.hq, params.qp, sizeof(uint16_t)}) ||
        !planner.Suballocate(tilingPayload.kScale, {params.b, params.hk, params.kp, sizeof(uint16_t)})) {
        return LogTilingFailure(__LINE__);
    }

    thread_local static uint64_t tlSystemWorkspace = 0;
    uint64_t systemReserved = tlSystemWorkspace;
    if (systemReserved == 0) {
        systemReserved = DEFAULT_SYSTEM_WORKSPACE;
        if (context->GetPlatformInfo()) {
            systemReserved = platform_ascendc::PlatformAscendC(context->GetPlatformInfo()).GetLibApiWorkSpaceSize();
        }
        tlSystemWorkspace = systemReserved;
    }

    uint64_t totalWorkspaceBytes = planner.GetTotalWorkspaceBytes();
    if (totalWorkspaceBytes > std::numeric_limits<size_t>::max() - systemReserved) {
        return LogTilingFailure(__LINE__);
    }

    auto workspaceSizes = context->GetWorkspaceSizes(1);
    auto rawPayload = context->GetRawTilingData();
    if (!workspaceSizes || !rawPayload || rawPayload->GetCapacity() < sizeof(tilingPayload)) {
        return LogTilingFailure(__LINE__);
    }
    std::memcpy(rawPayload->GetData(), &tilingPayload, sizeof(tilingPayload));
    rawPayload->SetDataSize(sizeof(tilingPayload));
    workspaceSizes[0] = static_cast<size_t>(totalWorkspaceBytes + systemReserved);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus ParseHardwareSpec(gert::TilingParseContext *context)
{
    if (!context || !context->GetPlatformInfo()) {
        return LogTilingFailure(__LINE__);
    }
    auto *spec = context->GetCompiledInfo<PlatformHardwareSpec>();
    if (!spec) {
        return LogTilingFailure(__LINE__);
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    spec->aicCoreCount = platform.GetCoreNumAic();
    spec->aivCoreCount = platform.GetCoreNumAiv();
    return ge::GRAPH_SUCCESS;
}

} // namespace sage::host_runtime

namespace optiling {
using sage::host_runtime::ComputeSageAttention2Tiling;
using sage::host_runtime::ParseHardwareSpec;
}
