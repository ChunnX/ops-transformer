/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <initializer_list>
#include <limits>
#include "acl/acl_rt.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../common/constants.h"
#include "../common/types.h"
#include "../op_kernel/sage_attention2_tiling_data.h"
#include "../op_kernel/sage_attention2_template_tiling_key.h"

namespace optiling {
namespace {

using namespace sage;

constexpr uint64_t WORKSPACE_MEMORY_ALIGNMENT = 512ULL;
constexpr uint64_t DEFAULT_SYSTEM_WORKSPACE = 16ULL * 1024ULL * 1024ULL;

inline ge::graphStatus LogTilingFailure(int sourceLine)
{
    std::fprintf(stderr, "[SageAttention2 Tiling] Validation gate failed at line %d\n", sourceLine);
    return ge::GRAPH_FAILED;
}

struct PlatformHardwareSpec {
    uint32_t aicCoreCount = 0;
    uint32_t aivCoreCount = 0;
};

// Discovers physical AIC and AIV core topology across Ascend platform descriptors and runtime queries.
bool DiscoverHardwareTopology(const gert::TilingContext *context, uint32_t &aicCores, uint32_t &aivCores)
{
    thread_local static uint32_t tlCachedAic = 0;
    thread_local static uint32_t tlCachedAiv = 0;

    if (tlCachedAic > 0 && tlCachedAiv >= 2 * tlCachedAic) {
        aicCores = tlCachedAic;
        aivCores = tlCachedAiv;
        return true;
    }

    if (context->GetPlatformInfo()) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        aicCores = platform.GetCoreNumAic();
        aivCores = platform.GetCoreNumAiv();
    }

    if (!aicCores && context->GetCompileInfo()) {
        const auto *spec = static_cast<const PlatformHardwareSpec *>(context->GetCompileInfo());
        aicCores = spec->aicCoreCount;
        aivCores = spec->aivCoreCount;
    }

    // Dynamic fallback query via runtime symbols
    if (!aicCores || aivCores < 2 * aicCores) {
        static auto fnGetDevice = reinterpret_cast<aclError (*)(int32_t *)>(dlsym(RTLD_DEFAULT, "aclrtGetDevice"));
        static auto fnGetDevInfo = reinterpret_cast<aclError (*)(uint32_t, aclrtDevAttr, int64_t *)>(
            dlsym(RTLD_DEFAULT, "aclrtGetDeviceInfo"));

        int32_t devId = 0;
        int64_t hwAic = 0, hwAiv = 0;
        if (fnGetDevice && fnGetDevInfo && fnGetDevice(&devId) == ACL_SUCCESS &&
            fnGetDevInfo(devId, ACL_DEV_ATTR_CUBE_CORE_NUM, &hwAic) == ACL_SUCCESS &&
            fnGetDevInfo(devId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &hwAiv) == ACL_SUCCESS &&
            hwAic > 0 && hwAic < UINT32_MAX && hwAiv >= 2 * hwAic && hwAiv < UINT32_MAX) {
            aicCores = static_cast<uint32_t>(hwAic);
            aivCores = static_cast<uint32_t>(hwAiv);
        }
    }

    if (aicCores > 0 && aivCores >= 2 * aicCores) {
        tlCachedAic = aicCores;
        tlCachedAiv = aivCores;
        return true;
    }
    return false;
}

// Memory block layout coordinator for device Global Memory workspace.
class WorkspacePlanner {
public:
    explicit WorkspacePlanner(uint64_t baseOffset = 0) : m_currentOffset(baseOffset) {}

    bool Suballocate(uint64_t &outOffset, std::initializer_list<uint64_t> dimensions)
    {
        uint64_t tensorBytes = 1;
        for (uint64_t dim : dimensions) {
            if (dim > 0 && tensorBytes > std::numeric_limits<uint64_t>::max() / dim) {
                return false;
            }
            tensorBytes *= dim;
        }

        if (tensorBytes > std::numeric_limits<uint64_t>::max() - (WORKSPACE_MEMORY_ALIGNMENT - 1)) {
            return false;
        }
        tensorBytes = RoundUpTo(tensorBytes, WORKSPACE_MEMORY_ALIGNMENT);

        if (m_currentOffset > std::numeric_limits<uint64_t>::max() - tensorBytes) {
            return false;
        }
        outOffset = m_currentOffset;
        m_currentOffset += tensorBytes;
        return true;
    }

    uint64_t GetTotalWorkspaceBytes() const noexcept { return m_currentOffset; }

private:
    uint64_t m_currentOffset;
};

// Main Tiling Entry Point for SageAttention2
ge::graphStatus ComputeSageAttention2Tiling(gert::TilingContext *context)
{
    if (!context || !context->GetAttrs()) {
        return LogTilingFailure(__LINE__);
    }

    // Validate 4D tensor input shapes: Query (0), Key (1), Value (2)
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

    // Compute FP16 pre-balanced factor from sm_scale
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

    // Strides
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

    // Wide N512 stream activation: head_dim == 128, aligned sequence, minimum Sk = 4096
    const bool isWideStream = (params.dp == 128 && params.d == params.dp &&
                               params.sq % 128 == 0 && params.sk % 512 == 0 && params.sk >= 4096);
    context->SetTilingKey(GET_TPL_TILING_KEY(static_cast<uint32_t>(inputDataType), isWideStream));
    context->SetScheduleMode(1);

    WorkspacePlanner planner(0);
    // When wide stream is active, on-chip streaming Q quantization eliminates GM roundtrip for Q
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

ge::graphStatus ParseHardwareSpec(gert::TilingParseContext *context)
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

} // namespace
} // namespace optiling

IMPL_OP_OPTILING(SageAttention2).Tiling(optiling::ComputeSageAttention2Tiling)
    .TilingParse<optiling::PlatformHardwareSpec>(optiling::ParseHardwareSpec);
