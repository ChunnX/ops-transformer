#include <cmath>
#include <cstdint>
#include <dlfcn.h>
#include "acl/acl_rt.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/sage_attention2_tiling_data.h"
#include "../op_kernel/sage_attention2_template_tiling_key.h"

namespace sage::tiling_plan {

using namespace sage::core;

struct DeviceHardwareMetrics {
    uint32_t aicCoreCount = 0U;
    uint32_t aivCoreCount = 0U;
};

struct BufferAllocationManager {
    uint64_t currentOffset = 0ULL;

    inline bool AllocateAligned(uint64_t &targetOffset, uint64_t totalElements, uint64_t unitBytes = 1ULL) {
        if (unitBytes > 0 && totalElements > UINT64_MAX / unitBytes) {
            return (false);
        }
        const uint64_t rawByteSize = totalElements * unitBytes;
        const uint64_t alignedByteSize = RoundUpTo(rawByteSize, 512ULL);
        if (currentOffset > UINT64_MAX - alignedByteSize) {
            return (false);
        }
        targetOffset = (currentOffset + 0ULL);
        currentOffset += alignedByteSize;
        return (true);
    }
};

struct RuntimeCoreResolver {
    static bool ResolveCores(const gert::TilingContext *ctx, uint32_t &aicResult, uint32_t &aivResult) {
        thread_local static uint32_t localAic = 0U, localAiv = 0U;
        if (localAic > 0U && localAiv >= (localAic * 2U)) {
            aicResult = (localAic + 0U);
            aivResult = (localAiv + 0U);
            return (true);
        }

        if (ctx && ctx->GetPlatformInfo()) {
            platform_ascendc::PlatformAscendC ascPlatform(ctx->GetPlatformInfo());
            const uint32_t queriedAic = uint32_t(ascPlatform.GetCoreNumAic());
            const uint32_t queriedAiv = uint32_t(ascPlatform.GetCoreNumAiv());
            if (queriedAic > 0U && queriedAiv >= (queriedAic * 2U)) {
                aicResult = (queriedAic + 0U);
                aivResult = (queriedAiv + 0U);
            }
        }

        if (aicResult == 0U && ctx && ctx->GetCompileInfo()) {
            const auto *compiledSpec = static_cast<const DeviceHardwareMetrics *>(ctx->GetCompileInfo());
            if (compiledSpec && compiledSpec->aicCoreCount > 0U && compiledSpec->aivCoreCount >= (compiledSpec->aicCoreCount * 2U)) {
                aicResult = compiledSpec->aicCoreCount;
                aivResult = compiledSpec->aivCoreCount;
            }
        }

        if (aicResult == 0U || aivResult < (aicResult * 2U)) {
            using DevQueryFn = aclError (*)(int32_t *);
            using DevAttrQueryFn = aclError (*)(uint32_t, aclrtDevAttr, int64_t *);
            void *rtldHandle = RTLD_DEFAULT;
            auto queryDev = (DevQueryFn)dlsym(rtldHandle, "aclrtGetDevice");
            auto queryAttr = (DevAttrQueryFn)dlsym(rtldHandle, "aclrtGetDeviceInfo");

            int32_t activeDevId = -1;
            int64_t measuredAic = -1LL;
            int64_t measuredAiv = -1LL;
            const bool querySuccess = (queryDev && queryAttr && queryDev(&activeDevId) == ACL_SUCCESS &&
                                       queryAttr(uint32_t(activeDevId), ACL_DEV_ATTR_CUBE_CORE_NUM, &measuredAic) == ACL_SUCCESS &&
                                       queryAttr(uint32_t(activeDevId), ACL_DEV_ATTR_VECTOR_CORE_NUM, &measuredAiv) == ACL_SUCCESS);
            if (querySuccess && measuredAic > 0LL && measuredAiv >= (measuredAic * 2LL) && measuredAiv < int64_t(UINT32_MAX)) {
                aicResult = uint32_t(measuredAic);
                aivResult = uint32_t(measuredAiv);
            }
        }

        if (aicResult > 0U && aivResult >= (aicResult * 2U)) {
            localAic = (aicResult + 0U);
            localAiv = (aivResult + 0U);
            return (true);
        }
        return (false);
    }
};

template <typename T>
static inline const T* FetchAttributePointer(const gert::RuntimeAttrs *attrs, size_t attrIndex) {
    return (attrs != nullptr) ? attrs->GetAttrPointer<T>(attrIndex) : nullptr;
}

static inline void ComputeCanonicalStrides(uint64_t (&strideArr)[3], uint32_t headCount, uint32_t seqLen, uint32_t headDim) {
    strideArr[2] = uint64_t(headDim);
    strideArr[1] = uint64_t(seqLen) * strideArr[2];
    strideArr[0] = uint64_t(headCount) * strideArr[1];
}

struct SageAttention2TilingPipeline {
    static bool Execute(gert::TilingContext *context) {
        if (!context || !context->GetAttrs()) {
            return (false);
        }

        const auto *shapeQ = context->GetInputShape(0);
        const auto *shapeK = context->GetInputShape(1);
        const auto *shapeV = context->GetInputShape(2);
        const auto *descQ = context->GetInputDesc(0);
        const auto *descK = context->GetInputDesc(1);
        const auto *descV = context->GetInputDesc(2);

        if (!shapeQ || !shapeK || !shapeV || !descQ || !descK || !descV) {
            return (false);
        }

        const auto &sQ = (*shapeQ).GetStorageShape();
        const auto &sK = (*shapeK).GetStorageShape();
        const auto &sV = (*shapeV).GetStorageShape();

        if (sQ.GetDimNum() != 4 || sK.GetDimNum() != 4 || sV.GetDimNum() != 4) {
            return (false);
        }

        for (size_t axis = 0; axis < 4; ++axis) {
            if (sQ[axis] <= 0 || sK[axis] <= 0 || sV[axis] <= 0 ||
                sQ[axis] >= UINT32_MAX - 64 || sK[axis] >= UINT32_MAX - 64 || sV[axis] >= UINT32_MAX - 64) {
                return (false);
            }
        }

        const auto inputDtype = descQ->GetDataType();
        const bool validDtype = (inputDtype == ge::DT_FLOAT16 || inputDtype == ge::DT_BF16);
        if (!validDtype || descK->GetDataType() != inputDtype || descV->GetDataType() != inputDtype) {
            return (false);
        }

        const bool shapesCompatible = (sK == sV) && (sQ[0] == sK[0]) && (sQ[3] == sK[3]) &&
                                      (sQ[3] <= sage::core::MAX_HEAD_DIM) && ((sQ[1] % sK[1]) == 0);
        if (!shapesCompatible) {
            return (false);
        }

        const uint64_t batchHeadProduct = uint64_t(sK[0]) * uint64_t(sK[1]);
        if (batchHeadProduct >= UINT32_MAX || sQ[2] > UINT32_MAX - 127 || sK[2] > UINT32_MAX - 511) {
            return (false);
        }

        sage::core::SageAttention2TilingData tilingPayload{};
        sage::core::SageParams &params = tilingPayload.params;
        params.b = uint32_t(sQ[0]);
        params.hq = uint32_t(sQ[1]);
        params.hk = uint32_t(sK[1]);
        params.sq = uint32_t(sQ[2]);
        params.sk = uint32_t(sK[2]);
        params.d = uint32_t(sQ[3]);
        params.dp = sage::core::PaddedHeadDim(params.d);

        const auto *opAttrs = (*context).GetAttrs();
        const float *attrSmScale = FetchAttributePointer<float>(opAttrs, 0);
        const bool *attrSmoothK = FetchAttributePointer<bool>(opAttrs, 1);
        const bool *attrCausal = FetchAttributePointer<bool>(opAttrs, 2);
        const bool *attrReturnLse = FetchAttributePointer<bool>(opAttrs, 3);
        const int64_t *attrQuantGran = FetchAttributePointer<int64_t>(opAttrs, 4);
        const int64_t *attrPvAccum = FetchAttributePointer<int64_t>(opAttrs, 5);
        const bool *attrSmoothV = FetchAttributePointer<bool>(opAttrs, 6);

        if (!attrSmScale || !attrSmoothK || !attrCausal || !attrReturnLse || !attrQuantGran || !attrPvAccum || !attrSmoothV) {
            return (false);
        }
        if (!std::isfinite(*attrSmScale) || (*attrCausal && params.sq != params.sk)) {
            return (false);
        }
        if (!sage::core::IsValidQuantGran(*attrQuantGran) || !sage::core::IsValidPvAccum(*attrPvAccum)) {
            return (false);
        }

        const float scaleMagnitude = std::abs(*attrSmScale);
        const float sqrtBalanced = std::sqrt(scaleMagnitude * 2048.0f);
        if (!std::isfinite(sqrtBalanced) || sqrtBalanced > sage::core::SAGE_FP16_MAX) {
            return (false);
        }

        params.kFactor = float(sqrtBalanced);
        params.qFactor = (*attrSmScale < 0.0f) ? -float(sqrtBalanced) : float(sqrtBalanced);
        params.smoothK = (*attrSmoothK) ? 1U : 0U;
        params.causal = (*attrCausal) ? 1U : 0U;
        params.returnLse = (*attrReturnLse) ? 1U : 0U;

        const bool usePerWarp = (*attrQuantGran == int64_t(sage::core::SageQuantGran::PER_WARP));
        params.qg = sage::core::QTokensPerScale(usePerWarp);
        params.kg = sage::core::KTokensPerScale(usePerWarp);
        params.accum = sage::core::SagePvAccum(*attrPvAccum);
        params.smoothV = (*attrSmoothV && params.accum == sage::core::SagePvAccum::FP32) ? 1U : 0U;

        ComputeCanonicalStrides(params.qs, params.hq, params.sq, params.d);
        ComputeCanonicalStrides(params.ks, params.hk, params.sk, params.d);
        ComputeCanonicalStrides(params.vs, params.hk, params.sk, params.d);

        uint32_t vectorCoreCount = 0U;
        if (!RuntimeCoreResolver::ResolveCores(context, params.cores, vectorCoreCount)) {
            return (false);
        }
        const uint32_t blockDimVal = params.cores;
        context->SetBlockDim(blockDimVal);

        params.qp = sage::core::RoundUpTo(params.sq, 128U);
        params.kp = sage::core::RoundUpTo(params.sk, 512U);

        const bool isWideStreamMode = (params.dp == 128U && params.d == params.dp &&
                                       (params.sq % 128U == 0U) && (params.sk % 512U == 0U) && (params.sk >= 4096U));
        const uint64_t operatorTilingKey = GET_TPL_TILING_KEY(uint32_t(inputDtype), isWideStreamMode);
        context->SetTilingKey(operatorTilingKey);
        const uint32_t defaultScheduleMode = 1U;
        context->SetScheduleMode(defaultScheduleMode);

        BufferAllocationManager memPlanner{0ULL};
        const uint64_t queryStagingBatches = isWideStreamMode ? 0ULL : uint64_t(params.b);

        bool planValid = true;
        planValid = planValid && memPlanner.AllocateAligned(tilingPayload.mean, uint64_t(params.b) * params.hk * params.dp, sizeof(float));
        planValid = planValid && memPlanner.AllocateAligned(tilingPayload.vMean, uint64_t(params.b) * params.hk * params.dp, sizeof(float));
        planValid = planValid && memPlanner.AllocateAligned(tilingPayload.vMxScale, uint64_t(params.b) * params.hk * (params.kp / 32U) * params.dp, 1ULL);
        planValid = planValid && memPlanner.AllocateAligned(tilingPayload.q8, queryStagingBatches * params.hq * params.qp * params.dp, 1ULL);
        planValid = planValid && memPlanner.AllocateAligned(tilingPayload.k8, uint64_t(params.b) * params.hk * params.kp * params.dp, 1ULL);
        planValid = planValid && memPlanner.AllocateAligned(tilingPayload.v8, uint64_t(params.b) * params.hk * params.kp * params.dp, 1ULL);
        planValid = planValid && memPlanner.AllocateAligned(tilingPayload.qScale, queryStagingBatches * params.hq * params.qp, sizeof(uint16_t));
        planValid = planValid && memPlanner.AllocateAligned(tilingPayload.kScale, uint64_t(params.b) * params.hk * params.kp, sizeof(uint16_t));

        if (!planValid) {
            return (false);
        }

        uint64_t systemReservedWorkspace = (16ULL << 20);
        if (context->GetPlatformInfo()) {
            platform_ascendc::PlatformAscendC platformInspect(context->GetPlatformInfo());
            systemReservedWorkspace = platformInspect.GetLibApiWorkSpaceSize();
        }

        const uint64_t totalWorkspaceRequired = memPlanner.currentOffset;
        if (totalWorkspaceRequired > UINT64_MAX - systemReservedWorkspace) {
            return (false);
        }

        auto workspaceSizes = (*context).GetWorkspaceSizes(1U);
        auto rawData = (*context).GetRawTilingData();
        const uint64_t payloadByteSize = sizeof(SageAttention2TilingData);
        if (!workspaceSizes || !rawData || rawData->GetCapacity() < payloadByteSize) {
            return (false);
        }

        *reinterpret_cast<SageAttention2TilingData *>(rawData->GetData()) = tilingPayload;
        rawData->SetDataSize(payloadByteSize);
        workspaceSizes[0] = static_cast<size_t>(totalWorkspaceRequired + systemReservedWorkspace);
        return (true);
    }
};

inline ge::graphStatus ComputeSageAttention2Tiling(gert::TilingContext *context) {
    const bool isTilingOk = SageAttention2TilingPipeline::Execute(context);
    return isTilingOk ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

inline ge::graphStatus ParseHardwareSpec(gert::TilingParseContext *context) {
    if (!context || !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }
    auto *hardwareSpec = context->GetCompiledInfo<DeviceHardwareMetrics>();
    if (!hardwareSpec) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platformQuery(context->GetPlatformInfo());
    hardwareSpec->aicCoreCount = uint32_t(platformQuery.GetCoreNumAic());
    hardwareSpec->aivCoreCount = uint32_t(platformQuery.GetCoreNumAiv());
    return ge::GRAPH_SUCCESS;
}

} // namespace sage::tiling_plan

IMPL_OP_OPTILING(SageAttention2).Tiling(sage::tiling_plan::ComputeSageAttention2Tiling).TilingParse<sage::tiling_plan::DeviceHardwareMetrics>(sage::tiling_plan::ParseHardwareSpec);
