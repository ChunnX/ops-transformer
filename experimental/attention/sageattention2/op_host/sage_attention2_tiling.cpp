#include <cmath>
#include <cstdint>
#include <dlfcn.h>
#include <utility>
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

template <uint64_t Alignment = 512ULL>
struct BufferAllocationManager {
    uint64_t currentOffset = 0ULL;

    template <typename... DimTs>
    inline bool AllocateAligned(uint64_t &targetOffset, DimTs... dimensions) noexcept {
        uint64_t totalBytes = 1ULL;
        auto multiplyFactor = [&](uint64_t factor) noexcept -> bool {
            if (factor == 0ULL) return true;
            if (totalBytes > UINT64_MAX / factor) return false;
            totalBytes *= factor;
            return true;
        };
        if (!(multiplyFactor(static_cast<uint64_t>(dimensions)) && ...)) {
            return false;
        }
        const uint64_t alignedBytes = sage::core::RoundUpTo(totalBytes, Alignment);
        if (currentOffset > UINT64_MAX - alignedBytes) {
            return false;
        }
        targetOffset = currentOffset;
        currentOffset += alignedBytes;
        return true;
    }
};

struct RuntimeCoreResolver {
    static bool ResolveCores(const gert::TilingContext *ctx, uint32_t &aicResult, uint32_t &aivResult) {
        thread_local static uint32_t localAic = 0U, localAiv = 0U;
        if (localAic > 0U && localAiv >= (localAic * 2U)) {
            aicResult = localAic;
            aivResult = localAiv;
            return true;
        }

        if (ctx && ctx->GetPlatformInfo()) {
            platform_ascendc::PlatformAscendC ascPlatform(ctx->GetPlatformInfo());
            const uint32_t queriedAic = uint32_t(ascPlatform.GetCoreNumAic());
            const uint32_t queriedAiv = uint32_t(ascPlatform.GetCoreNumAiv());
            if (queriedAic > 0U && queriedAiv >= (queriedAic * 2U)) {
                aicResult = queriedAic;
                aivResult = queriedAiv;
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
            localAic = aicResult;
            localAiv = aivResult;
            return true;
        }
        return false;
    }
};

template <typename... AttrTypes>
inline bool ExtractRuntimeAttributes(const gert::RuntimeAttrs *attrs, const AttrTypes* &... outPtrs) noexcept {
    if (!attrs) return false;
    size_t attrIdx = 0;
    return (((outPtrs = attrs->GetAttrPointer<AttrTypes>(attrIdx++)), outPtrs != nullptr) && ...);
}

template <size_t... Indices>
inline bool ValidateTensorInputs(const gert::TilingContext *ctx, std::index_sequence<Indices...>) noexcept {
    auto checkOne = [ctx](size_t idx) noexcept -> bool {
        const auto *shape = ctx->GetInputShape(idx);
        const auto *desc = ctx->GetInputDesc(idx);
        if (!shape || !desc) return false;
        const auto &storage = shape->GetStorageShape();
        if (storage.GetDimNum() != 4) return false;
        for (size_t d = 0; d < 4; ++d) {
            if (storage[d] <= 0 || storage[d] >= UINT32_MAX - 64) return false;
        }
        if (const auto *stride = ctx->GetInputStride(idx); stride != nullptr) {
            if (stride->GetDimNum() != 4 || (*stride)[3] != 1) return false;
        }
        return true;
    };
    return (checkOne(Indices) && ...);
}

template <size_t... Indices>
inline bool MatchInputDataTypes(const gert::TilingContext *ctx, ge::DataType expectedType, std::index_sequence<Indices...>) noexcept {
    return ((ctx->GetInputDesc(Indices)->GetDataType() == expectedType) && ...);
}

static inline void ComputeCanonicalStrides(uint64_t (&strideArr)[3], uint32_t headCount, uint32_t seqLen, uint32_t headDim) {
    strideArr[2] = uint64_t(headDim);
    strideArr[1] = uint64_t(seqLen) * strideArr[2];
    strideArr[0] = uint64_t(headCount) * strideArr[1];
}

inline constexpr bool IsWideStreamMode(uint32_t dp, uint32_t sq, uint32_t sk) noexcept {
    return (dp == 128U) && (sk >= 512U) && ((sk % 512U) == 0U) && ((sq % 64U) == 0U);
}

struct SageAttention2TilingPipeline {
    static bool Execute(gert::TilingContext *context) {
        if (!context || !context->GetAttrs()) {
            return false;
        }

        if (!ValidateTensorInputs(context, std::index_sequence<0, 1, 2>{})) {
            return false;
        }

        const auto inputDtype = context->GetInputDesc(0)->GetDataType();
        const bool validDtype = (inputDtype == ge::DT_FLOAT16 || inputDtype == ge::DT_BF16);
        if (!validDtype || !MatchInputDataTypes(context, inputDtype, std::index_sequence<1, 2>{})) {
            return false;
        }

        const auto &sQ = context->GetInputShape(0)->GetStorageShape();
        const auto &sK = context->GetInputShape(1)->GetStorageShape();
        const auto &sV = context->GetInputShape(2)->GetStorageShape();

        const bool shapesCompatible = (sK == sV) && (sQ[0] == sK[0]) && (sQ[3] == sK[3]) &&
                                      (sQ[3] <= sage::core::MAX_HEAD_DIM) && ((sQ[1] % sK[1]) == 0);
        if (!shapesCompatible) {
            return false;
        }

        const uint64_t batchHeadProduct = uint64_t(sK[0]) * uint64_t(sK[1]);
        if (batchHeadProduct >= UINT32_MAX || sQ[2] > UINT32_MAX - 127 || sK[2] > UINT32_MAX - 511) {
            return false;
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

        const float *attrSmScale = nullptr;
        const bool *attrSmoothK = nullptr;
        const bool *attrCausal = nullptr;
        const bool *attrReturnLse = nullptr;
        const int64_t *attrQuantGran = nullptr;
        const int64_t *attrPvAccum = nullptr;
        const bool *attrSmoothV = nullptr;

        if (!ExtractRuntimeAttributes(context->GetAttrs(), attrSmScale, attrSmoothK, attrCausal,
                                      attrReturnLse, attrQuantGran, attrPvAccum, attrSmoothV)) {
            return false;
        }
        if (!std::isfinite(*attrSmScale) || (*attrCausal && params.sq != params.sk)) {
            return false;
        }
        if (!sage::core::IsValidQuantGran(*attrQuantGran) || !sage::core::IsValidPvAccum(*attrPvAccum)) {
            return false;
        }

        const float scaleMagnitude = std::abs(*attrSmScale);
        const float sqrtBalanced = std::sqrt(scaleMagnitude * 2048.0f);
        if (!std::isfinite(sqrtBalanced) || sqrtBalanced > sage::core::SAGE_FP16_MAX) {
            return false;
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
            return false;
        }
        context->SetBlockDim(params.cores);

        params.qp = sage::core::RoundUpTo(params.sq, 128U);
        params.kp = sage::core::RoundUpTo(params.sk, 512U);

        const bool isWideStreamMode = IsWideStreamMode(params.dp, params.sq, params.sk);
        const bool isDp128 = (params.dp == 128U);
        const bool isCausal = (params.causal != 0U);
        const uint64_t operatorTilingKey = GET_TPL_TILING_KEY(uint32_t(inputDtype), isCausal, isDp128, isWideStreamMode);
        context->SetTilingKey(operatorTilingKey);
        context->SetScheduleMode(1U);

        BufferAllocationManager<512ULL> memPlanner{};
        const uint64_t queryStagingBatches = isWideStreamMode ? 0ULL : uint64_t(params.b);

        const bool planValid =
            memPlanner.AllocateAligned(tilingPayload.mean, params.b, params.hk, params.dp, sizeof(float)) &&
            memPlanner.AllocateAligned(tilingPayload.vMean, params.b, params.hk, params.dp, sizeof(float)) &&
            memPlanner.AllocateAligned(tilingPayload.vMxScale, params.b, params.hk, params.kp / 32U, params.dp, 1ULL) &&
            memPlanner.AllocateAligned(tilingPayload.q8, queryStagingBatches, params.hq, params.qp, params.dp, 1ULL) &&
            memPlanner.AllocateAligned(tilingPayload.k8, params.b, params.hk, params.kp, params.dp, 1ULL) &&
            memPlanner.AllocateAligned(tilingPayload.v8, params.b, params.hk, params.kp, params.dp, 1ULL) &&
            memPlanner.AllocateAligned(tilingPayload.qScale, queryStagingBatches, params.hq, params.qp, sizeof(uint16_t)) &&
            memPlanner.AllocateAligned(tilingPayload.kScale, params.b, params.hk, params.kp, sizeof(uint16_t));

        if (!planValid) {
            return false;
        }

        uint64_t systemReservedWorkspace = (16ULL << 20);
        if (context->GetPlatformInfo()) {
            platform_ascendc::PlatformAscendC platformInspect(context->GetPlatformInfo());
            systemReservedWorkspace = platformInspect.GetLibApiWorkSpaceSize();
        }

        const uint64_t totalWorkspaceRequired = memPlanner.currentOffset;
        if (totalWorkspaceRequired > UINT64_MAX - systemReservedWorkspace) {
            return false;
        }

        auto workspaceSizes = context->GetWorkspaceSizes(1U);
        auto rawData = context->GetRawTilingData();
        const uint64_t payloadByteSize = sizeof(SageAttention2TilingData);
        if (!workspaceSizes || !rawData || rawData->GetCapacity() < payloadByteSize) {
            return false;
        }

        *reinterpret_cast<SageAttention2TilingData *>(rawData->GetData()) = tilingPayload;
        rawData->SetDataSize(payloadByteSize);
        workspaceSizes[0] = static_cast<size_t>(totalWorkspaceRequired + systemReservedWorkspace);
        return true;
    }
};

inline ge::graphStatus ComputeSageAttention2Tiling(gert::TilingContext *context) {
    return SageAttention2TilingPipeline::Execute(context) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
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
