/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dispatch_ffn_combine_tiling.cpp
 * \brief Unified tiling for DispatchFFNCombine operator.
 *        Supports fp16/bf16, int8, int4/int32 weight types on Ascend910B/910_93 platforms.
 */


#include <vector>
#include <map>
#include <algorithm>
#include <string>
#include <limits>
#include <cstdlib>

#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "mc2_log.h"
#include "mc2_hcom_topo_info.h"
#include "ops_legacy/op_tiling/hcom_topo_info.h"
#include "platform/platform_infos_def.h"
#include "../../op_kernel/dispatch_ffn_combine_tiling.h"
#include "../../op_kernel/dispatch_ffn_combine_tiling_key.h"
#include "../../op_kernel/moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"
#include "dispatch_ffn_combine.h"  // local host config (DispatchFFNCombineConfig)

#include <cstring>

using namespace AscendC;
using namespace ge;

#define HCCL_BUFFSIZE "HCCL_BUFFSIZE"

namespace {

const char *K_INNER_DEBUG = "DispatchFFNCombine Tiling Debug";

// Symbolic name for input/attr indices follows `DispatchFFNCombineConfig`
// declared in `dispatch_ffn_combine.h` (host side). The actual numbers must
// match `dispatch_ffn_combine_def.cpp`.

// ---- Tiling key constants ----
constexpr uint32_t BLOCK_NUM = 20;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint64_t MB_SIZE = 1024 * 1024UL;
constexpr uint32_t RANK_NUM_PER_NODE_A2 = 8;

// String-to-code mapping for the `comm_alg` attribute. Kernel-side numeric
// codes are defined as `DispatchFFNCombineCommAlgCode` in the tiling header.
static uint32_t ParseCommAlgCode(const char *commAlg)
{
    if (commAlg == nullptr || commAlg[0] == '\0') {
        return DISPATCH_FFN_COMBINE_COMM_DEFAULT;
    }
    if (std::strstr(commAlg, "AlltoAll") != nullptr || std::strstr(commAlg, "alltoall") != nullptr) {
        return DISPATCH_FFN_COMBINE_COMM_ALLTOALL;
    }
    if (std::strstr(commAlg, "BatchWrite") != nullptr || std::strstr(commAlg, "batchwrite") != nullptr) {
        if (std::strstr(commAlg, "hierarchy") != nullptr) {
            return DISPATCH_FFN_COMBINE_COMM_HIERARCHY;
        }
        return DISPATCH_FFN_COMBINE_COMM_BATCHWRITE;
    }
    if (std::strstr(commAlg, "fullmesh") != nullptr) {
        return DISPATCH_FFN_COMBINE_COMM_FULLMESH;
    }
    if (std::strstr(commAlg, "pairwise") != nullptr) {
        return DISPATCH_FFN_COMBINE_COMM_PAIRWISE;
    }
    return DISPATCH_FFN_COMBINE_COMM_DEFAULT;
}

// String-to-code mapping for the `activation` attribute. Mirrors the codes
// declared as `DispatchFFNCombineActivationCode` in the kernel tiling header.
static uint32_t ParseActivationCode(const char *activation)
{
    if (activation == nullptr || activation[0] == '\0') {
        return DISPATCH_FFN_COMBINE_ACT_NONE;
    }
    if (std::strstr(activation, "swiglu") != nullptr || std::strstr(activation, "SwiGLU") != nullptr ||
        std::strstr(activation, "Swiglu") != nullptr) {
        return DISPATCH_FFN_COMBINE_ACT_SWIGLU;
    }
    if (std::strstr(activation, "gelu") != nullptr || std::strstr(activation, "GeLU") != nullptr ||
        std::strstr(activation, "Gelu") != nullptr) {
        return DISPATCH_FFN_COMBINE_ACT_GELU;
    }
    if (std::strstr(activation, "relu") != nullptr || std::strstr(activation, "ReLU") != nullptr ||
        std::strstr(activation, "Relu") != nullptr) {
        return DISPATCH_FFN_COMBINE_ACT_RELU;
    }
    if (std::strstr(activation, "silu") != nullptr || std::strstr(activation, "SiLU") != nullptr ||
        std::strstr(activation, "Silu") != nullptr) {
        return DISPATCH_FFN_COMBINE_ACT_SILU;
    }
    if (std::strstr(activation, "tanh") != nullptr || std::strstr(activation, "Tanh") != nullptr) {
        return DISPATCH_FFN_COMBINE_ACT_TANH;
    }
    return DISPATCH_FFN_COMBINE_ACT_NONE;
}

}  // namespace

namespace optiling {

static int32_t CeilDev(int32_t num, int32_t div)
{
    if (div == 0) {
        return 0;
    }
    return (num + div - 1) / div;
}

static uint64_t GetMaxWindowSize()
{
    uint16_t defaultWindowSize = 200;
    const char *hcclBuffsizeEnv = getenv(HCCL_BUFFSIZE);
    if (hcclBuffsizeEnv != nullptr) {
        try {
            std::string envStr(hcclBuffsizeEnv);
            unsigned long val = std::stoul(envStr);
            if (val <= std::numeric_limits<uint16_t>::max()) {
                defaultWindowSize = static_cast<uint16_t>(val);
            } else {
                OP_LOGW(K_INNER_DEBUG, "HCCL_BUFFSIZE value %lu is out of range, using default.", val);
            }
        } catch (const std::exception &e) {
            OP_LOGE(K_INNER_DEBUG, "Exception encountered when parsing env HCCL_BUFFSIZE: %s", e.what());
        }
    } else {
        OP_LOGD(K_INNER_DEBUG, "Env HCCL_BUFFSIZE not set");
    }
    const uint64_t maxWindowSize = static_cast<uint64_t>(defaultWindowSize) * MB_SIZE;
    OP_LOGD(K_INNER_DEBUG, "Get maxWindowSize is %lu", maxWindowSize);
    return maxWindowSize;
}

static std::string DispatchFFNCombineA2GetAlgConfig(int32_t epWorldSize, bool isLayered)
{
    if (epWorldSize <= RANK_NUM_PER_NODE_A2) {
        return "BatchWrite=level0:fullmesh";
    }
    return isLayered ? "BatchWrite=level1:hierarchy" : "BatchWrite=level1:fullmesh";
}

static ge::graphStatus DispatchFFNCombineCheckAttrAndSetTiling(gert::TilingContext *context,
                                                                DispatchFFNCombineInfo &info,
                                                                const DispatchFFNCombineConfig &config)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(K_INNER_DEBUG, "attrs is null."), return ge::GRAPH_FAILED);

    // -----------------------------------------------------------------
    // Read ALL 13 attributes declared in the unified op def. Required ones
    // produce a hard failure on missing pointers; OPTIONAL ones fall back to
    // the def-declared default value (0 / DT_UNDEFINED / "" / INFINITY / false).
    // -----------------------------------------------------------------
    auto moeExpertNumPtr        = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrMoeExpertNumIndex));
    auto epWorldSizePtr         = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrEpWorldSizeIndex));
    auto cclBufferSizePtr       = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCclBufferSizeIndex));
    auto maxRecvTokenNumPtr     = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrMaxRecvTokenNumIndex));
    auto dispatchQuantModePtr   = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrDispatchQuantModeIndex));
    auto dispatchQuantOutTypePtr= attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrDispatchQuantOutTypeIndex));
    auto combineQuantModePtr    = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCombineQuantModeIndex));
    auto commAlgPtr             = attrs->GetAttrPointer<char>   (static_cast<int>(config.attrCommAlgIndex));
    auto numMaxTokenPerRankPtr  = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrNumMaxTokenPerRankIndex));
    auto activationPtr          = attrs->GetAttrPointer<char>   (static_cast<int>(config.attrActivationIndex));
    auto activationClampPtr     = attrs->GetAttrPointer<float>  (static_cast<int>(config.attrActivationClampIndex));
    auto transposeW1Ptr         = attrs->GetAttrPointer<bool>   (static_cast<int>(config.attrTransposeWeight1Index));
    auto transposeW2Ptr         = attrs->GetAttrPointer<bool>   (static_cast<int>(config.attrTransposeWeight2Index));

    OP_TILING_CHECK(moeExpertNumPtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "moe_expert_num is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "ep_world_size is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(cclBufferSizePtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "ccl_buffer_size is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(maxRecvTokenNumPtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "max_recv_token_num is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(transposeW1Ptr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "transpose_weight1 is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(transposeW2Ptr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "transpose_weight2 is invalid."), return GRAPH_FAILED);

    info.moeExpertNum          = static_cast<uint32_t>(*moeExpertNumPtr);
    info.epWorldSize           = static_cast<uint32_t>(*epWorldSizePtr);
    info.worldSize             = info.epWorldSize;
    info.cclBufferSize         = static_cast<uint32_t>(*cclBufferSizePtr);
    info.maxRecvTokenNum       = static_cast<uint32_t>(*maxRecvTokenNumPtr);
    info.dispatchQuantMode     = (dispatchQuantModePtr   != nullptr) ? static_cast<uint32_t>(*dispatchQuantModePtr)   : 0U;
    info.dispatchQuantOutType  = (dispatchQuantOutTypePtr!= nullptr) ? static_cast<int32_t>(*dispatchQuantOutTypePtr) :
                                                                       static_cast<int32_t>(ge::DT_UNDEFINED);
    info.combineQuantMode      = (combineQuantModePtr    != nullptr) ? static_cast<uint32_t>(*combineQuantModePtr)    : 0U;
    info.commAlgCode           = ParseCommAlgCode(commAlgPtr);
    info.numMaxTokenPerRank    = (numMaxTokenPerRankPtr  != nullptr) ? static_cast<uint32_t>(*numMaxTokenPerRankPtr)  : 0U;
    info.activationCode        = ParseActivationCode(activationPtr);
    info.activationClamp       = (activationClampPtr     != nullptr) ? *activationClampPtr : std::numeric_limits<float>::infinity();
    info.isTransposeW1         = (*transposeW1Ptr) ? 1U : 0U;
    info.isTransposeW2         = (*transposeW2Ptr) ? 1U : 0U;

    OP_LOGD(K_INNER_DEBUG, "moeExpertNum=%u, epWorldSize=%u, cclBufferSize=%u, maxRecvTokenNum=%u",
            info.moeExpertNum, info.epWorldSize, info.cclBufferSize, info.maxRecvTokenNum);
    OP_LOGD(K_INNER_DEBUG, "dispatchQuantMode=%u, dispatchQuantOutType=%d, combineQuantMode=%u",
            info.dispatchQuantMode, info.dispatchQuantOutType, info.combineQuantMode);
    OP_LOGD(K_INNER_DEBUG, "commAlgCode=%u, numMaxTokenPerRank=%u, activationCode=%u, activationClamp=%f",
            info.commAlgCode, info.numMaxTokenPerRank, info.activationCode, info.activationClamp);
    OP_LOGD(K_INNER_DEBUG, "isTransposeW1=%u, isTransposeW2=%u", info.isTransposeW1, info.isTransposeW2);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchFFNCombineCheckShapeAndSetTiling(gert::TilingContext *context,
                                                                DispatchFFNCombineInfo &info,
                                                                const DispatchFFNCombineConfig &config)
{
    const char *nodeName = context->GetNodeName();

    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    auto topkIdsTensor = context->GetDynamicInputTensor(config.topkIdsIndex, 0);
    uint32_t M = xStorageShape->GetStorageShape().GetDim(0);
    uint32_t K = xStorageShape->GetStorageShape().GetDim(1);

    auto w1Tensor = context->GetDynamicInputTensor(config.weightOneIndex, 0);
    auto wDataType = w1Tensor->GetDataType();
    uint32_t wTensorDims = w1Tensor->GetOriginShape().GetDimNum();
    uint32_t N = w1Tensor->GetStorageShape().GetDim(wTensorDims - 1);

    if (ge::DT_INT32 == wDataType) {
        N = N * 8;
    }

    // Routing-quant decision: the BF16/FP16 weight path forwards X without any
    // quantization, so MoeInitRoutingV2 runs in its non-quant variant; every
    // other weight dtype (INT4, INT8, INT32-packed INT4) requires the input X
    // to be quantized to INT8 during routing.
    info.isQuantRouting = (wDataType == ge::DT_FLOAT16 || wDataType == ge::DT_BF16) ? 0U : 1U;

    uint32_t topK = topkIdsTensor->GetStorageShape().GetDim(1);
    uint32_t listLen = 0;
    while (true) {
        auto wTensorT = context->GetDynamicInputTensor(config.weightOneIndex, ++listLen);
        if (wTensorT == nullptr) {
            break;
        }
    }

    uint32_t expertPerRank;
    if (listLen == 1) {
        expertPerRank = w1Tensor->GetStorageShape().GetDim(0);
    } else {
        expertPerRank = listLen;
    }

    info.M = M;
    info.N = N;
    info.K = K;
    info.expertPerRank = expertPerRank;
    info.topK = topK;
    info.listLen = listLen;

    // -----------------------------------------------------------------
    // Detect presence of OPTIONAL inputs `x_active_mask` (index 8) and
    // `scales` (index 9). Kernel-side state mirrors what `mega_moe` does:
    // a nullptr GM_ADDR is forwarded when the input is absent, and the
    // boolean presence flags are passed in via tilingData so the kernel
    // can do compile-time pruning if needed.
    // -----------------------------------------------------------------
    auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
    auto scalesDesc      = context->GetOptionalInputDesc(config.scalesIndex);
    info.hasXActiveMask = (xActiveMaskDesc != nullptr) ? 1U : 0U;
    info.hasScales      = (scalesDesc      != nullptr) ? 1U : 0U;

    OP_LOGD(K_INNER_DEBUG, "M=%u ", info.M);
    OP_LOGD(K_INNER_DEBUG, "K=%u ", info.K);
    OP_LOGD(K_INNER_DEBUG, "N=%u ", info.N);
    OP_LOGD(K_INNER_DEBUG, "expertPerRank=%u ", info.expertPerRank);
    OP_LOGD(K_INNER_DEBUG, "topK=%u ", info.topK);
    OP_LOGD(K_INNER_DEBUG, "listLen=%u ", info.listLen);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchFFNCombineGetPlatformInfoAndSetTiling(gert::TilingContext *context,
                                                                     DispatchFFNCombineInfo &info)
{
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    info.aivNum = aivNum;
    info.totalUbSize = ubSize;

    std::string socVersion;
    (void)platformInfo->GetPlatformResWithLock("version", "Short_SoC_version", socVersion);
    info.isA2 = (socVersion == "Ascend910B");

    OP_LOGD(K_INNER_DEBUG, "aivNum=%u", info.aivNum);
    OP_LOGD(K_INNER_DEBUG, "ubSize=%lu", info.totalUbSize);
    OP_LOGD(K_INNER_DEBUG, "isA2=%d", info.isA2);

    return ge::GRAPH_SUCCESS;
}

void SetTilingData(CoCTiling &cocTilingData, DispatchFFNCombineInfo &info)
{
    cocTilingData.m0 = 128;
    cocTilingData.k0 = 256;
    cocTilingData.n0 = 256;
    cocTilingData.swizzleDirect = 1;
    cocTilingData.swizzleOffset = 7;
    cocTilingData.ubMoveNum = 16 * 1024;
    cocTilingData.pValue = 1;
    cocTilingData.commNpuSplit = info.worldSize;
    cocTilingData.commDataSplit = 1;
    cocTilingData.lenPerLoop = cocTilingData.m0 * cocTilingData.n0 / 2;
}

static ge::graphStatus DispatchFFNCombineTilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter DispatchFFNCombine tiling func.");

    // Local config carries all per-op input/attr indices, mirroring
    // `MegaMoeConfig`. The default values match the unified def order.
    DispatchFFNCombineConfig config;

    DispatchFFNCombineTilingData *tilingData = context->GetTilingData<DispatchFFNCombineTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."),
                    return ge::GRAPH_FAILED);
    OP_LOGI(nodeName, "DispatchFFNCombine get tilingData.");
    DispatchFFNCombineInfo &info = tilingData->dispatchFFNCombineInfo;
    OP_LOGI(nodeName, "DispatchFFNCombine get tilingData info.");

    OP_TILING_CHECK(DispatchFFNCombineCheckAttrAndSetTiling(context, info, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "DispatchFFNCombine CheckAttrAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(DispatchFFNCombineCheckShapeAndSetTiling(context, info, config) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "DispatchFFNCombine CheckShapeAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(DispatchFFNCombineGetPlatformInfoAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "DispatchFFNCombine GetPlatformInfoAndSetTiling Failed"),
        return ge::GRAPH_FAILED);

    SetTilingData(tilingData->cocTiling, info);

    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    uint64_t tilingKey = GET_TPL_TILING_KEY(
        static_cast<int64_t>(info.isTransposeW1),
        static_cast<int64_t>(info.isTransposeW2),
        static_cast<int64_t>(info.isA2));
    context->SetTilingKey(tilingKey);

    OP_LOGD(K_INNER_DEBUG, "tilingKey=%lu", tilingKey);

    // ------------------------------------------------------------------
    // MoeInitRoutingV2 tiling.
    // The output layout produced by MoeInitRoutingV2 differs between the
    // quant and non-quant variants (different gather/srcToDst sizes because
    // quant emits INT8 tokens + per-token scale while non-quant just emits
    // BF16/FP16 tokens). We dispatch on `info.isQuantRouting`, set
    // dispatch-time earlier from the `weight1` data type, and populate the
    // corresponding field on `cocTiling`.
    // We instantiate `MoeInitRoutingQuantV2TilingBase` in both cases (it is
    // the only concrete subclass of `InnerMoeInitRoutingV2TilingBase` we
    // currently ship), but pass `quantMode = 0` to make the routing kernel
    // emit the non-quant gather layout when needed.
    // ------------------------------------------------------------------
    optiling::MoeInitRoutingQuantV2TilingBase moeInitRoutingV2TilingBase;
    int64_t inuptXDtypeSize = sizeof(int16_t);
    int64_t scaleDim0 = 0;
    int64_t ubSize = 196352;
    int64_t expertCapacity = 0;
    int64_t expertNum = info.expertPerRank * info.worldSize;
    int64_t activeNum = info.M * info.topK;
    int64_t dropPadMode = 0;
    int64_t expertTokensCountOrCumsumFlag = 2;
    bool expertTokensBeforeCapacityFlag = false;
    int64_t quantMode = (info.isQuantRouting != 0U) ? 1 : 0;
    uint32_t aivNumInitRouting = 2 * BLOCK_NUM;
    moeInitRoutingV2TilingBase.DoTiling(info.M, info.K, info.topK, expertCapacity, expertNum, activeNum,
        dropPadMode, expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag,
        inuptXDtypeSize, quantMode, scaleDim0, aivNumInitRouting, ubSize);
    uint64_t initRoutingQuantTilingKey = moeInitRoutingV2TilingBase.tilingKey_;
    size_t initRoutingWorkspace = moeInitRoutingV2TilingBase.workspaceSize_;

    auto &quantTd = moeInitRoutingV2TilingBase.quantTilingData;
    if (info.isQuantRouting != 0U) {
        // Quant path: directly copy the full `MoeInitRoutingQuantV2TilingData`.
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData = quantTd;
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.vbsComputeParamsOp =
            quantTd.vbsComputeParamsOp;
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.vmsMiddleComputeParamsOp =
            quantTd.vmsMiddleComputeParamsOp;
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.sortOutComputeParamsOp =
            quantTd.sortOutComputeParamsOp;
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp =
            quantTd.srcToDstComputeParamsOp;
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp =
            quantTd.srcToDstCapacityComputeParamsOp;
        tilingData->cocTiling.moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp =
            quantTd.gatherOutComputeParamsOp;
    } else {
        // Non-quant path: lift only the InnerMoeInitRoutingV2TilingData subset
        // (the leading non-smoothType fields are layout-compatible).
        auto &nonQuantTd = tilingData->cocTiling.moeInitRoutingV2TilingData;
        nonQuantTd.coreNum = quantTd.coreNum;
        nonQuantTd.n = quantTd.n;
        nonQuantTd.cols = quantTd.cols;
        nonQuantTd.k = quantTd.k;
        nonQuantTd.expertCapacity = quantTd.expertCapacity;
        nonQuantTd.expertNum = quantTd.expertNum;
        nonQuantTd.dropPadMode = quantTd.dropPadMode;
        nonQuantTd.expertTokensCountOrCumsumFlag = quantTd.expertTokensCountOrCumsumFlag;
        nonQuantTd.expertTokensBeforeCapacityFlag = quantTd.expertTokensBeforeCapacityFlag;
        nonQuantTd.vbsComputeParamsOp = quantTd.vbsComputeParamsOp;
        nonQuantTd.vmsMiddleComputeParamsOp = quantTd.vmsMiddleComputeParamsOp;
        nonQuantTd.sortOutComputeParamsOp = quantTd.sortOutComputeParamsOp;
        nonQuantTd.srcToDstComputeParamsOp = quantTd.srcToDstComputeParamsOp;
        nonQuantTd.srcToDstCapacityComputeParamsOp = quantTd.srcToDstCapacityComputeParamsOp;
        nonQuantTd.gatherOutComputeParamsOp = quantTd.gatherOutComputeParamsOp;
    }
    tilingData->cocTiling.initRoutingQuantTilingKey = initRoutingQuantTilingKey;
    OP_LOGD(K_INNER_DEBUG, "initRoutingQuantTilingKey=%lu (quantMode=%ld, isQuantRouting=%u)",
            initRoutingQuantTilingKey, quantMode, info.isQuantRouting);

    uint64_t maxWindowSize = GetMaxWindowSize();
    uint64_t actualSize = static_cast<uint64_t>(info.M) * info.topK * info.K * sizeof(int8_t) * 3 + 10 * MB_SIZE;
    OP_TILING_CHECK((actualSize > maxWindowSize),
        OP_LOGE(nodeName, "HCCL_BUFFSIZE is too SMALL, m = %u, k = %u, topK = %u"
            " expected HCCL_BUFFSIZE is ((m * k * topK * sizeof(int8_t)) * 3 + 10MB)= %luMB, HCCL_BUFFSIZE=%luMB.",
            info.M, info.K, info.topK, (actualSize + MB_SIZE - 1) / MB_SIZE, maxWindowSize / MB_SIZE),
        return ge::GRAPH_FAILED);

    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."),
                    return ge::GRAPH_FAILED);

    uint32_t n2 = info.K;
    uint32_t k2 = info.N / 2;

    uint64_t cocWorkspace = (info.M + 256 - 1) / 256 * 256 * info.topK * sizeof(int32_t) +
                            info.worldSize * info.worldSize * info.expertPerRank * sizeof(int32_t) * 3 +
                            info.maxRecvTokenNum * sizeof(float) * 2 +
                            info.maxRecvTokenNum * info.N * sizeof(int16_t) * 2 +
                            info.maxRecvTokenNum * n2 * sizeof(int16_t) * 2 +
                            info.maxRecvTokenNum * info.K * sizeof(int8_t) +
                            info.maxRecvTokenNum * k2 * sizeof(int8_t) +
                            info.worldSize * sizeof(int32_t) * 16 +
                            (info.expertPerRank + info.worldSize) * sizeof(int32_t) * 16;

    workSpaces[0] = SYSTEM_NEED_WORKSPACE + std::max(cocWorkspace, initRoutingWorkspace);

    // Default `algConfig` is derived from platform (A2 vs A3). If the user
    // explicitly passes a non-empty `comm_alg` attribute, we honor it: the
    // raw attr string is forwarded to `Mc2CcTilingConfig` while the numeric
    // code is also stored in `info.commAlgCode` for the kernel.
    uint32_t opType = 0u;
    std::string algConfig;
    if (info.isA2) {
        opType = 18u;
        algConfig = DispatchFFNCombineA2GetAlgConfig(static_cast<int32_t>(info.worldSize), true);
    } else {
        opType = 8U;
        algConfig = "AlltoAll=level0:fullmesh;level1:pairwise";
    }
    auto attrsForAlg = context->GetAttrs();
    if (attrsForAlg != nullptr) {
        auto commAlgPtr = attrsForAlg->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
        if (commAlgPtr != nullptr && commAlgPtr[0] != '\0') {
            algConfig = std::string(commAlgPtr);
        }
    }
    auto contextTensor = context->GetInputTensor(static_cast<int>(config.contextIndex));
    auto group = (contextTensor != nullptr) ? static_cast<const char *>(contextTensor->GetData()) : nullptr;
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
    mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);

    OP_LOGI(nodeName, "Leave DispatchFFNCombine tiling func.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchFFNCombineTilingFunc(gert::TilingContext *context)
{
    return DispatchFFNCombineTilingFuncImpl(context);
}

struct DispatchFFNCombineCompileInfo {};
ge::graphStatus TilingParseForDispatchFFNCombine(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DispatchFFNCombine)
    .Tiling(DispatchFFNCombineTilingFunc)
    .TilingParse<DispatchFFNCombineCompileInfo>(TilingParseForDispatchFFNCombine);
}  // namespace optiling