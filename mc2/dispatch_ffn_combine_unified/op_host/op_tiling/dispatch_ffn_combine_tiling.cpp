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
using namespace Mc2Tiling;

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

static ge::graphStatus DispatchFFNCombineCheckAttrAndSetTiling(gert::TilingContext *context,
                                                                DispatchFFNCombineInfoCommon &info,
                                                                const DispatchFFNCombineConfig &config)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(K_INNER_DEBUG, "attrs is null."), return ge::GRAPH_FAILED);

    auto moeExpertNumPtr        = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrMoeExpertNumIndex));
    auto epWorldSizePtr         = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrEpWorldSizeIndex));
    auto cclBufferSizePtr       = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCclBufferSizeIndex));
    auto maxRecvTokenNumPtr     = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrMaxRecvTokenNumIndex));
    auto dispatchQuantModePtr   = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrDispatchQuantModeIndex));
    auto dispatchQuantOutTypePtr= attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrDispatchQuantOutTypeIndex));
    auto combineQuantModePtr    = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrCombineQuantModeIndex));
    auto commAlgPtr             = attrs->GetAttrPointer<char>  (static_cast<int>(config.attrCommAlgIndex));
    auto numMaxTokenPerRankPtr  = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrNumMaxTokenPerRankIndex));
    auto activationPtr          = attrs->GetAttrPointer<char>  (static_cast<int>(config.attrActivationIndex));
    auto activationClampPtr     = attrs->GetAttrPointer<float> (static_cast<int>(config.attrActivationClampIndex));
    auto transposeW1Ptr         = attrs->GetAttrPointer<bool>   (static_cast<int>(config.attrTransposeWeight1Index));
    auto transposeW2Ptr         = attrs->GetAttrPointer<bool>   (static_cast<int>(config.attrTransposeWeight2Index));
    auto activation_out_dtypePtr          = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrActivationOutDtypeIndex));
    auto weight1InterleavePtr   = attrs->GetAttrPointer<int64_t>(static_cast<int>(config.attrWeight1InterleaveIndex));

    OP_TILING_CHECK(moeExpertNumPtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "moe_expert_num is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "ep_world_size is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(cclBufferSizePtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "ccl_buffer_size is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(maxRecvTokenNumPtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "max_recv_token_num is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(std::strcmp(commAlgPtr, "") != 0,
        OP_LOGE(K_INNER_DEBUG, "Invalid commAlg(%s), expecting default value(\"\").", commAlgPtr),
        return ge::GRAPH_FAILED);
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
    // 需要将字符串转换
    uint32_t commAlgCode = 0U;
    if (commAlgPtr != nullptr) {
        if (strcmp(commAlgPtr, "fullmesh") == 0 || strcmp(commAlgPtr, "fullmesh_v1") == 0 ||
            strcmp(commAlgPtr, "fullmesh_v2") == 0) {
            commAlgCode = 1U;
        } else if (strcmp(commAlgPtr, "hierarchy") == 0 || strcmp(commAlgPtr, "doublering") == 0) {
            commAlgCode = 2U;
        } else if (strcmp(commAlgPtr, "switchwing") == 0) {
            commAlgCode = 3U;
        }
    }
    info.commAlgCode = commAlgCode;
    info.numMaxTokenPerRank    = (numMaxTokenPerRankPtr  != nullptr) ? static_cast<uint32_t>(*numMaxTokenPerRankPtr)  : 0U;
    uint32_t activationCode = 0U;
    // 需要将字符串转换
    if (activationPtr != nullptr) {
        if (strcmp(activationPtr, "swiglu") == 0) {
            activationCode = 1U;
        } else if (strcmp(activationPtr, "gelu") == 0) {
            activationCode = 2U;
        } else if (strcmp(activationPtr, "silu") == 0) {
            activationCode = 3U;
        } else if (strcmp(activationPtr, "relu") == 0) {
            activationCode = 4U;
        }
    }
    info.activationCode = activationCode;
    info.activationClamp       = (activationClampPtr     != nullptr) ? *activationClampPtr :
                                                                       std::numeric_limits<float>::infinity();
    info.isTransposeW1         = (*transposeW1Ptr) ? 1U : 0U;
    info.isTransposeW2         = (*transposeW2Ptr) ? 1U : 0U;
    info.activation_out_dtype            = (activation_out_dtypePtr        != nullptr) ? static_cast<int32_t>(*activation_out_dtypePtr) :
                                                                       static_cast<int32_t>(ge::DT_UNDEFINED);
    info.weight1Interleave     = (weight1InterleavePtr != nullptr) ? static_cast<uint32_t>(*weight1InterleavePtr) : 0U;

    OP_LOGD(K_INNER_DEBUG, "moeExpertNum=%u, epWorldSize=%u, cclBufferSize=%u, maxRecvTokenNum=%u",
            info.moeExpertNum, info.epWorldSize, info.cclBufferSize, info.maxRecvTokenNum);
    OP_LOGD(K_INNER_DEBUG, "dispatchQuantMode=%u, dispatchQuantOutType=%d, combineQuantMode=%u",
            info.dispatchQuantMode, info.dispatchQuantOutType, info.combineQuantMode);
    OP_LOGD(K_INNER_DEBUG, "commAlgCode=%u, numMaxTokenPerRank=%u, activationCode=%u, activationClamp=%f",
            info.commAlgCode, info.numMaxTokenPerRank, info.activationCode, info.activationClamp);
    OP_LOGD(K_INNER_DEBUG, "isTransposeW1=%u, isTransposeW2=%u",
            info.isTransposeW1, info.isTransposeW2);
    OP_LOGD(K_INNER_DEBUG, "activation_out_dtype=%d, weight1Interleave=%u",
            info.activation_out_dtype, info.weight1Interleave);
    OP_LOGD(K_INNER_DEBUG, "hasBias1=%u, hasBias2=%u",
            info.hasBias1, info.hasBias2);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchFFNCombineCheckShapeAndSetTiling(gert::TilingContext *context,
                                                                DispatchFFNCombineInfoCommon &info,
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
    info.dispatchQuantMode = info.isQuantRouting ? DISPATCH_FFN_COMBINE_QUANT_MODE_PER_TENSOR
                                                  : DISPATCH_FFN_COMBINE_QUANT_MODE_NO_QUANT;
    if (info.isQuantRouting) {
        info.dispatchQuantOutType = static_cast<int32_t>(ge::DT_INT8);
    }

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

    auto bias1Desc = context->GetOptionalInputDesc(config.biasOneIndex);
    auto bias2Desc = context->GetOptionalInputDesc(config.biasTwoIndex);
    info.hasBias1 = (bias1Desc != nullptr) ? 1U : 0U;
    info.hasBias2 = (bias2Desc != nullptr) ? 1U : 0U;

    OP_LOGD(K_INNER_DEBUG, "M=%u ", info.M);
    OP_LOGD(K_INNER_DEBUG, "K=%u ", info.K);
    OP_LOGD(K_INNER_DEBUG, "N=%u ", info.N);
    OP_LOGD(K_INNER_DEBUG, "expertPerRank=%u ", info.expertPerRank);
    OP_LOGD(K_INNER_DEBUG, "topK=%u ", info.topK);
    OP_LOGD(K_INNER_DEBUG, "listLen=%u ", info.listLen);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchFFNCombineGetPlatformInfoAndSetTiling(gert::TilingContext *context,
                                                                     DispatchFFNCombineInfoCommon &info)
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

static ge::graphStatus DispatchFFNCombineTilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter DispatchFFNCombine tiling func.");

    DispatchFFNCombineConfig config;

    // First pass: read attributes and shape to determine isQuantRouting,
    // then acquire the correct tiling data struct.
    DispatchFFNCombineInfoCommon infoCommon{};
    {
        DispatchFFNCombineTilingDataQuant *tilingDataQuant = context->GetTilingData<DispatchFFNCombineTilingDataQuant>();
        OP_TILING_CHECK(tilingDataQuant == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."),
                        return ge::GRAPH_FAILED);
        DispatchFFNCombineInfoCommon &info = tilingDataQuant->common;

        OP_TILING_CHECK(DispatchFFNCombineCheckAttrAndSetTiling(context, info, config) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "DispatchFFNCombine CheckAttrAndSetTiling Failed"),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(DispatchFFNCombineCheckShapeAndSetTiling(context, info, config) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "DispatchFFNCombine CheckShapeAndSetTiling Failed"),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(DispatchFFNCombineGetPlatformInfoAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "DispatchFFNCombine GetPlatformInfoAndSetTiling Failed"),
            return ge::GRAPH_FAILED);
        infoCommon = info;
    }

    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    uint32_t quantMode = infoCommon.isQuantRouting != 0U
        ? DISPATCH_FFN_COMBINE_QUANT_MODE_PER_TENSOR
        : DISPATCH_FFN_COMBINE_QUANT_MODE_NO_QUANT;
    uint32_t quantOutType = DISPATCH_FFN_COMBINE_QUANT_OUT_TYPE_UNDEFINED;
    if (infoCommon.dispatchQuantOutType == static_cast<int32_t>(ge::DT_INT8)) {
        quantOutType = DISPATCH_FFN_COMBINE_QUANT_OUT_TYPE_INT8;
    }
    if (quantMode == DISPATCH_FFN_COMBINE_QUANT_MODE_NO_QUANT) {
        OP_TILING_CHECK(quantOutType != DISPATCH_FFN_COMBINE_QUANT_OUT_TYPE_UNDEFINED,
            OP_LOGE(K_INNER_DEBUG, "dispatch_quant_out_type must be UNDEFINED in non-quant mode, got %u",
                    quantOutType), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(quantOutType != DISPATCH_FFN_COMBINE_QUANT_OUT_TYPE_INT8,
            OP_LOGE(K_INNER_DEBUG, "dispatch_quant_out_type must be INT8 in quant mode, got %u",
                    quantOutType), return ge::GRAPH_FAILED);
    }
    uint32_t archCode = infoCommon.isA2 ? SOC_ASCEND910B : SOC_ASCEND910_93;

    uint64_t tilingKey = GET_TPL_TILING_KEY(
        static_cast<int64_t>(infoCommon.isTransposeW1),
        static_cast<int64_t>(infoCommon.isTransposeW2),
        static_cast<int64_t>(quantMode),
        static_cast<int64_t>(quantOutType),
        static_cast<int64_t>(archCode));
    context->SetTilingKey(tilingKey);

    OP_LOGD(K_INNER_DEBUG, "tilingKey=%lu", tilingKey);

    int64_t inuptXDtypeSize = sizeof(int16_t);
    int64_t scaleDim0 = 0;
    int64_t ubSize = 196352;
    int64_t expertCapacity = 0;
    int64_t expertNum = infoCommon.expertPerRank * infoCommon.worldSize;
    int64_t activeNum = infoCommon.M * infoCommon.topK;
    int64_t dropPadMode = 0;
    int64_t expertTokensCountOrCumsumFlag = 2;
    bool expertTokensBeforeCapacityFlag = false;
    uint32_t aivNumInitRouting = 2 * BLOCK_NUM;

    uint64_t initRoutingQuantTilingKey = 0;
    size_t initRoutingWorkspace = 0;

    if (infoCommon.isQuantRouting != 0U) {
        DispatchFFNCombineTilingDataQuant *tilingData = context->GetTilingData<DispatchFFNCombineTilingDataQuant>();
        OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."),
                        return ge::GRAPH_FAILED);
        tilingData->common = infoCommon;

        optiling::MoeInitRoutingQuantV2TilingBase moeInitRoutingQuantV2TilingBase;
        int64_t quantModeRouting = 1;
        moeInitRoutingQuantV2TilingBase.DoTiling(infoCommon.M, infoCommon.K, infoCommon.topK, expertCapacity, expertNum, activeNum,
            dropPadMode, expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag,
            inuptXDtypeSize, quantModeRouting, scaleDim0, aivNumInitRouting, ubSize);
        initRoutingQuantTilingKey = moeInitRoutingQuantV2TilingBase.tilingKey_;
        initRoutingWorkspace = moeInitRoutingQuantV2TilingBase.workspaceSize_;
        tilingData->moeInitRoutingQuantV2TilingData = moeInitRoutingQuantV2TilingBase.quantTilingData;
        tilingData->common.initRoutingQuantTilingKey = initRoutingQuantTilingKey;
    } else {
        DispatchFFNCombineTilingDataNonQuant *tilingData = context->GetTilingData<DispatchFFNCombineTilingDataNonQuant>();
        OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."),
                        return ge::GRAPH_FAILED);
        tilingData->common = infoCommon;

        optiling::MoeInitRoutingV2TilingBase moeInitRoutingV2TilingBase;
        int64_t quantModeRouting = 0;
        moeInitRoutingV2TilingBase.DoTiling(infoCommon.M, infoCommon.K, infoCommon.topK, expertCapacity, expertNum, activeNum,
            dropPadMode, expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag,
            inuptXDtypeSize, quantModeRouting, scaleDim0, aivNumInitRouting, ubSize);
        initRoutingQuantTilingKey = moeInitRoutingV2TilingBase.tilingKey_;
        initRoutingWorkspace = moeInitRoutingV2TilingBase.workspaceSize_;
        tilingData->moeInitRoutingV2TilingData = moeInitRoutingV2TilingBase.moeInitRoutingTilingData;
        tilingData->common.initRoutingQuantTilingKey = initRoutingQuantTilingKey;
    }

    OP_LOGD(K_INNER_DEBUG, "initRoutingQuantTilingKey=%lu (isQuantRouting=%u)",
            initRoutingQuantTilingKey, infoCommon.isQuantRouting);

    uint64_t maxWindowSize = GetMaxWindowSize();
    uint64_t actualSize = static_cast<uint64_t>(infoCommon.M) * infoCommon.topK * infoCommon.K * sizeof(int8_t) * 3 + 10 * MB_SIZE;
    OP_TILING_CHECK((actualSize > maxWindowSize),
        OP_LOGE(nodeName, "HCCL_BUFFSIZE is too SMALL, m = %u, k = %u, topK = %u"
            " expected HCCL_BUFFSIZE is ((m * k * topK * sizeof(int8_t)) * 3 + 10MB)= %luMB, HCCL_BUFFSIZE=%luMB.",
            infoCommon.M, infoCommon.K, infoCommon.topK, (actualSize + MB_SIZE - 1) / MB_SIZE, maxWindowSize / MB_SIZE),
        return ge::GRAPH_FAILED);

    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."),
                    return ge::GRAPH_FAILED);

    uint32_t n2 = infoCommon.K;
    uint32_t k2 = infoCommon.N / 2;

    uint64_t cocWorkspace = (infoCommon.M + 256 - 1) / 256 * 256 * infoCommon.topK * sizeof(int32_t) +
                            infoCommon.worldSize * infoCommon.worldSize * infoCommon.expertPerRank * sizeof(int32_t) * 3 +
                            infoCommon.maxRecvTokenNum * sizeof(float) * 2 +
                            infoCommon.maxRecvTokenNum * infoCommon.N * sizeof(int16_t) * 2 +
                            infoCommon.maxRecvTokenNum * n2 * sizeof(int16_t) * 2 +
                            infoCommon.maxRecvTokenNum * infoCommon.K * sizeof(int8_t) +
                            infoCommon.maxRecvTokenNum * k2 * sizeof(int8_t) +
                            infoCommon.worldSize * sizeof(int32_t) * 16 +
                            (infoCommon.expertPerRank + infoCommon.worldSize) * sizeof(int32_t) * 16;

    workSpaces[0] = SYSTEM_NEED_WORKSPACE + std::max(cocWorkspace, initRoutingWorkspace);

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