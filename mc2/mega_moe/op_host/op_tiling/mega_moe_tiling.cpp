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
 * \file mega_moe_tiling.cpp
 * \brief
 */

#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "mc2_hcom_topo_info.h"
#include "mega_moe.h"
#include "../../op_kernel/mega_moe_tiling.h"
#include "../../op_kernel/mega_moe_tiling_key.h"
#include "../../op_kernel/mega_moe_workspace_info.h"

using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;

namespace optiling {
namespace {
    const static int64_t NUM_TWO = 2LL;
    const static int64_t NUM_FOUR = 4LL;
    const static int64_t UB_BLOCK_SIZE = 32LL;
    const static int64_t KV_FACTOR = 2LL;
    const static int64_t EXPERT_IDX_MAX = 10240LL;
    const static int64_t KV_MODE_EXPERT_IDX_MAX = EXPERT_IDX_MAX / KV_FACTOR;

    const static int64_t ROW_IDX_GATHER = 0LL;
    const static int64_t ROW_IDX_SCATTER = 1LL;
    const static int64_t EXPERT_TOKENS_TYPE_COUNT = 1LL;
    const static int64_t EXPERT_TOKENS_TYPE_KEY_VALUE = 2LL;
    const static int64_t DROP_PAD_MODE_DROPLESS = 0LL;

    const static int64_t FOUR_DIMS = 4LL;
    const static int64_t THREE_DIMS = 3LL;
    const static int64_t TWO_DIMS = 2LL;
    const static int64_t ONE_DIM = 1LL;
    const static int64_t MIN_BS = 1LL;
    const static int64_t MAX_BS = 512LL;
    const static int64_t MIN_EXPERT_PER_RANK = 1LL;
    const static int64_t MAX_EXPERT_PER_RANK = 16LL;
    const static int64_t H_BASE = 1024LL;
    const static int64_t HIDDEN_DIM_BASE = 1024LL;
    const static int64_t MIN_EP_WORLD_SIZE = 2LL;
    const static int64_t MAX_EP_WORLD_SIZE = 768LL;
    const static int64_t MAX_MOE_EXPERT_NUM = 1024LL;
    const static int64_t INPUT_WEIGHT_SCALES_CEIL_ALIGN = 64LL;
    const static int64_t RESERVED_WORKSPACE_SIZE = 1024 * 1024 * 50LL;
}

void PrintMegaMoeTilingData(const MegaMoeTilingData* tilingData, const char *nodeName)
{
    OP_TILING_CHECK(tilingData == nullptr,
        OP_LOGE(nodeName, "tilingData is nullptr."), return);
    OP_LOGD(nodeName, "========== MegaMoeTilingData ==========");
    OP_LOGD(nodeName, "BS is %u", tilingData->bs);
    OP_LOGD(nodeName, "H is %u", tilingData->h);
    OP_LOGD(nodeName, "hiddenDim is %u", tilingData->hiddenDim);

    OP_LOGD(nodeName, "topK is %u", tilingData->topK);
    OP_LOGD(nodeName, "expertPerRank is %u", tilingData->expertPerRank);
    OP_LOGD(nodeName, "groupListType is %u", tilingData->groupListType);

    OP_LOGD(nodeName, "epWorldSize is %u", tilingData->epWorldSize);
    OP_LOGD(nodeName, "maxOutputSize is %u", tilingData->maxOutputSize);

    OP_LOGD(nodeName, "transX is %s", (tilingData->transX ? "True" : "False"));
    OP_LOGD(nodeName, "transW is %s", (tilingData->transW ? "True" : "False"));
    OP_LOGD(nodeName, "transW2 is %s", (tilingData->transW2 ? "True" : "False"));
}

void printWorkspaceInfo(const struct WorkspaceInfo *info, const char *nodeName)
{
    OP_LOGD(nodeName, "dispatchRevDataPtr:         %ld\n", info->dispatchRevDataPtr);
    OP_LOGD(nodeName, "dispatchRevScalePtr:        %ld\n", info->dispatchRevScalePtr);
    OP_LOGD(nodeName, "swigluQuantDataPtr:         %ld\n", info->swigluQuantDataPtr);
    OP_LOGD(nodeName, "swigluQuantScalePtr:        %ld\n", info->swigluQuantScalePtr);
    OP_LOGD(nodeName, "expertRevTokenNumsPtr:      %ld\n", info->expertRevTokenNumsPtr);
    OP_LOGD(nodeName, "tripleInfoPtr:              %ld\n", info->tripleInfoPtr);
    OP_LOGD(nodeName, "flagSwiGluToGmm2Ptr:        %ld\n", info->flagSwiGluToGmm2Ptr);
    OP_LOGD(nodeName, "flagDispatchToGmm1Ptr:      %ld\n", info->flagDispatchToGmm1Ptr);
}

void printPeermemInfo(const MegaMoeTilingData* tilingData, const char *nodeName)
{
    OP_LOGD(nodeName, "========== PeermemInfo ==========");
    int64_t rankSyncInWorldSize = PEERMEM_DATA_OFFSET;
    OP_LOGD(nodeName, "rankSyncInWorldSize: {%ld}\n", rankSyncInWorldSize);
    int64_t sendTotalNum = static_cast<int64_t>(tilingData->bs) * tilingData->topK;
    int64_t compareCount = Ops::CeilAlign(sendTotalNum * (int64_t)sizeof(int32_t), (int64_t)ALIGN_256) /
        (int64_t)sizeof(int32_t);
    int64_t maskAlignSize = Ops::CeilAlign(compareCount / 8, (int64_t)ALIGN_32);
    int64_t maskSlotSize = maskAlignSize + (int64_t)ALIGN_32;  // mask + 32B count
    int64_t maskRecvSize = Ops::CeilAlign(
        (int64_t)tilingData->expertPerRank * tilingData->epWorldSize * maskSlotSize, (int64_t)ALIGN_512);
    OP_LOGD(nodeName, "maskRecvSize: {%ld}\n", maskRecvSize);
    uint32_t mxScaleNum = Ops::CeilDiv(tilingData->h, static_cast<uint32_t>(ALIGN_32));
    uint32_t dataBytes = Ops::CeilAlign(tilingData->h, static_cast<uint32_t>(ALIGN_256)) * sizeof(int8_t);
    uint32_t scaleBytes = mxScaleNum * sizeof(int8_t);
    uint32_t tokenBytes = Ops::CeilAlign(dataBytes + scaleBytes, static_cast<uint32_t>(ALIGN_32));
    int64_t quantTokenScaleSize = Ops::CeilAlign((int64_t)(tilingData->bs * tokenBytes * sizeof(int8_t)), (int64_t)ALIGN_512);
    OP_LOGD(nodeName, "quantTokenScaleSize: {%ld}\n", quantTokenScaleSize);
    int64_t combineSendSize = sendTotalNum * tilingData->h * sizeof(bfloat16_t);
    OP_LOGD(nodeName, "combineSendSize: {%ld}\n", combineSendSize);
    OP_LOGD(nodeName, "total PeermemInfo Size: {%ld}\n", rankSyncInWorldSize + maskRecvSize + quantTokenScaleSize + combineSendSize);
}

static ge::DataType GetDataTypeByOpQuantMode(const int64_t opQuantMode)
{
    // unsupport UNQUANT, STATIC, DYNAMIC currently
    switch (opQuantMode) {
        case DISPATCH_QUANT_OUT_TYPE_E5M2: return ge::DT_FLOAT8_E5M2;
        case DISPATCH_QUANT_OUT_TYPE_E4M3FN: return ge::DT_FLOAT8_E4M3FN;
        case DISPATCH_QUANT_OUT_TYPE_E2M1: return ge::DT_FLOAT4_E2M1;
        default: return ge::DT_UNDEFINED;
    }
    return ge::DT_UNDEFINED;
}

static int64_t GetOpQuantModeByAttrDispatchOutType(const gert::TilingContext *context, MegaMoeConfig &config)
{
    auto attrs = context->GetAttrs();
    auto dispatchQuantOutTypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutTypeIndex));
    int64_t dispatchQuantOutType = static_cast<int64_t>(*dispatchQuantOutTypePtr);

    int64_t opQuantMode;
    if (dispatchQuantOutType == static_cast<int64_t>(ge::DT_FLOAT8_E5M2)) {
        opQuantMode = DISPATCH_QUANT_OUT_TYPE_E5M2;
    } else if (dispatchQuantOutType == static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN)) {
        opQuantMode = DISPATCH_QUANT_OUT_TYPE_E4M3FN;
    } else {
        opQuantMode = DISPATCH_QUANT_OUT_TYPE_E2M1;
    }

    return opQuantMode;
}

static uint64_t CalTilingKey(const gert::TilingContext *context, MegaMoeConfig &config,
    MegaMoeTilingData *tilingData, const char *nodeName)
{
    auto attrs = context->GetAttrs();

    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    int64_t opQuantMode = GetOpQuantModeByAttrDispatchOutType(context, config);

    return GET_TPL_TILING_KEY(static_cast<int64_t>(*dispatchQuantModePtr), opQuantMode);
}

static ge::graphStatus CheckAttrPtrNullptr(const gert::TilingContext *context,
    MegaMoeConfig &config, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is null."), return ge::GRAPH_FAILED);

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>((config.attrCclBufferSizeIndex));
    auto maxRecvTokenNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex));
    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    auto dispatchQuantOutTypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutTypeIndex));
    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCombineQuantModeIndex));
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    auto globalBs = attrs->GetAttrPointer<int64_t>((config.attrGlobalBsIndex));

    OP_TILING_CHECK(moeExpertNumPtr == nullptr,
        OP_LOGE(nodeName, "moeExpertNumPtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr,
        OP_LOGE(nodeName, "epWorldSizePtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(cclBufferSizePtr == nullptr || *cclBufferSizePtr < 0,
        OP_LOGE(nodeName, "cclBufferSizePtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(maxRecvTokenNumPtr == nullptr,
        OP_LOGE(nodeName, "maxRecvTokenNumPtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(dispatchQuantModePtr == nullptr,
        OP_LOGE(nodeName, "dispatchQuantModePtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(dispatchQuantOutTypePtr == nullptr,
        OP_LOGE(nodeName, "dispatchQuantOutTypePtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(combineQuantModePtr == nullptr,
        OP_LOGE(nodeName, "combineQuantModePtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commAlgPtr == nullptr,
        OP_LOGE(nodeName, "commAlgPtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(globalBs == nullptr,
        OP_LOGE(nodeName, "globalBs is nullptr."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrParams(const gert::TilingContext *context, MegaMoeConfig &config, const char *nodeName)
{
    auto attrs = context->GetAttrs();

    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    auto weightOneStorageShape = context->GetDynamicInputShape(config.weightOneIndex, 0);
    auto yDesc = context->GetOutputDesc(config.yIndex);
    
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdsStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);

    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);
    int64_t topK = topkIdsStorageShape->GetStorageShape().GetDim(1);
    int64_t expertPerRank = weightOneStorageShape->GetStorageShape().GetDim(0);
    int64_t n = weightOneStorageShape->GetStorageShape().GetDim(1);
    
    ge::DataType yDtype = yDesc->GetDataType();
    int64_t yDtypeSize = ge::GetSizeByDataType(yDtype);

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    int64_t epWorldSize = static_cast<int64_t>(*epWorldSizePtr);
    OP_TILING_CHECK(epWorldSize < MIN_EP_WORLD_SIZE || epWorldSize > MAX_EP_WORLD_SIZE,
        OP_LOGE(nodeName,
        "epWorldSize should in [%ld, %ld], but now epWorldSize is %ld.",
        MIN_EP_WORLD_SIZE, MAX_EP_WORLD_SIZE, epWorldSize),
        return ge::GRAPH_FAILED);

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    int64_t moeExpertNum = static_cast<int64_t>(*moeExpertNumPtr);
    OP_TILING_CHECK((moeExpertNum < epWorldSize || moeExpertNum > MAX_MOE_EXPERT_NUM) || (moeExpertNum % epWorldSize),
        OP_LOGE(nodeName,
            "moeExpertNum(%ld) should in [%ld, %ld] and mod(moeExpertNum, epWorldSize(%ld)) should be zero.",
            moeExpertNum, epWorldSize, MAX_MOE_EXPERT_NUM, epWorldSize),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(moeExpertNum != expertPerRank * epWorldSize,
        OP_LOGE(nodeName,
            "moeExpertNum(%ld) should be equal to expertPerRank(%ld) * epWorldSize(%ld) = %ld.",
            moeExpertNum, expertPerRank, epWorldSize, expertPerRank * epWorldSize),
        return ge::GRAPH_FAILED);

    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>((config.attrCclBufferSizeIndex));
    int64_t cclBufferSize = static_cast<int64_t>(*cclBufferSizePtr);
    // leastCclBufferSize = PEERMEM_DATA_OFFSET + tokenPerExpert + quantTokenScale + combineOut
    int64_t leastCclBufferSize = PEERMEM_DATA_OFFSET +
        (epWorldSize * ops::CeilAlign(epWorldSize * expertPerRank, ALIGN_128) * sizeof(int32_t)) + // tokenPerExpert
        (ops::CeilAlign(bs * topK * (h + h / MXFP_SCALE_GROUP_NUM), ALIGN_512) * sizeof(int8_t)) + // quantTokenScale
        (ops::CeilAlign(bs * h * topK * yDtypeSize, ALIGN_512)); // combineOut
    OP_TILING_CHECK(cclBufferSize < leastCclBufferSize,
        OP_LOGE(nodeName, "cclBufferSize(%ld) should equal or larger than leastCclBufferSize(%ld).",
            cclBufferSize, leastCclBufferSize),
        return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "cclBufferSize is %ld, leastCclBufferSize is %ld", cclBufferSize, leastCclBufferSize);

    auto maxRecvTokenNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex));
    int64_t maxRecvTokenNum = static_cast<int64_t>(*maxRecvTokenNumPtr);
    OP_TILING_CHECK(maxRecvTokenNum < 0 || maxRecvTokenNum > bs * epWorldSize * std::min(topK, expertPerRank),
        OP_LOGE(nodeName,
            "maxRecvTokenNum(%ld) should in [0, %ld], right bound is bs * epWorldSize * min(topK, expertPerRank).",
            maxRecvTokenNum, bs * epWorldSize * std::min(topK, expertPerRank)),
        return ge::GRAPH_FAILED);

    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    int64_t dispatchQuantMode = static_cast<int64_t>(*dispatchQuantModePtr);
    OP_TILING_CHECK(dispatchQuantMode != DISPATCH_QUANT_MODE_MXFP,
        OP_LOGE(nodeName,
            "Invalid dispatchQuantMode, only support mxfp(%ld), but now is %ld.",
            DISPATCH_QUANT_MODE_MXFP, dispatchQuantMode),
        return ge::GRAPH_FAILED);

    auto dispatchQuantOutTypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutTypeIndex));
    int64_t dispatchQuantOutType = static_cast<int64_t>(*dispatchQuantOutTypePtr);
    OP_TILING_CHECK(dispatchQuantOutType != (static_cast<int64_t>(ge::DT_FLOAT8_E5M2)) &&
                    dispatchQuantOutType != (static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN)) &&
                    dispatchQuantOutType != (static_cast<int64_t>(ge::DT_FLOAT4_E2M1)),
        OP_LOGE(nodeName,
            "Invalid dispatchQuantOutType, only support fp8_e5m2, fp8_e4m3fn and fp4_e2m1, "
            "but now is %ld.", dispatchQuantOutType),
        return ge::GRAPH_FAILED);

    auto weightOneDesc = context->GetDynamicInputDesc(config.weightOneIndex, 0);
    int64_t opQuantMode = GetOpQuantModeByAttrDispatchOutType(context, config);
    ge::DataType refWeightDataType = GetDataTypeByOpQuantMode(opQuantMode);
    OP_TILING_CHECK(refWeightDataType == ge::DT_UNDEFINED,
        OP_LOGE(nodeName,
            "unsupported dispatchQuantMode(%ld), leading out data type to being DT_UNDEFINED.", dispatchQuantMode),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((refWeightDataType != weightOneDesc->GetDataType()) &&
                    (weightOneDesc->GetDataType() != ge::DT_FLOAT4_E2M1),
        OP_LOGE(nodeName, "refWeightDataType(%s) should be equal to weightOne & weightTwo dataType(%s).",
        Ops::Base::ToString(refWeightDataType).c_str(), Ops::Base::ToString(weightOneDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);

    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCombineQuantModeIndex));
    OP_TILING_CHECK(*combineQuantModePtr != 0,
        OP_LOGE(nodeName, "Invalid combineQuantMode(%ld), expecting default value(0).", *combineQuantModePtr),
        return ge::GRAPH_FAILED);

    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    OP_TILING_CHECK(std::strcmp(commAlgPtr, "") != 0,
        OP_LOGE(nodeName, "Invalid commAlg(%s), expecting default value("").", commAlgPtr),
        return ge::GRAPH_FAILED);

    auto globalBsPtr = attrs->GetAttrPointer<int64_t>((config.attrGlobalBsIndex));
    int64_t globalBs = static_cast<int64_t>(*globalBsPtr);
    if (globalBs != 0) {
        OP_TILING_CHECK(globalBs < 0 || bs * epWorldSize > globalBs || globalBs % epWorldSize != 0,
            OP_LOGE(nodeName, "globalBs(%ld) either be 0 or (maxBs * EP and mod(globalBs, EP(%ld)) == 0).",
                globalBs, epWorldSize),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetAttrParams(const gert::TilingContext *context, MegaMoeConfig &config,
    MegaMoeTilingData *tilingData, const char *nodeName, const uint32_t aicNum)
{
    auto attrs = context->GetAttrs();

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    auto maxRecvTokenNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex));

    tilingData->epWorldSize = *epWorldSizePtr;
    tilingData->maxOutputSize = *maxRecvTokenNumPtr != 0 ?
        *maxRecvTokenNumPtr :
        tilingData->bs * tilingData->epWorldSize *
        std::min(tilingData->topK, tilingData->expertPerRank);
    tilingData->blockNumPerEP = std::max(static_cast<uint32_t>(1), aicNum / tilingData->epWorldSize);
    tilingData->dispatchRows = 2;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrAndSetTilingData(const gert::TilingContext *context, MegaMoeConfig &config,
    MegaMoeTilingData *tilingData, const uint32_t aicNum)
{
    const char *nodeName = context->GetNodeName();

    OP_TILING_CHECK(CheckAttrPtrNullptr(context, config, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "params check nulld failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAttrParams(context, config, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "check attr params failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetAttrParams(context, config, tilingData, nodeName, aicNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "set attr params failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(
    gert::TilingContext *context, WorkspaceInfo& workspaceInfo, const char *nodeName)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, OP_LOGE(nodeName, "workspace is nullptr."),
        return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(workspaceInfo.workspaceSize == 0LL,
        OP_LOGE(nodeName, "workspaceSize from tilingData is 0."),
        return ge::GRAPH_FAILED);

    int64_t workspaceSize = sysWorkspaceSize + workspaceInfo.workspaceSize + RESERVED_WORKSPACE_SIZE;
    workspace[0] = workspaceSize;

    OP_LOGD(nodeName, "sysWorkspaceSize: %ld \n", sysWorkspaceSize);
    OP_LOGD(nodeName, "mega_moe_tiling workspaceSize: %ld \n", workspaceSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorPtrNullptr(const gert::TilingContext *context,
    MegaMoeConfig &config, const char *nodeName)
{
    auto contextDesc = context->GetInputDesc(config.contextIndex);
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);

    auto weightOneDesc = context->GetDynamicInputDesc(config.weightOneIndex, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(config.weightTwoIndex, 0);
    auto weightScalesOneDesc = context->GetDynamicInputDesc(config.weightScalesOneIndex, 0);
    auto weightScalesTwoDesc = context->GetDynamicInputDesc(config.weightScalesTwoIndex, 0);

    auto yDesc = context->GetOutputDesc(config.yIndex);
    auto expertTokenNumsDesc = context->GetOutputDesc(config.expertTokenNumsIndex);

    OP_CHECK_NULL_WITH_CONTEXT(context, contextDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkWeightsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumsDesc);

    auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
    auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
    OP_TILING_CHECK(xActiveMaskDesc != nullptr,
        OP_LOGE(nodeName, "xActiveMaskDesc should be null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(scalesDesc != nullptr, OP_LOGE(nodeName, "scalesDesc should be null."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWeightTensorDim(const gert::TilingContext *context,
    MegaMoeConfig &config, const char *nodeName)
{
    auto weightOneStorageShape = context->GetDynamicInputShape(config.weightOneIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    OP_TILING_CHECK(weightOneStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "weight1",
        (std::to_string(weightOneStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "3D"),
        return ge::GRAPH_FAILED);
    const int64_t weightOneDim0 = weightOneStorageShape->GetStorageShape().GetDim(0);
    const int64_t weightOneDim1 = weightOneStorageShape->GetStorageShape().GetDim(1);
    const int64_t weightOneDim2 = weightOneStorageShape->GetStorageShape().GetDim(2);
    OP_LOGD(nodeName, "weightOne dim0 = %ld", weightOneDim0);
    OP_LOGD(nodeName, "weightOne dim1 = %ld", weightOneDim1);
    OP_LOGD(nodeName, "weightOne dim2 = %ld", weightOneDim2);

    auto weightTwoStorageShape = context->GetDynamicInputShape(config.weightTwoIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoStorageShape);
    OP_TILING_CHECK(weightTwoStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "weight2",
        (std::to_string(weightTwoStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "3D"),
        return ge::GRAPH_FAILED);
    const int64_t weightTwoDim0 = weightTwoStorageShape->GetStorageShape().GetDim(0);
    const int64_t weightTwoDim1 = weightTwoStorageShape->GetStorageShape().GetDim(1);
    const int64_t weightTwoDim2 = weightTwoStorageShape->GetStorageShape().GetDim(2);
    OP_LOGD(nodeName, "weightTwo dim0 = %ld", weightTwoDim0);
    OP_LOGD(nodeName, "weightTwo dim1 = %ld", weightTwoDim1);
    OP_LOGD(nodeName, "weightTwo dim2 = %ld", weightTwoDim2);

    OP_TILING_CHECK(weightOneDim0 != weightTwoDim0,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, "weight1 and weight2",
        (std::string("[") + std::to_string(weightOneDim0) + ", " + std::to_string(weightTwoDim0) + "]").c_str(),
        "dim0 of weight1 and weight2 should be equal"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(weightOneDim2 != weightTwoDim1 || h != weightOneDim2,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, "weight1, weight2 and x",
        (std::string("[") + std::to_string(weightOneDim2) + ", " + std::to_string(weightTwoDim1) +
         ", " + std::to_string(h) + "]").c_str(),
        "dim2 of weight1 and dim1 of weight2 should equal h of x"),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(weightOneDim1 != weightTwoDim2 * NUM_TWO,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, "weight1 and weight2",
        (std::string("[") + std::to_string(weightOneDim1) + ", " + std::to_string(weightTwoDim2) + "]").c_str(),
        "dim1 of weight1 should equal dim2 of weight2 * 2"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutputTensorDim(const gert::TilingContext *context,
    MegaMoeConfig &config, const char *nodeName)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    auto weightOneStorageShape = context->GetDynamicInputShape(config.weightOneIndex, 0);

    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);
    int64_t expertPerRank = weightOneStorageShape->GetStorageShape().GetDim(0);

    auto yStorageShape = context->GetOutputShape(config.yIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorageShape);
    OP_TILING_CHECK(yStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "y",
        (std::to_string(yStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
        return ge::GRAPH_FAILED);
    const int64_t yDim0 = yStorageShape->GetStorageShape().GetDim(0);
    const int64_t yDim1 = yStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "y dim0 = %ld", yDim0);
    OP_LOGD(nodeName, "y dim1 = %ld", yDim1);

    OP_TILING_CHECK(yDim0 != bs || yDim1 != h,
        OP_LOGE(nodeName, "(yDim0(%ld), yDim1(%ld)) should be equal to (bs(%ld), h(%ld)).",
            yDim0, yDim1, bs, h),
        return ge::GRAPH_FAILED);

    auto expertTokenNumsStorageShape = context->GetOutputShape(config.expertTokenNumsIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumsStorageShape);
    OP_TILING_CHECK(expertTokenNumsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "expert_token_nums",
        (std::to_string(expertTokenNumsStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
        return ge::GRAPH_FAILED);
    const int64_t expertTokenNumsDim0 = expertTokenNumsStorageShape->GetStorageShape().GetDim(0);
    OP_LOGD(nodeName, "expertTokenNums dim0 = %ld", expertTokenNumsDim0);

    OP_TILING_CHECK(expertTokenNumsDim0 != expertPerRank,
        OP_LOGE(nodeName, "expertTokenNumsDim0(%ld) should be equal to expertPerRank(%ld).",
            expertTokenNumsDim0, expertPerRank),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWeightScalesTensorDim(const gert::TilingContext *context,
    MegaMoeConfig &config, const char *nodeName)
{
    auto weightScalesOneStorageShape = context->GetDynamicInputShape(config.weightScalesOneIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneStorageShape);
    OP_TILING_CHECK(weightScalesOneStorageShape->GetStorageShape().GetDimNum() != FOUR_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "weight_scales1",
        (std::to_string(weightScalesOneStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "4D"),
        return ge::GRAPH_FAILED);
    const int64_t weightScalesOneDim0 = weightScalesOneStorageShape->GetStorageShape().GetDim(0);
    const int64_t weightScalesOneDim1 = weightScalesOneStorageShape->GetStorageShape().GetDim(1);
    const int64_t weightScalesOneDim2 = weightScalesOneStorageShape->GetStorageShape().GetDim(2);
    const int64_t weightScalesOneDim3 = weightScalesOneStorageShape->GetStorageShape().GetDim(3);
    OP_LOGD(nodeName, "weightScalesOne dim0 = %ld", weightScalesOneDim0);
    OP_LOGD(nodeName, "weightScalesOne dim1 = %ld", weightScalesOneDim1);
    OP_LOGD(nodeName, "weightScalesOne dim2 = %ld", weightScalesOneDim2);
    OP_LOGD(nodeName, "weightScalesOne dim3 = %ld", weightScalesOneDim3);

    auto weightScalesTwoStorageShape = context->GetDynamicInputShape(config.weightScalesTwoIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoStorageShape);
    OP_TILING_CHECK(weightScalesTwoStorageShape->GetStorageShape().GetDimNum() != FOUR_DIMS,
        OP_LOGE(nodeName, "weightScalesTwoStorageShape dims must be 4, but current dim num is %zu.",
        weightScalesTwoStorageShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    const int64_t weightScalesTwoDim0 = weightScalesTwoStorageShape->GetStorageShape().GetDim(0);
    const int64_t weightScalesTwoDim1 = weightScalesTwoStorageShape->GetStorageShape().GetDim(1);
    const int64_t weightScalesTwoDim2 = weightScalesTwoStorageShape->GetStorageShape().GetDim(2);
    const int64_t weightScalesTwoDim3 = weightScalesTwoStorageShape->GetStorageShape().GetDim(3);
    OP_LOGD(nodeName, "weightScalesTwo dim0 = %ld", weightScalesTwoDim0);
    OP_LOGD(nodeName, "weightScalesTwo dim1 = %ld", weightScalesTwoDim1);
    OP_LOGD(nodeName, "weightScalesTwo dim2 = %ld", weightScalesTwoDim2);
    OP_LOGD(nodeName, "weightScalesTwo dim3 = %ld", weightScalesTwoDim3);

    OP_TILING_CHECK(weightScalesOneDim0 != weightScalesTwoDim0,
        OP_LOGE(nodeName, "weightScalesOneDim0(%ld) and weightScalesTwoDim0(%ld) should be equal.",
            weightScalesOneDim0, weightScalesTwoDim0),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(weightScalesOneDim3 != NUM_TWO || weightScalesOneDim3 != weightScalesTwoDim3,
        OP_LOGE(nodeName, "weightScalesOneDim3(%ld) and weightScalesTwoDim3(%ld) should be equal to 2.",
            weightScalesOneDim3, weightScalesTwoDim3),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(weightScalesTwoDim1 != h,
        OP_LOGE(nodeName, "weightScalesTwoDim1(%ld) should be equal to h(%ld).",
            weightScalesTwoDim1, h),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(weightScalesOneDim2 != ops::CeilDiv(h, INPUT_WEIGHT_SCALES_CEIL_ALIGN),
        OP_LOGE(nodeName,
            "weightScalesOneDim2(%ld) should equal CeilDiv(h(%ld), INPUT_WEIGHT_SCALES_CEIL_ALIGN(%ld)) = %ld.",
            weightScalesOneDim2, h,
            INPUT_WEIGHT_SCALES_CEIL_ALIGN, ops::CeilDiv(h, INPUT_WEIGHT_SCALES_CEIL_ALIGN)),
        return ge::GRAPH_FAILED);

    const int64_t n = weightScalesOneDim1;
    OP_TILING_CHECK(weightScalesTwoDim2 != ops::CeilDiv(n / NUM_TWO, INPUT_WEIGHT_SCALES_CEIL_ALIGN),
        OP_LOGE(nodeName,
            "weightScalesTwoDim2(%ld) should equal CeilDiv(n(%ld) / 2, INPUT_WEIGHT_SCALES_CEIL_ALIGN(%ld)) = %ld.",
            weightScalesTwoDim2, n,
            INPUT_WEIGHT_SCALES_CEIL_ALIGN, ops::CeilDiv(n / NUM_TWO, INPUT_WEIGHT_SCALES_CEIL_ALIGN)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorDim(const gert::TilingContext *context, MegaMoeConfig &config, const char *nodeName)
{
    const gert::StorageShape *contextStorageShape = context->GetInputShape(config.contextIndex);
    OP_TILING_CHECK(contextStorageShape == nullptr,
        OP_LOGE(nodeName, "contextShape is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(contextStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "contextShape dims must be 1, but current dim num is %zu.",
        contextStorageShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    int64_t contextDim0 = contextStorageShape->GetStorageShape().GetDim(0);
    OP_LOGD(nodeName, "context dim0 = %ld", contextDim0);

    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "xShape is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "xShape dims must be 2, but current dim num is %zu.",
        xStorageShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "x dim0 = %ld", xDim0);
    OP_LOGD(nodeName, "x dim1 = %ld", xDim1);

    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    OP_TILING_CHECK(topkIdsStorageShape == nullptr,
        OP_LOGE(nodeName, "topkIdsStorageShape is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(topkIdsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "topkIdsStorageShape dims must be 2, but current dim num is %zu.",
        topkIdsStorageShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    const int64_t topkIdsDim0 = topkIdsStorageShape->GetStorageShape().GetDim(0);
    const int64_t topkIdsDim1 = topkIdsStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "topkIds dim0 = %ld", topkIdsDim0);
    OP_LOGD(nodeName, "topkIds dim1 = %ld", topkIdsDim1);

    const gert::StorageShape *topkWeightsStorageShape = context->GetInputShape(config.topkWeightsIndex);
    OP_TILING_CHECK(topkWeightsStorageShape == nullptr,
        OP_LOGE(nodeName, "topkWeightsStorageShape is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(topkWeightsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "topkWeightsStorageShape dims must be 2, but current dim num is %zu.",
        topkWeightsStorageShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    const int64_t topkWeightsDim0 = topkWeightsStorageShape->GetStorageShape().GetDim(0);
    const int64_t topkWeightsDim1 = topkWeightsStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "topkWeights dim0 = %ld", topkWeightsDim0);
    OP_LOGD(nodeName, "topkWeights dim1 = %ld", topkWeightsDim1);

    OP_TILING_CHECK(xDim0 != topkIdsDim0 && xDim0 != topkWeightsDim0 && topkIdsDim0 != topkWeightsDim0,
        OP_LOGE(nodeName, "xDim0(%ld), topkIdsDim0(%ld), topkWeightsDim0(%ld) should all be equal.",
            xDim0, topkIdsDim0, topkWeightsDim0),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(topkIdsDim1 != topkWeightsDim1,
        OP_LOGE(nodeName, "topkIdsDim1(%ld), topkWeightsDim1(%ld) should be equal.",
            topkIdsDim1, topkWeightsDim1),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckWeightTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "weight params shape is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckWeightScalesTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "optional params shape is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckOutputTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "output params shape is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorDataType(const gert::TilingContext *context,
    MegaMoeConfig &config, const char *nodeName)
{
    auto contextDesc = context->GetInputDesc(config.contextIndex);
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);
    
    auto weightOneDesc = context->GetDynamicInputDesc(config.weightOneIndex, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(config.weightTwoIndex, 0);
    auto weightScalesOneDesc = context->GetDynamicInputDesc(config.weightScalesOneIndex, 0);
    auto weightScalesTwoDesc = context->GetDynamicInputDesc(config.weightScalesTwoIndex, 0);

    auto yDesc = context->GetOutputDesc(config.yIndex);
    auto expertTokenNumsDesc = context->GetOutputDesc(config.expertTokenNumsIndex);

    OP_TILING_CHECK(contextDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "context dataType is invalid, dataType should be DT_INT32, but is %s.",
        Ops::Base::ToString(contextDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(xDesc->GetDataType() != ge::DT_BF16,
        OP_LOGE(nodeName, "x dataType is invalid, dataType should be DT_BF16, but is %s.",
        Ops::Base::ToString(xDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(topkIdsDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "topkIds dataType is invalid, dataType should be DT_INT32, but is %s.",
        Ops::Base::ToString(topkIdsDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    OP_TILING_CHECK((
        (topkWeightsDesc->GetDataType() != ge::DT_BF16) &&
        (topkWeightsDesc->GetDataType() != ge::DT_FLOAT)),
        OP_LOGE(nodeName, "topkWeights dataType is invalid, dataType should be DT_FLOAT or DT_BF16, but is %s.",
        Ops::Base::ToString(topkWeightsDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    OP_TILING_CHECK((
        (weightOneDesc->GetDataType() != ge::DT_FLOAT8_E5M2) &&
        (weightOneDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN) &&
        (weightOneDesc->GetDataType() != ge::DT_FLOAT4_E2M1)),
        OP_LOGE(nodeName,
            "weightOne dataType is invalid, dataType should be DT_FLOAT8_E5M2 or DT_FLOAT8_E4M3FN or DT_FLOAT4_E2M1, "
            "but is %s.", Ops::Base::ToString(weightOneDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK((
        (weightTwoDesc->GetDataType() != ge::DT_FLOAT8_E5M2) &&
        (weightTwoDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN) &&
        (weightTwoDesc->GetDataType() != ge::DT_FLOAT4_E2M1)),
        OP_LOGE(nodeName,
            "weightTwo dataType is invalid, dataType should be DT_FLOAT8_E5M2 or DT_FLOAT8_E4M3FN or DT_FLOAT4_E2M1, "
            "but is %s.", Ops::Base::ToString(weightTwoDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(weightOneDesc->GetDataType() != weightTwoDesc->GetDataType(),
        OP_LOGE(nodeName,
            "weightOne dataType(%s) and weightTwo dataType(%s) should be equal.",
        Ops::Base::ToString(weightOneDesc->GetDataType()).c_str(),
        Ops::Base::ToString(weightTwoDesc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(weightScalesOneDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
        OP_LOGE(nodeName,
            "weightScalesOne dataType is invalid, dataType should be DT_FLOAT8_E8M0, but is %s.",
        Ops::Base::ToString(weightScalesOneDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(weightScalesTwoDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
        OP_LOGE(nodeName,
            "weightScalesTwo dataType is invalid, dataType should be DT_FLOAT8_E8M0, but is %s.",
        Ops::Base::ToString(weightScalesTwoDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(yDesc->GetDataType() != ge::DT_BF16,
        OP_LOGE(nodeName, "y dataType is invalid, dataType should be DT_BF16, but is %s.",
        Ops::Base::ToString(yDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(expertTokenNumsDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "expertTokenNums dataType is invalid, dataType should be DT_INT32, but is %s.",
        Ops::Base::ToString(expertTokenNumsDesc->GetDataType()).c_str()), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorFormat(const gert::TilingContext *context,
    MegaMoeConfig &config, const char *nodeName)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);
    
    auto weightOneDesc = context->GetDynamicInputDesc(config.weightOneIndex, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(config.weightTwoIndex, 0);
    auto weightScalesOneDesc = context->GetDynamicInputDesc(config.weightScalesOneIndex, 0);
    auto weightScalesTwoDesc = context->GetDynamicInputDesc(config.weightScalesTwoIndex, 0);

    auto yDesc = context->GetOutputDesc(config.yIndex);
    auto expertTokenNumsDesc = context->GetOutputDesc(config.expertTokenNumsIndex);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName,
            "x format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(topkIdsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName,
            "topkIds format is invalid."), return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(topkWeightsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName,
            "topkWeights format is invalid."), return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(weightOneDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName,
            "weightOne format is invalid."), return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(weightTwoDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName,
            "weightTwo format is invalid."), return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(weightScalesOneDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName,
            "weightScalesOne format is invalid."), return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(weightScalesTwoDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName,
            "weightScalesTwo format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(yDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName,
            "y format is invalid."), return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expertTokenNumsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName,
            "expertTokenNums format is invalid."), return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingCheckMegaMoe(const gert::TilingContext *context,
    MegaMoeConfig &config, const char *nodeName)
{
    OP_TILING_CHECK(CheckTensorPtrNullptr(context, config, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "params check nulld failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "params shape is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorDataType(context, config, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "params dataType is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorFormat(context, config, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "params format is invalid."), return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputParam(const gert::TilingContext *context, MegaMoeConfig &config, const char *nodeName)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);

    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(xDim0 < MIN_BS || xDim0 > MAX_BS,
        OP_LOGE(nodeName, "BS should in [%ld, %ld], but now BS is %ld.", MIN_BS, MAX_BS, xDim0),
        return ge::GRAPH_FAILED);

    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(xDim1 != 4LL * H_BASE && xDim1 != 5LL * H_BASE && xDim1 != 7LL * H_BASE,
        OP_LOGE(nodeName, "H only support 4k/5k/7k, but now H is %ld.", xDim1),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    int64_t topkIdsDim1 = topkIdsStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(topkIdsDim1 != 6LL && topkIdsDim1 != 8LL,
        OP_LOGE(nodeName, "topK only support 6/8, but now topK is %ld.", topkIdsDim1),
        return ge::GRAPH_FAILED);

    auto weightOneStorageShape = context->GetDynamicInputShape(config.weightOneIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    int64_t weightOneDim0 = weightOneStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(weightOneDim0 < MIN_EXPERT_PER_RANK || weightOneDim0 > MAX_EXPERT_PER_RANK,
        OP_LOGE(nodeName,
            "expertPerRank should in [%ld, %ld], but now expertPerRank is %ld.",
            MIN_EXPERT_PER_RANK, MAX_EXPERT_PER_RANK, weightOneDim0),
        return ge::GRAPH_FAILED);

    int64_t weightOneDim1 = weightOneStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(weightOneDim1 != 1LL * HIDDEN_DIM_BASE && weightOneDim1 != 2LL * HIDDEN_DIM_BASE &&
                    weightOneDim1 != 3LL * HIDDEN_DIM_BASE && weightOneDim1 != 4LL * HIDDEN_DIM_BASE &&
                    weightOneDim1 != 7LL * HIDDEN_DIM_BASE,
        OP_LOGE(nodeName, "hiddenDim only support 1k/2k/3k/4k/7k, but now hiddenDim is %ld.", weightOneDim1),
        return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetInputParam(const gert::TilingContext *context,
    MegaMoeTilingData *tilingData, MegaMoeConfig &config)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);

    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    int64_t topK = topkIdsStorageShape->GetStorageShape().GetDim(1);

    auto weightOneStorageShape = context->GetDynamicInputShape(config.weightOneIndex, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    int64_t expertPerRank = weightOneStorageShape->GetStorageShape().GetDim(0);
    int64_t hiddenDim = weightOneStorageShape->GetStorageShape().GetDim(1);

    tilingData->bs = static_cast<uint32_t>(bs);
    tilingData->h = static_cast<uint32_t>(h);
    tilingData->hiddenDim = static_cast<uint32_t>(hiddenDim);
    tilingData->topK = static_cast<uint32_t>(topK);
    tilingData->expertPerRank = static_cast<uint32_t>(expertPerRank);
    tilingData->groupListType = 1;

    tilingData->transX = false;
    tilingData->transW = true;
    tilingData->transW2 = true;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetInput(const gert::TilingContext *context, MegaMoeTilingData *tilingData,
    MegaMoeConfig &config, const char *nodeName)
{
    OP_TILING_CHECK(TilingCheckMegaMoe(context, config, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "check input failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckInputParam(context, config, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "check input param failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetInputParam(context, tilingData, config) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "set input param failed."), return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MegaMoeTilingFuncImplPublic(gert::TilingContext *context, MegaMoeConfig &config)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter MegaMoe tiling check func.");

    MegaMoeTilingData *tilingData = context->GetTilingData<MegaMoeTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."),
        return ge::GRAPH_FAILED);

    // Input check & set
    OP_TILING_CHECK(CheckAndSetInput(context, tilingData, config, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "Check and set input failed."), return ge::GRAPH_FAILED);

    // Platform Info
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_TILING_CHECK(aivNum <= 0 || aicNum <= 0,
        OP_LOGE(nodeName, "aivNum(%u) or aicNum(%u) less than or equal 0.", aivNum, aicNum),
        return ge::GRAPH_FAILED);
    tilingData->aicNum = aicNum;

    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum));
    context->SetScheduleMode(1); // batch model, all cores start at the same time
    OP_LOGI(nodeName, "TilingData Init: aivNum: %u, aicNum: %u, ubSize:%lu \n", aivNum, aicNum, ubSize);

    // Attr check & set
    OP_TILING_CHECK(CheckAttrAndSetTilingData(context, config, tilingData, aicNum) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "Getting attr failed."), return ge::GRAPH_FAILED);

    // Cal TilingKey
    uint64_t tilingKey = CalTilingKey(context, config, tilingData, nodeName);
    OP_LOGI(nodeName, "OP TilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);

    // WorkspaceSize
    WorkspaceInfo workspaceInfo((uint8_t*)0, tilingData);
    OP_TILING_CHECK(SetWorkspace(context, workspaceInfo1, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "Tiling set workspace Failed"), return ge::GRAPH_FAILED);

    // Print Info
    PrintMegaMoeTilingData(tilingData, nodeName);
    printWorkspaceInfo(&workspaceInfo, nodeName);
    printPeermemInfo(tilingData, nodeName);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MegaMoeTilingFunc(gert::TilingContext* context)
{
    MegaMoeConfig config;
    ge::graphStatus ret;

    ret = MegaMoeTilingFuncImplPublic(context, config);

    return ret;
}

struct MegaMoeCompileInfo {};
static ge::graphStatus TilingParseForMegaMoe(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MegaMoe)
    .Tiling(MegaMoeTilingFunc)
    .TilingParse<MegaMoeCompileInfo>(TilingParseForMegaMoe);
} // namespace optiling