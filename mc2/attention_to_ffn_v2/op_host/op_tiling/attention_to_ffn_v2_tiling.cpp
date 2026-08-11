/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file attention_to_ffn_v2_tiling.cpp
 * \brief
 */

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "mc2_hcom_topo_info.h"
#include "attention_to_ffn_v2_tiling.h"
#include "../../op_kernel/attention_to_ffn_v2_tiling.h"
#include "../../op_kernel/attention_to_ffn_v2_tiling_key.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace MC2Tiling {
namespace {
constexpr uint32_t ATTR_GROUP_INDEX = 0U;
constexpr uint32_t ATTR_WORLD_SIZE_INDEX = 1U;
constexpr uint32_t ATTR_FFN_TOKEN_INFO_SHAPE_INDEX = 2U;
constexpr uint32_t ATTR_FFN_TOKEN_DATA_SHAPE_INDEX = 3U;
constexpr uint32_t ATTR_ATTN_TOKEN_INFO_SHAPE_INDEX = 4U;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 5U;
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 6U;
constexpr uint32_t ATTR_SYNC_FLAG_INDEX = 7U;
constexpr uint32_t ATTR_FFN_START_RANK_ID_INDEX = 8U;
constexpr uint32_t ATTR_CCL_BUFFER_SIZE_INDEX = 9U;

constexpr uint32_t DIM_ONE = 1U;
constexpr uint32_t DIM_TWO = 2U;
constexpr uint32_t DIM_THREE = 3U;
constexpr uint32_t DIM_FIVE = 5U;
constexpr uint32_t H_MIN = 1024U;
constexpr uint32_t H_MAX = 8192U;
constexpr uint32_t BS_MAX = 512U;
constexpr uint32_t K_MAX = 16U;
constexpr uint32_t MOE_EXPERT_MAX_NUM = 1024U;
constexpr uint32_t SHARED_EXPERT_MAX_NUM = 4U;
constexpr uint32_t WORLD_SIZE_MIN = 2U;
constexpr uint32_t WORLD_SIZE_MAX = 768U;
constexpr uint32_t WORKSPACE_ELEMENT_OFFSET = 128U;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;

constexpr uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return (value + align - 1U) / align * align;
}

ge::graphStatus CheckQuantMode(const char *nodeName, uint32_t quantMode, bool isScales)
{
    OP_TILING_CHECK(quantMode != ATTN_FFN_TILINGKEY_NO_QUANT &&
                        quantMode != ATTN_FFN_TILINGKEY_PERTOKEN_QUANT &&
                        quantMode != ATTN_FFN_TILINGKEY_PERGROUP_QUANT &&
                        quantMode != ATTN_FFN_TILINGKEY_MX_QUANT &&
                        quantMode != ATTN_FFN_TILINGKEY_MX_QUANT_CLIP,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "quant_mode", std::to_string(quantMode).c_str(),
                                              "0, 2, 3, 4 or 5"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((quantMode == ATTN_FFN_TILINGKEY_MX_QUANT ||
                     quantMode == ATTN_FFN_TILINGKEY_MX_QUANT_CLIP) &&
                        isScales,
                    OP_LOGE(nodeName, "scales must be absent in MX and MX-clip modes"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetV2TilingKey(gert::TilingContext *context, const AttentionToFfnV2Info &info)
{
    const uint64_t tilingKey = GET_TPL_TILING_KEY(
        info.quantMode, info.isScales, info.syncFlag == 1U, info.isActiveMask, TILINGKEY_TPL_A5);
    context->SetTilingKey(tilingKey);
    OP_LOGD(context->GetNodeName(), "AttentionToFfnV2 cur case tilingKey is %lu", tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetPlatformAndCommTiling(gert::TilingContext *context, const char *group,
                                         AttentionToFfnV2TilingData &tilingData)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingData.attentionToFfnV2Info.aivNum = aivNum;
    tilingData.attentionToFfnV2Info.totalUbSize = ubSize;
    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum));

    uint32_t ffnNum = tilingData.attentionToFfnV2Info.worldSize -
                      tilingData.attentionToFfnV2Info.attentionWorkerNum;
    uint32_t ffnNumAlignSize = AlignUp(ffnNum * sizeof(int32_t), WORKSPACE_ELEMENT_OFFSET);
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "workSpaces"),
                    return ge::GRAPH_FAILED);
    workSpaces[0] = ascendcPlatform.GetLibApiWorkSpaceSize() + static_cast<size_t>(ffnNumAlignSize * aivNum);

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, OP_TYPE_ALL_TO_ALL,
                                                  "AlltoAll=level0:fullmesh;level1:pairwise");
    mc2CcTilingConfig.GetTiling(tilingData.mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData.mc2CcTiling1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingNewQuantMode(gert::TilingContext *context, uint32_t quantMode)
{
    const char *nodeName = context->GetNodeName();
    auto *tilingData = context->GetTilingData<AttentionToFfnV2TilingData>();
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    auto groupPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    auto worldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_WORLD_SIZE_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto syncFlagPtr = attrs->GetAttrPointer<int64_t>(ATTR_SYNC_FLAG_INDEX);
    auto ffnStartRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_FFN_START_RANK_ID_INDEX);
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_CCL_BUFFER_SIZE_INDEX);
    auto ffnInfoShapeAttr = attrs->GetListInt(ATTR_FFN_TOKEN_INFO_SHAPE_INDEX);
    auto ffnDataShapeAttr = attrs->GetListInt(ATTR_FFN_TOKEN_DATA_SHAPE_INDEX);
    auto attnInfoShapeAttr = attrs->GetListInt(ATTR_ATTN_TOKEN_INFO_SHAPE_INDEX);
    OP_TILING_CHECK(groupPtr == nullptr || worldSizePtr == nullptr || moeExpertNumPtr == nullptr ||
                        syncFlagPtr == nullptr || ffnStartRankIdPtr == nullptr || cclBufferSizePtr == nullptr ||
                        ffnInfoShapeAttr == nullptr || ffnDataShapeAttr == nullptr || attnInfoShapeAttr == nullptr,
                    OP_LOGE(nodeName, "required attribute is null"), return ge::GRAPH_FAILED);

    const gert::StorageShape *contextShape = context->GetInputShape(0U);
    const gert::StorageShape *xShape = context->GetInputShape(1U);
    const gert::StorageShape *sessionIdShape = context->GetInputShape(2U);
    const gert::StorageShape *microBatchIdShape = context->GetInputShape(3U);
    const gert::StorageShape *layerIdShape = context->GetInputShape(4U);
    const gert::StorageShape *expertIdsShape = context->GetInputShape(5U);
    const gert::StorageShape *expertRankTableShape = context->GetInputShape(6U);
    OP_TILING_CHECK(contextShape == nullptr || xShape == nullptr || sessionIdShape == nullptr ||
                        microBatchIdShape == nullptr || layerIdShape == nullptr || expertIdsShape == nullptr ||
                        expertRankTableShape == nullptr,
                    OP_LOGE(nodeName, "required input shape is null"), return ge::GRAPH_FAILED);

    auto contextDesc = context->GetInputDesc(0U);
    auto xDesc = context->GetInputDesc(1U);
    auto sessionIdDesc = context->GetInputDesc(2U);
    auto microBatchIdDesc = context->GetInputDesc(3U);
    auto layerIdDesc = context->GetInputDesc(4U);
    auto expertIdsDesc = context->GetInputDesc(5U);
    auto expertRankTableDesc = context->GetInputDesc(6U);
    OP_TILING_CHECK(contextDesc == nullptr || xDesc == nullptr || sessionIdDesc == nullptr ||
                        microBatchIdDesc == nullptr || layerIdDesc == nullptr || expertIdsDesc == nullptr ||
                        expertRankTableDesc == nullptr,
                    OP_LOGE(nodeName, "required input desc is null"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(contextShape->GetStorageShape().GetDimNum() != DIM_ONE ||
                        contextDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "context must be a 1D INT32 tensor"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xShape->GetStorageShape().GetDimNum() != DIM_THREE ||
                        (xDesc->GetDataType() != ge::DT_FLOAT16 && xDesc->GetDataType() != ge::DT_BF16),
                    OP_LOGE(nodeName, "x must be a 3D FLOAT16/BF16 tensor"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sessionIdShape->GetStorageShape().GetDimNum() != DIM_ONE ||
                        microBatchIdShape->GetStorageShape().GetDimNum() != DIM_ONE ||
                        layerIdShape->GetStorageShape().GetDimNum() != DIM_ONE ||
                        sessionIdDesc->GetDataType() != ge::DT_INT32 ||
                        microBatchIdDesc->GetDataType() != ge::DT_INT32 || layerIdDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "session_id, micro_batch_id and layer_id must be 1D INT32 tensors"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertIdsShape->GetStorageShape().GetDimNum() != DIM_THREE ||
                        expertIdsDesc->GetDataType() != ge::DT_INT32 ||
                        expertRankTableShape->GetStorageShape().GetDimNum() != DIM_THREE ||
                        expertRankTableDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "expert_ids and expert_rank_table must be 3D INT32 tensors"),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *scalesShape = context->GetOptionalInputShape(7U);
    const gert::StorageShape *activeMaskShape = context->GetOptionalInputShape(8U);
    bool isScales = scalesShape != nullptr && scalesShape->GetStorageShape().GetDimNum() != 0U;
    bool isActiveMask = activeMaskShape != nullptr && activeMaskShape->GetStorageShape().GetDimNum() != 0U;
    OP_TILING_CHECK(CheckQuantMode(nodeName, quantMode, isScales) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "quant mode validation failed"), return ge::GRAPH_FAILED);

    const int64_t xDim0 = xShape->GetStorageShape().GetDim(0);
    const int64_t bs = xShape->GetStorageShape().GetDim(1);
    const int64_t h = xShape->GetStorageShape().GetDim(2);
    const int64_t expertIdsK = expertIdsShape->GetStorageShape().GetDim(2);
    const int64_t layerNum = expertRankTableShape->GetStorageShape().GetDim(0);
    const int64_t expertNum = expertRankTableShape->GetStorageShape().GetDim(1);
    const int64_t expRankTableM = expertRankTableShape->GetStorageShape().GetDim(2);
    OP_TILING_CHECK(xDim0 != 1 || bs <= 0 || bs > BS_MAX || h < H_MIN || h > H_MAX,
                    OP_LOGE(nodeName, "x shape is out of range"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertIdsShape->GetStorageShape().GetDim(0) != xDim0 ||
                        expertIdsShape->GetStorageShape().GetDim(1) != bs || expertIdsK <= 0 || expertIdsK > K_MAX ||
                        expertIdsK > *moeExpertNumPtr,
                    OP_LOGE(nodeName, "expert_ids shape is invalid"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*moeExpertNumPtr <= 0 || *moeExpertNumPtr > MOE_EXPERT_MAX_NUM ||
                        expertNum < *moeExpertNumPtr || expertNum > *moeExpertNumPtr + SHARED_EXPERT_MAX_NUM,
                    OP_LOGE(nodeName, "expert count is invalid"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*worldSizePtr < WORLD_SIZE_MIN || *worldSizePtr > WORLD_SIZE_MAX || *syncFlagPtr < 0 ||
                        *syncFlagPtr > 1 || *ffnStartRankIdPtr < 0 || *ffnStartRankIdPtr >= *worldSizePtr ||
                        *cclBufferSizePtr <= 0,
                    OP_LOGE(nodeName, "communication attribute is invalid"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(ffnInfoShapeAttr->GetSize() != DIM_THREE || ffnDataShapeAttr->GetSize() != DIM_FIVE ||
                        attnInfoShapeAttr->GetSize() != DIM_THREE,
                    OP_LOGE(nodeName, "token table shape attribute has invalid rank"), return ge::GRAPH_FAILED);
    const int64_t *ffnInfoShape = ffnInfoShapeAttr->GetData();
    const int64_t *ffnDataShape = ffnDataShapeAttr->GetData();
    const int64_t *attnInfoShape = attnInfoShapeAttr->GetData();
    const int64_t sharedExpertNum = expertNum - *moeExpertNumPtr;
    const int64_t kAndShared = expertIdsK + sharedExpertNum;
    OP_TILING_CHECK(ffnInfoShape[0] != ffnDataShape[0] || ffnInfoShape[1] != ffnDataShape[1] ||
                        ffnDataShape[0] <= 0 || ffnDataShape[0] >= *worldSizePtr || ffnDataShape[2] != bs ||
                        ffnDataShape[3] != kAndShared || attnInfoShape[0] != ffnDataShape[1] ||
                        attnInfoShape[1] != bs || attnInfoShape[2] != kAndShared ||
                        ffnInfoShape[2] != 2 + bs * kAndShared,
                    OP_LOGE(nodeName, "token table shape attribute is inconsistent with inputs"),
                    return ge::GRAPH_FAILED);

    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(7U);
        OP_TILING_CHECK(scalesDesc == nullptr || scalesDesc->GetDataType() != ge::DT_FLOAT ||
                            scalesShape->GetStorageShape().GetDimNum() != DIM_THREE ||
                            scalesShape->GetStorageShape().GetDim(0) != layerNum ||
                            scalesShape->GetStorageShape().GetDim(1) != expertNum ||
                            scalesShape->GetStorageShape().GetDim(2) != h,
                        OP_LOGE(nodeName, "scales must have shape [L, expertNum, H] and FLOAT dtype"),
                        return ge::GRAPH_FAILED);
    }
    if (isActiveMask) {
        auto activeMaskDesc = context->GetOptionalInputDesc(8U);
        OP_TILING_CHECK(activeMaskDesc == nullptr || activeMaskDesc->GetDataType() != ge::DT_BOOL ||
                            activeMaskShape->GetStorageShape().GetDimNum() != DIM_TWO ||
                            activeMaskShape->GetStorageShape().GetDim(0) != xDim0 ||
                            activeMaskShape->GetStorageShape().GetDim(1) != bs,
                        OP_LOGE(nodeName, "active_mask must have shape [1, BS] and BOOL dtype"),
                        return ge::GRAPH_FAILED);
    }

    auto &info = tilingData->attentionToFfnV2Info;
    info.X = static_cast<uint32_t>(xDim0);
    info.BS = static_cast<uint32_t>(bs);
    info.H = static_cast<uint32_t>(h);
    info.K = static_cast<uint32_t>(expertIdsK);
    info.L = static_cast<uint32_t>(layerNum);
    info.expertNum = static_cast<uint32_t>(expertNum);
    info.moeExpertNum = static_cast<uint32_t>(*moeExpertNumPtr);
    info.sharedExpertNum = static_cast<uint32_t>(sharedExpertNum);
    info.HS = static_cast<uint32_t>(ffnDataShape[4]);
    info.expRankTableM = static_cast<uint32_t>(expRankTableM);
    info.microBatchNum = static_cast<uint32_t>(ffnDataShape[1]);
    info.attentionWorkerNum = static_cast<uint32_t>(ffnDataShape[0]);
    info.infoTableLastDimNum = static_cast<uint32_t>(ffnInfoShape[2]);
    info.worldSize = static_cast<uint32_t>(*worldSizePtr);
    info.syncFlag = static_cast<uint32_t>(*syncFlagPtr);
    info.quantMode = quantMode;
    info.ffnStartRankId = static_cast<uint32_t>(*ffnStartRankIdPtr);
    info.isScales = isScales;
    info.isActiveMask = isActiveMask;
    info.windowType = 0U;

    uint64_t tokenInfoBytes = static_cast<uint64_t>(info.attentionWorkerNum) * info.microBatchNum *
                              info.infoTableLastDimNum * sizeof(int32_t);
    uint64_t tokenDataBytes = static_cast<uint64_t>(info.attentionWorkerNum) * info.microBatchNum * info.BS *
                              (info.K + info.sharedExpertNum) * info.HS;
    uint64_t requiredWindowBytes = tokenInfoBytes + tokenDataBytes;
    uint64_t maxWindowSize = mc2tiling::Mc2TilingUtils::GetMaxWindowSize();
    OP_TILING_CHECK(requiredWindowBytes > maxWindowSize ||
                        requiredWindowBytes > static_cast<uint64_t>(*cclBufferSizePtr),
                    OP_LOGE(nodeName, "window buffer is smaller than required size %lu", requiredWindowBytes),
                    return ge::GRAPH_FAILED);
    info.totalWinSize = maxWindowSize;

    OP_TILING_CHECK(SetPlatformAndCommTiling(context, groupPtr, *tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "failed to set platform or communication tiling"), return ge::GRAPH_FAILED);
    return SetV2TilingKey(context, info);
}
} // namespace

ge::graphStatus AttentionToFfnV2TilingFunc(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "attrs"),
                    return ge::GRAPH_FAILED);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    OP_TILING_CHECK(quantModePtr == nullptr, OP_LOGE(context->GetNodeName(), "quant_mode is null"),
                    return ge::GRAPH_FAILED);
    uint32_t quantMode = static_cast<uint32_t>(*quantModePtr);

    const gert::StorageShape *scalesShape = context->GetOptionalInputShape(7U);
    bool isScales = scalesShape != nullptr && scalesShape->GetStorageShape().GetDimNum() != 0U;
    OP_TILING_CHECK(CheckQuantMode(context->GetNodeName(), quantMode, isScales) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "quant mode validation failed"),
                    return ge::GRAPH_FAILED);

    if (quantMode != ATTN_FFN_TILINGKEY_NO_QUANT &&
        quantMode != ATTN_FFN_TILINGKEY_PERTOKEN_QUANT) {
        return TilingNewQuantMode(context, quantMode);
    }

    AttentionToFFNTilingConfig config;
    config.contextIndex = 0U;
    config.xIndex = 1U;
    config.sessionIdIndex = 2U;
    config.microBatchIdIndex = 3U;
    config.layerIdIndex = 4U;
    config.expertIdsIndex = 5U;
    config.expertRankTableIndex = 6U;
    config.scalesIndex = 7U;
    config.activeMaskIndex = 8U;
    config.attrGroupIndex = 0U;
    config.attrWorldSizeIndex = 1U;
    config.attrFfnTokenInfoTableShapeIndex = 2U;
    config.attrFfnTokenDataShapeIndex = 3U;
    config.attrAttnTokenInfoTableShapeIndex = 4U;
    config.attrMoeExpertNumIndex = 5U;
    config.attrQuantModeIndex = 6U;
    config.attrSyncFlagIndex = ATTR_SYNC_FLAG_INDEX;
    config.attrFfnStartRankIdIndex = ATTR_FFN_START_RANK_ID_INDEX;
    config.attrCclBufferSizeIndex = ATTR_CCL_BUFFER_SIZE_INDEX;
    config.isMc2Context = true;

    auto ret = AttentionToFFNTilingFuncBase(context, config);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // Override V1's A3 key with the V2 A5 key.
    AttentionToFfnV2TilingData *tilingData = context->GetTilingData<AttentionToFfnV2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "tilingData"),
                    return ge::GRAPH_FAILED);
    return SetV2TilingKey(context, tilingData->attentionToFfnV2Info);
}

struct AttentionToFfnV2CompileInfo {};
ge::graphStatus TilingParseForAttentionToFfnV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AttentionToFfnV2)
    .Tiling(AttentionToFfnV2TilingFunc)
    .TilingParse<AttentionToFfnV2CompileInfo>(TilingParseForAttentionToFfnV2);
} // namespace MC2Tiling
