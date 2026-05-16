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
 * \file moe_distribute_dispatch_tiling_arch35.cpp
 * \brief
 */

#include "moe_distribute_dispatch_v2_tiling_arch35.h"

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <sys/types.h>
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "../../../op_kernel/moe_distribute_dispatch_v2_tiling.h"
#include "../../../op_kernel/moe_distribute_dispatch_v2_tiling_key.h"
using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;

namespace {
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
constexpr uint32_t ATTR_TP_WORLD_SIZE_INDEX = 5;
constexpr uint32_t ATTR_TP_RANK_ID_INDEX = 6;
constexpr uint32_t ATTR_EXPERT_SHARD_TYPE_INDEX = 7;
constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 8;
constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 9;
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 10;
constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 11;
constexpr uint32_t ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX = 12;
constexpr uint32_t ATTR_Y_DATATYPE_INDEX = 13;
constexpr uint32_t ATTR_COMM_ALG_INDEX = 13;

const size_t MAX_GROUP_NAME_LENGTH = 128UL;
const int64_t MAX_TP_WORLD_SIZE = 2;
const int64_t BS_UPPER_BOUND = 512;
const int64_t COUNT_OFFSET = 512;
const uint64_t BUFFER_NUM = 2;
const uint64_t EVEN_ALIGN = 2;
const uint64_t COMM_ALIGN = 512U;

constexpr uint32_t HCCL_CMD_ALLGATHER = 6U;
constexpr uint32_t HCCL_CMD_ALLTOALLV = 8U;
constexpr uint32_t HCCL_VERSION = 3U;

constexpr uint32_t NUM_0 = 0;
constexpr uint32_t NUM_1 = 1;
constexpr uint32_t NUM_10 = 10;
constexpr uint32_t NUM_100 = 100;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U;

constexpr uint32_t EP_WORLD_SIZE_FOUR = 4;
constexpr uint32_t EP_WORLD_SIZE_TWO = 2;

constexpr uint64_t MX_BLOCK_SIZE = 32U;
constexpr uint64_t MX_PAD_ALIGN = 256U;
constexpr uint64_t PERGROUP_BLOCK_SIZE = 128U;
constexpr uint64_t PERGROUP_PAD_ALIGN = 128U;
constexpr uint64_t STATUS_SIZE = 512U;

constexpr uint64_t STATIC_SCALE_DIM_0 = 1;
constexpr uint64_t HIF8_SCALE_DIM_0 = 1;
constexpr uint64_t ONE_DIM_SCALE_COL_NUM = 1;

constexpr uint32_t MAX_UINT32 = 4294967295;

const std::string OP_NAME = "MoeDistributeDispatchA5";

// V1、V2对外接口约束不一致，按版本区分范围校验的变量名
constexpr int64_t MAX_MOE_EXPERT_NUM_V2 = 1024;
constexpr int64_t MAX_MOE_EXPERT_NUM_V1 = 512;
const int64_t H_V1 = 7168;
const int64_t H_UPPER_BOUND_V2 = 8192;
const int64_t H_LOWER_BOUND_V2 = 1024;
const int64_t MAX_EP_WORLD_SIZE_V1 = 288;
const int64_t MAX_EP_WORLD_SIZE_V2 = 768;
const int64_t MIN_EP_WORLD_SIZE_V2 = 2;
const int64_t MAX_SHARED_EXPERT_NUM_V2 = 4;
const int64_t MIN_SHARED_EXPERT_NUM_V2 = 0;
constexpr int64_t K_UPPER_BOUND_V1 = 8;
const int64_t K_UPPER_BOUND_V2 = 16;

// A5
constexpr uint32_t OP_VERSION_2 = 2;
} // namespace

namespace optiling {
static void PrintTilingDataInfo(const char *nodeName, MoeDistributeDispatchV2TilingData &tilingData)
{
    OP_LOGD(nodeName, "epWorldSize is %u.", tilingData.moeDistributeDispatchV2Info.epWorldSize);
    OP_LOGD(nodeName, "tpWorldSize is %u.", tilingData.moeDistributeDispatchV2Info.tpWorldSize);
    OP_LOGD(nodeName, "epRankId is %u.", tilingData.moeDistributeDispatchV2Info.epRankId);
    OP_LOGD(nodeName, "tpRankId is %u.", tilingData.moeDistributeDispatchV2Info.tpRankId);
    OP_LOGD(nodeName, "expertShardType is %u.", tilingData.moeDistributeDispatchV2Info.expertShardType);
    OP_LOGD(nodeName, "sharedExpertRankNum is %u.", tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum);
    OP_LOGD(nodeName, "moeExpertNum is %u.", tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    OP_LOGD(nodeName, "quantMode is %u.", tilingData.moeDistributeDispatchV2Info.quantMode);
    OP_LOGD(nodeName, "globalBs is %u.", tilingData.moeDistributeDispatchV2Info.globalBs);
    OP_LOGD(nodeName, "bs is %u.", tilingData.moeDistributeDispatchV2Info.bs);
    OP_LOGD(nodeName, "k is %u.", tilingData.moeDistributeDispatchV2Info.k);
    OP_LOGD(nodeName, "h is %u.", tilingData.moeDistributeDispatchV2Info.h);
    OP_LOGD(nodeName, "aivNum is %u.", tilingData.moeDistributeDispatchV2Info.aivNum);
    OP_LOGD(nodeName, "totalUbSize is %lu.", tilingData.moeDistributeDispatchV2Info.totalUbSize);
    OP_LOGD(nodeName, "totalWinSizeEp is %lu.", tilingData.moeDistributeDispatchV2Info.totalWinSizeEp);
    OP_LOGD(nodeName, "expertTokenNumsType is %u.", tilingData.moeDistributeDispatchV2Info.expertTokenNumsType);

    OP_LOGD(nodeName, "scalesCol is %lu.", tilingData.moeDistributeDispatchV2Info.scalesCol);
    OP_LOGD(nodeName, "scalesRow is %lu.", tilingData.moeDistributeDispatchV2Info.scalesRow);
    OP_LOGD(nodeName, "scalesTypeSize is %u.", tilingData.moeDistributeDispatchV2Info.scalesTypeSize);
    OP_LOGD(nodeName, "scalesCount is %lu.", tilingData.moeDistributeDispatchV2Info.scalesCount);
}

inline ge::graphStatus CheckTpAttrs(const char *nodeName, const int64_t tpWorldSize, const int64_t tpRankId,
                                    const char *groupTpPtr, std::string &groupTp)
{
    OP_TILING_CHECK((tpWorldSize < 0) || (tpWorldSize > MAX_TP_WORLD_SIZE),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "tp_world_size",
                                                          std::to_string(tpWorldSize).c_str(), "valid range is [0, 2]"),
                    return ge::GRAPH_FAILED);
    if (tpWorldSize > 1) {
        OP_TILING_CHECK(
            (tpRankId < 0) || (tpRankId >= tpWorldSize),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "tp_rank_id", std::to_string(tpRankId).c_str(),
                                                  ("valid range is [0, " + std::to_string(tpWorldSize) + ")").c_str()),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK((groupTpPtr == nullptr), OP_LOGE_WITH_INVALID_INPUT(nodeName, "group_tp"),
                        return ge::GRAPH_FAILED);
        uint64_t len = strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH);
        OP_TILING_CHECK((len == 0) || (len == MAX_GROUP_NAME_LENGTH),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "group_tp", std::to_string(len).c_str(),
                                                              "valid length range is (0, 128)"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(tpRankId != 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "tp_rank_id", std::to_string(tpRankId).c_str(),
                                                              "expected value is 0 in NoTp mode"),
                        return ge::GRAPH_FAILED);
    }
    groupTp = std::string(groupTpPtr);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckEpAttrs(const char *nodeName, const int64_t epWorldSize, const int64_t epRankId,
                                    const char *groupEpPtr, std::string &groupEp)
{
    OP_TILING_CHECK(groupEpPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "group_ep"), return ge::GRAPH_FAILED);
    uint64_t len = strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH);
    OP_TILING_CHECK((len == 0) || (len == MAX_GROUP_NAME_LENGTH),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "group_ep", std::to_string(len).c_str(),
                                                          "valid length range is (0, 128)"),
                    return ge::GRAPH_FAILED);
    if (OpVersionManager::GetInstance().GetVersion() == OP_VERSION_1) {
        OP_TILING_CHECK((epWorldSize <= 0) || (epWorldSize > MAX_EP_WORLD_SIZE_V1),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "ep_world_size",
                                                              std::to_string(epWorldSize).c_str(),
                                                              "valid range is (0, 288] for V1"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK((epWorldSize < MIN_EP_WORLD_SIZE_V2) || (epWorldSize > MAX_EP_WORLD_SIZE_V2),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "ep_world_size",
                                                              std::to_string(epWorldSize).c_str(),
                                                              "valid range is [2, 768] for V2"),
                        return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(
        (epRankId < 0) || (epRankId >= epWorldSize),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "ep_rank_id", std::to_string(epRankId).c_str(),
                                              ("valid range is [0, " + std::to_string(epWorldSize) + ")").c_str()),
        return ge::GRAPH_FAILED);
    groupEp = std::string(groupEpPtr);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckQuantAndExpertAttrs(const char *nodeName, const int64_t sharedExpertNum,
                                                const int64_t quantMode, const int64_t expertTokenNumsType)
{
    OP_TILING_CHECK((quantMode < static_cast<int64_t>(QuantModeA5::NON_QUANT)) ||
                        (quantMode >= static_cast<int64_t>(QuantModeA5::BUTT)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "quant_mode", std::to_string(quantMode).c_str(),
                                                          "valid range is [0, 5)"),
                    return ge::GRAPH_FAILED);
    if (OpVersionManager::GetInstance().GetVersion() == OP_VERSION_1) {
        OP_TILING_CHECK(sharedExpertNum != 1,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "shared_expert_num",
                                                              std::to_string(sharedExpertNum).c_str(),
                                                              "expected value is 1 for V1"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK((sharedExpertNum < MIN_SHARED_EXPERT_NUM_V2) || (sharedExpertNum > MAX_SHARED_EXPERT_NUM_V2),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "shared_expert_num",
                                                              std::to_string(sharedExpertNum).c_str(),
                                                              "valid range is [0, 4] for V2"),
                        return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK((expertTokenNumsType != 0) && (expertTokenNumsType != 1),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "expert_token_nums_type",
                                                          std::to_string(expertTokenNumsType).c_str(),
                                                          "expected value is 0 or 1"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckSharedExpertAttrs(const char *nodeName, const int64_t expertShard,
                                              const int64_t sharedExpertRankNum, const int64_t epWorldSize)
{
    OP_TILING_CHECK(expertShard != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "expert_shard_type",
                                                          std::to_string(expertShard).c_str(), "expected value is 0"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((sharedExpertRankNum < 0) || (sharedExpertRankNum >= epWorldSize),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "shared_expert_rank_num", std::to_string(sharedExpertRankNum).c_str(),
                        ("valid range is [0, " + std::to_string(epWorldSize) + ")").c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckMoeExpertAttrs(const char *nodeName, const int64_t moeExpertNum)
{
    if (OpVersionManager::GetInstance().GetVersion() == OP_VERSION_1) {
        OP_TILING_CHECK((moeExpertNum <= 0) || (moeExpertNum > MAX_MOE_EXPERT_NUM_V1),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "moe_expert_num",
                                                              std::to_string(moeExpertNum).c_str(),
                                                              "valid range is (0, 512] for V1"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK((moeExpertNum <= 0) || (moeExpertNum > MAX_MOE_EXPERT_NUM_V2),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "moe_expert_num",
                                                              std::to_string(moeExpertNum).c_str(),
                                                              "valid range is (0, 1024] for V2"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckOutputDataType(const gert::TilingContext *context, const char *nodeName,
                                           const int64_t quantMode)
{
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expand_x"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (quantMode == static_cast<int64_t>(QuantModeA5::NON_QUANT)) &&
            (NON_QUANT_DTYPE.find(static_cast<ge::DataType>(expandXDesc->GetDataType())) == NON_QUANT_DTYPE.end()),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expand_x", Ops::Base::ToString(expandXDesc->GetDataType()).c_str(),
                                  "bf16/fp16/hif8/fp8_e4m3fn/fp8_e5m2"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((quantMode == static_cast<int64_t>(QuantModeA5::STATIC_QUANT)) &&
                        (expandXDesc->GetDataType() != ge::DT_HIFLOAT8) && (expandXDesc->GetDataType() != ge::DT_INT8),
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expand_x",
                                              Ops::Base::ToString(expandXDesc->GetDataType()).c_str(), "int8/hif8"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (quantMode == static_cast<int64_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) &&
            (expandXDesc->GetDataType() != ge::DT_INT8) && (expandXDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN) &&
            (expandXDesc->GetDataType() != ge::DT_FLOAT8_E5M2),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expand_x", Ops::Base::ToString(expandXDesc->GetDataType()).c_str(),
                                  "int8/fp8_e4m3fn/fp8_e5m2"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        ((quantMode == static_cast<int64_t>(QuantModeA5::PERGROUP_DYNAMIC_QUANT)) ||
         (quantMode == static_cast<int64_t>(QuantModeA5::MX_QUANT))) &&
            (expandXDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN) && (expandXDesc->GetDataType() != ge::DT_FLOAT8_E5M2),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "expand_x", Ops::Base::ToString(expandXDesc->GetDataType()).c_str(),
                                  "fp8_e4m3fn/fp8_e5m2"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline void SetSharedExpertNum(MoeDistributeDispatchV2TilingData &tilingData, const uint32_t sharedExpertRankNum,
                               const uint32_t sharedExpertNum)
{
    if ((sharedExpertRankNum == NUM_0) && (sharedExpertNum == NUM_1)) {
        tilingData.moeDistributeDispatchV2Info.sharedExpertNum = NUM_0;
    } else {
        tilingData.moeDistributeDispatchV2Info.sharedExpertNum = static_cast<uint32_t>(sharedExpertNum);
    }
}

inline void SetEpTpInfo(MoeDistributeDispatchV2TilingData &tilingData, const uint32_t epWorldSize,
                        const uint32_t tpWorldSize, const uint32_t epRankId, const uint32_t tpRankId)
{
    tilingData.moeDistributeDispatchV2Info.epWorldSize = epWorldSize;
    tilingData.moeDistributeDispatchV2Info.tpWorldSize = tpWorldSize;
    tilingData.moeDistributeDispatchV2Info.epRankId = epRankId;
    tilingData.moeDistributeDispatchV2Info.tpRankId = tpRankId;
}

inline void SetExpertInfo(MoeDistributeDispatchV2TilingData &tilingData, const uint32_t expertShard,
                          const uint32_t sharedExpertRankNum, const uint32_t moeExpertNum)
{
    tilingData.moeDistributeDispatchV2Info.expertShardType = expertShard;
    tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum = sharedExpertRankNum;
    tilingData.moeDistributeDispatchV2Info.moeExpertNum = moeExpertNum;
}

static ge::graphStatus GetContextAttrs(const gert::TilingContext *context, const char *nodeName,
                                       MoeDistributeDispatchV2TilingData &tilingData, std::string &groupEp,
                                       std::string &groupTp)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto groupTpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_TP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_RANK_ID_INDEX);
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_SHARED_EXPERT_NUM_INDEX));
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX));

    // 判空
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "ep_world_size"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tp_world_size"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "ep_rank_id"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tp_rank_id"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertShardPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expert_shard_type"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "shared_expert_rank_num"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "moe_expert_num"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(quantModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "quant_mode"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "shared_expert_num"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertTokenNumsTypePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expert_token_nums_type"),
                    return ge::GRAPH_FAILED);

    // 判断是否满足uint32_t及其他限制
    OP_TILING_CHECK(
        (CheckEpAttrs(nodeName, *epWorldSizePtr, *epRankIdPtr, groupEpPtr, groupEp) != ge::GRAPH_SUCCESS) ||
            (CheckTpAttrs(nodeName, *tpWorldSizePtr, *tpRankIdPtr, groupTpPtr, groupTp) != ge::GRAPH_SUCCESS),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "ep_attrs,tp_attrs", "invalid", "see detailed error above"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((CheckSharedExpertAttrs(nodeName, *expertShardPtr, *sharedExpertRankNumPtr, *epWorldSizePtr) !=
                     ge::GRAPH_SUCCESS) ||
                        (CheckMoeExpertAttrs(nodeName, *moeExpertNumPtr) != ge::GRAPH_SUCCESS) ||
                        (CheckQuantAndExpertAttrs(nodeName, *sharedExpertNumPtr, *quantModePtr,
                                                  *expertTokenNumsTypePtr) != ge::GRAPH_SUCCESS),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "quant_attrs,expert_attrs", "invalid",
                                                          "see detailed error above"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOutputDataType(context, nodeName, *quantModePtr) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "output", "invalid", "see detailed error above"),
                    return ge::GRAPH_FAILED);
    SetEpTpInfo(tilingData, static_cast<uint32_t>(*epWorldSizePtr), static_cast<uint32_t>(*tpWorldSizePtr),
                static_cast<uint32_t>(*epRankIdPtr), static_cast<uint32_t>(*tpRankIdPtr));
    SetExpertInfo(tilingData, static_cast<uint32_t>(*expertShardPtr), static_cast<uint32_t>(*sharedExpertRankNumPtr),
                  static_cast<uint32_t>(*moeExpertNumPtr));
    tilingData.moeDistributeDispatchV2Info.quantMode = static_cast<uint32_t>(*quantModePtr);
    tilingData.moeDistributeDispatchV2Info.expertTokenNumsType = static_cast<uint32_t>(*expertTokenNumsTypePtr);
    SetSharedExpertNum(tilingData, static_cast<uint32_t>(*sharedExpertRankNumPtr),
                       static_cast<uint32_t>(*sharedExpertNumPtr));
    return ge::GRAPH_SUCCESS;
}

inline uint32_t CheckQuantModeAndExpandXType(const gert::TilingContext *context, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    QuantModeA5 quantMode = static_cast<QuantModeA5>(*quantModePtr);
    auto modeToFind = QUANT_MODE_MAP.find({quantMode, static_cast<ge::DataType>(expandXDesc->GetDataType())});
    OP_TILING_CHECK(modeToFind == QUANT_MODE_MAP.end(),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "quant_mode,expand_x",
                                                          (std::to_string(static_cast<uint32_t>(quantMode)) + "," +
                                                           Ops::Base::ToString(expandXDesc->GetDataType()))
                                                              .c_str(),
                                                          "Failed to find real mode"),
                    return static_cast<uint32_t>(RealModeA5::INVALID_MODE));
    OP_LOGD(nodeName, "quantMode=%u, expandX datatype=%s, get realMode=%u\n", static_cast<uint32_t>(quantMode),
            Ops::Base::ToString(expandXDesc->GetDataType()).c_str(), static_cast<uint32_t>(modeToFind->second));
    return static_cast<uint32_t>(modeToFind->second);
}

static ge::graphStatus CheckQuantModeAndScales(const gert::TilingContext *context, const char *nodeName, bool isScales,
                                               const uint32_t quantMode)
{
    OP_TILING_CHECK(isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "scales", "provided",
                        ("scales should be nullptr when quantMode is " + std::to_string(quantMode)).c_str()),
                    return ge::GRAPH_FAILED);
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) &&
                        ((xDesc->GetDataType() == ge::DT_BF16) || (xDesc->GetDataType() == ge::DT_FLOAT16)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "scales", "provided",
                                                          ("scales should be nullptr when quantMode is " +
                                                           std::to_string(quantMode) + " and X datatype is " +
                                                           Ops::Base::ToString(xDesc->GetDataType()))
                                                              .c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) &&
                        ((xDesc->GetDataType() == ge::DT_HIFLOAT8) || (xDesc->GetDataType() == ge::DT_FLOAT8_E5M2) ||
                         (xDesc->GetDataType() == ge::DT_FLOAT8_E4M3FN)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "scales", "nullptr",
                                                          ("scales should not be nullptr when quantMode is " +
                                                           std::to_string(quantMode) + " and X datatype is " +
                                                           Ops::Base::ToString(xDesc->GetDataType()))
                                                              .c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "scales", "nullptr",
                        ("scales should not be nullptr when quantMode is " + std::to_string(quantMode)).c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckEpWorldSizeV1(const char *nodeName, uint32_t epWorldSize, uint32_t sharedExpertRankNum)
{
    // 校验ep能否均分共享专家
    OP_TILING_CHECK((sharedExpertRankNum != 0) && (epWorldSize % sharedExpertRankNum != 0),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "ep_world_size,shared_expert_rank_num",
                        (std::to_string(epWorldSize) + "," + std::to_string(sharedExpertRankNum)).c_str(),
                        "epWorldSize should be divisible by sharedExpertRankNum"),
                    return ge::GRAPH_FAILED);

    // Only the value 4 or 2 is supported currently
    if ((epWorldSize == EP_WORLD_SIZE_FOUR) || (epWorldSize == EP_WORLD_SIZE_TWO)) {
        OP_LOGD(nodeName, "epWorldSize=%u, skip validation\n", epWorldSize);
    } else {
        // 检验epWorldSize是否是8的倍数
        OP_TILING_CHECK(epWorldSize % 8 != 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            nodeName, "ep_world_size", std::to_string(epWorldSize).c_str(), "must be a multiple of 8"),
                        return ge::GRAPH_FAILED);

        OP_TILING_CHECK(
            (256 % epWorldSize != 0) && (epWorldSize % 144 != 0),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "ep_world_size", std::to_string(epWorldSize).c_str(),
                                                  "must be in the list [8, 16, 32, 64, 128, 144, 256, 288]"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckSharedAttrs(const char *nodeName, MoeDistributeDispatchV2TilingData &tilingData)
{
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;

    // 校验共享专家卡数和共享专家数是否只有一个为0
    OP_TILING_CHECK(((sharedExpertNum == 0U) && (sharedExpertRankNum > 0U)) ||
                        ((sharedExpertNum > 0U) && (sharedExpertRankNum == 0U)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "shared_expert_rank_num,shared_expert_num",
                        (std::to_string(sharedExpertRankNum) + "," + std::to_string(sharedExpertNum)).c_str(),
                        "shared_expert_rank_num and shared_expert_num must both be zero or both non-zero"),
                    return ge::GRAPH_FAILED);

    if ((sharedExpertNum > 0U) && (sharedExpertRankNum > 0U)) {
        OP_TILING_CHECK(((sharedExpertRankNum % sharedExpertNum) != 0U),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            nodeName, "shared_expert_rank_num,shared_expert_num",
                            (std::to_string(sharedExpertRankNum) + "," + std::to_string(sharedExpertNum)).c_str(),
                            "shared_expert_rank_num should be divisible by shared_expert_num"),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrs(const gert::TilingContext *context, MoeDistributeDispatchV2TilingData &tilingData,
                                  uint32_t &localMoeExpertNum, const bool isTokenMask)
{
    // nodeName已在调用处判空
    const char *nodeName = context->GetNodeName();
    uint32_t epWorldSize = tilingData.moeDistributeDispatchV2Info.epWorldSize;
    uint32_t tpWorldSize = tilingData.moeDistributeDispatchV2Info.tpWorldSize;
    uint32_t moeExpertNum = tilingData.moeDistributeDispatchV2Info.moeExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;

    if (OpVersionManager::GetInstance().GetVersion() == OP_VERSION_1) {
        OP_TILING_CHECK(CheckEpWorldSizeV1(nodeName, epWorldSize, sharedExpertRankNum) != ge::GRAPH_SUCCESS,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            nodeName, "ep_world_size", std::to_string(epWorldSize).c_str(), "see detailed error above"),
                        return ge::GRAPH_FAILED);
    }

    localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    OP_TILING_CHECK(moeExpertNum % (epWorldSize - sharedExpertRankNum) != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "moe_expert_num,ep_world_size,shared_expert_rank_num",
                        (std::to_string(moeExpertNum) + "," + std::to_string(epWorldSize) + "," +
                         std::to_string(sharedExpertRankNum))
                            .c_str(),
                        "moe_expert_num should be divisible by (ep_world_size - shared_expert_rank_num)"),
                    return ge::GRAPH_FAILED);
}

// 校验moe专家数量能否均分给多机
localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
OP_TILING_CHECK(moeExpertNum % (epWorldSize - sharedExpertRankNum) != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    nodeName, "moe_expert_num,ep_world_size,shared_expert_rank_num",
                    (std::to_string(moeExpertNum) + "," + std::to_string(epWorldSize) + "," +
                     std::to_string(sharedExpertRankNum))
                        .c_str(),
                    "moe_expert_num should be divisible by (ep_world_size - shared_expert_rank_num)"),
                return ge::GRAPH_FAILED);
OP_TILING_CHECK(localMoeExpertNum <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "local_moe_expert_num",
                                                      std::to_string(localMoeExpertNum).c_str(),
                                                      "must be greater than 0"),
                return ge::GRAPH_FAILED);
// tpWorldSize 当前仅支持1
OP_TILING_CHECK(tpWorldSize != 1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "tp_world_size", std::to_string(tpWorldSize).c_str(),
                                                      "must be 1 in current version"),
                return ge::GRAPH_FAILED);

OP_TILING_CHECK(CheckSharedAttrs(nodeName, tilingData) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "shared_attrs", "invalid", "see detailed error above"),
                return ge::GRAPH_FAILED);

// 校验输入x的dim 0并设bs
const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x"), return ge::GRAPH_FAILED);
const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
OP_TILING_CHECK((xDim0 > BS_UPPER_BOUND) || (xDim0 <= 0),
                OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "x", std::to_string(xDim0).c_str(),
                                             "dim0 should be in range [1, 512]"),
                return ge::GRAPH_FAILED);
tilingData.moeDistributeDispatchV2Info.bs = static_cast<uint32_t>(xDim0);

// 校验globalBS
auto attrs = context->GetAttrs();
OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);
auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "global_bs"), return ge::GRAPH_FAILED);
OP_LOGD(nodeName, "MoeDistributeDispatch *globalBsPtr=%ld, bs=%ld, epWorldSize=%u\n", *globalBsPtr, xDim0, epWorldSize);
OP_TILING_CHECK((*globalBsPtr != 0) && ((*globalBsPtr < xDim0 * static_cast<int64_t>(epWorldSize)) ||
                                        ((*globalBsPtr) % (static_cast<int64_t>(epWorldSize)) != 0)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "global_bs", std::to_string(*globalBsPtr).c_str(),
                                                      ("invalid, only support 0 or maxBs * ep_world_size, bs=" +
                                                       std::to_string(xDim0) +
                                                       ", ep_world_size=" + std::to_string(epWorldSize))
                                                          .c_str()),
                return ge::GRAPH_FAILED);
OP_TILING_CHECK(((*globalBsPtr > (xDim0 * static_cast<int64_t>(epWorldSize))) && isTokenMask),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    nodeName, "global_bs", std::to_string(*globalBsPtr).c_str(),
                    ("different bs on different rank cannot work when isActiveMask=true, bs=" + std::to_string(xDim0) +
                     ", ep_world_size=" + std::to_string(epWorldSize))
                        .c_str()),
                return ge::GRAPH_FAILED);
if (*globalBsPtr == 0) {
    tilingData.moeDistributeDispatchV2Info.globalBs = static_cast<uint32_t>(xDim0) * epWorldSize;
} else {
    tilingData.moeDistributeDispatchV2Info.globalBs = static_cast<uint32_t>(*globalBsPtr);
}

return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckTwoDimScalesShape(const gert::TilingContext *context, const char *nodeName,
                                              const MoeDistributeDispatchV2TilingData &tilingData,
                                              const int64_t scalesDim0, const int64_t scalesDim1)
{
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x"), return ge::GRAPH_FAILED);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    if (sharedExpertRankNum == 0U) {
        OP_TILING_CHECK(
            scalesDim0 != moeExpertNum,
            OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "scales", std::to_string(scalesDim0).c_str(),
                                         ("dim0 should equal moe_expert_num=" + std::to_string(moeExpertNum)).c_str()),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(scalesDim0 != (moeExpertNum + sharedExpertNum),
                        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "scales", std::to_string(scalesDim0).c_str(),
                                                     ("dim0 should equal moe_expert_num + shared_expert_num=" +
                                                      std::to_string(moeExpertNum + sharedExpertNum))
                                                         .c_str()),
                        return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(xDim1 != scalesDim1,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "scales", std::to_string(scalesDim1).c_str(),
                                                 ("dim1 should equal x's dim1=" + std::to_string(xDim1)).c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckAndSetScalesInfo(const gert::TilingContext *context, const char *nodeName,
                                             MoeDistributeDispatchV2TilingData &tilingData, bool isScales,
                                             const uint32_t quantMode)
{
    // 校验scales的维度
    // bs and h have been set in CheckAttrs
    uint32_t h = tilingData.moeDistributeDispatchV2Info.h;
    uint32_t bs = tilingData.moeDistributeDispatchV2Info.bs;
    uint64_t scalesRow = 0;
    uint64_t scalesCol = 0;
    uint32_t scalesTypeSize = 0;
    uint64_t scalesCount = 0;
    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(SCALES_INDEX);
        const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
        OP_TILING_CHECK(scalesStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "scales"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "scales"), return ge::GRAPH_FAILED);
        size_t scalesDimNum = scalesStorageShape->GetStorageShape().GetDimNum();
        const int64_t scalesDim0 = scalesStorageShape->GetStorageShape().GetDim(0);
        scalesRow = static_cast<uint64_t>(scalesDim0);
        scalesTypeSize = ge::GetSizeByDataType(scalesDesc->GetDataType());
        if (scalesDimNum == ONE_DIM) {
            // realMode 1 or 9
            OP_TILING_CHECK(
                (quantMode == static_cast<uint32_t>(RealModeA5::STATIC_SCALES)) && (scalesDim0 != h) &&
                    (scalesDim0 != STATIC_SCALE_DIM_0),
                OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "scales", std::to_string(scalesDim0).c_str(),
                                             ("dim0 should be " + std::to_string(h) + " or 1 in static quant").c_str()),
                return ge::GRAPH_FAILED);
            OP_TILING_CHECK(
                (quantMode == static_cast<uint32_t>(RealModeA5::HIF8_SCALES)) && (scalesDim0 != HIF8_SCALE_DIM_0),
                OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "scales", std::to_string(scalesDim0).c_str(),
                                             "dim0 should be 1 when expandX datatype is hif8 in static quant"),
                return ge::GRAPH_FAILED);
            scalesCol = ONE_DIM_SCALE_COL_NUM;
            scalesCount = static_cast<uint64_t>(scalesDim0);
        } else if (quantMode == static_cast<uint32_t>(RealModeA5::NO_SCALES)) {
            OP_TILING_CHECK(
                scalesDim0 != bs,
                OP_LOGE_FOR_INVALID_SHAPEDIM(
                    nodeName, "scales", std::to_string(scalesDim0).c_str(),
                    ("dim0 should be " + std::to_string(bs) + " when scales is not null in non-quant").c_str()),
                return ge::GRAPH_FAILED);
        } else {
            const int64_t scalesDim1 = scalesStorageShape->GetStorageShape().GetDim(1);
            OP_TILING_CHECK(CheckTwoDimScalesShape(context, nodeName, tilingData, scalesDim0, scalesDim1) !=
                                ge::GRAPH_SUCCESS,
                            OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "scales", "invalid",
                                                         "CheckTwoDimScalesShape failed, see detailed error above"),
                            return ge::GRAPH_FAILED);
            scalesCol = static_cast<uint64_t>(scalesDim1);
            scalesCount = static_cast<uint64_t>(scalesDim0 * scalesDim1);
        }
    }
    tilingData.moeDistributeDispatchV2Info.scalesRow = scalesRow;
    tilingData.moeDistributeDispatchV2Info.scalesCol = scalesCol;
    tilingData.moeDistributeDispatchV2Info.scalesCount = scalesCount;
    tilingData.moeDistributeDispatchV2Info.scalesTypeSize = scalesTypeSize;
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckExpandXShape(const gert::TilingContext *context, const char *nodeName,
                                         const MoeDistributeDispatchV2TilingData &tilingData, uint32_t A)
{
    // H has been set in CheckAndSetH
    const uint32_t xDim1 = tilingData.moeDistributeDispatchV2Info.h;
    // 校验expandX的维度
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.tpWorldSize);
    const gert::StorageShape *expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    const int64_t expandXDim0 = expandXStorageShape->GetStorageShape().GetDim(0);
    const int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(
        expandXDim0 < tpWorldSize * static_cast<int64_t>(A),
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "expand_x", std::to_string(expandXDim0).c_str(),
                                     ("dim0 should be >= A*tp_world_size=" + std::to_string(tpWorldSize * A)).c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expandXDim1 != static_cast<int64_t>(xDim1),
                    OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "expand_x", std::to_string(expandXDim1).c_str(),
                                                 ("dim1 should equal x's dim1=" + std::to_string(xDim1)).c_str()),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckDynamicScalesShape(const gert::TilingContext *context, const char *nodeName,
                                               const MoeDistributeDispatchV2TilingData &tilingData,
                                               const uint32_t quantMode, uint32_t A)
{
    // 校验dynamicScales的维度
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.tpWorldSize);
    uint64_t h = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.h);
    if ((quantMode != static_cast<uint32_t>(QuantModeA5::NON_QUANT)) &&
        (quantMode != static_cast<uint32_t>(QuantModeA5::STATIC_QUANT))) {
        // Dim0
        const gert::StorageShape *dynamicScalesStorageShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
        const int64_t dynamicScalesDim0 = dynamicScalesStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(dynamicScalesDim0 < static_cast<int64_t>(A) * tpWorldSize,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(
                            nodeName, "dynamic_scales", std::to_string(dynamicScalesDim0).c_str(),
                            ("dim0 should be >= A*tp_world_size=" + std::to_string(A * tpWorldSize)).c_str()),
                        return ge::GRAPH_FAILED);
        // Dim1, only for pergroup and mx
        if (quantMode != static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
            const uint64_t dynamicScalesDim1 =
                static_cast<uint64_t>(dynamicScalesStorageShape->GetStorageShape().GetDim(1));
            OP_TILING_CHECK(
                (quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT)) &&
                    (dynamicScalesDim1 !=
                     ops::CeilAlign(static_cast<uint64_t>(ops::CeilDiv(h, MX_BLOCK_SIZE)), EVEN_ALIGN)),
                OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "dynamic_scales", std::to_string(dynamicScalesDim1).c_str(),
                                             ("dim1 should be " + std::to_string(ops::CeilDiv(h, MX_BLOCK_SIZE)) +
                                              " and even when quantMode=" + std::to_string(quantMode))
                                                 .c_str()),
                return ge::GRAPH_FAILED);
            OP_TILING_CHECK(
                (dynamicScalesDim1 != ops::CeilDiv(h, PERGROUP_BLOCK_SIZE)) &&
                    (quantMode == static_cast<uint32_t>(QuantModeA5::PERGROUP_DYNAMIC_QUANT)),
                OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "dynamic_scales", std::to_string(dynamicScalesDim1).c_str(),
                                             ("dim1 should be " + std::to_string(ops::CeilDiv(h, PERGROUP_BLOCK_SIZE)) +
                                              " when quantMode=" + std::to_string(quantMode))
                                                 .c_str()),
                return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckExpandIdxAndMaskShape(const gert::TilingContext *context, const char *nodeName,
                                                  const int64_t xDim0, const int64_t expertIdsDim1)
{
    // 校验expandIdx的维度
    const gert::StorageShape *expandIdxStorageShape = context->GetOutputShape(OUTPUT_EXPAND_IDX_INDEX);
    const int64_t expandIdxDim0 = expandIdxStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(
        expandIdxDim0 < expertIdsDim1 * xDim0,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "expand_idx", std::to_string(expandIdxDim0).c_str(),
                                     ("dim0 should be >= bs * k = " + std::to_string(xDim0 * expertIdsDim1)).c_str()),
        return ge::GRAPH_FAILED);
    // 校验xActiveMask的维度
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    if ((OpVersionManager::GetInstance().GetVersion() != OP_VERSION_1) && (xActiveMaskStorageShape != nullptr)) {
        const int64_t xActiveMaskDim0 = xActiveMaskStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(xActiveMaskDim0 != xDim0,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "x_active_mask", std::to_string(xActiveMaskDim0).c_str(),
                                                     ("dim0 should be equal to bs=" + std::to_string(xDim0)).c_str()),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckExpertTokenNumsShape(const gert::TilingContext *context, const char *nodeName,
                                                 const bool isSharedExpert, const int64_t localMoeExpertNum)
{
    // 校验expertTokenNums的维度
    const gert::StorageShape *expertTokenNumsStorageShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    const int64_t expertTokenNumsDim0 = expertTokenNumsStorageShape->GetStorageShape().GetDim(0);
    if (isSharedExpert) {
        OP_TILING_CHECK(expertTokenNumsDim0 != 1,
                        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "expert_token_nums",
                                                     std::to_string(expertTokenNumsDim0).c_str(),
                                                     "dim0 should be 1 for shared expert"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(
            expertTokenNumsDim0 != localMoeExpertNum,
            OP_LOGE_FOR_INVALID_SHAPEDIM(
                nodeName, "expert_token_nums", std::to_string(expertTokenNumsDim0).c_str(),
                ("dim0 should be equal to local_moe_expert_num=" + std::to_string(localMoeExpertNum)).c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckEpTpTecvTensorShape(const gert::TilingContext *context, const char *nodeName,
                                                const MoeDistributeDispatchV2TilingData &tilingData,
                                                const bool isSharedExpert, const int64_t localMoeExpertNum)
{
    // 校验epRecvCount和tpRecvCount的维度
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.tpWorldSize);
    int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);
    const gert::StorageShape *epRecvCountStorageShape = context->GetOutputShape(OUTPUT_EP_RECV_COUNTS_INDEX);
    const gert::StorageShape *tpRecvCountStorageShape = context->GetOutputShape(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "ep_recv_counts"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRecvCountStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tp_recv_counts"),
                    return ge::GRAPH_FAILED);
    const int64_t epRecvCountDim0 = epRecvCountStorageShape->GetStorageShape().GetDim(0);
    const int64_t tpRecvCountDim0 = tpRecvCountStorageShape->GetStorageShape().GetDim(0);
    int64_t epRecvCount = (isSharedExpert) ? epWorldSize : epWorldSize * localMoeExpertNum;
    if (tpWorldSize == MAX_TP_WORLD_SIZE) {
        epRecvCount *= tpWorldSize;
    }
    OP_TILING_CHECK(
        epRecvCountDim0 < epRecvCount,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            nodeName, "ep_recv_counts", std::to_string(epRecvCountDim0).c_str(),
            ("dim0 should be >= ep_world_size * local_moe_expert_num * tp_world_size=" + std::to_string(epRecvCount))
                .c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        tpRecvCountDim0 != tpWorldSize,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "tp_recv_counts", std::to_string(tpRecvCountDim0).c_str(),
                                     ("dim0 should be equal to tp_world_size=" + std::to_string(tpWorldSize)).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetH(const char *nodeName, MoeDistributeDispatchV2TilingData &tilingData,
                                    const int64_t xDim1)
{
    // 校验输入x的维度1并设h, bs已校验过
    if (OpVersionManager::GetInstance().GetVersion() == OP_VERSION_1) {
        OP_TILING_CHECK((xDim1 != H_V1),
                        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "x", std::to_string(xDim1).c_str(),
                                                     "dim1(H) only supports 7168 for V1"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK((xDim1 < H_LOWER_BOUND_V2) || (xDim1 > H_UPPER_BOUND_V2),
                        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "x", std::to_string(xDim1).c_str(),
                                                     "dim1(H) should be in range [1024, 8192] for V2"),
                        return ge::GRAPH_FAILED);
    }
    tilingData.moeDistributeDispatchV2Info.h = static_cast<uint32_t>(xDim1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckExpertIdsAndSetK(const gert::TilingContext *context, const char *nodeName,
                                             MoeDistributeDispatchV2TilingData &tilingData, const int64_t xDim0)
{
    // 校验expert_id的维度并设k
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "expert_ids"),
                    return ge::GRAPH_FAILED);
    const int64_t expertIdsDim0 = expertIdStorageShape->GetStorageShape().GetDim(0);
    const int64_t expertIdsDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    const int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    OP_TILING_CHECK(xDim0 != expertIdsDim0,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "expert_ids", std::to_string(expertIdsDim0).c_str(),
                                                 ("dim0 should equal x's dim0=" + std::to_string(xDim0)).c_str()),
                    return ge::GRAPH_FAILED);
    const int64_t maxK =
        (OpVersionManager::GetInstance().GetVersion() == OP_VERSION_1) ? K_UPPER_BOUND_V1 : K_UPPER_BOUND_V2;
    OP_TILING_CHECK((expertIdsDim1 <= 0) || (expertIdsDim1 > maxK) || (expertIdsDim1 > moeExpertNum),
                    OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "expert_ids", std::to_string(expertIdsDim1).c_str(),
                                                 ("dim1(k) should be in (0, min(" + std::to_string(maxK) +
                                                  ", moe_expert_num=" + std::to_string(moeExpertNum) + "))")
                                                     .c_str()),
                    return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchV2Info.k = static_cast<uint32_t>(expertIdsDim1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorShape(const gert::TilingContext *context,
                                        MoeDistributeDispatchV2TilingData &tilingData, const bool isSharedExpert,
                                        const int64_t localMoeExpertNum)
{
    // nodeName已在调用处判空
    const char *nodeName = context->GetNodeName();
    uint32_t quantMode = tilingData.moeDistributeDispatchV2Info.quantMode;
    uint32_t A = 0;
    uint32_t globalBs = tilingData.moeDistributeDispatchV2Info.globalBs;
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x"), return ge::GRAPH_FAILED);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(
        CheckAndSetH(nodeName, tilingData, xDim1) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "x", "invalid", "CheckAndSetH failed, see detailed error above"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckExpertIdsAndSetK(context, nodeName, tilingData, xDim0) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "expert_ids", "invalid",
                                                 "CheckExpertIdsAndSetK failed, see detailed error above"),
                    return ge::GRAPH_FAILED);
    int64_t k = tilingData.moeDistributeDispatchV2Info.k;
    if (isSharedExpert) {
        if (OpVersionManager::GetInstance().GetVersion() == OP_VERSION_1) {
            A = globalBs / sharedExpertRankNum;
        } else {
            uint32_t rankNumPerSharedExpert = 0;
            uint32_t epWorldSizeU32 = tilingData.moeDistributeDispatchV2Info.epWorldSize;
            uint32_t maxBs = globalBs / epWorldSizeU32;
            uint32_t maxSharedGroupNum = 0;
            if ((sharedExpertNum != 0U) && (sharedExpertRankNum != 0U)) {
                rankNumPerSharedExpert = sharedExpertRankNum / sharedExpertNum;
                maxSharedGroupNum = (epWorldSizeU32 + rankNumPerSharedExpert - 1U) / rankNumPerSharedExpert;
            }
            A = maxBs * maxSharedGroupNum;
        }
    } else {
        A = globalBs * std::min(localMoeExpertNum, k);
    }
    OP_TILING_CHECK(
        CheckExpandXShape(context, nodeName, tilingData, A) != ge::GRAPH_SUCCESS ||
            CheckDynamicScalesShape(context, nodeName, tilingData, quantMode, A) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "expand_x,dynamic_scales", "invalid", "see detailed error above"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckExpandIdxAndMaskShape(context, nodeName, xDim0, k) != ge::GRAPH_SUCCESS ||
            CheckExpertTokenNumsShape(context, nodeName, isSharedExpert, localMoeExpertNum) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "expand_idx,expert_token_nums", "invalid", "see detailed error above"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckEpTpTecvTensorShape(context, nodeName, tilingData, isSharedExpert, localMoeExpertNum) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "ep_recv_counts,tp_recv_counts", "invalid", "see detailed error above"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, MoeDistributeDispatchV2TilingData &tilingData,
                                    uint32_t localMoeExpertNum)
{
    const char *nodeName = context->GetNodeName();
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "workspace"), return ge::GRAPH_FAILED);
    uint64_t epWorldSize = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);
    uint64_t maxBs = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.globalBs) / epWorldSize;
    uint64_t alignCnt = UB_ALIGN;
    uint32_t quantMode = tilingData.moeDistributeDispatchV2Info.quantMode;
    if (quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT)) {
        alignCnt = MX_PAD_ALIGN;
    } else if (quantMode == static_cast<uint32_t>(QuantModeA5::PERGROUP_DYNAMIC_QUANT)) {
        alignCnt = PERGROUP_PAD_ALIGN;
    }
    uint64_t alignedH = ops::CeilAlign(static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.h), alignCnt);

    // Non-null check in CheckQuantModeAndScales
    auto xDesc = context->GetInputDesc(X_INDEX);
    workSpaces[0] =
        SYSTEM_NEED_WORKSPACE + epWorldSize * sizeof(uint64_t) * BUFFER_NUM * BUFFER_NUM +
        epWorldSize *
            (COUNT_OFFSET + maxBs * ops::CeilAlign(alignedH * GetSizeByDataType(xDesc->GetDataType()), COMM_ALIGN) *
                                localMoeExpertNum);
    // A5 实际物理分配为 HCCL_BUFFSIZE 的 2 倍
    tilingData.moeDistributeDispatchV2Info.totalWinSizeEp = mc2tiling::Mc2TilingUtils::GetMaxWindowSize() * 2UL;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetCommTiling(const gert::TilingContext *context, MoeDistributeDispatchV2TilingData &tilingData,
                                     std::string &groupEp, std::string &groupTp)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "MoeDistributeDispatchV2 groupEp = %s, groupTp = %s", groupEp.c_str(), groupTp.c_str());
    // Only HalfAllToAllV is set, as A5 does not support TP communication.
    // Setting op types other than HalfAllToAllV will result in an error.
    uint32_t opType = static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_HALFALLTOALLV);
    std::string algConfigStr = "AlltoAll=level0:fullmesh;level1:pairwise";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType, algConfigStr);
    mc2CcTilingConfig.SetCommEngine(mc2tiling::A5_CCU_ENGINE);
    mc2CcTilingConfig.GetTiling(tilingData.mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData.mc2CcTiling1);

    mc2CcTilingConfig.SetGroupName(groupTp);
    mc2CcTilingConfig.GetTiling(tilingData.mc2CcTiling2);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GenTilingKey(gert::TilingContext *context, uint32_t quantMode, bool isScales)
{
    bool tp = false;
    uint32_t tilingKeyQuantMode = quantMode;
    bool scaleMode = false;
    uint32_t fullMesh = TILINGKEY_NO_FULLMESH;
    uint32_t commMode = TILINGKEY_TPL_CCU;

    if (isScales) {
        scaleMode = true;
    }
    uint64_t tilingKey = GET_TPL_TILING_KEY(tp, tilingKeyQuantMode, scaleMode, fullMesh, commMode, TILINGKEY_TPL_A5);
    const char *nodeName = context->GetNodeName();
    // Only tpWorldSize 1 is supported currently
    OP_LOGD(nodeName, "tilingKey=%lu", tilingKey);
    OP_LOGD(nodeName, "tp=%d", tp);
    OP_LOGD(nodeName, "tilingKeyQuantMode=%u", tilingKeyQuantMode);
    OP_LOGD(nodeName, "scaleMode=%d", scaleMode);
    OP_LOGD(nodeName, "fullMesh=%u", fullMesh);
    OP_LOGD(nodeName, "commMode=%u", commMode);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckCommAttrs(const char *nodeName, const MoeDistributeDispatchV2TilingData &tilingData,
                                      uint32_t localMoeExpertNum)
{
    // A5 实际物理分配为 HCCL_BUFFSIZE 的 2 倍
    uint64_t maxWindowSize = mc2tiling::Mc2TilingUtils::GetMaxWindowSize() * 2UL;
    uint64_t alignCnt = UB_ALIGN;
    uint32_t quantMode = tilingData.moeDistributeDispatchV2Info.quantMode;
    uint32_t aivNum = tilingData.moeDistributeDispatchV2Info.aivNum;
    if (quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT)) {
        alignCnt = MX_PAD_ALIGN;
    } else if (quantMode == static_cast<uint32_t>(QuantModeA5::PERGROUP_DYNAMIC_QUANT)) {
        alignCnt = PERGROUP_PAD_ALIGN;
    }
    uint64_t h = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.h);
    uint64_t alignedH = ops::CeilAlign(h, alignCnt);
    uint64_t epWorldSize = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);
    uint64_t maxBs = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.globalBs) / epWorldSize;
    uint64_t actualSize = aivNum * STATUS_SIZE + 2UL * epWorldSize *
                                                     (maxBs * ops::CeilAlign(alignedH * 2UL, COMM_ALIGN) *
                                                          static_cast<uint64_t>(localMoeExpertNum) +
                                                      COUNT_OFFSET);
    if (actualSize > maxWindowSize) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            nodeName, "HCCL_BUFFSIZE", (std::to_string(actualSize / MB_SIZE + 1UL) + "MB").c_str(),
            ("too small, exceeds maxWindowSize=" + std::to_string(maxWindowSize / MB_SIZE) + "MB").c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline void SetPlatformInfo(gert::TilingContext *context, MoeDistributeDispatchV2TilingData &tilingData,
                            const char *nodeName)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t numBlocks = 1U;
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    tilingData.moeDistributeDispatchV2Info.totalUbSize = ubSize;
    tilingData.moeDistributeDispatchV2Info.aivNum = aivNum;
    OP_LOGD(nodeName, "numBlocks=%u, aivNum=%u, ubSize=%lu", numBlocks, aivNum, ubSize);
}

ge::graphStatus MoeDistributeDispatchTilingImpl(gert::TilingContext *context, uint32_t opVersion)
{
    OpVersionManager::SetVersion(opVersion);
    // Tiling implementation
    OP_TILING_CHECK(context == nullptr, OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "tiling_context"), return ge::GRAPH_FAILED);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "node_name"), return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "Start MoeDistributeDispatch tiling.");
    MoeDistributeDispatchV2TilingData *tilingData = context->GetTilingData<MoeDistributeDispatchV2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tiling_data"),
                    return ge::GRAPH_FAILED);
    std::string groupEp = "";
    std::string groupTp = "";
    uint32_t localMoeExpertNum = 1;
    // Attrs
    OP_TILING_CHECK(GetContextAttrs(context, nodeName, *tilingData, groupEp, groupTp) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "attrs", "invalid", "Get attr and set tiling data failed, see detailed error above"),
                    return ge::GRAPH_FAILED);
    uint32_t quantMode = tilingData->moeDistributeDispatchV2Info.quantMode;
    uint32_t realMode = CheckQuantModeAndExpandXType(context, nodeName);
    OP_TILING_CHECK(
        realMode == static_cast<uint32_t>(RealModeA5::INVALID_MODE),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "quant_mode,expand_x", "invalid",
                                              "CheckQuantModeAndExpandXType failed, see detailed error above"),
        return ge::GRAPH_FAILED);
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    bool isTokenMask = (xActiveMaskStorageShape != nullptr);
    tilingData->moeDistributeDispatchV2Info.isTokenMask = isTokenMask;
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    bool isScales = (scalesStorageShape != nullptr);
    OP_TILING_CHECK(CheckQuantModeAndScales(context, nodeName, isScales, quantMode) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "quant_mode,scales",
                        (std::to_string(quantMode) + "," + std::to_string(static_cast<int32_t>(isScales))).c_str(),
                        "quant_mode and scales not match"),
                    return ge::GRAPH_FAILED);
    // Output and Input
    OP_TILING_CHECK(MoeDistributeDispatchTilingHelper::TilingCheckMoeDistributeDispatchA5(
                        context, isScales, quantMode, isTokenMask) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "param", "invalid",
                                                          "Tiling check param failed, see detailed error above"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAttrs(context, *tilingData, localMoeExpertNum, isTokenMask) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "attrs", "invalid",
                                                          "Check attr failed, see detailed error above"),
                    return ge::GRAPH_FAILED);

    bool isSharedExpert = (tilingData->moeDistributeDispatchV2Info.epRankId >=
                           tilingData->moeDistributeDispatchV2Info.sharedExpertRankNum) ?
                              false :
                              true;

    OP_TILING_CHECK(CheckTensorShape(context, *tilingData, isSharedExpert, static_cast<int64_t>(localMoeExpertNum)) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "tensor", "invalid",
                                                 "Check tensor shape failed, see detailed error above"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAndSetScalesInfo(context, nodeName, *tilingData, isScales, realMode) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "scales", "invalid",
                                                          "Check scales info failed, see detailed error above"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckCommAttrs(nodeName, *tilingData, localMoeExpertNum) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "comm_attrs", "invalid",
                                                          "CheckCommAttrs failed, see detailed error above"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SetCommTiling(context, *tilingData, groupEp, groupTp) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "comm_tiling", "invalid",
                                                          "SetCommTiling failed, see detailed error above"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SetWorkSpace(context, *tilingData, localMoeExpertNum) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "workspace", "invalid",
                                                          "SetWorkSpace failed, see detailed error above"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(GenTilingKey(context, quantMode, isScales) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "tiling_key", "invalid",
                                                          "Fail to get tiling key, see detailed error above"),
                    return ge::GRAPH_FAILED);
    // Platform
    SetPlatformInfo(context, *tilingData, nodeName);
    PrintTilingDataInfo(nodeName, *tilingData);
    OP_LOGD(nodeName, "Finish MoeDistributeDispatch tiling.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchTilingA5::DoOpTiling()
{
    return MoeDistributeDispatchTilingImpl(context_, OP_VERSION_1);
}

uint64_t MoeDistributeDispatchTilingA5::GetTilingKey() const
{
    // TilingKey calculation is done in DoOptiling
    const uint64_t tilingKey = context_->GetTilingKey();
    const char *nodeName = context_->GetNodeName();
    OP_LOGD(nodeName, "MoeDistributeDispatchTilingA5 get tiling key %lu", tilingKey);
    return tilingKey;
}

bool MoeDistributeDispatchTilingA5::IsCapable()
{
    if (npuArch_ == NpuArch::DAV_3510) {
        const char *nodeName = context_->GetNodeName();
        OP_LOGD(nodeName, "Do MoeDistributeDispatchTilingA5 tiling.");
        return true;
    }
    return false;
}

static bool CheckTensorDataTypeStaticOrDynamic(const gert::TilingContext *context, const char *nodeName, bool isScales,
                                               DispatchV2Config &config)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
    OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dynamic_scales"), return false);
    OP_TILING_CHECK(
        (xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(), "bf16/fp16"),
        return false);
    // Scales: fp32, optional for dynamic/pertoken/pertile, required for static/hif8
    // isScales has been checked in CheckQuantModeAndScales
    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "scales"), return false);
        OP_TILING_CHECK((scalesDesc->GetDataType() != ge::DT_FLOAT),
                        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "scales",
                                                  Ops::Base::ToString(scalesDesc->GetDataType()).c_str(), "float"),
                        return false);
    }
    OP_TILING_CHECK(dynamicScalesDesc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "dynamic_scales",
                                              Ops::Base::ToString(dynamicScalesDesc->GetDataType()).c_str(), "float"),
                    return false);
    return true;
}

static bool CheckTensorDataTypeMxfp8(const gert::TilingContext *context, const char *nodeName, DispatchV2Config &config)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
    OP_TILING_CHECK(
        (xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(), "bf16/fp16"),
        return false);
    // No Scales input
    OP_TILING_CHECK(dynamicScalesDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "dynamic_scales",
                                              Ops::Base::ToString(dynamicScalesDesc->GetDataType()).c_str(), "e8m0"),
                    return false);
    return true;
}

static bool CheckTensorDataTypeNonQuant(const gert::TilingContext *context, const char *nodeName, const bool isScales,
                                        DispatchV2Config &config)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK((NON_QUANT_DTYPE.find(static_cast<ge::DataType>(xDesc->GetDataType())) == NON_QUANT_DTYPE.end()),
                    OP_LOGE_FOR_INVALID_DTYPE(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(),
                                              "bf16/fp16/e5m2/e4m3fn/hif8"),
                    return false);
    // ExpandX: the same as X
    OP_TILING_CHECK(
        (expandXDesc->GetDataType() != xDesc->GetDataType() && (expandXDesc->GetDataType() != ge::DT_HIFLOAT8)),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            nodeName, "expand_x", Ops::Base::ToString(expandXDesc->GetDataType()).c_str(),
            ("datatype should be equal to x datatype " + Ops::Base::ToString(xDesc->GetDataType())).c_str()),
        return false);
    // Scales: bf16/fp16: nullptr; hif8: fp32; e5m2/e4m3fn: float/e8m0
    // Dynamic scales: the same as scales, and no validations for bf16/fp16
    // If X is bf16/fp16, the scales must be nullptr, which is validated in CheckQuantModeAndScales
    // Hence the datatype of X must be e5m2/e4m3fn/hif8 when isScales is true
    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dynamic_scales"),
                        return false);
        OP_TILING_CHECK((xDesc->GetDataType() == ge::DT_HIFLOAT8) && (scalesDesc->GetDataType() != ge::DT_FLOAT),
                        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "scales",
                                                  Ops::Base::ToString(scalesDesc->GetDataType()).c_str(), "float"),
                        return false);
        OP_TILING_CHECK(
            ((xDesc->GetDataType() == ge::DT_FLOAT4_E2M1) || (xDesc->GetDataType() == ge::DT_FLOAT4_E1M2)) &&
                (scalesDesc->GetDataType() != ge::DT_FLOAT8_E8M0),
            OP_LOGE_FOR_INVALID_DTYPE(nodeName, "scales", Ops::Base::ToString(scalesDesc->GetDataType()).c_str(),
                                      "e8m0"),
            return false);
        OP_TILING_CHECK((scalesDesc->GetDataType() != ge::DT_FLOAT) &&
                            (scalesDesc->GetDataType() != ge::DT_FLOAT8_E8M0),
                        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "scales",
                                                  Ops::Base::ToString(scalesDesc->GetDataType()).c_str(), "float/e8m0"),
                        return false);
        OP_TILING_CHECK(
            dynamicScalesDesc->GetDataType() != scalesDesc->GetDataType(),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                nodeName, "dynamic_scales", Ops::Base::ToString(dynamicScalesDesc->GetDataType()).c_str(),
                ("datatype should be equal to scales datatype " + Ops::Base::ToString(scalesDesc->GetDataType()))
                    .c_str()),
            return false);
    }
    return true;
}

static bool CheckDistinctTensorDataType(const gert::TilingContext *context, const char *nodeName, const bool isScales,
                                        const uint32_t quantMode, DispatchV2Config &config)
{
    if (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT)) {
        OP_TILING_CHECK(!CheckTensorDataTypeNonQuant(context, nodeName, isScales, config),
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            nodeName, "tensor", "invalid",
                            "CheckTensorDataType for nonquant mode failed, see detailed error above"),
                        return false);
    } else if (quantMode == static_cast<uint32_t>(QuantModeA5::MX_QUANT)) {
        OP_TILING_CHECK(!CheckTensorDataTypeMxfp8(context, nodeName, config),
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            nodeName, "tensor", "invalid",
                            "CheckTensorDataType for mx quant mode failed, see detailed error above"),
                        return false);
    } else {
        OP_TILING_CHECK(
            !CheckTensorDataTypeStaticOrDynamic(context, nodeName, isScales, config),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "tensor", "invalid",
                                                  ("CheckTensorDataType for quant_mode=" + std::to_string(quantMode) +
                                                   " failed, see detailed error above")
                                                      .c_str()),
            return false);
    }
    return true;
}

static bool CheckQuantModeAndExpandXTypeStatic(const gert::TilingContext *context, const char *nodeName,
                                               DispatchV2Config &config)
{
    auto attrs = context->GetAttrs();
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(config.attrQuantModeIndex);
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    QuantModeA5 quantMode = static_cast<QuantModeA5>(*quantModePtr);
    auto modeToFind = QUANT_MODE_MAP.find({quantMode, static_cast<ge::DataType>(expandXDesc->GetDataType())});
    OP_TILING_CHECK(modeToFind == QUANT_MODE_MAP.end(),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "quant_mode,expand_x",
                                                          (std::to_string(static_cast<uint32_t>(quantMode)) + "," +
                                                           Ops::Base::ToString(expandXDesc->GetDataType()))
                                                              .c_str(),
                                                          "Failed to find real mode"),
                    return false);
    OP_LOGD(nodeName, "quantMode=%u, expandx datatype=%s, get realMode=%u\n", static_cast<uint32_t>(quantMode),
            Ops::Base::ToString(expandXDesc->GetDataType()).c_str(), static_cast<uint32_t>(modeToFind->second));
    return true;
}

bool MoeDistributeDispatchV2TilingFuncA5::CheckTensorDataType(const gert::TilingContext *context, const char *nodeName,
                                                              const bool isScales, const uint32_t quantMode,
                                                              const bool isActiveMask, const bool hasElasticInfo,
                                                              const bool isPerformance, DispatchV2Config &config)
{
    OP_TILING_CHECK(
        !CheckQuantModeAndExpandXTypeStatic(context, nodeName, config),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "quant_mode,expand_x", "invalid",
                                              "CheckQuantModeAndExpandXTypeStatic failed, see detailed error above"),
        return false);
    OP_TILING_CHECK(!CheckDistinctTensorDataType(context, nodeName, isScales, quantMode, config),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "tensor", "invalid", "CheckDistinctTensorDataType failed, see detailed error above"),
                    return false);
    OP_TILING_CHECK(
        !CheckCommomOtherInputTensorDataType(context, nodeName, isActiveMask, hasElasticInfo, isPerformance, config),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "input", "invalid",
                                              "CheckCommomOtherInputTensorDataType failed, see detailed error above"),
        return false);
    OP_TILING_CHECK(
        !CheckCommomOutputTensorDataType(context, nodeName),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "output", "invalid",
                                              "CheckCommomOutputTensorDataType failed, see detailed error above"),
        return false);

    return true;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncA5::MoeDistributeDispatchTilingFuncImpl(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    const char *nodeName = context->GetNodeName();
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_COMM_ALG_INDEX));
    OP_LOGD(nodeName, "Set 'commAlg' as '%s'", commAlgPtr);
    // 检查 commAlg 参数合法性校验
    bool isNullOrEmpty = (commAlgPtr == nullptr) || (std::strlen(commAlgPtr) == 0);
    bool isFullmeshV1 = (std::strcmp(commAlgPtr, "fullmesh_v1") == 0);
    bool isFullmeshV2 = (std::strcmp(commAlgPtr, "fullmesh_v2") == 0);
    bool isMte = isFullmeshV1 || isFullmeshV2;
    bool isCcu = std::strcmp(commAlgPtr, "ccu") == 0;
    OP_TILING_CHECK(!(isNullOrEmpty || isMte || isCcu),
                    OP_LOGE_FOR_INVALID_ATTR(nodeName, "comm_alg", (commAlgPtr != nullptr) ? commAlgPtr : "nullptr",
                                             "fullmesh_v1/fullmesh_v2/ccu"),
                    return ge::GRAPH_FAILED);
    if (isCcu) {
        // CCU 调用 A5 tiling 实现
        return MoeDistributeDispatchTilingImpl(context, OP_VERSION_2);
    }
    // 默认空指针和空字符走 MTE 方式
    if (isNullOrEmpty) {
        OP_LOGI(nodeName, "Parameter 'commAlg' is nullptr/empty, defaulting to 'fullmesh_v1'.");
    }
    DispatchV2Config config;
    config.xIndex = 0U;                       // 0: 根据dispatchV2算子原型标志位初始化groupEp索引
    config.expertIdsIndex = 1U;               // 1: 根据dispatchV2算子原型标志位初始化expertIds索引
    config.scalesIndex = 2U;                  // 2: 根据dispatchV2算子原型标志位初始化scales索引
    config.xActiveMaskIndex = 3U;             // 3: 根据dispatchV2算子原型标志位初始化xActiveMask索引
    config.expertScalesIndex = 4U;            // 4: 根据dispatchV2算子原型标志位初始化expertScales索引
    config.elasticInfoIndex = 5U;             // 5: 根据dispatchV2算子原型标志位初始化elasticInfo索引
    config.performanceInfoIndex = 6U;         // 6: 根据dispatchV2算子原型标志位初始化performanceInfo索引
    config.attrGroupEpIndex = 0;              // 0: 根据dispatchV2算子原型标志位初始化groupEp索引
    config.attrEpWorldSizeIndex = 1;          // 1: 根据dispatchV2算子原型标志位初始化epWorldSize索引
    config.attrEpRankIdIndex = 2;             // 2: 根据dispatchV2算子原型标志位初始化epRankId索引
    config.attrMoeExpertNumIndex = 3;         // 3: 根据dispatchV2算子原型标志位初始化moeExpertNum索引
    config.attrGroupTpIndex = 4;              // 4: 根据dispatchV2算子原型标志位初始化groupTp索引
    config.attrTpWorldSizeIndex = 5;          // 5: 根据dispatchV2算子原型标志位初始化tpWorldSize索引
    config.attrTpRankIdIndex = 6;             // 6: 根据dispatchV2算子原型标志位初始化tpRankId索引
    config.attrExpertSharedTypeIndex = 7;     // 7: 根据dispatchV2算子原型标志位初始化expertSharedType索引
    config.attrSharedExpertNumIndex = 8;      // 8: 根据dispatchV2算子原型标志位初始化sharedExpertNum索引
    config.attrSharedExpertRankNumIndex = 9;  // 9: 根据dispatchV2算子原型标志位初始化sharedExpertRankNum索引
    config.attrQuantModeIndex = 10;           // 10: 根据dispatchV2算子原型标志位初始化quantMode索引
    config.attrGlobalBsIndex = 11;            // 11: 根据dispatchV2算子原型标志位初始化globalBs索引
    config.attrExpertTokenNumsTypeIndex = 12; // 12: 根据dispatchV2算子原型标志位初始化expertTokenNumType索引
    config.attrCommAlgIndex = 13;             // 13: 根据dispatchV2算子原型标志位初始化commAlg索引
    config.attrZeroExpertNumIndex = 14;       // 14: 根据dispatchV2算子原型标志位初始化zeroExpertNumIndex索引
    config.attrCopyExpertNumIndex = 15;       // 15: 根据dispatchV2算子原型标志位初始化copyExpertNumIndex索引
    config.attrConstExpertNumIndex = 16;      // 16: 根据dispatchV2算子原型标志位初始化constExpertNumIndex索引
    config.isMc2Context = false;
    // MTE 调用 A3 tiling 实现
    return MoeDistributeDispatchA3TilingFuncImplPublic(context, config);
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncA5::MoeDistributeDispatchV2TilingFunc(gert::TilingContext *context)
{
    ge::graphStatus ret = MoeDistributeDispatchTilingFuncImpl(context);
    return ret;
}

uint64_t MoeDistributeDispatchV2TilingFuncA5::CalTilingKey(const bool isScales, const uint32_t quantMode,
                                                           const uint32_t tpWorldSize, const bool isSetFullMeshV2,
                                                           bool isLayered)
{
    uint32_t templateDispatch = TILINGKEY_NO_FULLMESH;
    bool tp = false;
    uint32_t tilingKeyQuantMode = quantMode;
    bool scaleMode = false;
    uint64_t tilingKey;
    uint32_t commMode = TILINGKEY_TPL_MTE;
    if (isScales) {
        scaleMode = true;
    }
    if (isSetFullMeshV2) {
        templateDispatch = TILINGKEY_ENABLE_FULLMESH;
    }
    if (isLayered) {
        templateDispatch = TILINGKEY_ENABLE_HIERARCHY;
    }
    tilingKey = GET_TPL_TILING_KEY(tp, tilingKeyQuantMode, scaleMode, templateDispatch, commMode, TILINGKEY_TPL_A5);
    return tilingKey;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncA5::CheckQuantModeMatchScales(gert::TilingContext *context,
                                                                               const char *nodeName, bool isScales,
                                                                               uint32_t quantMode,
                                                                               DispatchV2Config &config)
{
    OP_TILING_CHECK(CheckQuantModeAndScales(context, nodeName, isScales, quantMode) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        nodeName, "quant_mode,scales",
                        (std::to_string(quantMode) + "," + std::to_string(static_cast<int32_t>(isScales))).c_str(),
                        "quant_mode and scales not match"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncA5::CheckCommAlgPtr(const char *commAlgPtr, const char *nodeName)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncA5::CheckQuantModePtr(const int64_t *quantModePtr,
                                                                       const char *nodeName)
{
    OP_TILING_CHECK((*quantModePtr < static_cast<int64_t>(QuantModeA5::NON_QUANT)) ||
                        (*quantModePtr > static_cast<int64_t>(QuantModeA5::MX_QUANT)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "quant_mode", std::to_string(*quantModePtr).c_str(),
                                                          "valid range is [0, 4]"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

struct MoeDistributeDispatchCompileInfo {};
static ge::graphStatus TilingParseForMoeDistributeDispatchV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchV2TilingFuncImpl(gert::TilingContext *context)
{
    MoeDistributeDispatchV2TilingFuncA5 impl;
    return impl.MoeDistributeDispatchV2TilingFunc(context);
}

IMPL_OP_OPTILING(MoeDistributeDispatchV2)
    .Tiling(MoeDistributeDispatchV2TilingFuncImpl)
    .TilingParse<MoeDistributeDispatchCompileInfo>(TilingParseForMoeDistributeDispatchV2);
} // namespace optiling
