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
 * \file moe_distribute_combine_tiling.cc
 * \brief
 */

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"

#include "op_host/op_tiling/moe_tiling_base.h"
#include "moe_distribute_combine_tiling_base.h"
#include "mc2_hcom_topo_info.h"

using namespace Ops::Transformer::OpTiling;
using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;

namespace {
constexpr uint32_t ARR_LENGTH = 128U;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;     // numeric representation of AlltoAll
constexpr uint32_t OP_TYPE_REDUCE_SCATTER = 7U; // numeric representation of AlltoAll
constexpr int64_t K_MAX = 8;
constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
constexpr uint32_t ATTR_GROUP_TP_INDEX = 4;
constexpr uint32_t ATTR_TP_WORLD_SIZE_INDEX = 5;
constexpr uint32_t ATTR_TP_RANK_ID_INDEX = 6;
constexpr uint32_t ATTR_EXPERT_SHARD_TYPE_INDEX = 7;
constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 8;
constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 9;
constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 10;
constexpr uint32_t ATTR_COMM_QUANT_MODE_INDEX = 12;

constexpr uint32_t INT8_COMM_QUANT = 2U;
constexpr uint32_t EXPAND_IDX_DIMS = 1U;
constexpr uint64_t INIT_TILINGKEY_TP_2 = 1100UL;
constexpr uint64_t INIT_TILINGKEY_TP_1 = 1000UL;
constexpr uint32_t TILINGKEY_INT8_COMM_QUANT = 20U;

const int64_t BS_UPPER_BOUND = 512;
const size_t MAX_GROUP_NAME_LENGTH = 128UL;

constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024; // Bytes
constexpr uint32_t VERSION_2 = 2;
constexpr uint32_t HCOMMCNT_2 = 2;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
const int64_t MAX_EP_WORLD_SIZE = 288;
const int64_t MAX_TP_WORLD_SIZE = 2;
constexpr uint64_t TP_WORLD_SIZE_TWO = 2;
constexpr int64_t MOE_EXPERT_MAX_NUM = 512;

} // namespace

namespace optiling {

void MoeDistributeCombineTilingBase::PrintTilingDataInfo(const char *nodeName,
    MoeDistributeCombineTilingData &tilingData)
{
    OP_LOGD(nodeName, "epWorldSize is %u.", tilingData.moeDistributeCombineInfo.epWorldSize);
    OP_LOGD(nodeName, "tpWorldSize is %u.", tilingData.moeDistributeCombineInfo.tpWorldSize);
    OP_LOGD(nodeName, "epRankId is %u.", tilingData.moeDistributeCombineInfo.epRankId);
    OP_LOGD(nodeName, "tpRankId is %u.", tilingData.moeDistributeCombineInfo.tpRankId);
    OP_LOGD(nodeName, "expertShardType is %u.", tilingData.moeDistributeCombineInfo.expertShardType);
    OP_LOGD(nodeName, "sharedExpertRankNum is %u.", tilingData.moeDistributeCombineInfo.sharedExpertRankNum);
    OP_LOGD(nodeName, "moeExpertNum is %u.", tilingData.moeDistributeCombineInfo.moeExpertNum);
    OP_LOGD(nodeName, "moeExpertPerRankNum is %u.", tilingData.moeDistributeCombineInfo.moeExpertPerRankNum);
    OP_LOGD(nodeName, "globalBs is %u.", tilingData.moeDistributeCombineInfo.globalBs);
    OP_LOGD(nodeName, "bs is %d.", tilingData.moeDistributeCombineInfo.bs);
    OP_LOGD(nodeName, "k is %d.", tilingData.moeDistributeCombineInfo.k);
    OP_LOGD(nodeName, "h is %d.", tilingData.moeDistributeCombineInfo.h);
    OP_LOGD(nodeName, "aivNum is %d.", tilingData.moeDistributeCombineInfo.aivNum);
    OP_LOGD(nodeName, "totalUbSize is %ld.", tilingData.moeDistributeCombineInfo.totalUbSize);
    OP_LOGD(nodeName, "totalWinSizeEP is %lu.", tilingData.moeDistributeCombineInfo.totalWinSizeEp);
    OP_LOGD(nodeName, "totalWinSizeTP is %lu.", tilingData.moeDistributeCombineInfo.totalWinSizeTp);
}

static ge::graphStatus CheckAttrPointersNotNull(gert::TilingContext *context, const char *nodeName) 
{
    auto attrs = context->GetAttrs();
    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_RANK_ID_INDEX);
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_SHARED_EXPERT_NUM_INDEX));
    auto commQuantModePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_COMM_QUANT_MODE_INDEX));
    
    OP_TILING_CHECK((groupEpPtr == nullptr) || (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
        (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
        OP_LOGE(nodeName, "groupEp is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "epWorldSize is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE(nodeName, "tpWorldSize is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName, "epRankId is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE(nodeName, "tpRankId is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertShardPtr == nullptr, OP_LOGE(nodeName, "expertShardType is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE(nodeName, "sharedExpertRankNum is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE(nodeName, "moeExpertNum is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertNumPtr == nullptr, OP_LOGE(nodeName, "sharedExpertNum is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commQuantModePtr == nullptr, OP_LOGE(nodeName, "commQuantMode is null."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckCommAttrValuesValid(gert::TilingContext *context, const char *nodeName, std::string &groupTp)
{
    auto attrs = context->GetAttrs();
    auto groupTpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_TP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_RANK_ID_INDEX);
    auto commQuantModePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_COMM_QUANT_MODE_INDEX));

    OP_TILING_CHECK((*epWorldSizePtr <= 0) || (*epWorldSizePtr > MAX_EP_WORLD_SIZE),
        OP_LOGE(nodeName, "epWorldSize is invalid, only support (0, %ld], but got epWorldSize=%ld.",
        MAX_EP_WORLD_SIZE, *epWorldSizePtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*tpWorldSizePtr < 0) || (*tpWorldSizePtr > MAX_TP_WORLD_SIZE),
        OP_LOGE(nodeName, "tpWorldSize is invalid, only support [0, %ld], but got tpWorldSize=%ld.",
        MAX_TP_WORLD_SIZE, *tpWorldSizePtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*epRankIdPtr < 0) || (*epRankIdPtr >= *epWorldSizePtr),
        OP_LOGE(nodeName, "epRankId is invalid, only support [0, %ld), but got epRankId=%ld.",
        *epWorldSizePtr, *epRankIdPtr), return ge::GRAPH_FAILED);
    if (*tpWorldSizePtr > 1) {
        OP_TILING_CHECK((*tpRankIdPtr < 0) || (*tpRankIdPtr >= *tpWorldSizePtr),
            OP_LOGE(nodeName, "tpRankId is invalid, only support [0, %ld), but got tpRankId=%ld.",
            *tpWorldSizePtr, *tpRankIdPtr), return ge::GRAPH_FAILED);
        OP_TILING_CHECK((groupTpPtr == nullptr) || (strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
            (strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
            OP_LOGE(nodeName, "groupTpPtr is null."), return ge::GRAPH_FAILED);
        OP_TILING_CHECK((*commQuantModePtr != 0), OP_LOGE(nodeName,
            "commQuantMode only supports 0 when tpWorldSize > 1, but got commQuantMode=%ld, tpWorldSize=%ld.",
            *commQuantModePtr, *tpWorldSizePtr), return ge::GRAPH_FAILED);
        groupTp = std::string(groupTpPtr);
    } else {
        OP_TILING_CHECK(*tpRankIdPtr != 0, OP_LOGE(nodeName,
            "tpRankId is invalid, NoTp mode only support 0, but got tpRankId=%ld.", *tpRankIdPtr),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrValuesValid(gert::TilingContext *context, const char *nodeName, std::string &groupTp) 
{
    auto attrs = context->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_SHARED_EXPERT_NUM_INDEX));
    auto commQuantModePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_COMM_QUANT_MODE_INDEX));

    OP_TILING_CHECK(CheckCommAttrValuesValid(context, nodeName, groupTp) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "ep tp attrs is invalid"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*expertShardPtr != 0,
        OP_LOGE(nodeName, "expertShardType is invalid, only support 0, but got expertShardType=%ld.", *expertShardPtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*sharedExpertRankNumPtr < 0) || (*sharedExpertRankNumPtr >= *epWorldSizePtr),
        OP_LOGE(nodeName, "sharedExpertRankNum is invalid, only support [0, %ld), but got sharedExpertRankNum=%ld.",
        *epWorldSizePtr, *sharedExpertRankNumPtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*sharedExpertNumPtr != 1,
        OP_LOGE(nodeName, "sharedExpertNum only support 1, but got sharedExpertNum=%ld.", *sharedExpertNumPtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*moeExpertNumPtr <= 0) || (*moeExpertNumPtr > MOE_EXPERT_MAX_NUM),
        OP_LOGE(nodeName, "moeExpertNum is invalid, only support (0, %ld], but got moeExpertNum=%ld.",
        MOE_EXPERT_MAX_NUM, *moeExpertNumPtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*commQuantModePtr != 0) && (*commQuantModePtr != 2),
        OP_LOGE(nodeName, "commQuantMode only support 0 or 2, but got commQuantMode=%ld.", *commQuantModePtr),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTilingBase::GetAttrAndSetTilingData(gert::TilingContext *context,
    MoeDistributeCombineTilingData &tilingData,
    const char *nodeName, std::string &groupEp, std::string &groupTp, uint32_t &commQuantMode)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is null."), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto groupTpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_TP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_RANK_ID_INDEX);
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_SHARED_EXPERT_NUM_INDEX));
    auto commQuantModePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_COMM_QUANT_MODE_INDEX));

    // 校验指针非空
    OP_TILING_CHECK(CheckAttrPointersNotNull(context, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check attr pointers not null failed."), return ge::GRAPH_FAILED);

    // 判断是否满足uint32_t及其他限制
    OP_TILING_CHECK(CheckAttrValuesValid(context, nodeName, groupTp) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check attr values valid failed."), return ge::GRAPH_FAILED);

    commQuantMode = static_cast<uint32_t>(*commQuantModePtr);
    groupEp = string(groupEpPtr);
    tilingData.moeDistributeCombineInfo.epWorldSize = static_cast<uint32_t>(*epWorldSizePtr);
    tilingData.moeDistributeCombineInfo.tpWorldSize = static_cast<uint32_t>(*tpWorldSizePtr);
    tilingData.moeDistributeCombineInfo.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData.moeDistributeCombineInfo.tpRankId = static_cast<uint32_t>(*tpRankIdPtr);
    tilingData.moeDistributeCombineInfo.expertShardType = static_cast<uint32_t>(*expertShardPtr);
    tilingData.moeDistributeCombineInfo.sharedExpertRankNum = static_cast<uint32_t>(*sharedExpertRankNumPtr);
    tilingData.moeDistributeCombineInfo.moeExpertNum = static_cast<uint32_t>(*moeExpertNumPtr);

    return ge::GRAPH_SUCCESS;
}

static bool CheckExpertAttrs(gert::TilingContext *context, 
    MoeDistributeCombineTilingData &tilingData, const char *nodeName, uint32_t &localMoeExpertNum) 
{
    uint32_t epWorldSize = tilingData.moeDistributeCombineInfo.epWorldSize;
    uint32_t tpWorldSize = tilingData.moeDistributeCombineInfo.tpWorldSize;
    uint32_t moeExpertNum = tilingData.moeDistributeCombineInfo.moeExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeCombineInfo.sharedExpertRankNum;

    // 校验ep能均分共享
    OP_TILING_CHECK((sharedExpertRankNum != 0) && (epWorldSize % sharedExpertRankNum != 0),
        OP_LOGE(nodeName, "epWorldSize should be divisible by sharedExpertRankNum, but epWorldSize=%u, "
        "sharedExpertRankNum=%u.", epWorldSize, sharedExpertRankNum), return false);

    // 校验moe专家数量能否均分给多机
    OP_TILING_CHECK(moeExpertNum % (epWorldSize - sharedExpertRankNum) != 0,
        OP_LOGE(nodeName, "moeExpertNum should be divisible by (epWorldSize - sharedExpertRankNum), "
        "but got moeExpertNum=%u, epWorldSize=%u, sharedExpertRankNum=%u.", moeExpertNum, epWorldSize, 
        sharedExpertRankNum), return false);
    localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    OP_TILING_CHECK(localMoeExpertNum <= 0,
        OP_LOGE(nodeName, "localMoeExpertNum is invalid, localMoeExpertNum = %u", localMoeExpertNum),
        return false);
    OP_TILING_CHECK((localMoeExpertNum > 1) && (tpWorldSize > 1),
        OP_LOGE(nodeName, "Cannot support multi-moeExpert %u in a rank when tpWorldSize = %u > 1",
        localMoeExpertNum, tpWorldSize), return false);
    tilingData.moeDistributeCombineInfo.moeExpertPerRankNum = localMoeExpertNum;

    // 校验k > moeExpertNum
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    const int64_t expertIdsDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    uint32_t K = static_cast<uint32_t>(expertIdsDim1);
    OP_TILING_CHECK(K > moeExpertNum, OP_LOGE(nodeName, "K is larger than moeExpertNum, "
        "k is %u, moeExpertNum is %u.", K, moeExpertNum), return false);

    return true;
}

bool MoeDistributeCombineTilingBase::CheckEpWorldSizeAttrs(gert::TilingContext *context,
    MoeDistributeCombineTilingData &tilingData, const char *nodeName)
{
    uint32_t epWorldSize = tilingData.moeDistributeCombineInfo.epWorldSize;

    return CheckEpWorldSize(nodeName, epWorldSize);
}

static bool CheckBatchAttrs(gert::TilingContext *context,
    MoeDistributeCombineTilingData &tilingData, const char *nodeName)
{
    uint32_t epWorldSize = tilingData.moeDistributeCombineInfo.epWorldSize;

    // 校验输入expertIds的维度0并设bs
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdsStorageShape == nullptr, OP_LOGE(nodeName, "expertIds id null."), return false);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK((expertIdsDim0 <= 0) || (expertIdsDim0 > BS_UPPER_BOUND),
        OP_LOGE(nodeName, "Invalid expertIds dims0(BS) %ld. Should be between [1, %ld].", expertIdsDim0,
        BS_UPPER_BOUND), return false);
    tilingData.moeDistributeCombineInfo.bs = static_cast<uint32_t>(expertIdsDim0);

    // 校验globalBS
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is null."), return false);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE(nodeName, "globalBs is null."), return false);
    OP_LOGD(nodeName, "MoeDistributeCombine *globalBsPtr = %ld, bs = %ld, epWorldSize = %u\n", 
        *globalBsPtr, expertIdsDim0, epWorldSize);

    OP_TILING_CHECK((*globalBsPtr != 0) && ((*globalBsPtr < static_cast<int64_t>(epWorldSize) * expertIdsDim0) ||
        ((*globalBsPtr) % (static_cast<int64_t>(epWorldSize)) != 0)), OP_LOGE(nodeName, "globalBS is invalid, only "
        "support 0 or maxBs(maxBs is the largest bs on all ranks) * epWorldSize, but got globalBS=%ld, "
        "bs=%ld, epWorldSize=%u.", *globalBsPtr, expertIdsDim0, epWorldSize), return false);
    if (*globalBsPtr == 0) {
        tilingData.moeDistributeCombineInfo.globalBs = static_cast<uint32_t>(expertIdsDim0) * epWorldSize;
    } else {
        tilingData.moeDistributeCombineInfo.globalBs = static_cast<uint32_t>(*globalBsPtr);
    }

    return true;
}

static bool CheckBasicInputTensorShape(gert::TilingContext *context, MoeDistributeCombineTilingData &tilingData,
    const char *nodeName, bool isShared, uint32_t localExpertNum) 
{ 
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    int64_t expertIdsDim1 = expertIdsStorageShape->GetStorageShape().GetDim(1);

    uint32_t A = 0;
    uint32_t globalBs = tilingData.moeDistributeCombineInfo.globalBs;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeCombineInfo.sharedExpertRankNum;
    if (isShared) { // 本卡为共享专家
        A = globalBs / sharedExpertRankNum;
    } else { // 本卡为moe专家
        A = globalBs * std::min(static_cast<int64_t>(localExpertNum), expertIdsDim1);
    }
    tilingData.moeDistributeCombineInfo.a = A;

    // 校验expandX的维度并设h
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.moeDistributeCombineInfo.tpWorldSize);
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(EXPAND_X_INDEX);
    int64_t expandXDim0 = expandXStorageShape->GetStorageShape().GetDim(0);
    int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expandXDim0 < tpWorldSize * static_cast<int64_t>(A), OP_LOGE(nodeName, 
        "expandX's dim0 not greater than or equal to A * tpWorldSize, expandXDim0 = %ld, A = %ld, "
        "tpWorldSize = %ld", expandXDim0, static_cast<int64_t>(A), tpWorldSize), return false);
    OP_TILING_CHECK((expandXDim1 != 7168),
        OP_LOGE(nodeName, "expandX dims1(H) only supports 7168, but got %ld.", expandXDim1), return false);
    tilingData.moeDistributeCombineInfo.h = static_cast<uint32_t>(expandXDim1);

    OP_TILING_CHECK((expertIdsDim1 <= 0) || (expertIdsDim1 > K_MAX), OP_LOGE(nodeName,
        "expertIdShape's dim1(k) should be in (0, %ld], but got expertIdShape's dim1=%ld.", K_MAX,
        expertIdsDim1), return false);
    tilingData.moeDistributeCombineInfo.k = static_cast<uint32_t>(expertIdsDim1);

    // 校验expandIdx的维度
    const gert::StorageShape *expandIdxStorageShape = context->GetInputShape(EXPAND_IDX_INDEX);
    int64_t expandIdxDim0 = expandIdxStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(expandIdxDim0 != expertIdsDim0 * expertIdsDim1,
        OP_LOGE(nodeName, "expandIdxDim0 != bs * k, expandIdxDim0 is %ld, bs * k is %ld.", expandIdxDim0,
        expertIdsDim0 * expertIdsDim1), return false);

    return true;
}

static bool CheckCommInputTensorShape(gert::TilingContext *context, MoeDistributeCombineTilingData &tilingData,
    const char *nodeName, bool isShared)
{ 
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    int64_t expertIdsDim1 = expertIdsStorageShape->GetStorageShape().GetDim(1);
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.moeDistributeCombineInfo.tpWorldSize);

    // 校验epSendCount和tpSendCount的维度
    int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeCombineInfo.epWorldSize);
    int64_t moeExpertPerRankNum = static_cast<int64_t>(tilingData.moeDistributeCombineInfo.moeExpertPerRankNum);
    const gert::StorageShape *epSendCountStorageShape = context->GetInputShape(EP_SEND_COUNTS_INDEX);
    const gert::StorageShape *tpSendCountStorageShape = context->GetOptionalInputShape(TP_SEND_COUNTS_INDEX);
    const int64_t epSendCountDim0 = epSendCountStorageShape->GetStorageShape().GetDim(0);
    const int64_t tpSendCountDim0 = tpSendCountStorageShape->GetStorageShape().GetDim(0);
    int64_t epSendCount = (isShared) ? epWorldSize : epWorldSize * moeExpertPerRankNum;
    OP_TILING_CHECK(epSendCountDim0 < epSendCount * tpWorldSize, OP_LOGE(nodeName,
        "epSendCountDim0 not greater than or equal to epSendCount * tpWorldSize, epSendCountDim0 is %ld, epSendCount is %ld, "
        "tpWorldSize is %ld.", epSendCountDim0, epSendCount, tpWorldSize), return false);
    OP_TILING_CHECK(tpSendCountDim0 != tpWorldSize, OP_LOGE(nodeName,
        "tpSendCountDim0 not equal to tpWorldSize, tpSendCountDim0 is %ld, tpWorldSize is %ld.",
        tpSendCountDim0, tpWorldSize), return false);

    // 校验expertScales的维度
    const gert::StorageShape *expertScalesStorageShape = context->GetInputShape(EXPERT_SCALES_INDEX);
    int64_t expertScalesDim0 = expertScalesStorageShape->GetStorageShape().GetDim(0);
    int64_t expertScalesDim1 = expertScalesStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expertScalesDim0 != expertIdsDim0, OP_LOGE(nodeName, 
        "expertScales' dim0 not equal to bs, expertScalesDim0 = %ld, bs = %ld", expertScalesDim0, expertIdsDim0),
        return false);
    OP_TILING_CHECK(expertScalesDim1 != expertIdsDim1, OP_LOGE(nodeName, 
        "expertScales' dim1 not equal to k, expertScalesDim1 = %ld, k = %ld", expertScalesDim1, expertIdsDim1),
        return false);

    return true;
}

bool MoeDistributeCombineTilingBase::CheckTensorShape(gert::TilingContext *context,
    MoeDistributeCombineTilingData &tilingData,
    const char *nodeName, bool isShared, uint32_t localExpertNum)
{
    // 校验输入expertIds的维度1并设k, bs已校验过
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    int64_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    const gert::StorageShape *expandXStorageShape = context->GetInputShape(EXPAND_X_INDEX);
    int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);

    OP_TILING_CHECK(!CheckBasicInputTensorShape(context, tilingData, nodeName, isShared, localExpertNum),
        OP_LOGE(nodeName, "Check basic input shapes failed."), return false);

    OP_TILING_CHECK(!CheckCommInputTensorShape(context, tilingData, nodeName, isShared),
        OP_LOGE(nodeName, "Check comm input shapes failed."), return false);

    // 校验x的维度
    const gert::StorageShape *xStorageShape = context->GetOutputShape(OUTPUT_X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "x is null."), return false);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(xDim0 != expertIdsDim0,
                    OP_LOGE(nodeName, "xDim0 not equal to bs, bs = %ld, xDim0 = %ld", expertIdsDim0, xDim0),
                    return false);
    OP_TILING_CHECK(xDim1 != expandXDim1,
                    OP_LOGE(nodeName, "xDim1 not equal to h, xDim1 = %ld, h = %ld", xDim1, expandXDim1), return false);

    return true;
}

ge::graphStatus MoeDistributeCombineTilingBase::SetWorkspace(gert::TilingContext *context, const char *nodeName)
{
    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, VECTOR_INNER_ERR_REPORT_TILING(nodeName, "get workspace failed"),
                    return ge::GRAPH_FAILED);
    workspace[0] = SYSTEM_NEED_WORKSPACE;
    OP_LOGD(nodeName, "workspace[0] size is %ld", workspace[0]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTilingBase::SetHCommCfg(const gert::TilingContext *context,
    MoeDistributeCombineTilingData *tiling,
    const std::string groupEp, const std::string groupTp, const uint32_t tpWorldSize)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "MoeDistributeCombine groupEp = %s", groupEp.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    uint32_t opType2 = OP_TYPE_REDUCE_SCATTER;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    std::string algConfigReduceScatterStr = "ReduceScatter=level0:ring";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.SetCommEngine(mc2tiling::AIV_ENGINE);   // 通过不拉起AICPU，提高算子退出性能
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2tiling GetTiling mc2InitTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2tiling GetTiling mc2CcTiling1 failed"), return ge::GRAPH_FAILED);

    if (tpWorldSize > 1) {
        OP_LOGD(nodeName, "MoeDistributeCombine groupTp = %s", groupTp.c_str());
        mc2CcTilingConfig.SetGroupName(groupTp);
        mc2CcTilingConfig.SetOpType(opType2);
        mc2CcTilingConfig.SetAlgConfig(algConfigReduceScatterStr);
        OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling2) != 0,
            OP_LOGE(nodeName, "mc2CcTilingConfig mc2tiling GetTiling mc2CcTiling2 failed"), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTilingBase::CheckWinSize(const gert::TilingContext *context,
    MoeDistributeCombineTilingData* tilingData,
    const char *nodeName, uint32_t localMoeExpertNum)
{
    auto attrs = context->GetAttrs();
    uint64_t hcclBufferSizeEp = 0;
    uint64_t maxWindowSizeEp = 0;
    OP_TILING_CHECK(mc2tiling::GetEpWinSize(
        context, nodeName, hcclBufferSizeEp, maxWindowSizeEp, ATTR_GROUP_EP_INDEX, false) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Get EP WinSize failed"), return ge::GRAPH_FAILED);
    uint64_t h = static_cast<uint64_t>(tilingData->moeDistributeCombineInfo.h);
    uint64_t epWorldSize = static_cast<uint64_t>(tilingData->moeDistributeCombineInfo.epWorldSize);
    uint64_t maxBs = static_cast<uint64_t>(tilingData->moeDistributeCombineInfo.globalBs) / epWorldSize;
    uint64_t actualSize = epWorldSize * maxBs * h * 2UL * 2UL * static_cast<uint64_t>(localMoeExpertNum);
    if (actualSize > maxWindowSizeEp) {
        OP_LOGE(nodeName,
                "HCCL_BUFFSIZE is too SMALL, maxBs = %lu, h = %lu, epWorldSize = %lu, localMoeExpertNum = %u,"
                "ep_worldsize * maxBs * h * 2 * 2 * localMoeExpertNum = %luMB, HCCL_BUFFSIZE=%luMB.",
                maxBs, h, epWorldSize, localMoeExpertNum, actualSize / MB_SIZE + 1UL, hcclBufferSizeEp / MB_SIZE);
        return ge::GRAPH_FAILED;
    }
    tilingData->moeDistributeCombineInfo.totalWinSizeEp = maxWindowSizeEp;
    OP_LOGD(nodeName, "EpwindowSize = %lu", maxWindowSizeEp);

    uint64_t tpWorldSize = static_cast<uint64_t>(tilingData->moeDistributeCombineInfo.tpWorldSize);
    if (tpWorldSize == TP_WORLD_SIZE_TWO) {
        uint64_t maxWindowSizeTp = 0;
        auto groupTpHccl = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_TP_INDEX));
        OP_TILING_CHECK(mc2tiling::GetCclBufferSize(groupTpHccl, &maxWindowSizeTp, nodeName) != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "Get Ep HcclBufferSizeTP failed, HcclBufferSizeTP is %lu", maxWindowSizeTp),
            return ge::GRAPH_FAILED);
        actualSize = static_cast<uint64_t>(tilingData->moeDistributeCombineInfo.a) * (h * 2UL + 128UL) * 2UL;
        OP_TILING_CHECK((actualSize > maxWindowSizeTp), OP_LOGE(nodeName,
            "TP HCCL_BUFFSIZE is too SMALL, A = %u, h = %lu, NEEDED_HCCL_BUFFSIZE(A * (h * 2UL + 128UL) * 2UL)"
            " = %luMB, TP HCCL_BUFFSIZE=%luMB.", tilingData->moeDistributeCombineInfo.a,
            h, actualSize / MB_SIZE + 1UL, maxWindowSizeTp / MB_SIZE), return ge::GRAPH_FAILED);
        tilingData->moeDistributeCombineInfo.totalWinSizeTp = maxWindowSizeTp;
        OP_LOGD(nodeName, "TpwindowSize = %lu", maxWindowSizeTp);
    }
    return ge::GRAPH_SUCCESS;
}

bool MoeDistributeCombineTilingBase::CheckAttrs(gert::TilingContext *context,
    MoeDistributeCombineTilingData &tilingData,
    const char *nodeName, uint32_t &localMoeExpertNum)
{
    OP_TILING_CHECK(!CheckExpertAttrs(context, tilingData, nodeName, localMoeExpertNum),
        OP_LOGE(nodeName, "Check expert params failed."), return false);

    OP_TILING_CHECK(!CheckEpWorldSizeAttrs(context, tilingData, nodeName),
        OP_LOGE(nodeName, "Check epworldsize params failed."), return false);

    OP_TILING_CHECK(!CheckBatchAttrs(context, tilingData, nodeName),
        OP_LOGE(nodeName, "Check batch params failed."), return false);

    return true;
}

ge::graphStatus MoeDistributeCombineTilingBase::MoeDistributeCombineTilingFunc(gert::TilingContext *context)
{
    // 不支持 expandX数据类型为int32 type
    auto expandXDesc = context->GetInputDesc(EXPAND_X_INDEX);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandxDesc is null."), return ge::GRAPH_FAILED);
    // 检查expandX数据类型为DT_INT32
    OP_TILING_CHECK((expandXDesc->GetDataType() == ge::DT_INT32),
                    OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be bf16 or float16, but is %s",
                            Ops::Base::ToString(expandXDesc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);

    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;

    std::string socVersion;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);
    ge::graphStatus ret = MoeDistributeCombineTilingFuncImpl(socVersion, context);
    return ret;
}

struct MoeDistributeCombineCompileInfo {};
ge::graphStatus TilingParseForMoeDistributeCombine(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTilingBase::DoOpTiling()
{
    return MoeDistributeCombineTilingFunc(context_);
}

ge::graphStatus MoeDistributeCombineTiling(gert::TilingContext *context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

uint64_t MoeDistributeCombineTilingBase::GetTilingKey() const
{
    // TilingKey calculation is done in DoOptiling
    const uint64_t tilingKey = context_->GetTilingKey();
    const char *nodeName = context_->GetNodeName();
    OP_LOGD(nodeName, "MoeDistributeCombineTilingBase get tiling key %lu", tilingKey);
    return tilingKey;
}

bool MoeDistributeCombineTilingBase::IsCapable()
{
    return true;
}

IMPL_OP_OPTILING(MoeDistributeCombine)
    .Tiling(MoeDistributeCombineTiling)
    .TilingParse<MoeDistributeCombineCompileInfo>(TilingParseForMoeDistributeCombine);

} // namespace optiling