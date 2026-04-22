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
#include "tiling_base/tiling_templates_registry.h"
#include "mc2_hcom_topo_info.h"

using namespace Ops::Transformer::OpTiling;
using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;

namespace {
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

constexpr uint32_t ARR_LENGTH = 128U;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;     // numeric representation of AlltoAll
constexpr uint32_t OP_TYPE_REDUCE_SCATTER = 7U; // numeric representation of AlltoAll

const int64_t BS_UPPER_BOUND = 512;
constexpr uint64_t TP_WORLD_SIZE_TWO = 2;

constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024; // Bytes
constexpr uint32_t VERSION_2 = 2;
constexpr uint32_t HCOMMCNT_2 = 2;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;

} // namespace

namespace optiling {

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

bool MoeDistributeCombineTilingBase::CheckAttrs(gert::TilingContext *context, MoeDistributeCombineTilingData &tilingData, 
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
    ge::graphStatus ret;
    ret = MoeDistributeCombineTilingFuncImpl(socVersion, context);
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