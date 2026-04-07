/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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

#include "../../op_kernel/mega_moe_tiling_def.h"

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <iomanip>
#include <algorithm>

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "mc2_hcom_topo_info.h"
#include "mega_moe_tiling_v2.h"
#include "moe_init_routing_v3_arch35_tiling.h"
#include "moe_finalize_routing_v2_standalone_tiling.h"
#include "../../op_kernel/workspace_info.h"
#include "../../op_kernel/gmmsq_base.h"
using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;


namespace optiling {
namespace {
    constexpr uint32_t X_IDX = 1;
    constexpr uint32_t EXPERT_IDS_IDX = 2;
    constexpr uint32_t W1_IDX = 6;
    constexpr uint32_t W2_IDX = 8;
    constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
    constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
    constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
    constexpr uint32_t ATTR_BUFFER_SIZE_INDEX = 4;
    constexpr uint32_t ATTR_MAX_RECV_TOKEN_NUM = 5;
}

void PrintMoeV3Arch35VBSComputeTilingData(const MoeV3Arch35VBSComputeTilingData& data, int indent = 0) {
    std::string prefix(indent, ' ');
    std::cout << prefix << "MoeV3Arch35VBSComputeTilingData:" << std::endl;
    std::cout << prefix << "  needCoreNum: " << data.needCoreNum << std::endl;
    std::cout << prefix << "  perCoreElements: " << data.perCoreElements << std::endl;
    std::cout << prefix << "  perCoreLoops: " << data.perCoreLoops << std::endl;
    std::cout << prefix << "  perCorePerLoopElements: " << data.perCorePerLoopElements << std::endl;
    std::cout << prefix << "  perCoreLastLoopElements: " << data.perCoreLastLoopElements << std::endl;
    std::cout << prefix << "  lastCoreElements: " << data.lastCoreElements << std::endl;
    std::cout << prefix << "  lastCoreLoops: " << data.lastCoreLoops << std::endl;
    std::cout << prefix << "  lastCorePerLoopElements: " << data.lastCorePerLoopElements << std::endl;
    std::cout << prefix << "  lastCoreLastLoopElements: " << data.lastCoreLastLoopElements << std::endl;
    std::cout << prefix << "  oneLoopMaxElements: " << data.oneLoopMaxElements << std::endl;
}

void PrintMoeV3Arch35VMSMiddleComputeTilingData(const MoeV3Arch35VMSMiddleComputeTilingData& data, int indent = 0) {
    std::string prefix(indent, ' ');
    std::cout << prefix << "MoeV3Arch35VMSMiddleComputeTilingData:" << std::endl;
    std::cout << prefix << "  needCoreNum: " << data.needCoreNum << std::endl;
}

void PrintMoeV3Arch35SortOutComputeTilingData(const MoeV3Arch35SortOutComputeTilingData& data, int indent = 0) {
    std::string prefix(indent, ' ');
    std::cout << prefix << "MoeV3Arch35SortOutComputeTilingData:" << std::endl;
    std::cout << prefix << "  oneLoopMaxElements: " << data.oneLoopMaxElements << std::endl;
}

void PrintMoeV3Arch35ExpertTokensCountTilingData(const MoeV3Arch35ExpertTokensCountTilingData& data, int indent = 0) {
    std::string prefix(indent, ' ');
    std::cout << prefix << "MoeV3Arch35ExpertTokensCountTilingData:" << std::endl;
    std::cout << prefix << "  needCoreNum: " << data.needCoreNum << std::endl;
    std::cout << prefix << "  perCoreElements: " << data.perCoreElements << std::endl;
    std::cout << prefix << "  lastCoreElements: " << data.lastCoreElements << std::endl;
    std::cout << prefix << "  perCoreLoops: " << data.perCoreLoops << std::endl;
    std::cout << prefix << "  perCorePerLoopElements: " << data.perCorePerLoopElements << std::endl;
    std::cout << prefix << "  perCoreLastLoopElements: " << data.perCoreLastLoopElements << std::endl;
    std::cout << prefix << "  lastCoreLoops: " << data.lastCoreLoops << std::endl;
    std::cout << prefix << "  lastCorePerLoopElements: " << data.lastCorePerLoopElements << std::endl;
    std::cout << prefix << "  lastCoreLastLoopElements: " << data.lastCoreLastLoopElements << std::endl;
}

void PrintMoeV3Arch35GatherOutComputeTilingData(const MoeV3Arch35GatherOutComputeTilingData& data, int indent = 0) {
    std::string prefix(indent, ' ');
    std::cout << prefix << "MoeV3Arch35GatherOutComputeTilingData:" << std::endl;
    std::cout << prefix << "  needCoreNum: " << data.needCoreNum << std::endl;
    std::cout << prefix << "  perCoreIndicesElements: " << data.perCoreIndicesElements << std::endl;
    std::cout << prefix << "  lastCoreIndicesElements: " << data.lastCoreIndicesElements << std::endl;
    std::cout << prefix << "  perCoreIndicesLoops: " << data.perCoreIndicesLoops << std::endl;
    std::cout << prefix << "  perCorePerLoopIndicesElements: " << data.perCorePerLoopIndicesElements << std::endl;
    std::cout << prefix << "  perCoreLastLoopIndicesElements: " << data.perCoreLastLoopIndicesElements << std::endl;
    std::cout << prefix << "  lastCoreIndicesLoops: " << data.lastCoreIndicesLoops << std::endl;
    std::cout << prefix << "  lastCorePerLoopIndicesElements: " << data.lastCorePerLoopIndicesElements << std::endl;
    std::cout << prefix << "  lastCoreLastLoopIndicesElements: " << data.lastCoreLastLoopIndicesElements << std::endl;
    std::cout << prefix << "  colsLoops: " << data.colsLoops << std::endl;
    std::cout << prefix << "  perLoopCols: " << data.perLoopCols << std::endl;
    std::cout << prefix << "  lastLoopCols: " << data.lastLoopCols << std::endl;
    std::cout << prefix << "  activeNum: " << data.activeNum << std::endl;
}

void PrintMoeInitRoutingV3Arch35TilingData(const MoeInitRoutingV3Arch35TilingData& data, int indent = 0) {
    std::string prefix(indent, ' ');
    std::cout << prefix << "MoeInitRoutingV3Arch35TilingData:" << std::endl;
    std::cout << prefix << "  coreNum: " << data.coreNum << std::endl;
    std::cout << prefix << "  n: " << data.n << std::endl;
    std::cout << prefix << "  cols: " << data.cols << std::endl;
    std::cout << prefix << "  k: " << data.k << std::endl;
    std::cout << prefix << "  expertStart: " << data.expertStart << std::endl;
    std::cout << prefix << "  expertEnd: " << data.expertEnd << std::endl;
    std::cout << prefix << "  actualExpertNum: " << data.actualExpertNum << std::endl;
    std::cout << prefix << "  quantMode: " << data.quantMode << std::endl;
    std::cout << prefix << "  rowIdxType: " << data.rowIdxType << std::endl;
    std::cout << prefix << "  isInputScale: " << data.isInputScale << std::endl;
    std::cout << prefix << "  isInputOffset: " << data.isInputOffset << std::endl;
    std::cout << prefix << "  expertNum: " << data.expertNum << std::endl;
    std::cout << prefix << "  expertTokensNumType: " << data.expertTokensNumType << std::endl;
    std::cout << prefix << "  expertTokensNumFlag: " << data.expertTokensNumFlag << std::endl;
    std::cout << prefix << "  gatherFirstFullload: " << data.gatherFirstFullload << std::endl;
    std::cout << prefix << "  epFullload: " << data.epFullload << std::endl;
    std::cout << prefix << "  activeNum: " << data.activeNum << std::endl;
    std::cout << prefix << "  dropPadMode: " << data.dropPadMode << std::endl;
    std::cout << prefix << "  smoothType: " << data.smoothType << std::endl;
    PrintMoeV3Arch35VBSComputeTilingData(data.vbsComputeParamsOp, indent + 2);
    PrintMoeV3Arch35VMSMiddleComputeTilingData(data.vmsMiddleComputeParamsOp, indent + 2);
    PrintMoeV3Arch35SortOutComputeTilingData(data.sortOutComputeParamsOp, indent + 2);
    PrintMoeV3Arch35ExpertTokensCountTilingData(data.expertTokensCountTilingDataOp, indent + 2);
    PrintMoeV3Arch35GatherOutComputeTilingData(data.gatherOutComputeParamsOp, indent + 2);
}

void PrintMegaMoeTilingData(const MegaMoeTilingData* tilingData) {
    if (tilingData == nullptr) {
        std::cout << "[ERROR] TilingData pointer is null." << std::endl;
        return;
    }

    std::cout << "\n========== MegaMoeTilingData ==========" << std::endl;
    PrintMoeInitRoutingV3Arch35TilingData(tilingData->moeInitRoutingTilingData, 2);
    // 基础维度信息
    std::cout << std::left << std::setw(20) << "m:" << tilingData->m << std::endl;
    std::cout << std::left << std::setw(20) << "k:" << tilingData->k << std::endl;
    std::cout << std::left << std::setw(20) << "n:" << tilingData->n << std::endl;
    
    // 专家与路由信息
    std::cout << std::left << std::setw(20) << "topK:" << tilingData->topK << std::endl;
    std::cout << std::left << std::setw(20) << "expertPerRank:" << tilingData->expertPerRank << std::endl;
    std::cout << std::left << std::setw(20) << "groupListType:" << tilingData->groupListType << std::endl;
    
    // 分布式环境信息 (Expert Parallelism)
    std::cout << std::left << std::setw(20) << "rank (EP Rank):" << tilingData->rank << std::endl;
    std::cout << std::left << std::setw(20) << "EP:" << tilingData->EP << std::endl;
    std::cout << std::left << std::setw(20) << "maxOutputSize:" << tilingData->maxOutputSize << std::endl;
    
    // 转置标志位
    std::cout << std::left << std::setw(20) << "transX:" << (tilingData->transX ? "True" : "False") << std::endl;
    std::cout << std::left << std::setw(20) << "transW:" << (tilingData->transW ? "True" : "False") << std::endl;
    std::cout << std::left << std::setw(20) << "transW2:" << (tilingData->transW2 ? "True" : "False") << std::endl;
    
    // 打印 Finalize Routing 的底层 Tiling 参数
    auto &FinalizeRoutingTilingData = tilingData->moeFinalizeRoutingTilingData;
    printf(
    "MoeFinalizeRoutingV2 tiling data: row[%ld] e[%ld] c[%ld] h[%ld] hAligned[%ld] k[%ld] rowOfFormerBlock[%ld] rowOfTailBlock[%ld] rowLoopOfFormerBlock[%ld] rowLoopOfTailBlock[%ld] rowFactor[%ld] tailRowFactorOfFormerBlock[%ld] tailRowFactorOfTailBlock[%ld] hLoop[%ld] hFactor[%ld] tailHFactor[%ld] kLoop[%ld] kFactor[%ld] tailKFactor[%ld] activeNum[%ld] tilingKey[%ld]\n\n",
    FinalizeRoutingTilingData.row, FinalizeRoutingTilingData.e, FinalizeRoutingTilingData.c, FinalizeRoutingTilingData.h, FinalizeRoutingTilingData.hAligned, FinalizeRoutingTilingData.k,
    FinalizeRoutingTilingData.rowOfFormerBlock, FinalizeRoutingTilingData.rowOfTailBlock, FinalizeRoutingTilingData.rowLoopOfFormerBlock,
    FinalizeRoutingTilingData.rowLoopOfTailBlock, FinalizeRoutingTilingData.rowFactor, FinalizeRoutingTilingData.tailRowFactorOfFormerBlock,
    FinalizeRoutingTilingData.tailRowFactorOfTailBlock, FinalizeRoutingTilingData.hLoop, FinalizeRoutingTilingData.hFactor, FinalizeRoutingTilingData.tailHFactor,
    FinalizeRoutingTilingData.kLoop, FinalizeRoutingTilingData.kFactor, FinalizeRoutingTilingData.tailKFactor, FinalizeRoutingTilingData.activeNum, tilingData->moeFinalizeRoutingTilingKey);

    std::cout << "===============================================\n" << std::endl;
}

void printWorkspaceInfo(const struct WorkspaceInfo *info) {
    printf("gmCtx:             %ld\n", info->gmCtx);
    printf("ptrA:              %ld\n", info->ptrA);
    printf("ptrAScale:         %ld\n", info->ptrAScale);
    printf("ptrA2:             %ld\n", info->ptrA2);
    printf("ptrA2Scale:        %ld\n", info->ptrA2Scale);
    printf("ptrcumsumMM:       %ld\n", info->ptrcumsumMM);
    printf("expandedRowIdx:    %ld\n", info->expandedRowIdx);
    printf("ptrSumBeforeRank:  %ld\n", info->ptrSumBeforeRank);
    printf("ptrFlagList:       %ld\n", info->ptrFlagList);
    printf("ptrFlagList2:      %ld\n", info->ptrFlagList2);
    printf("workspaceSize:     %ld\n", info->workspaceSize);
}

ge::graphStatus MegaMoeTilingFuncImplPublic(gert::TilingContext *context, MegaMoeConfig &config)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter Mega_Moe tiling check func.");
    MegaMoeTilingData *tilingData = context->GetTilingData<MegaMoeTilingData>();

    const gert::StorageShape *aStorageShape = context->GetInputShape(X_IDX);
    auto expertIdxTensor = context->GetDynamicInputTensor(EXPERT_IDS_IDX, 0);
    uint32_t M = aStorageShape->GetStorageShape().GetDim(0);
    uint32_t K = aStorageShape->GetStorageShape().GetDim(1);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    context->SetBlockDim(aicNum);
    context->SetScheduleMode(1); // batch model模型，所有核同时启动

    printf("TilingData Init: aivNum: %u, aicNum: %u, ubSize:%lu \n", aivNum, aicNum, ubSize);

    auto wTensor = context->GetDynamicInputTensor(W1_IDX, 0);
    uint32_t wTensorDims = wTensor->GetOriginShape().GetDimNum();
    uint32_t wdim2 = wTensor->GetStorageShape().GetDim(2);
    uint32_t wdim1 = wTensor->GetStorageShape().GetDim(1);
    uint32_t wdim0 = wTensor->GetStorageShape().GetDim(0);
    auto N = (K == wdim1) ? wdim2 : wdim1;
    uint32_t topK = expertIdxTensor->GetStorageShape().GetDim(1);
    tilingData->expertPerRank = wdim0;
    tilingData->groupListType = 1;
    tilingData->m = M;
    tilingData->k = K;
    tilingData->n = N;
    tilingData->transX = false;
    tilingData->transW = true;
    tilingData->transW2 = true;
    tilingData->topK = topK;
    tilingData->m0 = M < NUM_256 ? NUM_128 : NUM_256;
    tilingData->k0 = M < NUM_256 ? NUM_512 : NUM_384;
    tilingData->n0 = NUM_256;
    auto attrs = context->GetAttrs();
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((ATTR_EP_WORLD_SIZE_INDEX));
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "epWorldSize is null."), return ge::GRAPH_FAILED);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>((ATTR_EP_RANK_ID_INDEX));
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName, "epRankId is null."), return ge::GRAPH_FAILED);
    tilingData->rank = *epRankIdPtr;
    tilingData->EP = *epWorldSizePtr;
    tilingData->maxOutputSize = M * topK * tilingData->EP * 3;
    auto maxOutputSizePtr = attrs->GetAttrPointer<int64_t>((ATTR_MAX_RECV_TOKEN_NUM));
    if (maxOutputSizePtr != nullptr) {
        tilingData->maxOutputSize = *maxOutputSizePtr;
    }

    tilingData->blockNumPerEP = std::max(static_cast<uint32_t>(1), aicNum / tilingData->EP);
    tilingData->dispatchRows = 2;

    optiling::GetMoeInitRoutingV3Tiling(*tilingData, aivNum, ubSize);
    optiling::v2::CalcMoeFinalizeRoutingV2Tiling(
        tilingData->m,    // int64_t row,
        tilingData->k,    // int64_t h,
        tilingData->topK,    // int64_t k,
        2,    // int64_t dtypeSize,
        2,    // int64_t scaleDtypeSize,
        32,    // int64_t blockSize,
        ubSize,    // int64_t ubSize,
        aivNum,    // int64_t coreNum,
        false,    // bool hasX1,
        false,    // bool hasX2,
        false,    // bool hasBias,
        true,    // bool hasScales,
        true,    // bool isScaleBf16,
        tilingData->moeFinalizeRoutingTilingData,    // MoeFinalizeRoutingV2RegbaseTilingData &tilingData,
        tilingData->moeFinalizeRoutingTilingKey    // uint64_t &tilingKey)
    );

    int32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *workspace = context->GetWorkspaceSizes(1);
    WorkspaceInfo workspaceInfo((uint8_t*)0, tilingData);
    int64_t workspaceSize = sysWorkspaceSize + workspaceInfo.workspaceSize + 1024 * 1024 * 50;
    workspace[0] = workspaceSize;

    printf("sysWorkspaceSize: %u \n", sysWorkspaceSize);
    printf("mega_moe_tiling workspaceSize: %ld \n", workspaceSize);

    PrintMegaMoeTilingData(tilingData);
    printWorkspaceInfo(&workspaceInfo);

    int64_t peermemSize = tilingData->m * tilingData->topK * (tilingData->k + tilingData->k / MXFP_SCALE_GROUP_NUM) * sizeof(int8_t) + 60 * 1024 + tilingData->EP * 128 * sizeof(int32_t) + tilingData->m * tilingData->topK * tilingData->k * sizeof(int16_t);
    printf("peermemSize: {%ld}\n", peermemSize);
    int64_t offset = PEERMEM_DATA_OFFSET;
    printf("ptrA0: {%ld}\n", PEERMEM_DATA_OFFSET);
    offset += tilingData->m * tilingData->topK * (tilingData->k + tilingData->k / MXFP_SCALE_GROUP_NUM) * sizeof(int8_t);
    printf("ptrTokenPerExpert: {%ld}\n", offset);
    offset += tilingData->EP * CeilAlign(tilingData->EP * tilingData->expertPerRank, ALIGN_128) * sizeof(int32_t);
    printf("ptrD: {%ld}\n", offset);

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