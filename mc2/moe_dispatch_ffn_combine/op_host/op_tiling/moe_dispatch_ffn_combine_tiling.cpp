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
 * \brief
 */

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
#include <limits>

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "mc2_hcom_topo_info.h"
#include "moe_dispatch_ffn_combine.h"
#include "../../op_kernel/moe_dispatch_ffn_combine_tiling.h"
#include "../../op_kernel/moe_dispatch_ffn_combine_tiling_key.h"
#include "../../op_kernel/moe_dispatch_ffn_combine_workspace_info.h"

using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;

namespace optiling {
namespace {
    // init routing
    const static int64_t SIMT_DCACHE_SIZE = 64 * 1024LL;
    const static int64_t SORT_API_MAX_ELEM = 32 * 255LL;
    const static int64_t MRG_SORT_API_MAX_ELEM = 1024LL;
    const static int64_t MX_QUANT_BLOCK_SIZE = 32LL;

    const static int64_t NUM_TWO = 2LL;
    const static int64_t NUM_THREE = 3LL;
    const static int64_t NUM_FOUR = 4LL;
    const static int64_t MRG_LIST_NUM = 4LL;
    const static int64_t SORT32_ALIGN_ELEMENT = 32LL;
    const static int64_t UB_BLOCK_SIZE = 32LL;
    const static size_t DIM_ONE = 1ULL;
    const static size_t DIM_TWO = 2ULL;
    const static int64_t KV_FACTOR = 2LL;
    const static int64_t EXPERT_IDX_MAX = 10240LL;
    const static int64_t KV_MODE_EXPERT_IDX_MAX = EXPERT_IDX_MAX / KV_FACTOR;

    const static int64_t DYNAMIC_QUANT_COLS_BUFFER = 21LL;

    const static int64_t ROW_IDX_GATHER = 0LL;
    const static int64_t ROW_IDX_SCATTER = 1LL;
    const static int64_t EXPERT_TOKENS_TYPE_COUNT = 1LL;
    const static int64_t EXPERT_TOKENS_TYPE_KEY_VALUE = 2LL;
    const static int64_t DROP_PAD_MODE_DROPLESS = 0LL;
    const static int64_t SORT_CORE_TILINGKEY_BASE = 100000LL;

    const static int64_t THREE_DIMS = 3LL;
    const static int64_t TWO_DIMS = 2LL;
    const static int64_t ONE_DIM = 1LL;
    const static int64_t MIN_BS = 1LL;
    const static int64_t MAX_BS = 512LL;
    const static int64_t MIN_EXPERT_PER_RANK = 1LL;
    const static int64_t MAX_EXPERT_PER_RANK = 16LL;
    const static int64_t H_BASE = 1024LL;
    const static int64_t N_BASE = 1024LL;
    const static int64_t MIN_EP_WORLD_SIZE = 2LL;
    const static int64_t MAX_EP_WORLD_SIZE = 768LL;
    const static int64_t MAX_MOE_EXPERT_NUM = 1024LL;
    const static int64_t DISABLE_EXPERT_CAPACITY = -1LL;
    const static int64_t DISPATCH_QUANT_OUT_TYPE_E5M2 = 23LL;
    const static int64_t DISPATCH_QUANT_OUT_TYPE_E4M3 = 24LL;
}

template<typename T>
inline static T CeilDiv(T x, T y)
{
    return (x + y - 1) / y;
}

template<typename T>
inline static T CeilAlign(T x, T align)
{
    return ((x + align - 1) / align) * align;
}

inline static int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE * UB_BLOCK_SIZE / bytes;
}

inline static int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE * UB_BLOCK_SIZE;
}

inline static int64_t CeilLog4(int64_t x)
{
    if (x <= 1) return 0;
    return static_cast<int64_t>(std::ceil(std::log(x) / std::log(NUM_FOUR)));
}

inline static int64_t OpQuantModeToInitRoutingQuantMode(const int64_t opQuantMode)
{
    // 暂不支持 UNQUANT, STATIC, DYNAMIC
    switch (opQuantMode) {
        case DISPATCH_QUANT_MODE_MXFP8_E5M2: return 2;
        case DISPATCH_QUANT_MODE_MXFP8_E4M3FN: return 3;
        default: return -1;
    }
    return -1;
}

void PrintMoeV3Arch35VBSComputeTilingData(const MoeV3Arch35VBSComputeTilingData& data, const char *nodeName) {
    OP_LOGD(nodeName, "========== MoeV3Arch35VBSComputeTilingData ==========");
    OP_LOGD(nodeName, "needCoreNum is %ld", data.needCoreNum);
    OP_LOGD(nodeName, "perCoreElements is %ld", data.perCoreElements);
    OP_LOGD(nodeName, "perCoreLoops is %ld", data.perCoreLoops);
    OP_LOGD(nodeName, "perCorePerLoopElements is %ld", data.perCorePerLoopElements);
    OP_LOGD(nodeName, "perCoreLastLoopElements is %ld", data.perCoreLastLoopElements);
    OP_LOGD(nodeName, "lastCoreElements is %ld", data.lastCoreElements);
    OP_LOGD(nodeName, "lastCoreLoops is %ld", data.lastCoreLoops);
    OP_LOGD(nodeName, "lastCorePerLoopElements is %ld", data.lastCorePerLoopElements);
    OP_LOGD(nodeName, "lastCoreLastLoopElements is %ld", data.lastCoreLastLoopElements);
    OP_LOGD(nodeName, "oneLoopMaxElements is %ld", data.oneLoopMaxElements);
}

void PrintMoeV3Arch35VMSMiddleComputeTilingData(const MoeV3Arch35VMSMiddleComputeTilingData& data, const char *nodeName) {
    OP_LOGD(nodeName, "========== MoeV3Arch35VMSMiddleComputeTilingData ==========");
    OP_LOGD(nodeName, "needCoreNum is %ld", data.needCoreNum);
}

void PrintMoeV3Arch35SortOutComputeTilingData(const MoeV3Arch35SortOutComputeTilingData& data, const char *nodeName) {
    OP_LOGD(nodeName, "========== MoeV3Arch35SortOutComputeTilingData ==========");
    OP_LOGD(nodeName, "oneLoopMaxElements is %ld", data.oneLoopMaxElements);
}

void PrintMoeV3Arch35ExpertTokensCountTilingData(const MoeV3Arch35ExpertTokensCountTilingData& data, const char *nodeName) {
    OP_LOGD(nodeName, "========== MoeV3Arch35ExpertTokensCountTilingData ==========");
    OP_LOGD(nodeName, "needCoreNum is %ld", data.needCoreNum);
    OP_LOGD(nodeName, "perCoreElements is %ld", data.perCoreElements);
    OP_LOGD(nodeName, "lastCoreElements is %ld", data.lastCoreElements);
    OP_LOGD(nodeName, "perCoreLoops is %ld", data.perCoreLoops);
    OP_LOGD(nodeName, "perCorePerLoopElements is %ld", data.perCorePerLoopElements);
    OP_LOGD(nodeName, "perCoreLastLoopElements is %ld", data.perCoreLastLoopElements);
    OP_LOGD(nodeName, "lastCoreLoops is %ld", data.lastCoreLoops);
    OP_LOGD(nodeName, "lastCorePerLoopElements is %ld", data.lastCorePerLoopElements);
    OP_LOGD(nodeName, "lastCoreLastLoopElements is %ld", data.lastCoreLastLoopElements);
}

void PrintMoeV3Arch35GatherOutComputeTilingData(const MoeV3Arch35GatherOutComputeTilingData& data, const char *nodeName) {
    OP_LOGD(nodeName, "========== MoeV3Arch35GatherOutComputeTilingData ==========");
    OP_LOGD(nodeName, "needCoreNum is %ld", data.needCoreNum);
    OP_LOGD(nodeName, "perCoreIndicesElements is %ld", data.perCoreIndicesElements);
    OP_LOGD(nodeName, "lastCoreIndicesElements is %ld", data.lastCoreIndicesElements);
    OP_LOGD(nodeName, "perCoreIndicesLoops is %ld", data.perCoreIndicesLoops);
    OP_LOGD(nodeName, "perCorePerLoopIndicesElements is %ld", data.perCorePerLoopIndicesElements);
    OP_LOGD(nodeName, "perCoreLastLoopIndicesElements is %ld", data.perCoreLastLoopIndicesElements);
    OP_LOGD(nodeName, "lastCoreIndicesLoops is %ld", data.lastCoreIndicesLoops);
    OP_LOGD(nodeName, "lastCorePerLoopIndicesElements is %ld", data.lastCorePerLoopIndicesElements);
    OP_LOGD(nodeName, "lastCoreLastLoopIndicesElements is %ld", data.lastCoreLastLoopIndicesElements);
    OP_LOGD(nodeName, "colsLoops is %ld", data.colsLoops);
    OP_LOGD(nodeName, "perLoopCols is %ld", data.perLoopCols);
    OP_LOGD(nodeName, "lastLoopCols is %ld", data.lastLoopCols);
    OP_LOGD(nodeName, "activeNum is %ld", data.activeNum);
}

void PrintMoeInitRoutingV3Arch35TilingData(const MoeInitRoutingV3Arch35TilingData& data, const char *nodeName) {
    OP_LOGD(nodeName, "========== MoeInitRoutingV3Arch35TilingData ==========");
    OP_LOGD(nodeName, "coreNum is %ld", data.coreNum);
    OP_LOGD(nodeName, "BS is %ld", data.n);
    OP_LOGD(nodeName, "H is %ld", data.cols);
    OP_LOGD(nodeName, "topK is %ld", data.k);
    OP_LOGD(nodeName, "expertStart is %ld", data.expertStart);
    OP_LOGD(nodeName, "expertEnd is %ld", data.expertEnd);
    OP_LOGD(nodeName, "actualExpertNum is %ld", data.actualExpertNum);
    OP_LOGD(nodeName, "quantMode is %ld", data.quantMode);
    OP_LOGD(nodeName, "rowIdxType is %ld", data.rowIdxType);
    OP_LOGD(nodeName, "isInputScale is %s", data.isInputScale ? "True" : "False");
    OP_LOGD(nodeName, "isInputOffset is %s", data.isInputOffset ? "True" : "False");
    OP_LOGD(nodeName, "expertNum is %ld", data.expertNum);
    OP_LOGD(nodeName, "expertTokensNumType is %ld", data.expertTokensNumType);
    OP_LOGD(nodeName, "expertTokensNumFlag is %ld", data.expertTokensNumFlag);
    OP_LOGD(nodeName, "gatherFirstFullload is %ld", data.gatherFirstFullload);
    OP_LOGD(nodeName, "epFullload is %ld", data.epFullload);
    OP_LOGD(nodeName, "activeNum is %ld", data.activeNum);
    OP_LOGD(nodeName, "dropPadMode is %ld", data.dropPadMode);
    OP_LOGD(nodeName, "smoothType is %ld", data.smoothType);
    OP_LOGD(nodeName, "InitRouting TilingKey is %ld", data.tilingKey);

    PrintMoeV3Arch35VBSComputeTilingData(data.vbsComputeParamsOp, nodeName);
    PrintMoeV3Arch35VMSMiddleComputeTilingData(data.vmsMiddleComputeParamsOp, nodeName);
    PrintMoeV3Arch35SortOutComputeTilingData(data.sortOutComputeParamsOp, nodeName);
    PrintMoeV3Arch35ExpertTokensCountTilingData(data.expertTokensCountTilingDataOp, nodeName);
    PrintMoeV3Arch35GatherOutComputeTilingData(data.gatherOutComputeParamsOp, nodeName);
}

void PrintMoeDispatchFFNCombineTilingData(const MoeDispatchFFNCombineTilingData* tilingData, const char *nodeName) {
    OP_TILING_CHECK(tilingData == nullptr, VECTOR_INNER_ERR_REPORT_TILING(nodeName, "tilingData is nullptr."), return);

    OP_LOGD(nodeName, "========== MoeDispatchFFNCombineTilingData ==========");
    
    PrintMoeInitRoutingV3Arch35TilingData(tilingData->moeInitRoutingTilingData, nodeName);

    OP_LOGD(nodeName, "BS is %u", tilingData->m);
    OP_LOGD(nodeName, "H is %u", tilingData->k);
    OP_LOGD(nodeName, "N is %u", tilingData->n);

    OP_LOGD(nodeName, "topK is %u", tilingData->topK);
    OP_LOGD(nodeName, "expertPerRank is %u", tilingData->expertPerRank);
    OP_LOGD(nodeName, "groupListType is %u", tilingData->groupListType);

    OP_LOGD(nodeName, "EP is %u", tilingData->EP);
    OP_LOGD(nodeName, "maxOutputSize is %u", tilingData->maxOutputSize);

    OP_LOGD(nodeName, "transX is %s", (tilingData->transX ? "True" : "False"));
    OP_LOGD(nodeName, "transW is %s", (tilingData->transW ? "True" : "False"));
    OP_LOGD(nodeName, "transW2 is %s", (tilingData->transW2 ? "True" : "False"));
}

void printWorkspaceInfo(const struct WorkspaceInfo *info, const char *nodeName) 
{
    OP_LOGD(nodeName, "gmCtx:             %ld\n", info->gmCtx);
    OP_LOGD(nodeName, "ptrA:              %ld\n", info->ptrA);
    OP_LOGD(nodeName, "ptrAScale:         %ld\n", info->ptrAScale);
    OP_LOGD(nodeName, "ptrA2:             %ld\n", info->ptrA2);
    OP_LOGD(nodeName, "ptrA2Scale:        %ld\n", info->ptrA2Scale);
    OP_LOGD(nodeName, "ptrcumsumMM:       %ld\n", info->ptrcumsumMM);
    OP_LOGD(nodeName, "expandedRowIdx:    %ld\n", info->expandedRowIdx);
    OP_LOGD(nodeName, "ptrSumBeforeRank:  %ld\n", info->ptrSumBeforeRank);
    OP_LOGD(nodeName, "ptrFlagList:       %ld\n", info->ptrFlagList);
    OP_LOGD(nodeName, "ptrFlagList2:      %ld\n", info->ptrFlagList2);
    OP_LOGD(nodeName, "workspaceSize:     %ld\n", info->workspaceSize);
}

void printPeermemInfo(const MoeDispatchFFNCombineTilingData* tilingData, const char *nodeName) 
{
    OP_LOGD(nodeName, "========== PeermemInfo ==========");

    int64_t peermemSize = tilingData->m * tilingData->topK *
            (tilingData->k + tilingData->k / MXFP_SCALE_GROUP_NUM) * sizeof(int8_t) + 60 * 1024 +
            tilingData->EP * 128 * sizeof(int32_t) +
            tilingData->m * tilingData->topK * tilingData->k * sizeof(int16_t);
    OP_LOGD(nodeName, "peermemSize: {%ld}\n", peermemSize);

    int64_t offset = PEERMEM_DATA_OFFSET;
    OP_LOGD(nodeName, "ptrA0: {%ld}\n", PEERMEM_DATA_OFFSET);
    offset += tilingData->m * tilingData->topK * (tilingData->k + tilingData->k / MXFP_SCALE_GROUP_NUM) * sizeof(int8_t);
    OP_LOGD(nodeName, "ptrTokenPerExpert: {%ld}\n", offset);
    offset += tilingData->EP * CeilAlign(tilingData->EP * tilingData->expertPerRank, ALIGN_128) * sizeof(int32_t);
    OP_LOGD(nodeName, "ptrD: {%ld}\n", offset);
}

void printAuxiliaryTilingCtx(const MoeDispatchFFNCombineAuxTilingContext* ctx, const uint32_t expertPerRank, const uint32_t EP,
    const std::vector<int64_t>& x_shape, const std::vector<int64_t>& expert_idx_shape, ge::DataType xDtype, const char *nodeName)
{
    OP_LOGD(nodeName, "========== AuxiliaryTilingCtx ==========");
    OP_LOGD(nodeName, "BS is %u", ctx->n);
    OP_LOGD(nodeName, "H is %u", ctx->cols);
    OP_LOGD(nodeName, "topK is %u", ctx->k);
    OP_LOGD(nodeName, "expertPerRank is %u", expertPerRank);
    OP_LOGD(nodeName, "EP is %u", EP);
    OP_LOGD(nodeName, "expert_idx_shape is [%ld, %ld]", expert_idx_shape[0], expert_idx_shape[1]);
    
    OP_LOGD(nodeName, "Dtypes - x is %s, inputXDtypeSize = %ld", Ops::Base::ToString(xDtype).c_str(), ctx->inputXDtypeSize);
    OP_LOGD(nodeName, "Flags - is_input_scale is %s, is_input_offset is %s",
        ctx->isInputScale ? "True" : "False", ctx->isInputOffset ? "True" : "False");

    OP_LOGD(nodeName, "active_num is %u", ctx->activeNum);
    OP_LOGD(nodeName, "expert_capacity is %u", ctx->expertCapacity);
    OP_LOGD(nodeName, "expert_num is %u", ctx->expertNum);
    OP_LOGD(nodeName, "drop_pad_mode is %ld, expert_tokens_num_type is %ld",
        ctx->dropPadMode, ctx->expertTokensNumType);
    OP_LOGD(nodeName, "expert_tokens_num_flag is %s, quant_mode is %ld",
        ctx->expertTokensNumFlag ? "True" : "False", ctx->quantMode);
    OP_LOGD(nodeName, "expert_range is [%ld, %ld]", ctx->expertStart, ctx->expertEnd);
    OP_LOGD(nodeName, "row_idx_type is %u", ctx->rowIdxType);
}

static int64_t GetQuantModeByAttrDispatchOutType(const gert::TilingContext *context, MoeDispatchFFNCombineConfig &config)
{
    auto attrs = context->GetAttrs();
    auto dispatchQuantOutTypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutTypeIndex));
    int64_t dispatchQuantOutType = static_cast<int64_t>(*dispatchQuantOutTypePtr);

    int64_t dispatchQuantMode;
    if (dispatchQuantOutType == DISPATCH_QUANT_OUT_TYPE_E5M2) {
        dispatchQuantMode = DISPATCH_QUANT_MODE_MXFP8_E5M2;
    } else {
        dispatchQuantMode = DISPATCH_QUANT_MODE_MXFP8_E4M3FN;
    }

    return dispatchQuantMode;
}

static uint64_t CalTilingKey(const gert::TilingContext *context, MoeDispatchFFNCombineConfig &config, MoeDispatchFFNCombineTilingData *tilingData, const char *nodeName)
{
    int64_t dispatchQuantMode = GetQuantModeByAttrDispatchOutType(context, config);

    return GET_TPL_TILING_KEY(dispatchQuantMode);
}

static ge::graphStatus CheckinitTilingData(
    const int64_t expert_tokens_num_type, int64_t expert_num, const int64_t drop_pad_mode, const bool expert_tokens_num_flag,
    const int64_t quant_mode, const int64_t row_idx_type, const std::vector<int64_t> expert_range,
    const std::vector<int64_t> x_shape, const std::vector<int64_t> expert_idx_shape,
    const char* nodeName)
{
    OP_TILING_CHECK(
        expert_tokens_num_type != EXPERT_TOKENS_TYPE_COUNT && expert_tokens_num_type != EXPERT_TOKENS_TYPE_KEY_VALUE,
        OP_LOGE(nodeName, "Only support EXPERT_TOKENS_TYPE_COUNT(%ld), EXPERT_TOKENS_TYPE_KEY_VALUE(%ld), but now expert_tokens_num_type is %ld.",
            EXPERT_TOKENS_TYPE_COUNT, EXPERT_TOKENS_TYPE_KEY_VALUE, expert_tokens_num_type),
        return ge::GRAPH_FAILED);

    int64_t maxExpertNum = (expert_tokens_num_type == EXPERT_TOKENS_TYPE_COUNT) ?
                            EXPERT_IDX_MAX : KV_MODE_EXPERT_IDX_MAX;
    OP_TILING_CHECK(
        expert_num <= 0 || expert_num > maxExpertNum,
        OP_LOGE(nodeName, "expert_num out of range, supporting [0, %ld], expert_num is %ld.",
            maxExpertNum, expert_num),
        return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(
        drop_pad_mode != DROP_PAD_MODE_DROPLESS,
        OP_LOGE(nodeName, "Only support DROP_PAD_MODE_DROPLESS(%ld), but now drop_pad_mode is %ld.",
            DROP_PAD_MODE_DROPLESS, drop_pad_mode),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        !expert_tokens_num_flag,
        OP_LOGE(nodeName, "Failed, expert_tokens_num_flag must be true."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        quant_mode != DISPATCH_QUANT_MODE_MXFP8_E5M2 && quant_mode != DISPATCH_QUANT_MODE_MXFP8_E4M3FN,
        OP_LOGE(nodeName, "quant_mode is %ld, only support DISPATCH_QUANT_MODE_MXFP8_E5M2(%d), DISPATCH_QUANT_MODE_MXFP8_E4M3FN(%d), but now is %ld.",
            quant_mode, DISPATCH_QUANT_MODE_MXFP8_E5M2, DISPATCH_QUANT_MODE_MXFP8_E4M3FN, quant_mode),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        row_idx_type != ROW_IDX_SCATTER && row_idx_type != ROW_IDX_GATHER,
        OP_LOGE(nodeName, "Only support ROW_IDX_SCATTER(%ld), ROW_IDX_GATHER(%ld), but now row_idx_type is %ld.",
            ROW_IDX_SCATTER, ROW_IDX_GATHER, row_idx_type),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        expert_range.size() != TWO_DIMS,
        OP_LOGE(nodeName, "expert_range.size() must be 2, but now expert_range.size() is %zu.", expert_range.size()),
        return ge::GRAPH_FAILED);

    int64_t expertStart = expert_range[0];
    int64_t expertEnd = expert_range[1];
    OP_TILING_CHECK(
        expertStart < 0 || expertStart >= expertEnd || expertEnd > expert_num,
        OP_LOGE(nodeName, "expert_num out of range, supporting [%ld, %ld], but now expert_num is %ld.",
            expertStart, expertEnd, expert_num),
        return ge::GRAPH_FAILED);
    
    OP_TILING_CHECK(
        x_shape.size() != TWO_DIMS,
        OP_LOGE(nodeName, "x_shape.size() must be 2, but now x_shape.size() is %zu.", x_shape.size()),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        expert_idx_shape.size() != TWO_DIMS || expert_idx_shape[0] != x_shape[0],
        OP_LOGE(nodeName, "invalid expert_idx_shape, expecting expert_idx_shape.size() = 2, expert_idx_shape[0] is %ld.", x_shape[0]),
        return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetinitTilingData(
    MoeDispatchFFNCombineTilingData &tilingData, MoeInitRoutingV3Arch35TilingData &initTilingData,
    const uint32_t aivNum, const int64_t quantMode, const char *nodeName)
{
    // Get Param
    auto bs = tilingData.m;
    auto h = tilingData.k;
    auto topK = tilingData.topK;
    auto expertPerRank = tilingData.expertPerRank;
    auto EP = tilingData.EP;

    int64_t expert_tokens_num_type = EXPERT_TOKENS_TYPE_COUNT;
    int64_t expert_num = EP * expertPerRank;
    int64_t drop_pad_mode = DROP_PAD_MODE_DROPLESS;
    bool expert_tokens_num_flag = true;
    int64_t quant_mode = quantMode;
    int64_t row_idx_type = ROW_IDX_SCATTER;

    const std::vector<int64_t> expert_range = {0, expert_num};
    std::vector<int64_t> x_shape = {bs, h};
    std::vector<int64_t> expert_idx_shape = {bs, topK};

    bool is_input_scale = false;
    bool is_input_offset = false;
    
    // Check Param
    OP_TILING_CHECK(
        CheckinitTilingData(
            expert_tokens_num_type, expert_num, drop_pad_mode, expert_tokens_num_flag, quant_mode,
            row_idx_type, expert_range, x_shape, expert_idx_shape, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "CheckinitTilingData failed."), return ge::GRAPH_FAILED);

    // Set Param
    initTilingData.expertTokensNumType = expert_tokens_num_type;
    initTilingData.expertNum = expert_num;
    initTilingData.dropPadMode = drop_pad_mode;
    initTilingData.expertTokensNumFlag = expert_tokens_num_flag ? 1 : 0;
    initTilingData.quantMode = OpQuantModeToInitRoutingQuantMode(quant_mode);
    initTilingData.rowIdxType = row_idx_type;

    initTilingData.expertStart = expert_range[0];
    initTilingData.expertEnd = expert_range[1];
    initTilingData.actualExpertNum = expert_range[1] - expert_range[0];

    initTilingData.n = bs;
    initTilingData.cols = h;
    initTilingData.k = topK;
    initTilingData.isInputScale = is_input_scale ? 1 : 0;
    initTilingData.isInputOffset = is_input_offset ? 1 : 0;

    initTilingData.coreNum = aivNum;

    // Reserved
    initTilingData.gatherFirstFullload = 0;
    initTilingData.activeNum = initTilingData.n * initTilingData.k;
    initTilingData.epFullload = 0;  // unsupported
    initTilingData.smoothType = 0;  // unsupported

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetAuxiliaryTilingCtx(
    MoeInitRoutingV3Arch35TilingData &initTilingData, MoeDispatchFFNCombineAuxTilingContext &ctx,
    const char* nodeName, const uint32_t aivNum, const uint64_t ubSize, ge::DataType xDtype)
{
    int64_t inputXDtypeSize = ge::GetSizeByDataType(xDtype);
    OP_TILING_CHECK(
        inputXDtypeSize < 0,
        OP_LOGE(nodeName, "invalid x dtype(%s).", Ops::Base::ToString(xDtype).c_str()),
        return ge::GRAPH_FAILED);

    ctx.aivCoreNum = aivNum;
    ctx.totalUbSize = ubSize;
    ctx.availUbSize = ubSize - SIMT_DCACHE_SIZE;

    ctx.n = initTilingData.n;
    ctx.cols = initTilingData.cols;
    ctx.k = initTilingData.k;
    ctx.inputXDtypeSize = inputXDtypeSize;
    ctx.isInputScale = initTilingData.isInputScale;
    ctx.isInputOffset = initTilingData.isInputOffset;

    ctx.activeNum = initTilingData.activeNum;
    ctx.expertCapacity = DISABLE_EXPERT_CAPACITY;
    ctx.expertNum = initTilingData.expertNum;
    ctx.dropPadMode = initTilingData.dropPadMode;
    ctx.expertTokensNumType = initTilingData.expertTokensNumType;
    ctx.expertTokensNumFlag = initTilingData.expertTokensNumFlag;
    ctx.quantMode = initTilingData.quantMode;
    
    ctx.expertStart = initTilingData.expertStart;
    ctx.expertEnd = initTilingData.expertEnd;
    ctx.rowIdxType = initTilingData.rowIdxType;
    ctx.totalLength = initTilingData.n * initTilingData.k;

    ctx.sortLoopMaxElement = ctx.availUbSize / (NUM_FOUR * NUM_TWO * NUM_FOUR) / SORT32_ALIGN_ELEMENT * SORT32_ALIGN_ELEMENT;
    ctx.sortLoopMaxElement = std::min(ctx.sortLoopMaxElement, SORT_API_MAX_ELEM);

    return ge::GRAPH_SUCCESS;
}

inline static void Tiling4VBSOneCore(MoeV3Arch35VBSComputeTilingData *vbsTiling,
                                     const MoeDispatchFFNCombineAuxTilingContext& ctx) {
    vbsTiling->needCoreNum = 1;
    vbsTiling->perCoreElements = ctx.totalLength;
    vbsTiling->perCoreLoops = CeilDiv(ctx.totalLength, ctx.sortLoopMaxElement);
    vbsTiling->perCorePerLoopElements = std::min(ctx.totalLength, ctx.sortLoopMaxElement);
    vbsTiling->perCoreLastLoopElements = vbsTiling->perCoreElements;
    vbsTiling->lastCoreElements = vbsTiling->perCoreElements;
    vbsTiling->lastCoreLoops = 1;
    vbsTiling->lastCorePerLoopElements = vbsTiling->perCoreElements;
    vbsTiling->lastCoreLastLoopElements = vbsTiling->perCoreElements;
}

inline static void Tiling4VBSMultiCore(MoeV3Arch35VBSComputeTilingData *vbsTiling,
                                        MoeDispatchFFNCombineAuxTilingContext& ctx) {
    int64_t needCoreNum = CeilDiv(ctx.totalLength, ctx.sortLoopMaxElement);
    needCoreNum = static_cast<int64_t>(std::pow(4, CeilLog4(needCoreNum)));
    needCoreNum = std::min(needCoreNum, ctx.aivCoreNum);

    if (needCoreNum == 0) {
        needCoreNum = 1;
    }
    int64_t perCoreElements = (needCoreNum == 0) ? 0 : (ctx.totalLength / needCoreNum);
    int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
    int64_t lastCoreElement = ctx.totalLength - (needCoreNum - 1) * alineFloorPerCoreElements;
    int64_t alineCeilPerCoreElements = perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
    if (lastCoreElement > alineCeilPerCoreElements) {
        perCoreElements = alineCeilPerCoreElements;
        needCoreNum = CeilDiv(ctx.totalLength, perCoreElements);
    } else {
        perCoreElements = alineFloorPerCoreElements;
    }

    vbsTiling->needCoreNum = needCoreNum;
    do {
        vbsTiling->perCoreElements = perCoreElements;
        vbsTiling->perCoreLoops = CeilDiv(vbsTiling->perCoreElements, ctx.sortLoopMaxElement);
        vbsTiling->perCorePerLoopElements = std::min(vbsTiling->perCoreElements, ctx.sortLoopMaxElement);

        vbsTiling->perCoreLastLoopElements =
            vbsTiling->perCoreElements - (vbsTiling->perCoreLoops - 1) * vbsTiling->perCorePerLoopElements;

        vbsTiling->lastCoreElements = ctx.totalLength - (vbsTiling->needCoreNum - 1) * vbsTiling->perCoreElements;
        vbsTiling->lastCoreLoops = vbsTiling->perCoreLoops;
        int64_t lastCorePerLoopElements =
            CeilAlign(CeilDiv(vbsTiling->lastCoreElements, vbsTiling->lastCoreLoops), SORT32_ALIGN_ELEMENT);
        vbsTiling->lastCorePerLoopElements = lastCorePerLoopElements;
        vbsTiling->lastCoreLastLoopElements =
            vbsTiling->lastCoreElements - (vbsTiling->lastCoreLoops - 1) * vbsTiling->lastCorePerLoopElements;
        perCoreElements -= SORT32_ALIGN_ELEMENT;
    } while (vbsTiling->lastCoreLastLoopElements <= 0 && perCoreElements > 0);
}

inline static void Tiling4VBSCompute(MoeInitRoutingV3Arch35TilingData *tilingData,
                                     MoeDispatchFFNCombineAuxTilingContext& ctx) {
    auto *vbsTiling = &(tilingData->vbsComputeParamsOp);
    vbsTiling->oneLoopMaxElements = ctx.sortLoopMaxElement;

    if (ctx.totalLength <= ctx.sortLoopMaxElement) {
        ctx.sortMode = 0;
        Tiling4VBSOneCore(vbsTiling, ctx);
    } else {
        ctx.sortMode = 1;
        Tiling4VBSMultiCore(vbsTiling, ctx);
    }
    tilingData->tilingKey = ctx.sortMode * SORT_CORE_TILINGKEY_BASE + 1031000;
}

inline static void Tiling4VMSMiddleCompute(MoeInitRoutingV3Arch35TilingData *tilingData) {
    auto *vbsTiling = &(tilingData->vbsComputeParamsOp);
    auto *vmsMiddleTiling = &(tilingData->vmsMiddleComputeParamsOp);

    if (vbsTiling->needCoreNum <= MRG_LIST_NUM) {
        vmsMiddleTiling->needCoreNum = 0;
        return;
    }
    int64_t needCoreNum = CeilDiv(vbsTiling->needCoreNum, MRG_LIST_NUM);
    vmsMiddleTiling->needCoreNum = needCoreNum;
}

inline static void Tiling4SortOutCompute(MoeInitRoutingV3Arch35TilingData *tilingData) {
    auto *sortOutTiling = &(tilingData->sortOutComputeParamsOp);
    sortOutTiling->oneLoopMaxElements = MRG_SORT_API_MAX_ELEM;
}

inline static void Tiling4ExpertTokensCountCompute(MoeInitRoutingV3Arch35TilingData *tilingData,
                                                    const MoeDispatchFFNCombineAuxTilingContext& ctx) {
    auto *tokensCountTiling = &(tilingData->expertTokensCountTilingDataOp);
    int64_t totalElements = tilingData->n * tilingData->k;
    int64_t perCoreElements = CeilDiv(totalElements, ctx.aivCoreNum);
    int64_t needCoreNum = CeilDiv(totalElements, perCoreElements);
    int64_t lastCoreElements = totalElements - (needCoreNum - 1) * perCoreElements;

    tokensCountTiling->needCoreNum = needCoreNum;
    tokensCountTiling->perCoreElements = perCoreElements;
    tokensCountTiling->lastCoreElements = lastCoreElements;

    int64_t expertNumElement = (tilingData->expertTokensNumType != EXPERT_TOKENS_TYPE_KEY_VALUE) ?
                                   tilingData->actualExpertNum :
                                   (tilingData->actualExpertNum + 1) * DIM_TWO;
    int64_t maxElementsPerLoop =
        (ctx.availUbSize -
         CeilAlign(expertNumElement, UB_BLOCK_SIZE) *
             (static_cast<int64_t>(sizeof(int32_t)) * NUM_TWO + static_cast<int64_t>(sizeof(int64_t))) -
         UB_BLOCK_SIZE) /
        static_cast<int64_t>(sizeof(int32_t));

    int64_t perCoreLoops = CeilDiv(perCoreElements, maxElementsPerLoop);
    int64_t perCorePerLoopElements = CeilDiv(perCoreElements, perCoreLoops);
    int64_t perCoreLastLoopElements = perCoreElements - (perCoreLoops - 1) * perCorePerLoopElements;
    tokensCountTiling->perCoreLoops = perCoreLoops;
    tokensCountTiling->perCorePerLoopElements = perCorePerLoopElements;
    tokensCountTiling->perCoreLastLoopElements = perCoreLastLoopElements;

    int64_t lastCoreLoops = CeilDiv(lastCoreElements, maxElementsPerLoop);
    int64_t lastCorePerLoopElements = CeilDiv(lastCoreElements, lastCoreLoops);
    int64_t lastCoreLastLoopElements = lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoopElements;
    tokensCountTiling->lastCoreLoops = lastCoreLoops;
    tokensCountTiling->lastCorePerLoopElements = lastCorePerLoopElements;
    tokensCountTiling->lastCoreLastLoopElements = lastCoreLastLoopElements;
}

inline static int64_t CalcMaxRowIdxPerLoopMxQuant(int64_t perLoopCols, int64_t inputXDtypeSize,
                                                    int64_t availUbSize) {
    int64_t xInSize = AlignBytes(perLoopCols, inputXDtypeSize) + AlignBytes(perLoopCols, sizeof(int8_t));
    int64_t scaleSize = 2 * AlignBytes(perLoopCols / MX_QUANT_BLOCK_SIZE, inputXDtypeSize) +
                        AlignBytes(perLoopCols / MX_QUANT_BLOCK_SIZE, sizeof(int8_t));
    int64_t xOutSize = Align(perLoopCols / 4, sizeof(int8_t)) * 4;
    return (availUbSize - (xInSize + scaleSize + xOutSize)) / static_cast<int64_t>(sizeof(int32_t));
}

inline static void Tiling4GatherOutMxQuant(MoeInitRoutingV3Arch35TilingData *tilingData,
                                           const MoeDispatchFFNCombineAuxTilingContext& ctx) {
    auto *gatherOutTiling = &(tilingData->gatherOutComputeParamsOp);
    int64_t perCoreIndicesElements = CeilDiv(ctx.totalLength, ctx.aivCoreNum);
    if (perCoreIndicesElements <= 0) {
        gatherOutTiling->needCoreNum = 0;
        return;
    }
    int64_t needCoreNum = CeilDiv(ctx.totalLength, perCoreIndicesElements);
    int64_t lastCoreIndicesElements = ctx.totalLength - (needCoreNum - 1) * perCoreIndicesElements;

    int64_t perLoopCols = CeilAlign(tilingData->cols, MX_QUANT_BLOCK_SIZE);
    int64_t perLoopMaxIndicesElements = CalcMaxRowIdxPerLoopMxQuant(perLoopCols, ctx.inputXDtypeSize, ctx.availUbSize);
    while (perLoopMaxIndicesElements <= 0) {
        perLoopCols = CeilAlign(CeilDiv(perLoopCols, NUM_TWO), MX_QUANT_BLOCK_SIZE);
        perLoopMaxIndicesElements = CalcMaxRowIdxPerLoopMxQuant(perLoopCols, ctx.inputXDtypeSize, ctx.availUbSize);
    }
    int64_t colsLoops = CeilDiv(tilingData->cols, perLoopCols);
    int64_t lastLoopCols = tilingData->cols - (colsLoops - 1) * perLoopCols;
    gatherOutTiling->needCoreNum = needCoreNum;
    gatherOutTiling->perCoreIndicesElements = perCoreIndicesElements;
    gatherOutTiling->lastCoreIndicesElements = lastCoreIndicesElements;
    gatherOutTiling->colsLoops = colsLoops;
    gatherOutTiling->perLoopCols = perLoopCols;
    gatherOutTiling->lastLoopCols = lastLoopCols;

    int64_t perCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, perCoreIndicesElements);
    int64_t perCoreIndicesLoops = CeilDiv(perCoreIndicesElements, perCorePerLoopIndicesElements);
    int64_t perCoreLastLoopIndicesElements =
        perCoreIndicesElements - (perCoreIndicesLoops - 1) * perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreIndicesLoops = perCoreIndicesLoops;
    gatherOutTiling->perCorePerLoopIndicesElements = perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreLastLoopIndicesElements = perCoreLastLoopIndicesElements;

    int64_t lastCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, lastCoreIndicesElements);
    int64_t lastCoreIndicesLoops = CeilDiv(lastCoreIndicesElements, lastCorePerLoopIndicesElements);
    int64_t lastCoreLastLoopIndicesElements =
        lastCoreIndicesElements - (lastCoreIndicesLoops - 1) * lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreIndicesLoops = lastCoreIndicesLoops;
    gatherOutTiling->lastCorePerLoopIndicesElements = lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreLastLoopIndicesElements = lastCoreLastLoopIndicesElements;
}

MoeInitRoutingV3Arch35TilingData ComputeMoeInitRoutingV3Tiling(
    MoeInitRoutingV3Arch35TilingData &tilingData, MoeDispatchFFNCombineAuxTilingContext &ctx) {
    Tiling4VBSCompute(&tilingData, ctx);
    Tiling4VMSMiddleCompute(&tilingData);
    Tiling4SortOutCompute(&tilingData);
    Tiling4ExpertTokensCountCompute(&tilingData, ctx);
    Tiling4GatherOutMxQuant(&tilingData, ctx); // 目前只支持 fp8_e4m3, fp8_e5m2
    return tilingData;
}

static ge::graphStatus GetMoeInitRoutingV3Tiling(MoeDispatchFFNCombineTilingData &tilingData,
    const uint32_t aivNum, const uint64_t ubSize, const int64_t quantMode, ge::DataType xDtype, const char *nodeName)
{
    MoeInitRoutingV3Arch35TilingData initTilingData;
    OP_TILING_CHECK(
        CheckAndSetinitTilingData(tilingData, initTilingData, aivNum, quantMode, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "SetinitTilingData failed."), return ge::GRAPH_FAILED);

    MoeDispatchFFNCombineAuxTilingContext ctx;
    OP_TILING_CHECK(
        SetAuxiliaryTilingCtx(initTilingData, ctx, nodeName, aivNum, ubSize, xDtype) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "SetAuxiliaryTilingCtx failed."), return ge::GRAPH_FAILED);

    printAuxiliaryTilingCtx(&ctx, tilingData.expertPerRank, tilingData.EP,
        {tilingData.m, tilingData.k}, {tilingData.m, tilingData.topK}, xDtype, nodeName);

    auto ss_tiling = ComputeMoeInitRoutingV3Tiling(initTilingData, ctx);

    int64_t leastCoreNum = std::max(
        std::max(initTilingData.vbsComputeParamsOp.needCoreNum, initTilingData.vmsMiddleComputeParamsOp.needCoreNum), 
        std::max(initTilingData.expertTokensCountTilingDataOp.needCoreNum, initTilingData.gatherOutComputeParamsOp.needCoreNum));
    bool isCoreNumValid = initTilingData.coreNum >= leastCoreNum;
    OP_TILING_CHECK(
        !isCoreNumValid,
        OP_LOGE(nodeName, "aivNum(%ld) < leastCoreNum(%ld).", initTilingData.coreNum, leastCoreNum),
        return ge::GRAPH_FAILED);
    
    tilingData.moeInitRoutingTilingData = ss_tiling;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrPtrNullptr(const gert::TilingContext *context, MoeDispatchFFNCombineConfig &config, const char *nodeName)
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

    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE(nodeName, "moeExpertNumPtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "epWorldSizePtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(cclBufferSizePtr == nullptr || *cclBufferSizePtr < 0, OP_LOGE(nodeName, "cclBufferSizePtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(maxRecvTokenNumPtr == nullptr, OP_LOGE(nodeName, "maxRecvTokenNumPtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(dispatchQuantModePtr == nullptr, OP_LOGE(nodeName, "dispatchQuantModePtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(dispatchQuantOutTypePtr == nullptr, OP_LOGE(nodeName, "dispatchQuantOutTypePtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(combineQuantModePtr == nullptr, OP_LOGE(nodeName, "combineQuantModePtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commAlgPtr == nullptr, OP_LOGE(nodeName, "commAlgPtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(globalBs == nullptr, OP_LOGE(nodeName, "globalBs is nullptr."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrParams(const gert::TilingContext *context, MoeDispatchFFNCombineConfig &config, const char *nodeName)
{
    auto attrs = context->GetAttrs();

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    int64_t epWorldSize = static_cast<int64_t>(*epWorldSizePtr);
    OP_TILING_CHECK(epWorldSize < MIN_EP_WORLD_SIZE || epWorldSize > MAX_EP_WORLD_SIZE,
        OP_LOGE(nodeName,
        "epWorldSize should in [%ld, %ld], but now epWorldSize is %ld.", MIN_EP_WORLD_SIZE, MAX_EP_WORLD_SIZE, epWorldSize),
        return ge::GRAPH_FAILED);

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    int64_t moeExpertNum = static_cast<int64_t>(*moeExpertNumPtr);
    OP_TILING_CHECK((moeExpertNum < epWorldSize || moeExpertNum > MAX_MOE_EXPERT_NUM) || (moeExpertNum % epWorldSize),
        OP_LOGE(nodeName,
            "moeExpertNum should in [%ld, %ld] and mod(moeExpertNum, epWorldSize) should be zero, but now moeExpertNum is %ld.",
            epWorldSize, MAX_MOE_EXPERT_NUM, moeExpertNum),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *weightOneStorageShape = context->GetInputShape(config.weightOneIndex);
    int64_t expertPerRank = weightOneStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(moeExpertNum != expertPerRank * epWorldSize,
        OP_LOGE(nodeName,
            "moeExpertNum(%ld) should be equal to expertPerRank(%ld) * epWorldSize(%ld) = %ld.",
            moeExpertNum, expertPerRank, epWorldSize, expertPerRank * epWorldSize),
        return ge::GRAPH_FAILED);

    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    int64_t dispatchQuantMode = static_cast<int64_t>(*dispatchQuantModePtr);
    OP_TILING_CHECK(dispatchQuantMode != 4,
        OP_LOGE(nodeName, "Invalid dispatchQuantMode, only support mxfp(4), but now is %ld.", dispatchQuantMode),
        return ge::GRAPH_FAILED);

    auto dispatchQuantOutTypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutTypeIndex));
    int64_t dispatchQuantOutType = static_cast<int64_t>(*dispatchQuantOutTypePtr);
    OP_TILING_CHECK(dispatchQuantOutType != DISPATCH_QUANT_OUT_TYPE_E5M2 && dispatchQuantOutType != DISPATCH_QUANT_OUT_TYPE_E4M3,
        OP_LOGE(nodeName, "Invalid dispatchQuantOutType, only support fp8_e5m2(23) and fp8_e4m3fn(24), but now is %ld.", dispatchQuantOutType),
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
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    if (globalBs != 0) {
        OP_TILING_CHECK(globalBs < 0 || bs * epWorldSize > globalBs || globalBs % epWorldSize != 0,
            OP_LOGE(nodeName, "globalBs(%ld) either be 0 or (maxBs * EP and mod(globalBs, EP(%ld)) == 0).",
                globalBs, epWorldSize),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetAttrParams(const gert::TilingContext *context, MoeDispatchFFNCombineConfig &config,
    MoeDispatchFFNCombineTilingData *tilingData, const char *nodeName, const uint32_t aicNum, int64_t &CclBufferSize)
{
    auto attrs = context->GetAttrs();

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    auto CclBufferSizePtr = attrs->GetAttrPointer<int64_t>((config.attrCclBufferSizeIndex));
    auto maxRecvTokenNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex));

    CclBufferSize = *CclBufferSizePtr;
    tilingData->EP = *epWorldSizePtr;
    tilingData->maxOutputSize = *maxRecvTokenNumPtr;
    tilingData->blockNumPerEP = std::max(static_cast<uint32_t>(1), aicNum / tilingData->EP);
    tilingData->dispatchRows = 2;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrAndSetTilingData(const gert::TilingContext *context, MoeDispatchFFNCombineConfig &config,
    MoeDispatchFFNCombineTilingData *tilingData, const uint32_t aicNum, int64_t &CclBufferSize)
{
    const char *nodeName = context->GetNodeName();

    OP_TILING_CHECK(CheckAttrPtrNullptr(context, config, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "params check nulld failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAttrParams(context, config, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "check attr params failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetAttrParams(context, config, tilingData, nodeName, aicNum, CclBufferSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "set attr params failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(
    gert::TilingContext *context, const char *nodeName, WorkspaceInfo& workspaceInfo, const int64_t CclBufferSize)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, VECTOR_INNER_ERR_REPORT_TILING(nodeName, "workspace is nullptr."),
        return ge::GRAPH_FAILED);

    int64_t workspaceSize = sysWorkspaceSize + workspaceInfo.workspaceSize + 1024 * 1024 * 50;

    OP_TILING_CHECK(
        CclBufferSize < workspaceSize, 
        OP_LOGE(nodeName, "CclBufferSize(%ld) less than workspaceSize(%ld).", CclBufferSize, workspaceSize),
        return ge::GRAPH_FAILED);

    workspace[0] = workspaceSize;

    OP_LOGD(nodeName, "sysWorkspaceSize: %u \n", sysWorkspaceSize);
    OP_LOGD(nodeName, "Moe_Dispatch_FFN_Combine_tiling workspaceSize: %ld \n", workspaceSize);

    return ge::GRAPH_SUCCESS;
}

static bool CheckTensorPtrNullptr(const gert::TilingContext *context, MoeDispatchFFNCombineConfig &config, const char *nodeName)
{
    auto contextDesc = context->GetInputDesc(config.contextIndex);
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);
    auto weightOneDesc = context->GetInputDesc(config.weightOneIndex);
    auto weightTwoDesc = context->GetInputDesc(config.weightTwoIndex);
    auto weightScalesOneDesc = context->GetOptionalInputDesc(config.weightScalesOneIndex);
    auto weightScalesTwoDesc = context->GetOptionalInputDesc(config.weightScalesTwoIndex);

    OP_TILING_CHECK(contextDesc == nullptr, OP_LOGE(nodeName, "contextDesc is null."), return false);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    OP_TILING_CHECK(topkIdsDesc == nullptr, OP_LOGE(nodeName, "topkIdsDesc is null."), return false);
    OP_TILING_CHECK(topkWeightsDesc == nullptr, OP_LOGE(nodeName, "topkWeightsDesc is null."), return false);
    OP_TILING_CHECK(weightOneDesc == nullptr, OP_LOGE(nodeName, "weightOneDesc is null."), return false);
    OP_TILING_CHECK(weightTwoDesc == nullptr, OP_LOGE(nodeName, "weightTwoDesc is null."), return false);
    OP_TILING_CHECK(weightScalesOneDesc == nullptr, OP_LOGE(nodeName, "weightScalesOneDesc is null."), return false);
    OP_TILING_CHECK(weightScalesTwoDesc == nullptr, OP_LOGE(nodeName, "weightScalesTwoDesc is null."), return false);

    return true; 
}

static bool CheckWeightTensorDim(const gert::TilingContext *context, MoeDispatchFFNCombineConfig &config, const char *nodeName)
{
    const gert::StorageShape *weightOneStorageShape = context->GetInputShape(config.weightOneIndex);
    OP_TILING_CHECK(weightOneStorageShape == nullptr, OP_LOGE(nodeName, "weightOneStorageShape is null."), return false);
    OP_TILING_CHECK(weightOneStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
        OP_LOGE(nodeName, "weightOneStorageShape dims must be 3, but current dim num is %lu.",
        weightOneStorageShape->GetStorageShape().GetDimNum()), return false);
    const int64_t weightOneDim0 = weightOneStorageShape->GetStorageShape().GetDim(0);
    const int64_t weightOneDim1 = weightOneStorageShape->GetStorageShape().GetDim(1);
    const int64_t weightOneDim2 = weightOneStorageShape->GetStorageShape().GetDim(2);
    OP_LOGD(nodeName, "weightOne dim0 = %ld", weightOneDim0);
    OP_LOGD(nodeName, "weightOne dim1 = %ld", weightOneDim1);
    OP_LOGD(nodeName, "weightOne dim2 = %ld", weightOneDim2);

    const gert::StorageShape *weightTwoStorageShape = context->GetInputShape(config.weightTwoIndex);
    OP_TILING_CHECK(weightTwoStorageShape == nullptr, OP_LOGE(nodeName, "weightTwoStorageShape is null."), return false);
    OP_TILING_CHECK(weightTwoStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
        OP_LOGE(nodeName, "weightTwoStorageShape dims must be 3, but current dim num is %lu.",
        weightTwoStorageShape->GetStorageShape().GetDimNum()), return false);
    const int64_t weightTwoDim0 = weightTwoStorageShape->GetStorageShape().GetDim(0);
    const int64_t weightTwoDim1 = weightTwoStorageShape->GetStorageShape().GetDim(1);
    const int64_t weightTwoDim2 = weightTwoStorageShape->GetStorageShape().GetDim(2);
    OP_LOGD(nodeName, "weightTwo dim0 = %ld", weightTwoDim0);
    OP_LOGD(nodeName, "weightTwo dim1 = %ld", weightTwoDim1);
    OP_LOGD(nodeName, "weightTwo dim2 = %ld", weightTwoDim2);

    OP_TILING_CHECK(weightOneDim0 != weightTwoDim0,
        OP_LOGE(nodeName, "weightOneDim0(%ld) and weightTwoDim0(%ld) should be equal.",
            weightOneDim0, weightTwoDim0),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(weightOneDim1 != weightTwoDim2,
        OP_LOGE(nodeName, "weightOneDim1(%ld) and weightTwoDim2(%ld) should be equal.",
            weightOneDim1, weightTwoDim2),
        return ge::GRAPH_FAILED);

    return true;
}

static bool CheckOptionalTensorDim(const gert::TilingContext *context, MoeDispatchFFNCombineConfig &config, const char *nodeName)
{
    const gert::StorageShape *weightScalesOneStorageShape = context->GetOptionalInputShape(config.weightScalesOneIndex);
    OP_TILING_CHECK(weightScalesOneStorageShape == nullptr, OP_LOGE(nodeName, "weightScalesOneStorageShape is null."), return false);
    OP_TILING_CHECK(weightScalesOneStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
        OP_LOGE(nodeName, "weightScalesOneStorageShape dims must be 3, but current dim num is %lu.",
        weightScalesOneStorageShape->GetStorageShape().GetDimNum()), return false);
    const int64_t weightScalesOneDim0 = weightScalesOneStorageShape->GetStorageShape().GetDim(0);
    const int64_t weightScalesOneDim1 = weightScalesOneStorageShape->GetStorageShape().GetDim(1);
    const int64_t weightScalesOneDim2 = weightScalesOneStorageShape->GetStorageShape().GetDim(2);
    OP_LOGD(nodeName, "weightScalesOne dim0 = %ld", weightScalesOneDim0);
    OP_LOGD(nodeName, "weightScalesOne dim1 = %ld", weightScalesOneDim1);
    OP_LOGD(nodeName, "weightScalesOne dim2 = %ld", weightScalesOneDim2);

    const gert::StorageShape *weightScalesTwoStorageShape = context->GetOptionalInputShape(config.weightScalesTwoIndex);
    OP_TILING_CHECK(weightScalesTwoStorageShape == nullptr, OP_LOGE(nodeName, "weightScalesTwoStorageShape is null."), return false);
    OP_TILING_CHECK(weightScalesTwoStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
        OP_LOGE(nodeName, "weightScalesTwoStorageShape dims must be 3, but current dim num is %lu.",
        weightScalesTwoStorageShape->GetStorageShape().GetDimNum()), return false);
    const int64_t weightScalesTwoDim0 = weightScalesTwoStorageShape->GetStorageShape().GetDim(0);
    const int64_t weightScalesTwoDim1 = weightScalesTwoStorageShape->GetStorageShape().GetDim(1);
    const int64_t weightScalesTwoDim2 = weightScalesTwoStorageShape->GetStorageShape().GetDim(2);
    OP_LOGD(nodeName, "weightScalesTwo dim0 = %ld", weightScalesTwoDim0);
    OP_LOGD(nodeName, "weightScalesTwo dim1 = %ld", weightScalesTwoDim1);
    OP_LOGD(nodeName, "weightScalesTwo dim2 = %ld", weightScalesTwoDim2);

    OP_TILING_CHECK(weightScalesOneDim0 != weightScalesTwoDim0,
        OP_LOGE(nodeName, "weightScalesOneDim0(%ld) and weightScalesTwoDim0(%ld) should be equal.",
            weightScalesOneDim0, weightScalesTwoDim0),
        return ge::GRAPH_FAILED);

    return true;
}

static bool CheckTensorDim(const gert::TilingContext *context, MoeDispatchFFNCombineConfig &config, const char *nodeName)
{
    const gert::StorageShape *contextStorageShape = context->GetInputShape(config.contextIndex);
    OP_TILING_CHECK(contextStorageShape == nullptr, OP_LOGE(nodeName, "contextShape is null."), return false);
    OP_TILING_CHECK(contextStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "contextShape dims must be 1, but current dim num is %lu.",
        contextStorageShape->GetStorageShape().GetDimNum()), return false);
    int64_t contextDim0 = contextStorageShape->GetStorageShape().GetDim(0);
    OP_LOGD(nodeName, "context dim0 = %ld", contextDim0);

    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "xShape is null."), return false);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "xShape dims must be 2, but current dim num is %lu.",
        xStorageShape->GetStorageShape().GetDimNum()), return false);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "x dim0 = %ld", xDim0);
    OP_LOGD(nodeName, "x dim1 = %ld", xDim1);

    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    OP_TILING_CHECK(topkIdsStorageShape == nullptr, OP_LOGE(nodeName, "topkIdsStorageShape is null."), return false);
    OP_TILING_CHECK(topkIdsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "topkIdsStorageShape dims must be 2, but current dim num is %lu.",
        topkIdsStorageShape->GetStorageShape().GetDimNum()), return false);
    const int64_t topkIdsDim0 = topkIdsStorageShape->GetStorageShape().GetDim(0);
    const int64_t topkIdsDim1 = topkIdsStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "topkIds dim0 = %ld", topkIdsDim0);
    OP_LOGD(nodeName, "topkIds dim1 = %ld", topkIdsDim1);

    const gert::StorageShape *topkWeightsStorageShape = context->GetInputShape(config.topkWeightsIndex);
    OP_TILING_CHECK(topkWeightsStorageShape == nullptr, OP_LOGE(nodeName, "topkWeightsStorageShape is null."), return false);
    OP_TILING_CHECK(topkWeightsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "topkWeightsStorageShape dims must be 2, but current dim num is %lu.",
        topkWeightsStorageShape->GetStorageShape().GetDimNum()), return false);
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

    OP_TILING_CHECK(!CheckWeightTensorDim(context, config, nodeName),
        OP_LOGE(nodeName, "weight params shape is invalid."), return false);

    OP_TILING_CHECK(!CheckOptionalTensorDim(context, config, nodeName),
        OP_LOGE(nodeName, "optional params shape is invalid."), return false);

    return true;
}

static bool CheckTensorDataType(const gert::TilingContext *context, MoeDispatchFFNCombineConfig &config, const char *nodeName)
{
    auto contextDesc = context->GetInputDesc(config.contextIndex);
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);
    auto weightOneDesc = context->GetInputDesc(config.weightOneIndex);
    auto weightTwoDesc = context->GetInputDesc(config.weightTwoIndex);
    auto weightScalesOneDesc = context->GetOptionalInputDesc(config.weightScalesOneIndex);
    auto weightScalesTwoDesc = context->GetOptionalInputDesc(config.weightScalesTwoIndex);

    OP_TILING_CHECK(contextDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "context dataType is invalid, dataType should be DT_INT32, but is %s.",
        Ops::Base::ToString(contextDesc->GetDataType()).c_str()), return false);

    OP_TILING_CHECK(xDesc->GetDataType() != ge::DT_BF16,
        OP_LOGE(nodeName, "x dataType is invalid, dataType should be DT_BF16, but is %s.",
        Ops::Base::ToString(xDesc->GetDataType()).c_str()), return false);

    OP_TILING_CHECK(topkIdsDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "topkIds dataType is invalid, dataType should be DT_INT32, but is %s.",
        Ops::Base::ToString(topkIdsDesc->GetDataType()).c_str()), return false);

    OP_TILING_CHECK(((topkWeightsDesc->GetDataType() != ge::DT_BF16) && (topkWeightsDesc->GetDataType() != ge::DT_FLOAT)),
        OP_LOGE(nodeName, "topkWeights dataType is invalid, dataType should be DT_FLOAT or DT_BF16, but is %s.",
        Ops::Base::ToString(topkWeightsDesc->GetDataType()).c_str()), return false);

    OP_TILING_CHECK(((weightOneDesc->GetDataType() != ge::DT_FLOAT8_E5M2) && (weightOneDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN)),
        OP_LOGE(nodeName, "weightOne dataType is invalid, dataType should be DT_FLOAT8_E5M2 or DT_FLOAT8_E4M3FN, but is %s.",
        Ops::Base::ToString(weightOneDesc->GetDataType()).c_str()), return false);
    
    OP_TILING_CHECK(((weightTwoDesc->GetDataType() != ge::DT_FLOAT8_E5M2) && (weightTwoDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN)),
        OP_LOGE(nodeName, "weightTwo dataType is invalid, dataType should be DT_FLOAT8_E5M2 or DT_FLOAT8_E4M3FN, but is %s.",
        Ops::Base::ToString(weightTwoDesc->GetDataType()).c_str()), return false);

    OP_TILING_CHECK(weightScalesOneDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
        OP_LOGE(nodeName, "weightScalesOne dataType is invalid, dataType should be DT_FLOAT8_E8M0, but is %s.",
        Ops::Base::ToString(weightScalesOneDesc->GetDataType()).c_str()), return false);

    OP_TILING_CHECK(weightScalesTwoDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
        OP_LOGE(nodeName, "weightScalesTwo dataType is invalid, dataType should be DT_FLOAT8_E8M0, but is %s.",
        Ops::Base::ToString(weightScalesTwoDesc->GetDataType()).c_str()), return false);
    
    return true;
}

static bool CheckTensorFormat(const gert::TilingContext *context, MoeDispatchFFNCombineConfig &config, const char *nodeName)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);
    auto weightOneDesc = context->GetInputDesc(config.weightOneIndex);
    auto weightTwoDesc = context->GetInputDesc(config.weightTwoIndex);
    auto weightScalesOneDesc = context->GetOptionalInputDesc(config.weightScalesOneIndex);
    auto weightScalesTwoDesc = context->GetOptionalInputDesc(config.weightScalesTwoIndex);

    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) != ge::FORMAT_ND,
        OP_LOGE(nodeName, "x format is invalid."), return false);

    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(topkIdsDesc->GetStorageFormat())) != ge::FORMAT_ND,
        OP_LOGE(nodeName, "topkIds format is invalid."), return false);
    
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(topkWeightsDesc->GetStorageFormat())) != ge::FORMAT_ND,
        OP_LOGE(nodeName, "topkWeights format is invalid."), return false);
    
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(weightOneDesc->GetStorageFormat())) != ge::FORMAT_ND,
        OP_LOGE(nodeName, "weightOne format is invalid."), return false);
    
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(weightTwoDesc->GetStorageFormat())) != ge::FORMAT_ND,
        OP_LOGE(nodeName, "weightTwo format is invalid."), return false);
    
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(weightScalesOneDesc->GetStorageFormat())) != ge::FORMAT_ND,
        OP_LOGE(nodeName, "weightScalesOne format is invalid."), return false);
    
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(weightScalesTwoDesc->GetStorageFormat())) != ge::FORMAT_ND,
        OP_LOGE(nodeName, "weightScalesTwo format is invalid."), return false);
    
    return true;
}

static ge::graphStatus TilingCheckMoeDispatchFFNCombine(const gert::TilingContext *context, MoeDispatchFFNCombineConfig &config, const char *nodeName)
{
    OP_TILING_CHECK(!CheckTensorPtrNullptr(context, config, nodeName),
        OP_LOGE(nodeName, "params check nulld failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(!CheckTensorDim(context, config, nodeName),
        OP_LOGE(nodeName, "params shape is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(!CheckTensorDataType(context, config, nodeName),
        OP_LOGE(nodeName, "params dataType is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(!CheckTensorFormat(context, config, nodeName),
        OP_LOGE(nodeName, "params format is invalid."), return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputParam(const gert::TilingContext *context, MoeDispatchFFNCombineConfig &config, const char *nodeName)
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

    const gert::StorageShape *weightOneStorageShape = context->GetInputShape(config.weightOneIndex);
    int64_t weightOneDim0 = weightOneStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(weightOneDim0 < MIN_EXPERT_PER_RANK || weightOneDim0 > MAX_EXPERT_PER_RANK,
        OP_LOGE(nodeName, "expertPerRank should in [%ld, %ld], but now expertPerRank is %ld.", MIN_EXPERT_PER_RANK, MAX_EXPERT_PER_RANK, weightOneDim0),
        return ge::GRAPH_FAILED);

    int64_t weightOneDim2 = weightOneStorageShape->GetStorageShape().GetDim(2);
    OP_TILING_CHECK(weightOneDim2 != 1LL * N_BASE && weightOneDim2 != 2LL * N_BASE && weightOneDim2 != 3LL * N_BASE &&
                    weightOneDim2 != 4LL * N_BASE && weightOneDim2 != 7LL * N_BASE,
        OP_LOGE(nodeName, "N only support 1k/2k/3k/4k/7k, but now N is %ld.", weightOneDim2),
        return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetInputParam(const gert::TilingContext *context, 
    MoeDispatchFFNCombineTilingData *tilingData, MoeDispatchFFNCombineConfig &config)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    int64_t BS = xStorageShape->GetStorageShape().GetDim(0);
    int64_t H = xStorageShape->GetStorageShape().GetDim(1);

    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    int64_t topK = topkIdsStorageShape->GetStorageShape().GetDim(1);

    const gert::StorageShape *weightOneStorageShape = context->GetInputShape(config.weightOneIndex);
    int64_t expertPerRank = weightOneStorageShape->GetStorageShape().GetDim(0);
    int64_t N = weightOneStorageShape->GetStorageShape().GetDim(2);

    tilingData->m = static_cast<uint32_t>(BS);
    tilingData->k = static_cast<uint32_t>(H);
    tilingData->n = static_cast<uint32_t>(N);;
    tilingData->topK = static_cast<uint32_t>(topK);
    tilingData->expertPerRank = static_cast<uint32_t>(expertPerRank);
    tilingData->groupListType = 1;

    tilingData->transX = false;
    tilingData->transW = true;
    tilingData->transW2 = true;
    
    tilingData->m0 = BS < NUM_256 ? NUM_128 : NUM_256;
    tilingData->k0 = BS < NUM_256 ? NUM_512 : NUM_384;
    tilingData->n0 = NUM_256;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetInput(const gert::TilingContext *context, MoeDispatchFFNCombineTilingData *tilingData,
    MoeDispatchFFNCombineConfig &config, const char *nodeName)
{
    OP_TILING_CHECK(TilingCheckMoeDispatchFFNCombine(context, config, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "check input failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckInputParam(context, config, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "check input param failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetInputParam(context, tilingData, config) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "set input param failed."), return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDispatchFFNCombineTilingFuncImplPublic(gert::TilingContext *context, MoeDispatchFFNCombineConfig &config)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter Moe_Dispatch_FFN_Combine tiling check func.");

    MoeDispatchFFNCombineTilingData *tilingData = context->GetTilingData<MoeDispatchFFNCombineTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, VECTOR_INNER_ERR_REPORT_TILING(nodeName, "tilingData is nullptr."),
        return ge::GRAPH_FAILED);

    // Input check & set
    OP_TILING_CHECK(CheckAndSetInput(context, tilingData, config, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "Check and set input failed."), return ge::GRAPH_FAILED);

    // Platform Info
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum));
    context->SetScheduleMode(1); // batch model模型，所有核同时启动
    OP_LOGI(nodeName, "TilingData Init: aivNum: %u, aicNum: %u, ubSize:%lu \n", aivNum, aicNum, ubSize);

    // Attr check & set
    int64_t CclBufferSize = -1;
    OP_TILING_CHECK(CheckAttrAndSetTilingData(context, config, tilingData, aicNum, CclBufferSize) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "Getting attr failed."), return ge::GRAPH_FAILED);

    // InitRouting
    int64_t quantMode = GetQuantModeByAttrDispatchOutType(context, config);
    ge::DataType xDtype = context->GetInputDesc(config.xIndex)->GetDataType();
    OP_TILING_CHECK(GetMoeInitRoutingV3Tiling(*tilingData, aivNum, ubSize, quantMode, xDtype, nodeName) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "Moe init routing failed."), return ge::GRAPH_FAILED);

    // Cal TilingKey
    uint64_t tilingKey = CalTilingKey(context, config, tilingData, nodeName);
    OP_LOGI(nodeName, "OP TilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);

    // WorkspaceSize
    WorkspaceInfo workspaceInfo((uint8_t*)0, tilingData);
    OP_TILING_CHECK(SetWorkspace(context, nodeName, workspaceInfo, CclBufferSize) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "Tiling set workspace Failed"), return ge::GRAPH_FAILED);

    // Print Info
    PrintMoeDispatchFFNCombineTilingData(tilingData, nodeName);
    printWorkspaceInfo(&workspaceInfo, nodeName);
    printPeermemInfo(tilingData, nodeName);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDispatchFFNCombineTilingFunc(gert::TilingContext* context)
{
    MoeDispatchFFNCombineConfig config;
    ge::graphStatus ret;

    ret = MoeDispatchFFNCombineTilingFuncImplPublic(context, config);

    return ret;
}

struct MoeDispatchFFNCombineCompileInfo {};
static ge::graphStatus TilingParseForMoeDispatchFFNCombine(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeDispatchFFNCombine)
    .Tiling(MoeDispatchFFNCombineTilingFunc)
    .TilingParse<MoeDispatchFFNCombineCompileInfo>(TilingParseForMoeDispatchFFNCombine);
} // namespace optiling