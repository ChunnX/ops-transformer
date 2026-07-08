/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file ffn_worker_batching_apt.cpp
 * \brief arch35 (Ascend950 / DAV_3510) kernel entry for FfnWorkerBatching.
 *        迭代二整合：NORM(100) + RECV(101) 双 TilingKey 完整整合，单/多核 sort 全路径打通。
 *        - NORM：单核 KernelSortMaskOneCore / 多核 KernelSortMaskMultiCore(+MrgSort 归并)。
 *        - RECV：scan_token_info 前置扫描 → 三种 sort 分支
 *                (KernelScanSortMaskOneCore / KernelScanSortMaskMultiCore /
 *                 KernelScanSortMaskMultiCoreBsK，由 isNormalBsK 分流)。
 *        三步算法照搬 A2（ffn_worker_batching.cpp），仅按运行时 UB/核数重算切分阈值。
 *        102(RECV_GATHER) 为 A2 保留分支，tiling 从不 SetTilingKey(102)，此处不分发。
 */
#include "kernel_operator.h"
#include "arch35/ffn_worker_batching_arch35_tiling_def.h"
#include "arch35/ffn_wb_common.h"
#include "arch35/ffn_wb_get_schedule_context.h"
#include "arch35/ffn_wb_gather_out_all.h"
#include "arch35/ffn_wb_group_listing.h"
#include "arch35/ffn_wb_sort_one_core.h"
#include "arch35/ffn_wb_sort_multi_core.h"
#include "arch35/ffn_wb_scan_token_info.h"
#include "arch35/ffn_wb_scan_sort_one_core.h"
#include "arch35/ffn_wb_scan_sort_multi_core.h"
#include "arch35/ffn_wb_scan_sort_multi_core_bsk.h"
#include "arch35/ffn_wb_scan_group_listing_one_core.h"

#define TILING_KEY_NORM 100
#define TILING_KEY_RECV 101

using namespace AscendC;
using namespace FfnWbBatching;

// NORM(100)：sort（单/多核）→ gather / group_listing 核间分工。三步照搬 A2。
extern "C" __aicore__ inline void FfnWorkerBatchingProcess(GM_ADDR schedule_context,
    GM_ADDR y, GM_ADDR group_list, GM_ADDR session_ids, GM_ADDR micro_batch_ids, GM_ADDR token_ids,
    GM_ADDR expert_offsets, GM_ADDR dynamic_scale, GM_ADDR actual_token_num,
    GM_ADDR userWS, const FfnWorkerBatchingArch35TilingData *tilingData)
{
    TPipe ffnPipe;
    ScheduleContextInfo contextInfo;
    ScheduleContextInfoCompute<false>(schedule_context, tilingData, contextInfo, &ffnPipe);
    ffnPipe.Reset();

    // step 1 sort：单核（小 shape）vs 多核 + MrgSort 归并（大 shape 触发）。
    SortCustomTilingDataKernel tilingdataSort;
    TilingSort(&tilingdataSort, &contextInfo);
    if (tilingdataSort.tilingKey == TILINGKEY_ONECORE_SORT) {
        KernelSortMaskOneCore op;
        op.Init(reinterpret_cast<GM_ADDR>(contextInfo.bufferPtr.expertIdsBuf), userWS, &tilingdataSort,
                &contextInfo, &ffnPipe);
        op.Process();
        ffnPipe.Reset();
    } else {
        KernelSortMaskMultiCore op;
        op.Init(reinterpret_cast<GM_ADDR>(contextInfo.bufferPtr.expertIdsBuf), userWS, &tilingdataSort,
                &contextInfo, &ffnPipe);
        op.Process();
        ffnPipe.Reset();
    }

    // 求 gather_idx 有效长度 + 写 actual_token_num
    ValidGatherIdxLengthCompute(userWS, contextInfo, actual_token_num);

    KernelFfnWBGroupListing expertTokenOutMultiOp;

    int64_t groupListingDealFlag = 0;
    uint32_t usedCoreNum = GROUP_LISTING_AIV_NUM;
    if (contextInfo.validGatherIdxLength > GROUP_LISTING_MULTI_CORE_LENGTH) {
        groupListingDealFlag = 1;
        usedCoreNum = GROUP_LISTING_MULTI_AIV_NUM;
    }

    // step 2 gather / step 3 group_listing：核间分工（部分核 gather，其余核 group_listing）
    if (GetBlockIdx() < contextInfo.coreNum - usedCoreNum) {
        KernelFfnWBGatherOutAll<false> op;
        op.Init(userWS + (OFFSET_SORTED_EXPERT_IDS + contextInfo.sortNumWorkSpace + tilingdataSort.totalLength) *
            sizeof(int32_t),
            y, session_ids, micro_batch_ids, token_ids, expert_offsets, dynamic_scale,
            &contextInfo, &ffnPipe, usedCoreNum);
        op.Process();
        ffnPipe.Reset();
    } else {
        expertTokenOutMultiOp.Init(
            userWS + (OFFSET_SORTED_EXPERT_IDS + contextInfo.sortNumWorkSpace) * sizeof(int32_t), group_list,
            userWS + (OFFSET_SORTED_EXPERT_IDS + contextInfo.sortNumWorkSpace * (NUM_TWO * NUM_FOUR + 1)) *
                sizeof(int32_t),
            &contextInfo, &ffnPipe, groupListingDealFlag);
        expertTokenOutMultiOp.Process(groupListingDealFlag);
    }

    SyncAll();
    if (groupListingDealFlag && (GetBlockIdx() == contextInfo.coreNum - 1)) {
        expertTokenOutMultiOp.ProcessExpertCount();
    }
}

// RECV(101)：scan_token_info 前置扫描 → 三分支 scan sort → gather / group_listing。三步照搬 A2。
extern "C" __aicore__ inline void FfnWorkerBatchingRecvProcess(GM_ADDR schedule_context,
    GM_ADDR y, GM_ADDR group_list, GM_ADDR session_ids, GM_ADDR micro_batch_ids, GM_ADDR token_ids,
    GM_ADDR expert_offsets, GM_ADDR dynamic_scale, GM_ADDR actual_token_num,
    GM_ADDR userWS, const FfnWorkerBatchingArch35TilingData *tilingData)
{
    TPipe ffnPipe;
    ScheduleContextInfo contextInfo;
    ScheduleContextInfoCompute<true>(schedule_context, tilingData, contextInfo, &ffnPipe);
    ffnPipe.Reset();

    // RECV 专属：先扫描接收 token_info（block0 忙等当前 micro batch ready，推进 poll idx）
    GM_ADDR tokenInfoGmAddr = reinterpret_cast<GM_ADDR>(contextInfo.bufferPtr.tokenInfoBuf);
    KernelScanTokenInfo scanTokenInfo;
    scanTokenInfo.Init(schedule_context, tokenInfoGmAddr, userWS, &contextInfo, &ffnPipe);
    scanTokenInfo.Process();
    SyncAll();
    ffnPipe.Reset();

    // step 1 sort（scan 变体，isNormalBsK 分流）
    SortCustomTilingDataKernel tilingdataSort;
    TilingScanSort(&tilingdataSort, &contextInfo);
    if (tilingdataSort.tilingKey == TILINGKEY_ONECORE_SORT) {
        KernelScanSortMaskOneCore op;
        op.Init(tokenInfoGmAddr, userWS, &tilingdataSort, &contextInfo, &ffnPipe);
        op.Process();
        ffnPipe.Reset();
    } else if (tilingdataSort.isNormalBsK != NORMAL_BSK) {
        KernelScanSortMaskMultiCoreBsK op;
        op.Init(tokenInfoGmAddr, userWS, &tilingdataSort, &contextInfo, &ffnPipe);
        op.Process();
        ffnPipe.Reset();
    } else {
        KernelScanSortMaskMultiCore op;
        op.Init(tokenInfoGmAddr, userWS, &tilingdataSort, &contextInfo, &ffnPipe);
        op.Process();
        ffnPipe.Reset();
    }

    // 求 gather_idx 有效长度 + 写 actual_token_num
    ValidGatherIdxLengthCompute(userWS, contextInfo, actual_token_num);

    // step 2 gather / step 3 group_listing：RECV 保留 GROUP_LISTING_SCAN_AIV_NUM 核做单核 group_listing
    int64_t blockIdx = GetBlockIdx();
    if (blockIdx < tilingData->coreNum - GROUP_LISTING_SCAN_AIV_NUM) {
        KernelFfnWBGatherOutAll<true> op;
        op.Init(userWS + (OFFSET_SORTED_EXPERT_IDS + contextInfo.sortNumWorkSpace +
            tilingdataSort.totalLengthWithPad) * sizeof(int32_t),
            y, session_ids, micro_batch_ids, token_ids, expert_offsets, dynamic_scale,
            &contextInfo, &ffnPipe, GROUP_LISTING_SCAN_AIV_NUM);
        op.Process();
        ffnPipe.Reset();
    } else {
        KernelFfnWBScanGroupListingOneCore expertTokenOutOp;
        expertTokenOutOp.Init(userWS + (OFFSET_SORTED_EXPERT_IDS + contextInfo.sortNumWorkSpace) * sizeof(int32_t),
                              group_list, &contextInfo, &ffnPipe);
        expertTokenOutOp.Process();
        ffnPipe.Reset();
    }
}

extern "C" __global__ __aicore__ void ffn_worker_batching(GM_ADDR schedule_context,
                    GM_ADDR y, GM_ADDR group_list, GM_ADDR session_ids,
                    GM_ADDR micro_batch_ids, GM_ADDR token_ids,
                    GM_ADDR expert_offsets, GM_ADDR dynamic_scale, GM_ADDR actual_token_num,
                    GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(FfnWorkerBatchingArch35TilingData);
    GET_TILING_DATA_WITH_STRUCT(FfnWorkerBatchingArch35TilingData, tilingDataIn, tiling);
    const FfnWorkerBatchingArch35TilingData *tilingData = &tilingDataIn;

    if (TILING_KEY_IS(TILING_KEY_NORM)) {
        FfnWorkerBatchingProcess(schedule_context, y, group_list, session_ids, micro_batch_ids, token_ids,
            expert_offsets, dynamic_scale, actual_token_num, userWS, tilingData);
    } else if (TILING_KEY_IS(TILING_KEY_RECV)) {
        FfnWorkerBatchingRecvProcess(schedule_context, y, group_list, session_ids, micro_batch_ids, token_ids,
            expert_offsets, dynamic_scale, actual_token_num, userWS, tilingData);
    }
}
