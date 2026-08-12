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
 *        - NORM：单核 KernelSortMaskOneCore / 多核 FfnWbSortMultiCoreArch35(+MrgSort 归并)。
 *        - RECV：scan_token_info 前置扫描 → 三种 sort 分支
 *                (KernelScanSortMaskOneCore / FfnWbScanSortMultiCoreArch35 /
 *                 FfnWbScanSortMultiCoreBskArch35，由 isNormalBsK 分流)。
 *        三步算法照搬 A2（ffn_worker_batching.cpp），仅按运行时 UB/核数重算切分阈值。
 *        102(RECV_GATHER) 为 A2 保留分支，tiling 从不 SetTilingKey(102)，此处不分发。
 */
#include "kernel_operator.h"
#include "arch35/ffn_worker_batching_arch35_tiling_def.h"

// arch35 平铺 tiling struct 与 A2 字段一致;kernel TU 内起别名,使复用的 A2 头中的 FfnWorkerBatchingTilingData 解析到本
// struct(host 侧不受影响)。
using FfnWorkerBatchingTilingData = FfnWorkerBatchingArch35TilingData;
// 复用 A2 的 11 个公共 kernel 头 + 命名空间桥接(须先于下方 arch35 差异头)
#include "arch35/ffn_wb_arch35_reuse.h"
// 以下 4 个为 arch35 真实代次差异实现(k=0 守卫 / DAV-3510 缓冲初始化+同步)
#include "arch35/ffn_wb_group_listing_arch35.h"
#include "arch35/ffn_wb_sort_multi_core_arch35.h"
#include "arch35/ffn_wb_scan_sort_multi_core_arch35.h"
#include "arch35/ffn_wb_scan_sort_multi_core_bsk_arch35.h"

#define TILING_KEY_NORM 100
#define TILING_KEY_RECV 101

using namespace AscendC;
using namespace FfnWbBatchingArch35;
using namespace FfnWbBatching;

// step 2 gather / step 3 group_listing 核间分工(部分核 gather,其余核多核 O(Y) group_listing)。
// NORM(isScanFlag=false)/RECV(isScanFlag=true) 共用;仅 gather 的 expertIdx 工作区偏移不同,由调用方传入
// (NORM 用 totalLength、RECV 用 totalLengthWithPad)。
template <bool isScanFlag>
__aicore__ inline void GatherAndGroupListing(GM_ADDR gatherExpertIdxWS, GM_ADDR y, GM_ADDR group_list,
                                             GM_ADDR session_ids, GM_ADDR micro_batch_ids, GM_ADDR token_ids,
                                             GM_ADDR expert_offsets, GM_ADDR dynamic_scale, GM_ADDR userWS,
                                             ScheduleContextInfo &contextInfo, TPipe &ffnPipe)
{
    FfnWbGroupListingArch35 expertTokenOutMultiOp;
    int64_t groupListingDealFlag = 0;
    uint32_t usedCoreNum = GROUP_LISTING_AIV_NUM;
    if (contextInfo.validGatherIdxLength > GROUP_LISTING_MULTI_CORE_LENGTH) {
        groupListingDealFlag = 1;
        usedCoreNum = GROUP_LISTING_MULTI_AIV_NUM;
    }

    if (GetBlockIdx() < contextInfo.coreNum - usedCoreNum) {
        KernelFfnWBGatherOutAll<isScanFlag> op;
        op.Init(gatherExpertIdxWS, y, session_ids, micro_batch_ids, token_ids, expert_offsets, dynamic_scale,
                &contextInfo, &ffnPipe, usedCoreNum);
        op.Process();
        ffnPipe.Reset();
    } else {
        expertTokenOutMultiOp.Init(
            userWS + (OFFSET_SORTED_EXPERT_IDS + contextInfo.sortNumWorkSpace) * sizeof(int32_t), group_list,
            userWS +
                (OFFSET_SORTED_EXPERT_IDS + contextInfo.sortNumWorkSpace * (NUM_TWO * NUM_FOUR + 1)) * sizeof(int32_t),
            &contextInfo, &ffnPipe, groupListingDealFlag);
        expertTokenOutMultiOp.Process(groupListingDealFlag);
    }

    SyncAll();
    if (groupListingDealFlag && (GetBlockIdx() == contextInfo.coreNum - 1)) {
        expertTokenOutMultiOp.ProcessExpertCount();
    }
}

// NORM(100)：sort（单/多核）→ gather / group_listing 核间分工。三步照搬 A2。
extern "C" __aicore__ inline void
FfnWorkerBatchingProcess(GM_ADDR schedule_context, GM_ADDR y, GM_ADDR group_list, GM_ADDR session_ids,
                         GM_ADDR micro_batch_ids, GM_ADDR token_ids, GM_ADDR expert_offsets, GM_ADDR dynamic_scale,
                         GM_ADDR actual_token_num, GM_ADDR userWS, const FfnWorkerBatchingTilingData *tilingData)
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
        op.Init(reinterpret_cast<GM_ADDR>(contextInfo.bufferPtr.expertIdsBuf), userWS, &tilingdataSort, &contextInfo,
                &ffnPipe);
        op.Process();
        ffnPipe.Reset();
    } else {
        FfnWbSortMultiCoreArch35 op;
        op.Init(reinterpret_cast<GM_ADDR>(contextInfo.bufferPtr.expertIdsBuf), userWS, &tilingdataSort, &contextInfo,
                &ffnPipe);
        op.Process();
        ffnPipe.Reset();
    }

    // 求 gather_idx 有效长度 + 写 actual_token_num
    ValidGatherIdxLengthCompute(userWS, contextInfo, actual_token_num);

    // step 2 gather / step 3 group_listing（NORM：gather 用 totalLength 偏移）
    GatherAndGroupListing<false>(
        userWS +
            (OFFSET_SORTED_EXPERT_IDS + contextInfo.sortNumWorkSpace + tilingdataSort.totalLength) * sizeof(int32_t),
        y, group_list, session_ids, micro_batch_ids, token_ids, expert_offsets, dynamic_scale, userWS, contextInfo,
        ffnPipe);
}

// RECV(101)：scan_token_info 前置扫描 → 三分支 scan sort → gather / group_listing。三步照搬 A2。
extern "C" __aicore__ inline void
FfnWorkerBatchingRecvProcess(GM_ADDR schedule_context, GM_ADDR y, GM_ADDR group_list, GM_ADDR session_ids,
                             GM_ADDR micro_batch_ids, GM_ADDR token_ids, GM_ADDR expert_offsets, GM_ADDR dynamic_scale,
                             GM_ADDR actual_token_num, GM_ADDR userWS, const FfnWorkerBatchingTilingData *tilingData)
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
        FfnWbScanSortMultiCoreBskArch35 op;
        op.Init(tokenInfoGmAddr, userWS, &tilingdataSort, &contextInfo, &ffnPipe);
        op.Process();
        ffnPipe.Reset();
    } else {
        FfnWbScanSortMultiCoreArch35 op;
        op.Init(tokenInfoGmAddr, userWS, &tilingdataSort, &contextInfo, &ffnPipe);
        op.Process();
        ffnPipe.Reset();
    }

    // 求 gather_idx 有效长度 + 写 actual_token_num
    ValidGatherIdxLengthCompute(userWS, contextInfo, actual_token_num);

    // step 2 gather / step 3 group_listing：RECV 复用 NORM 的 O(Y) 多核 group_listing（替换原 O(E×Y) 单核实现）。
    // 原单核版对每个 expert 都要 CompareScalar/GatherMask 扫全块（O(E×Y)），高 expert 数（DeepSeek 级 E=128/256）
    // 下成为串行长尾；NORM 版为单遍 O(Y) + 多核，E 全范围平坦。输入（排序 expert_ids @128+Y）与输出语义一致。
    GatherAndGroupListing<true>(
        userWS + (OFFSET_SORTED_EXPERT_IDS + contextInfo.sortNumWorkSpace + tilingdataSort.totalLengthWithPad) *
                     sizeof(int32_t),
        y, group_list, session_ids, micro_batch_ids, token_ids, expert_offsets, dynamic_scale, userWS, contextInfo,
        ffnPipe);
}

extern "C" __global__ __aicore__ void ffn_worker_batching(GM_ADDR schedule_context, GM_ADDR y, GM_ADDR group_list,
                                                          GM_ADDR session_ids, GM_ADDR micro_batch_ids,
                                                          GM_ADDR token_ids, GM_ADDR expert_offsets,
                                                          GM_ADDR dynamic_scale, GM_ADDR actual_token_num,
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

    REGISTER_TILING_DEFAULT(FfnWorkerBatchingTilingData);
    GET_TILING_DATA_WITH_STRUCT(FfnWorkerBatchingTilingData, tilingDataIn, tiling);
    const FfnWorkerBatchingTilingData *tilingData = &tilingDataIn;

    if (TILING_KEY_IS(TILING_KEY_NORM)) {
        FfnWorkerBatchingProcess(schedule_context, y, group_list, session_ids, micro_batch_ids, token_ids,
                                 expert_offsets, dynamic_scale, actual_token_num, userWS, tilingData);
    } else if (TILING_KEY_IS(TILING_KEY_RECV)) {
        FfnWorkerBatchingRecvProcess(schedule_context, y, group_list, session_ids, micro_batch_ids, token_ids,
                                     expert_offsets, dynamic_scale, actual_token_num, userWS, tilingData);
    }
}
