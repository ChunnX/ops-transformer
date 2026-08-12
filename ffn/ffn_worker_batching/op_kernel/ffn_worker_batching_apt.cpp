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
 * \brief arch35 (Ascend950 / DAV_3510) kernel 入口 —— 纯 A5 实现,不复用任何 A2 kernel 头。
 *
 *   phase0 prepare   NORM/RECV 的 expert_id 归一到同一个扁平缓冲(RECV 另含握手等待与回写)
 *      ↓ SyncAll
 *   phase1 段内排序   各段压掉被 mask 的 token 后排序,结果为 proposal 对
 *      ↓ SyncAll
 *   phase2 段间归并   每轮 4 路 MrgSort,两块工作区乒乓,直到剩一路(每轮之间 SyncAll)
 *   phase3 拆包       proposal 对 → 有序 expert_id + gather_idx
 *      ↓ SyncAll
 *   phase4 group_list 有序序列计数(SIMT)→ 汇总 → 压成 [expert_id, tokenNum] 行
 *      ↓ SyncAll
 *   phase5 gather     按 gather_idx 搬运 y 与四个 id 输出
 *
 * 切分、单次搬运量、workspace 段偏移全部来自 host tiling;kernel 不自算任何布局。
 */

#include "kernel_operator.h"
#include "arch35/ffn_worker_batching_arch35_tiling_def.h"

using FfnWorkerBatchingTilingData = FfnWorkerBatchingArch35TilingData;

#include "arch35/ffn_wb_a5_context.h"
#include "arch35/ffn_wb_a5_prepare.h"
#include "arch35/ffn_wb_a5_sort.h"
#include "arch35/ffn_wb_a5_group_list.h"
#include "arch35/ffn_wb_a5_gather.h"

#define TILING_KEY_NORM 100
#define TILING_KEY_RECV 101

using namespace AscendC;
using namespace FfnWbBatchingArch35;

namespace {
// actual_token_num 由 0 号向量核写出;经 UB 整段搬运,不对 GM 做标量写。
__aicore__ inline void WriteActualTokenNum(GM_ADDR actualTokenNum, int64_t value, TPipe *pipe)
{
    const int64_t subNum = GetTaskRation() > 0 ? GetTaskRation() : 1;
    if (GetBlockIdx() * subNum + GetSubBlockIdx() != 0) {
        return;
    }
    GlobalTensor<int64_t> outGm;
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(actualTokenNum), 1);
    TBuf<TPosition::VECCALC> buf;
    pipe->InitBuffer(buf, ONE_BLK_SIZE);
    LocalTensor<int64_t> local = buf.Get<int64_t>();
    local.SetValue(0, value);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyExtParams cp{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(outGm, local, cp);
    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
}
}  // namespace

template <bool isRecv>
__aicore__ inline void FfnWorkerBatchingA5(GM_ADDR schedule_context, GM_ADDR y, GM_ADDR group_list,
                                           GM_ADDR session_ids, GM_ADDR micro_batch_ids, GM_ADDR token_ids,
                                           GM_ADDR expert_offsets, GM_ADDR dynamic_scale, GM_ADDR actual_token_num,
                                           GM_ADDR userWS, const FfnWorkerBatchingTilingData *tilingData)
{
    TPipe pipe;
    ScheduleContextInfo ctx;
    ScheduleContextParse<isRecv>(schedule_context, tilingData, ctx, &pipe);
    pipe.Reset();

    const int64_t wordSize = static_cast<int64_t>(sizeof(int32_t));
    GM_ADDR flatIdsWs = userWS + tilingData->wsFlatIds * wordSize;
    GM_ADDR pairAWs = userWS + tilingData->wsPairA * wordSize;
    GM_ADDR pairBWs = userWS + tilingData->wsPairB * wordSize;
    GM_ADDR segCntWs = userWS + tilingData->wsSegCnt * wordSize;
    GM_ADDR sortedIdsWs = userWS + tilingData->wsSortedIds * wordSize;
    GM_ADDR gatherIdxWs = userWS + tilingData->wsGatherIdx * wordSize;

    // ---------------- phase0:expert_id 归一 ----------------
    if constexpr (isRecv) {
        // 轮询下标由本次调用推进(回写见 FfnWbA5RecvWait),而每个核是在 ScheduleContextParse 里
        // 各自从 GM 读它的。回写没有任何屏障挡着,起步偏晚的核会读到已经加一的值——实测
        // A=512/M=4 时 micro_batch_ids 出现 0 与 1 各半、y 也跟着取错 micro batch。
        // 故回写前先等所有核把上下文读完。
        SyncAll();
        // 先等本 micro batch 的全部 session 就绪,并推进轮询下标
        GM_ADDR tokenInfoBuf = reinterpret_cast<GM_ADDR>(ctx.bufferPtr.tokenInfoBuf);
        FfnWbA5RecvWait waiter;
        waiter.Init(schedule_context, tokenInfoBuf, &ctx, &pipe);
        waiter.Process();
        pipe.Reset();
        SyncAll();

        FfnWbPrepareArch35 prep;
        prep.Init(flatIdsWs, &ctx, &pipe, tilingData->flatElements, tilingData->preparePerLoopRows);
        prep.ProcessRecv(tokenInfoBuf);
        pipe.Reset();
    } else {
        FfnWbPrepareArch35 prep;
        prep.Init(flatIdsWs, &ctx, &pipe, tilingData->flatElements, tilingData->preparePerLoopRows);
        prep.ProcessNorm(reinterpret_cast<GM_ADDR>(ctx.bufferPtr.expertIdsBuf));
        pipe.Reset();
    }
    SyncAll();

    // ---------------- phase1:段内排序 ----------------
    {
        A5SortSegParam sp;
        sp.segNum = tilingData->sortSegNum;
        sp.perSegElements = tilingData->sortPerSegElements;
        sp.totalElements = tilingData->flatElements;
        sp.expertStart = tilingData->expertStart;
        sp.sortLenPerSeg = tilingData->sortLenPerSeg;

        FfnWbA5SegSort sorter;
        sorter.Init(flatIdsWs, pairAWs, segCntWs, sp, &ctx, &pipe);
        sorter.Process();
        pipe.Reset();
    }
    SyncAll();

#ifdef FFN_WB_A5_DIAG
    // [临时诊断] 借 expert_offsets 输出通道回读中间量:参数 + 扁平序列前几个值 + 各段有效计数
    {
        const int64_t subNumD = GetTaskRation() > 0 ? GetTaskRation() : 1;
        if (GetBlockIdx() * subNumD + GetSubBlockIdx() == 0) {
            TBuf<TPosition::VECCALC> dbgBuf;
            pipe.InitBuffer(dbgBuf, 4096);
            LocalTensor<int32_t> dbg = dbgBuf.Get<int32_t>();
            GlobalTensor<int32_t> flatDbg;
            flatDbg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(flatIdsWs), tilingData->flatElements);
            GlobalTensor<int32_t> cntDbg;
            cntDbg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(segCntWs), tilingData->sortSegNum * 8);
            DataCopyPadExtParams<int32_t> padD{false, 0, 0, 0};
            // [8,16) 扁平序列前 8 个;[16,32) 前 16 段的计数(每段取槽首)
            DataCopyExtParams cpF{static_cast<uint16_t>(1), static_cast<uint32_t>(8 * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(dbg[8], flatDbg, cpF, padD);
            DataCopyExtParams cpC{static_cast<uint16_t>(16), static_cast<uint32_t>(sizeof(int32_t)),
                                  static_cast<uint32_t>(7 * sizeof(int32_t)), 0, 0};
            DataCopyPad(dbg[16], cntDbg, cpC, padD);
            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            dbg.SetValue(0, static_cast<int32_t>(tilingData->flatElements));
            dbg.SetValue(1, static_cast<int32_t>(tilingData->sortSegNum));
            dbg.SetValue(2, static_cast<int32_t>(tilingData->sortPerSegElements));
            dbg.SetValue(3, static_cast<int32_t>(tilingData->sortLenPerSeg));
            dbg.SetValue(4, static_cast<int32_t>(tilingData->mergeRounds));
            dbg.SetValue(5, static_cast<int32_t>(tilingData->expertStart));
            dbg.SetValue(6, static_cast<int32_t>(ctx.A));
            dbg.SetValue(7, static_cast<int32_t>(ctx.BS * ctx.K));
            GlobalTensor<int32_t> outDbg;
            outDbg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(dynamic_scale), 64);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyExtParams cpO{static_cast<uint16_t>(1), static_cast<uint32_t>(48 * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(outDbg, dbg, cpO);
            SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        }
    }
#endif

    // ---------------- phase2:段间归并 ----------------
    int64_t finalWs = 0;  // 0=pairA,1=pairB
    int64_t totalValid = 0;
    {
        A5MergeParam mp;
        mp.segNum = tilingData->sortSegNum;
        mp.sortLenPerSeg = tilingData->sortLenPerSeg;
        mp.oneLoopMaxElements = tilingData->mergeOneLoopElements;
        mp.rounds = tilingData->mergeRounds;

        FfnWbA5MrgSort merger;
        merger.Init(pairAWs, pairBWs, segCntWs, mp, &pipe);
        for (int64_t r = 0; r < tilingData->mergeRounds; r++) {
            finalWs = merger.ProcessRound(r, finalWs);
            SyncAll();
        }
        totalValid = merger.TotalValid();
        pipe.Reset();
    }
    ctx.validGatherIdxLength = totalValid;

    // ---------------- phase3:拆包 ----------------
    if (totalValid > 0) {
        A5ExtractParam ep;
        ep.totalValid = totalValid;
        ep.perLoopElements = tilingData->extractPerLoopElements;

        FfnWbA5Extract extractor;
        extractor.Init((finalWs == 0) ? pairAWs : pairBWs, sortedIdsWs, gatherIdxWs, ep, &pipe);
        extractor.Process();
        pipe.Reset();
    }
    SyncAll();

    // ---------------- phase4:group_list ----------------
    {
        A5GroupListParam gp;
        gp.rowsPerLoop = tilingData->glRowsPerLoop;

        FfnWbA5GroupList gl;
        gl.Init(sortedIdsWs, group_list, gp, &ctx, &pipe);
        gl.Process();
        pipe.Reset();
    }

    WriteActualTokenNum(actual_token_num, totalValid, &pipe);
    pipe.Reset();

    // ---------------- phase5:gather ----------------
    if (totalValid > 0) {
        FfnWbA5Gather<isRecv> op;
        op.Init(gatherIdxWs, y, session_ids, micro_batch_ids, token_ids, expert_offsets, dynamic_scale, &ctx, &pipe, 0);
        op.Process();
        pipe.Reset();
    }
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
        FfnWorkerBatchingA5<false>(schedule_context, y, group_list, session_ids, micro_batch_ids, token_ids,
                                   expert_offsets, dynamic_scale, actual_token_num, userWS, tilingData);
    } else if (TILING_KEY_IS(TILING_KEY_RECV)) {
        FfnWorkerBatchingA5<true>(schedule_context, y, group_list, session_ids, micro_batch_ids, token_ids,
                                  expert_offsets, dynamic_scale, actual_token_num, userWS, tilingData);
    }
}
