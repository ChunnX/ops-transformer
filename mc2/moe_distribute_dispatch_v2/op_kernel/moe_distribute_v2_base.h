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
 * \file moe_distribute_v2_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_V2_BASE_H
#define MOE_DISTRIBUTE_V2_BASE_H

#include "moe_distribute_v2_constant.h"

#if __has_include("../common/inc/kernel/moe_distribute_base.h")
#include "../common/inc/kernel/moe_distribute_base.h"
#include "../common/inc/kernel/mc2_kernel_utils.h"
#else
#include "../../common/inc/kernel/moe_distribute_base.h"
#include "../../common/inc/kernel/mc2_kernel_utils.h"
#endif

namespace MoeDistributeV2Base {

using namespace AscendC;
using namespace Mc2Kernel;

constexpr uint64_t SQ_WQE_DUMP_BYTES = 64UL * 2048UL;
constexpr uint64_t SQ_WQE_DUMP_OFFSET = 800UL * 1024UL;
constexpr uint32_t SQ_WQE_DUMP_COPY_U64_NUM = UB_ALIGN / sizeof(uint64_t);
constexpr uint32_t DB_HISTORY_ENTRY_NUM = 500U;
constexpr uint32_t DB_HISTORY_ENTRY_U32_NUM = 8U;
constexpr uint64_t DB_HISTORY_DUMP_OFFSET = SQ_WQE_DUMP_OFFSET + SQ_WQE_DUMP_BYTES;

__aicore__ inline void DumpSQWqeToWin(GM_ADDR sqBaseAddr, uint32_t wqeSize, uint32_t depth,
    LocalTensor<uint64_t>& ubLocalTensor, GlobalTensor<uint32_t>& selfDataStatusGMAiv)
{
    uint64_t sqDumpBytes = static_cast<uint64_t>(wqeSize) * static_cast<uint64_t>(depth);
    if (sqDumpBytes > SQ_WQE_DUMP_BYTES) {
        sqDumpBytes = SQ_WQE_DUMP_BYTES;
    }
    if (sqDumpBytes > STATE_SIZE - SQ_WQE_DUMP_OFFSET) {
        sqDumpBytes = STATE_SIZE - SQ_WQE_DUMP_OFFSET;
    }
    sqDumpBytes = sqDumpBytes / UB_ALIGN * UB_ALIGN;
    if (sqDumpBytes == 0) {
        return;
    }

    uint32_t aivId = GetBlockIdx();
    GM_ADDR statusDataSpaceGm = (GM_ADDR)selfDataStatusGMAiv.GetPhyAddr() - aivId * 64;
    GlobalTensor<uint64_t> sqDumpSrc;
    GlobalTensor<uint64_t> sqDumpDst;
    sqDumpSrc.SetGlobalBuffer((__gm__ uint64_t*)sqBaseAddr);
    sqDumpDst.SetGlobalBuffer((__gm__ uint64_t*)(statusDataSpaceGm + SQ_WQE_DUMP_OFFSET));

    uint64_t sqDumpU64Num = sqDumpBytes / sizeof(uint64_t);
    for (uint64_t offset = 0; offset < sqDumpU64Num; offset += SQ_WQE_DUMP_COPY_U64_NUM) {
        DataCopy(ubLocalTensor, sqDumpSrc[offset], SQ_WQE_DUMP_COPY_U64_NUM);
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        DataCopy(sqDumpDst[offset], ubLocalTensor, SQ_WQE_DUMP_COPY_U64_NUM);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }

    GlobalTensor<uint32_t> selfDataStatusGMAiv2;
    selfDataStatusGMAiv2.SetGlobalBuffer((__gm__ uint32_t*)(statusDataSpaceGm + aivId * 64 + 48 * 64));
    selfDataStatusGMAiv2.SetValue(10, static_cast<uint32_t>(sqDumpBytes));
    DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(selfDataStatusGMAiv2);
}

__aicore__ inline uint32_t InitWinState(GlobalTensor<uint32_t> selfDataStatusGMTensor, uint32_t epRankIdHccl, uint32_t epWorldSizeHccl, uint32_t epRankIdOriginal,
                                           uint32_t moeExpertNum, uint32_t epWorldSizeOriginal, uint32_t globalBS, TBuf<> dataStateBuf)
{
    LocalTensor<uint64_t> dataStateLocalTensor64 = dataStateBuf.Get<uint64_t>();
    LocalTensor<uint32_t> dataStateLocalTensor = dataStateBuf.Get<uint32_t>();
    DataCopy(dataStateLocalTensor, selfDataStatusGMTensor, UB_ALIGN / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint32_t dataState = dataStateLocalTensor.GetValue(ZERONE_STATE_POS);
    dataStateLocalTensor.SetValue(ZERONE_STATE_POS, dataState == 0 ? 1 : 0);
    dataStateLocalTensor.SetValue(OPOSITION_POS, 1);
    dataStateLocalTensor.SetValue(TILING_EPRANKID_POS, epRankIdOriginal);
    dataStateLocalTensor.SetValue(MOE_NUM_POS, moeExpertNum);
    dataStateLocalTensor.SetValue(TILING_WORLDSIZE_POS, epWorldSizeOriginal);
    dataStateLocalTensor.SetValue(GLOBALBS_POS, globalBS);
    uint32_t opCnt = dataStateLocalTensor64.GetValue(OP_CNT_POSUL);
    opCnt = opCnt + 1;
    dataStateLocalTensor64.SetValue(OP_CNT_POSUL, opCnt);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(selfDataStatusGMTensor, dataStateLocalTensor, UB_ALIGN / sizeof(uint32_t));

    if ((epRankIdOriginal != epRankIdHccl) || (epWorldSizeOriginal != epWorldSizeHccl)) {
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        DataCopyParams hcclDatacopyParams{1U, HCCL_DFX_NUM * sizeof(uint32_t), 0U, 0U};
        dataStateLocalTensor.SetValue(HCCL_EPRANKId_POS, epRankIdHccl);
        dataStateLocalTensor.SetValue(HCCL_WORLDSIZE_POS, epWorldSizeHccl);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(selfDataStatusGMTensor[HCCL_DFX_POS], dataStateLocalTensor, hcclDatacopyParams);
    }
    return dataState;
}

 __aicore__ inline void AIVRDMAPostDoorBell(__gm__ HcclAiRMAWQ* qp_ctx_entry, uint64_t curHead, __gm__ uint32_t* curHardwareHead, 
    LocalTensor<uint64_t>& ubLocalTensor, LocalTensor<uint32_t>& ubLocalHeadTensor,
    GlobalTensor<uint32_t>& selfDataStatusGMAiv)
{
    uint64_t doorBellInfo = 0;
    doorBellInfo |= qp_ctx_entry->wqn; // [0:23] DB_TAG (qp_num)
    doorBellInfo |= 0UL << 24UL; // [24:27] DB_CMD = HNS_ROCE_V2_SQ_DB (0)
    doorBellInfo |= (curHead % 65536UL) << 32UL; // [32:47] DB_PI = sq.head
    doorBellInfo |= (uint64_t)(qp_ctx_entry->sl) << 48UL; // [48:50] DB_SL = qp.sl

    __gm__ uint64_t* doorBellAddr = (__gm__ uint64_t* )(qp_ctx_entry->dbAddr);
    PipeBarrier<PIPE_ALL>();

    ubLocalTensor.SetValue(0, doorBellInfo);
    AscendC::GlobalTensor<uint64_t> DBGlobalTensor;
    DBGlobalTensor.SetGlobalBuffer(doorBellAddr);
    AscendC::DataCopyExtParams copyParams{1, 1 * sizeof(uint64_t), 0, 0, 0};
    PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(DBGlobalTensor, ubLocalTensor, copyParams);
    PipeBarrier<PIPE_ALL>();

    ubLocalHeadTensor.SetValue(0, (uint32_t)curHead);
    AscendC::GlobalTensor<uint32_t> HeadGlobalTensor;
    HeadGlobalTensor.SetGlobalBuffer((__gm__ uint32_t*)curHardwareHead);
    AscendC::DataCopyExtParams copyParamsHead{1, 1 * sizeof(uint32_t), 0, 0, 0};
    PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(HeadGlobalTensor, ubLocalHeadTensor, copyParamsHead);
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<uint32_t> selfDataStatusGMAiv2;
    uint32_t aivId = GetBlockIdx();
    GM_ADDR statusDataSpaceGm = (GM_ADDR)selfDataStatusGMAiv.GetPhyAddr() - aivId * 64;
    selfDataStatusGMAiv2.SetGlobalBuffer((__gm__ uint32_t*)(statusDataSpaceGm + aivId * 64 + 48 * 64));
    uint32_t doorBellCnt = selfDataStatusGMAiv2.GetValue(12) + 1;
    auto curHardwareTailAddr = qp_ctx_entry->tailAddr;
    cacheWriteThrough((__gm__ uint8_t*)curHardwareTailAddr, CACHEWRITESIZE);
    uint32_t curTail = *(__gm__ uint32_t*)(curHardwareTailAddr);
    GlobalTensor<uint32_t> dbHistoryGm;
    dbHistoryGm.SetGlobalBuffer((__gm__ uint32_t*)(statusDataSpaceGm + DB_HISTORY_DUMP_OFFSET));
    uint32_t dbHistoryOffset = ((doorBellCnt - 1) % DB_HISTORY_ENTRY_NUM) * DB_HISTORY_ENTRY_U32_NUM;
    dbHistoryGm.SetValue(dbHistoryOffset, doorBellCnt);
    dbHistoryGm.SetValue(dbHistoryOffset + 1, static_cast<uint32_t>(doorBellInfo));
    dbHistoryGm.SetValue(dbHistoryOffset + 2, static_cast<uint32_t>(doorBellInfo >> 32));
    dbHistoryGm.SetValue(dbHistoryOffset + 3, static_cast<uint32_t>(curHead));
    dbHistoryGm.SetValue(dbHistoryOffset + 4, static_cast<uint32_t>(curHead >> 32));
    dbHistoryGm.SetValue(dbHistoryOffset + 5, curTail);
    dbHistoryGm.SetValue(dbHistoryOffset + 6, static_cast<uint32_t>((uint64_t)doorBellAddr));
    dbHistoryGm.SetValue(dbHistoryOffset + 7, static_cast<uint32_t>((uint64_t)doorBellAddr >> 32));
    DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(dbHistoryGm[dbHistoryOffset]);
    selfDataStatusGMAiv2.SetValue(6, static_cast<uint32_t>(doorBellInfo));
    selfDataStatusGMAiv2.SetValue(7, static_cast<uint32_t>(doorBellInfo >> 32));
    selfDataStatusGMAiv2.SetValue(9, curTail);
    selfDataStatusGMAiv2.SetValue(12, doorBellCnt);
    selfDataStatusGMAiv2.SetValue(13, static_cast<uint32_t>((uint64_t)curHardwareHead));
    selfDataStatusGMAiv2.SetValue(14, static_cast<uint32_t>((uint64_t)curHardwareHead >> 32));
    selfDataStatusGMAiv2.SetValue(15, static_cast<uint32_t>(curHead));
    DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(selfDataStatusGMAiv2);
}
    
__aicore__ inline void AIVRDMAPostSend(
    GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr, uint64_t destRankId, uint64_t messageLen, __gm__ HcclAiRMAInfo* QpInfo,
    LocalTensor<uint64_t>& ubLocalTensor, LocalTensor<uint32_t>& ubLocalHeadTensor, GlobalTensor<uint32_t>& selfDataStatusGMAiv, uint32_t flag)
{
    auto qpNum = ((__gm__ HcclAiRMAInfo*)QpInfo)->qpNum;
    auto qp_ctx_entry = (__gm__ HcclAiRMAWQ*)(((__gm__ HcclAiRMAInfo*)QpInfo)->sqPtr +
        destRankId * qpNum * (uint64_t)(((__gm__ HcclAiRMAInfo*)QpInfo)->sizeOfAiRMAWQ));
    auto mem_info_table = ((__gm__ HcclAiRMAInfo*)QpInfo)->memPtr;
    auto sizeof_memdetail = ((__gm__ HcclAiRMAInfo*)QpInfo)->sizeOfAiRMAMem;
    auto cur_rank_id = (((__gm__ HcclAiRMAInfo*)QpInfo)->curRankId);
    auto sqBaseAddr = qp_ctx_entry->bufAddr;
    auto wqeSize = qp_ctx_entry->wqeSize;
    auto curHardwareHead = qp_ctx_entry->headAddr;
    cacheWriteThrough((__gm__ uint8_t*)curHardwareHead, 8);
    uint64_t curHead = *(__gm__ uint32_t*)(curHardwareHead);
    auto curHardwareTailAddr = qp_ctx_entry->tailAddr;
    auto QP_DEPTH = qp_ctx_entry->depth;
    // uint64_t shift = 11;
    uint64_t shift = 15;
    // uint64_t shift = 63U - AscendC::CountLeadingZero(QP_DEPTH);

    PipeBarrier<PIPE_ALL>();
    if (flag != -1) {
        selfDataStatusGMAiv.SetValue(0, 12 + flag); // 初始化为0，表示数据未准备好
        DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(selfDataStatusGMAiv);
    }
    

    // Make sure we don't overflow the SQ in an infinite loop - no need to mitigate endless loop as the host
    // will timeout and kill the kernel, same as all2all kernel if it fails to complete (e.g. in case of link loss)
    while(1) {
        cacheWriteThrough((__gm__ uint8_t*)curHardwareTailAddr, CACHEWRITESIZE);
        if ((curHead - *(__gm__ uint32_t*)(curHardwareTailAddr)) < QP_DEPTH - 1) {
            break;
        }
        int64_t systemCycleAfter = AscendC::GetSystemCycle();
        int64_t systemCycleAfter1 = AscendC::GetSystemCycle();
    }

    if (flag != -1) {
        PipeBarrier<PIPE_ALL>();
        selfDataStatusGMAiv.SetValue(0, 13 + flag); // 初始化为0，表示数据未准备好
        DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(selfDataStatusGMAiv);
    }
    

    __gm__ uint8_t* wqeAddr = (__gm__ uint8_t*)(sqBaseAddr + wqeSize * (curHead % QP_DEPTH));

    // Write the WQE to GM
    uint64_t ownBit = (curHead >> shift) & 1U;
    uint32_t byte_4 = 3U;                       // [0:4] opcode=0x3(RDMA_WRITE)
    byte_4 |= ((~ownBit) << 7U) & (1U << 7U);   // [7] owner_bit
    byte_4 |= 1U << 8U;                         // [8:8] IBV_SEND_SIGNALED

    *(__gm__ uint32_t*)(wqeAddr) = byte_4;          // Control set by local parameter see above lines
    *(__gm__ uint32_t*)(wqeAddr + 4) = messageLen;  // message size
    *(__gm__ uint32_t*)(wqeAddr + 8) = 0;           // immtdata is always 0 till we provide poll CQ flow in AIV
    *(__gm__ uint32_t*)(wqeAddr + 12) = 1U << 24U;  // [120:127] num_sge = 1
    *(__gm__ uint32_t*)(wqeAddr + 16) = 0;          // [128:151] start_sge_idx = 0;
    __gm__ HcclAiRMAMemInfo* memDetail = (__gm__ HcclAiRMAMemInfo*)(mem_info_table + sizeof_memdetail * destRankId);
    *(__gm__ uint32_t*)(wqeAddr + 20) = ((__gm__ MemDetails*)(memDetail->memDetailPtr +
        memDetail->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::REMOTE_INPUT)))->key;
    *(__gm__ uint64_t*)(wqeAddr + 24) = (uint64_t)destDmaAddr; // destination VA

    // Setup SGE and write to GM
    __gm__ uint8_t* sgeAddr = wqeAddr + sizeof(struct hns_roce_rc_sq_wqe);
    *(__gm__ uint32_t*)(sgeAddr) = messageLen;
    memDetail = (__gm__ HcclAiRMAMemInfo*)(mem_info_table + sizeof_memdetail * destRankId);
    *(__gm__ uint32_t*)(sgeAddr + sizeof(uint32_t)) = ((__gm__ MemDetails*)(memDetail->memDetailPtr +
        memDetail->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::LOCAL_OUTPUT)))->key; // L_Key
    *(__gm__ uint64_t*)(sgeAddr + 2 * sizeof(uint32_t)) = (uint64_t)srcDmaAddr; // src VA addr memory registered by RNIC

    // wqe & sge cache flush
    cacheWriteThrough(wqeAddr, sizeof(struct hns_roce_rc_sq_wqe) + sizeof(struct hns_roce_lite_wqe_data_seg));
    PipeBarrier<PIPE_ALL>();
    curHead++;
    AIVRDMAPostDoorBell(qp_ctx_entry, curHead, (__gm__ uint32_t*) curHardwareHead, ubLocalTensor, ubLocalHeadTensor,
        selfDataStatusGMAiv);
}

__aicore__ inline void AIVRDMAPostRead(
    GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr, uint64_t destRankId, uint64_t messageLen, __gm__ HcclAiRMAInfo* QpInfo,
    LocalTensor<uint64_t>& ubLocalTensor, LocalTensor<uint32_t>& ubLocalHeadTensor, GlobalTensor<uint32_t>& selfDataStatusGMAiv, uint32_t flag)
{
    auto qpNum = ((__gm__ HcclAiRMAInfo*)QpInfo)->qpNum;
    auto qp_ctx_entry = (__gm__ HcclAiRMAWQ*)(((__gm__ HcclAiRMAInfo*)QpInfo)->sqPtr +
        destRankId * qpNum * (uint64_t)(((__gm__ HcclAiRMAInfo*)QpInfo)->sizeOfAiRMAWQ));
    auto mem_info_table = ((__gm__ HcclAiRMAInfo*)QpInfo)->memPtr;
    auto sizeof_memdetail = ((__gm__ HcclAiRMAInfo*)QpInfo)->sizeOfAiRMAMem;
    auto cur_rank_id = (((__gm__ HcclAiRMAInfo*)QpInfo)->curRankId);
    auto sqBaseAddr = qp_ctx_entry->bufAddr;
    auto wqeSize = qp_ctx_entry->wqeSize;
    auto curHardwareHead = qp_ctx_entry->headAddr;
    cacheWriteThrough((__gm__ uint8_t*)curHardwareHead, 8);
    uint64_t curHead = *(__gm__ uint32_t*)(curHardwareHead);
    auto curHardwareTailAddr = qp_ctx_entry->tailAddr;
    auto QP_DEPTH = qp_ctx_entry->depth;
    // uint64_t shift = 63U - AscendC::CountLeadingZero(QP_DEPTH);
    // uint64_t shift = 11;
    uint64_t shift = 15;

    PipeBarrier<PIPE_ALL>();
    if (flag != -1) {
        selfDataStatusGMAiv.SetValue(0, 12 + flag); // 初始化为0，表示数据未准备好
        DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(selfDataStatusGMAiv);
    }
    

    // Make sure we don't overflow the SQ in an infinite loop - no need to mitigate endless loop as the host
    // will timeout and kill the kernel, same as all2all kernel if it fails to complete (e.g. in case of link loss)
    while(1) {
        cacheWriteThrough((__gm__ uint8_t*)curHardwareTailAddr, CACHEWRITESIZE);
        if ((curHead - *(__gm__ uint32_t*)(curHardwareTailAddr)) < QP_DEPTH - 1) {
            break;
        }
        int64_t systemCycleAfter = AscendC::GetSystemCycle();
        int64_t systemCycleAfter1 = AscendC::GetSystemCycle();
    }

    if (flag != -1) {
        PipeBarrier<PIPE_ALL>();
        selfDataStatusGMAiv.SetValue(0, 13 + flag); // 初始化为0，表示数据未准备好
        DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(selfDataStatusGMAiv);
    }
    

    __gm__ uint8_t* wqeAddr = (__gm__ uint8_t*)(sqBaseAddr + wqeSize * (curHead % QP_DEPTH));

    // Write the WQE to GM
    uint64_t ownBit = (curHead >> shift) & 1U;
    uint32_t byte_4 = 5U;                       // [0:4] opcode=0x3(RDMA_WRITE)
    byte_4 |= ((~ownBit) << 7U) & (1U << 7U);   // [7] owner_bit
    byte_4 |= 1U << 8U;                         // [8:8] IBV_SEND_SIGNALED

    *(__gm__ uint32_t*)(wqeAddr) = byte_4;          // Control set by local parameter see above lines
    *(__gm__ uint32_t*)(wqeAddr + 4) = messageLen;  // message size
    *(__gm__ uint32_t*)(wqeAddr + 8) = 0;           // immtdata is always 0 till we provide poll CQ flow in AIV
    *(__gm__ uint32_t*)(wqeAddr + 12) = 1U << 24U;  // [120:127] num_sge = 1
    *(__gm__ uint32_t*)(wqeAddr + 16) = 0;          // [128:151] start_sge_idx = 0;
    __gm__ HcclAiRMAMemInfo* memDetail = (__gm__ HcclAiRMAMemInfo*)(mem_info_table + sizeof_memdetail * destRankId);
    *(__gm__ uint32_t*)(wqeAddr + 20) = ((__gm__ MemDetails*)(memDetail->memDetailPtr +
        memDetail->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::REMOTE_INPUT)))->key;
    *(__gm__ uint64_t*)(wqeAddr + 24) = (uint64_t)destDmaAddr; // destination VA

    // Setup SGE and write to GM
    __gm__ uint8_t* sgeAddr = wqeAddr + sizeof(struct hns_roce_rc_sq_wqe);
    *(__gm__ uint32_t*)(sgeAddr) = messageLen;
    memDetail = (__gm__ HcclAiRMAMemInfo*)(mem_info_table + sizeof_memdetail * destRankId);
    *(__gm__ uint32_t*)(sgeAddr + sizeof(uint32_t)) = ((__gm__ MemDetails*)(memDetail->memDetailPtr +
        memDetail->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::LOCAL_OUTPUT)))->key; // L_Key
    *(__gm__ uint64_t*)(sgeAddr + 2 * sizeof(uint32_t)) = (uint64_t)srcDmaAddr; // src VA addr memory registered by RNIC

    // wqe & sge cache flush
    cacheWriteThrough(wqeAddr, sizeof(struct hns_roce_rc_sq_wqe) + sizeof(struct hns_roce_lite_wqe_data_seg));
    PipeBarrier<PIPE_ALL>();
    curHead++;
    AIVRDMAPostDoorBell(qp_ctx_entry, curHead, (__gm__ uint32_t*) curHardwareHead, ubLocalTensor, ubLocalHeadTensor,
        selfDataStatusGMAiv);
    PipeBarrier<PIPE_ALL>();
    DumpSQWqeToWin((GM_ADDR)sqBaseAddr, wqeSize, QP_DEPTH, ubLocalTensor, selfDataStatusGMAiv);
}

}
#endif // MOE_DISTRIBUTE_V2_BASE_H
